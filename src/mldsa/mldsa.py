"""Pure-Python implementation of ML-DSA (FIPS 204) signature verification."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from hashlib import shake_128, shake_256

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = ["Parameters", "VerificationError", "VerificationKey"]

Q = 8380417
N = 256


@dataclass(frozen=True, slots=True)
class _Parameters:
    name: str
    public_key_size: int
    signature_size: int
    k: int
    l: int
    η: int
    γ1: int
    γ2: int
    λ: int
    τ: int
    ω: int


class Parameters(Enum):
    """ML-DSA parameter sets as defined in FIPS 204."""

    ML_DSA_44 = _Parameters(name="ML-DSA-44", public_key_size=1312, signature_size=2420,
        k=4, l=4, η=2, γ1=17, γ2=88, λ=128, τ=39, ω=80)  # fmt: skip
    ML_DSA_65 = _Parameters(name="ML-DSA-65", public_key_size=1952, signature_size=3309,
        k=6, l=5, η=4, γ1=19, γ2=32, λ=192, τ=49, ω=55)  # fmt: skip
    ML_DSA_87 = _Parameters(name="ML-DSA-87", public_key_size=2592, signature_size=4627,
        k=8, l=7, η=2, γ1=19, γ2=32, λ=256, τ=60, ω=75)  # fmt: skip

    @property
    def public_key_size(self) -> int:
        """Return the encoded public key size in bytes."""
        return self.value.public_key_size

    @property
    def signature_size(self) -> int:
        """Return the signature size in bytes."""
        return self.value.signature_size

    def __str__(self) -> str:
        """Return the human-readable parameter set name, e.g. ``ML-DSA-44``."""
        return self.value.name


class VerificationError(Exception):
    """Raised when signature verification fails."""


class VerificationKey:
    """An ML-DSA public key."""

    _p: _Parameters

    def __init__(self, pk: bytes, /, parameters: Parameters | None = None) -> None:
        """Decode an ML-DSA public key.

        If *parameters* is ``None``, the parameter set is inferred from
        the length of *pk*.

        Raises:
            ValueError: If the public key size is invalid or doesn't match
                the specified parameter set.
        """
        if parameters is None:
            size_to_params = {p.public_key_size: p for p in Parameters}
            if len(pk) not in size_to_params:
                raise ValueError(f"unexpected public key size {len(pk)}")
            parameters = size_to_params[len(pk)]
        self._p = parameters.value

        if len(pk) != self._p.public_key_size:
            raise ValueError(f"expected {self._p.public_key_size} bytes, got {len(pk)}")
        self._enc = pk
        self._tr = public_key_hash(pk)
        ρ = pk[:32]
        pk = pk[32:]

        self._t1: list[NTTPoly] = []  # NTT(t₁ ⋅ 2ᵈ)
        for _ in range(self._p.k):
            self._t1.append(ntt(Poly([z << 13 for z in unpack(pk[:320], N, 10)])))
            pk = pk[320:]

        self._A: list[list[NTTPoly]] = [[] for _ in range(self._p.k)]
        for r in range(self._p.k):
            for s in range(self._p.l):
                self._A[r].append(sample_ntt(ρ, s, r))

    def __bytes__(self) -> bytes:
        """Return the encoded public key."""
        return self._enc

    @property
    def parameters(self) -> Parameters:
        """Return the parameter set of this key."""
        return Parameters(self._p)

    def verify(self, message: bytes, signature: bytes, *, context: bytes = b"") -> None:
        """Verify a signature over *message*.

        Raises:
            VerificationError: If the signature is invalid.
            ValueError: If the signature size or encoding is invalid.
        """
        μ = message_hash(self._tr, message, context)

        if len(signature) != self._p.signature_size:
            raise ValueError(f"expected {self._p.signature_size} bytes, got {len(signature)}")
        ch = signature[: self._p.λ // 4]
        signature = signature[self._p.λ // 4 :]
        z: list[Poly] = []
        for _ in range(self._p.l):
            length = (self._p.γ1 + 1) * N // 8
            z.append(Poly(unpack_signed(signature[:length], N, self._p.γ1 + 1)))
            signature = signature[length:]
        h: list[list[int]] = [[0] * N for _ in range(self._p.k)]
        idx = 0
        for i in range(self._p.k):
            limit = signature[self._p.ω + i]
            if limit < idx or limit > self._p.ω:
                raise ValueError("invalid signature encoding")
            first = idx
            while idx < limit:
                if idx > first and signature[idx - 1] >= signature[idx]:
                    raise ValueError("invalid signature encoding")
                h[i][signature[idx]] = 1
                idx += 1
        for i in range(idx, self._p.ω):
            if signature[i] != 0:
                raise ValueError("invalid signature encoding")

        c = ntt(sample_in_ball(ch, self._p))

        z_hat = [ntt(x) for x in z]
        w: list[Poly] = []  # Â ∘ NTT(z) − NTT(c) ∘ NTT(t₁ ⋅ 2ᵈ)
        for i in range(self._p.k):
            w_hat = NTTPoly.zero()
            for j in range(self._p.l):
                w_hat += z_hat[j] * self._A[i][j]
            w_hat -= c * self._t1[i]
            w.append(inverse_ntt(w_hat))

        w1 = [use_hint(w[i], h[i], self._p) for i in range(self._p.k)]

        H = shake_256()
        H.update(μ)
        w1_bit_length = ((Q - 1) // (2 * self._p.γ2) - 1).bit_length()
        for i in range(self._p.k):
            H.update(pack(w1[i], w1_bit_length))
        if H.digest(self._p.λ // 4) != ch:
            raise VerificationError("invalid signature")

        β = self._p.τ * self._p.η
        γ1 = 1 << self._p.γ1
        γ1β = γ1 - β

        for v in z:
            if any(x.infinity_norm() >= γ1β for x in v.cs):
                raise VerificationError("invalid signature")


def public_key_hash(pk: bytes) -> bytes:
    h = shake_256()
    h.update(pk)
    return h.digest(64)


def message_hash(tr: bytes, m: bytes, ctx: bytes) -> bytes:
    if len(ctx) > 255:
        raise ValueError(f"expected context of at most 255 bytes, got {len(ctx)}")
    h = shake_256()
    h.update(tr)
    h.update(b"\x00")
    h.update(bytes([len(ctx)]))
    h.update(ctx)
    h.update(m)
    return h.digest(64)


class F:
    __slots__ = ("v",)

    def __init__(self, v: int) -> None:
        if not (0 <= v < Q):
            raise ValueError(f"expected 0 <= v < {Q}, got {v}")
        self.v = v

    @classmethod
    def reduce(cls, v: int) -> F:
        return cls(v % Q)

    def __add__(self, other: F) -> F:
        return F.reduce(self.v + other.v)

    def __sub__(self, other: F) -> F:
        return F.reduce(self.v - other.v)

    def __mul__(self, other: F) -> F:
        return F.reduce(self.v * other.v)

    def __lshift__(self, other: int) -> F:
        return F(self.v << other)

    def infinity_norm(self) -> int:
        return self.v if self.v <= Q // 2 else Q - self.v


def centered_mod(v: int, m: int) -> int:
    r = v % m
    if r > m // 2:
        r -= m
    return r


def decompose(r: F, p: _Parameters) -> tuple[int, int]:
    r0 = centered_mod(r.v, 2 * p.γ2)
    if r.v - r0 == Q - 1:
        return 0, r0 - 1
    r1 = (r.v - r0) // (2 * p.γ2)
    return r1, r0


def unpack(buf: bytes, n: int, bit_length: int) -> list[F]:
    if n * bit_length != len(buf) * 8:
        raise ValueError(f"expected {(n * bit_length) / 8} bytes, got {len(buf)}")
    res: list[F] = []
    acc = 0
    acc_len = 0
    for b in buf:
        acc |= b << acc_len
        acc_len += 8
        while acc_len >= bit_length:
            res.append(F(acc & ((1 << bit_length) - 1)))
            acc >>= bit_length
            acc_len -= bit_length
    return res


def unpack_signed(buf: bytes, n: int, bit_length: int) -> list[F]:
    b = F(1 << (bit_length - 1))
    return [b - x for x in unpack(buf, n, bit_length)]


def pack(cs: list[int], bit_length: int) -> bytes:
    acc = 0
    acc_len = 0
    res = bytearray()
    for c in cs:
        acc |= c << acc_len
        acc_len += bit_length
        while acc_len >= 8:
            res.append(acc & 0xFF)
            acc >>= 8
            acc_len -= 8
    if acc_len > 0:
        res.append(acc & 0xFF)
    return bytes(res)


class Poly:
    __slots__ = ("cs",)
    cs: list[F]

    def __init__(self, cs: list[F]) -> None:
        if len(cs) != N:
            raise ValueError(f"expected {N} coefficients, got {len(cs)}")
        self.cs = cs

    @classmethod
    def zero(cls) -> Self:
        return cls([F(0)] * N)

    def __add__(self, other: Self) -> Self:
        type_must_match(self, other)
        return type(self)([a + b for a, b in zip(self.cs, other.cs)])

    def __sub__(self, other: Self) -> Self:
        type_must_match(self, other)
        return type(self)([a - b for a, b in zip(self.cs, other.cs)])


class NTTPoly(Poly):
    def __mul__(self, other: NTTPoly) -> NTTPoly:
        type_must_match(self, other)
        return NTTPoly([a * b for a, b in zip(self.cs, other.cs)])


def ntt(f: Poly) -> NTTPoly:
    m = 0
    w = f.cs
    for len in [128, 64, 32, 16, 8, 4, 2, 1]:
        for start in range(0, N, 2 * len):
            m += 1
            zeta = ZETAS[m]
            for j in range(start, start + len):
                t = zeta * w[j + len]
                w[j + len] = w[j] - t
                w[j] = w[j] + t
    return NTTPoly(w)


def inverse_ntt(f: NTTPoly) -> Poly:
    m = 255
    w = f.cs
    for len in [1, 2, 4, 8, 16, 32, 64, 128]:
        for start in range(0, N, 2 * len):
            zeta = ZETAS[m]
            m -= 1
            for j in range(start, start + len):
                t = w[j]
                w[j] = t + w[j + len]
                w[j + len] = t - w[j + len]
                w[j + len] = zeta * w[j + len]
    return Poly([f * F(8347681) for f in w])


def sample_ntt(ρ: bytes, s: int, r: int) -> NTTPoly:
    G = shake_128()
    G.update(ρ)
    G.update(bytes([s, r]))

    a: list[F] = []
    buf = G.digest(168)
    while len(a) < N:
        if len(buf) == 0:
            buf = G.digest(168)
        v = int.from_bytes(buf[:3], "little") & 0x7FFFFF
        buf = buf[3:]
        if v < Q:
            a.append(F(v))
    return ntt(Poly(a))


def sample_in_ball(rho: bytes, p: _Parameters) -> Poly:
    G = shake_256()
    G.update(rho)
    s = G.digest(8)

    c = [F(0)] * N
    for i in range(256 - p.τ, 256):
        j = G.digest(1)
        while j[0] > i:
            j = G.digest(1)
        c[i] = c[j[0]]
        bit_idx = i + p.τ - 256
        bit = (s[bit_idx // 8] >> (bit_idx % 8)) & 1
        c[j[0]] = F(1) if bit == 0 else F(Q - 1)

    return Poly(c)


def use_hint(w: Poly, h: list[int], p: _Parameters) -> list[int]:
    m = (Q - 1) // (2 * p.γ2)
    w1: list[int] = []
    for i in range(N):
        r1, r0 = decompose(w.cs[i], p)
        if h[i] == 0:
            w1.append(r1)
        elif r0 > 0:
            w1.append((r1 + 1) % m)
        else:
            w1.append((r1 - 1) % m)
    return w1


ZETAS = [F(4193792), F(25847), F(5771523), F(7861508), F(237124), F(7602457), F(7504169), F(466468), F(1826347), F(2353451), F(8021166), F(6288512), F(3119733), F(5495562), F(3111497), F(2680103), F(2725464), F(1024112), F(7300517), F(3585928), F(7830929), F(7260833), F(2619752), F(6271868), F(6262231), F(4520680), F(6980856), F(5102745), F(1757237), F(8360995), F(4010497), F(280005), F(2706023), F(95776), F(3077325), F(3530437), F(6718724), F(4788269), F(5842901), F(3915439), F(4519302), F(5336701), F(3574422), F(5512770), F(3539968), F(8079950), F(2348700), F(7841118), F(6681150), F(6736599), F(3505694), F(4558682), F(3507263), F(6239768), F(6779997), F(3699596), F(811944), F(531354), F(954230), F(3881043), F(3900724), F(5823537), F(2071892), F(5582638), F(4450022), F(6851714), F(4702672), F(5339162), F(6927966), F(3475950), F(2176455), F(6795196), F(7122806), F(1939314), F(4296819), F(7380215), F(5190273), F(5223087), F(4747489), F(126922), F(3412210), F(7396998), F(2147896), F(2715295), F(5412772), F(4686924), F(7969390), F(5903370), F(7709315), F(7151892), F(8357436), F(7072248), F(7998430), F(1349076), F(1852771), F(6949987), F(5037034), F(264944), F(508951), F(3097992), F(44288), F(7280319), F(904516), F(3958618), F(4656075), F(8371839), F(1653064), F(5130689), F(2389356), F(8169440), F(759969), F(7063561), F(189548), F(4827145), F(3159746), F(6529015), F(5971092), F(8202977), F(1315589), F(1341330), F(1285669), F(6795489), F(7567685), F(6940675), F(5361315), F(4499357), F(4751448), F(3839961), F(2091667), F(3407706), F(2316500), F(3817976), F(5037939), F(2244091), F(5933984), F(4817955), F(266997), F(2434439), F(7144689), F(3513181), F(4860065), F(4621053), F(7183191), F(5187039), F(900702), F(1859098), F(909542), F(819034), F(495491), F(6767243), F(8337157), F(7857917), F(7725090), F(5257975), F(2031748), F(3207046), F(4823422), F(7855319), F(7611795), F(4784579), F(342297), F(286988), F(5942594), F(4108315), F(3437287), F(5038140), F(1735879), F(203044), F(2842341), F(2691481), F(5790267), F(1265009), F(4055324), F(1247620), F(2486353), F(1595974), F(4613401), F(1250494), F(2635921), F(4832145), F(5386378), F(1869119), F(1903435), F(7329447), F(7047359), F(1237275), F(5062207), F(6950192), F(7929317), F(1312455), F(3306115), F(6417775), F(7100756), F(1917081), F(5834105), F(7005614), F(1500165), F(777191), F(2235880), F(3406031), F(7838005), F(5548557), F(6709241), F(6533464), F(5796124), F(4656147), F(594136), F(4603424), F(6366809), F(2432395), F(2454455), F(8215696), F(1957272), F(3369112), F(185531), F(7173032), F(5196991), F(162844), F(1616392), F(3014001), F(810149), F(1652634), F(4686184), F(6581310), F(5341501), F(3523897), F(3866901), F(269760), F(2213111), F(7404533), F(1717735), F(472078), F(7953734), F(1723600), F(6577327), F(1910376), F(6712985), F(7276084), F(8119771), F(4546524), F(5441381), F(6144432), F(7959518), F(6094090), F(183443), F(7403526), F(1612842), F(4834730), F(7826001), F(3919660), F(8332111), F(7018208), F(3937738), F(1400424), F(7534263), F(1976782)]  # fmt: skip  # noqa: E501


def type_must_match(a: object, b: object) -> None:
    if type(a) is not type(b):
        raise TypeError(f"expected {type(a).__name__}, got {type(b).__name__}")
