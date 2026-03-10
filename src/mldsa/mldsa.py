# mldsa-py by Filippo Valsorda is marked CC0 1.0 Universal. To view a copy of
# this mark, visit https://creativecommons.org/publicdomain/zero/1.0/
#
# Alternatively, use of this source code is governed by the 0BSD license that
# can be found in the LICENSE file.

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


@dataclass(frozen=True)
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
        k=4, l=4, η=2, γ1=17, γ2=(Q - 1) // 88, λ=128, τ=39, ω=80)  # fmt: skip
    ML_DSA_65 = _Parameters(name="ML-DSA-65", public_key_size=1952, signature_size=3309,
        k=6, l=5, η=4, γ1=19, γ2=(Q - 1) // 32, λ=192, τ=49, ω=55)  # fmt: skip
    ML_DSA_87 = _Parameters(name="ML-DSA-87", public_key_size=2592, signature_size=4627,
        k=8, l=7, η=2, γ1=19, γ2=(Q - 1) // 32, λ=256, τ=60, ω=75)  # fmt: skip

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
        ρ = bytes(pk[:32])
        pkv = memoryview(pk[32:])

        self._t1: list[NTTPoly] = []  # NTT(t₁ ⋅ 2ᵈ)
        for _ in range(self._p.k):
            self._t1.append(ntt(Poly([F(z.v << 13) for z in unpack(bytes(pkv[:320]), N, 10)])))
            pkv = pkv[320:]

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
            ValueError: If the context is too long (more than 255 bytes).
        """
        μ = message_hash(self._tr, message, context)

        if len(signature) != self._p.signature_size:
            raise VerificationError(
                f"invalid signature size: expected {self._p.signature_size} bytes, got {len(signature)}"
            )
        ch = bytes(signature[: self._p.λ // 4])
        sigv = memoryview(signature[self._p.λ // 4 :])
        z: list[Poly] = []
        for _ in range(self._p.l):
            length = (self._p.γ1 + 1) * N // 8
            z.append(Poly(unpack_signed(bytes(sigv[:length]), N, self._p.γ1 + 1)))
            sigv = sigv[length:]
        h: list[list[int]] = [[0] * N for _ in range(self._p.k)]
        idx = 0
        for i in range(self._p.k):
            limit = sigv[self._p.ω + i]
            if limit < idx or limit > self._p.ω:
                raise VerificationError("invalid signature encoding")
            first = idx
            while idx < limit:
                if idx > first and sigv[idx - 1] >= sigv[idx]:
                    raise VerificationError("invalid signature encoding")
                h[i][sigv[idx]] = 1
                idx += 1
        for i in range(idx, self._p.ω):
            if sigv[i] != 0:
                raise VerificationError("invalid signature encoding")

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
        if type(self) is not type(other):
            return NotImplemented
        return type(self)([a + b for a, b in zip(self.cs, other.cs)])

    def __sub__(self, other: Self) -> Self:
        if type(self) is not type(other):
            return NotImplemented
        return type(self)([a - b for a, b in zip(self.cs, other.cs)])


class NTTPoly(Poly):
    def __mul__(self, other: NTTPoly) -> NTTPoly:
        if type(self) is not type(other):
            return NotImplemented
        return NTTPoly([a * b for a, b in zip(self.cs, other.cs)])


def ntt(f: Poly) -> NTTPoly:
    m = 0
    w = list(f.cs)
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
    w = list(f.cs)
    for len in [1, 2, 4, 8, 16, 32, 64, 128]:
        for start in range(0, N, 2 * len):
            zeta = ZETAS[m]
            m -= 1
            for j in range(start, start + len):
                t = w[j]
                w[j] = t + w[j + len]
                w[j + len] = w[j + len] - t
                w[j + len] = zeta * w[j + len]
    return Poly([f * F(8347681) for f in w])


def sample_ntt(ρ: bytes, s: int, r: int) -> NTTPoly:
    G = shake_128()
    G.update(ρ)
    G.update(bytes([s, r]))
    buf = G.digest(894)

    a: list[F] = []
    while len(a) < N:
        v = int.from_bytes(buf[:3], "little") & 0x7FFFFF
        buf = buf[3:]
        if v < Q:
            a.append(F(v))
    return NTTPoly(a)


def sample_in_ball(rho: bytes, p: _Parameters) -> Poly:
    G = shake_256()
    G.update(rho)
    buf = G.digest(221)
    s = buf[:8]
    j = memoryview(buf)[8:]

    c = [F(0)] * N
    for i in range(256 - p.τ, 256):
        while j[0] > i:
            j = j[1:]
        c[i] = c[j[0]]
        bit_idx = i + p.τ - 256
        bit = (s[bit_idx // 8] >> (bit_idx % 8)) & 1
        c[j[0]] = F(1) if bit == 0 else F(Q - 1)
        j = j[1:]

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


ZETAS = [F(1), F(4808194), F(3765607), F(3761513), F(5178923), F(5496691), F(5234739), F(5178987), F(7778734), F(3542485), F(2682288), F(2129892), F(3764867), F(7375178), F(557458), F(7159240), F(5010068), F(4317364), F(2663378), F(6705802), F(4855975), F(7946292), F(676590), F(7044481), F(5152541), F(1714295), F(2453983), F(1460718), F(7737789), F(4795319), F(2815639), F(2283733), F(3602218), F(3182878), F(2740543), F(4793971), F(5269599), F(2101410), F(3704823), F(1159875), F(394148), F(928749), F(1095468), F(4874037), F(2071829), F(4361428), F(3241972), F(2156050), F(3415069), F(1759347), F(7562881), F(4805951), F(3756790), F(6444618), F(6663429), F(4430364), F(5483103), F(3192354), F(556856), F(3870317), F(2917338), F(1853806), F(3345963), F(1858416), F(3073009), F(1277625), F(5744944), F(3852015), F(4183372), F(5157610), F(5258977), F(8106357), F(2508980), F(2028118), F(1937570), F(4564692), F(2811291), F(5396636), F(7270901), F(4158088), F(1528066), F(482649), F(1148858), F(5418153), F(7814814), F(169688), F(2462444), F(5046034), F(4213992), F(4892034), F(1987814), F(5183169), F(1736313), F(235407), F(5130263), F(3258457), F(5801164), F(1787943), F(5989328), F(6125690), F(3482206), F(4197502), F(7080401), F(6018354), F(7062739), F(2461387), F(3035980), F(621164), F(3901472), F(7153756), F(2925816), F(3374250), F(1356448), F(5604662), F(2683270), F(5601629), F(4912752), F(2312838), F(7727142), F(7921254), F(348812), F(8052569), F(1011223), F(6026202), F(4561790), F(6458164), F(6143691), F(1744507), F(1753), F(6444997), F(5720892), F(6924527), F(2660408), F(6600190), F(8321269), F(2772600), F(1182243), F(87208), F(636927), F(4415111), F(4423672), F(6084020), F(5095502), F(4663471), F(8352605), F(822541), F(1009365), F(5926272), F(6400920), F(1596822), F(4423473), F(4620952), F(6695264), F(4969849), F(2678278), F(4611469), F(4829411), F(635956), F(8129971), F(5925040), F(4234153), F(6607829), F(2192938), F(6653329), F(2387513), F(4768667), F(8111961), F(5199961), F(3747250), F(2296099), F(1239911), F(4541938), F(3195676), F(2642980), F(1254190), F(8368000), F(2998219), F(141835), F(8291116), F(2513018), F(7025525), F(613238), F(7070156), F(6161950), F(7921677), F(6458423), F(4040196), F(4908348), F(2039144), F(6500539), F(7561656), F(6201452), F(6757063), F(2105286), F(6006015), F(6346610), F(586241), F(7200804), F(527981), F(5637006), F(6903432), F(1994046), F(2491325), F(6987258), F(507927), F(7192532), F(7655613), F(6545891), F(5346675), F(8041997), F(2647994), F(3009748), F(5767564), F(4148469), F(749577), F(4357667), F(3980599), F(2569011), F(6764887), F(1723229), F(1665318), F(2028038), F(1163598), F(5011144), F(3994671), F(8368538), F(7009900), F(3020393), F(3363542), F(214880), F(545376), F(7609976), F(3105558), F(7277073), F(508145), F(7826699), F(860144), F(3430436), F(140244), F(6866265), F(6195333), F(3123762), F(2358373), F(6187330), F(5365997), F(6663603), F(2926054), F(7987710), F(8077412), F(3531229), F(4405932), F(4606686), F(1900052), F(7598542), F(1054478), F(7648983)]  # fmt: skip  # noqa: E501
