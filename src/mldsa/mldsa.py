# mldsa-py by Filippo Valsorda is marked CC0 1.0 Universal. To view a copy of
# this mark, visit https://creativecommons.org/publicdomain/zero/1.0/
#
# Alternatively, you may use this source code under the terms of the 0BSD
# license that can be found in the LICENSE file.

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


class InvalidPublicKeyError(ValueError):
    """Raised when a public key is invalid."""


class InvalidContextError(ValueError):
    """Raised when a context string is invalid."""


class VerificationKey:
    """An ML-DSA public key."""

    _p: _Parameters

    def __init__(self, pk: bytes, /, parameters: Parameters | None = None) -> None:
        """Decode an ML-DSA public key.

        If *parameters* is ``None``, the parameter set is inferred from
        the length of *pk*.

        Raises:
            InvalidPublicKeyError: If the public key size is invalid or doesn't match
                the specified parameter set.
        """
        if parameters is None:
            size_to_params = {p.public_key_size: p for p in Parameters}
            if len(pk) not in size_to_params:
                raise InvalidPublicKeyError(f"unexpected public key size {len(pk)}")
            parameters = size_to_params[len(pk)]
        self._p = parameters.value

        if len(pk) != self._p.public_key_size:
            raise InvalidPublicKeyError(f"expected {self._p.public_key_size} bytes, got {len(pk)}")
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
            InvalidContextError: If the context is too long (more than 255 bytes).
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
        raise InvalidContextError(f"expected context of at most 255 bytes, got {len(ctx)}")
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
        assert 0 <= v < Q
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
    assert n * bit_length == len(buf) * 8
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
        assert len(cs) == N
        self.cs = cs

    def __add__(self, other: Self) -> Self:
        if type(self) is not type(other):
            return NotImplemented
        return type(self)([a + b for a, b in zip(self.cs, other.cs)])

    def __sub__(self, other: Self) -> Self:
        if type(self) is not type(other):
            return NotImplemented
        return type(self)([a - b for a, b in zip(self.cs, other.cs)])


class NTTPoly:
    __slots__ = ("cs",)
    cs: list[int]  # don't use F to avoid function call overhead in hot loops

    def __init__(self, cs: list[int]) -> None:
        assert len(cs) == N
        self.cs = cs

    @classmethod
    def zero(cls) -> Self:
        return cls([0 for _ in range(N)])

    def __iadd__(self, other: Self) -> Self:
        if type(self) is not type(other):
            return NotImplemented
        for i in range(N):
            self.cs[i] = (self.cs[i] + other.cs[i]) % Q
        return self

    def __isub__(self, other: Self) -> Self:
        if type(self) is not type(other):
            return NotImplemented
        for i in range(N):
            self.cs[i] = (self.cs[i] - other.cs[i]) % Q
        return self

    def __mul__(self, other: NTTPoly) -> NTTPoly:
        if type(self) is not type(other):
            return NotImplemented
        return NTTPoly([a * b % Q for a, b in zip(self.cs, other.cs)])


def ntt(f: Poly) -> NTTPoly:
    m = 0
    w = [c.v for c in f.cs]
    for len in [128, 64, 32, 16, 8, 4, 2, 1]:
        for start in range(0, N, 2 * len):
            m += 1
            zeta = ZETAS[m]
            for j in range(start, start + len):
                t = zeta * w[j + len] % Q
                w[j + len] = (w[j] - t) % Q
                w[j] = (w[j] + t) % Q
    return NTTPoly(w)


def inverse_ntt(f: NTTPoly) -> Poly:
    m = 255
    w = [c for c in f.cs]
    for len in [1, 2, 4, 8, 16, 32, 64, 128]:
        for start in range(0, N, 2 * len):
            zeta = ZETAS[m]
            m -= 1
            for j in range(start, start + len):
                t = w[j]
                w[j] = (t + w[j + len]) % Q
                w[j + len] = zeta * (w[j + len] - t) % Q
    return Poly([F(v * 8347681 % Q) for v in w])


def sample_ntt(ρ: bytes, s: int, r: int) -> NTTPoly:
    G = shake_128()
    G.update(ρ)
    G.update(bytes([s, r]))
    buf = G.digest(894)

    a: list[int] = []
    while len(a) < N:
        v = int.from_bytes(buf[:3], "little") & 0x7FFFFF
        buf = buf[3:]
        if v < Q:
            a.append(v)
    return NTTPoly(a)


def sample_in_ball(rho: bytes, p: _Parameters) -> Poly:
    G = shake_256()
    G.update(rho)
    buf = G.digest(221)
    s = buf[:8]
    j = memoryview(buf)[8:]

    c = [F(0) for _ in range(N)]
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


ZETAS = [1, 4808194, 3765607, 3761513, 5178923, 5496691, 5234739, 5178987, 7778734, 3542485, 2682288, 2129892, 3764867, 7375178, 557458, 7159240, 5010068, 4317364, 2663378, 6705802, 4855975, 7946292, 676590, 7044481, 5152541, 1714295, 2453983, 1460718, 7737789, 4795319, 2815639, 2283733, 3602218, 3182878, 2740543, 4793971, 5269599, 2101410, 3704823, 1159875, 394148, 928749, 1095468, 4874037, 2071829, 4361428, 3241972, 2156050, 3415069, 1759347, 7562881, 4805951, 3756790, 6444618, 6663429, 4430364, 5483103, 3192354, 556856, 3870317, 2917338, 1853806, 3345963, 1858416, 3073009, 1277625, 5744944, 3852015, 4183372, 5157610, 5258977, 8106357, 2508980, 2028118, 1937570, 4564692, 2811291, 5396636, 7270901, 4158088, 1528066, 482649, 1148858, 5418153, 7814814, 169688, 2462444, 5046034, 4213992, 4892034, 1987814, 5183169, 1736313, 235407, 5130263, 3258457, 5801164, 1787943, 5989328, 6125690, 3482206, 4197502, 7080401, 6018354, 7062739, 2461387, 3035980, 621164, 3901472, 7153756, 2925816, 3374250, 1356448, 5604662, 2683270, 5601629, 4912752, 2312838, 7727142, 7921254, 348812, 8052569, 1011223, 6026202, 4561790, 6458164, 6143691, 1744507, 1753, 6444997, 5720892, 6924527, 2660408, 6600190, 8321269, 2772600, 1182243, 87208, 636927, 4415111, 4423672, 6084020, 5095502, 4663471, 8352605, 822541, 1009365, 5926272, 6400920, 1596822, 4423473, 4620952, 6695264, 4969849, 2678278, 4611469, 4829411, 635956, 8129971, 5925040, 4234153, 6607829, 2192938, 6653329, 2387513, 4768667, 8111961, 5199961, 3747250, 2296099, 1239911, 4541938, 3195676, 2642980, 1254190, 8368000, 2998219, 141835, 8291116, 2513018, 7025525, 613238, 7070156, 6161950, 7921677, 6458423, 4040196, 4908348, 2039144, 6500539, 7561656, 6201452, 6757063, 2105286, 6006015, 6346610, 586241, 7200804, 527981, 5637006, 6903432, 1994046, 2491325, 6987258, 507927, 7192532, 7655613, 6545891, 5346675, 8041997, 2647994, 3009748, 5767564, 4148469, 749577, 4357667, 3980599, 2569011, 6764887, 1723229, 1665318, 2028038, 1163598, 5011144, 3994671, 8368538, 7009900, 3020393, 3363542, 214880, 545376, 7609976, 3105558, 7277073, 508145, 7826699, 860144, 3430436, 140244, 6866265, 6195333, 3123762, 2358373, 6187330, 5365997, 6663603, 2926054, 7987710, 8077412, 3531229, 4405932, 4606686, 1900052, 7598542, 1054478, 7648983]  # fmt: skip  # noqa: E501
