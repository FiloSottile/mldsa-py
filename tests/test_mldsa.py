"""Tests for the mldsa package, ported from Go crypto/internal/fips140/mldsa."""

from __future__ import annotations

import hashlib
import random
from array import array
from typing import Any

import pytest

from mldsa import (
    InvalidContextError,
    InvalidVerificationKeyError,
    ParameterSet,
    VerificationError,
    VerificationKey,
)
from mldsa.mldsa import (
    ZETAS,
    F,
    N,
    NTTPoly,
    Poly,
    Q,
    _Parameters,
    centered_mod,
    decompose,
    inverse_ntt,
    message_hash,
    ntt,
    pack,
    sample_in_ball,
    sample_ntt,
    unpack,
    unpack_signed,
    use_hint,
)


def interesting_values() -> list[int]:
    """Return a set of interesting field values for testing, matching the Go test."""
    return [0, 1, 2, 3, Q - 3, Q - 2, Q - 1, Q // 2, (Q + 1) // 2]


class TestFieldArithmetic:
    def test_field_add(self) -> None:
        for a in interesting_values():
            for b in interesting_values():
                got = (F(a) + F(b)).v
                exp = (a + b) % Q
                assert got == exp, f"{a} + {b} = {got}, expected {exp}"

    def test_field_sub(self) -> None:
        for a in interesting_values():
            for b in interesting_values():
                got = (F(a) - F(b)).v
                exp = (a - b) % Q
                assert got == exp, f"{a} - {b} = {got}, expected {exp}"

    def test_field_mul(self) -> None:
        for a in interesting_values():
            for b in interesting_values():
                got = (F(a) * F(b)).v
                exp = (a * b) % Q
                assert got == exp, f"{a} * {b} = {got}, expected {exp}"

    def test_field_reduce(self) -> None:
        for v in [Q, Q + 1, 2 * Q - 1, 2 * Q, -1, -Q]:
            got = F.reduce(v).v
            exp = v % Q
            assert got == exp, f"F.reduce({v}) = {got}, expected {exp}"

    def test_field_bounds(self) -> None:
        with pytest.raises(AssertionError):
            F(Q)
        with pytest.raises(AssertionError):
            F(-1)


class TestCenteredMod:
    def test_centered_mod_congruent(self) -> None:
        for x in interesting_values():
            for m in [2 * 95232, 2 * 261888]:
                r = centered_mod(x, m)
                assert x % m == r % m, f"centered_mod({x}, {m}) = {r} not congruent"

    def test_centered_mod_range(self) -> None:
        for m in [2 * 95232, 2 * 261888]:
            for x in range(0, m * 3, m // 7):
                r = centered_mod(x, m)
                assert r > -(m // 2) - 1, f"centered_mod({x}, {m}) = {r} too small"
                assert r <= m // 2, f"centered_mod({x}, {m}) = {r} too large"


class TestInfinityNorm:
    def test_infinity_norm(self) -> None:
        for x in interesting_values():
            got = F(x).infinity_norm()
            # The infinity norm is |centered_mod(x, Q)|.
            c = x if x <= Q // 2 else Q - x
            assert got == c, f"F({x}).infinity_norm() = {got}, expected {c}"

    def test_infinity_norm_one(self) -> None:
        assert F(1).infinity_norm() == 1

    def test_infinity_norm_minus_one(self) -> None:
        assert F(Q - 1).infinity_norm() == 1


class TestZetas:
    @staticmethod
    def bit_rev8(n: int) -> int:
        """Reverse the bits of an 8-bit integer."""
        r = 0
        for i in range(8):
            r |= ((n >> i) & 1) << (7 - i)
        return r

    def test_zetas(self) -> None:
        zeta = 1753
        for k, z in enumerate(ZETAS):
            exp = pow(zeta, self.bit_rev8(k), Q)
            assert z == exp, f"ZETAS[{k}] = {z}, expected {exp}"


class TestDecompose:
    def _test_decompose(self, p: _Parameters) -> None:
        # Test all interesting values plus a random sample.
        test_values = interesting_values() + random.sample(range(Q), 1000)
        for x in test_values:
            r1, r0 = decompose(F(x), p)
            # Check reconstruction: x == r1 * 2γ2 + r0 (mod Q).
            reconstructed = (r1 * 2 * p.γ2 + r0) % Q
            assert reconstructed == x, f"decompose({x}) = ({r1}, {r0}), reconstructs to {reconstructed}"

    def test_decompose_44(self) -> None:
        self._test_decompose(ParameterSet.ML_DSA_44.value)

    def test_decompose_65(self) -> None:
        self._test_decompose(ParameterSet.ML_DSA_65.value)

    def test_decompose_87(self) -> None:
        self._test_decompose(ParameterSet.ML_DSA_87.value)


class TestUseHint:
    def _test_use_hint_no_hint(self, p: _Parameters) -> None:
        cs = [F(v) for v in random.sample(range(Q), N)]
        w = Poly(cs)
        h = [0] * N
        w1 = use_hint(w, h, p)
        for i in range(N):
            r1, _ = decompose(cs[i], p)
            assert w1[i] == r1, f"use_hint mismatch at index {i}"

    def test_use_hint_44(self) -> None:
        self._test_use_hint_no_hint(ParameterSet.ML_DSA_44.value)

    def test_use_hint_65(self) -> None:
        self._test_use_hint_no_hint(ParameterSet.ML_DSA_65.value)


class TestNTT:
    def test_ntt_inverse_roundtrip(self) -> None:
        random.seed(42)
        original = [random.randrange(Q) for _ in range(N)]
        f = Poly([F(v) for v in original])
        result = inverse_ntt(ntt(f))
        for i in range(N):
            assert result.cs[i].v == original[i], f"NTT roundtrip failed at index {i}"

    def test_ntt_linearity(self) -> None:
        """NTT(a + b) == NTT(a) + NTT(b)."""
        random.seed(43)
        a = Poly([F(random.randrange(Q)) for _ in range(N)])
        b = Poly([F(random.randrange(Q)) for _ in range(N)])
        ntt_sum = ntt(a + b)
        sum_ntt = ntt(a)
        sum_ntt += ntt(b)
        for i in range(N):
            assert ntt_sum.cs[i] == sum_ntt.cs[i], f"linearity failed at {i}"

    def test_ntt_mul_is_polynomial_mul(self) -> None:
        """NTT pointwise multiplication corresponds to polynomial multiplication
        modulo X^256 + 1."""
        random.seed(44)
        a_vals = [random.randrange(Q) for _ in range(N)]
        b_vals = [random.randrange(Q) for _ in range(N)]
        a = Poly([F(v) for v in a_vals])
        b = Poly([F(v) for v in b_vals])
        result = inverse_ntt(ntt(a) * ntt(b))
        # Compute the schoolbook product mod X^256 + 1 mod Q.
        product = [0] * N
        for i in range(N):
            for j in range(N):
                idx = i + j
                if idx < N:
                    product[idx] = (product[idx] + a_vals[i] * b_vals[j]) % Q
                else:
                    product[idx - N] = (product[idx - N] - a_vals[i] * b_vals[j]) % Q
        for i in range(N):
            assert result.cs[i].v == product[i], f"poly mul mismatch at {i}"


class TestPackUnpack:
    def test_pack_unpack_10bit(self) -> None:
        random.seed(50)
        values = [random.randrange(1024) for _ in range(N)]
        buf = pack(values, 10)
        assert len(buf) == N * 10 // 8
        result = unpack(buf, N, 10)
        for i in range(N):
            assert result[i].v == values[i], f"10-bit pack/unpack failed at {i}"

    def test_pack_unpack_4bit(self) -> None:
        random.seed(51)
        values = [random.randrange(16) for _ in range(N)]
        buf = pack(values, 4)
        assert len(buf) == N * 4 // 8
        result = unpack(buf, N, 4)
        for i in range(N):
            assert result[i].v == values[i], f"4-bit pack/unpack failed at {i}"

    def test_unpack_signed(self) -> None:
        # For bit_length=18 (γ1+1 for ML-DSA-44), b = 1 << 17 = 131072.
        # unpack_signed returns b - x for each x from unpack.
        bit_length = 18
        b = 1 << (bit_length - 1)
        # A buffer of all zeros should give b - 0 = b for each element.
        buf = bytes(N * bit_length // 8)
        result = unpack_signed(buf, N, bit_length)
        for i in range(N):
            assert result[i].v == b, f"unpack_signed zero buf failed at {i}"


class TestSampleNTT:
    def test_deterministic(self) -> None:
        rho = bytes(32)
        a1 = sample_ntt(rho, 0, 0)
        a2 = sample_ntt(rho, 0, 0)
        for i in range(N):
            assert a1.cs[i] == a2.cs[i]

    def test_different_indices(self) -> None:
        rho = bytes(32)
        a1 = sample_ntt(rho, 0, 0)
        a2 = sample_ntt(rho, 1, 0)
        assert any(a1.cs[i] != a2.cs[i] for i in range(N))

    def test_coefficients_in_range(self) -> None:
        rho = bytes(range(32))
        a = sample_ntt(rho, 3, 2)
        for i in range(N):
            assert 0 <= a.cs[i] < Q


class TestSampleInBall:
    def _test_sample_in_ball(self, p: _Parameters) -> None:
        rho = bytes(p.λ // 4)
        c = sample_in_ball(rho, p)
        # Count nonzero coefficients: should be exactly tau.
        nonzero = sum(1 for x in c.cs if x.v != 0)
        assert nonzero == p.τ, f"expected {p.τ} nonzero, got {nonzero}"
        # All nonzero coefficients should be +1 or -1 (i.e., 1 or Q-1).
        for i, x in enumerate(c.cs):
            if x.v != 0:
                assert x.v in (1, Q - 1), f"c[{i}] = {x.v}, expected 1 or {Q - 1}"

    def test_sample_in_ball_44(self) -> None:
        self._test_sample_in_ball(ParameterSet.ML_DSA_44.value)

    def test_sample_in_ball_65(self) -> None:
        self._test_sample_in_ball(ParameterSet.ML_DSA_65.value)

    def test_sample_in_ball_87(self) -> None:
        self._test_sample_in_ball(ParameterSet.ML_DSA_87.value)


class TestConstants:
    def test_parameter_sizes(self) -> None:
        assert ParameterSet.ML_DSA_44.verification_key_size == 1312
        assert ParameterSet.ML_DSA_65.verification_key_size == 1952
        assert ParameterSet.ML_DSA_87.verification_key_size == 2592
        assert ParameterSet.ML_DSA_44.signature_size == 2420
        assert ParameterSet.ML_DSA_65.signature_size == 3309
        assert ParameterSet.ML_DSA_87.signature_size == 4627

    def test_parameter_names(self) -> None:
        assert str(ParameterSet.ML_DSA_44) == "ML-DSA-44"
        assert str(ParameterSet.ML_DSA_65) == "ML-DSA-65"
        assert str(ParameterSet.ML_DSA_87) == "ML-DSA-87"

    def test_parameter_enum(self) -> None:
        for p in ParameterSet:
            vk = VerificationKey(bytes(p.verification_key_size), parameters=p)
            assert vk.parameters is p

    def test_q(self) -> None:
        assert Q == 8380417
        assert Q == 2**23 - 2**13 + 1

    def test_n(self) -> None:
        assert N == 256


class TestVerificationKey:
    def test_wrong_size(self) -> None:
        with pytest.raises(InvalidVerificationKeyError):
            VerificationKey(b"too short")

    def test_wrong_size_for_parameters(self) -> None:
        with pytest.raises(InvalidVerificationKeyError):
            VerificationKey(bytes(1952), parameters=ParameterSet.ML_DSA_44)

    def test_auto_detect_parameters(self) -> None:
        for p in ParameterSet:
            vk = VerificationKey(bytes(p.verification_key_size))
            assert vk.parameters is p

    def test_bytes_roundtrip(self) -> None:
        for p in ParameterSet:
            pk = bytes(p.verification_key_size)
            vk = VerificationKey(pk)
            assert bytes(vk) == pk

    def test_verify_wrong_size_signature(self) -> None:
        vk = VerificationKey(bytes(ParameterSet.ML_DSA_44.verification_key_size))
        with pytest.raises(VerificationError):
            vk.verify(b"bad sig", b"message")

    def test_verify_context_too_long(self) -> None:
        vk = VerificationKey(bytes(ParameterSet.ML_DSA_44.verification_key_size))
        with pytest.raises(InvalidContextError, match="context"):
            vk.verify(bytes(2420), b"message", context=bytes(256))


def buffer_variants(b: bytes) -> list[Any]:
    """Return the same bytes exposed through several Buffer implementations.

    ``array("I")`` holds 4-byte items, so ``len()`` on a memoryview of it is the
    item count rather than the byte count. Sizes must be measured in bytes.
    """
    variants: list[Any] = [b, bytearray(b), memoryview(b), array("B", b)]
    if len(b) % 4 == 0:
        a = array("I")
        a.frombytes(b)
        variants.append(a)
    return variants


class TestBufferTypes:
    def test_key_size_measured_in_bytes(self) -> None:
        pk = bytes(range(256)) * 5 + bytes(32)
        assert len(pk) == ParameterSet.ML_DSA_44.verification_key_size
        for buf in buffer_variants(pk):
            vk = VerificationKey(buf)
            assert bytes(vk) == pk
            assert vk.parameters is ParameterSet.ML_DSA_44

    def test_key_wrong_byte_size_in_non_byte_buffer(self) -> None:
        # 1312 four-byte items is 5248 bytes, not a valid key size.
        with pytest.raises(InvalidVerificationKeyError, match="5248"):
            VerificationKey(array("I", [0] * 1312))

    def test_signature_size_measured_in_bytes(self) -> None:
        vk = VerificationKey(bytes(ParameterSet.ML_DSA_44.verification_key_size))
        # 2420 four-byte items is 9680 bytes, not a valid signature size.
        with pytest.raises(VerificationError, match="9680"):
            vk.verify(array("I", [0] * 2420), b"message")

    def test_signature_buffer_types_agree(self) -> None:
        vk = VerificationKey(bytes(ParameterSet.ML_DSA_44.verification_key_size))
        for buf in buffer_variants(bytes(2420)):
            with pytest.raises(VerificationError, match=r"^invalid signature$"):
                vk.verify(buf, b"message")

    def test_context_length_measured_in_bytes(self) -> None:
        # 255 four-byte items is 1020 bytes, over the 255 byte limit.
        with pytest.raises(InvalidContextError, match="1020"):
            message_hash(bytes(64), b"message", array("I", [0] * 255))

    def test_context_buffer_types_agree(self) -> None:
        ctx = bytes(range(64))
        expected = message_hash(bytes(64), b"message", ctx)
        for buf in buffer_variants(ctx):
            assert message_hash(bytes(64), b"message", buf) == expected

    def test_non_contiguous_buffer_rejected(self) -> None:
        vk = VerificationKey(bytes(ParameterSet.ML_DSA_44.verification_key_size))
        with pytest.raises(TypeError):
            vk.verify(memoryview(bytearray(4840))[::2], b"message")
        with pytest.raises(TypeError):
            VerificationKey(memoryview(bytearray(2624))[::2])


class TestPolyTypes:
    def test_poly_add_type_mismatch(self) -> None:
        p = Poly([F(0) for _ in range(N)])
        n = NTTPoly.zero()
        with pytest.raises(TypeError):
            p + n  # ty: ignore[unsupported-operator]

    def test_poly_sub_type_mismatch(self) -> None:
        p = Poly([F(0) for _ in range(N)])
        n = NTTPoly.zero()
        with pytest.raises(TypeError):
            p - n  # ty: ignore[unsupported-operator]

    def test_ntt_poly_mul(self) -> None:
        a = NTTPoly([1] * N)
        b = NTTPoly([2] * N)
        c = a * b
        for i in range(N):
            assert c.cs[i] == 2

    def test_poly_wrong_length(self) -> None:
        with pytest.raises(AssertionError):
            Poly([F(0)] * 100)


def power2round(r: int) -> tuple[int, int]:
    r1 = (r + (1 << 12) - 1) >> 13
    r0 = (r - (r1 << 13)) % Q
    return r1, r0


class TestAccumulated:
    @pytest.mark.slow
    def test_accumulated(self) -> None:
        expected = "f930663417278156ab05d940294a77210a809c924d8ab63ec72f4526247602c7"
        o = hashlib.shake_128()

        p44 = ParameterSet.ML_DSA_44.value
        p65 = ParameterSet.ML_DSA_65.value

        for batch in range(0, Q, N):
            size = min(N, Q - batch)
            xs = list(range(batch, batch + size))
            cs = [F(x) for x in xs]
            # Pad to N for Poly if this is the last (partial) batch.
            if size < N:
                cs_padded = cs + [F(0)] * (N - size)
            else:
                cs_padded = cs
            w = Poly(cs_padded)
            h0 = [0] * N
            h1 = [1] * N
            w1_44_h0 = use_hint(w, h0, p44)
            w1_44_h1 = use_hint(w, h1, p44)
            w1_65_h0 = use_hint(w, h0, p65)
            w1_65_h1 = use_hint(w, h1, p65)

            for i in range(size):
                x = xs[i]
                o.update(f"{centered_mod(x, Q)}\n".encode())
                o.update(f"{F(x).infinity_norm()}\n".encode())
                hi, lo = power2round(x)
                o.update(f"{hi}\n".encode())
                o.update(f"{lo}\n".encode())
                r1, r0 = decompose(F(x), p44)
                assert w1_44_h0[i] == r1
                o.update(f"{r1}\n".encode())
                o.update(f"{w1_44_h1[i]}\n".encode())
                o.update(f"{r0}\n".encode())
                o.update(f"{abs(r0)}\n".encode())
                r1, r0 = decompose(F(x), p65)
                assert w1_65_h0[i] == r1
                o.update(f"{r1}\n".encode())
                o.update(f"{w1_65_h1[i]}\n".encode())
                o.update(f"{r0}\n".encode())
                o.update(f"{abs(r0)}\n".encode())

        got = o.hexdigest(32)
        assert got == expected, f"got {got}, expected {expected}"
