"""Wycheproof test vectors for ML-DSA signature verification.

Test vectors from https://github.com/C2SP/wycheproof at commit ee7b4f7e611928cbe163dc6f5e54527bfd166f34.
"""

import json
from pathlib import Path

import pytest

from mldsa import (
    InvalidContextError,
    InvalidVerificationKeyError,
    ParameterSet,
    VerificationError,
    VerificationKey,
)

from .test_mldsa import buffer_variants

TESTDATA = Path(__file__).parent / "testdata"

PARAM_MAP = {
    "ML-DSA-44": ParameterSet.ML_DSA_44,
    "ML-DSA-65": ParameterSet.ML_DSA_65,
    "ML-DSA-87": ParameterSet.ML_DSA_87,
}


def load_verify_vectors():
    """Load all Wycheproof ML-DSA verify test vectors."""
    vectors = []
    for filename in sorted(TESTDATA.glob("mldsa_*_verify_test.json")):
        data = json.loads(filename.read_text())
        algorithm = data["algorithm"]
        params = PARAM_MAP[algorithm]
        for group in data["testGroups"]:
            pk_hex = group["publicKey"]
            for test in group["tests"]:
                tc_id = test["tcId"]
                comment = test["comment"]
                msg = bytes.fromhex(test["msg"])
                ctx = bytes.fromhex(test.get("ctx", ""))
                sig = bytes.fromhex(test["sig"])
                result = test["result"]
                flags = test["flags"]
                vectors.append(
                    pytest.param(
                        params,
                        pk_hex,
                        msg,
                        ctx,
                        sig,
                        result,
                        flags,
                        id=f"{algorithm}-{tc_id}-{comment}",
                    )
                )
    return vectors


def load_valid_vector():
    """Return (params, pk, msg, ctx, sig) for a valid vector with a non-empty context.

    Raises:
        AssertionError: If the test vectors contain no such vector.
    """
    for filename in sorted(TESTDATA.glob("mldsa_*_verify_test.json")):
        data = json.loads(filename.read_text())
        params = PARAM_MAP[data["algorithm"]]
        for group in data["testGroups"]:
            pk = bytes.fromhex(group["publicKey"])
            for test in group["tests"]:
                if test["result"] == "valid" and test.get("ctx"):
                    return (
                        params,
                        pk,
                        bytes.fromhex(test["msg"]),
                        bytes.fromhex(test["ctx"]),
                        bytes.fromhex(test["sig"]),
                    )
    raise AssertionError("no valid test vector with a context found")


def test_verify_accepts_any_buffer_type():
    """A valid signature verifies whichever Buffer implementation carries each input."""
    params, pk, msg, ctx, sig = load_valid_vector()
    vk = VerificationKey(pk, parameters=params)
    vk.verify(sig, msg, context=ctx)
    for buf in buffer_variants(pk):
        VerificationKey(buf, parameters=params).verify(sig, msg, context=ctx)
    for buf in buffer_variants(sig):
        vk.verify(buf, msg, context=ctx)
    for buf in buffer_variants(msg):
        vk.verify(sig, buf, context=ctx)
    for buf in buffer_variants(ctx):
        vk.verify(sig, msg, context=buf)


@pytest.mark.parametrize("params,pk_hex,msg,ctx,sig,result,flags", load_verify_vectors())
def test_wycheproof_verify(params, pk_hex, msg, ctx, sig, result, flags):
    pk = bytes.fromhex(pk_hex)
    if result == "valid":
        vk = VerificationKey(pk, parameters=params)
        vk.verify(sig, msg, context=ctx)
    elif result == "invalid":
        with pytest.raises((VerificationError, InvalidVerificationKeyError, InvalidContextError)):
            vk = VerificationKey(pk, parameters=params)
            vk.verify(sig, msg, context=ctx)
    elif result == "acceptable":
        # Acceptable results may pass or fail; just ensure no crash.
        try:
            vk = VerificationKey(pk, parameters=params)
            vk.verify(sig, msg, context=ctx)
        except (VerificationError, InvalidVerificationKeyError, InvalidContextError):
            pass
