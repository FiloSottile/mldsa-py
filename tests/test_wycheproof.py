"""Wycheproof test vectors for ML-DSA signature verification.

Test vectors from https://github.com/C2SP/wycheproof at commit d1b0cb0337202fa113b735b0e119f451af3c542d.
"""

import json
from pathlib import Path

import pytest

from mldsa import Parameters, VerificationError, VerificationKey

TESTDATA = Path(__file__).parent / "testdata"

PARAM_MAP = {
    "ML-DSA-44": Parameters.ML_DSA_44,
    "ML-DSA-65": Parameters.ML_DSA_65,
    "ML-DSA-87": Parameters.ML_DSA_87,
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
                vectors.append(pytest.param(
                    params, pk_hex, msg, ctx, sig, result, flags,
                    id=f"{algorithm}-{tc_id}-{comment}",
                ))
    return vectors


@pytest.mark.parametrize("params,pk_hex,msg,ctx,sig,result,flags", load_verify_vectors())
def test_wycheproof_verify(params, pk_hex, msg, ctx, sig, result, flags):
    pk = bytes.fromhex(pk_hex)
    if result == "valid":
        vk = VerificationKey(pk, parameters=params)
        vk.verify(msg, sig, context=ctx)
    elif result == "invalid":
        with pytest.raises((VerificationError, ValueError)):
            vk = VerificationKey(pk, parameters=params)
            vk.verify(msg, sig, context=ctx)
    elif result == "acceptable":
        # Acceptable results may pass or fail; just ensure no crash.
        try:
            vk = VerificationKey(pk, parameters=params)
            vk.verify(msg, sig, context=ctx)
        except (VerificationError, ValueError):
            pass
