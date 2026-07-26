from __future__ import annotations

import argparse
import json
import timeit
from pathlib import Path
from typing import Callable, Iterator

from mldsa import ParameterSet, VerificationKey

TESTDATA = Path(__file__).parent / "testdata"


def benchmark(name: str, fn: Callable[[], None]) -> None:
    """Time *fn* and print one result line, in the format of Go benchmarks."""
    n, elapsed = timeit.Timer(fn).autorange()
    print(f"Benchmark{name}\t{n:8d}\t{elapsed / n * 1e9:12.0f} ns/op", flush=True)


def benchmark_verify(parameters: ParameterSet, vk: bytes, sig: bytes, msg: bytes, ctx: bytes) -> None:
    def whole() -> None:
        VerificationKey(vk).verify(sig, msg, context=ctx)

    benchmark(f"Verify/{parameters}/Whole", whole)

    key = VerificationKey(vk)

    def precomputed() -> None:
        key.verify(sig, msg, context=ctx)

    benchmark(f"Verify/{parameters}/Precomputed", precomputed)


def valid_vectors() -> Iterator[tuple[ParameterSet, bytes, bytes, bytes, bytes]]:
    """Yield one valid vector for each parameter set.

    Yields:
        (parameters, vk, sig, msg, ctx) tuples.
    """
    parameter_sets = {str(p): p for p in ParameterSet}
    for filename in sorted(TESTDATA.glob("mldsa_*_verify_test.json")):
        data = json.loads(filename.read_text())
        group, test = next(
            (g, t) for g in data["testGroups"] for t in g["tests"] if t["result"] == "valid"
        )
        yield (
            parameter_sets[data["algorithm"]],
            bytes.fromhex(group["publicKey"]),
            bytes.fromhex(test["sig"]),
            bytes.fromhex(test["msg"]),
            bytes.fromhex(test.get("ctx", "")),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-count", type=int, default=1, help="run each benchmark N times")
    args = parser.parse_args()

    vectors = list(valid_vectors())
    for _ in range(args.count):
        for vector in vectors:
            benchmark_verify(*vector)
