# mldsa-py

```
pip install mldsa
```

This is a pure-Python production implementation of ML-DSA (FIPS 204)
post-quantum signature verification.

It does not provide key or signature generation, because secrets can't be
handled in constant-time in Python.

```python
from mldsa import VerificationKey, VerificationError

vk = VerificationKey(verification_key_bytes)

try:
    vk.verify(signature, message)
except VerificationError:
    print("invalid signature!")
```

The parameter set (ML-DSA-44, ML-DSA-65, or ML-DSA-87) is inferred from the
verification key size, or it can be specified explicitly.

```python
from mldsa import ParameterSet, VerificationKey

vk = VerificationKey(verification_key_bytes, parameters=ParameterSet.ML_DSA_87)
vk.verify(signature, message, context=b"example.com/foo token")
```

The non-test code is [a single-file module](https://github.com/FiloSottile/mldsa-py/blob/main/src/mldsa/mldsa.py)
of less than 500 lines, with no dependencies.

It works with Python 3.8 and later.

## Development

To run tests, use

```bash
uv run ruff check
uv run ty check
uv run pytest
go install github.com/FiloSottile/mostly-harmless/muzoo@latest
muzoo -mutations tests/testdata/mutations test -- uv run pytest -x
```

This project uses tests from [Wycheproof](https://github.com/C2SP/wycheproof).

## License

This work is marked CC0 1.0 Universal. To view a copy of this mark, visit
[creativecommons.org](https://creativecommons.org/publicdomain/zero/1.0/).

Alternatively, you may use this source code under the terms of the 0BSD license
that can be found in the LICENSE file.
In short, you can do whatever you want with this code.
