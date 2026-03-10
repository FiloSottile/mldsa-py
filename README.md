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

vk = VerificationKey(public_key_bytes)

try:
    vk.verify(message, signature)
except VerificationError:
    print("invalid signature!")
```

The parameter set (ML-DSA-44, ML-DSA-65, or ML-DSA-87) is inferred from the
public key size, or it can be specified explicitly.

```python
from mldsa import Parameters, VerificationKey

vk = VerificationKey(public_key_bytes, parameters=Parameters.ML_DSA_87)
vk.verify(message, signature, context=b"example.com/foo token")
```

The non-test code is [a single-file module](src/mldsa/mldsa.py) of less than 400
lines, with no dependencies.

It works with Python 3.8 and later.

## Development

To run tests, use

```bash
uv run ruff check
uv run ty check
uv run pytest
```

This project uses tests from [Wycheproof](https://github.com/C2SP/wycheproof).

## License

This work is marked CC0 1.0 Universal. To view a copy of this mark, visit
[creativecommons.org](https://creativecommons.org/publicdomain/zero/1.0/).

Alternatively, you may use this source code under the terms of the 0BSD license
that can be found in the LICENSE file.
In short, you can do whatever you want with this code.
