# warpy-kernels

CuTe DSL kernels for local LLMs. Currently supports SM120 only; further support
will be made for SM89 and SM80.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install warpy-kernels
```

```python
from warpy_kernels import attn_bwd, attn_fwd, gemm
```
