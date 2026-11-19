# warpy-kernels

CuTe DSL kernels for local LLMs. Currently supports SM120 and SM89; further support will be made for SM80 and SM86.

> [!NOTE]
> SM80 is supported albeit poorly. We do not recommend to use this package to target the architecture for now.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install "git+https://github.com/tan0210/warpy-kernels"
```

```python
from warpy_kernels import attn_bwd, attn_fwd, gemm
```
