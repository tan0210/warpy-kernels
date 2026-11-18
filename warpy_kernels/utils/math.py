# type: ignore

from cutlass import Float32, cute


@cute.jit
def gelu(x):
    return (
        Float32(0.5)
        * x
        * (Float32(1.0) + cute.math.erf(x * Float32(0.7071067811865476)))
    )


@cute.jit
def dgelu(x):
    cdf = Float32(0.5) * (Float32(1.0) + cute.math.erf(x * Float32(0.7071067811865476)))
    pdf = Float32(0.3989422804014327) * cute.math.exp2(
        x * x * Float32(-0.7213475204444817), fastmath=True
    )
    return cdf + x * pdf
