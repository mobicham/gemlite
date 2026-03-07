"""
Benchmark activation quantization kernels from quant_utils.py (v5 integrated)
vs flashinfer nvfp4_quantize.
"""
import torch
import triton

torch.manual_seed(0)
device = "cuda:0"
dtype = torch.bfloat16

# Import directly from quant_utils (now v5 by default)
from gemlite.quant_utils import (
    scale_activations_mxfp4,       # v5
    scale_activations_nvfp4,       # v5
    scale_activations_mxfp4_triton_v3 as mxfp4_v3,
    scale_activations_nvfp4_triton_v3 as nvfp4_v3,
)

from flashinfer import nvfp4_quantize, SfLayout

shapes = [
    (1024,  4096),
    (1024,  16384),
    (4096,  4096),
    (4096,  16384),
    (8192,  4096),
    (8192,  16384),
    (16384, 16384),
]

KERNELS = {
    "gemlite mxfp4 (default=v5)": scale_activations_mxfp4,
    "gemlite mxfp4 v3 (old)":     mxfp4_v3,
    "gemlite nvfp4 (default=v5)": scale_activations_nvfp4,
    "gemlite nvfp4 v3 (old)":     nvfp4_v3,
    "flashinfer nvfp4 (kernel)":  None,
}

print(f"{'Kernel':<40} {'Shape':>14} {'Time (us)':>10} {'GB/s':>8}")
print("=" * 76)

for M, K in shapes:
    x = torch.randn(M, K, device=device, dtype=dtype)
    global_sf = (448.0 * 6.0) / x.float().abs().nan_to_num().amax().clamp(min=1e-12)
    bytes_read = M * K * x.element_size()

    for name, fn in KERNELS.items():
        if fn is None:
            fn_bench = lambda: nvfp4_quantize(x, global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        else:
            fn_bench = lambda fn=fn: fn(x)

        try:
            ms = triton.testing.do_bench(fn_bench, warmup=200, rep=200)
            us = ms * 1000
            gbps = bytes_read / (ms * 1e-3) / 1e9
            print(f"  {name:<38} {str((M,K)):>14} {us:>10.1f} {gbps:>8.1f}")
        except Exception as e:
            print(f"  {name:<38} {str((M,K)):>14} {'FAILED':>10}  {str(e)[:40]}")

    print()
