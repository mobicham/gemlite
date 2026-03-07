"""
Benchmark activation quantization kernels:
  - gemlite MXFP4 (v1, v2, v3)
  - gemlite NVFP4 (v1, v2, v3)
  - flashinfer nvfp4_quantize
"""
import torch
import triton

torch.manual_seed(0)
device = "cuda:0"
dtype = torch.bfloat16

# ---- gemlite quant kernels ----
from gemlite.quant_utils import (
    scale_activations_mxfp4_triton as mxfp4_v1,
    scale_activations_mxfp4_triton_v2 as mxfp4_v2,
    scale_activations_mxfp4_triton_v3 as mxfp4_v3,
    scale_activations_nvfp4_triton as nvfp4_v1,
    scale_activations_nvfp4_triton_v2 as nvfp4_v2,
    scale_activations_nvfp4_triton_v3 as nvfp4_v3,
)

# ---- flashinfer ----
from flashinfer import nvfp4_quantize, SfLayout

def flashinfer_nvfp4_quant(x):
    global_sf = (448.0 * 6.0) / x.float().abs().nan_to_num().amax().clamp(min=1e-12)
    return nvfp4_quantize(x, global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)

def flashinfer_nvfp4_quant_no_scale(x):
    """Just the quantize kernel, pre-computed global scale."""
    return nvfp4_quantize(x, x._global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)

# ---- benchmark ----
KERNELS = {
    "gemlite mxfp4 v1": mxfp4_v1,
    "gemlite mxfp4 v2": mxfp4_v2,
    "gemlite mxfp4 v3": mxfp4_v3,
    "gemlite nvfp4 v1": nvfp4_v1,
    "gemlite nvfp4 v2": nvfp4_v2,
    "gemlite nvfp4 v3": nvfp4_v3,
    "flashinfer nvfp4 (with global_sf)": flashinfer_nvfp4_quant,
    "flashinfer nvfp4 (kernel only)": None,  # special case
}

shapes = [
    (1024,  4096),
    (1024,  16384),
    (4096,  4096),
    (4096,  16384),
    (8192,  4096),
    (8192,  16384),
    (16384, 16384),
]

print(f"{'Kernel':<40} {'Shape':>14} {'Time (us)':>10} {'GB/s':>8}")
print("=" * 76)

for M, K in shapes:
    x = torch.randn(M, K, device=device, dtype=dtype)
    # Pre-compute for flashinfer kernel-only variant
    global_sf = (448.0 * 6.0) / x.float().abs().nan_to_num().amax().clamp(min=1e-12)
    x._global_sf = global_sf

    bytes_read = M * K * x.element_size()  # input bytes

    for name, fn in KERNELS.items():
        if name == "flashinfer nvfp4 (kernel only)":
            fn_bench = lambda: nvfp4_quantize(x, global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        else:
            fn_bench = lambda fn=fn: fn(x)

        try:
            ms = triton.testing.do_bench(fn_bench, warmup=200, rep=200)
            us = ms * 1000
            gbps = bytes_read / (ms * 1e-3) / 1e9
            print(f"  {name:<38} {str((M,K)):>14} {us:>10.1f} {gbps:>8.1f}")
        except Exception as e:
            print(f"  {name:<38} {str((M,K)):>14} {'FAILED':>10}  {str(e)[:30]}")

    print()
