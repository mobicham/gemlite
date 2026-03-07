"""
v5 NVFP4 activation quant: 2D grid like v3, but with BLOCK_SIZE_K processing
multiple groups per block. Keeps the simplicity of v3 while reducing block count.
"""
import torch
import triton
import triton.language as tl
from typing import Tuple

torch.manual_seed(0)
device = "cuda:0"
dtype = torch.bfloat16

from gemlite.quant_utils import (
    scale_activations_nvfp4_triton_v3 as nvfp4_v3,
    scale_activations_mxfp4_triton_v3 as mxfp4_v3,
    thr_pos,
    NVFP4_META_SCALE,
    next_power_of_2_triton,
)
from flashinfer import nvfp4_quantize, SfLayout


def prune_large_blocks(configs, nargs, **kwargs):
    M = nargs['M']
    K = nargs['K']
    for config in configs:
        bm = config.kwargs['BLOCK_SIZE_M']
        bk = config.kwargs['BLOCK_SIZE_K']
        if bm > M or bk > K:
            continue
        yield config


# ---- NVFP4 v5: 2D grid, multi-group per block ----
@triton.autotune(
    configs=[
        # BLOCK_SIZE_K must be multiple of GROUP_SIZE=16
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 16},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 16},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 16},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 16},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 32},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 32},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 32},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 256}, num_warps=8, num_stages=1),
        # Multi-stage
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 32},  num_warps=8, num_stages=2),
    ],
    key=['M', 'K'],
    prune_configs_by={'early_config_prune': prune_large_blocks},
)
@triton.jit
def scale_activations_nvfp4_kernel_v5(
    tensor_ptr, out_ptr, scales_ptr, thr_pos_ptr,
    M, K,
    stride_m_t: tl.constexpr, stride_k_t: tl.constexpr,
    stride_m_s: tl.constexpr, stride_k_s: tl.constexpr,
    stride_m_o: tl.constexpr, stride_k_o: tl.constexpr,
    eps: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    meta_scales: tl.constexpr = NVFP4_META_SCALE,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    fp8_dtype: tl.constexpr = tl.float8e4nv
    max_fp8: tl.constexpr = 448.
    HALF_BLOCK_K: tl.constexpr = BLOCK_SIZE_K // 2
    GROUPS_PER_BLOCK: tl.constexpr = BLOCK_SIZE_K // GROUP_SIZE
    FLAT_M: tl.constexpr = BLOCK_SIZE_M * GROUPS_PER_BLOCK
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty

    thr0 = tl.load(thr_pos_ptr + 0)
    thr1 = tl.load(thr_pos_ptr + 1)
    thr2 = tl.load(thr_pos_ptr + 2)
    thr3 = tl.load(thr_pos_ptr + 3)
    thr4 = tl.load(thr_pos_ptr + 4)
    thr5 = tl.load(thr_pos_ptr + 5)
    thr6 = tl.load(thr_pos_ptr + 6)
    thr7 = tl.load(thr_pos_ptr + 7)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # Load BLOCK_SIZE_K elements (multiple groups)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
    tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Reshape to [FLAT_M, GROUP_SIZE] for per-group reduction
    tensor_flat = tl.reshape(tensor, (FLAT_M, GROUP_SIZE))

    # FP8 scales per group
    abs_max = tl.max(tl.abs(tensor_flat), axis=1, keep_dims=True)
    scales_raw = abs_max / (6. * meta_scales)
    scales_fp8 = tl.minimum(scales_raw, max_fp8).to(fp8_dtype)
    scales_full = tl.maximum(scales_fp8.to(tl.float32) * meta_scales, eps)

    # Scalar threshold comparisons
    wq = tensor_flat / scales_full
    abs_wq = tl.abs(wq)
    idx_abs = ((abs_wq > thr0).to(tl.int32) + (abs_wq > thr1).to(tl.int32) +
               (abs_wq > thr2).to(tl.int32) + (abs_wq > thr3).to(tl.int32) +
               (abs_wq > thr4).to(tl.int32) + (abs_wq > thr5).to(tl.int32) +
               (abs_wq > thr6).to(tl.int32) + (abs_wq > thr7).to(tl.int32))
    out = tl.where(wq >= 0, idx_abs, idx_abs + 8).to(out_dtype)

    # Reshape back to [BLOCK_SIZE_M, BLOCK_SIZE_K] and pack pairs
    out = tl.reshape(out, (BLOCK_SIZE_M, BLOCK_SIZE_K))
    lo, hi = tl.split(out.reshape((BLOCK_SIZE_M, HALF_BLOCK_K, 2), can_reorder=False))
    out = lo | (hi << 4)

    # Store packed output
    offs_k_out = pid_k * HALF_BLOCK_K + tl.arange(0, HALF_BLOCK_K)
    out_mask = ((offs_m[:, None] < M) & (offs_k_out[None, :] < (K // 2))).to(tl.int1)
    tl.store(out_ptr + (offs_m[:, None] * stride_m_o + offs_k_out[None, :] * stride_k_o), out, mask=out_mask)

    # Store scales [BLOCK_SIZE_M, GROUPS_PER_BLOCK]
    scales_2d = tl.reshape(scales_fp8, (BLOCK_SIZE_M, GROUPS_PER_BLOCK))
    base_group = pid_k * GROUPS_PER_BLOCK
    offs_g = base_group + tl.arange(0, GROUPS_PER_BLOCK)
    g_mask = offs_g < tl.cdiv(K, GROUP_SIZE)
    tl.store(
        scales_ptr + offs_m[:, None] * stride_m_s + offs_g[None, :] * stride_k_s,
        scales_2d, mask=(offs_m[:, None] < M) & g_mask[None, :]
    )


def scale_activations_nvfp4_v5(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    group_size: int = 16
    eps: float = 1e-6
    fp8_dtype = torch.float8_e4m3fn

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K // 2), device=tensor.device, dtype=torch.uint8)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=fp8_dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(K, meta['BLOCK_SIZE_K']))
    device_index = tensor.device.index

    scale_activations_nvfp4_kernel_v5[grid](
        tensor, out, scales, thr_pos[device_index],
        M, K,
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        eps=eps,
        GROUP_SIZE=group_size,
    )
    return out, scales


# Same for MXFP4
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 32},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 32},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 32},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 256}, num_warps=8, num_stages=1),
        # Multi-stage
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 32},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_K': 64},  num_warps=8, num_stages=2),
    ],
    key=['M', 'K'],
    prune_configs_by={'early_config_prune': prune_large_blocks},
)
@triton.jit
def scale_activations_mxfp4_kernel_v5(
    tensor_ptr, out_ptr, scales_ptr, thr_pos_ptr,
    M, K,
    stride_m_t: tl.constexpr, stride_k_t: tl.constexpr,
    stride_m_s: tl.constexpr, stride_k_s: tl.constexpr,
    stride_m_o: tl.constexpr, stride_k_o: tl.constexpr,
    eps_exp: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    HALF_BLOCK_K: tl.constexpr = BLOCK_SIZE_K // 2
    GROUPS_PER_BLOCK: tl.constexpr = BLOCK_SIZE_K // GROUP_SIZE
    FLAT_M: tl.constexpr = BLOCK_SIZE_M * GROUPS_PER_BLOCK
    out_dtype: tl.constexpr = out_ptr.dtype.element_ty

    thr0 = tl.load(thr_pos_ptr + 0)
    thr1 = tl.load(thr_pos_ptr + 1)
    thr2 = tl.load(thr_pos_ptr + 2)
    thr3 = tl.load(thr_pos_ptr + 3)
    thr4 = tl.load(thr_pos_ptr + 4)
    thr5 = tl.load(thr_pos_ptr + 5)
    thr6 = tl.load(thr_pos_ptr + 6)
    thr7 = tl.load(thr_pos_ptr + 7)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).to(tl.int1)
    tensor_ptrs = tensor_ptr + (offs_m[:, None] * stride_m_t + offs_k[None, :] * stride_k_t)
    tensor = tl.load(tensor_ptrs, mask=mask, other=0.0).to(tl.float32)

    tensor_flat = tl.reshape(tensor, (FLAT_M, GROUP_SIZE))

    scales, scales_log2 = next_power_of_2_triton(
        tl.max(tl.abs(tensor_flat), axis=1, keep_dims=True) / 6., eps_exp
    )

    wq = tensor_flat / scales
    abs_wq = tl.abs(wq)
    idx_abs = ((abs_wq > thr0).to(tl.int32) + (abs_wq > thr1).to(tl.int32) +
               (abs_wq > thr2).to(tl.int32) + (abs_wq > thr3).to(tl.int32) +
               (abs_wq > thr4).to(tl.int32) + (abs_wq > thr5).to(tl.int32) +
               (abs_wq > thr6).to(tl.int32) + (abs_wq > thr7).to(tl.int32))
    out = tl.where(wq >= 0, idx_abs, idx_abs + 8).to(out_dtype)

    out = tl.reshape(out, (BLOCK_SIZE_M, BLOCK_SIZE_K))
    lo, hi = tl.split(out.reshape((BLOCK_SIZE_M, HALF_BLOCK_K, 2), can_reorder=False))
    out = lo | (hi << 4)

    offs_k_out = pid_k * HALF_BLOCK_K + tl.arange(0, HALF_BLOCK_K)
    out_mask = ((offs_m[:, None] < M) & (offs_k_out[None, :] < (K // 2))).to(tl.int1)
    tl.store(out_ptr + (offs_m[:, None] * stride_m_o + offs_k_out[None, :] * stride_k_o), out, mask=out_mask)

    scales_2d = tl.reshape(scales_log2, (BLOCK_SIZE_M, GROUPS_PER_BLOCK))
    base_group = pid_k * GROUPS_PER_BLOCK
    offs_g = base_group + tl.arange(0, GROUPS_PER_BLOCK)
    g_mask = offs_g < tl.cdiv(K, GROUP_SIZE)
    tl.store(
        scales_ptr + offs_m[:, None] * stride_m_s + offs_g[None, :] * stride_k_s,
        scales_2d, mask=(offs_m[:, None] < M) & g_mask[None, :]
    )


def scale_activations_mxfp4_v5(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    group_size: int = 32
    eps_exp: int = -30

    tensor = tensor.contiguous()
    tensor = tensor.view(-1, tensor.shape[-1])
    M, K = tensor.shape

    pad_m = (group_size - M % group_size) % group_size
    M_padded = M + pad_m

    out = torch.empty((M, K // 2), device=tensor.device, dtype=torch.uint8)
    scales = torch.empty((M_padded, K // group_size), device=tensor.device, dtype=torch.uint8)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(K, meta['BLOCK_SIZE_K']))
    device_index = tensor.device.index

    scale_activations_mxfp4_kernel_v5[grid](
        tensor, out, scales, thr_pos[device_index],
        M, K,
        tensor.stride(0), tensor.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        eps_exp=eps_exp,
        GROUP_SIZE=group_size,
    )
    return out, scales


# ---- Benchmark ----
def flashinfer_nvfp4_kernel_only(x, global_sf):
    return nvfp4_quantize(x, global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)


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
    "gemlite mxfp4 v3": mxfp4_v3,
    "gemlite mxfp4 v5": scale_activations_mxfp4_v5,
    "gemlite nvfp4 v3": nvfp4_v3,
    "gemlite nvfp4 v5": scale_activations_nvfp4_v5,
    "flashinfer nvfp4 (kernel only)": None,
}

print(f"{'Kernel':<40} {'Shape':>14} {'Time (us)':>10} {'GB/s':>8}")
print("=" * 76)

for M, K in shapes:
    x = torch.randn(M, K, device=device, dtype=dtype)
    global_sf = (448.0 * 6.0) / x.float().abs().nan_to_num().amax().clamp(min=1e-12)
    bytes_read = M * K * x.element_size()

    for name, fn in KERNELS.items():
        if name == "flashinfer nvfp4 (kernel only)":
            fn_bench = lambda: flashinfer_nvfp4_kernel_only(x, global_sf)
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
