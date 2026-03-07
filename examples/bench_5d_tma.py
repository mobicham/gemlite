"""
Standalone benchmark: compare pointer-based vs 5D TMA scale loading in block-scaled GEMM.
Tests the NVFP4 case (group_size=16, e4m3 scales).
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

device = "cuda:0"
dtype = torch.bfloat16

# Required for TMA tensor descriptors
from typing import Optional
def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)
triton.set_allocator(alloc_fn)


def preshuffle_scales(scales_2d, N, K_S):
    """Convert [N, K_S] scales to 5D preshuffled layout for TMA.

    Follows the Triton tutorial layout: [1, N//128, K_S//4, 2, 256]
    Preserves dtype (fp8_e4m3fn for NVFP4, uint8 for MXFP4).
    """
    return (
        scales_2d
        .reshape(N // 128, 4, 32, K_S // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(1, N // 128, K_S // 4, 2, 256)
        .contiguous()
    )


# Kernel with pointer-based scale loading (current gemlite approach)
@triton.jit
def gemm_fp4_pointer_scales(
    a_ptr, b_ptr, c_ptr,
    scales_b_ptr, scales_a_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bn: tl.constexpr, stride_bk: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    stride_sb_n: tl.constexpr, stride_sb_g: tl.constexpr,
    stride_sa_m: tl.constexpr, stride_sa_g: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    meta_scale_norm: tl.constexpr = 0.0025,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    BLOCK_K_A: tl.constexpr = BLOCK_K // 2  # packed FP4
    BLOCK_K_B: tl.constexpr = BLOCK_K // 2
    BLOCK_K_S: tl.constexpr = BLOCK_K // group_size

    # TMA for data
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K // 2], [stride_am, stride_ak], [BLOCK_M, BLOCK_K_A])
    b_desc = tl.make_tensor_descriptor(b_ptr, [N, K // 2], [stride_bn, stride_bk], [BLOCK_N, BLOCK_K_B])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, stride_cn], [BLOCK_M, BLOCK_N])

    # Pointer-based scales
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k_s = tl.arange(0, BLOCK_K_S)
    scales_b_ptrs = scales_b_ptr + offs_n[:, None] * stride_sb_n + offs_k_s[None, :] * stride_sb_g

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    scales_a_ptrs = scales_a_ptr + offs_m[:, None] * stride_sa_m + offs_k_s[None, :] * stride_sa_g

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k = tl.cdiv(K, BLOCK_K)
    for k in tl.range(num_k, num_stages=NUM_STAGES):
        a = tl.load_tensor_descriptor(a_desc, [pid_m * BLOCK_M, k * BLOCK_K_A])
        b = tl.load_tensor_descriptor(b_desc, [pid_n * BLOCK_N, k * BLOCK_K_B]).T

        k_m = k * BLOCK_K_S
        scales_b = tl.load(scales_b_ptrs + k_m * stride_sb_g)
        scales_a = tl.load(scales_a_ptrs + k_m * stride_sa_g)

        acc = tl.dot_scaled(a, scales_a, "e2m1", b, scales_b, "e2m1", acc)

    if group_size == 16:
        acc *= meta_scale_norm

    tl.store_tensor_descriptor(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], value=acc)


# Kernel with 5D TMA scale loading (tutorial approach)
@triton.jit
def gemm_fp4_5d_tma_scales(
    a_ptr, b_ptr, c_ptr,
    scales_b_5d_ptr, scales_a_5d_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bn: tl.constexpr, stride_bk: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    sb_s0: tl.constexpr, sb_s1: tl.constexpr, sb_s2: tl.constexpr, sb_s3: tl.constexpr, sb_s4: tl.constexpr,
    sa_s0: tl.constexpr, sa_s1: tl.constexpr, sa_s2: tl.constexpr, sa_s3: tl.constexpr, sa_s4: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    meta_scale_norm: tl.constexpr = 0.0025,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    BLOCK_K_A: tl.constexpr = BLOCK_K // 2
    BLOCK_K_B: tl.constexpr = BLOCK_K // 2
    BLOCK_K_S: tl.constexpr = BLOCK_K // group_size
    VEC_SIZE: tl.constexpr = group_size

    rep_m: tl.constexpr = BLOCK_M // 128
    rep_n: tl.constexpr = BLOCK_N // 128
    rep_k: tl.constexpr = BLOCK_K // VEC_SIZE // 4

    # TMA for data
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K // 2], [stride_am, stride_ak], [BLOCK_M, BLOCK_K_A])
    b_desc = tl.make_tensor_descriptor(b_ptr, [N, K // 2], [stride_bn, stride_bk], [BLOCK_N, BLOCK_K_B])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, stride_cn], [BLOCK_M, BLOCK_N])

    # 5D TMA for scales
    scales_b_shape1: tl.constexpr = N // 128
    scales_b_shape2: tl.constexpr = K // VEC_SIZE // 4
    scales_b_desc = tl.make_tensor_descriptor(
        scales_b_5d_ptr,
        [1, scales_b_shape1, scales_b_shape2, 2, 256],
        [sb_s0, sb_s1, sb_s2, sb_s3, sb_s4],
        [1, rep_n, rep_k, 2, 256],
    )

    scales_a_shape1: tl.constexpr = M // 128
    scales_a_shape2: tl.constexpr = K // VEC_SIZE // 4
    scales_a_desc = tl.make_tensor_descriptor(
        scales_a_5d_ptr,
        [1, scales_a_shape1, scales_a_shape2, 2, 256],
        [sa_s0, sa_s1, sa_s2, sa_s3, sa_s4],
        [1, rep_m, rep_k, 2, 256],
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k = tl.cdiv(K, BLOCK_K)
    for k in tl.range(num_k, num_stages=NUM_STAGES):
        a = tl.load_tensor_descriptor(a_desc, [pid_m * BLOCK_M, k * BLOCK_K_A])
        b = tl.load_tensor_descriptor(b_desc, [pid_n * BLOCK_N, k * BLOCK_K_B]).T

        # 5D TMA scale loads
        scale_b_raw = tl.load_tensor_descriptor(scales_b_desc, [0, pid_n * rep_n, k * rep_k, 0, 0])
        scales_b = scale_b_raw.reshape(rep_n, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K_S)

        scale_a_raw = tl.load_tensor_descriptor(scales_a_desc, [0, pid_m * rep_m, k * rep_k, 0, 0])
        scales_a = scale_a_raw.reshape(rep_m, rep_k, 32, 4, 4).trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K_S)

        acc = tl.dot_scaled(a, scales_a, "e2m1", b, scales_b, "e2m1", acc)

    if group_size == 16:
        acc *= meta_scale_norm

    tl.store_tensor_descriptor(c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], value=acc)


def bench(M, N, K, group_size=16, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, NUM_STAGES=4, num_warps=4):
    VEC_SIZE = group_size
    K_S = K // group_size

    # Create random FP4 data (packed as uint8)
    a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)
    b = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)

    # 2D scales for pointer-based kernel
    scales_b_2d = torch.randn(N, K_S, device=device).to(torch.float8_e4m3fn)  # [N, K_S]
    scales_a_2d = torch.randn(M, K_S, device=device).to(torch.float8_e4m3fn)  # [M, K_S]

    # Transposed view (matching gemlite's current layout)
    scales_b_T = scales_b_2d.T  # [K_S, N] with strides (1, K_S)
    scales_a_T = scales_a_2d.T  # not used directly, pointer from original

    # 5D preshuffled scales (keep fp8_e4m3fn dtype for NVFP4)
    scales_b_5d = preshuffle_scales(scales_b_2d, N, K_S)
    scales_a_5d = preshuffle_scales(scales_a_2d, M, K_S)

    c_ptr = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    c_5d = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Pointer-based kernel
    def run_pointer():
        gemm_fp4_pointer_scales[grid](
            a, b, c_ptr,
            scales_b_T, scales_a_2d,  # scales_b is transposed, scales_a is row-major
            M, N, K, group_size,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c_ptr.stride(0), c_ptr.stride(1),
            scales_b_T.stride(0), scales_b_T.stride(1),  # stride_sb_n=1, stride_sb_g=K_S
            scales_a_2d.stride(0), scales_a_2d.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_STAGES=NUM_STAGES,
            num_warps=num_warps,
        )

    # 5D TMA kernel
    def run_5d_tma():
        gemm_fp4_5d_tma_scales[grid](
            a, b, c_5d,
            scales_b_5d, scales_a_5d,
            M, N, K, group_size,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c_5d.stride(0), c_5d.stride(1),
            scales_b_5d.stride(0), scales_b_5d.stride(1), scales_b_5d.stride(2), scales_b_5d.stride(3), scales_b_5d.stride(4),
            scales_a_5d.stride(0), scales_a_5d.stride(1), scales_a_5d.stride(2), scales_a_5d.stride(3), scales_a_5d.stride(4),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_STAGES=NUM_STAGES,
            num_warps=num_warps,
        )

    ms_ptr = triton.testing.do_bench(run_pointer, warmup=200, rep=500)
    ms_5d = triton.testing.do_bench(run_5d_tma, warmup=200, rep=500)

    flops = 2.0 * M * N * K
    tflops_ptr = flops / (ms_ptr * 1e-3) / 1e12
    tflops_5d = flops / (ms_5d * 1e-3) / 1e12

    print(f"  Pointer scales: {ms_ptr:.3f} ms, {tflops_ptr:.1f} TFLOP/s")
    print(f"  5D TMA scales:  {ms_5d:.3f} ms, {tflops_5d:.1f} TFLOP/s")
    print(f"  Speedup: {ms_ptr / ms_5d:.3f}x")
    return ms_ptr, ms_5d


if __name__ == "__main__":
    M, N, K = 8192, 16384, 16384
    group_size = 16  # NVFP4

    configs = [
        # (BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, num_warps)
        # Best so far: 128x128x128, 3 stages, 4 warps = 1217.5 TFLOP/s
        (128, 128, 128, 2, 4),
        (128, 128, 128, 3, 4),
        (128, 128, 128, 3, 8),
        (128, 128, 128, 4, 4),
        (128, 128, 128, 5, 4),
        (128, 128, 256, 2, 4),
        (128, 128, 256, 2, 8),
        (128, 256, 128, 2, 4),
        (128, 256, 128, 2, 8),
        (128, 256, 128, 3, 4),
        (128, 256, 256, 2, 4),
        (128, 256, 256, 2, 8),
    ]

    print(f"M={M}, N={N}, K={K}, group_size={group_size}")
    for bm, bn, bk, ns, nw in configs:
        rep_k = bk // group_size // 4
        if rep_k < 1:
            print(f"\n  Skipping BLOCK_M={bm}, BLOCK_N={bn}, BLOCK_K={bk} (rep_k < 1)")
            continue
        print(f"\n  BLOCK_M={bm}, BLOCK_N={bn}, BLOCK_K={bk}, stages={ns}, warps={nw}")
        try:
            bench(M, N, K, group_size, bm, bn, bk, ns, nw)
        except Exception as e:
            print(f"  FAILED: {e}")
