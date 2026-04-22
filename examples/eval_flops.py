# For blackwell, you need to export ptxas: TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.0/bin/ptxas 
import torch
import time, gc
import gemlite
from gemlite.helper import *
import argparse
import torch._dynamo
torch._dynamo.config.recompile_limit = 256
import torch._inductor.config as _inductor_config
import triton

device, dtype = 'cuda:0', torch.bfloat16
repeat = 32

gemlite.reset_config()
#gemlite.enable_cudagraph_autotune(True)
#gemlite.enable_tma(True)
#gemlite.set_fast_nvfp4(True)
#gemlite.set_ptx_fp4_pack(True) # Requires ptxas CUDA 13+ via ptxas-blackwell / TRITON_PTXAS_BLACKWELL_PATH
#gemlite.set_autotune("max")
#gemlite.core.enable_activation_scaling(2)

def get_model(K, N, repeat=repeat):
    torch.manual_seed(0)
    model = torch.nn.Sequential(*[
        torch.nn.Linear(N, K, dtype=dtype, device=device, bias=False)
        for _ in range(repeat)
    ])
    model.requires_grad_(False)
    return model


@torch.no_grad()
def eval_model(model, M, K, iters=50, verbose=False):
    torch.manual_seed(0)
    t = []
    for i in range(iters):
        x = torch.randn(M, K, dtype=dtype, device=device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = model(x)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        _time = (t2 - t1) * 1000
        t.append(_time)
        if verbose:
            print(f"Took: {_time} ms")
        del x
        torch.cuda.empty_cache()
    t = t[-(iters // 2):]
    time_torch = (sum(t) / len(t))
    gc.collect()
    return time_torch


def get_flops(M, K, N, perf_time_ms):
    flops_per_linear = 2 * M * N * K
    tflops = flops_per_linear / (perf_time_ms * 1e-3) / 1e12
    return tflops


def cleanup(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


###########################################################################################################################
# Pytorch INT8 dynamic reference
###########################################################################################################################
class NativePyTorchINT8Dynamic(torch.nn.Module):
    def __init__(self, linear_layer):
        super().__init__()
        w_fp16 = linear_layer.weight.data
        self.w_scales = w_fp16.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-5) / 127.0
        w_int8 = torch.round(w_fp16 / self.w_scales).to(torch.int8)
        self.w_int8 = w_int8.contiguous()
        self.w_scales = self.w_scales.view(1, -1)

    def forward(self, x):
        x_scales = x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5) / 127.0
        x_int8 = torch.round(x / x_scales).to(torch.int8)
        out_int32 = torch._int_mm(x_int8, self.w_int8.t())
        return out_int32.to(x.dtype) * (x_scales * self.w_scales)


def patch_model_native_int8(model):
    for i, layer in enumerate(model):
        if isinstance(layer, torch.nn.Linear):
            model[i] = NativePyTorchINT8Dynamic(layer)


###########################################################################################################################
# Pytorch FP8 dynamic reference
###########################################################################################################################
def _to_fp8_and_inv_scale(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    dim: int | tuple[int, ...] | None,
    keepdim: bool,
    clamp_min: float = 1e-12,
):
    finfo = torch.finfo(fp8_dtype)
    x_fp32 = x.float()
    if dim is None:
        amax = x_fp32.abs().amax().clamp(min=clamp_min)
    else:
        amax = x_fp32.abs().amax(dim=dim, keepdim=keepdim).clamp(min=clamp_min)

    scale_gain = (finfo.max / amax)
    x_scaled_sat = (x_fp32 * scale_gain).clamp(min=finfo.min, max=finfo.max)
    x_fp8 = x_scaled_sat.to(fp8_dtype)
    inv_scale = scale_gain.reciprocal().to(torch.float32)
    return x_fp8, inv_scale


class NativePyTorchFP8Dynamic(torch.nn.Module):
    def __init__(
        self,
        linear_layer: torch.nn.Linear,
        fp8_dtype: torch.dtype = torch.float8_e4m3fn,
        use_fast_accum: bool = False,
    ):
        super().__init__()
        self.fp8_dtype = fp8_dtype
        self.use_fast_accum = use_fast_accum

        w_hp = linear_layer.weight.data
        w_fp8, w_inv_scale_row = _to_fp8_and_inv_scale(w_hp, fp8_dtype=fp8_dtype, dim=1, keepdim=True)
        self.register_buffer("w_fp8", w_fp8.contiguous().t())
        self.register_buffer("w_inv_scale", w_inv_scale_row.view(1, -1).contiguous())

        if linear_layer.bias is not None:
            self.register_buffer("bias", linear_layer.bias.data.contiguous())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor):
        x_fp8, x_inv_scale = _to_fp8_and_inv_scale(x, fp8_dtype=self.fp8_dtype, dim=-1, keepdim=True)
        out = torch._scaled_mm(
            x_fp8,
            self.w_fp8,
            scale_a=x_inv_scale,
            scale_b=self.w_inv_scale,
            bias=self.bias,
            out_dtype=x.dtype,
            use_fast_accum=self.use_fast_accum,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out


def patch_model_native_fp8(model, fp8_dtype=torch.float8_e4m3fn, use_fast_accum=False):
    for i, layer in enumerate(model):
        if isinstance(layer, torch.nn.Linear):
            model[i] = NativePyTorchFP8Dynamic(
                layer, fp8_dtype=fp8_dtype, use_fast_accum=use_fast_accum,
            )


###########################################################################################################################
# flashinfer NVFP4 reference (CUTLASS-based, supports sm_120)
###########################################################################################################################
def _get_flashinfer():
    """Check if flashinfer with NVFP4 support is available."""
    try:
        from flashinfer import nvfp4_quantize, mm_fp4, SfLayout
        return True, None
    except ImportError:
        return False, "flashinfer not installed (pip install flashinfer)"


# ---- custom_op wrappers for torch.compile compatibility ----
@torch.library.custom_op("flashinfer_bench::nvfp4_quantize", mutates_args=())
def _nvfp4_quantize_op(
    a: torch.Tensor, a_global_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer import nvfp4_quantize, SfLayout
    a_fp4, a_sf = nvfp4_quantize(a, a_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    return a_fp4, a_sf


@torch.library.register_fake("flashinfer_bench::nvfp4_quantize")
def _nvfp4_quantize_fake(
    a: torch.Tensor, a_global_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    M, K = a.shape
    a_fp4 = torch.empty((M, K // 2), dtype=torch.uint8, device=a.device)
    a_sf = torch.empty((M, K // 16), dtype=torch.uint8, device=a.device)
    return a_fp4, a_sf


@torch.library.custom_op("flashinfer_bench::mm_fp4", mutates_args=())
def _mm_fp4_op(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: torch.Tensor,
    out_N: int,
) -> torch.Tensor:
    from flashinfer import mm_fp4
    return mm_fp4(a, b, a_descale, b_descale, alpha, torch.bfloat16, backend="cutlass")


@torch.library.register_fake("flashinfer_bench::mm_fp4")
def _mm_fp4_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: torch.Tensor,
    out_N: int,
) -> torch.Tensor:
    M = a.shape[0]
    return torch.empty((M, out_N), dtype=torch.bfloat16, device=a.device)


class FlashinferNVFP4Dynamic(torch.nn.Module):
    """
    NVFP4 dynamic quantization using flashinfer CUTLASS backend.
    Weights quantized offline in __init__; activations quantized on-the-fly in forward.
    Compatible with torch.compile via custom_op wrappers.
    """

    def __init__(self, linear_layer: torch.nn.Linear):
        super().__init__()
        from flashinfer import nvfp4_quantize, SfLayout

        w_bf16 = linear_layer.weight.data  # [N, K]
        N, K = w_bf16.shape

        # Quantize weights offline
        w_global_sf = (448.0 * 6.0) / w_bf16.float().abs().nan_to_num().amax().clamp(min=1e-12)
        w_fp4, w_sf = nvfp4_quantize(
            w_bf16, w_global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False
        )

        # Store pre-transposed for mm_fp4: b=[K//2, N], b_descale=[K//16, N]
        self.register_buffer("w_fp4_t", w_fp4.T.contiguous())
        self.register_buffer("w_sf_t", w_sf.T.contiguous())
        self.register_buffer(
            "w_global_sf_inv",
            (1.0 / w_global_sf).to(torch.float32).contiguous(),
        )
        self.N = N

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Activation quantization: compute global scale with pytorch ops
        x_global_sf = (448.0 * 6.0) / x.float().abs().nan_to_num().amax().clamp(min=1e-12)

        # Quantize activation via custom_op (flashinfer CUDA kernel)
        x_fp4, x_sf = torch.ops.flashinfer_bench.nvfp4_quantize(x, x_global_sf)

        # alpha = 1 / (x_global_sf * w_global_sf)
        alpha = self.w_global_sf_inv / x_global_sf

        # CUTLASS FP4 matmul via custom_op
        return torch.ops.flashinfer_bench.mm_fp4(
            x_fp4, self.w_fp4_t, x_sf, self.w_sf_t, alpha, self.N
        )


def patch_model_flashinfer_nvfp4(model):
    for i, layer in enumerate(model):
        if isinstance(layer, torch.nn.Linear):
            model[i] = FlashinferNVFP4Dynamic(layer)



###########################################################################################################################
def run_benchmark(proc_name, M, K, N):
    """
    Unified benchmark runner. Returns (label, M, K, N, tflops) or None on skip.
    Handles gemlite processors, native PyTorch INT8/FP8, and flashinfer NVFP4.
    """
    torch._dynamo.reset()
    has_flashinfer, fi_err = _get_flashinfer()

    # ---- flashinfer NVFP4 dynamic (torch.compile + activation quant) ----
    if proc_name == "flashinfer_nvfp4_dynamic":
        if not has_flashinfer:
            print(f"  Skipping {proc_name}: {fi_err}")
            return None
        # Disable cudagraph trees: flashinfer CUTLASS does internal workspace allocs
        old_cudagraph = _inductor_config.triton.cudagraph_trees
        _inductor_config.triton.cudagraph_trees = False

        # NOTE: flashinfer's CUTLASS NVFP4 kernel requires M to be a multiple of 128.
        # When M < 128, we pad M up to 128 so the kernel doesn't crash. The TFLOP/s
        # are computed using the padded M to keep the comparison fair (same actual work).
        M_padded = max(M, 128)
        M_padded = ((M_padded + 127) // 128) * 128

        model = get_model(K, N, repeat=repeat)
        patch_model_flashinfer_nvfp4(model)
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

        perf_time_ms = eval_model(model, M_padded, K) / repeat
        tflops = get_flops(M, K, N, perf_time_ms)
        label = "flashinfer NVFP4 (dynamic)"
        if M_padded != M:
            print(f"  {label} | {M}, {K}, {N} | {tflops:.2f} TFLOP/s  (M padded to {M_padded} internally)")
        else:
            print(f"  {label} | {M}, {K}, {N} | {tflops:.2f} TFLOP/s")

        cleanup(model)
        _inductor_config.triton.cudagraph_trees = old_cudagraph
        return (label, M, K, N, tflops)

    # ---- Native PyTorch INT8 dynamic ----
    if proc_name == "native_int8":
        if M <= 16:
            print(f"  Skipping native_int8 for M={M} (requires M > 16)")
            return None
        model = get_model(K, N, repeat=repeat)
        patch_model_native_int8(model)
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

        perf_time_ms = eval_model(model, M, K) / repeat
        tflops = get_flops(M, K, N, perf_time_ms)
        label = "PyTorch Native INT8"
        print(f"  {label} | {M}, {K}, {N} | {tflops:.2f} TFLOP/s")

        cleanup(model)
        return (label, M, K, N, tflops)

    # ---- Native PyTorch FP8 dynamic ----
    if proc_name == "native_fp8":
        model = get_model(K, N, repeat=repeat)
        patch_model_native_fp8(model, fp8_dtype=torch.float8_e4m3fn, use_fast_accum=False)
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

        perf_time_ms = eval_model(model, M, K) / repeat
        tflops = get_flops(M, K, N, perf_time_ms)
        label = "PyTorch Native FP8"
        print(f"  {label} | {M}, {K}, {N} | {tflops:.2f} TFLOP/s")

        cleanup(model)
        return (label, M, K, N, tflops)

    # ---- GemLite processors + BF16 baseline ----
    GEMLITE_MAP = {
        "A16W8_INT8": lambda: A16W8_INT8(),
        "A16W8_FP8": lambda: A16W8_FP8(),
        "A16W4_HQQ_INT": lambda: A16W4_HQQ_INT(),
        "A8W8_INT8_dynamic": lambda: A8W8_INT8_dynamic(),
        "A8W8_FP8_dynamic": lambda: A8W8_FP8_dynamic(),
        "A8W8_INT8_block_dynamic": lambda: A8W8_INT8_dynamic(block_quant=True),
        "A8W8_FP8_block_dynamic": lambda: A8W8_FP8_dynamic(block_quant=True),
        "A8W8_MXFP_dynamic_post_scale": lambda: A8W8_MXFP_dynamic(dtype=dtype, post_scale=True),
        "A8W8_MXFP_dynamic": lambda: A8W8_MXFP_dynamic(dtype=dtype, post_scale=False),
        "A4W4_MXFP_dynamic": lambda: A4W4_MXFP_dynamic(dtype=dtype),
        "A4W4_NVFP_dynamic": lambda: A4W4_NVFP_dynamic(dtype=dtype),
        "none": lambda: None,
        "fp16": lambda: None,
    }

    if proc_name not in GEMLITE_MAP:
        print(f"  Unknown processor: {proc_name}, skipping.")
        return None

    procesor = GEMLITE_MAP[proc_name]()

    model = get_model(K, N, repeat=repeat)
    if procesor is not None:
        patch_model(model, device=device, processor=procesor)
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    perf_time_ms = eval_model(model, M, K) / repeat
    label = proc_name if procesor is not None else "BF16 (no processor)"
    tflops = get_flops(M, K, N, perf_time_ms)
    print(f"  {label} | {M}, {K}, {N} | {tflops:.2f} TFLOP/s")

    cleanup(model)
    return (label, M, K, N, tflops)


ALL_PROCESSORS = [
    "none",
    "A16W8_INT8",
    "A16W8_FP8",
    "A16W4_HQQ_INT",
    "A8W8_INT8_dynamic",
    "A8W8_FP8_dynamic",
    "A8W8_INT8_block_dynamic",
    "A8W8_FP8_block_dynamic",
    "A8W8_MXFP_dynamic_post_scale",
    "A8W8_MXFP_dynamic",
    "A4W4_MXFP_dynamic",
    "A4W4_NVFP_dynamic",
    "native_int8",
    "native_fp8",
    "flashinfer_nvfp4_dynamic",
]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TFLOP/s for various quantized matmul processors.",
        epilog="""
Examples:
  # Run with default parameters (all processors)
  python3 eval_flops.py

  # Run with specific dimensions:
  python3 eval_flops.py --M 8192 --K 8192 --N 8192

  # Run only specific processors (comma-separated):
  python3 eval_flops.py --processor A4W4_MXFP_dynamic,flashinfer_nvfp4_dynamic,native_fp8

  # Run only BF16 baseline (no quantization):
  python3 eval_flops.py --processor none

  # Available processors:
  #   GemLite:    A16W8_INT8, A16W8_FP8, A16W4_HQQ_INT,
  #               A8W8_INT8_dynamic, A8W8_FP8_dynamic,
  #               A8W8_MXFP_dynamic_post_scale, A8W8_MXFP_dynamic,
  #               A4W4_MXFP_dynamic, A4W4_NVFP_dynamic
  #   PyTorch:    native_int8, native_fp8
  #   flashinfer: flashinfer_nvfp4_dynamic
  #   Baseline:   none / fp16 (BF16, no quantization)
  #   Use "all" to run every processor.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--M", type=str, default="8192", help="Comma-separated M values (e.g., 1,8,16,64,128,256,512,1024,2048,4096)")
    parser.add_argument("--K", type=int, default=8192, help="Input feature dimension")
    parser.add_argument("--N", type=int, default=8192, help="Output feature dimension")
    parser.add_argument("--processor", type=str, default="all",
                        help='Comma-separated processor names or "all" (default: all)')
    args = parser.parse_args()

    M_values = [int(m.strip()) for m in args.M.split(",")]
    K, N = args.K, args.N

    if args.processor == "all":
        processor_names = list(ALL_PROCESSORS)
    else:
        processor_names = [p.strip() for p in args.processor.split(",")]

    # results_map[proc_name][M] = tflops
    results_map = {}
    proc_labels = {}

    for proc_name in processor_names:
        for M in M_values:
            result = run_benchmark(proc_name, M, K, N)
            if result is not None:
                label, m, k, n, tflops = result
                proc_labels[proc_name] = label
                if proc_name not in results_map:
                    results_map[proc_name] = {}
                results_map[proc_name][M] = tflops

    # ---- Summary Table ----
    active_procs = [p for p in processor_names if p in results_map]
    if not active_procs:
        print("No results to display.")
        return

    gpu_name = torch.cuda.get_device_name(device)
    print(f"\n{'=' * 120}")
    print(f"SUMMARY  (GPU: {gpu_name},  K={K}, N={N})  [TFLOP/s]")
    print(f"{'=' * 120}")

    # Column widths
    col_w = max(12, max(len(proc_labels.get(p, p)) for p in active_procs) + 2)
    m_col_w = 8

    # Header
    header = f"{'M':>{m_col_w}}"
    for p in active_procs:
        header += f"  {proc_labels.get(p, p):>{col_w}}"
    print(header)
    print("-" * len(header))

    # Rows
    for M in M_values:
        row = f"{M:>{m_col_w}}"
        for p in active_procs:
            val = results_map.get(p, {}).get(M)
            if val is not None:
                row += f"  {val:>{col_w}.2f}"
            else:
                row += f"  {'--':>{col_w}}"
        print(row)

    print(f"{'=' * len(header)}")


if __name__ == "__main__":
    main()
