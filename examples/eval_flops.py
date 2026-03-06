import torch
import time, gc
import gemlite
from gemlite.helper import *
import argparse
import torch._dynamo
torch._dynamo.config.recompile_limit = 256

device, dtype = 'cuda:0', torch.bfloat16
repeat = 32

#gemlite.reset_cache()
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
    t = t[-(iters // 2):]
    time_torch = (sum(t) / len(t))
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
                layer, fp8_dtype=fp8_dtype, use_fast_accum=use_fast_accum
            )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TFLOP/s for various quantized matmul processors.",
        epilog="""
Examples:
  # Run with default parameters
  python eval_flops.py

  # Run with specific dimensions:
  python eval_flops.py --M 128 --K 4096 --N 4096

  # Run only specific processors (comma-separated):
  python eval_flops.py --processor A16W8_INT8,A8W8_FP8_dynamic

  # Run only BF16 baseline (no quantization):
  python eval_flops.py --processor none

  # Available processors:
  #   A16W8_INT8, A16W8_FP8, A8W8_INT8_dynamic, A8W8_FP8_dynamic,
  #   A8W8_MXFP_dynamic_post_scale, A8W8_MXFP_dynamic_no_post_scale,
  #   A4W4_MXFP_dynamic, A4W4_NVFP_dynamic, none (BF16 baseline)
  #   Use "all" to run every processor.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--M", type=int, default=8192, help="Batch/sequence dimension")
    parser.add_argument("--K", type=int, default=16384, help="Input feature dimension")
    parser.add_argument("--N", type=int, default=16384, help="Output feature dimension")
    parser.add_argument("--processor", type=str, default="all",
                        help='Comma-separated processor names or "all" (default: all)')
    args = parser.parse_args()

    M, K, N = args.M, args.K, args.N

    PROCESSOR_MAP = {
        "A16W8_INT8": lambda: A16W8_INT8(),
        "A16W8_FP8": lambda: A16W8_FP8(),
        "A16W4_HQQ_INT": lambda: A16W4_HQQ_INT(),
        "A8W8_INT8_dynamic": lambda: A8W8_INT8_dynamic(),
        "A8W8_FP8_dynamic": lambda: A8W8_FP8_dynamic(),
        "A8W8_MXFP_dynamic_post_scale": lambda: A8W8_MXFP_dynamic(dtype=dtype, post_scale=True),
        "A8W8_MXFP_dynamic": lambda: A8W8_MXFP_dynamic(dtype=dtype, post_scale=False),
        "A4W4_MXFP_dynamic": lambda: A4W4_MXFP_dynamic(dtype=dtype),
        "A4W4_NVFP_dynamic": lambda: A4W4_NVFP_dynamic(dtype=dtype),
        "none": lambda: None,
        "fp16": lambda: None,
    }

    if args.processor == "all":
        processor_names = list(PROCESSOR_MAP.keys())
    else:
        processor_names = [p.strip() for p in args.processor.split(",")]

    results = []

    # ---- GemLite processors ----
    for proc_name in processor_names:
        if proc_name not in PROCESSOR_MAP:
            print(f"Unknown processor: {proc_name}, skipping.")
            continue

        procesor = PROCESSOR_MAP[proc_name]()

        model = get_model(K, N, repeat=repeat)
        if procesor is not None:
            patch_model(model, device=device, processor=procesor)
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

        perf_time_ms = eval_model(model, M, K) / repeat
        label = proc_name if procesor is not None else "FP16 (no processor)"
        tflops = get_flops(M, K, N, perf_time_ms)
        print(f"Processor: {label} | {M}, {K}, {N} | {tflops:.2f} TFLOP/s")
        results.append((label, M, K, N, tflops))

        cleanup(model)

    # ---- PyTorch Native INT8 dynamic reference ----
    if M >= 16:
        model = get_model(K, N, repeat=repeat)
        patch_model_native_int8(model)
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

        perf_time_ms = eval_model(model, M, K) / repeat
        tflops = get_flops(M, K, N, perf_time_ms)
        print(f"PyTorch Native INT8 | {M}, {K}, {N} | {tflops:.2f} TFLOP/s")
        results.append(("PyTorch Native INT8", M, K, N, tflops))

        cleanup(model)
    else:
        print(f"Skipping PyTorch Native INT8 for M={M} (requires M >= 16).")

    # ---- PyTorch Native FP8 dynamic reference ----
    model = get_model(K, N, repeat=repeat)
    patch_model_native_fp8(model, fp8_dtype=torch.float8_e4m3fn, use_fast_accum=False)
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    perf_time_ms = eval_model(model, M, K) / repeat
    tflops = get_flops(M, K, N, perf_time_ms)
    print(f"PyTorch Native FP8 | {M}, {K}, {N} | {tflops:.2f} TFLOP/s")
    results.append(("PyTorch Native FP8", M, K, N, tflops))

    cleanup(model)

    # ---- Summary ----
    print("\n" + "=" * 70)
    gpu_name = torch.cuda.get_device_name(device)
    print(f"SUMMARY  (GPU: {gpu_name})")
    print("=" * 70)
    max_label_len = max(len(r[0]) for r in results) if results else 0
    for label, m, k, n, tflops in results:
        print(f"  {label:<{max_label_len}} | {m}, {k}, {n} | {tflops:.2f} TFLOP/s")
    print("=" * 70)


if __name__ == "__main__":
    main()
