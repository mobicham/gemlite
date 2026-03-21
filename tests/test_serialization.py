# Usage: python3 test_serialization.py [--autotune]
import sys
_autotune = '--autotune' in sys.argv
if _autotune: sys.argv.remove('--autotune')

import unittest
import torch
from gemlite import reset_config, set_autotune
from gemlite.core import GemLiteLinearTriton, DType, TORCH_TO_DTYPE
from gemlite.triton_kernels.config import KERNEL
from gemlite.helper import A4W4_MXFP_dynamic, A4W4_NVFP_dynamic, patch_model

def is_fp8_supported():
    if not torch.cuda.is_available():
        return False
    capability = torch.cuda.get_device_capability(0)
    return capability >= (8, 9)

device        = 'cuda:0'
compute_dtype = torch.bfloat16
gemlite_dtype = TORCH_TO_DTYPE[compute_dtype]

reset_config()
if _autotune is False: set_autotune(False)
KERNEL.ENABLE_CACHING = False

def _check_serialization(test_case, gemlite_linear, matmul_type='GEMM', batch_size=32, tol=1e-7):
    """Shared serialization round-trip check."""
    in_features = gemlite_linear.in_features

    torch.save(gemlite_linear.state_dict(), '/tmp/_test_serial.pt')

    loaded = GemLiteLinearTriton()
    loaded.load_state_dict(torch.load('/tmp/_test_serial.pt'))

    # Check meta_args match
    ref_meta = gemlite_linear.get_meta_args()
    loaded_meta = loaded.get_meta_args()
    for i in range(len(ref_meta)):
        test_case.assertEqual(ref_meta[i], loaded_meta[i], f"meta_args mismatch at {i}: {ref_meta[i]} != {loaded_meta[i]}")

    # Check tensor_args match
    ref_tensors = gemlite_linear.get_tensor_args()
    loaded_tensors = loaded.get_tensor_args()
    for i in range(len(ref_tensors)):
        if ref_tensors[i].numel() > 0:
            diff = (ref_tensors[i].float() - loaded_tensors[i].float()).abs().mean().item()
            test_case.assertEqual(diff, 0, f"tensor_args mismatch at {i}: mean diff = {diff}")

    # Check inference matches
    x = torch.randn(batch_size, in_features, dtype=compute_dtype, device=device) / 10.
    y_ref = gemlite_linear.forward_manual(x, matmul_type=matmul_type)
    y_loaded = loaded.forward_manual(x, matmul_type=matmul_type)
    diff = (y_ref - y_loaded).abs().mean().item()
    test_case.assertTrue(diff < tol, f"Inference mismatch: mean diff = {diff}, expected < {tol}")


class TestSerializationINT(unittest.TestCase):
    """Serialization tests for INT quantized layers."""

    def test_A16W4(self):
        in_features, out_features = 4096, 2048
        W_nbits, group_size = 4, 128

        W_q = torch.randint(0, 2**W_nbits - 1, (out_features, in_features), device=device).to(torch.uint8)
        gs = W_q.numel() // group_size
        scales = torch.ones((gs, 1), device=device, dtype=compute_dtype) * 0.001
        zeros  = torch.zeros((gs, 1), device=device, dtype=compute_dtype) * ((2**W_nbits - 1)//2)

        gemlite_linear = GemLiteLinearTriton(W_nbits,
                        group_size=group_size,
                        in_features=in_features,
                        out_features=out_features,
                        input_dtype=gemlite_dtype,
                        output_dtype=gemlite_dtype)
        gemlite_linear.pack(W_q, scales, zeros, None)

        _check_serialization(self, gemlite_linear)

    @unittest.skipIf(not is_fp8_supported(), "Skipping test: GPU does not support FP8")
    def test_A8W8(self):
        in_features, out_features = 4096, 2048
        fp8_dtype = torch.float8_e4m3fn

        W = torch.randn((out_features, in_features), dtype=compute_dtype, device=device) / 10.
        _scales = torch.randn((1, out_features), dtype=compute_dtype, device=device) * 1e-4

        gemlite_linear = GemLiteLinearTriton(W_nbits=8,
                        group_size=in_features,
                        in_features=in_features,
                        out_features=out_features,
                        input_dtype=TORCH_TO_DTYPE[fp8_dtype],
                        output_dtype=gemlite_dtype,
                        scaled_activations=True)
        gemlite_linear.pack(W.to(fp8_dtype), scales=_scales, zeros=None, bias=None)

        _check_serialization(self, gemlite_linear)


class TestSerializationMX(unittest.TestCase):
    """Serialization tests for MXFP/NVFP quantized layers."""

    def setUp(self):
        self.in_features, self.out_features = 4224, 2048
        torch.manual_seed(42)
        self.linear_layer = torch.nn.Linear(
            self.in_features, self.out_features, dtype=compute_dtype, device=device, bias=False
        )
        self.linear_layer.weight.data /= 10.
        self.linear_layer.weight.requires_grad = False

    def _quantize(self, processor_fn):
        model = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, self.out_features, dtype=compute_dtype, device=device, bias=False)
        )
        model.requires_grad_(False)
        model[0].weight.data = self.linear_layer.weight.data.clone()
        processor = processor_fn(dtype=compute_dtype)
        patch_model(model, device=device, processor=processor)
        return model[0]

    def test_A4W4_MXFP(self):
        gemlite_linear = self._quantize(A4W4_MXFP_dynamic)
        _check_serialization(self, gemlite_linear, matmul_type='GEMM')
        _check_serialization(self, gemlite_linear, matmul_type='GEMM_SPLITK', batch_size=2)

    def test_A4W4_NVFP(self):
        gemlite_linear = self._quantize(A4W4_NVFP_dynamic)
        _check_serialization(self, gemlite_linear, matmul_type='GEMM')
        _check_serialization(self, gemlite_linear, matmul_type='GEMM_SPLITK', batch_size=2)


if __name__ == '__main__':
    unittest.main()
