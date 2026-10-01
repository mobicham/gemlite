# SPDX-License-Identifier: Apache-2.0
"""Run with: python3 -m pytest tests/vllm_test.py"""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from compressed_tensors.quantization import QuantizationArgs
from gemlite.vllm import backend
from gemlite.vllm.schemes import GemliteCTWNA16Int, GemliteNvFp4LinearMethod, _fp8_channel_scales, _mxfp4_scales
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import modelopt


@pytest.mark.parametrize("legacy_name", [False, True])
def test_modelopt_import_compatibility(legacy_name):
    # Emulate the two exported API names in a fresh process so imports cannot
    # accidentally pass because backend.py was already cached in sys.modules.
    code = f"""
from vllm.model_executor.layers.quantization import modelopt
legacy = getattr(modelopt, 'ModelOptNvFp4LinearMethod', None)
merged = getattr(modelopt, 'ModelOptLinearMethod', None)
method = legacy if legacy is not None else merged
assert method is not None
if {legacy_name!r}:
    modelopt.ModelOptNvFp4LinearMethod = method
else:
    if legacy is not None:
        del modelopt.ModelOptNvFp4LinearMethod
    modelopt.ModelOptLinearMethod = method
from gemlite.vllm.backend import ModelOptNvFp4LinearMethod
assert ModelOptNvFp4LinearMethod is method
"""
    env = os.environ.copy()
    for name in ("VLLM_GEMLITE_ENABLE", "VLLM_GEMLITE_ONTHEFLY_QUANT"):
        env.pop(name, None)
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def routing_layer():
    # Dispatch does not need allocated weights or a distributed process group.
    config = VllmConfig()
    config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    with set_current_vllm_config(config):
        yield LinearBase(128, 128, params_dtype=torch.bfloat16, disable_tp=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_modelopt_nvfp4_routing(monkeypatch, routing_layer, enabled):
    monkeypatch.setattr(backend, "_ENABLED",
                        {"A4W4_NVFP_DYNAMIC"} if enabled else set())
    config = backend.GemliteModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True)
    method = config.get_quant_method(routing_layer, "proj")
    if enabled:
        assert isinstance(method, GemliteNvFp4LinearMethod)
        assert isinstance(method._stock, backend.ModelOptNvFp4LinearMethod)
    else:
        assert isinstance(method, backend.ModelOptNvFp4LinearMethod)


def test_modelopt_excluded_layer_keeps_stock_method(monkeypatch, routing_layer):
    monkeypatch.setattr(backend, "_ENABLED", {"A4W4_NVFP_DYNAMIC"})
    config = backend.GemliteModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True, exclude_modules=["proj"])
    method = config.get_quant_method(routing_layer, "proj")
    assert isinstance(method, UnquantizedLinearMethod)


@pytest.mark.skipif(not hasattr(modelopt, "ModelOptLinearMethod"),
                    reason="The merged ModelOpt API is not available")
def test_modelopt_weight_only_keeps_stock_method(monkeypatch, routing_layer):
    monkeypatch.setattr(backend, "_ENABLED", {"A4W4_NVFP_DYNAMIC"})
    config = backend.GemliteModelOptNvFp4Config(
        quant_method="W4A16_NVFP4", is_checkpoint_nvfp4_serialized=True)
    method = config.get_quant_method(routing_layer, "proj")
    assert isinstance(method, modelopt.ModelOptLinearMethod)
    assert method.spec.activation is None


@pytest.mark.parametrize("num_bits", [4, 8])
@pytest.mark.parametrize("actorder", [None, "static"])
def test_compressed_tensors_packed_int_routing(monkeypatch, num_bits, actorder):
    monkeypatch.setattr(backend, "_ENABLED", {f"A16W{num_bits}_HQQ_INT"})
    weights = QuantizationArgs(num_bits=num_bits, type="int", strategy="group",
                               group_size=128, symmetric=True, actorder=actorder)
    config = backend.GemliteCompressedTensorsConfig.from_config({
        "format": "pack-quantized", "ignore": [],
        "config_groups": {"group_0": {
            "targets": ["Linear"], "weights": weights.model_dump(mode="json"),
            "input_activations": None,
        }},
    })
    scheme = config._get_scheme_from_parts(weights, None,
                                           format="pack-quantized", layer_name="proj")
    assert isinstance(scheme, GemliteCTWNA16Int)
    assert scheme.num_bits == num_bits
    assert scheme.group_size == 128


@pytest.mark.parametrize("toggle", ["A8W8_FP8_DYNAMIC", "A16W8_FP8"])
def test_native_tensor_fp8_routes_through_gemlite(monkeypatch, routing_layer, toggle):
    monkeypatch.setattr(backend, "_ENABLED", {toggle})
    config = backend.GemliteFp8Config(is_checkpoint_fp8_serialized=True,
                                     activation_scheme="dynamic")
    method = config.get_quant_method(routing_layer, "proj")
    assert type(method).__name__ == "GemliteFp8PerTensorLinearMethod"


@pytest.mark.parametrize("toggle", ["A16W8_FP8", "A16W8_INT8"])
def test_weight_only_flags_register_compressed_tensors(monkeypatch, toggle):
    monkeypatch.setattr(backend, "_ENABLED", {toggle})
    assert backend._build_overrides()["compressed-tensors"] is backend.GemliteCompressedTensorsConfig


@pytest.mark.parametrize("toggle,expected", [
    ("A8W8_FP8_DYNAMIC", "GemliteCTW8A8Fp8"),
    ("A16W8_FP8", "GemliteCTW8A16Fp8"),
])
def test_compressed_tensor_fp8_activation_mode(monkeypatch, routing_layer, toggle, expected):
    monkeypatch.setattr(backend, "_ENABLED", {toggle})
    weights = QuantizationArgs(num_bits=8, type="float", strategy="channel", symmetric=True)
    inputs = QuantizationArgs(num_bits=8, type="float", strategy="token", dynamic=True, symmetric=True)
    config = backend.GemliteCompressedTensorsConfig.from_config({
        "format": "float-quantized", "ignore": [],
        "config_groups": {"group_0": {"targets": ["Linear"],
            "weights": weights.model_dump(mode="json"),
            "input_activations": inputs.model_dump(mode="json")}},
    })
    scheme = config._get_scheme_from_parts(weights, inputs,
                                           format="float-quantized", layer_name="proj")
    assert type(scheme).__name__ == expected


@pytest.mark.parametrize("scales,widths,expected", [
    ([2.0], [4], [2.0, 2.0, 2.0, 2.0]),
    ([1.0, 2.0, 3.0, 4.0], [4], [1.0, 2.0, 3.0, 4.0]),
    ([0.25, 0.5, 1.0], [2, 1, 1], [0.25, 0.25, 0.5, 1.0]),
])
def test_fp8_scales_preserve_logical_shard_dequantization(scales, widths, expected):
    layer = torch.nn.Module()
    layer.logical_widths = widths
    layer.weight_scale = torch.nn.Parameter(torch.tensor(scales), requires_grad=False)
    quantized = torch.tensor([[1.0, 2.0]] * sum(widths)).to(torch.float8_e4m3fn)
    expanded = _fp8_channel_scales(layer, quantized)
    actual_weights = quantized.float() * expanded
    reference_weights = quantized.float() * torch.tensor(expected).view(-1, 1)
    torch.testing.assert_close(actual_weights, reference_weights)


def test_fp8_scales_reject_unmatched_shards():
    layer = torch.nn.Module()
    layer.logical_widths = [2, 1, 1]
    layer.weight_scale = torch.nn.Parameter(torch.tensor([0.25, 0.5]), requires_grad=False)
    with pytest.raises(ValueError, match="does not match"):
        _fp8_channel_scales(layer, torch.zeros(4, 2))


@pytest.mark.parametrize("typed", [False, True])
def test_mxfp4_scale_bytes_preserve_physical_scale(typed):
    encoded = torch.tensor([118, 127, 130], dtype=torch.uint8)
    scale = encoded.view(torch.float8_e8m0fnu) if typed else encoded
    layer = torch.nn.Module()
    layer.register_buffer("weight_scale", scale)
    decoded = _mxfp4_scales(layer)
    assert decoded.dtype == torch.float8_e8m0fnu
    torch.testing.assert_close(decoded.float(), torch.tensor([2 ** -9, 1.0, 8.0]))


def test_mxfp4_rejects_nvfp4_scale_format():
    layer = torch.nn.Module()
    layer.register_buffer("weight_scale", torch.tensor([1.0]).to(torch.float8_e4m3fn))
    with pytest.raises(ValueError, match="Unsupported MXFP4"):
        _mxfp4_scales(layer)
