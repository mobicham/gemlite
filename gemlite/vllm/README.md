# gemlite.vllm — gemlite integration for vLLM

Route vLLM's quantized forward path through gemlite's Triton kernels, or
quantize fp16/bf16 checkpoints on the fly at load time.

Supported pre-quantized formats: FP8 (dynamic block and per-tensor/channel,
weight-only), NVFP4 W4A4, MXFP4 weight-only/dynamic, INT8 weight-only and
dynamic, GPTQ / AWQ / GPTQMarlin / AWQMarlin int4 and int8, compressed-tensors
NVFP4/MXFP4/WNA16, GGUF (Q4_0 / Q4_1 / Q4_K / Q8_0 / Q2_K).

Entry points:
- `enable_gemlite(names=None)` — route pre-quantized checkpoints through gemlite.
- `set_onthefly_quant(...)` — quantize fp16/bf16 checkpoints at load time.
- `patch_vllm()` — env-driven application of both (idempotent).
- `register()` — vLLM plugin entry point (calls `patch_vllm()`).

## How the patch works

`enable_gemlite()` swaps vLLM's `get_quantization_config` registry so requested
quant types load through gemlite `LinearMethod`s. Weight materialization is
delegated to stock vLLM (so HF loading, sharding, fused qkv/gate_up all keep
working); only `process_weights_after_loading` and `apply` are replaced.

**The swap must run before vLLM resolves the quant-config class for the
model** — i.e. before `LLM(...)` or `vllm serve` finishes importing the
engine. All examples below follow that ordering.

Any checkpoint / layer that gemlite doesn't handle (K-quants other than
Q4_K / Q2_K, I-quants, MoE experts, embeddings, etc.) falls through to stock
vLLM with a warning. Nothing hard-fails.

## Requirements

- `gemlite` (this package)
- `vllm`
- `hqq` — only if you use on-the-fly `int4_weightonly`

On Blackwell, make sure you use CUDA 13 PTXAS
```
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.0/bin/ptxas
```

## 1. Interactive Python / offline `LLM`

Import and enable **before** constructing `LLM`:

```python
from gemlite.vllm import enable_gemlite
enable_gemlite()                          # all schemes in SUPPORTED

from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen3-4B-Instruct-2507-FP8", dtype="bfloat16")
out = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=16))
print(out[0].outputs[0].text)
```

To restrict to a subset:

```python
enable_gemlite(["A8W8_FP8_DYNAMIC", "A16W4_HQQ_INT"])
```

## 2. `vllm serve` / OpenAI-compatible server

`gemlite` registers a `vllm.general_plugins` entry point (see `setup.py`),
so plain `vllm serve` auto-discovers and installs the patch at engine
startup. Set `VLLM_GEMLITE_ENABLE=1` to opt in:

```bash
export VLLM_GEMLITE_ENABLE=1
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 --dtype bfloat16 --port 8000
```

Same env var works with `python -m vllm.entrypoints.openai.api_server ...`.

Restrict to a subset:

```bash
export VLLM_GEMLITE_ENABLE_LIST=A8W8_FP8_DYNAMIC,A16W4_HQQ_INT
```

### Bootstrap (fallback)

If the plugin entry point isn't discovered (e.g. an editable install that
skipped `pip install -e .` after adding the entry point), pre-import
`gemlite.vllm` in a wrapper instead:

```bash
export VLLM_GEMLITE_ENABLE=1
python3 -c "
import sys, gemlite.vllm            # triggers patch_vllm() via env var
sys.argv = ['vllm', 'serve', 'Qwen/Qwen3-4B-Instruct-2507-FP8',
            '--dtype', 'bfloat16', '--port', '8000']
from vllm.entrypoints.cli.main import main
main()
"
```

## 3. Pre-quantized checkpoints

`enable_gemlite(names=None)` enables every scheme below; pass a list to
restrict.

| Scheme name          | Matches checkpoints                                  |
| -------------------- | ---------------------------------------------------- |
| `A8W8_FP8_DYNAMIC`   | FP8 dynamic (DeepSeek 128×128 block, per-tensor, per-channel) |
| `A16W8_FP8`          | FP8 weight-only per-channel, dynamic activations     |
| `A4W4_NVFP_DYNAMIC`  | NVFP4 W4A4 (ModelOpt and compressed-tensors)         |
| `A4W4_MXFP_DYNAMIC`  | MXFP4 dynamic                                        |
| `A16W4_MXFP`         | MXFP4 weight-only (compressed-tensors)               |
| `A16W8_INT8`         | INT8 weight-only                                     |
| `A8W8_INT8_DYNAMIC`  | INT8 dynamic                                         |
| `A16W4_HQQ_INT`      | GPTQ / AWQ / GPTQMarlin / AWQMarlin int4, HQQ int4, GGUF Q4_0 / Q4_1 / Q4_K, CT pack_quantized int4 |
| `A16W8_HQQ_INT`      | GPTQ / AWQ int8, GGUF Q8_0, CT pack_quantized int8   |
| `A16W2_HQQ_INT`      | GGUF Q2_K                                            |

GGUF routing is available only on vLLM releases that provide the `gguf`
quantization backend. vLLM 0.23 no longer provides that backend, so GemLite
leaves `gguf` unregistered on that version instead of installing a broken
override.

Aliases: `A16W4_INT` → `A16W4_HQQ_INT`, `A16W8_INT` → `A16W8_HQQ_INT`.

## 4. On-the-fly quantization

Quantize an **fp16 / bf16** checkpoint at load time. This path is a no-op
on already-quantized checkpoints (FP8 / AWQ / GPTQ / GGUF / …) — those
arrive with a `quant_config` attached, and the on-the-fly hook only
replaces layers whose `quant_config is None`. For pre-quantized models use
`VLLM_GEMLITE_ENABLE=1` (section 2/3) instead.

Via env var (uses a named preset) — **the recommended path** for `vllm serve`
and offline `LLM`, because it propagates to v1 worker subprocesses:

```bash
export VLLM_GEMLITE_ONTHEFLY_QUANT=A16W8_INT8
export VLLM_GEMLITE_SKIP_MODULES=lm_head,visual,vision
vllm serve Qwen/Qwen3-4B --dtype bfloat16 --port 8000
```

`VLLM_GEMLITE_ONTHEFLY_QUANT` alone is enough — gemlite's plugin
`register()` runs `patch_vllm()` in every worker, which calls
`set_onthefly_quant(...)` with the matching preset.

Programmatic — works for offline `LLM` only when v1 spawn is disabled
(`VLLM_USE_V1=0`) or for code paths that don't fork workers; under v1 the
parent-process monkey-patch does **not** propagate to workers, so use the
env-var path for `vllm serve` / multi-process scenarios:

```python
from gemlite.vllm import set_onthefly_quant
set_onthefly_quant(
    weight_bits=8, group_size=None, quant_mode="int8_weightonly",
    skip_modules=["lm_head", "vision", "visual"],
)

from vllm import LLM
llm = LLM(model="Qwen/Qwen3-4B", dtype="bfloat16")
```

If you need a custom `weight_bits` / `group_size` combo not in the preset
table, add an entry to `_ONTHEFLY_PRESETS` in `gemlite/vllm/backend.py` and
select it via `VLLM_GEMLITE_ONTHEFLY_QUANT`.

### Presets

| Preset                     | weight_bits | group_size | quant_mode          | block_quant |
| -------------------------- | ----------- | ---------- | ------------------- | ----------- |
| `A16W8_INT8`               | 8           | —          | `int8_weightonly`   | —           |
| `A16W8_FP8`                | 8           | —          | `fp8_weightonly`    | —           |
| `A16W4_INT4_HQQ`           | 4           | 64         | `int4_weightonly`   | —           |
| `A8W8_INT8_DYNAMIC`        | 8           | —          | `int8_dynamic`      | false       |
| `A8W8_FP8_DYNAMIC`         | 8           | —          | `fp8_dynamic`       | false       |
| `A8W8_FP8_DYNAMIC_BLOCK`   | 8           | —          | `fp8_dynamic`       | true        |
| `MXFP8_DYNAMIC`            | 8           | 32         | `mxfp8_dynamic`     | —           |
| `MXFP4_WEIGHTONLY`         | 4           | —          | `mxfp4_weightonly`  | —           |
| `MXFP4_DYNAMIC`            | 4           | —          | `mxfp4_dynamic`     | —           |
| `A8W4_MXFP_DYNAMIC`        | 4           | —          | `mxfp8_dynamic`     | —           |
| `NVFP4_DYNAMIC`            | 4           | —          | `nvfp4_dynamic`     | —           |

`int4_weightonly` requires `pip install hqq`.

## Environment variables

| Name                          | Default   | Purpose                                              |
| ----------------------------- | --------- | ---------------------------------------------------- |
| `VLLM_GEMLITE_ENABLE`         | `0`       | Set to `"1"` to route pre-quantized checkpoints through gemlite. Required for `vllm serve` (both plugin and bootstrap paths). |
| `VLLM_GEMLITE_ENABLE_LIST`    | (unset)   | Comma-separated subset of `SUPPORTED` scheme names.  |
| `VLLM_GEMLITE_ONTHEFLY_QUANT` | (unset)   | Preset name — enables on-the-fly quantization (fp16/bf16 checkpoints only). |
| `VLLM_GEMLITE_SKIP_MODULES`   | `lm_head,visual,vision` | Comma-separated module names to leave unquantized (on-the-fly only). |

## Notes

- **Autotune cache** — first call on a new shape runs Triton autotune (can
  take minutes). Decisions are persisted to `/tmp/gemlite_cache.json` and
  reused on subsequent runs.
- **CUDA graphs** — keep them on (vLLM default). Gemlite kernels are
  captured correctly under `torch.compile`'s PIECEWISE mode.
- **Fallback on unsupported layers** — a warning is logged and that layer
  keeps its stock vLLM forward path. This applies per-layer, not per-model:
  a model with unsupported GGUF tensors still uses gemlite on the supported
  ones.

## Tested

Verified end-to-end on RTX PRO 6000 Blackwell (sm_120, CUDA 13, vLLM 0.19.2):
across all three activation paths (offline `LLM`, `vllm serve` bootstrap,
plain `vllm serve` via plugin):

| Model                                        | Format                     | Scheme                          |
| -------------------------------------------- | -------------------------- | ------------------------------- |
| `Firworks/Qwen3-4B-Instruct-2507-nvfp4`      | CT NVFP4 W4A4              | `A4W4_NVFP_DYNAMIC`             |
| `Qwen/Qwen3-4B-Instruct-2507-FP8`            | DeepSeek block FP8 128×128 | `A8W8_FP8_DYNAMIC`              |
| `cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit`   | CT pack-quantized int4     | `A16W4_HQQ_INT`                 |
| `JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4`  | GPTQ int4 (→ gptq_marlin)  | `A16W4_HQQ_INT`                 |
| `unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_1`   | GGUF Q4_1                  | `A16W4_HQQ_INT`                 |

Compatibility-checked on vLLM
`0.23.1rc1.dev1279+gdcfebf93f` with the same Blackwell/CUDA 13 setup. This
includes the plugin/env entry point, block FP8, compressed-tensors NVFP4 and
MXFP4 weight loading, AutoAWQ/AutoGPTQ routing, and on-the-fly `LinearBase`
construction. The mixed block-FP8/NVFP4 model
`dropbox-dash/Qwen3.5-4B_glm-52-fp8_deepspeed_v2_take3_extradata_1_vlm-NVFP4-MIX-FP8KV`
also loads and serves through the GemLite routes on this version.

### vLLM 0.26 nightly

Validated on vLLM `0.26.1rc1.dev255+g5e35a6f4f`, PyTorch 2.13, Triton
3.7.1, and CUDA 13 on RTX PRO 6000 Blackwell. Unless noted otherwise, these
checks used vLLM's defaults: `enforce_eager=False`, `VLLM_COMPILE`, and all
51 PIECEWISE plus all 51 FULL CUDA graph capture sizes.

The following on-the-fly presets load, compile, capture, and generate
successfully:

| Preset                           | Result |
| -------------------------------- | ------ |
| `A16W8_INT8`                     | Pass   |
| `A16W8_FP8`                      | Pass   |
| `A16W4_INT4_HQQ`                 | Pass   |
| `A8W8_INT8_DYNAMIC`              | Pass   |
| `A8W8_FP8_DYNAMIC`               | Pass   |
| `A8W8_FP8_DYNAMIC_BLOCK`         | Pass   |
| `MXFP8_DYNAMIC`                  | Pass   |
| `MXFP4_WEIGHTONLY`               | Pass   |
| `A8W4_MXFP_DYNAMIC`              | Pass   |
| `NVFP4_DYNAMIC`                  | Pass   |
| `MXFP4_DYNAMIC`                  | Execution passes; see quality note below |

Direct kernel comparisons for the INT8, MXFP8, MXFP4, and NVFP4 dynamic
paths produced finite outputs and stayed within 0.43% relative L2 error of
an explicitly dequantized reference across the sampled shapes. Fully dynamic
MXFP4 also matched its quantized reference, but quantizing both weights and
activations to FP4 caused roughly 16-17% error relative to BF16 and visibly
degraded Qwen3-0.6B generation. Treat `MXFP4_DYNAMIC` as an aggressive
quality/performance tradeoff, not as a generally quality-safe default.

Triton 3.7 requires native BF16/FP16 operands of mixed MXFP4
`tl.dot_scaled` to use a null lhs scale; GemLite follows that contract for
weight-only MXFP4 on sm_120. Triton does not provide the corresponding
native BF16/FP16 x NVFP4 weight-only `tl.dot_scaled` path. W4A4
`NVFP4_DYNAMIC` is supported because both operands are FP4.

The following pre-quantized checkpoints were also validated through GemLite
with offline `LLM` and the same default compile/CUDA-graph settings:

| Model                                        | Format                     | Result |
| -------------------------------------------- | -------------------------- | ------ |
| `Firworks/Qwen3-4B-Instruct-2507-nvfp4`      | CT NVFP4 W4A4              | Pass   |
| `Qwen/Qwen3-4B-Instruct-2507-FP8`            | DeepSeek block FP8 128x128 | Pass   |
| `cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit`   | CT pack-quantized int4     | Pass   |
| `JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4`  | GPTQ int4                  | Pass   |

Newer vLLM's DeepGEMM warmup discovers block-FP8 layers through the stock
`fp8_linear` selector. GemLite clears that selector after replacing the stock
weights so the warmup does not inspect tensors that have already been packed
and removed. The GGUF checkpoint cannot be run on this nightly because this
vLLM release does not expose a GGUF quantization backend. The mixed Dropbox
FP8/NVFP4 checkpoint was not rechecked because it requires access to its
private Hugging Face repository.

### vLLM 0.30 nightly

Validated on 2026-10-01 with vLLM `0.30.1rc1.dev396+gac68c3087`, PyTorch
`2.13.0+cu130`, Triton `3.7.1`, and CUDA 13 on RTX PRO 6000 Blackwell
(`sm_120`). The matrix ran actual `vllm serve` processes and HTTP requests;
it includes pre-quantized checkpoints and all 11 on-the-fly presets.

All 64 GemLite configuration checks pass. Each checks worker-side
GemLite methods and activation precision, cold/warm chat, concurrent requests
including longer prefill, streaming, batched completions, finite token
logprobs, and schema-valid JSON. Arithmetic and capital-city smoke checks pass.
Both all-enabled and individually selected formats are covered, as are the
`A16W4_INT` and `A16W8_INT` legacy aliases.

| On-the-fly preset | Model | Serve/API |
| --- | --- | --- |
| `A16W4_INT4_HQQ` | `Qwen/Qwen3-4B-Instruct-2507` | Pass |
| `A16W8_FP8` | `Qwen/Qwen3-4B-Instruct-2507` | Pass |
| `A16W8_INT8` | `Qwen/Qwen3-4B-Instruct-2507` | Pass |
| `A8W4_MXFP_DYNAMIC` | `Qwen/Qwen3-4B-Instruct-2507` | Pass |
| `A8W8_FP8_DYNAMIC` | `Qwen/Qwen3-4B-Instruct-2507` | Pass |
| `A8W8_FP8_DYNAMIC_BLOCK` | `Qwen/Qwen3-4B-Instruct-2507` | Pass |
| `A8W8_INT8_DYNAMIC` | `Qwen/Qwen3-4B-Instruct-2507` | Pass |
| `MXFP4_DYNAMIC` | `Qwen/Qwen3-4B-Instruct-2507` | Pass |
| `MXFP4_WEIGHTONLY` | `Qwen/Qwen3-4B-Instruct-2507` | Pass |
| `MXFP8_DYNAMIC` | `Qwen/Qwen3-4B-Instruct-2507` | Pass |
| `NVFP4_DYNAMIC` | `Qwen/Qwen3-4B-Instruct-2507` | Pass |

| Pre-quantized path | Checkpoints / layouts | Serve/API |
| --- | --- | --- |
| ModelOpt A4W4 NVFP4 | `baseten/Qwen3-4B-NVFP4-PTQ` | Pass |
| Compressed-tensors A4W4 NVFP4 | `Firworks/Qwen3-4B-Instruct-2507-nvfp4` | Pass |
| Native block FP8 | `Qwen/Qwen3-4B-Instruct-2507-FP8` | Pass |
| Compressed-tensors channel FP8 | `RedHatAI/gemma-3-4b-it-FP8-dynamic` | Pass |
| Compressed-tensors packed int4 | `cyankiwi/Qwen3-4B-Instruct-2507-AWQ-4bit` | Pass |
| AutoAWQ / Marlin selection | `Qwen/Qwen3-4B-AWQ` | Pass |
| GPTQ int4 | `JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int4` | Pass |
| GPTQ int8 | `JunHowie/Qwen3-4B-Instruct-2507-GPTQ-Int8` | Pass |
| Compressed-tensors dynamic INT8 | `zhiqing/Qwen3-1.7B-INT8` | Pass |
| Native per-tensor FP8 | Saved FP8 weights, A8W8 and A16W8 | Pass |
| Compressed-tensors FP8 | Tensor, channel, block 128x128; dynamic and weight-only | Pass |
| Compressed-tensors packed INT4/INT8 | Group/channel, symmetric/asymmetric | Pass |
| Compressed-tensors MXFP4 | A4W4 dynamic and A16W4 weight-only | Pass |

NVFP4 uses the original A4W4 checkpoints above, with packed FP4 weights,
E4M3 block scales, group size 16, and 4-bit activation quantization. AutoRound
checkpoints are not used as NVFP4 evidence.

Additional layout fixtures contain real `Qwen/Qwen3-4B-Instruct-2507` weights
quantized and saved with the installed compressed-tensors compressors before
starting the server. Block-FP8 CT coverage also uses unchanged vendor FP8
weights/scales with the scale key renamed to the CT convention. These are
saved pre-quantized checkpoints, not on-the-fly-only tests. Provenance and
the exact fixture/configuration for every run are recorded in the artifacts.

The tests use BF16, TP=1, `max_model_len=2048`, `max_num_seqs=16`,
`max_num_batched_tokens=2048`, and `gpu_memory_utilization=0.5`, with
`enforce_eager=False`, `VLLM_COMPILE`, and `FULL_AND_PIECEWISE` CUDA graphs.
Graph capture sizes are `[1, 2, 4, 8, 16, 24, 32]`. Gemma requests use text.

The matrix exposed and fixed ModelOpt's renamed linear class, WNA16's removed
`actorder` argument, missing selective FP8/INT8 weight-only routes, incorrect
per-tensor FP8 fused-shard scale expansion, and MXFP4 E8M0 byte interpretation.
`A16W8_FP8` now actually keeps BF16/FP16 activations. Native per-tensor FP8
also responds to the `A8W8_FP8_DYNAMIC` toggle.

After fixing the MXFP4 scale view, weight-only output is within 0.32% relative
L2 of an independently dequantized layer reference; A4W4's additional
activation quantization gives roughly 11.7% relative L2 versus that BF16-input
reference. Both modes produce correct generation smoke checks and valid JSON.

Eight stock-vLLM control configurations also pass. The stock MXFP4 control
fails with FlashInfer's `mm_fp4 does not support backend 'cute-dsl' with
capability 120`; GemLite's MXFP4 paths pass on the same device. GGUF Q2/Q4/Q8
cannot run because this nightly has removed the GGUF quantization backend;
the capability check reports `Invalid quantization method: gguf`.

Run the focused import, routing, and numerical scale regressions with:

```bash
python3 -m pytest tests/vllm_test.py
```

All 23 cases pass. The legacy ModelOpt symbol is tested in an isolated
process; an older vLLM runtime was not reinstalled. Ada/L4, FP16 serving, and
multi-GPU configurations were not tested. These are compatibility/generation
smoke checks, not model-accuracy or latency benchmarks.

When updating GemLite code, start with a fresh vLLM compilation cache. The
matrix found that stock graphs compiled before a route fix can be reused
after the new method replaces the stock tensors. Repaired routes were
rechecked with an isolated `VLLM_CACHE_ROOT`.

The 2026-10-01 artifacts are under `/tmp/gemlite-vllm-full-matrix/`:
`manifest.json`, `coverage.csv`, `validation-summary.json`, and per-case
JSON/log files retain every attempt, including the failures and repairs.

GGUF checkpoints require `--hf-config-path <hf-repo>` on `vllm serve`, and
must use `--dtype float16` (vLLM rejects `bfloat16` for GGUF).

## Troubleshooting

- **`KeyError: 'gemlite_linear'` after toggling `VLLM_GEMLITE_ENABLE`** —
  recent vLLM releases expose their environment-variable registry; GemLite
  registers its plugin settings there so different modes receive different
  torch.compile cache keys. On older releases without that registry, a graph
  compiled with GemLite enabled can be reused with GemLite disabled (or vice
  versa). Wipe the cache when switching backends on those releases:

  ```bash
  rm -rf ~/.cache/vllm/torch_compile_cache
  ```

- **GGUF + bfloat16 rejected** — vLLM only accepts `float16` / `float32`
  for GGUF quant. Pass `--dtype float16`. Also pass `--hf-config-path
  <unquantized-repo>` if the GGUF repo doesn't ship a `config.json`
  (e.g. `unsloth/*-GGUF` repos).

- **`collective_rpc` with custom functions** — set
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1` to let vLLM pickle arbitrary
  callables to workers. Only relevant if you pass your own probe/closure,
  not for normal inference.

- **Plugin not firing under plain `vllm serve`** — verify the entry point
  is registered:

  ```bash
  python3 -c "from importlib.metadata import entry_points; print([e for e in entry_points().select(group='vllm.general_plugins')])"
  ```

  should include `gemlite`. If missing, reinstall: `pip install -e /path/to/gemlite`.
