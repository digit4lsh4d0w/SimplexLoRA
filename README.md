## Requirements

- [uv](https://docs.astral.sh/uv/) 0.13.6 or newer.
- Python 3.15 or newer. The local development version is stored in
  `.python-version` as `3.15`, so uv selects the latest 3.15 patch release
  automatically.
- Linux and a driver compatible with the selected CUDA or ROCm PyTorch build.

`requires-python` describes the compatibility range of the project, while
`.python-version` selects the interpreter used for local development.

## Dependencies

All dependency management and command execution go through uv. Select one GPU
profile when creating the environment:

```bash
uv sync --extra rocm
# or
uv sync --extra cuda
```

The `cuda` and `rocm` profiles are mutually exclusive. A plain `uv sync`
installs only the backend-independent dependencies. The ROCm profile is tuned
for the RX 9070 XT (`gfx1201`); use another `device-gfx*` extra and matching
override when targeting a different AMD GPU.

`uv sync --extra cuda` installs the CUDA PyTorch wheel from
`download.pytorch.org`, while its NVIDIA runtime dependencies come from the
standard PyPI package index. `uv sync --extra rocm` installs PyTorch and the
required ROCm SDK packages from the official ROCm 7.14 index.

The project uses the public PEFT release managed through [`uv.lock`](./uv.lock).
Simplex LoRA is implemented as an out-of-tree PEFT layer in
`src/simplexlora/peft_plugin.py`; it registers a custom linear module on a
standard `LoraConfig` and does not replace the installed PEFT package.

Use `uv sync --extra cuda --no-dev` for a runtime-only NVIDIA environment
(`--extra rocm` on AMD). Do not install packages with conda or pip.

The selected GPU profile must also be passed to every `uv run` command. The
examples below use `--extra cuda` because they target an NVIDIA host; replace
it with `--extra rocm` on AMD. Omitting the extra does not select an implicit
GPU backend and can leave the command without the intended PyTorch runtime.

## Simplex LoRA plugin

PEFT's custom-module mapping is runtime-only and is not persisted in
`adapter_config.json`. Use the helpers from `simplexlora.peft_plugin` whenever
an adapter is created or loaded:

```python
from simplexlora.peft_plugin import (
    get_simplex_peft_model,
    load_simplex_peft_model,
    make_simplex_lora_config,
)

config = make_simplex_lora_config(r=64, lora_alpha=128, target_modules=["q_proj", "v_proj"])
model = get_simplex_peft_model(base_model, config)

restored = load_simplex_peft_model(base_model, "path/to/adapter")
```

The custom-module hook is marked experimental by PEFT. Keeping an official
PEFT installation means future PEFT releases can still provide their other
bug fixes and tuners, but the hook must be re-registered before loading a
Simplex LoRA adapter.

## ToDo: PEFT plugin parity and validation

The current plugin reproduced the vendored implementation in one 50-step
Qwen2.5-Coder-7B NF4 run: it selected the same layers for all five
Simplex projections and produced the same final rank distribution. The
following checks are still required before treating the implementations as
fully interchangeable.

- [ ] **Test quantized merge and unmerge separately for 4-bit and 8-bit BnB.**
  Load a small quantized model, run one fixed batch, and save its logits before
  merging. Set deterministic values for `lora_A`, `lora_B`, and `lora_weight`,
  call `merge_adapter()`, compare logits, then call `unmerge_adapter()` and
  compare both logits and base weights. Run this on ROCm and CUDA and record
  finite-value checks, maximum absolute error, and relative error. Do not use
  the same tolerance for NF4/INT8 and bf16: quantized merge may introduce
  expected rounding drift. The current plugin wraps quantized layers through
  the generic `nn.Linear` mapping, while the vendored implementation had
  specialized BnB LoRA classes.

- [ ] **Test checkpoint interoperability in both directions.**
  Save an adapter with the vendored PEFT and load it with the public-PEFT
  plugin; then perform the reverse operation. In both cases use
  `load_simplex_lora_config()` or `load_simplex_peft_model()` before loading
  weights, verify that every expected `lora_weight` key exists, and compare
  fixed-batch logits with the source model. Record how the old
  `use_weight_lora` config field is handled by current PEFT and confirm that no
  adapter weights are silently dropped. Repeat with 4-bit base weights.

- [ ] **Test save, reload, and resume after a rank-update cycle.**
  Save a checkpoint immediately after at least one `fat_step`, restore the
  model and optimizer state, and continue for another complete cycle. Verify
  that `lora_weight` objects and optimizer references still point to the same
  parameters, that configured ranks are restored, and that the resumed loss
  trajectory matches an uninterrupted run within expected floating-point
  tolerance.

- [ ] **Test multiple adapters and mixed-adapter batches.**
  Attach two Simplex adapters to the same model, verify `set_adapter()`,
  `enable_adapters()`, `disable_adapters()`, `delete_adapter()`, and
  `adapter_names` forwarding. For mixed batches, compare the result against
  separate per-adapter forward passes and confirm that each `lora_weight` is
  applied only to its own sub-batch.

- [ ] **Define and test supported layer types and LoRA options.**
  Document that the current custom module targets `nn.Linear` and subclasses,
  including BnB linear layers. Add explicit tests or validation errors for
  `use_dora`, `lora_bias`, `use_qalora`, multi-adapter variants, and other PEFT
  LoRA variants. In particular, verify whether `lora_bias=True` must multiply
  the bias contribution by `lora_weight`; if that behavior is not implemented,
  reject `lora_bias=True` instead of silently producing a different update.

- [ ] **Monitor the experimental PEFT custom-module API.**
  `_register_custom_module()` is private and may change even in a patch or minor
  PEFT release. Run the CPU and GPU plugin tests whenever PEFT is updated.
  Compare release notes and the implementation of `LoraConfig`,
  `LoraModel._create_new_module()`, BnB dispatch, and adapter state-dict key
  handling. Add an upper dependency bound only if compatibility testing shows
  that a release actually breaks the plugin, rather than pre-emptively.

- [ ] **Evaluate importance estimation with a representative calibration set.**
  Use multiple data seeds and the paper's calibration budget rather than a
  single 50-example run: 50 optimizer steps with batch 4 gives 200 examples;
  batch 16 gives 800. Report the distribution after every projection, including
  entropy, effective number of layers, top-k share, zero-rank layers, and rank
  totals. Confirm that the selected layers remain stable across seeds and are
  not an artifact of one small sample. Keep `weight_lr` as an explicit
  sensitivity parameter because the current objective can become
  winner-take-all.

- [ ] **Export the final layer plan before the final adapter reset.**
  `_final_lora_rank_update()` replaces the selected A/B matrices and fixes
  `lora_weight` at one. Add a stable exporter that records the final projection
  weights and rank plan before that reset, preferably as `layer_plan.json` with
  module names, ranks, weights, and total rank budget. Use this file as the
  handoff format for Unsloth instead of parsing training logs.

- [ ] **Validate the Unsloth handoff end to end.**
  Load Qwen2.5-Coder in Unsloth, apply the exported ranks through
  `rank_pattern`, train with the same data and parameter budget as a standard
  LoRA control, and evaluate on a held-out split. Compare task quality,
  trainable parameters, peak VRAM, tokens per second, and final adapter size.
  The 50-step Simplex phase is only rank calibration; meaningful model quality
  must be measured after the subsequent optimized LoRA training.

- [ ] **Add repeatable GPU regression tests.**
  Keep ordinary unit tests CPU-only, but add opt-in pytest markers for
  ROCm/CUDA tests covering BnB forward/backward, rank changes, merge/unmerge,
  checkpoint reload, and a tiny end-to-end calibration run. Record the GPU
  architecture, PyTorch, PEFT, Transformers, and bitsandbytes versions in every
  test report so numerical differences can be separated from backend changes.

## Checks

```bash
uv lock --check
uv run --frozen --extra cuda ruff check .
uv run --frozen --extra cuda ruff format --check src tests
uv run --frozen --extra cuda python -m unittest discover -s tests -v
uv check --no-project
uv build --all-packages
```

`uv check` type-checks the complete package under `src/` and the test suite
configured by `[tool.ty.src]`. The `--no-project` flag avoids a uv 0.13 metadata
failure caused by the intentionally conflicting CUDA and ROCm extras; it does
not disable any ty diagnostics.

## Project layout

The installable package uses the uv `src` layout:

```text
src/simplexlora/             core configuration, optimizers and utilities
src/simplexlora/experiments/ runnable experiment modules and CLI entry points
src/simplexlora/experiments/configs/ TOML experiment matrices
src/simplexlora/tools/       maintenance utilities
src/simplexlora/peft_plugin.py Simplex LoRA custom PEFT layer and helpers
```

Python entry points are defined in `pyproject.toml`, so experiment code does
not rely on `sys.path` changes or sibling-file imports.

## Notebook sources

Notebook sources are stored as Jupytext `py:percent` scripts. The `.py` files
are the canonical, version-controlled form and generated `.ipynb` files should
not be committed. Keep markdown sections in `# %% [markdown]` cells so that the
method notes and experiment context remain part of the source.

To inspect or run a source as a temporary notebook, convert it outside the
repository:

```bash
uv run --extra cuda jupytext --to notebook additional_experiments/visualization.py \
  --output /tmp/visualization.ipynb
```

When importing an external notebook, convert it directly to the project format:

```bash
uv run --extra cuda jupytext --to py:percent path/to/external.ipynb
```

## Run experiments

Run experiments through their uv-managed entry points:

```bash
uv run --extra cuda simplexlora-grid --help
uv run --extra cuda simplexlora-grid glue-deberta --dry-run
uv run --extra cuda simplexlora-grid glue-deberta --select simplex
uv run --extra cuda simplexlora-grid nlg --select lora
uv run --extra cuda simplexlora-grid squad --select simplex-lora
uv run --extra cuda simplexlora-grid qwen
uv run --extra cuda simplexlora-grid additional --select memory

uv run --extra cuda simplexlora-glue --help
uv run --extra cuda simplexlora-nlg --help
uv run --extra cuda simplexlora-squad --help
uv run --extra cuda simplexlora-llm --help
uv run --extra cuda simplexlora-dotprod --help
uv run --extra cuda simplexlora-memory --help
uv run --extra cuda python additional_experiments/visualization.py
```

Built-in grid configurations cover GLUE DeBERTa, NLG, SQuAD, Qwen and the
memory/dot-product experiments. Use `--resume` to skip successful runs recorded
in the JSON manifest and `--keep-going` to continue after failures. See the
[grid configuration guide](./src/simplexlora/experiments/configs/README.md) for
the TOML schema.

## Decompilation training

The first SimplexLoRA decompilation run uses a deterministic 20,000-example
subset of `LLM4Binary/decompile-bench`:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
uv run --extra rocm simplexlora-decompile \
  --offline \
  --max-train-samples 20000 \
  --max-seq-length 1024 \
  --num-train-epochs 3
```

This writes the PEFT adapter, tokenizer files and memory metrics to
`outputs/qwen2.5-coder-decompile-simplex`.
