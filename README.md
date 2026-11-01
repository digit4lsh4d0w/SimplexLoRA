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

The local [`peft`](./peft) package is installed in editable mode because it
contains the SimplexLoRA modifications.

Use `uv sync --no-dev` for a runtime-only environment. Do not install packages
with conda or pip.

## Checks

```bash
uv run ruff check .
```

## Notebooks

The notebooks are converted to Python with the locked `nbconvert` version:

```bash
uv run jupyter-nbconvert --to script path/to/notebook.ipynb --output notebook
```

The visualization notebook has already been exported to
`additional_experiments/visualization.py`.

## Run experiments

Run scripts and Python modules through uv:

```bash
uv run bash run_glue_experiments.sh
uv run python additional_experiments/visualization.py
```

To conduct your own experiments, check [glue_experiment/scripts/](./glue_experiment/scripts/)

## Decompilation training

The first SimplexLoRA decompilation run uses a deterministic 20,000-example
subset of `LLM4Binary/decompile-bench`:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
uv run --extra rocm python run_experiment/train_decompile.py \
  --offline \
  --max-train-samples 20000 \
  --max-seq-length 1024 \
  --num-train-epochs 3
```

This writes the PEFT adapter, tokenizer files and memory metrics to
`outputs/qwen2.5-coder-decompile-simplex`.
