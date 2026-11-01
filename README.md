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

The root project and local [`peft`](./peft) package form a uv workspace with a
single [`uv.lock`](./uv.lock). Workspace packages are installed in editable mode;
`peft` contains the SimplexLoRA modifications and uses the same Python and
dependency resolution as the root project.

Use `uv sync --no-dev` for a runtime-only environment. Do not install packages
with conda or pip.

## Checks

```bash
uv run ruff check .
uv run python -m unittest discover -s tests -v
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
peft/                        local workspace member with SimplexLoRA PEFT changes
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
uv run jupytext --to notebook additional_experiments/visualization.py \
  --output /tmp/visualization.ipynb
```

When importing an external notebook, convert it directly to the project format:

```bash
uv run jupytext --to py:percent path/to/external.ipynb
```

## Run experiments

Run experiments through their uv-managed entry points:

```bash
uv run simplexlora-grid --help
uv run simplexlora-grid glue-deberta --dry-run
uv run simplexlora-grid glue-deberta --select simplex
uv run simplexlora-grid nlg --select lora
uv run simplexlora-grid squad --select simplex-lora
uv run simplexlora-grid qwen
uv run simplexlora-grid additional --select memory

uv run simplexlora-glue --help
uv run simplexlora-nlg --help
uv run simplexlora-squad --help
uv run simplexlora-llm --help
uv run simplexlora-dotprod --help
uv run simplexlora-memory --help
uv run python additional_experiments/visualization.py
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
