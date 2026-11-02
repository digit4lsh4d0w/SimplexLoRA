# Vendored Simplex LoRA research baseline

This branch is based on `805b81e^`, the last revision before Simplex LoRA was
moved from the vendored PEFT fork to the out-of-tree plugin. It intentionally
retains the old optimizer and PEFT layer so the original behavior can be tested
without the current plugin.

## Setup

```bash
uv sync --extra cuda
```

## Prepare the same O0 data

Generate the optimization cluster mapping with the current `main` branch first,
then export a deterministic subset from this branch:

```bash
uv run --extra cuda python tools/make_o0_subset.py \
  --output-dir /tmp/decompile-bench-o0-2000 \
  --max-samples 2000
```

The script only selects rows using the existing cluster parquet. It does not
modify training data or execute generated code.

## Known-answer test

```bash
uv run --extra cuda python -m unittest tests.test_vendored_known_importance -v
```

The test checks that the vendored implementation selects the two causally
active layers in two independent configurations.

## Qwen CUDA comparison

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 \
uv run --extra cuda simplexlora-decompile \
  --offline \
  --dataset-path /tmp/decompile-bench-o0-2000 \
  --model-id Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --output-dir /tmp/simplexlora-old-qwen05-o0-50 \
  --max-train-samples 2000 \
  --max-validation-samples 0 \
  --max-seq-length 1024 \
  --max-steps 50 \
  --num-train-epochs 1 \
  --learning-rate 1e-4 \
  --weight-lr 5e-3 \
  --weight-decay 0.01 \
  --lora-r 128 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --fat-step 10 \
  --max-fat-steps 5 \
  --eval-steps 0 \
  --save-steps 999999 \
  --logging-steps 1 \
  --seed 42
```

The old optimizer prints `self.lora_ranks` and `New chosen layers` after each
projection. Preserve the complete output. Do not use the saved adapter for the
Unsloth comparison yet; this run is only a behavioral baseline.

Known old-code limitations:

- `lora_dropout=0` crashes rank updates because `nn.Identity` has no `.p`;
- repeated parameter replacement causes ROCm memory faults;
- the rank update creates a new `lora_weight`, while the optimizer's weight
  parameter group is not updated to the new object.
