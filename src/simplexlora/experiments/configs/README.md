# Experiment grid configurations

Each TOML file expands `[[experiment]]` tables into a sequence of commands. Run
the built-in configurations by name:

```bash
uv run simplexlora-grid glue-deberta --dry-run
uv run simplexlora-grid glue-deberta --select simplex
uv run simplexlora-grid nlg --select lora
uv run simplexlora-grid squad --select weight-lora
uv run simplexlora-grid qwen
uv run simplexlora-grid additional --select dotprod
```

`--select` accepts exact names, glob patterns, repeated values, or comma-separated
values. Without `--select` or `--all`, the runner uses `settings.default_select`
when present and otherwise runs every experiment.

## Configuration schema

```toml
[settings]
manifest = "outputs/grid/example.json"
default_select = ["simplex"]
stop_on_error = true

[settings.defaults]
env = { CUDA_VISIBLE_DEVICES = "0" }

[settings.defaults.arguments]
model_name_or_path = "microsoft/deberta-v3-base"

[[experiment]]
name = "simplex"
command = "simplexlora-glue"

[experiment.matrix]
task_name = ["mrpc", "rte"]
seed = [18, 52]

[experiment.arguments]
task_name = "{task_name}"
seed = "{seed}"
do_eval = true
```

The runner starts `command` directly, so commands must be installed executable
entry points available through uv. `env` is merged into the current environment.
Arguments with boolean `true` become flags; boolean `false`, `None`, and empty
strings are omitted.

Matrix tables use a Cartesian product. A matrix value may also contain tables:

```toml
[experiment.matrix]
dataset = [
  { dataset_name = "xsum", dataset_config = "" },
  { dataset_name = "cnn_dailymail", dataset_config = "3.0.0" },
]
```

All table fields become placeholders, so `{dataset_name}` and
`{dataset_config}` can be used in arguments and environment variables.

## Execution state

Before every command the runner writes a JSON manifest atomically and records
the expanded context, command, environment, return code, status and timestamps.
Use `--resume` to skip runs recorded as successful. Failed or interrupted runs
are executed again.

Useful options:

```bash
uv run simplexlora-grid glue-deberta --dry-run
uv run simplexlora-grid glue-deberta --resume
uv run simplexlora-grid nlg --keep-going
uv run simplexlora-grid additional --select memory --max-runs 5
uv run simplexlora-grid /path/to/custom.toml --manifest /tmp/custom-grid.json
```

## SimplexLoRA arguments

- `lora_r`: base LoRA rank.
- `fat_step`: optimizer steps between rank recalculations.
- `max_fat_steps`: number of rank recalculations.
- `learning_rate`: learning rate for the LoRA A/B matrices.
- `learning_rate_w`: learning rate for layer weights.
- `ft_strategy`: PEFT strategy. SimplexLoRA uses `WeightLoRA` plus `use_fat`.
- `lora_extention`: adapter resizing implementation.
- `use_fat`: enables rank redistribution.

Keep `ft_strategy`, `lora_extention`, and `use_fat` consistent when editing a
SimplexLoRA run.
