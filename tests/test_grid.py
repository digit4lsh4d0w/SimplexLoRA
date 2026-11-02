from __future__ import annotations

import json
import unittest
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

import tomllib

from simplexlora.experiments.grid import GridRun, build_runs, main


def load_builtin_config(name: str) -> dict:
    resource = files("simplexlora.experiments.configs").joinpath(name)
    return tomllib.loads(resource.read_text(encoding="utf-8"))


class GridTests(unittest.TestCase):
    def test_expands_matrix_and_renders_arguments(self) -> None:
        config = {
            "experiment": [
                {
                    "name": "sample",
                    "command": "simplexlora-glue",
                    "env": {"GPU": "{gpu}"},
                    "matrix": {"gpu": [0, 1], "seed": [10, 20]},
                    "arguments": {
                        "task_name": "mrpc",
                        "gpu": "{gpu}",
                        "seed": "{seed}",
                        "do_eval": True,
                        "do_predict": False,
                        "empty": "",
                    },
                }
            ]
        }

        runs = build_runs(config, Path("/tmp"))

        self.assertEqual(len(runs), 4)
        self.assertEqual(runs[0].env, {"GPU": "0"})
        self.assertIn("--do-eval", runs[0].command)
        self.assertNotIn("--do-predict", runs[0].command)
        self.assertNotIn("--empty", runs[0].command)
        self.assertEqual(runs[0].cwd, Path("/tmp"))
        self.assertEqual(len({run.run_id for run in runs}), 4)

    def test_run_id_is_stable(self) -> None:
        first = GridRun("test", {"seed": 1}, ("tool", "--seed", "1"), {}, Path("/tmp"))
        second = GridRun("test", {"seed": 1}, ("tool", "--seed", "1"), {}, Path("/tmp"))

        self.assertEqual(first.run_id, second.run_id)

    def test_builtin_configs_expand(self) -> None:
        expected_counts = {
            "decompile-hparam.toml": 6,
            "glue-deberta.toml": 20,
            "nlg.toml": 357,
            "squad.toml": 16,
            "qwen.toml": 6,
            "additional.toml": 216,
        }

        for name, expected in expected_counts.items():
            with self.subTest(config=name):
                runs = build_runs(load_builtin_config(name), Path.cwd())
                self.assertEqual(len(runs), expected)

    def test_runs_command_and_records_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "grid.toml"
            manifest_path = root / "manifest.json"
            config_path.write_text(
                """
[[experiment]]
name = "true"
command = ["/bin/true"]
""",
                encoding="utf-8",
            )

            returncode = main([str(config_path), "--manifest", str(manifest_path)])
            first_manifest = manifest_path.read_text(encoding="utf-8")
            resume_returncode = main([str(config_path), "--manifest", str(manifest_path), "--resume"])
            resumed_manifest = manifest_path.read_text(encoding="utf-8")

        data = json.loads(first_manifest)
        resumed_data = json.loads(resumed_manifest)
        self.assertEqual(returncode, 0)
        self.assertEqual(resume_returncode, 0)
        self.assertEqual(next(iter(data["runs"].values()))["status"], "success")
        self.assertEqual(
            next(iter(data["runs"].values()))["started_at"],
            next(iter(resumed_data["runs"].values()))["started_at"],
        )


if __name__ == "__main__":
    unittest.main()
