#!/usr/bin/env python3
"""Run and summarize GAE relational-feature ablations on HIH predictions.

This launcher deliberately reuses the formal Foot and Cobb trainers.  The
three ablation conditions zero-mask selected inputs to the unchanged GAE:

* distance only: distance edge channels;
* distance + orientation: all edge channels;
* distance + triplets: distance edge channels and the triplet branch.

The full-relational row is not retrained.  ``--task summarize`` imports it from
the existing formal standardized GAE results and verifies that every ablation
has the same trainable parameter count.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "outputs" / "benchmarks" / "hih_gae_feature_ablation"

ABLATION_MODELS = (
    "gae_distance_only",
    "gae_distance_orientation",
    "gae_distance_triplets",
)
CONDITION_ORDER = (*ABLATION_MODELS, "gae")
CONDITION_LABELS = {
    "gae_distance_only": "Distance only",
    "gae_distance_orientation": "Distance + orientation",
    "gae_distance_triplets": "Distance + triplets",
    "gae": "Full relational features (GAE)",
}

TASKS = {
    "foot": {
        "trainer": EXPERIMENT_ROOT / "train_paired_joint_geometry_models.py",
        "config": EXPERIMENT_ROOT / "controlled_geometry_standardized_config.json",
        "output": OUTPUT_ROOT / "foot",
        "full_results": (
            REPO_ROOT / "outputs" / "benchmarks" /
            "joint_geometry_hih_oof_paired_coordstem_standardized" / "results.json"
        ),
        "summary_key": "test_summary_across_pipeline_seeds",
        "targets": ("HVA", "IMA"),
        "expected_parameters": 50114,
    },
    "cobb": {
        "trainer": EXPERIMENT_ROOT / "train_joint_geometry_models.py",
        "config": EXPERIMENT_ROOT / "aasce_geometry_coordstem_standardized_config.json",
        "output": OUTPUT_ROOT / "cobb",
        "full_results": (
            REPO_ROOT / "outputs" / "benchmarks" /
            "joint_geometry_aasce_hih_oof_coordstem_standardized" / "results.json"
        ),
        "summary_key": "test_summary_across_seeds",
        "targets": ("Cobb",),
        "expected_parameters": 63873,
    },
}

METRICS = (
    "mae", "icc_2_1", "within_tolerance_percent",
    "bias", "loa_lower", "loa_upper",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("foot", "cobb", "summarize"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_training(task: str, device: str, dry_run: bool) -> None:
    specification = TASKS[task]
    command = [
        sys.executable,
        str(specification["trainer"]),
        "--repo-root", str(REPO_ROOT),
        "--config", str(specification["config"]),
        "--output", str(specification["output"]),
        "--models", *ABLATION_MODELS,
        "--device", device,
    ]
    if dry_run:
        command.append("--dry-run")
    print(json.dumps({
        "status": "launching",
        "task": task,
        "full_relational_policy": "reuse existing formal GAE; do not retrain",
        "command": command,
    }, indent=2), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def paired_seeds(document: dict, task: str) -> list[tuple[int, int]]:
    if task == "foot":
        return [
            (int(item["detector_seed"]), int(item["downstream_seed"]))
            for item in document["pipeline_seed_pairing"]
        ]
    return [
        (int(item["detector_seed"]), int(item["downstream_seed"]))
        for item in document["seed_pairs"]
    ]


def summarize() -> None:
    output = {
        "protocol_version": "hih-predicted-gae-feature-ablation-v1",
        "feature_protocol": {
            "distance": "edge channels 0:2 (distance/max and log distance/median)",
            "orientation": "edge channels 2:4 (cos theta and sin theta)",
            "triplets": "complete ordered vertex-angle cosine/sine branch",
            "control": (
                "all conditions use the original full GAE architecture; omitted "
                "components are zero-masked, so parameter count is identical"
            ),
            "full_relational": "existing formal GAE result, not retrained",
        },
        "tasks": {},
    }
    csv_rows = []
    for task, specification in TASKS.items():
        ablation_path = Path(specification["output"]) / "results.json"
        ablation = read_json(ablation_path)
        full = read_json(Path(specification["full_results"]))
        expected_pairs = [(17, 17), (42, 42), (73, 73)]
        if paired_seeds(ablation, task) != expected_pairs:
            raise ValueError(f"{task} ablation is not the formal paired-seed protocol")
        if paired_seeds(full, task) != expected_pairs:
            raise ValueError(f"{task} full GAE reference is not paired 17/42/73")

        task_block = {
            "ablation_results": str(ablation_path),
            "full_gae_results": str(specification["full_results"]),
            "paired_seeds": expected_pairs,
            "conditions": {},
        }
        for model_name in CONDITION_ORDER:
            source = full if model_name == "gae" else ablation
            model = source.get("models", {}).get(model_name)
            if model is None:
                raise ValueError(f"{task} is missing completed model {model_name}")
            parameter_count = int(source["architecture_audit"][model_name]["parameters"])
            if parameter_count != int(specification["expected_parameters"]):
                raise ValueError(
                    f"{task}/{model_name} has {parameter_count} parameters; expected "
                    f"{specification['expected_parameters']}"
                )
            summary = model[specification["summary_key"]]
            condition = {
                "label": CONDITION_LABELS[model_name],
                "parameters": parameter_count,
                "targets": {},
            }
            for target in specification["targets"]:
                condition["targets"][target] = {
                    metric: summary[target][metric] for metric in METRICS
                }
                row = {
                    "task": task,
                    "target": target,
                    "condition": CONDITION_LABELS[model_name],
                    "parameters": parameter_count,
                }
                for metric in METRICS:
                    row[f"{metric}_mean"] = summary[target][metric]["mean"]
                    row[f"{metric}_sd"] = summary[target][metric]["sd"]
                csv_rows.append(row)
            task_block["conditions"][model_name] = condition
        output["tasks"][task] = task_block

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2) + "\n")
    csv_path = OUTPUT_ROOT / "table.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    compact = {"status": "PASS", "summary": str(summary_path), "csv": str(csv_path)}
    for task, task_block in output["tasks"].items():
        compact[task] = {}
        for condition_name, condition in task_block["conditions"].items():
            compact[task][condition_name] = {
                target: {
                    "mae": values["mae"],
                    "icc_2_1": values["icc_2_1"],
                }
                for target, values in condition["targets"].items()
            }
    print(json.dumps(compact, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.task == "summarize":
        if args.dry_run:
            raise ValueError("--dry-run applies only to foot/cobb training")
        summarize()
    else:
        run_training(args.task, args.device, args.dry_run)


if __name__ == "__main__":
    main()
