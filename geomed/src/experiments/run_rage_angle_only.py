#!/usr/bin/env python3
"""Train the RAGE angle-only control and summarize relational ablations.

The new condition retains the complete GAE and prediction head but masks all
four pairwise-edge input channels.  Only the ordered triplet-angle branch
contains case-specific geometry.  Existing feature-ablation and formal GAE
results are read, never modified, when producing:

1. module-level: Edge only / Angle only / Edge + Angle;
2. within-edge: Distance only / Distance + orientation.
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
OUTPUT_ROOT = REPO_ROOT / "outputs" / "benchmarks" / "hih_gae_angle_only"
OLD_ABLATION_ROOT = (
    REPO_ROOT / "outputs" / "benchmarks" / "hih_gae_feature_ablation"
)

TASKS = {
    "foot": {
        "trainer": EXPERIMENT_ROOT / "train_paired_joint_geometry_models.py",
        "config": (
            EXPERIMENT_ROOT / "controlled_geometry_coordstem_standardized_config.json"
        ),
        "output": OUTPUT_ROOT / "foot",
        "old_ablation": OLD_ABLATION_ROOT / "foot" / "results.json",
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
        "old_ablation": OLD_ABLATION_ROOT / "cobb" / "results.json",
        "full_results": (
            REPO_ROOT / "outputs" / "benchmarks" /
            "joint_geometry_aasce_hih_oof_coordstem_standardized" / "results.json"
        ),
        "summary_key": "test_summary_across_seeds",
        "targets": ("Cobb",),
        "expected_parameters": 63873,
    },
}

MODULE_CONDITIONS = (
    ("edge_only", "Edge only", "old", "gae_distance_orientation"),
    ("angle_only", "Angle only", "new", "gae_angle_only"),
    ("edge_angle", "Edge + Angle (GAE)", "full", "gae"),
)
WITHIN_EDGE_CONDITIONS = (
    ("distance_only", "Distance only", "old", "gae_distance_only"),
    (
        "distance_orientation", "Distance + orientation (Edge only)",
        "old", "gae_distance_orientation",
    ),
)
METRICS = (
    "mae", "icc_2_1", "within_tolerance_percent",
    "bias", "loa_lower", "loa_upper",
)
FORMAL_PROTOCOL_KEYS = (
    "batch_size", "num_workers", "amp", "max_epochs", "patience",
    "optimizer", "scheduler", "loss", "gradient_clip_norm",
    "search_space", "checkpoint_selection", "target_standardization",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("foot", "cobb", "summarize"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true",
        help="Allow replacing this new angle-only task output only.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def paired_seeds(document: dict, task: str) -> list[tuple[int, int]]:
    key = "pipeline_seed_pairing" if task == "foot" else "seed_pairs"
    return [
        (int(item["detector_seed"]), int(item["downstream_seed"]))
        for item in document[key]
    ]


def validate_formal_protocol(task: str, specification: dict) -> dict:
    """Fail before training if the selected config differs from formal GAE."""
    config = read_json(specification["config"])
    formal = read_json(specification["full_results"])
    if config["protocol_version"] != formal["protocol_version"]:
        raise ValueError(
            f"{task} protocol mismatch: {config['protocol_version']} != "
            f"{formal['protocol_version']}"
        )
    formal_training = formal["training_protocol"]
    mismatches = [
        key for key in FORMAL_PROTOCOL_KEYS
        if config.get(key) != formal_training.get(key)
    ]
    if mismatches:
        raise ValueError(f"{task} differs from formal GAE for: {mismatches}")
    expected_pairs = [(17, 17), (42, 42), (73, 73)]
    if paired_seeds(formal, task) != expected_pairs:
        raise ValueError(f"{task} formal GAE does not use paired seeds 17/42/73")
    return {
        "status": "PASS",
        "formal_protocol_version": formal["protocol_version"],
        "matched_fields": list(FORMAL_PROTOCOL_KEYS),
        "paired_seeds": expected_pairs,
    }


def train(task: str, device: str, dry_run: bool, force: bool) -> None:
    specification = TASKS[task]
    formal_protocol_audit = validate_formal_protocol(task, specification)
    results_path = specification["output"] / "results.json"
    if results_path.exists() and not force and not dry_run:
        raise FileExistsError(f"{results_path} exists; pass --force to replace it")
    command = [
        sys.executable,
        str(specification["trainer"]),
        "--repo-root", str(REPO_ROOT),
        "--config", str(specification["config"]),
        "--output", str(specification["output"]),
        "--models", "gae_angle_only",
        "--device", device,
    ]
    if dry_run:
        command.append("--dry-run")
    print(json.dumps({
        "status": "launching",
        "task": task,
        "condition": "Angle only",
        "edge_policy": "mask all four edge channels",
        "triplet_policy": "retain complete ordered angle branch",
        "existing_results_modified": False,
        "formal_gae_protocol_audit": formal_protocol_audit,
        "command": command,
    }, indent=2), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def condition_block(
    task: str, specification: dict, documents: dict,
    source_name: str, model_name: str, label: str,
) -> tuple[dict, list[dict]]:
    source = documents[source_name]
    model = source.get("models", {}).get(model_name)
    if model is None:
        raise ValueError(f"{task}/{source_name} is missing {model_name}")
    parameters = int(source["architecture_audit"][model_name]["parameters"])
    if parameters != int(specification["expected_parameters"]):
        raise ValueError(
            f"{task}/{model_name} has {parameters} parameters; expected "
            f"{specification['expected_parameters']}"
        )
    summary = model[specification["summary_key"]]
    block = {"label": label, "parameters": parameters, "targets": {}}
    rows = []
    for target in specification["targets"]:
        block["targets"][target] = {
            metric: summary[target][metric] for metric in METRICS
        }
        row = {
            "task": task, "target": target, "condition": label,
            "parameters": parameters,
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = summary[target][metric]["mean"]
            row[f"{metric}_sd"] = summary[target][metric]["sd"]
        rows.append(row)
    return block, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize() -> None:
    expected_pairs = [(17, 17), (42, 42), (73, 73)]
    output = {
        "protocol_version": "hih-predicted-gae-module-ablation-v1",
        "new_training": "angle-only only; existing edge-only/full results reused",
        "mask_definition": {
            "edge_only": "distance and orientation edge channels; triplets masked",
            "angle_only": "all edge channels masked; triplet-angle branch retained",
            "edge_angle": "distance, orientation, and triplet-angle branches",
        },
        "tasks": {},
    }
    module_rows: list[dict] = []
    within_rows: list[dict] = []
    for task, specification in TASKS.items():
        documents = {
            "new": read_json(specification["output"] / "results.json"),
            "old": read_json(specification["old_ablation"]),
            "full": read_json(specification["full_results"]),
        }
        for source_name, document in documents.items():
            if paired_seeds(document, task) != expected_pairs:
                raise ValueError(
                    f"{task}/{source_name} is not paired detector/downstream "
                    "seeds 17/17, 42/42, 73/73"
                )
        reference_training = documents["full"]["training_protocol"]
        for source_name, document in documents.items():
            mismatches = [
                key for key in FORMAL_PROTOCOL_KEYS
                if document["training_protocol"].get(key)
                != reference_training.get(key)
            ]
            if mismatches:
                raise ValueError(
                    f"{task}/{source_name} training protocol differs from full "
                    f"GAE for: {mismatches}"
                )
        task_output = {
            "paired_seeds": expected_pairs,
            "sources": {
                "angle_only": str(specification["output"] / "results.json"),
                "existing_feature_ablation": str(specification["old_ablation"]),
                "existing_full_gae": str(specification["full_results"]),
            },
            "module_level": {},
            "within_edge": {},
        }
        for key, label, source_name, model_name in MODULE_CONDITIONS:
            block, rows = condition_block(
                task, specification, documents, source_name, model_name, label
            )
            task_output["module_level"][key] = block
            module_rows.extend(rows)
        for key, label, source_name, model_name in WITHIN_EDGE_CONDITIONS:
            block, rows = condition_block(
                task, specification, documents, source_name, model_name, label
            )
            task_output["within_edge"][key] = block
            within_rows.extend(rows)
        output["tasks"][task] = task_output

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(output, indent=2) + "\n")
    module_csv = OUTPUT_ROOT / "module_level_table.csv"
    within_csv = OUTPUT_ROOT / "within_edge_table.csv"
    write_csv(module_csv, module_rows)
    write_csv(within_csv, within_rows)
    print(json.dumps({
        "status": "PASS",
        "summary": str(summary_path),
        "module_level_table": str(module_csv),
        "within_edge_table": str(within_csv),
    }, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.task == "summarize":
        if args.dry_run or args.force:
            raise ValueError("--dry-run/--force do not apply to summarize")
        summarize()
    else:
        train(args.task, args.device, args.dry_run, args.force)


if __name__ == "__main__":
    main()
