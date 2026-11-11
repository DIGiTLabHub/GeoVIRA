#!/usr/bin/env python3
"""Repeated realistic-donor null for the frozen geometry grounding audit.

Attention maps, valid-token masks, predicted measurement geometry, and model
outputs are immutable.  The only operation is repeated within-test-set donor
derangement of real HIH-predicted geometry.  No model is loaded or trained.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "geomed" / "src"))
sys.path.insert(0, str(SCRIPT_ROOT))

import audit_geometry as fixed_control  # noqa: E402
import audit_ordinary_attention_geometry_alignment as alignment  # noqa: E402
import audit_geometry as geometry_helpers  # noqa: E402
import train_predicted_landmark_frozen_vision_fusion as fusion  # noqa: E402


TASKS = alignment.TASKS
SEEDS = alignment.SEEDS
DEFAULT_REPETITIONS = 1000
BOOTSTRAP_REPLICATES = 10000
AUDIT_ROOT = alignment.DEFAULT_OUTPUT


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def donor_permutation(task: str, arrays, detector_seed: int, repetition: int):
    # Distinct deterministic seeds; the construction, not a searched result,
    # decides each donor set.  Offset 155921 is the established test offset.
    shuffle_seed = (
        int(detector_seed) * 1_000_003 + 155_921
        + (int(repetition) + 1) * 32_452_843
    )
    if task == "foot":
        permutation = fusion.patient_derangement(
            arrays["test"].patient_ids, shuffle_seed
        )
        if np.any(
            arrays["test"].patient_ids[permutation]
            == arrays["test"].patient_ids
        ):
            raise AssertionError("Foot repeated donor shares recipient patient")
    else:
        permutation = fusion.sattolo_derangement(
            len(arrays["test"].identifiers), shuffle_seed
        )
    if np.any(permutation == np.arange(len(permutation))):
        raise AssertionError("repeated donor contains self-pair")
    return permutation


def repeated_null(weights, valid, raw_geometry, permutations):
    """Compute R_donor efficiently while preserving recipient valid regions."""
    average = np.asarray(weights, np.float64).mean(axis=1)
    average *= valid
    average /= average.sum(axis=1, keepdims=True)
    valid_count = valid.sum(axis=1)
    null = np.empty((len(valid), len(permutations)), np.float32)
    for repetition, permutation in enumerate(permutations):
        donor = raw_geometry[permutation] & valid
        geometry_count = donor.sum(axis=1)
        if np.any(geometry_count == 0):
            raise ValueError("donor geometry is empty in recipient valid region")
        mass = (average * donor).sum(axis=1)
        null[:, repetition] = mass / (geometry_count / valid_count)
    return null


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repetitions < 500:
        raise ValueError("formal repeated-donor audit requires at least 500 repetitions")
    original = json.loads(
        (AUDIT_ROOT / "geometry_attention_alignment_summary.json").read_text()
    )
    rows, summary = [], {
        "analysis_only": True, "models_loaded": 0, "models_trained": 0,
        "attention_maps_changed": False, "geometry_definition_changed": False,
        "repetitions": args.repetitions,
        "null": "real predicted measurement geometry from other test cases",
        "constraints": {
            "foot": "different donor patient at every repetition",
            "cobb": "no self donor at every repetition",
        },
        "per_case_null_p": "(1 + count(R_donor >= R_self)) / (B + 1)",
        "delta_R": "R_self minus mean repeated-donor R",
        "tasks": {},
    }
    maximum_self_error = 0.0
    for task in TASKS:
        adapter = fusion.load_adapter(
            task, fusion.DEFAULT_FOOT_CONFIG, fusion.DEFAULT_COBB_CONFIG
        )
        records = fusion.records_for_task(task)
        valid = alignment.valid_token_masks(records, "test")
        summary["tasks"][task] = {"seeds": {}}
        for seed in SEEDS:
            arrays, audit = adapter.benchmark.load_data(
                REPO_ROOT, adapter.config, seed
            )
            geometry = geometry_helpers.geometry_for_seed(task, arrays, records)["test"]
            raw_geometry = alignment.bilinear_support_mask(geometry["segments"])
            attention_path = Path(
                original["tasks"][task]["seeds"][str(seed)]["attention_file"]
            )
            with np.load(attention_path) as archive:
                identifiers = archive["identifiers"].astype(str)
                weights = np.asarray(archive["attention_per_head"], np.float64)
                saved_self = np.asarray(archive["R_geo"], np.float64)
            if not np.array_equal(identifiers, arrays["test"].identifiers):
                raise ValueError(f"identifier order changed: {task}/{seed}")
            self_ratio, _, _, _ = alignment.enrichment(
                weights, valid, raw_geometry
            )
            self_error = float(np.max(np.abs(self_ratio - saved_self)))
            maximum_self_error = max(maximum_self_error, self_error)
            if self_error > 2e-6:
                raise RuntimeError(f"R_self reconstruction failed: {self_error}")

            permutations = [
                donor_permutation(task, arrays, seed, repetition)
                for repetition in range(args.repetitions)
            ]
            unique_permutations = len({p.tobytes() for p in permutations})
            if unique_permutations != args.repetitions:
                raise RuntimeError("repeated donor generator produced duplicate permutations")
            null = repeated_null(weights, valid, raw_geometry, permutations)
            null_mean = null.mean(axis=1, dtype=np.float64)
            delta = self_ratio - null_mean
            per_case_p = (
                1.0 + np.sum(null >= self_ratio[:, None], axis=1)
            ) / float(args.repetitions + 1)
            clusters = (
                np.asarray(arrays["test"].patient_ids)
                if task == "foot" else np.arange(len(identifiers))
            )
            paired = fixed_control.paired_cluster_bootstrap(
                delta, clusters,
                1_000_003 * seed + (211 if task == "foot" else 307),
            )
            # Dataset-level empirical comparison against B donor-set means.
            null_dataset_means = null.mean(axis=0, dtype=np.float64)
            dataset_p = float(
                (1 + np.sum(null_dataset_means >= self_ratio.mean()))
                / (args.repetitions + 1)
            )
            item = {
                "n": len(identifiers), "repetitions": args.repetitions,
                "unique_donor_permutations": unique_permutations,
                "predicted_geometry_source": audit["paths"]["test"],
                "R_self_mean": float(self_ratio.mean()),
                "R_self_median": float(np.median(self_ratio)),
                "R_donor_null_mean": float(null.mean()),
                "R_donor_null_median_of_case_means": float(np.median(null_mean)),
                "R_donor_dataset_mean_q1": float(np.quantile(null_dataset_means, .25)),
                "R_donor_dataset_mean_q3": float(np.quantile(null_dataset_means, .75)),
                "delta_R_mean": float(delta.mean()),
                "delta_R_median": float(np.median(delta)),
                "fraction_delta_R_positive": float(np.mean(delta > 0)),
                "paired_bootstrap": paired,
                "dataset_level_empirical_p": dataset_p,
                "per_case_empirical_p_median": float(np.median(per_case_p)),
                "fraction_per_case_p_below_0_05": float(np.mean(per_case_p < .05)),
                "fraction_per_case_p_below_0_01": float(np.mean(per_case_p < .01)),
                "R_self_reconstruction_max_abs_error": self_error,
            }
            summary["tasks"][task]["seeds"][str(seed)] = item
            null_path = (
                AUDIT_ROOT / "repeated_donor_null" / task /
                f"seed_{seed}_R_donor_null.npz"
            )
            null_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                null_path, identifiers=identifiers,
                R_self=self_ratio.astype(np.float32),
                R_donor_null=null,
                R_donor_mean_per_case=null_mean.astype(np.float32),
                delta_R=delta.astype(np.float32),
                empirical_p=per_case_p.astype(np.float32),
                donor_permutations=np.asarray(permutations, np.int16),
                repetitions=np.asarray(args.repetitions),
                no_gt_geometry=np.asarray(True),
            )
            item["full_null_file"] = str(null_path)
            for index, identifier in enumerate(identifiers):
                rows.append({
                    "task": task, "seed": seed, "identifier": identifier,
                    "patient_id": str(clusters[index]) if task == "foot" else "",
                    "R_self": float(self_ratio[index]),
                    "R_donor_mean": float(null_mean[index]),
                    "R_donor_median": float(np.median(null[index])),
                    "R_donor_q1": float(np.quantile(null[index], .25)),
                    "R_donor_q3": float(np.quantile(null[index], .75)),
                    "delta_R": float(delta[index]),
                    "empirical_p": float(per_case_p[index]),
                })
            print(json.dumps({
                "task": task, "seed": seed,
                "R_self_mean": item["R_self_mean"],
                "R_donor_null_mean": item["R_donor_null_mean"],
                "delta_R_mean": item["delta_R_mean"],
                "mean_delta_ci95": paired["mean_delta_ci95"],
                "dataset_level_empirical_p": dataset_p,
            }), flush=True)
    summary["R_self_reconstruction_max_abs_error"] = maximum_self_error
    summary["no_gt_landmarks_or_geometry"] = True
    summary["scientific_caveat"] = (
        "Post-hoc attention correspondence does not establish causal dependence."
    )
    csv_path = AUDIT_ROOT / "repeated_donor_null_per_case.csv"
    json_path = AUDIT_ROOT / "repeated_donor_null_summary.json"
    write_csv(csv_path, rows); alignment.write_json(json_path, summary)
    print(json.dumps({"status": "PASS", "models_trained": 0,
                      "summary": str(json_path), "per_case": str(csv_path),
                      "full_null_directory": str(AUDIT_ROOT / "repeated_donor_null")},
                     indent=2))


if __name__ == "__main__":
    main()
