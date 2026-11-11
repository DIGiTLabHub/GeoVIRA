#!/usr/bin/env python3
"""Case-shuffled controls for the completed localized MedImageInsight models.

This script trains exactly one new condition per invocation:

* ``--task foot``: axis-aligned HVA/IMA geometry receives all three localized
  MII axis tokens from a different test patient/case;
* ``--task cobb``: line-aligned Cobb geometry receives both localized MII line
  tokens from a different image.

Two explicitly named shuffle levels are supported. ``token-set`` reproduces
the original control by permuting already-pooled localized token sets.
``feature-map`` permutes complete frozen 15x15 MII maps and then resamples each
donor map with the recipient case's predicted coordinates. Neither mode loads
the image encoder or an HIH checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "geomed" / "src"))
sys.path.insert(0, str(SCRIPT_ROOT))

import train_aasce_cobb_measurement_line_mii as cobb  # noqa: E402
import train_hvangle_axis_aligned_mii as foot  # noqa: E402
import train_predicted_landmark_mii_cross_attention as cross  # noqa: E402
import train_predicted_landmark_frozen_vision_fusion as shuffle  # noqa: E402


SEEDS = (17, 42, 73)
SPLITS = ("train", "val", "test")
FOOT_CONDITION = "axis_aligned_image_shuffled"
COBB_CONDITION = "measurement_line_aligned_image_shuffled"
FOOT_MAP_CONDITION = "axis_aligned_feature_map_shuffled"
COBB_MAP_CONDITION = "measurement_line_aligned_feature_map_shuffled"
FOOT_SOURCE = REPO_ROOT / "outputs" / "benchmarks" / "hvangle_axis_aligned_mii"
COBB_SOURCE = (
    REPO_ROOT / "outputs" / "benchmarks" / "aasce_cobb_measurement_line_mii"
)
FOOT_OUTPUT = (
    REPO_ROOT / "outputs" / "benchmarks" /
    "hvangle_axis_aligned_mii_image_shuffled"
)
COBB_OUTPUT = (
    REPO_ROOT / "outputs" / "benchmarks" /
    "aasce_cobb_line_aligned_mii_image_shuffled"
)
FOOT_MAP_OUTPUT = (
    REPO_ROOT / "outputs" / "benchmarks" /
    "hvangle_axis_aligned_mii_feature_map_shuffled"
)
COBB_MAP_OUTPUT = (
    REPO_ROOT / "outputs" / "benchmarks" /
    "aasce_cobb_line_aligned_mii_feature_map_shuffled"
)
CROSS_ATTENTION_CACHE = (
    REPO_ROOT / "outputs" / "benchmarks" /
    "predicted_landmark_mii_cross_attention"
)


def load_task(task: str, config: dict):
    arrays_by_seed, data_audits = {}, {}
    module = foot if task == "foot" else cobb
    for seed in SEEDS:
        arrays_by_seed[seed], data_audits[str(seed)] = module.benchmark.load_data(
            REPO_ROOT, config, seed
        )
    return arrays_by_seed, data_audits


def load_source_features(task: str, arrays: dict, seed: int):
    if task == "foot":
        return foot.load_axis_features(FOOT_SOURCE, arrays, seed)
    return cobb.load_features(
        COBB_SOURCE, "cobb_axis_aligned_matched", arrays, seed,
        cobb.DEFAULT_ALPHA,
    )


def shuffled_features(task: str, arrays: dict, seed: int):
    source = load_source_features(task, arrays, seed)
    permutations = shuffle.split_permutations(task, arrays, seed)
    audit = shuffle.permutation_audit(
        task, arrays, None, permutations, seed
    )
    result = {}
    for split in SPLITS:
        tokens = np.asarray(source[split], dtype=np.float32)
        permutation = permutations[split]
        result[split] = np.ascontiguousarray(tokens[permutation])
        # A permutation must preserve every token exactly, not resample or
        # recompute it.  Invert it and require exact floating-point equality.
        inverse = np.argsort(permutation)
        if not np.array_equal(result[split][inverse], tokens):
            raise AssertionError(f"{task}/{seed}/{split}: token permutation changed values")
        audit[split].update({
            "token_shape": list(tokens.shape),
            "token_multiset_preserved_exactly": True,
            "anatomical_role_order_preserved": True,
        })
    return result, audit


def recipient_mappings(records: dict) -> dict[str, np.ndarray]:
    """Reproduce the exact original-image to 224-letterbox mapping."""
    result = {}
    for split in SPLITS:
        rows = []
        for record in records[split]:
            with Image.open(record.image_path) as image:
                _, mapping = foot.letterbox_with_mapping(image.convert("RGB"))
            rows.append(mapping)
        result[split] = np.asarray(rows, dtype=np.float32)
    return result


def feature_map_shuffled_features(
    task: str,
    arrays: dict,
    data_audits: dict,
    records: dict,
    mappings: dict,
    seed: int,
    device: torch.device,
    batch_size: int,
    lines_by_seed: dict | None,
):
    """Shuffle full MII maps, then sample with recipient predicted geometry."""
    if not cross.validate_patch_cache(CROSS_ATTENTION_CACHE, task, records):
        raise FileNotFoundError(
            f"frozen MII map cache is unavailable under {CROSS_ATTENTION_CACHE}"
        )
    permutations = shuffle.split_permutations(task, arrays, seed)
    audit = shuffle.permutation_audit(task, arrays, records, permutations, seed)
    old_localized = load_source_features(task, arrays, seed)
    if task == "foot":
        geometry = {
            split: foot.load_predicted_points(data_audits, {seed: arrays}, seed, split)
            for split in SPLITS
        }
    else:
        if lines_by_seed is None:
            raise ValueError("Cobb feature-map shuffling requires predicted lines")
        geometry = lines_by_seed[seed]

    result = {}
    for split in SPLITS:
        patch_file = cross.patch_path(CROSS_ATTENTION_CACHE, task, split)
        patches = np.load(patch_file, mmap_mode="r")
        permutation = permutations[split]
        number_of_roles = 3 if task == "foot" else 2
        tokens = np.empty(
            (len(permutation), number_of_roles, foot.SPATIAL_CHANNELS),
            dtype=np.float32,
        )
        invalid_count = 0
        for start in range(0, len(permutation), batch_size):
            stop = min(start + batch_size, len(permutation))
            donor_maps = np.asarray(
                patches[permutation[start:stop]], dtype=np.float32
            )
            donor_maps_t = torch.from_numpy(donor_maps).to(device)
            if task == "foot":
                sampled = foot.sample_axis_tokens(
                    donor_maps_t, (15, 15), geometry[split][start:stop],
                    mappings[split][start:stop],
                )
            else:
                sampled, diagnostics = cobb.sample_lines(
                    donor_maps_t, (15, 15), geometry[split][start:stop],
                    mappings[split][start:stop],
                )
                invalid_count += int(np.asarray(diagnostics["invalid"]).sum())
            tokens[start:stop] = sampled.float().cpu().numpy()
        if invalid_count:
            raise RuntimeError(
                f"{task}/{seed}/{split}: {invalid_count} recipient primitives invalid"
            )
        direct_token_shuffle = np.asarray(
            old_localized[split][permutation], dtype=np.float32
        )
        difference = np.abs(tokens - direct_token_shuffle)
        if not np.any(difference > 1e-6):
            raise AssertionError(
                "map/current-coordinate resampling equals token-set shuffling"
            )
        result[split] = tokens
        audit[split].update({
            "token_shape": list(tokens.shape),
            "full_feature_map_shape": [15, 15, foot.SPATIAL_CHANNELS],
            "full_feature_map_deranged": True,
            "recipient_predicted_coordinates_used": True,
            "localized_tokens_recomputed_after_map_derangement": True,
            "anatomical_role_order_preserved": True,
            "mean_abs_difference_vs_token_set_shuffle": float(difference.mean()),
            "maximum_abs_difference_vs_token_set_shuffle": float(difference.max()),
            "invalid_primitive_count": invalid_count,
            "patch_cache": str(patch_file),
        })
    return result, audit


def architecture_audit(task: str, arrays_by_seed: dict, condition: str):
    sample = arrays_by_seed[SEEDS[0]]["train"]
    relational = torch.from_numpy(sample.relational[:2])
    edges = torch.from_numpy(sample.edges[:2])
    if task == "foot":
        model = foot.AxisAlignedGAE(
            sample.relational.shape[1], sample.edges.shape[1]
        )
        visual = torch.randn(2, 3, foot.SPATIAL_CHANNELS)
        output = model(relational, edges, visual, condition)
        roles = ["toe", "mt1", "mt2"]
    else:
        model = cobb.CobbLineAlignedGAE(
            sample.relational.shape[1], sample.edges.shape[1]
        )
        visual = torch.randn(2, 2, foot.SPATIAL_CHANNELS)
        output = model(relational, edges, visual)
        roles = ["upper", "lower"]
    output.sum().backward()
    if not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise RuntimeError("non-finite gradient in architecture smoke test")
    return {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "output_shape": list(output.shape),
        "forward_backward_smoke_test": "passed",
        "unchanged_model_class": f"{type(model).__module__}.{type(model).__name__}",
        "visual_role_order": roles,
    }


def existing_matched(task: str):
    path = (FOOT_SOURCE if task == "foot" else COBB_SOURCE) / "results.json"
    if not path.is_file():
        return {"path": str(path), "available": False}
    saved = json.loads(path.read_text())
    key = "axis_aligned_matched" if task == "foot" else "cobb_axis_aligned_matched"
    return {
        "path": str(path), "available": True,
        "condition": key,
        "test_summary_across_pipeline_seeds": saved["conditions"][key][
            "test_summary_across_pipeline_seeds"
        ],
    }


def run(args) -> None:
    task = args.task
    module = foot if task == "foot" else cobb
    feature_map_mode = args.shuffle_mode == "feature-map"
    condition = (
        FOOT_MAP_CONDITION if task == "foot" and feature_map_mode
        else COBB_MAP_CONDITION if task == "cobb" and feature_map_mode
        else FOOT_CONDITION if task == "foot" else COBB_CONDITION
    )
    config_path = args.config or (
        foot.DEFAULT_CONFIG if task == "foot" else cobb.DEFAULT_CONFIG
    )
    output = args.output or (
        FOOT_MAP_OUTPUT if task == "foot" and feature_map_mode
        else COBB_MAP_OUTPUT if task == "cobb" and feature_map_mode
        else FOOT_OUTPUT if task == "foot" else COBB_OUTPUT
    )
    config = json.loads(config_path.read_text())
    if tuple(int(seed) for seed in config["seeds"]) != SEEDS:
        raise ValueError("formal paired seeds must be exactly 17, 42, 73")
    if len(config["search_space"]) != 9:
        raise ValueError("formal protocol requires the existing 3x3 LR/WD grid")
    device = foot.resolve_device(args.device)
    arrays_by_seed, data_audits = load_task(task, config)
    architecture = architecture_audit(task, arrays_by_seed, condition)

    records = shuffle.records_for_task(task)
    mappings = recipient_mappings(records) if feature_map_mode else None
    lines_by_seed = None
    line_verification = None
    if task == "cobb" and feature_map_mode:
        lines_by_seed, line_verification = cobb.selected_lines_by_seed(
            arrays_by_seed, data_audits, records
        )

    features_by_seed, shuffle_audits = {}, {}
    for seed in SEEDS:
        if feature_map_mode:
            features_by_seed[seed], shuffle_audits[str(seed)] = (
                feature_map_shuffled_features(
                    task, arrays_by_seed[seed], data_audits, records, mappings,
                    seed, device, args.sampling_batch_size, lines_by_seed,
                )
            )
        else:
            features_by_seed[seed], shuffle_audits[str(seed)] = shuffled_features(
                task, arrays_by_seed[seed], seed
            )

    audit = {
        "status": "PASS",
        "task": task,
        "condition": condition,
        "paired_pipeline_seeds": list(SEEDS),
        "device": str(device),
        "shuffle_mode": args.shuffle_mode,
        "source_feature_cache": str(
            CROSS_ATTENTION_CACHE if feature_map_mode
            else FOOT_SOURCE if task == "foot" else COBB_SOURCE
        ),
        "source_condition": (
            "axis_aligned_matched" if task == "foot"
            else "cobb_axis_aligned_matched"
        ),
        "localized_tokens_recomputed": feature_map_mode,
        "image_encoder_loaded": False,
        "hih_retrained": False,
        "geometry_source": "paired-seed HIH predicted landmarks only",
        "shuffle_unit": (
            "complete frozen 15x15x2048 map before recipient-coordinate sampling"
            if feature_map_mode
            else "case row containing the complete localized token set"
        ),
        "shuffle_constraint": (
            "no self and different patient" if task == "foot" else "no self"
        ),
        "architecture": architecture,
        "cobb_line_verification": line_verification,
        "pairings": shuffle_audits,
    }
    terminal_audit = {
        key: value for key, value in audit.items() if key != "pairings"
    }
    terminal_audit["pairing_summary"] = {
        seed: {
            split: {
                "n": len(row["permutation"]),
                "no_self_pairing": row["no_self_pairing"],
                "different_patient_pairing": row["different_patient_pairing"],
                "token_shape": row["token_shape"],
                "token_multiset_preserved_exactly": row.get(
                    "token_multiset_preserved_exactly"
                ),
                "full_feature_map_deranged": row.get("full_feature_map_deranged"),
                "recipient_predicted_coordinates_used": row.get(
                    "recipient_predicted_coordinates_used"
                ),
                "mean_abs_difference_vs_token_set_shuffle": row.get(
                    "mean_abs_difference_vs_token_set_shuffle"
                ),
            }
            for split, row in splits.items()
        }
        for seed, splits in shuffle_audits.items()
    }
    print(json.dumps(terminal_audit, indent=2), flush=True)
    if args.dry_run:
        return

    results_path = output / "results.json"
    progress_path = output / "training_progress.json"
    if results_path.exists() or progress_path.exists():
        existing = results_path if results_path.exists() else progress_path
        raise FileExistsError(
            f"{existing} exists; formal shuffled-control outputs are never overwritten"
        )
    output.mkdir(parents=True, exist_ok=True)
    results = {
        "protocol_version": "localized-mii-case-shuffled-v2",
        "task": task,
        "condition": condition,
        "paired_pipeline_seeds": list(SEEDS),
        "formal_config": str(config_path),
        "training_protocol": {
            key: config[key] for key in (
                "batch_size", "num_workers", "amp", "max_epochs", "patience",
                "optimizer", "scheduler", "target_standardization", "loss",
                "gradient_clip_norm", "checkpoint_selection", "search_space",
            )
        },
        "audit": audit,
        "data_audit_by_seed": data_audits,
        "existing_matched": existing_matched(task),
        "runs": {},
    }
    foot.write_json(progress_path, results)

    for seed in SEEDS:
        arrays = arrays_by_seed[seed]
        features = features_by_seed[seed]
        trials = []
        for trial, hp in enumerate(config["search_space"]):
            checkpoint = (
                output / condition / "checkpoints" /
                f"pipeline_seed_{seed}_trial_{trial}.pt"
            )
            if task == "foot":
                trained = foot.train_trial(
                    condition, seed, hp, trial, arrays, features, config,
                    device, checkpoint,
                )
            else:
                trained = cobb.train_trial(
                    condition, seed, hp, trial, arrays, features, config,
                    device, checkpoint,
                )
            trials.append(trained)
        selected = min(trials, key=lambda row: row["validation_selection_score"])
        if task == "foot":
            evaluation = foot.evaluate_selected(
                condition, seed, selected, arrays, features, config, device, output
            )
        else:
            evaluation = cobb.evaluate_selected(
                condition, seed, selected, arrays, features, config, device, output
            )
        results["runs"][str(seed)] = {
            "pipeline_seed": seed,
            "detector_seed": seed,
            "downstream_seed": seed,
            "visual_pairing_seed": seed,
            "all_validation_trials": trials,
            "selected_trial": selected,
            **evaluation,
        }
        foot.write_json(progress_path, results)

    results["test_summary_across_pipeline_seeds"] = (
        module.benchmark.summarize_seeds(results["runs"])
    )
    results["comparison"] = {
        "matched": results["existing_matched"].get(
            "test_summary_across_pipeline_seeds"
        ),
        "image_shuffled": results["test_summary_across_pipeline_seeds"],
    }
    foot.write_json(results_path, results)
    progress_path.unlink()
    foot.write_json(output / "compact_comparison.json", results["comparison"])
    print(json.dumps({
        "status": "complete", "task": task,
        "condition": condition,
        "image_shuffled": results["test_summary_across_pipeline_seeds"],
        "matched": results["comparison"]["matched"],
        "paths": {
            "results": str(results_path),
            "comparison": str(output / "compact_comparison.json"),
        },
    }, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("foot", "cobb"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--shuffle-mode", choices=("token-set", "feature-map"),
        default="feature-map",
    )
    parser.add_argument("--sampling-batch-size", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
