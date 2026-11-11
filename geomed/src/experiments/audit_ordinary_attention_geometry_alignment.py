#!/usr/bin/env python3
"""Representation-only audit of ordinary GeoMed cross-attention.

The trained network and its inference graph are unchanged.  This script loads
the already selected ordinary cross-attention checkpoints in eval mode and
logs the attention tensor that the existing MHA module already computes.  It
then compares attention allocation with the same HIH-predicted measurement
primitives used to construct the relational geometry input.  No GT landmark,
retraining, gradient, or checkpoint mutation is permitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "geomed" / "src"))
sys.path.insert(0, str(SCRIPT_ROOT))

import audit_geometry as geometry_helpers  # noqa: E402
import train_predicted_landmark_frozen_vision_fusion as fusion  # noqa: E402
import train_predicted_landmark_mii_cross_attention as ordinary  # noqa: E402


TASKS = ("foot", "cobb")
SEEDS = (17, 42, 73)
SPLITS = ("test",)
LINE_SAMPLES = 16
BOOTSTRAP_REPLICATES = 5000
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs" / "benchmarks" /
    "ordinary_cross_attention_geometry_alignment_audit"
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def bilinear_support_mask(segments: np.ndarray) -> np.ndarray:
    """Union of patch tokens touched by established 16-point line sampling.

    This is a representation audit footprint, not a newly chosen line width:
    each continuous sample contributes exactly the at-most-four 15x15 tokens
    used by bilinear interpolation, with border handling matching grid_sample.
    """
    segments = np.asarray(segments, np.float64)
    masks = np.zeros((len(segments), ordinary.PATCH_COUNT), dtype=bool)
    fractions = np.linspace(0.0, 1.0, LINE_SAMPLES)
    for case_index, case_segments in enumerate(segments):
        for segment in case_segments:
            points = (
                segment[0, None] * (1.0 - fractions[:, None])
                + segment[1, None] * fractions[:, None]
            )
            # padding_mode='border' in the established localized sampler.
            points = np.clip(points, 0.0, 14.0)
            for x, y in points:
                xs = {int(math.floor(x)), int(math.ceil(x))}
                ys = {int(math.floor(y)), int(math.ceil(y))}
                for yy in ys:
                    for xx in xs:
                        masks[case_index, yy * 15 + xx] = True
    return masks


def valid_token_masks(records: dict, split: str) -> np.ndarray:
    """Patch-centre validity under the exact existing 224 letterbox."""
    yy, xx = np.meshgrid(np.arange(15), np.arange(15), indexing="ij")
    x_canvas = (xx.reshape(-1) + 0.5) * geometry_helpers.foot_local.LETTERBOX_SIZE / 15.0 - 0.5
    y_canvas = (yy.reshape(-1) + 0.5) * geometry_helpers.foot_local.LETTERBOX_SIZE / 15.0 - 0.5
    masks = []
    for record in records[split]:
        with Image.open(record.image_path) as image:
            width, height = image.size
            _, mapping = geometry_helpers.foot_local.letterbox_with_mapping(image.convert("RGB"))
        contained_width = mapping[0] * width
        contained_height = mapping[1] * height
        masks.append(
            (x_canvas >= mapping[2]) & (x_canvas < mapping[2] + contained_width)
            & (y_canvas >= mapping[3]) & (y_canvas < mapping[3] + contained_height)
        )
    return np.asarray(masks, bool)


def attention_with_existing_modules(model, relational, edges, patches):
    """Exact ordinary forward plus already-computed per-head MHA weights."""
    geometry = model.geometry_encoder(relational, edges)
    normalized = torch.nn.functional.normalize(patches.float(), dim=-1)
    keys_values = model.patch_projection(normalized)
    query = model.query_projection(geometry).unsqueeze(1)
    attended, weights = model.cross_attention(
        query, keys_values, keys_values,
        need_weights=True, average_attn_weights=False,
    )
    prediction = model.head(torch.cat((attended.squeeze(1), geometry), dim=1))
    return prediction, weights.squeeze(2)


def enrichment(weights: np.ndarray, valid: np.ndarray, geometry: np.ndarray):
    """Return main head-average and supplementary per-head R_geo."""
    if weights.ndim != 3 or weights.shape[1:] != (ordinary.ATTENTION_HEADS, 225):
        raise ValueError(f"unexpected attention shape {weights.shape}")
    geometry_valid = geometry & valid
    valid_count = valid.sum(axis=1)
    geometry_count = geometry_valid.sum(axis=1)
    if np.any(valid_count == 0) or np.any(geometry_count == 0):
        raise ValueError("empty valid-token or geometry-token region")

    # Main analysis: average the eight heads first, then renormalize within V.
    average = weights.mean(axis=1)
    average_valid = average * valid
    average_valid /= average_valid.sum(axis=1, keepdims=True)
    mass = (average_valid * geometry_valid).sum(axis=1)
    chance = geometry_count / valid_count
    main_ratio = mass / chance

    # Supplement: independently renormalize each head within V.
    per_head_valid = weights * valid[:, None, :]
    per_head_valid /= per_head_valid.sum(axis=2, keepdims=True)
    per_head_mass = (per_head_valid * geometry_valid[:, None, :]).sum(axis=2)
    per_head_ratio = per_head_mass / chance[:, None]
    return main_ratio, per_head_ratio, mass, chance


def bootstrap(values: np.ndarray, clusters: np.ndarray, seed: int) -> dict:
    unique = np.unique(clusters)
    members = {item: np.flatnonzero(clusters == item) for item in unique}
    rng = np.random.default_rng(seed)
    means, medians = [], []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([members[item] for item in sampled])
        means.append(float(values[indices].mean()))
        medians.append(float(np.median(values[indices])))
    return {
        "unit": "patient" if len(unique) < len(clusters) else "case",
        "replicates": BOOTSTRAP_REPLICATES,
        "mean_ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
        "median_ci95": [float(np.quantile(medians, .025)), float(np.quantile(medians, .975))],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device index")
    return parser.parse_args()


def main():
    args = parse_args()
    device = fusion.resolve_device(args.device)
    ordinary_results = json.loads((ordinary.DEFAULT_OUTPUT / "results.json").read_text())
    output = DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = [], {
        "audit_only": True, "models_trained": 0,
        "network_changed": False, "checkpoints_changed": False,
        "model": "existing ordinary cross-attention",
        "main_attention": "unselected 8-head average",
        "geometry_region": (
            "union of tokens in the exact bilinear support of 16 samples on each "
            "HIH-predicted measurement primitive"
        ),
        "padding_policy": (
            "tokens whose centres fall in letterbox padding are removed; attention "
            "is renormalized over valid tokens V"
        ),
        "R_geo": "attention mass in G divided by |G|/|V|",
        "tasks": {},
    }
    global_max_saved_difference = 0.0
    global_max_direct_difference = 0.0
    global_max_logged_difference = 0.0
    for task in TASKS:
        adapter = fusion.load_adapter(
            task, fusion.DEFAULT_FOOT_CONFIG, fusion.DEFAULT_COBB_CONFIG
        )
        records = fusion.records_for_task(task)
        valid = valid_token_masks(records, "test")
        task_summary = {"seeds": {}}
        summary["tasks"][task] = task_summary
        for seed in SEEDS:
            arrays, audit = adapter.benchmark.load_data(REPO_ROOT, adapter.config, seed)
            geometry = geometry_helpers.geometry_for_seed(task, arrays, records)["test"]
            geometry_mask = bilinear_support_mask(geometry["segments"]) & valid
            run = ordinary_results["tasks"][task]["conditions"]["matched"]["runs"][str(seed)]
            checkpoint = torch.load(
                run["selected_trial"]["checkpoint"], map_location=device,
                weights_only=False,
            )
            model = ordinary.MIICrossAttentionGAE(
                arrays["train"].relational.shape[1], arrays["train"].edges.shape[1],
                len(adapter.targets),
            ).to(device)
            model.load_state_dict(checkpoint["model"]); model.eval()
            dataset = ordinary.PatchFusionDataset(
                arrays["test"], ordinary.patch_path(ordinary.DEFAULT_OUTPUT, task, "test"),
                np.arange(len(arrays["test"].identifiers)),
            )
            loader = DataLoader(dataset, batch_size=int(adapter.config["batch_size"]),
                                shuffle=False, num_workers=0,
                                pin_memory=device.type == "cuda")
            predictions, direct_predictions, logged_predictions = [], [], []
            weights_all, indices = [], []
            transform = adapter.transform(arrays)
            target_mean, target_sd = ordinary.transform_arrays(transform)
            with torch.inference_mode():
                for relational, edges, patches, _, index in loader:
                    relational = relational.to(device, non_blocking=True)
                    edges = edges.to(device, non_blocking=True)
                    patches = patches.to(device, non_blocking=True)
                    audit_prediction, weights = attention_with_existing_modules(
                        model, relational, edges, patches
                    )
                    direct_prediction = model(relational, edges, patches)
                    logged_prediction, _ = model(
                        relational, edges, patches, return_attention=True
                    )
                    predictions.append(audit_prediction.cpu().numpy() * target_sd + target_mean)
                    direct_predictions.append(direct_prediction.cpu().numpy() * target_sd + target_mean)
                    logged_predictions.append(logged_prediction.cpu().numpy() * target_sd + target_mean)
                    weights_all.append(weights.float().cpu().numpy())
                    indices.append(index.numpy())
            order = np.concatenate(indices).argsort()
            prediction = np.concatenate(predictions)[order]
            direct = np.concatenate(direct_predictions)[order]
            logged = np.concatenate(logged_predictions)[order]
            weights = np.concatenate(weights_all)[order]
            saved_path = Path(run["test"]["prediction_file"])
            with np.load(saved_path) as saved:
                if not np.array_equal(saved["identifiers"].astype(str), arrays["test"].identifiers):
                    raise ValueError("saved prediction identifiers changed")
                saved_prediction = np.asarray(saved["prediction"], np.float64)
            saved_difference = float(np.max(np.abs(prediction - saved_prediction)))
            direct_difference = float(np.max(np.abs(prediction - direct)))
            logged_difference = float(np.max(np.abs(prediction - logged)))
            global_max_saved_difference = max(global_max_saved_difference, saved_difference)
            global_max_direct_difference = max(global_max_direct_difference, direct_difference)
            global_max_logged_difference = max(global_max_logged_difference, logged_difference)
            if (saved_difference > 5e-5 or direct_difference > 5e-5
                    or logged_difference > 2e-6):
                raise RuntimeError(
                    f"prediction invariance failed {task}/{seed}: "
                    f"saved={saved_difference}, direct={direct_difference}, "
                    f"logged={logged_difference}"
                )
            ratio, per_head, mass, chance = enrichment(weights, valid, geometry_mask)
            clusters = (
                arrays["test"].patient_ids if task == "foot"
                else np.arange(len(arrays["test"].identifiers))
            )
            boot = bootstrap(ratio, clusters, 1_000_003 * seed + (0 if task == "foot" else 71))
            head_pair_correlations = []
            for left in range(ordinary.ATTENTION_HEADS):
                for right in range(left + 1, ordinary.ATTENTION_HEADS):
                    head_pair_correlations.append(
                        float(spearmanr(per_head[:, left], per_head[:, right]).statistic)
                    )
            seed_summary = {
                "n": len(ratio), "checkpoint": run["selected_trial"]["checkpoint"],
                "prediction_file": str(saved_path),
                "maximum_saved_prediction_difference_deg": saved_difference,
                "maximum_direct_forward_difference_deg": direct_difference,
                "maximum_existing_logged_forward_difference_deg": logged_difference,
                "R_geo_mean": float(ratio.mean()),
                "R_geo_median": float(np.median(ratio)),
                "R_geo_q1": float(np.quantile(ratio, .25)),
                "R_geo_q3": float(np.quantile(ratio, .75)),
                "fraction_R_geo_above_1": float(np.mean(ratio > 1.0)),
                "bootstrap": boot,
                "valid_token_count_median": float(np.median(valid.sum(axis=1))),
                "geometry_token_count_median": float(np.median(geometry_mask.sum(axis=1))),
                "per_head_R_geo_mean": per_head.mean(axis=0).tolist(),
                "per_head_R_geo_median": np.median(per_head, axis=0).tolist(),
                "head_pair_spearman_median": float(np.median(head_pair_correlations)),
                "head_pair_spearman_iqr": [
                    float(np.quantile(head_pair_correlations, .25)),
                    float(np.quantile(head_pair_correlations, .75)),
                ],
            }
            task_summary["seeds"][str(seed)] = seed_summary
            npz_path = output / "attention" / task / f"seed_{seed}_test_attention.npz"
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_path, identifiers=arrays["test"].identifiers,
                attention_per_head=weights.astype(np.float32),
                valid_token_mask=valid, geometry_token_mask=geometry_mask,
                R_geo=ratio.astype(np.float32), R_geo_per_head=per_head.astype(np.float32),
                geometry_attention_mass=mass.astype(np.float32),
                uniform_geometry_fraction=chance.astype(np.float32),
                prediction=prediction.astype(np.float32),
                predicted_geometry_source=np.asarray(audit["paths"]["test"]),
            )
            seed_summary["attention_file"] = str(npz_path)
            for index, identifier in enumerate(arrays["test"].identifiers):
                row = {
                    "task": task, "seed": seed, "identifier": identifier,
                    "patient_id": str(clusters[index]) if task == "foot" else "",
                    "valid_token_count": int(valid[index].sum()),
                    "geometry_token_count": int(geometry_mask[index].sum()),
                    "uniform_geometry_fraction": float(chance[index]),
                    "geometry_attention_mass": float(mass[index]),
                    "R_geo": float(ratio[index]),
                }
                for head in range(ordinary.ATTENTION_HEADS):
                    row[f"R_geo_head_{head}"] = float(per_head[index, head])
                rows.append(row)
            print(json.dumps({"task": task, "seed": seed,
                              "R_geo_mean": seed_summary["R_geo_mean"],
                              "R_geo_median": seed_summary["R_geo_median"],
                              "saved_prediction_difference": saved_difference}), flush=True)
    summary["prediction_invariance"] = {
        "passed": True, "tolerance_saved_deg": 5e-5,
        "tolerance_direct_deg": 5e-5,
        "tolerance_existing_logged_forward_deg": 2e-6,
        "maximum_saved_prediction_difference_deg": global_max_saved_difference,
        "maximum_direct_forward_difference_deg": global_max_direct_difference,
        "maximum_existing_logged_forward_difference_deg": global_max_logged_difference,
        "statement": "Attention logging did not alter model inference.",
    }
    summary["no_gt_landmarks_or_geometry"] = True
    summary["predicted_geometry_only"] = {
        "foot": "HIH predicted L6 -> Toe/MT1/MT2 union",
        "cobb": "HIH predicted L68 -> deterministic Auto-EV L8 -> Upper/Lower union",
    }
    csv_path = output / "geometry_attention_alignment_per_case.csv"
    summary_path = output / "geometry_attention_alignment_summary.json"
    write_csv(csv_path, rows); write_json(summary_path, summary)
    print(json.dumps({"status": "PASS", "models_trained": 0,
                      "summary": str(summary_path), "per_case": str(csv_path),
                      "attention_directory": str(output / "attention")}, indent=2))


if __name__ == "__main__":
    main()
