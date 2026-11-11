"""Predicted-geometry construction and patient/case bootstrap for attention audits."""
from __future__ import annotations
import numpy as np
from PIL import Image
import train_hvangle_axis_aligned_mii as foot_local
import train_aasce_cobb_measurement_line_mii as cobb_local

SPLITS = ("train", "val", "test")
ROLES = {"foot": ("toe", "mt1", "mt2"), "cobb": ("upper", "lower")}
BOOTSTRAP_REPLICATES = 10000

def canvas_to_feature(points: np.ndarray) -> np.ndarray:
    """Exact align_corners=False mapping used by localized sampling."""
    return (np.asarray(points, np.float32) + 0.5) * 15.0 / float(
        foot_local.LETTERBOX_SIZE
    ) - 0.5

def map_original_to_feature(points: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    canvas = np.asarray(points, np.float32).copy()
    canvas[..., 0] = canvas[..., 0] * mapping[0] + mapping[2]
    canvas[..., 1] = canvas[..., 1] * mapping[1] + mapping[3]
    return canvas_to_feature(canvas)

def finite_segment_distance_sq(segments: np.ndarray) -> np.ndarray:
    """Squared distance from all 15x15 patch centres to finite segments."""
    y, x = np.meshgrid(np.arange(15), np.arange(15), indexing="ij")
    grid = np.stack((x, y), axis=-1).reshape(1, 1, 225, 2).astype(np.float32)
    a = segments[:, :, 0, None, :]
    b = segments[:, :, 1, None, :]
    vector = b - a
    denominator = np.sum(vector * vector, axis=-1, keepdims=True)
    safe = np.where(denominator > 1e-12, denominator, 1.0)
    t = np.sum((grid - a) * vector, axis=-1, keepdims=True) / safe
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * vector
    return np.sum((grid - closest) ** 2, axis=-1).astype(np.float32)

def geometry_for_seed(task: str, arrays: dict, records: dict) -> dict:
    """Construct exact predicted primitives and their 15x15 distance maps."""
    result = {}
    for split in SPLITS:
        split_arrays = arrays[split]
        if len(records[split]) != len(split_arrays.identifiers):
            raise ValueError(f"{task}/{split}: record count changed")
        segments = []
        for coordinates, record in zip(split_arrays.coordinates, records[split]):
            with Image.open(record.image_path) as image:
                width, height = image.size
                _, mapping = foot_local.letterbox_with_mapping(image.convert("RGB"))
            predicted = np.asarray(coordinates, np.float32) * float(max(width, height))
            if task == "foot":
                if predicted.shape != (6, 2):
                    raise ValueError(f"unexpected predicted Foot L6: {predicted.shape}")
                primitive = predicted.reshape(3, 2, 2)
            else:
                if predicted.shape != (8, 2):
                    raise ValueError(f"unexpected Auto-EV Cobb L8: {predicted.shape}")
                primitive = cobb_local.exact_reference_lines(predicted)
            feature_segments = map_original_to_feature(primitive, mapping)
            segments.append(feature_segments)
        segments = np.asarray(segments, np.float32)
        expected = (len(split_arrays.identifiers), len(ROLES[task]), 2, 2)
        if segments.shape != expected or not np.isfinite(segments).all():
            raise ValueError(f"{task}/{split}: invalid primitive segments {segments.shape}")
        distance_sq = finite_segment_distance_sq(segments)
        lengths = np.linalg.norm(segments[:, :, 1] - segments[:, :, 0], axis=-1)
        if task == "foot":
            descriptors = split_arrays.edges[:, foot_local.AXIS_EDGE_INDICES, :]
        else:
            descriptors = np.stack(
                [split_arrays.edges[:, pair, :].mean(axis=1)
                 for pair in cobb_local.ENDPLATE_EDGE_INDICES], axis=1
            )
        if descriptors.shape != (len(segments), len(ROLES[task]), 4):
            raise ValueError(f"{task}/{split}: primitive descriptor mismatch")
        result[split] = {
            "segments": segments,
            "distance_sq": distance_sq,
            "lengths": lengths.astype(np.float32),
            "descriptors": np.asarray(descriptors, np.float32),
        }
    return result

def paired_cluster_bootstrap(delta, clusters, seed):
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    members = {item: np.flatnonzero(clusters == item) for item in unique}
    rng = np.random.default_rng(seed)
    means, medians = [], []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([members[item] for item in sampled])
        means.append(float(np.mean(delta[indices])))
        medians.append(float(np.median(delta[indices])))
    return {
        "unit": "patient" if len(unique) < len(clusters) else "case",
        "clusters": int(len(unique)), "replicates": BOOTSTRAP_REPLICATES,
        "mean_delta_ci95": [float(np.quantile(means, .025)),
                            float(np.quantile(means, .975))],
        "median_delta_ci95": [float(np.quantile(medians, .025)),
                              float(np.quantile(medians, .975))],
    }
