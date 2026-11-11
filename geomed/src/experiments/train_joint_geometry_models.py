#!/usr/bin/env python3
"""Train Cobb regressors from fixed AASCE-HIH-68 landmark predictions.

The three input archives are treated as immutable:

* train uses five-fold image-OOF AASCE-HIH-68 predictions;
* validation selects epochs and hyperparameters;
* test is evaluated once after validation selection.

Each predicted L68 is reduced by the deterministic Auto-EV selector to the
measurement-defining L8. All models share a 64-dimensional representation and
the same one-output Cobb regression head. Patient IDs are unavailable, so the
frozen 336/72/73 split is image-level and must not be called patient-disjoint.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("aasce_geometry_coordstem_standardized_config.json")
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "benchmarks" / "joint_geometry_aasce_hih_oof_coordstem_standardized"
sys.path.insert(0, str(REPO_ROOT / "geomed" / "src"))

from geomed.data import load_aasce_records  # noqa: E402
from geomed.models import GeometryAttentionEncoder  # noqa: E402
from cobb_geometry import select_primary_cobb  # noqa: E402


TARGETS = ("Cobb",)
POINT_COUNT = 8
MODEL_NAMES = (
    "coordmlp", "deepsets", "deepsets_role_coordstem",
    "gnn", "gnn_role_coordstem", "gae",
    "gae_distance_only", "gae_distance_orientation", "gae_distance_triplets",
    "gae_angle_only",
)

GAE_FEATURE_ABLATIONS = {
    "gae_distance_only": {
        "distance": True, "orientation": False, "triplets": False,
        "label": "Distance only",
    },
    "gae_distance_orientation": {
        "distance": True, "orientation": True, "triplets": False,
        "label": "Distance + orientation",
    },
    "gae_distance_triplets": {
        "distance": True, "orientation": False, "triplets": True,
        "label": "Distance + triplets",
    },
    "gae_angle_only": {
        "distance": False, "orientation": False, "triplets": True,
        "label": "Angle only",
    },
    "gae": {
        "distance": True, "orientation": True, "triplets": True,
        "label": "Full relational features",
    },
}


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def icc_2_1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Two-way random-effects, absolute-agreement, single-measure ICC."""
    values = np.column_stack([np.asarray(y_true, float), np.asarray(y_pred, float)])
    n, k = values.shape
    if n < 2:
        return float("nan")
    row_mean = values.mean(axis=1)
    column_mean = values.mean(axis=0)
    grand_mean = values.mean()
    ms_rows = k * np.square(row_mean - grand_mean).sum() / (n - 1)
    ms_columns = n * np.square(column_mean - grand_mean).sum() / (k - 1)
    residual = np.square(
        values - row_mean[:, None] - column_mean[None, :] + grand_mean
    ).sum()
    ms_error = residual / ((n - 1) * (k - 1))
    denominator = (
        ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    )
    return float((ms_rows - ms_error) / denominator) if denominator else float("nan")


def measurement_metrics(truth: np.ndarray, prediction: np.ndarray,
                        tolerance: float) -> dict:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    difference = prediction - truth
    absolute = np.abs(difference)
    bias = float(difference.mean())
    standard_deviation = float(difference.std(ddof=1))
    return {
        "n": int(len(truth)),
        "mae": float(absolute.mean()),
        "mae_sd": float(absolute.std(ddof=1)),
        "icc_2_1": icc_2_1(truth, prediction),
        "within_tolerance_percent": float(100.0 * np.mean(absolute <= tolerance)),
        "tolerance_deg": float(tolerance),
        "bias": bias,
        "loa_lower": bias - 1.96 * standard_deviation,
        "loa_upper": bias + 1.96 * standard_deviation,
    }


def metric_block(truth: np.ndarray, prediction: np.ndarray,
                 tolerances: dict[str, float]) -> dict:
    return {
        name: measurement_metrics(truth[:, index], prediction[:, index], tolerances[name])
        for index, name in enumerate(TARGETS)
    }


@dataclass(frozen=True)
class GeometryArrays:
    identifiers: np.ndarray
    coordinates: np.ndarray
    relational: np.ndarray
    edges: np.ndarray
    truth: np.ndarray
    analytic: np.ndarray
    folds: np.ndarray | None
    selected_pairs: np.ndarray
    selector_fallback: np.ndarray
    degenerate_vertebra_count: np.ndarray


class GeometryDataset(Dataset):
    def __init__(self, arrays: GeometryArrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays.identifiers)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.arrays.coordinates[index]),
            torch.from_numpy(self.arrays.relational[index]),
            torch.from_numpy(self.arrays.edges[index]),
            torch.from_numpy(self.arrays.truth[index]),
            int(index),
        )


def predicted_relational_features(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Paper GAE features with a defined encoding for coincident predictions.

    GT endpoints are non-degenerate, but two independently decoded HIH peaks can
    coincide.  Such edges retain zero distance and receive zero orientation;
    triplets containing a zero vector receive the neutral ``(0, 0)`` encoding.
    Every non-degenerate value is identical to ``geomed.features.relational_features``.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (POINT_COUNT, 2) or not np.isfinite(points).all():
        raise ValueError(f"predicted AASCE selected points must have shape ({POINT_COUNT},2)")
    edge_index = list(combinations(range(POINT_COUNT), 2))
    distances = np.asarray([
        np.linalg.norm(points[right] - points[left]) for left, right in edge_index
    ], dtype=np.float32)
    positive = distances[distances > 1e-6]
    maximum = float(distances.max()) if len(positive) else 1.0
    median = float(np.median(distances))
    if median <= 1e-6:
        median = float(np.median(positive)) if len(positive) else 1.0

    edge_rows = []
    adjacency = {index: [] for index in range(POINT_COUNT)}
    for distance, (left, right) in zip(distances, edge_index):
        adjacency[left].append(right)
        adjacency[right].append(left)
        if distance <= 1e-6:
            edge_rows.append([0.0, 0.0, 0.0, 0.0])
            continue
        dx, dy = points[right] - points[left]
        theta = math.atan2(float(dy), float(dx))
        edge_rows.append([
            float(distance / maximum),
            math.log(float(distance / median) + 1e-6),
            math.cos(theta),
            math.sin(theta),
        ])

    triplets = []
    for center in range(POINT_COUNT):
        for left, right in combinations(sorted(adjacency[center]), 2):
            first = points[left] - points[center]
            second = points[right] - points[center]
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denominator <= 1e-12:
                triplets.append([0.0, 0.0])
                continue
            cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
            triplets.append([cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine))])
    edges = np.asarray(edge_rows, dtype=np.float32)
    relational = np.concatenate([edges.reshape(-1), np.asarray(triplets).reshape(-1)])
    return relational.astype(np.float32), edges


class JointRegressionHead(nn.Sequential):
    """The same MLP head is used after every 64-d model representation."""

    def __init__(self, representation_dim: int = 64):
        super().__init__(
            nn.LayerNorm(representation_dim),
            nn.Linear(representation_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, len(TARGETS)),
        )


class CoordMLP(nn.Module):
    """MLP over eight size-normalized selected corner coordinates."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(2 * POINT_COUNT),
            nn.Linear(2 * POINT_COUNT, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.GELU(),
        )
        self.head = JointRegressionHead()

    def forward(self, coordinates, relational, edges):
        del relational, edges
        return self.head(self.encoder(coordinates.flatten(1)))


class DeepSets(nn.Module):
    """Shared point encoder plus symmetric pooling over labeled landmarks."""

    def __init__(self, number_of_points: int = POINT_COUNT):
        super().__init__()
        self.register_buffer("point_ids", torch.arange(number_of_points), persistent=False)
        self.identity = nn.Embedding(number_of_points, 16)
        self.point_encoder = nn.Sequential(
            nn.Linear(18, 128), nn.GELU(),
            nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(),
        )
        self.head = JointRegressionHead()

    def forward(self, coordinates, relational, edges):
        del relational, edges
        identity = self.identity(self.point_ids).unsqueeze(0).expand(coordinates.shape[0], -1, -1)
        point_tokens = self.point_encoder(torch.cat([coordinates, identity], dim=-1))
        return self.head(point_tokens.mean(dim=1))


def normalize_landmark_cloud(coordinates: torch.Tensor) -> torch.Tensor:
    """Remove translation and isotropic scale without changing Cobb angles."""
    centered = coordinates - coordinates.mean(dim=1, keepdim=True)
    rms_radius = torch.sqrt(
        centered.square().sum(dim=-1).mean(dim=1, keepdim=True)
    ).unsqueeze(-1)
    return centered / rms_radius.clamp_min(1e-6)


class CoordinateIdentityStem(nn.Module):
    """Encode corner coordinates and fixed anatomical roles separately."""

    def __init__(self, number_of_points: int = POINT_COUNT, hidden_dim: int = 64):
        super().__init__()
        self.register_buffer(
            "point_ids", torch.arange(number_of_points), persistent=False
        )
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
        )
        self.identity = nn.Embedding(number_of_points, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = normalize_landmark_cloud(coordinates)
        coordinate_features = self.coordinate_encoder(normalized)
        identity_features = self.identity(self.point_ids).unsqueeze(0).expand(
            coordinates.shape[0], -1, -1
        )
        return self.fusion(
            torch.cat([coordinate_features, identity_features], dim=-1)
        ), normalized


class DeepSetsRoleConcatCoordinateStem(nn.Module):
    """Role-preserving DeepSets over the eight Auto-EV-selected corners."""

    def __init__(self, number_of_points: int = POINT_COUNT):
        super().__init__()
        self.stem = CoordinateIdentityStem(number_of_points)
        self.point_encoder = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(number_of_points * 64),
            nn.Linear(number_of_points * 64, 64), nn.GELU(),
        )
        self.head = JointRegressionHead()

    def forward(self, coordinates, relational, edges):
        del relational, edges
        tokens, _ = self.stem(coordinates)
        return self.head(self.readout(self.point_encoder(tokens).flatten(1)))


class MessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim + 3, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        batch, number_of_points, hidden_dim = hidden.shape
        receiver = hidden[:, :, None, :].expand(-1, -1, number_of_points, -1)
        sender = hidden[:, None, :, :].expand(-1, number_of_points, -1, -1)
        delta = coordinates[:, None, :, :] - coordinates[:, :, None, :]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        messages = self.message(torch.cat([receiver, sender, delta, distance], dim=-1))
        mask = (~torch.eye(number_of_points, dtype=torch.bool, device=hidden.device)).view(
            1, number_of_points, number_of_points, 1
        )
        aggregated = (messages * mask).sum(dim=2) / max(1, number_of_points - 1)
        update = self.update(torch.cat([hidden, aggregated], dim=-1))
        return torch.nn.functional.gelu(self.normalization(hidden + update))


class FullyConnectedGNN(nn.Module):
    """Two-layer message-passing GNN on the complete eight-corner graph."""

    def __init__(self, number_of_points: int = POINT_COUNT):
        super().__init__()
        self.register_buffer("point_ids", torch.arange(number_of_points), persistent=False)
        self.identity = nn.Embedding(number_of_points, 16)
        self.input_projection = nn.Sequential(
            nn.Linear(18, 64), nn.GELU(), nn.LayerNorm(64)
        )
        self.layers = nn.ModuleList([MessagePassingLayer(), MessagePassingLayer()])
        self.head = JointRegressionHead()

    def forward(self, coordinates, relational, edges):
        del relational, edges
        identity = self.identity(self.point_ids).unsqueeze(0).expand(coordinates.shape[0], -1, -1)
        hidden = self.input_projection(torch.cat([coordinates, identity], dim=-1))
        for layer in self.layers:
            hidden = layer(hidden, coordinates)
        return self.head(hidden.mean(dim=1))


class RolePreservingGNNCoordinateStem(nn.Module):
    """Residual complete-graph GNN with the same coordinate/role stem."""

    def __init__(self, number_of_points: int = POINT_COUNT):
        super().__init__()
        self.stem = CoordinateIdentityStem(number_of_points)
        self.layers = nn.ModuleList([MessagePassingLayer(), MessagePassingLayer()])
        self.readout = nn.Sequential(
            nn.LayerNorm(number_of_points * 64),
            nn.Linear(number_of_points * 64, 64), nn.GELU(),
        )
        self.head = JointRegressionHead()

    def forward(self, coordinates, relational, edges):
        del relational, edges
        hidden, normalized_coordinates = self.stem(coordinates)
        for layer in self.layers:
            hidden = layer(hidden, normalized_coordinates)
        return self.head(self.readout(hidden.flatten(1)))


class JointGAE(nn.Module):
    """The paper's edge-attention/triplet GAE with a single Cobb head."""

    def __init__(self, relational_dim: int, number_of_edges: int):
        super().__init__()
        self.encoder = GeometryAttentionEncoder(
            relational_dim, number_of_edges, output_dim=64,
        )
        self.head = JointRegressionHead()

    def forward(self, coordinates, relational, edges):
        del coordinates
        return self.head(self.encoder(relational, edges))


class JointGAEFeatureAblation(nn.Module):
    """Original Cobb GAE with relational components zero-masked.

    All variants retain the full edge-attention branch, triplet branch, output
    projection, and regression head.  Consequently their parameter count is
    identical to full GAE and only available geometric evidence changes.
    """

    def __init__(self, relational_dim: int, number_of_edges: int,
                 *, use_distance: bool, use_orientation: bool,
                 use_triplets: bool):
        super().__init__()
        self.number_of_edges = int(number_of_edges)
        self.edge_dim = self.number_of_edges * 4
        edge_mask = [
            float(use_distance), float(use_distance),
            float(use_orientation), float(use_orientation),
        ]
        self.register_buffer(
            "edge_feature_mask",
            torch.tensor(edge_mask, dtype=torch.float32).view(1, 1, 4),
            persistent=True,
        )
        self.register_buffer(
            "triplet_feature_mask",
            torch.tensor(float(use_triplets), dtype=torch.float32),
            persistent=True,
        )
        self.encoder = GeometryAttentionEncoder(
            relational_dim, number_of_edges, output_dim=64
        )
        self.head = JointRegressionHead()

    def forward(self, coordinates, relational, edges):
        del coordinates
        masked_edges = edges * self.edge_feature_mask
        masked_triplets = relational[:, self.edge_dim:] * self.triplet_feature_mask
        masked_relational = torch.cat(
            [masked_edges.flatten(1), masked_triplets], dim=1
        )
        return self.head(self.encoder(masked_relational, masked_edges))


def build_model(name: str, relational_dim: int, number_of_edges: int) -> nn.Module:
    if name == "coordmlp":
        return CoordMLP()
    if name == "deepsets":
        return DeepSets()
    if name == "deepsets_role_coordstem":
        return DeepSetsRoleConcatCoordinateStem()
    if name == "gnn":
        return FullyConnectedGNN()
    if name == "gnn_role_coordstem":
        return RolePreservingGNNCoordinateStem()
    if name == "gae":
        return JointGAE(relational_dim, number_of_edges)
    if name in GAE_FEATURE_ABLATIONS and name != "gae":
        specification = GAE_FEATURE_ABLATIONS[name]
        return JointGAEFeatureAblation(
            relational_dim,
            number_of_edges,
            use_distance=bool(specification["distance"]),
            use_orientation=bool(specification["orientation"]),
            use_triplets=bool(specification["triplets"]),
        )
    raise ValueError(f"unknown model: {name}")


def load_split(path: Path, records, split: str, expected_size: int) -> GeometryArrays:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as archive:
        required = {"identifiers", "predicted_points", "truth", "analytic"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        identifiers = archive["identifiers"].astype(str)
        predicted_points = np.asarray(archive["predicted_points"], dtype=np.float32)
        truth = np.asarray(archive["truth"], dtype=np.float32)
        analytic = np.asarray(archive["analytic"], dtype=np.float32)
        folds = np.asarray(archive["fold"], dtype=np.int64) if "fold" in archive.files else None
        archived_selected = (
            np.asarray(archive["selected_points"], dtype=np.float32)
            if "selected_points" in archive.files else None
        )
        archived_pairs = (
            np.asarray(archive["selected_pairs"], dtype=np.int64)
            if "selected_pairs" in archive.files else None
        )

    expected_identifiers = np.asarray([record.identifier for record in records], dtype=str)
    expected_truth = np.asarray([record.targets for record in records], dtype=np.float32)
    if truth.ndim == 1:
        truth = truth[:, None]
    if analytic.ndim == 1:
        analytic = analytic[:, None]
    if len(identifiers) != expected_size or len(records) != expected_size:
        raise ValueError(f"unexpected {split} size: archive={len(identifiers)}, records={len(records)}")
    if not np.array_equal(identifiers, expected_identifiers):
        raise ValueError(f"{split} identifiers do not match the frozen AASCE image split and order")
    if predicted_points.shape != (expected_size, 68, 2):
        raise ValueError(f"unexpected {split} L68 shape: {predicted_points.shape}")
    if truth.shape != (expected_size, 1) or analytic.shape != (expected_size, 1):
        raise ValueError(f"unexpected {split} point/target shapes")
    if not np.allclose(truth, expected_truth, rtol=0.0, atol=1e-5):
        raise ValueError(f"{split} targets differ from AASCE angles.csv column 0")
    if not all(np.isfinite(item).all() for item in (predicted_points, truth, analytic)):
        raise ValueError(f"{split} contains non-finite values")
    if split == "train" and folds is None:
        raise ValueError("training archive must contain an image-OOF fold for every row")
    if split != "train" and folds is not None:
        raise ValueError(f"{split} archive must not masquerade as an OOF training archive")
    if folds is not None and folds.shape != (expected_size,):
        raise ValueError(f"unexpected {split} fold shape: {folds.shape}")

    coordinates, relational_rows, edge_rows = [], [], []
    selected_rows, selected_pairs, fallback_rows, degenerate_counts = [], [], [], []
    recomputed_analytic = []
    for points, record in zip(predicted_points, records):
        # Isotropic normalization preserves directions/angles and removes image scale.
        with Image.open(record.image_path) as image:
            width, height = image.size
        angle, upper, lower, selected, diagnostics = select_primary_cobb(
            points, (width, height), return_diagnostics=True
        )
        coordinates.append(selected / float(max(width, height)))
        relational, edges = predicted_relational_features(selected)
        relational_rows.append(relational)
        edge_rows.append(edges)
        selected_rows.append(selected)
        selected_pairs.append((upper, lower))
        fallback_rows.append(bool(diagnostics["fallback"]))
        degenerate_counts.append(int(diagnostics["degenerate_vertebra_count"]))
        recomputed_analytic.append(angle)

    selected_rows = np.asarray(selected_rows, dtype=np.float32)
    selected_pairs = np.asarray(selected_pairs, dtype=np.int64)
    recomputed_analytic = np.asarray(recomputed_analytic, dtype=np.float32)[:, None]
    if not np.allclose(analytic, recomputed_analytic, rtol=0.0, atol=1e-4):
        difference = float(np.max(np.abs(analytic - recomputed_analytic)))
        raise ValueError(f"{split} stored analytic Cobb disagrees with Auto-EV: {difference:.6g}°")
    if archived_selected is not None and not np.allclose(
        archived_selected, selected_rows, rtol=0.0, atol=1e-4
    ):
        raise ValueError(f"{split} stored selected L8 disagrees with deterministic Auto-EV")
    if archived_pairs is not None and not np.array_equal(archived_pairs, selected_pairs):
        raise ValueError(f"{split} stored end-vertebra pairs disagree with deterministic Auto-EV")

    return GeometryArrays(
        identifiers=identifiers,
        coordinates=np.asarray(coordinates, dtype=np.float32),
        relational=np.asarray(relational_rows, dtype=np.float32),
        edges=np.asarray(edge_rows, dtype=np.float32),
        truth=truth,
        analytic=recomputed_analytic,
        folds=folds,
        selected_pairs=selected_pairs,
        selector_fallback=np.asarray(fallback_rows, dtype=bool),
        degenerate_vertebra_count=np.asarray(degenerate_counts, dtype=np.int64),
    )


def load_data(repo_root: Path, config: dict, detector_seed: int):
    dataset_config = config["dataset"]
    if dataset_config["target_mode"] != "single_primary_cobb":
        raise ValueError("configuration does not request single primary Cobb prediction")
    if tuple(dataset_config["target_order"]) != TARGETS:
        raise ValueError("target order must be Cobb")
    outer = dataset_config["outer_split"]
    if outer["mode"] != "fixed_image_level" or int(outer["split_seed"]) != 42:
        raise ValueError("AASCE geometry requires the frozen image-level split seed 42")
    if bool(outer.get("patient_ids_available", True)):
        raise ValueError("AASCE does not provide patient identifiers")
    records = load_aasce_records(repo_root)
    sizes = dataset_config["outer_split"]["sizes"]
    arrays = {}
    paths = {}
    for split in ("train", "val", "test"):
        relative = dataset_config["landmark_inputs"][split].format(
            detector_seed=detector_seed
        )
        paths[split] = repo_root / relative
        arrays[split] = load_split(paths[split], records[split], split, int(sizes[split]))

    if sorted(set(arrays["train"].folds.tolist())) != [0, 1, 2, 3, 4]:
        raise ValueError("training archive does not contain exactly five OOF folds")
    all_identifiers = [
        identifier for split in ("train", "val", "test")
        for identifier in arrays[split].identifiers.tolist()
    ]
    if len(all_identifiers) != len(set(all_identifiers)):
        raise ValueError("AASCE image identifiers overlap across train/val/test")

    audit = {
        "paths": {key: str(value) for key, value in paths.items()},
        "sizes": {key: len(value.identifiers) for key, value in arrays.items()},
        "split_mode": "fixed image-level (seed 42)",
        "patient_ids_available": False,
        "patient_disjoint": "not established",
        "train_prediction_policy": dataset_config["train_prediction_policy"],
        "val_test_prediction_policy": dataset_config["val_test_prediction_policy"],
        "geometry_policy": dataset_config["geometry_policy"],
        "train_fold_sizes": {
            str(fold): int(np.sum(arrays["train"].folds == fold)) for fold in range(5)
        },
        "detector_seed": detector_seed,
        "selector_diagnostics": {
            split: {
                "fallback_count": int(value.selector_fallback.sum()),
                "samples_with_degenerate_vertebrae": int(
                    np.sum(value.degenerate_vertebra_count > 0)
                ),
            }
            for split, value in arrays.items()
        },
        "analytic_mae": {
            split: {
                target: float(np.abs(value.analytic[:, index] - value.truth[:, index]).mean())
                for index, target in enumerate(TARGETS)
            }
            for split, value in arrays.items()
        },
    }
    return arrays, audit


def make_loaders(arrays: dict[str, GeometryArrays], batch_size: int,
                 workers: int, seed: int, device: torch.device):
    generator = torch.Generator().manual_seed(seed)
    return {
        split: DataLoader(
            GeometryDataset(values),
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=device.type == "cuda",
            generator=generator if split == "train" else None,
        )
        for split, values in arrays.items()
    }


def target_transform(arrays: dict[str, GeometryArrays], config: dict) -> dict:
    """Fit the scalar Cobb transform strictly on training-OOF targets."""
    specification = config.get("target_standardization", {"enabled": False})
    enabled = bool(specification.get("enabled", False))
    ddof = int(specification.get("ddof", 1))
    if ddof not in (0, 1):
        raise ValueError("target standardization ddof must be 0 or 1")
    if enabled:
        mean = arrays["train"].truth.mean(axis=0, dtype=np.float64)
        standard_deviation = arrays["train"].truth.std(
            axis=0, ddof=ddof, dtype=np.float64
        )
        if not np.isfinite(standard_deviation).all() or np.any(standard_deviation <= 0):
            raise ValueError("training Cobb targets have an invalid standard deviation")
    else:
        mean = np.zeros(len(TARGETS), dtype=np.float64)
        standard_deviation = np.ones(len(TARGETS), dtype=np.float64)
    return {
        "enabled": enabled,
        "source": "training OOF targets only" if enabled else "disabled",
        "ddof": ddof,
        "target_order": list(TARGETS),
        "mean_deg": mean.astype(float).tolist(),
        "sd_deg": standard_deviation.astype(float).tolist(),
    }


def _transform_arrays(transform: dict) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(transform["mean_deg"], dtype=np.float32),
        np.asarray(transform["sd_deg"], dtype=np.float32),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             transform: dict | None = None):
    model.eval()
    predictions, truths, indices = [], [], []
    if transform is None:
        mean = np.zeros(len(TARGETS), dtype=np.float32)
        standard_deviation = np.ones(len(TARGETS), dtype=np.float32)
    else:
        mean, standard_deviation = _transform_arrays(transform)
    with torch.inference_mode():
        for coordinates, relational, edges, target, index in loader:
            output = model(
                coordinates.to(device, non_blocking=True),
                relational.to(device, non_blocking=True),
                edges.to(device, non_blocking=True),
            )
            predictions.append(
                output.float().cpu().numpy() * standard_deviation + mean
            )
            truths.append(target.numpy())
            indices.append(index.numpy())
    order = np.concatenate(indices).argsort()
    return np.concatenate(truths)[order], np.concatenate(predictions)[order]


def train_trial(model_name: str, seed: int, hyperparameters: dict, trial_index: int,
                arrays: dict[str, GeometryArrays], config: dict, device: torch.device,
                checkpoint: Path):
    seed_everything(seed)
    model = build_model(
        model_name, arrays["train"].relational.shape[1], arrays["train"].edges.shape[1]
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    loaders = make_loaders(
        arrays, int(config["batch_size"]), int(config["num_workers"]), seed, device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(hyperparameters["lr"]),
        weight_decay=float(hyperparameters["weight_decay"]),
    )
    max_epochs = int(config["max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss_function = nn.HuberLoss(
        delta=float(config["loss"]["delta"]), reduction=config["loss"]["reduction"]
    )
    amp_enabled = bool(config["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    tolerances = {
        target: float(config["metrics"]["tolerance_deg"][target]) for target in TARGETS
    }
    transform = target_transform(arrays, config)
    target_mean, target_sd = _transform_arrays(transform)
    target_mean_tensor = torch.from_numpy(target_mean).to(device)
    target_sd_tensor = torch.from_numpy(target_sd).to(device)
    best_score, best_epoch, stale_epochs = float("inf"), 0, 0
    history = []
    started = time.time()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for coordinates, relational, edges, target, _ in loaders["train"]:
            coordinates = coordinates.to(device, non_blocking=True)
            relational = relational.to(device, non_blocking=True)
            edges = edges.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            normalized_target = (target - target_mean_tensor) / target_sd_tensor
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(coordinates, relational, edges)
                loss = loss_function(prediction, normalized_target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
        scheduler.step()

        validation_truth, validation_prediction = evaluate(
            model, loaders["val"], device, transform
        )
        validation_metrics = metric_block(validation_truth, validation_prediction, tolerances)
        selection_score = float(np.mean([
            validation_metrics[target]["mae"] for target in TARGETS
        ]))
        epoch_record = {
            "epoch": epoch,
            "train_loss_standardized": running_loss / max(1, len(loaders["train"])),
            "validation_cobb_mae_deg": validation_metrics["Cobb"]["mae"],
        }
        history.append(epoch_record)
        if selection_score < best_score:
            best_score, best_epoch, stale_epochs = selection_score, epoch, 0
            torch.save({
                "model": model.state_dict(), "model_name": model_name, "seed": seed,
                "trial_index": trial_index, "hyperparameters": hyperparameters,
                "best_epoch": best_epoch, "validation_selection_score": best_score,
                "target_transform": transform,
                "history_through_best_epoch": list(history),
            }, checkpoint)
        else:
            stale_epochs += 1
        print(json.dumps({
            "model": model_name, "seed": seed, "trial": trial_index,
            "epoch": epoch, "train_loss": running_loss / max(1, len(loaders["train"])),
            "val_cobb_mae": validation_metrics["Cobb"]["mae"],
            "val_selection_score": selection_score, "best_epoch": best_epoch,
        }), flush=True)
        if stale_epochs >= int(config["patience"]):
            break

    return {
        "trial_index": trial_index,
        "hyperparameters": hyperparameters,
        "best_epoch": best_epoch,
        "validation_selection_score": best_score,
        "parameter_count": parameter_count,
        "checkpoint": str(checkpoint),
        "target_transform": transform,
        "epoch_history": history,
        "elapsed_seconds": time.time() - started,
    }


def evaluate_selected(model_name: str, seed: int, selected_trial: dict,
                      arrays: dict[str, GeometryArrays], config: dict,
                      device: torch.device, output_root: Path, detector_seed: int):
    checkpoint = torch.load(selected_trial["checkpoint"], map_location=device)
    model = build_model(
        model_name, arrays["train"].relational.shape[1], arrays["train"].edges.shape[1]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    transform = target_transform(arrays, config)
    if checkpoint.get("target_transform", transform) != transform:
        raise ValueError("checkpoint target transform differs from training-OOF statistics")
    loaders = make_loaders(
        arrays, int(config["batch_size"]), int(config["num_workers"]), seed, device
    )
    tolerances = {
        target: float(config["metrics"]["tolerance_deg"][target]) for target in TARGETS
    }
    result = {}
    for split in ("val", "test"):
        truth, prediction = evaluate(model, loaders[split], device, transform)
        result[split] = metric_block(truth, prediction, tolerances)
        result[split]["prediction_diagnostics"] = {
            "Cobb": {
                "truth_mean_deg": float(truth[:, 0].mean()),
                "truth_sd_deg": float(truth[:, 0].std(ddof=1)),
                "prediction_mean_deg": float(prediction[:, 0].mean()),
                "prediction_sd_deg": float(prediction[:, 0].std(ddof=1)),
            }
        }
        prediction_path = (
            output_root / model_name /
            f"detector_{detector_seed}_seed_{seed}_{split}_predictions.npz"
        )
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_path,
            identifiers=arrays[split].identifiers,
            selected_pairs=arrays[split].selected_pairs,
            detector_seed=np.asarray(detector_seed),
            downstream_seed=np.asarray(seed),
            truth=truth,
            prediction=prediction,
            absolute_error=np.abs(prediction - truth),
        )
        result[split]["prediction_file"] = str(prediction_path)
    return result


def summarize_seeds(runs: dict, split: str = "test") -> dict:
    summary = {}
    for target in TARGETS:
        summary[target] = {}
        for metric in (
            "mae", "icc_2_1", "within_tolerance_percent", "bias", "loa_lower", "loa_upper"
        ):
            values = np.asarray([
                run[split][target][metric] for run in runs.values()
            ], dtype=np.float64)
            summary[target][metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "per_seed": values.tolist(),
            }
    return summary


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--detector-seed", type=int,
        help=("fixed-detector diagnostic mode; omit for the default paired "
              "detector/downstream seeds 17/17, 42/42, 73/73"),
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES,
        help="defaults to the model list declared in the selected config",
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", default="auto",
                        help="PyTorch device, e.g. cuda:0, cuda, cpu, or auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    model_names = args.models or list(config.get("models", MODEL_NAMES))
    unknown_models = sorted(set(model_names) - set(MODEL_NAMES))
    if unknown_models:
        raise ValueError(f"configuration requests unknown models: {unknown_models}")
    seeds = args.seeds if args.seeds is not None else [int(item) for item in config["seeds"]]
    if args.detector_seed is None:
        seed_pairs = [(seed, seed) for seed in seeds]
        seed_mode = "paired detector/downstream seeds"
    else:
        seed_pairs = [(int(args.detector_seed), seed) for seed in seeds]
        seed_mode = "fixed detector diagnostic; SD is conditional on one detector"
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device.index} was requested, but only "
                f"{torch.cuda.device_count()} device(s) are visible"
            )

    arrays_by_detector, data_audits = {}, {}
    for detector_seed, _ in seed_pairs:
        if detector_seed not in arrays_by_detector:
            arrays_by_detector[detector_seed], data_audits[str(detector_seed)] = load_data(
                args.repo_root, config, detector_seed
            )
    paired_input_audit = [
        {
            "detector_seed": detector_seed,
            "downstream_seed": downstream_seed,
            "inputs": data_audits[str(detector_seed)]["paths"],
        }
        for detector_seed, downstream_seed in seed_pairs
    ]
    arrays = arrays_by_detector[seed_pairs[0][0]]
    architecture_audit = {}
    sample = GeometryDataset(arrays["train"])
    batch = [item.unsqueeze(0) for item in sample[0][:-1]]
    for model_name in model_names:
        model = build_model(
            model_name, arrays["train"].relational.shape[1], arrays["train"].edges.shape[1]
        )
        output = model(batch[0], batch[1], batch[2])
        if output.shape != (1, 1):
            raise ValueError(f"{model_name} does not produce one Cobb output")
        transform = target_transform(arrays, config)
        mean, standard_deviation = _transform_arrays(transform)
        normalized_target = (
            batch[3] - torch.from_numpy(mean).unsqueeze(0)
        ) / torch.from_numpy(standard_deviation).unsqueeze(0)
        smoke_loss = nn.HuberLoss(delta=float(config["loss"]["delta"]))(
            output, normalized_target
        )
        smoke_loss.backward()
        if not any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise ValueError(f"{model_name} failed the backward-pass smoke test")
        architecture_audit[model_name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "representation_dim": 64,
            "output_shape": list(output.shape),
            "forward_backward_smoke_test": "passed",
        }
        if model_name in GAE_FEATURE_ABLATIONS:
            architecture_audit[model_name]["relational_features"] = dict(
                GAE_FEATURE_ABLATIONS[model_name]
            )

    print(json.dumps({
        "status": "data_and_architecture_audit_passed", "device": str(device),
        "seed_mode": seed_mode, "seed_pairs": seed_pairs,
        "paired_input_audit": paired_input_audit,
        "target_transforms": {
            str(detector_seed): target_transform(arrays_by_detector[detector_seed], config)
            for detector_seed in arrays_by_detector
        },
        "data_by_detector_seed": data_audits, "architectures": architecture_audit,
    }, indent=2), flush=True)
    if args.dry_run:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.json"
    results = {
        "protocol_version": config["protocol_version"],
        "task": "primary Cobb regression from predicted AASCE L68 selected to L8",
        "target_order": list(TARGETS),
        "seed_mode": seed_mode,
        "seed_pairs": [
            {"detector_seed": detector, "downstream_seed": downstream}
            for detector, downstream in seed_pairs
        ],
        "paired_input_audit": paired_input_audit,
        "target_transforms": {
            str(detector_seed): target_transform(arrays_by_detector[detector_seed], config)
            for detector_seed in arrays_by_detector
        },
        "models": {},
        "training_protocol": {
            key: config[key] for key in (
                "seeds", "batch_size", "num_workers", "amp", "max_epochs", "patience",
                "optimizer", "scheduler", "loss", "gradient_clip_norm", "search_space",
                "checkpoint_selection",
            )
        },
        "data_audit_by_detector_seed": data_audits,
        "architecture_audit": architecture_audit,
    }
    selected_ablation_models = [
        name for name in model_names if name in GAE_FEATURE_ABLATIONS
    ]
    if selected_ablation_models:
        results["gae_feature_ablation"] = {
            "design": (
                "identical full GAE architecture and parameter count; omitted "
                "relational components are deterministically zero-masked"
            ),
            "edge_channel_order": [
                "distance_over_max", "log_distance_over_median",
                "cos_orientation", "sin_orientation",
            ],
            "conditions": {
                name: GAE_FEATURE_ABLATIONS[name]
                for name in selected_ablation_models
            },
            "full_reference": (
                "existing model 'gae' under the same standardized paired-seed protocol"
            ),
        }
    if "target_standardization" in config:
        results["training_protocol"]["target_standardization"] = config[
            "target_standardization"
        ]
    if "coordinate_input_stem" in config:
        results["training_protocol"]["coordinate_input_stem"] = config[
            "coordinate_input_stem"
        ]
    tolerances = {
        target: float(config["metrics"]["tolerance_deg"][target]) for target in TARGETS
    }
    analytic_runs = {}
    for detector_seed, downstream_seed in seed_pairs:
        paired_arrays = arrays_by_detector[detector_seed]
        analytic_runs[str(downstream_seed)] = {
            "detector_seed": detector_seed,
            "downstream_seed": downstream_seed,
            "val": metric_block(
                paired_arrays["val"].truth, paired_arrays["val"].analytic, tolerances
            ),
            "test": metric_block(
                paired_arrays["test"].truth, paired_arrays["test"].analytic, tolerances
            ),
        }
    results["analytic_baseline"] = {
        "runs": analytic_runs,
        "test_summary_across_seeds": summarize_seeds(analytic_runs),
    }
    write_json(results_path, results)

    for model_name in model_names:
        model_root = args.output / model_name
        model_root.mkdir(parents=True, exist_ok=True)
        runs = {}
        for detector_seed, seed in seed_pairs:
            arrays = arrays_by_detector[detector_seed]
            trials = []
            for trial_index, hyperparameters in enumerate(config["search_space"]):
                checkpoint = (
                    model_root /
                    f"detector_{detector_seed}_seed_{seed}_trial_{trial_index}.pt"
                )
                trials.append(train_trial(
                    model_name, seed, hyperparameters, trial_index, arrays, config,
                    device, checkpoint,
                ))
            selected = min(trials, key=lambda item: item["validation_selection_score"])
            evaluations = evaluate_selected(
                model_name, seed, selected, arrays, config, device, args.output,
                detector_seed,
            )
            runs[str(seed)] = {
                "seed": seed,
                "detector_seed": detector_seed,
                "all_validation_trials": trials,
                "selected_trial": selected,
                **evaluations,
            }
            results["models"].setdefault(model_name, {})["runs"] = runs
            write_json(results_path, results)
        results["models"][model_name]["test_summary_across_seeds"] = summarize_seeds(runs)
        write_json(results_path, results)

    print(json.dumps({"status": "complete", "results": str(results_path)}, indent=2))


if __name__ == "__main__":
    main()
