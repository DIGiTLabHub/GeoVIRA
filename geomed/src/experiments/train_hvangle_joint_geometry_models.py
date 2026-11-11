#!/usr/bin/env python3
"""Train joint HVA/IMA regressors from fixed HIH landmark predictions.

The three input archives are treated as immutable:

* train uses patient-held-out OOF HIH predictions;
* validation selects epochs and hyperparameters;
* test is evaluated once after validation selection.

All models share a 64-dimensional representation and the same two-output
regression head.  One model jointly predicts HVA and IMA; no target-specific
models are trained.
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
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("controlled_geometry_coordstem_standardized_config.json")
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "benchmarks" / "joint_geometry_hih_oof"
sys.path.insert(0, str(REPO_ROOT / "geomed" / "src"))

from geomed.data import load_hvangle_records  # noqa: E402
from geomed.models import GeometryAttentionEncoder  # noqa: E402


TARGETS = ("HVA", "IMA")
MODEL_NAMES = ('coordmlp', 'deepsets', 'deepsets_meanmax', 'deepsets_role', 'deepsets_role_coordstem', 'gnn', 'gnn_role', 'gnn_role_coordstem', 'gae', 'gae_distance_only', 'gae_distance_orientation', 'gae_distance_triplets', 'gae_angle_only')

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
    patient_ids: np.ndarray
    coordinates: np.ndarray
    relational: np.ndarray
    edges: np.ndarray
    truth: np.ndarray
    analytic: np.ndarray
    folds: np.ndarray | None


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
    if points.shape != (6, 2) or not np.isfinite(points).all():
        raise ValueError("predicted HVAngleEst points must have shape (6,2)")
    edge_index = list(combinations(range(6), 2))
    distances = np.asarray([
        np.linalg.norm(points[right] - points[left]) for left, right in edge_index
    ], dtype=np.float32)
    positive = distances[distances > 1e-6]
    maximum = float(distances.max()) if len(positive) else 1.0
    median = float(np.median(distances))
    if median <= 1e-6:
        median = float(np.median(positive)) if len(positive) else 1.0

    edge_rows = []
    adjacency = {index: [] for index in range(6)}
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
    for center in range(6):
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
            nn.Linear(64, 2),
        )


class CoordMLP(nn.Module):
    """MLP over six size-normalized predicted landmark coordinates."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(12),
            nn.Linear(12, 256),
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

    def __init__(self, number_of_points: int = 6):
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


class DeepSetsMeanMax(DeepSets):
    """Stronger invariant readout using complementary mean and max statistics."""

    def __init__(self, number_of_points: int = 6):
        super().__init__(number_of_points)
        self.readout = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU()
        )

    def forward(self, coordinates, relational, edges):
        del relational, edges
        identity = self.identity(self.point_ids).unsqueeze(0).expand(
            coordinates.shape[0], -1, -1
        )
        point_tokens = self.point_encoder(torch.cat([coordinates, identity], dim=-1))
        pooled = torch.cat([point_tokens.mean(dim=1), point_tokens.amax(dim=1)], dim=-1)
        return self.head(self.readout(pooled))


class DeepSetsRoleConcat(DeepSets):
    """Identity-conditioned tokens with a fixed clinically labeled readout."""

    def __init__(self, number_of_points: int = 6):
        super().__init__(number_of_points)
        self.readout = nn.Sequential(
            nn.LayerNorm(number_of_points * 64),
            nn.Linear(number_of_points * 64, 64),
            nn.GELU(),
        )

    def forward(self, coordinates, relational, edges):
        del relational, edges
        identity = self.identity(self.point_ids).unsqueeze(0).expand(
            coordinates.shape[0], -1, -1
        )
        point_tokens = self.point_encoder(torch.cat([coordinates, identity], dim=-1))
        return self.head(self.readout(point_tokens.flatten(1)))


def normalize_landmark_cloud(coordinates: torch.Tensor) -> torch.Tensor:
    """Center and isotropically scale one landmark cloud at a time.

    The upstream coordinates are already divided by max(image width, height),
    which preserves angles.  This second normalization removes only global
    translation and isotropic scale, while retaining every measurement-bearing
    relative direction and distance ratio.  Unlike LayerNorm over individual
    points, it never rescales x and y independently.
    """
    centered = coordinates - coordinates.mean(dim=1, keepdim=True)
    rms_radius = torch.sqrt(
        centered.square().sum(dim=-1).mean(dim=1, keepdim=True)
    ).unsqueeze(-1)
    return centered / rms_radius.clamp_min(1e-6)


class CoordinateIdentityStem(nn.Module):
    """Encode geometry and landmark identity on separate, balanced branches."""

    def __init__(self, number_of_points: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.register_buffer(
            "point_ids", torch.arange(number_of_points), persistent=False
        )
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.identity = nn.Embedding(number_of_points, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = normalize_landmark_cloud(coordinates)
        coordinate_features = self.coordinate_encoder(normalized)
        identity_features = self.identity(self.point_ids).unsqueeze(0).expand(
            coordinates.shape[0], -1, -1
        )
        tokens = self.fusion(torch.cat([coordinate_features, identity_features], dim=-1))
        return tokens, normalized


class DeepSetsRoleConcatCoordinateStem(nn.Module):
    """Role-preserving DeepSets with a geometry-first coordinate stem."""

    def __init__(self, number_of_points: int = 6):
        super().__init__()
        self.stem = CoordinateIdentityStem(number_of_points)
        self.point_encoder = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(),
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(number_of_points * 64),
            nn.Linear(number_of_points * 64, 64),
            nn.GELU(),
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
    """Two-layer message-passing GNN on the complete six-landmark graph."""

    def __init__(self, number_of_points: int = 6):
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


class RolePreservingGNN(FullyConnectedGNN):
    """The same message passing followed by fixed-order labeled-node readout."""

    def __init__(self, number_of_points: int = 6):
        super().__init__(number_of_points)
        self.readout = nn.Sequential(
            nn.LayerNorm(number_of_points * 64),
            nn.Linear(number_of_points * 64, 64),
            nn.GELU(),
        )

    def forward(self, coordinates, relational, edges):
        del relational, edges
        identity = self.identity(self.point_ids).unsqueeze(0).expand(
            coordinates.shape[0], -1, -1
        )
        hidden = self.input_projection(torch.cat([coordinates, identity], dim=-1))
        for layer in self.layers:
            hidden = layer(hidden, coordinates)
        return self.head(self.readout(hidden.flatten(1)))


class RolePreservingGNNCoordinateStem(nn.Module):
    """Residual complete-graph GNN with geometry-first node initialization."""

    def __init__(self, number_of_points: int = 6):
        super().__init__()
        self.stem = CoordinateIdentityStem(number_of_points)
        self.layers = nn.ModuleList([MessagePassingLayer(), MessagePassingLayer()])
        self.readout = nn.Sequential(
            nn.LayerNorm(number_of_points * 64),
            nn.Linear(number_of_points * 64, 64),
            nn.GELU(),
        )
        self.head = JointRegressionHead()

    def forward(self, coordinates, relational, edges):
        del relational, edges
        hidden, normalized_coordinates = self.stem(coordinates)
        for layer in self.layers:
            hidden = layer(hidden, normalized_coordinates)
        return self.head(self.readout(hidden.flatten(1)))


class JointGAE(nn.Module):
    """The paper's edge-attention/triplet GAE with a joint two-angle head."""

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
    """Original GAE with selected relational inputs deterministically masked.

    The encoder and joint prediction head are unchanged, so every ablation has
    exactly the same trainable parameter count as full GAE.  Edge channels 0:2
    are the two distance features and channels 2:4 are cosine/sine orientation.
    The flattened triplet branch begins after ``number_of_edges * 4`` values.
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
    if name == "deepsets_meanmax":
        return DeepSetsMeanMax()
    if name == "deepsets_role":
        return DeepSetsRoleConcat()
    if name == "deepsets_role_coordstem":
        return DeepSetsRoleConcatCoordinateStem()
    if name == "gnn":
        return FullyConnectedGNN()
    if name == "gnn_role":
        return RolePreservingGNN()
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
        required = {
            "identifiers", "patient_ids", "predicted_points", "truth", "analytic"
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing fields: {sorted(missing)}")
        identifiers = archive["identifiers"].astype(str)
        patient_ids = np.asarray(archive["patient_ids"], dtype=np.int64)
        predicted_points = np.asarray(archive["predicted_points"], dtype=np.float32)
        truth = np.asarray(archive["truth"], dtype=np.float32)
        analytic = np.asarray(archive["analytic"], dtype=np.float32)
        folds = np.asarray(archive["fold"], dtype=np.int64) if "fold" in archive.files else None

    expected_identifiers = np.asarray([record.identifier for record in records])
    expected_patients = np.asarray(
        [record.metadata["patient_id"] for record in records], dtype=np.int64
    )
    expected_truth = np.asarray([record.targets for record in records], dtype=np.float32)
    if len(identifiers) != expected_size or len(records) != expected_size:
        raise ValueError(f"unexpected {split} size: archive={len(identifiers)}, records={len(records)}")
    if not np.array_equal(identifiers, expected_identifiers):
        raise ValueError(f"{split} identifiers do not match the frozen patient split and order")
    if not np.array_equal(patient_ids, expected_patients):
        raise ValueError(f"{split} patient IDs do not match the frozen manifest")
    if predicted_points.shape != (expected_size, 6, 2) or truth.shape != (expected_size, 2):
        raise ValueError(f"unexpected {split} point/target shapes")
    if patient_ids.shape != (expected_size,) or analytic.shape != (expected_size, 2):
        raise ValueError(f"unexpected {split} patient/analytic shapes")
    if split == "train" and folds is None:
        raise ValueError("training archive must contain a patient-OOF fold for every row")
    if split != "train" and folds is not None:
        raise ValueError(f"{split} archive must not contain training OOF folds")
    if folds is not None and folds.shape != (expected_size,):
        raise ValueError(f"unexpected {split} fold shape: {folds.shape}")
    if not np.allclose(truth, expected_truth, rtol=0.0, atol=1e-5):
        raise ValueError(f"{split} targets differ from the canonical HVA/IMA table")
    if not all(np.isfinite(item).all() for item in (predicted_points, truth, analytic)):
        raise ValueError(f"{split} contains non-finite values")

    coordinates, relational_rows, edge_rows = [], [], []
    for points, record in zip(predicted_points, records):
        # Isotropic normalization preserves directions/angles and removes image scale.
        from PIL import Image
        with Image.open(record.image_path) as image:
            width, height = image.size
        coordinates.append(points / float(max(width, height)))
        relational, edges = predicted_relational_features(points)
        relational_rows.append(relational)
        edge_rows.append(edges)

    return GeometryArrays(
        identifiers=identifiers,
        patient_ids=patient_ids,
        coordinates=np.asarray(coordinates, dtype=np.float32),
        relational=np.asarray(relational_rows, dtype=np.float32),
        edges=np.asarray(edge_rows, dtype=np.float32),
        truth=truth,
        analytic=analytic,
        folds=folds,
    )


def load_data(repo_root: Path, config: dict, detector_seed: int):
    dataset_config = config["datasets"]["hvangle"]
    if dataset_config["target_mode"] != "joint_hva_ima":
        raise ValueError("configuration does not request joint HVA/IMA prediction")
    if tuple(dataset_config["target_order"]) != TARGETS:
        raise ValueError("target order must be HVA, IMA")
    records = load_hvangle_records(
        repo_root, unilateral_only=True, split_mode="patient", split_seed=42
    )
    sizes = dataset_config["outer_split"]["sizes"]
    arrays = {}
    paths = {}
    for split in ("train", "val", "test"):
        relative = dataset_config["landmark_inputs"][split].format(
            detector_seed=detector_seed
        )
        paths[split] = repo_root / relative
        arrays[split] = load_split(paths[split], records[split], split, int(sizes[split]))

    patient_sets = {
        split: set(values.patient_ids.tolist()) for split, values in arrays.items()
    }
    if (patient_sets["train"] & patient_sets["val"] or
            patient_sets["train"] & patient_sets["test"] or
            patient_sets["val"] & patient_sets["test"]):
        raise ValueError("patient leakage detected across train/validation/test")
    if arrays["train"].folds is None:
        raise ValueError("training landmarks are not marked as OOF")
    patient_fold_counts = {}
    for patient, fold in zip(arrays["train"].patient_ids, arrays["train"].folds):
        patient_fold_counts.setdefault(int(patient), set()).add(int(fold))
    if any(len(value) != 1 for value in patient_fold_counts.values()):
        raise ValueError("one or more training patients cross OOF folds")
    if sorted(set(arrays["train"].folds.tolist())) != [0, 1, 2, 3, 4]:
        raise ValueError("training archive does not contain exactly five OOF folds")

    audit = {
        "paths": {key: str(value) for key, value in paths.items()},
        "sizes": {key: len(value.identifiers) for key, value in arrays.items()},
        "patient_counts": {
            key: len(set(value.patient_ids.tolist())) for key, value in arrays.items()
        },
        "patient_disjoint": True,
        "train_prediction_policy": "five-fold patient OOF",
        "train_fold_sizes": {
            str(fold): int(np.sum(arrays["train"].folds == fold)) for fold in range(5)
        },
        "detector_seed": detector_seed,
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
    """Fit the target transform strictly on training-OOF labels."""
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
            raise ValueError("training targets have an invalid standard deviation")
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
            output_degrees = output.float().cpu().numpy() * standard_deviation + mean
            predictions.append(output_degrees)
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
            "validation_hva_mae_deg": validation_metrics["HVA"]["mae"],
            "validation_ima_mae_deg": validation_metrics["IMA"]["mae"],
            "validation_joint_mae_deg": selection_score,
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
            "val_hva_mae": validation_metrics["HVA"]["mae"],
            "val_ima_mae": validation_metrics["IMA"]["mae"],
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
                      device: torch.device, output_root: Path,
                      splits: tuple[str, ...] = ("val", "test")):
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
    for split in splits:
        truth, prediction = evaluate(model, loaders[split], device, transform)
        result[split] = metric_block(truth, prediction, tolerances)
        result[split]["prediction_diagnostics"] = {
            target: {
                "truth_mean_deg": float(truth[:, index].mean()),
                "truth_sd_deg": float(truth[:, index].std(ddof=1)),
                "prediction_mean_deg": float(prediction[:, index].mean()),
                "prediction_sd_deg": float(prediction[:, index].std(ddof=1)),
            }
            for index, target in enumerate(TARGETS)
        }
        prediction_path = output_root / model_name / f"seed_{seed}_{split}_predictions.npz"
        np.savez_compressed(
            prediction_path,
            identifiers=arrays[split].identifiers,
            patient_ids=arrays[split].patient_ids,
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
    parser.add_argument("--detector-seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    seeds = args.seeds if args.seeds is not None else [int(item) for item in config["seeds"]]
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    arrays, data_audit = load_data(args.repo_root, config, args.detector_seed)
    architecture_audit = {}
    sample = GeometryDataset(arrays["train"])
    batch = [item.unsqueeze(0) for item in sample[0][:-1]]
    for model_name in args.models:
        model = build_model(
            model_name, arrays["train"].relational.shape[1], arrays["train"].edges.shape[1]
        )
        output = model(batch[0], batch[1], batch[2])
        if output.shape != (1, 2):
            raise ValueError(f"{model_name} does not produce joint HVA/IMA output")
        smoke_loss = nn.HuberLoss(delta=5.0)(output, batch[3])
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

    print(json.dumps({
        "status": "data_and_architecture_audit_passed", "device": str(device),
        "data": data_audit, "architectures": architecture_audit,
    }, indent=2), flush=True)
    if args.dry_run:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.json"
    results = {
        "protocol_version": config["protocol_version"],
        "task": "joint HVA/IMA regression from fixed predicted landmarks",
        "target_order": list(TARGETS),
        "models": {},
        "training_protocol": {
            key: config[key] for key in (
                "seeds", "batch_size", "num_workers", "amp", "max_epochs", "patience",
                "optimizer", "scheduler", "loss", "gradient_clip_norm", "search_space",
                "checkpoint_selection", "matched_reference",
            )
        },
        "data_audit": data_audit,
        "architecture_audit": architecture_audit,
    }
    write_json(results_path, results)

    for model_name in args.models:
        model_root = args.output / model_name
        model_root.mkdir(parents=True, exist_ok=True)
        runs = {}
        for seed in seeds:
            trials = []
            for trial_index, hyperparameters in enumerate(config["search_space"]):
                checkpoint = model_root / f"seed_{seed}_trial_{trial_index}.pt"
                trials.append(train_trial(
                    model_name, seed, hyperparameters, trial_index, arrays, config,
                    device, checkpoint,
                ))
            selected = min(trials, key=lambda item: item["validation_selection_score"])
            evaluations = evaluate_selected(
                model_name, seed, selected, arrays, config, device, args.output
            )
            runs[str(seed)] = {
                "seed": seed,
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
