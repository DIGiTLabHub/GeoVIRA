"""Relational geometry features and analytical angle measurements."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
import torch


@dataclass(frozen=True)
class RelationalGeometry:
    """Flat triplet-bearing representation plus structured edge features."""

    relational: torch.Tensor
    edges: torch.Tensor
    edge_index: tuple[tuple[int, int], ...]
    triplet_count: int

    @property
    def relational_dim(self) -> int:
        return int(self.relational.numel())


def _as_points(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
        raise ValueError("points must have shape (N, 2), N >= 2")
    if not np.isfinite(values).all():
        raise ValueError("points contain non-finite values")
    return values


def relational_features(
    points: np.ndarray,
    edge_index: list[tuple[int, int]] | tuple[tuple[int, int], ...] | None = None,
) -> RelationalGeometry:
    """Build the paper's distance/orientation and graph-centred angle features.

    Each edge is `[d/d_max, log(d/d_median + eps), cos(theta), sin(theta)]`.
    At every vertex, every unordered pair of graph neighbours yields one angle encoded
    as `(cos(phi), sin(phi))`. A complete six-point graph therefore produces 15 edges
    and 60 triplet angles, matching the manuscript.
    """

    pts = _as_points(points)
    n_points = len(pts)
    if edge_index is None:
        edge_index = list(combinations(range(n_points), 2))
    canonical = []
    seen = set()
    for left, right in edge_index:
        left, right = int(left), int(right)
        if not (0 <= left < n_points and 0 <= right < n_points) or left == right:
            raise ValueError(f"invalid edge {(left, right)} for {n_points} points")
        edge = (min(left, right), max(left, right))
        if edge not in seen:
            seen.add(edge); canonical.append(edge)
    if not canonical:
        raise ValueError("edge_index must contain at least one edge")

    raw_distances = np.asarray(
        [np.linalg.norm(pts[right] - pts[left]) for left, right in canonical],
        dtype=np.float32,
    )
    if np.any(raw_distances <= 1e-6):
        raise ValueError("geometry contains a zero-length edge")
    maximum = float(raw_distances.max())
    median = float(np.median(raw_distances))
    edge_rows = []
    adjacency = {index: [] for index in range(n_points)}
    for distance, (left, right) in zip(raw_distances, canonical):
        dx, dy = pts[right] - pts[left]
        theta = math.atan2(float(dy), float(dx))
        edge_rows.append([
            float(distance / maximum),
            math.log(float(distance / median) + 1e-6),
            math.cos(theta),
            math.sin(theta),
        ])
        adjacency[left].append(right); adjacency[right].append(left)

    triplets = []
    for center in range(n_points):
        for left, right in combinations(sorted(adjacency[center]), 2):
            first, second = pts[left] - pts[center], pts[right] - pts[center]
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denominator <= 1e-12:
                raise ValueError("geometry contains a degenerate triplet")
            cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
            triplets.append([cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine))])
    if not triplets:
        triplets = [[0.0, 0.0]]

    edges = torch.tensor(edge_rows, dtype=torch.float32)
    triplet_tensor = torch.tensor(triplets, dtype=torch.float32)
    relational = torch.cat([edges.reshape(-1), triplet_tensor.reshape(-1)])
    return RelationalGeometry(
        relational=relational,
        edges=edges,
        edge_index=tuple(canonical),
        triplet_count=len(triplets),
    )


def _acute_axis_angle(first: np.ndarray, second: np.ndarray) -> float:
    first, second = np.asarray(first, float), np.asarray(second, float)
    cosine = abs(float(np.dot(first, second))) / (
        float(np.linalg.norm(first) * np.linalg.norm(second)) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def hv_analytic_angles(points: np.ndarray) -> tuple[float, float]:
    """Return HVA and IMA from `[great-toe endpoints, first-MT endpoints, second-MT endpoints]`."""

    pts = _as_points(points)
    if pts.shape != (6, 2):
        raise ValueError("HVAngleEst requires six endpoints")
    great_toe = pts[1] - pts[0]
    first_metatarsal = pts[3] - pts[2]
    second_metatarsal = pts[5] - pts[4]
    return (
        _acute_axis_angle(great_toe, first_metatarsal),
        _acute_axis_angle(first_metatarsal, second_metatarsal),
    )


def aasce_primary_cobb(
    normalized_landmarks: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[float, int, int, np.ndarray]:
    """Reconstruct the released primary Cobb angle and selected eight corners."""

    values = np.asarray(normalized_landmarks, dtype=np.float64)
    if values.shape != (136,) or not np.isfinite(values).all():
        raise ValueError("AASCE landmarks must contain 136 finite values")
    width, height = image_size
    points = np.column_stack([values[:68] * width, values[68:] * height])
    vertebrae = points.reshape(17, 4, 2)
    directions = 0.5 * (
        (vertebrae[:, 1] - vertebrae[:, 0])
        + (vertebrae[:, 3] - vertebrae[:, 2])
    )
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("AASCE contains a zero-length vertebral direction")
    cosine = np.clip(
        directions @ directions.T / (norms[:, None] * norms[None, :]),
        -1.0,
        1.0,
    )
    angles = np.degrees(np.arccos(cosine))
    upper, lower = np.triu_indices(17, 1)
    selected = int(np.argmax(angles[upper, lower]))
    upper_index, lower_index = int(upper[selected]), int(lower[selected])
    selected_points = np.vstack([vertebrae[upper_index], vertebrae[lower_index]])
    return (
        float(angles[upper_index, lower_index]),
        upper_index,
        lower_index,
        selected_points.astype(np.float32),
    )

