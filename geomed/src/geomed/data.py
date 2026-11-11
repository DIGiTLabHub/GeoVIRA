"""Dataset tables and split protocols for GeoVIRA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image

from .features import aasce_primary_cobb, hv_analytic_angles, relational_features


Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class MeasurementRecord:
    identifier: str
    image_path: Path
    split: Split
    relational: np.ndarray
    edges: np.ndarray
    normalized_coordinates: np.ndarray
    targets: tuple[float, ...]
    analytic_targets: tuple[float, ...]
    metadata: dict


def _parse_segment(value: str, width: float, height: float) -> np.ndarray:
    numbers = np.asarray([float(item) for item in str(value).split(",")], dtype=np.float32)
    if numbers.shape != (4,):
        raise ValueError(f"expected four endpoint coordinates, got {numbers.shape}")
    numbers[0::2] *= float(width)
    numbers[1::2] *= float(height)
    return numbers.reshape(2, 2)


def load_hvangle_records(repo_root: Path, unilateral_only: bool = True,
                         split_mode: str = "official_image",
                         split_seed: int = 42) -> dict[Split, list[MeasurementRecord]]:
    """Load the released image split or the patient-disjoint experiment split."""

    root = Path(repo_root)
    data_dir = root / "HVAngleEst" / "HVAngleEst"
    split_dir = root / "HVAngleEst" / "splits"
    frame = pd.read_csv(data_dir / "datasets.csv")
    required = {
        "filename", "patient_id", "image_width", "image_height", "great_toe",
        "first_metatarsal", "second_metatarsal", "HVA", "IMA",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"HVAngleEst table is missing columns: {sorted(required - set(frame.columns))}")
    if unilateral_only:
        frame = frame[~frame.filename.duplicated(keep=False)]

    if split_mode == "patient":
        released_names = set().union(*(
            set(pd.read_csv(split_dir / f"{name}.csv")["filename"].astype(str))
            for name in ("train", "val", "test")
        ))
        frame = frame[frame.filename.isin(released_names)]
        manifest = (
            root / "outputs" / "benchmarks" / "controlled_cross_model_v2" /
            "hvangle" / "hvangle_results.json"
        )
        if manifest.is_file() and split_seed == 42:
            import json
            split_ids = json.loads(manifest.read_text())["split_ids"]
        else:
            patients = np.asarray(sorted(frame.patient_id.unique()))
            patients = patients[np.random.RandomState(split_seed).permutation(len(patients))]
            n_train, n_val = int(0.70 * len(patients)), int(0.15 * len(patients))
            patient_sets = {
                "train": set(patients[:n_train]),
                "val": set(patients[n_train:n_train + n_val]),
                "test": set(patients[n_train + n_val:]),
            }
            split_ids = {
                split: frame[frame.patient_id.isin(patient_sets[split])].filename.astype(str).tolist()
                for split in ("train", "val", "test")
            }
    elif split_mode == "official_image":
        split_ids = {
            split: pd.read_csv(split_dir / f"{split}.csv")["filename"].astype(str).tolist()
            for split in ("train", "val", "test")
        }
    else:
        raise ValueError(f"unsupported HVAngleEst split mode: {split_mode}")

    records: dict[Split, list[MeasurementRecord]] = {}
    for split in ("train", "val", "test"):
        split_names = split_ids[split]
        allowed = set(split_names)
        if split_mode == "patient":
            # The saved split manifest is the authority for both membership and order.
            rows = frame.set_index("filename", drop=False).loc[split_names].copy()
        else:
            # Match the submitted benchmark exactly: split CSVs define membership,
            # while datasets.csv defines the within-split image order.
            rows = frame[frame.filename.isin(allowed)].copy()
        split_records = []
        for row in rows.itertuples():
            points = np.vstack([
                _parse_segment(row.great_toe, row.image_width, row.image_height),
                _parse_segment(row.first_metatarsal, row.image_width, row.image_height),
                _parse_segment(row.second_metatarsal, row.image_width, row.image_height),
            ])
            normalized_points = points / np.asarray([row.image_width, row.image_height], dtype=np.float32)
            geometry = relational_features(points)
            analytic = hv_analytic_angles(points)
            image_path = data_dir / "images" / row.filename
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            split_records.append(MeasurementRecord(
                identifier=str(row.filename),
                image_path=image_path,
                split=split,
                relational=geometry.relational.numpy(),
                edges=geometry.edges.numpy(),
                normalized_coordinates=normalized_points.reshape(-1),
                targets=(float(row.HVA), float(row.IMA)),
                analytic_targets=analytic,
                metadata={"patient_id": int(row.patient_id), "point_count": 6},
            ))
        records[split] = split_records

    observed = [record.identifier for split in records.values() for record in split]
    if len(observed) != len(set(observed)):
        raise ValueError("HVAngleEst image identifiers overlap across released splits")
    expected = (
        {"train": 829, "val": 172, "test": 176}
        if split_mode == "patient" else
        {"train": 828, "val": 229, "test": 120}
    )
    actual = {split: len(rows) for split, rows in records.items()}
    if actual != expected:
        raise ValueError(f"unexpected unilateral HVAngleEst split sizes: {actual}, expected {expected}")
    if split_mode == "patient":
        patients = {
            split: {record.metadata["patient_id"] for record in rows}
            for split, rows in records.items()
        }
        if (patients["train"] & patients["val"] or
                patients["train"] & patients["test"] or
                patients["val"] & patients["test"]):
            raise ValueError("patient-disjoint split contains patient overlap")
    return records


def _aasce_split_ids(filenames: list[str]) -> dict[Split, list[str]]:
    permutation = np.random.RandomState(42).permutation(len(filenames))
    n_train, n_val = int(0.70 * len(filenames)), int(0.15 * len(filenames))
    train_set = {filenames[index] for index in permutation[:n_train]}
    val_set = {filenames[index] for index in permutation[n_train:n_train + n_val]}
    return {
        "train": [name for name in filenames if name in train_set],
        "val": [name for name in filenames if name in val_set],
        "test": [filenames[index] for index in permutation[n_train + n_val:]],
    }


def load_aasce_records(repo_root: Path) -> dict[Split, list[MeasurementRecord]]:
    """Load primary Cobb labels and deterministic GT end-vertebra geometry."""

    root = Path(repo_root)
    table_dir = root / "AASCE" / "data" / "train_txt"
    image_dir = root / "AASCE" / "data" / "train"
    filenames = pd.read_csv(table_dir / "filenames.csv", header=None)[0].astype(str).tolist()
    landmarks = pd.read_csv(table_dir / "landmarks.csv", header=None).to_numpy(np.float64)
    angles = pd.read_csv(table_dir / "angles.csv", header=None).to_numpy(np.float64)
    if len(filenames) != 481 or landmarks.shape != (481, 136) or angles.shape != (481, 3):
        raise ValueError("unexpected AASCE release table shapes")
    if len(filenames) != len(set(filenames)):
        raise ValueError("AASCE filenames are not unique")
    if not np.allclose(angles[:, 0], angles.max(axis=1), rtol=0.0, atol=1e-8):
        raise ValueError("AASCE column 0 is not the row-wise maximum released angle")

    by_name = {}
    reconstruction_error = []
    for identifier, landmark, target_row in zip(filenames, landmarks, angles):
        image_path = image_dir / identifier
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            analytic, upper, lower, selected_points = aasce_primary_cobb(landmark, image.size)
        reconstruction_error.append(abs(analytic - float(target_row[0])))
        geometry = relational_features(selected_points)
        by_name[identifier] = {
            "image_path": image_path,
            "geometry": geometry,
            "target": float(target_row[0]),
            "analytic": analytic,
            "upper": upper,
            "lower": lower,
            "normalized_points": selected_points / np.asarray(image.size, dtype=np.float32),
        }
    if max(reconstruction_error) > 0.02:
        raise ValueError(
            f"primary Cobb selector disagrees with angles.csv: max error {max(reconstruction_error):.6f}°"
        )

    split_ids = _aasce_split_ids(filenames)
    records: dict[Split, list[MeasurementRecord]] = {}
    for split, identifiers in split_ids.items():
        split_records = []
        for identifier in identifiers:
            item = by_name[identifier]
            geometry = item["geometry"]
            split_records.append(MeasurementRecord(
                identifier=identifier,
                image_path=item["image_path"],
                split=split,
                relational=geometry.relational.numpy(),
                edges=geometry.edges.numpy(),
                normalized_coordinates=item["normalized_points"].reshape(-1),
                targets=(item["target"],),
                analytic_targets=(item["analytic"],),
                metadata={
                    "upper_end_vertebra": item["upper"],
                    "lower_end_vertebra": item["lower"],
                    "point_count": 8,
                    "target_source": "angles.csv column 0",
                },
            ))
        records[split] = split_records
    expected = {"train": 336, "val": 72, "test": 73}
    actual = {split: len(rows) for split, rows in records.items()}
    if actual != expected:
        raise ValueError(f"unexpected AASCE split sizes: {actual}, expected {expected}")
    return records
