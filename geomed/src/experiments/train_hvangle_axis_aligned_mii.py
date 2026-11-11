#!/usr/bin/env python3
"""Train measurement-aligned foot regression with case-matched visual features.

The independently trained feature-map shuffle control is implemented in
train_localized_mii_image_shuffled.py.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "geomed" / "src"))
sys.path.insert(0, str(SCRIPT_ROOT))

import train_hvangle_joint_geometry_models as benchmark  # noqa: E402
from geomed.data import load_hvangle_records  # noqa: E402
from geomed.models import GeometryAttentionEncoder  # noqa: E402


DEFAULT_CONFIG = SCRIPT_ROOT / "controlled_geometry_standardized_config.json"
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs" / "benchmarks" / "hvangle_axis_aligned_mii"
)
CONDITIONS = ("axis_aligned_matched",)
EXTERNAL_CONDITIONS = (
    "axis_aligned_image_shuffled",
    "axis_aligned_feature_map_shuffled",
)
SEEDS = (17, 42, 73)
SPLITS = ("train", "val", "test")
AXIS_EDGE_INDICES = (0, 9, 14)  # (0,1), (2,3), (4,5) in complete-graph order.
MII_INPUT_SIZE = 480
LETTERBOX_SIZE = 224
SPATIAL_CHANNELS = 2048
AXIS_SAMPLES = 16


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    device = torch.device(
        "cuda" if value == "auto" and torch.cuda.is_available()
        else "cpu" if value == "auto" else value
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if (device.type == "cuda" and device.index is not None and
            device.index >= torch.cuda.device_count()):
        raise RuntimeError(f"requested {device}, but only {torch.cuda.device_count()} GPUs exist")
    return device


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def letterbox_with_mapping(image: Image.Image, size: int = LETTERBOX_SIZE):
    """Return the exact existing black letterbox plus original->canvas mapping."""
    width, height = image.size
    contained = ImageOps.contain(image, (size, size), Image.Resampling.BILINEAR)
    pad_x = (size - contained.width) // 2
    pad_y = (size - contained.height) // 2
    canvas = Image.new("RGB", (size, size), color=0)
    canvas.paste(contained, (pad_x, pad_y))
    mapping = np.asarray([
        contained.width / float(width), contained.height / float(height),
        float(pad_x), float(pad_y),
    ], dtype=np.float32)
    return canvas, mapping


def spatial_forward(image_encoder: nn.Module, pixels: torch.Tensor):
    """Expose DaViT's final token grid immediately before global average pooling."""
    x = pixels
    spatial_size = (x.size(2), x.size(3))
    for convolution, block in zip(image_encoder.convs, image_encoder.blocks):
        x, spatial_size = convolution(x, spatial_size)
        x, spatial_size = block(x, spatial_size)
    if x.ndim != 3 or x.shape[-1] != SPATIAL_CHANNELS:
        raise ValueError(f"unexpected MedImageInsight spatial token shape: {tuple(x.shape)}")
    height, width = int(spatial_size[0]), int(spatial_size[1])
    if height * width != x.shape[1]:
        raise ValueError(f"spatial metadata {spatial_size} disagrees with {x.shape[1]} tokens")
    return x, (height, width)


def verify_prepool_tokens(model, pixels, tokens) -> float:
    """Prove that the exposed tokens reproduce the released global pathway."""
    pooled = model.image_encoder.norms(tokens.mean(dim=1))
    manual = F.normalize(pooled @ model.image_projection, dim=-1)
    official = model.encode_image(pixels, norm=True)
    error = float(torch.max(torch.abs(manual - official)).item())
    if error > 2e-5:
        raise RuntimeError(
            f"pre-pooling token verification failed: max global difference={error:.8g}"
        )
    return error


def sample_axis_tokens(spatial_tokens: torch.Tensor, spatial_size: tuple[int, int],
                       predicted_points: np.ndarray, mappings: np.ndarray) -> torch.Tensor:
    """Bilinearly sample 16 points on each predicted clinical axis."""
    batch, _, channels = spatial_tokens.shape
    height, width = spatial_size
    feature_map = spatial_tokens.transpose(1, 2).reshape(batch, channels, height, width)
    # Clone because torch.as_tensor may share memory with the cached NumPy
    # predictions; the mapping below must never mutate the HIH archive array.
    points = torch.as_tensor(predicted_points, dtype=torch.float32,
                             device=feature_map.device).clone().reshape(batch, 3, 2, 2)
    mapping = torch.as_tensor(mappings, dtype=torch.float32, device=feature_map.device)
    points[..., 0] = points[..., 0] * mapping[:, None, None, 0] + mapping[:, None, None, 2]
    points[..., 1] = points[..., 1] * mapping[:, None, None, 1] + mapping[:, None, None, 3]

    fractions = torch.linspace(0.0, 1.0, AXIS_SAMPLES, device=feature_map.device)
    samples = (
        points[:, :, 0, None, :] * (1.0 - fractions[None, None, :, None]) +
        points[:, :, 1, None, :] * fractions[None, None, :, None]
    )
    # Coordinates refer to the exact 224 canvas fed to the released preprocessing.
    # align_corners=False maps normalized pixel-center coordinates consistently
    # across the 480-pixel encoder input and its 15x15 final feature map.
    # A coordinate may lie between the final pixel centre (223) and the right
    # image boundary (224).  Under align_corners=False that legitimately maps
    # to a grid value slightly above 1 and padding_mode="border" samples the
    # final feature cell.  Validate against the continuous canvas extent, not
    # the feature-centre interval [-1, 1].
    if torch.any(samples < -1e-4) or torch.any(samples >= LETTERBOX_SIZE + 1e-4):
        minimum = float(samples.min().item())
        maximum = float(samples.max().item())
        raise ValueError(
            f"predicted axis lies outside the {LETTERBOX_SIZE}px letterbox: "
            f"range=[{minimum:.6f}, {maximum:.6f}]"
        )
    grid = torch.empty_like(samples)
    grid[..., 0] = 2.0 * (samples[..., 0] + 0.5) / LETTERBOX_SIZE - 1.0
    grid[..., 1] = 2.0 * (samples[..., 1] + 0.5) / LETTERBOX_SIZE - 1.0
    sampled = F.grid_sample(
        feature_map.float(), grid, mode="bilinear", padding_mode="border",
        align_corners=False,
    )  # B,C,3,16
    return sampled.mean(dim=-1).transpose(1, 2)  # B,3,C


def feature_path(output: Path, seed: int, split: str) -> Path:
    return output / "axis_features" / f"seed_{seed}" / f"{split}.npz"


def load_predicted_points(audit: dict, arrays_by_seed: dict, seed: int, split: str):
    path = Path(audit[str(seed)]["paths"][split])
    with np.load(path) as archive:
        identifiers = archive["identifiers"].astype(str)
        points = np.asarray(archive["predicted_points"], dtype=np.float32)
    if not np.array_equal(identifiers, arrays_by_seed[seed][split].identifiers):
        raise ValueError(f"{seed}/{split} predicted point identifiers changed")
    if points.shape != (len(identifiers), 6, 2):
        raise ValueError(f"unexpected predicted L6 shape: {points.shape}")
    return points


def extract_axis_features(output: Path, arrays_by_seed: dict, audits: dict,
                          batch_size: int, device: torch.device) -> dict:
    """Extract only axis samples; never save a large generic spatial feature bank."""
    checkpoint = (
        REPO_ROOT / "MedImageInsights" / "2024.09.27" / "vision_model" /
        "medimageinsigt-v1.0.0.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    records = load_hvangle_records(
        REPO_ROOT, unilateral_only=True, split_mode="patient", split_seed=42
    )
    complete = all(
        feature_path(output, seed, split).is_file()
        for seed in SEEDS for split in SPLITS
    )
    if complete:
        # Never silently reuse tokens produced from a different geometry source,
        # landmark order, encoder preprocessing, or spatial representation.
        for seed in SEEDS:
            load_axis_features(output, arrays_by_seed[seed], seed)
        with np.load(feature_path(output, SEEDS[0], SPLITS[0])) as archive:
            spatial_shape = archive["spatial_shape"].astype(int).tolist()
            verification_error = float(archive["official_global_max_abs_error"])
        print("reusing complete MedImageInsight axis-feature cache", flush=True)
        return {
            "status": "reused", "encoder_loaded": False,
            "spatial_shape": spatial_shape,
            "official_global_max_abs_error": verification_error,
        }

    sys.path.insert(0, str(REPO_ROOT / "MedImageInsights"))
    from medimageinsightmodel import MedImageInsight
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model_dir = REPO_ROOT / "MedImageInsights" / "2024.09.27"
    wrapper = MedImageInsight(
        str(model_dir), "medimageinsigt-v1.0.0.pt", "language_model.pth"
    )
    wrapper.load_model()
    configured_size = tuple(int(value) for value in wrapper.opt["IMAGE_ENCODER"]["IMAGE_SIZE"])
    if configured_size != (MII_INPUT_SIZE, MII_INPUT_SIZE):
        raise ValueError(f"unexpected MedImageInsight input size: {configured_size}")
    if bool(wrapper.opt["TEST"]["CENTER_CROP"]):
        raise ValueError("spatial coordinate mapping does not permit a center crop")
    wrapper.model.to(device).eval()
    for parameter in wrapper.model.parameters():
        parameter.requires_grad_(False)

    verification_error = None
    spatial_shape = None
    for split in SPLITS:
        expected_ids = np.asarray([record.identifier for record in records[split]])
        for seed in SEEDS:
            if not np.array_equal(expected_ids, arrays_by_seed[seed][split].identifiers):
                raise ValueError(f"{seed}/{split} records do not match predicted geometry")
        points_by_seed = {
            seed: load_predicted_points(audits, arrays_by_seed, seed, split)
            for seed in SEEDS
        }
        collected = {seed: [] for seed in SEEDS}
        for start in range(0, len(records[split]), batch_size):
            batch_records = records[split][start:start + batch_size]
            tensors, mappings = [], []
            for record in batch_records:
                with Image.open(record.image_path) as image:
                    canvas, mapping = letterbox_with_mapping(image.convert("RGB"))
                # Match the existing global-MII cache path, including JPEG round trip.
                buffer = io.BytesIO()
                canvas.save(buffer, format="JPEG", quality=95)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                decoded = wrapper.decode_base64_image(encoded)
                tensors.append(wrapper.preprocess(decoded))
                mappings.append(mapping)
            pixels = torch.stack(tensors).to(device)
            if tuple(pixels.shape[-2:]) != (MII_INPUT_SIZE, MII_INPUT_SIZE):
                raise ValueError(f"official preprocessing returned {tuple(pixels.shape[-2:])}")
            with torch.inference_mode():
                spatial, shape = spatial_forward(wrapper.model.image_encoder, pixels)
                if verification_error is None:
                    verification_error = verify_prepool_tokens(wrapper.model, pixels, spatial)
                    spatial_shape = [int(shape[0]), int(shape[1]), int(spatial.shape[-1])]
                for seed in SEEDS:
                    batch_points = points_by_seed[seed][start:start + len(batch_records)]
                    tokens = sample_axis_tokens(
                        spatial, shape, batch_points, np.asarray(mappings, np.float32)
                    )
                    collected[seed].append(tokens.cpu().numpy().astype(np.float32))
            print(f"[{split}] {min(start + batch_size, len(records[split]))}/"
                  f"{len(records[split])}", flush=True)
        for seed in SEEDS:
            tokens = np.concatenate(collected[seed])
            if tokens.shape != (len(records[split]), 3, SPATIAL_CHANNELS):
                raise ValueError(f"unexpected axis token shape: {tokens.shape}")
            path = feature_path(output, seed, split)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path, identifiers=expected_ids, axis_tokens=tokens,
                roles=np.asarray(["toe", "mt1", "mt2"]),
                detector_seed=np.asarray(seed), source=np.asarray("HIH predicted L6"),
                encoder=np.asarray("MedImageInsight-v1.0.0 frozen pre-pooling DaViT"),
                spatial_shape=np.asarray(spatial_shape), axis_samples=np.asarray(AXIS_SAMPLES),
                letterbox_size=np.asarray(LETTERBOX_SIZE),
                encoder_input_size=np.asarray(MII_INPUT_SIZE),
                center_crop=np.asarray(False),
                official_global_max_abs_error=np.asarray(verification_error),
            )
    del wrapper
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "status": "extracted", "encoder_loaded": True,
        "spatial_shape": spatial_shape,
        "official_global_max_abs_error": verification_error,
    }


def load_axis_features(output: Path, arrays, seed: int) -> dict[str, np.ndarray]:
    result = {}
    for split in SPLITS:
        path = feature_path(output, seed, split)
        if not path.is_file():
            raise FileNotFoundError(f"{path}; run --extract-only first")
        with np.load(path) as archive:
            identifiers = archive["identifiers"].astype(str)
            tokens = np.asarray(archive["axis_tokens"], dtype=np.float32)
            source = str(archive["source"].item())
            roles = tuple(archive["roles"].astype(str).tolist())
            cached_seed = int(archive["detector_seed"])
            encoder = str(archive["encoder"].item())
            spatial_shape = tuple(archive["spatial_shape"].astype(int).tolist())
            axis_samples = int(archive["axis_samples"])
            letterbox_size = int(archive["letterbox_size"])
            encoder_input_size = int(archive["encoder_input_size"])
            center_crop = bool(archive["center_crop"])
            verification_error = float(archive["official_global_max_abs_error"])
        if not np.array_equal(identifiers, arrays[split].identifiers):
            raise ValueError(f"axis features do not match {seed}/{split} geometry")
        if tokens.shape != (len(identifiers), 3, SPATIAL_CHANNELS):
            raise ValueError(f"invalid cached axis feature shape: {tokens.shape}")
        if source != "HIH predicted L6":
            raise ValueError("axis cache is not derived from predicted landmarks")
        expected_metadata = (
            roles == ("toe", "mt1", "mt2") and cached_seed == seed and
            encoder == "MedImageInsight-v1.0.0 frozen pre-pooling DaViT" and
            spatial_shape == (15, 15, SPATIAL_CHANNELS) and
            axis_samples == AXIS_SAMPLES and letterbox_size == LETTERBOX_SIZE and
            encoder_input_size == MII_INPUT_SIZE and not center_crop and
            math.isfinite(verification_error) and verification_error <= 2e-5
        )
        if not expected_metadata:
            raise ValueError(f"stale or incompatible axis feature cache: {path}")
        result[split] = tokens
    return result


class AxisDataset(Dataset):
    def __init__(self, arrays, axis_tokens):
        self.arrays = arrays
        self.axis_tokens = torch.from_numpy(np.asarray(axis_tokens, np.float32))

    def __len__(self):
        return len(self.arrays.identifiers)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.arrays.relational[index]),
            torch.from_numpy(self.arrays.edges[index]),
            self.axis_tokens[index], torch.from_numpy(self.arrays.truth[index]),
            int(index),
        )


class AxisAlignedGAE(nn.Module):
    """Unchanged GAE plus shared small edge--axis visual alignment modules."""
    def __init__(self, relational_dim: int, number_of_edges: int,
                 visual_dim: int = SPATIAL_CHANNELS, alignment_dim: int = 32):
        super().__init__()
        self.gae = GeometryAttentionEncoder(
            relational_dim, number_of_edges, output_dim=64
        )
        self.geo_projector = nn.Sequential(
            nn.LayerNorm(4), nn.Linear(4, alignment_dim), nn.GELU()
        )
        self.visual_projector = nn.Sequential(
            nn.LayerNorm(visual_dim), nn.Linear(visual_dim, alignment_dim), nn.GELU()
        )
        self.alignment = nn.Sequential(
            nn.LayerNorm(2 * alignment_dim),
            nn.Linear(2 * alignment_dim, alignment_dim), nn.GELU(),
        )
        self.head = benchmark.JointRegressionHead(64 + 3 * alignment_dim)

    def forward(self, relational, edges, axis_visual, condition: str):
        # ``axis_aligned_image_shuffled`` is trained by the focused external
        # control script.  Its tensor layout is identical to matched: only the
        # case row supplying the already-extracted visual tokens is permuted.
        if condition not in CONDITIONS + EXTERNAL_CONDITIONS:
            raise ValueError(condition)
        geometry = self.gae(relational, edges)
        axis_edges = edges[:, AXIS_EDGE_INDICES, :]
        geo_tokens = self.geo_projector(axis_edges)
        visual_tokens = self.visual_projector(axis_visual)
        aligned = self.alignment(torch.cat([geo_tokens, visual_tokens], dim=-1))
        return self.head(torch.cat([geometry, aligned.flatten(1)], dim=1))


def make_loaders(arrays, features, config, seed, device):
    generator = torch.Generator().manual_seed(seed)
    return {
        split: DataLoader(
            AxisDataset(arrays[split], features[split]),
            batch_size=int(config["batch_size"]), shuffle=split == "train",
            num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda",
            generator=generator if split == "train" else None,
        )
        for split in SPLITS
    }


def transform_arrays(transform):
    return (
        np.asarray(transform["mean_deg"], np.float32),
        np.asarray(transform["sd_deg"], np.float32),
    )


def evaluate(model, loader, condition, device, transform):
    model.eval()
    mean, sd = transform_arrays(transform)
    predictions, truths, indices = [], [], []
    with torch.inference_mode():
        for relational, edges, visual, target, index in loader:
            output = model(
                relational.to(device, non_blocking=True),
                edges.to(device, non_blocking=True),
                visual.to(device, non_blocking=True), condition,
            )
            predictions.append(output.float().cpu().numpy() * sd + mean)
            truths.append(target.numpy())
            indices.append(index.numpy())
    order = np.concatenate(indices).argsort()
    return np.concatenate(truths)[order], np.concatenate(predictions)[order]


def train_trial(condition, seed, hp, trial_index, arrays, features, config,
                device, checkpoint):
    seed_everything(seed)
    model = AxisAlignedGAE(
        arrays["train"].relational.shape[1], arrays["train"].edges.shape[1]
    ).to(device)
    loaders = make_loaders(arrays, features, config, seed, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"])
    )
    max_epochs = int(config["max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.HuberLoss(
        delta=float(config["loss"]["delta"]), reduction=config["loss"]["reduction"]
    )
    amp = bool(config["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    transform = benchmark.target_transform(arrays, config)
    mean, sd = transform_arrays(transform)
    mean_t, sd_t = torch.from_numpy(mean).to(device), torch.from_numpy(sd).to(device)
    tolerances = {
        target: float(config["metrics"]["tolerance_deg"][target])
        for target in benchmark.TARGETS
    }
    best, best_epoch, stale, history = float("inf"), 0, 0, []
    started = time.time()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for relational, edges, visual, target, _ in loaders["train"]:
            target = target.to(device, non_blocking=True)
            normalized = (target - mean_t) / sd_t
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                prediction = model(
                    relational.to(device, non_blocking=True),
                    edges.to(device, non_blocking=True),
                    visual.to(device, non_blocking=True), condition,
                )
                loss = criterion(prediction, normalized)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())
        scheduler.step()
        truth, prediction = evaluate(model, loaders["val"], condition, device, transform)
        metrics = benchmark.metric_block(truth, prediction, tolerances)
        score = float(np.mean([metrics[target]["mae"] for target in benchmark.TARGETS]))
        history.append({
            "epoch": epoch, "train_loss_standardized": running / len(loaders["train"]),
            "validation_hva_mae_deg": metrics["HVA"]["mae"],
            "validation_ima_mae_deg": metrics["IMA"]["mae"],
            "validation_joint_mae_deg": score,
        })
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            torch.save({
                "model": model.state_dict(), "condition": condition, "seed": seed,
                "trial_index": trial_index, "hyperparameters": hp,
                "best_epoch": best_epoch, "validation_selection_score": best,
                "target_transform": transform,
            }, checkpoint)
        else:
            stale += 1
        print(json.dumps({
            "condition": condition, "seed": seed, "trial": trial_index,
            "epoch": epoch, "val_hva_mae": metrics["HVA"]["mae"],
            "val_ima_mae": metrics["IMA"]["mae"], "val_joint_mae": score,
            "best_epoch": best_epoch,
        }), flush=True)
        if stale >= int(config["patience"]):
            break
    return {
        "trial_index": trial_index, "hyperparameters": hp, "best_epoch": best_epoch,
        "validation_selection_score": best,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(checkpoint), "target_transform": transform,
        "epoch_history": history, "elapsed_seconds": time.time() - started,
    }


def evaluate_selected(condition, seed, selected, arrays, features, config,
                      device, output):
    checkpoint = torch.load(selected["checkpoint"], map_location=device, weights_only=False)
    model = AxisAlignedGAE(
        arrays["train"].relational.shape[1], arrays["train"].edges.shape[1]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    transform = benchmark.target_transform(arrays, config)
    if checkpoint["target_transform"] != transform:
        raise ValueError("checkpoint target transform mismatch")
    loaders = make_loaders(arrays, features, config, seed, device)
    tolerances = {
        target: float(config["metrics"]["tolerance_deg"][target])
        for target in benchmark.TARGETS
    }
    result = {}
    for split in ("val", "test"):
        truth, prediction = evaluate(model, loaders[split], condition, device, transform)
        result[split] = benchmark.metric_block(truth, prediction, tolerances)
        path = output / condition / f"seed_{seed}_{split}_predictions.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, identifiers=arrays[split].identifiers,
            patient_ids=arrays[split].patient_ids, truth=truth, prediction=prediction,
            absolute_error=np.abs(prediction - truth),
        )
        result[split]["prediction_file"] = str(path)
    return result


def existing_comparison() -> dict:
    geometry_path = (
        REPO_ROOT / "outputs" / "benchmarks" /
        "joint_geometry_hih_oof_paired_coordstem_standardized" / "results.json"
    )
    global_path = (
        REPO_ROOT / "outputs" / "benchmarks" /
        "predicted_landmark_frozen_vision_fusion_medimageinsight" / "results.json"
    )
    result = {}
    if geometry_path.is_file():
        saved = json.loads(geometry_path.read_text())
        result["gae_only"] = saved["models"]["gae"]["test_summary_across_pipeline_seeds"]
        result["gae_only_path"] = str(geometry_path)
    if global_path.is_file():
        saved = json.loads(global_path.read_text())
        result["medimageinsight_global_matched"] = saved["tasks"]["foot"][
            "conditions"
        ]["matched"]["test_summary_across_pipeline_seeds"]
        result["medimageinsight_global_matched_path"] = str(global_path)
    return result


def compact_report(results: dict) -> dict:
    rows = {}
    existing = results["existing_results"]
    if "gae_only" in existing:
        rows["GAE only"] = existing["gae_only"]
    if "medimageinsight_global_matched" in existing:
        rows["MedImageInsight global matched"] = existing[
            "medimageinsight_global_matched"
        ]
    for condition in CONDITIONS:
        if condition in results["conditions"]:
            rows[condition] = results["conditions"][condition][
                "test_summary_across_pipeline_seeds"
            ]
    compact = {}
    for name, summary in rows.items():
        compact[name] = {
            target: {
                "mae_mean": summary[target]["mae"]["mean"],
                "mae_sd": summary[target]["mae"]["sd"],
                "icc_mean": summary[target]["icc_2_1"]["mean"],
                "icc_sd": summary[target]["icc_2_1"]["sd"],
                "tolerance_mean_percent": summary[target][
                    "within_tolerance_percent"
                ]["mean"],
                "tolerance_sd_percent": summary[target][
                    "within_tolerance_percent"
                ]["sd"],
            }
            for target in benchmark.TARGETS
        }
    return compact


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--extract-batch-size", type=int, default=2)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    if tuple(int(seed) for seed in config["seeds"]) != SEEDS:
        raise ValueError("formal paired seeds must be 17, 42, 73")
    device = resolve_device(args.device)
    arrays_by_seed, audits = {}, {}
    for seed in SEEDS:
        arrays_by_seed[seed], audits[str(seed)] = benchmark.load_data(
            REPO_ROOT, config, seed
        )

    sample = arrays_by_seed[SEEDS[0]]["train"]
    model = AxisAlignedGAE(sample.relational.shape[1], sample.edges.shape[1])
    relational = torch.from_numpy(sample.relational[:2])
    edges = torch.from_numpy(sample.edges[:2])
    visual = torch.randn(2, 3, SPATIAL_CHANNELS)
    aligned = model(relational, edges, visual, "axis_aligned_matched")
    if aligned.shape != (2, 2):
        raise ValueError("axis alignment model output smoke test failed")
    aligned.sum().backward()
    architecture = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "unchanged_gae": True, "gae_token_dim": 64,
        "alignment_dim": 32, "axis_visual_dim": SPATIAL_CHANNELS,
        "axis_roles": ["toe", "mt1", "mt2"],
        "axis_edge_indices": list(AXIS_EDGE_INDICES),
        "output_shape": list(aligned.shape), "forward_backward_smoke_test": "passed",
    }
    print(json.dumps({
        "status": "protocol_and_architecture_audit_passed", "device": str(device),
        "conditions": list(CONDITIONS), "paired_seeds": list(SEEDS),
        "predicted_geometry_paths": {
            seed: audits[str(seed)]["paths"] for seed in SEEDS
        }, "architecture": architecture,
    }, indent=2), flush=True)
    if args.dry_run:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    extraction = extract_axis_features(
        args.output, arrays_by_seed, audits, args.extract_batch_size, device
    )
    if args.extract_only:
        print(json.dumps({
            "status": "axis_feature_extraction_complete", "output": str(args.output),
            "extraction": extraction,
        }, indent=2))
        return

    results_path = args.output / "results.json"
    if results_path.exists() and not args.force:
        raise FileExistsError(f"{results_path} exists; use --force only for an intentional rerun")
    results = {
        "protocol_version": "hvangle-axis-specific-mii-alignment-v1",
        "task": "joint HVA/IMA from patient-OOF HIH predicted L6",
        "conditions": {}, "architecture": architecture,
        "spatial_feature_extraction": extraction,
        "paired_pipeline_seeds": list(SEEDS),
        "data_audit_by_seed": audits,
        "training_protocol": {
            key: config[key] for key in (
                "batch_size", "num_workers", "amp", "max_epochs", "patience",
                "optimizer", "scheduler", "target_standardization", "loss",
                "gradient_clip_norm", "checkpoint_selection", "search_space",
            )
        },
        "controls": {
            "image_encoder": "frozen MedImageInsight-v1.0.0",
            "spatial_source": "final 15x15x2048 DaViT tokens before global pooling",
            "landmark_source": "HIH predicted L6 only",
            "samples_per_axis": AXIS_SAMPLES,
            "image_pairing": "case-matched visual features",
        },
        "existing_results": existing_comparison(),
    }
    write_json(results_path, results)
    for condition in CONDITIONS:
        runs = {}
        for seed in SEEDS:
            arrays = arrays_by_seed[seed]
            features = load_axis_features(args.output, arrays, seed)
            trials = []
            for trial_index, hp in enumerate(config["search_space"]):
                checkpoint = (
                    args.output / condition / "checkpoints" /
                    f"pipeline_seed_{seed}_trial_{trial_index}.pt"
                )
                trials.append(train_trial(
                    condition, seed, hp, trial_index, arrays, features, config,
                    device, checkpoint,
                ))
            selected = min(trials, key=lambda row: row["validation_selection_score"])
            evaluation = evaluate_selected(
                condition, seed, selected, arrays, features, config, device, args.output
            )
            runs[str(seed)] = {
                "pipeline_seed": seed, "detector_seed": seed,
                "downstream_seed": seed, "all_validation_trials": trials,
                "selected_trial": selected, **evaluation,
            }
            results["conditions"].setdefault(condition, {})["runs"] = runs
            write_json(results_path, results)
        results["conditions"][condition]["test_summary_across_pipeline_seeds"] = (
            benchmark.summarize_seeds(runs)
        )
        write_json(results_path, results)
    report = compact_report(results)
    write_json(args.output / "compact_comparison.json", report)
    print(json.dumps({
        "status": "complete", "comparison": report,
        "results": str(results_path),
        "compact_comparison": str(args.output / "compact_comparison.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
