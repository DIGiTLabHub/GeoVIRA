#!/usr/bin/env python3
"""Frozen-vision attribution using the established predicted-landmark GAE.

This is deliberately a small extension of the released geometry experiments:

* foot geometry is detector-predicted L6 (patient-OOF for training);
* Cobb geometry is detector-predicted L68 -> deterministic Auto-EV -> L8;
* the GAE is the unchanged ``GeometryAttentionEncoder``;
* the selected radiology encoder is frozen and its normalized global features
  are cached once;
* matched and shuffled-image models have exactly the same architecture and
  training protocol.  Shuffling is a deterministic within-split derangement.

No HIH model is trained by this script.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "geomed" / "src"))
sys.path.insert(0, str(SCRIPT_ROOT))

os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / "outputs" / "model_cache" / "torch"))
os.environ.setdefault(
    "HF_HOME", str(REPO_ROOT / "outputs" / "model_cache" / "huggingface")
)

import train_hvangle_joint_geometry_models as foot_benchmark  # noqa: E402
import train_joint_geometry_models as cobb_benchmark  # noqa: E402
from geomed.data import load_aasce_records, load_hvangle_records  # noqa: E402
from geomed.models import GeometryAttentionEncoder  # noqa: E402


ENCODERS = {
    "biomedclip": {
        "model": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "input_size": 224,
        "feature_dim": 512,
        "preprocessing": "black letterbox to 224, released BiomedCLIP transform",
    },
    "rad-dino": {
        "model": "microsoft/rad-dino",
        "input_size": 518,
        "feature_dim": 768,
        "preprocessing": "black letterbox to native 518, released RAD-DINO processor",
    },
    "medimageinsight": {
        "model": "MedImageInsight-v1.0.0",
        "input_size": 224,
        "feature_dim": 1024,
        "preprocessing": "black letterbox to 224, released MedImageInsight preprocessing",
    },
    "medclip-resnet50": {
        "model": "MedCLIP-ResNet50",
        "input_size": 224,
        "feature_dim": 512,
        "preprocessing": "224 grayscale letterbox, mean 0.5862786, SD 0.2795009",
    },
    "medclip-swin": {
        "model": "MedCLIP-Swin",
        "input_size": 224,
        "feature_dim": 512,
        "preprocessing": "224 grayscale letterbox, mean 0.5862786, SD 0.2795009",
    },
    "dax-resnet50-va": {
        "model": "DAX-ResNet50-vA",
        "input_size": 224,
        "feature_dim": 2048,
        "preprocessing": "224 RGB letterbox, channel mean/SD 0.5",
    },
}
SPLITS = ("train", "val", "test")
CONDITIONS = ("shuffled", "matched")
TASKS = ("foot", "cobb")
DEFAULT_FOOT_CONFIG = SCRIPT_ROOT / "controlled_geometry_standardized_config.json"
DEFAULT_COBB_CONFIG = SCRIPT_ROOT / "aasce_geometry_coordstem_standardized_config.json"


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def letterbox(image: Image.Image, size: int = 224) -> Image.Image:
    """The established frozen-vision preprocessing from visual-evidence runs."""
    contained = ImageOps.contain(image, (size, size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), color=0)
    canvas.paste(
        contained,
        ((size - contained.width) // 2, (size - contained.height) // 2),
    )
    return canvas


def resolve_device(value: str) -> torch.device:
    device = torch.device(
        "cuda" if value == "auto" and torch.cuda.is_available()
        else "cpu" if value == "auto" else value
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if (device.type == "cuda" and device.index is not None and
            device.index >= torch.cuda.device_count()):
        raise RuntimeError(
            f"CUDA device {device.index} requested, but only "
            f"{torch.cuda.device_count()} device(s) are visible"
        )
    return device


def records_for_task(task: str):
    if task == "foot":
        return load_hvangle_records(
            REPO_ROOT, unilateral_only=True, split_mode="patient", split_seed=42
        )
    if task == "cobb":
        return load_aasce_records(REPO_ROOT)
    raise ValueError(task)


def expected_ids(records) -> list[str]:
    return [record.identifier for split in SPLITS for record in records[split]]


def feature_cache_path(output: Path, task: str, encoder: str) -> Path:
    return output / "frozen_features" / f"{task}_{encoder.replace('-', '_')}_global.npz"


def validate_encoder_assets(encoder: str) -> dict:
    """Fail early when a local released checkpoint required by an adapter is absent."""
    required = {
        "medimageinsight": [
            REPO_ROOT / "MedImageInsights" / "2024.09.27" / "vision_model" /
            "medimageinsigt-v1.0.0.pt",
        ],
        "medclip-resnet50": [
            REPO_ROOT / "checkpoints" / "MedCLIP" / "resnet" / "pytorch_model.bin",
        ],
        "medclip-swin": [
            REPO_ROOT / "checkpoints" / "MedCLIP" / "vit" / "pytorch_model.bin",
        ],
        "dax-resnet50-va": [
            REPO_ROOT / "checkpoints" / "DAX" / "dax-resnet50-a.pth",
        ],
    }.get(encoder, [])
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{encoder} is missing released checkpoint(s): " +
            ", ".join(str(path) for path in missing)
        )
    return {
        "required_local_checkpoints": [str(path) for path in required],
        "all_present": True,
    }


def extract_features(task: str, encoder: str, output: Path, batch_size: int,
                     device: torch.device) -> Path:
    records = records_for_task(task)
    identifiers = expected_ids(records)
    specification = ENCODERS[encoder]
    model_name = specification["model"]
    input_size = int(specification["input_size"])
    feature_dim = int(specification["feature_dim"])
    cache_path = feature_cache_path(output, task, encoder)
    if cache_path.is_file():
        with np.load(cache_path) as archive:
            cached_ids = archive["identifiers"].astype(str).tolist()
            cached_model = str(archive["model"].item())
        if cached_ids != identifiers or cached_model != model_name:
            raise ValueError(f"stale frozen feature cache: {cache_path}")
        print(f"[{task}] reusing {cache_path}", flush=True)
        return cache_path

    wrapper = None
    if encoder == "biomedclip":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        processor = None
    elif encoder == "rad-dino":
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        preprocess = None
    elif encoder == "medimageinsight":
        sys.path.insert(0, str(REPO_ROOT / "MedImageInsights"))
        from medimageinsightmodel import MedImageInsight
        if device.type == "cuda":
            # The released wrapper uses generic ``cuda`` internally; setting the
            # current device makes that resolve to the explicitly requested GPU.
            torch.cuda.set_device(device)
        model_dir = REPO_ROOT / "MedImageInsights" / "2024.09.27"
        wrapper = MedImageInsight(
            str(model_dir), "medimageinsigt-v1.0.0.pt", "language_model.pth"
        )
        wrapper.load_model()
        wrapper.model.eval()
        model = wrapper.model
        preprocess = processor = None
    elif encoder in {"medclip-resnet50", "medclip-swin"}:
        from torchvision import transforms as T
        import transformers
        # Released MedCLIP imports the pre-v5 class name.  In Transformers 5
        # the same processor is exposed as CLIPImageProcessor.
        if not hasattr(transformers, "CLIPFeatureExtractor"):
            transformers.CLIPFeatureExtractor = transformers.CLIPImageProcessor
        sys.path.insert(0, str(REPO_ROOT / "external_models" / "MedCLIP"))
        from medclip import MedCLIPVisionModel, MedCLIPVisionModelViT
        is_resnet = encoder == "medclip-resnet50"
        checkpoint_dir = (
            REPO_ROOT / "checkpoints" / "MedCLIP" /
            ("resnet" if is_resnet else "vit")
        )
        vision_class = MedCLIPVisionModel if is_resnet else MedCLIPVisionModelViT
        model = vision_class(medclip_checkpoint=str(checkpoint_dir))
        preprocess = T.Compose([
            T.ToTensor(), T.Normalize([0.5862786], [0.2795009])
        ])
        processor = None
    elif encoder == "dax-resnet50-va":
        from torchvision import transforms as T
        import torchvision.models as tv_models
        checkpoint_path = REPO_ROOT / "checkpoints" / "DAX" / "dax-resnet50-a.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        teacher = checkpoint["teacher"]
        state = {
            key.replace("module.backbone.", ""): value
            for key, value in teacher.items() if key.startswith("module.backbone.")
        }
        model = tv_models.resnet50(weights=None)
        model.fc = nn.Identity()
        model.load_state_dict(state, strict=True)
        preprocess = T.Compose([
            T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        processor = None
    else:
        raise ValueError(encoder)
    model = model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    paths = [record.image_path for split in SPLITS for record in records[split]]
    chunks = []
    print(f"[{task}] extracting {len(paths)} frozen {encoder} features", flush=True)
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start:start + batch_size]:
            with Image.open(path) as image:
                mode = "L" if encoder.startswith("medclip-") else "RGB"
                images.append(letterbox(image.convert(mode), input_size).convert(mode))
        if encoder == "biomedclip":
            pixel_values = torch.stack([preprocess(image) for image in images]).to(device)
        elif encoder == "rad-dino":
            # The image is already square at RAD-DINO's native resolution, so
            # its official resize/crop stages cannot alter aspect or discard anatomy.
            pixel_values = processor(images=images, return_tensors="pt")[
                "pixel_values"
            ].to(device)
        elif encoder == "medimageinsight":
            encoded_images = []
            for image in images:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=95)
                encoded_images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
            features = np.asarray(
                wrapper.encode(images=encoded_images)["image_embeddings"],
                dtype=np.float32,
            )
            features /= np.linalg.norm(features, axis=1, keepdims=True).clip(1e-8)
            chunks.append(features)
            print(f"  {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
            continue
        else:
            pixel_values = torch.stack([preprocess(image) for image in images]).to(device)
        with torch.inference_mode():
            if encoder == "biomedclip":
                features = model.encode_image(pixel_values)
            elif encoder == "rad-dino":
                # RAD-DINO is a DINOv2 ViT; the established global descriptor is
                # the CLS token at position zero.
                features = model(pixel_values=pixel_values).last_hidden_state[:, 0, :]
            else:
                features = model(pixel_values)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        chunks.append(features.float().cpu().numpy())
        print(f"  {min(start + batch_size, len(paths))}/{len(paths)}", flush=True)
    features = np.concatenate(chunks).astype(np.float32)
    if features.shape != (len(identifiers), feature_dim):
        raise ValueError(f"unexpected {encoder} feature shape: {features.shape}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        identifiers=np.asarray(identifiers),
        features=features,
        encoder=np.asarray(encoder),
        model=np.asarray(model_name),
        preprocessing=np.asarray(specification["preprocessing"]),
        l2_normalized=np.asarray(True),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache_path


def load_feature_splits(task: str, encoder: str, output: Path,
                        records) -> dict[str, np.ndarray]:
    path = feature_cache_path(output, task, encoder)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}; run this script once with --extract-only (or without --dry-run)"
        )
    with np.load(path) as archive:
        identifiers = archive["identifiers"].astype(str).tolist()
        features = np.asarray(archive["features"], dtype=np.float32)
        model = str(archive["model"].item())
    if identifiers != expected_ids(records) or model != ENCODERS[encoder]["model"]:
        raise ValueError(f"frozen feature cache does not match {task} records")
    result, offset = {}, 0
    for split in SPLITS:
        stop = offset + len(records[split])
        result[split] = features[offset:stop]
        offset = stop
    return result


def sattolo_derangement(n: int, seed: int) -> np.ndarray:
    """Return one deterministic cycle, hence no sample can map to itself."""
    if n < 2:
        raise ValueError("a derangement requires at least two cases")
    rng = np.random.default_rng(seed)
    permutation = np.arange(n)
    for right in range(n - 1, 0, -1):
        left = int(rng.integers(0, right))
        permutation[left], permutation[right] = permutation[right], permutation[left]
    if np.any(permutation == np.arange(n)):
        raise AssertionError("Sattolo construction produced a fixed point")
    return permutation


def patient_derangement(patient_ids: np.ndarray, seed: int) -> np.ndarray:
    """Strengthen foot shuffling by also forbidding same-patient donors."""
    patient_ids = np.asarray(patient_ids)
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        permutation = rng.permutation(len(patient_ids))
        if (np.all(permutation != np.arange(len(patient_ids))) and
                np.all(patient_ids[permutation] != patient_ids)):
            return permutation
    raise RuntimeError("could not construct a no-self, different-patient derangement")


def split_permutations(task: str, arrays, seed: int) -> dict[str, np.ndarray]:
    offsets = {"train": 104729, "val": 130363, "test": 155921}
    result = {}
    for split in SPLITS:
        permutation_seed = int(seed) * 1_000_003 + offsets[split]
        if task == "foot":
            result[split] = patient_derangement(
                arrays[split].patient_ids, permutation_seed
            )
        else:
            result[split] = sattolo_derangement(
                len(arrays[split].identifiers), permutation_seed
            )
    return result


class FusionDataset(Dataset):
    def __init__(self, arrays, image_features: np.ndarray):
        if len(arrays.identifiers) != len(image_features):
            raise ValueError("geometry/image feature row count mismatch")
        self.arrays = arrays
        self.image_features = torch.from_numpy(
            np.asarray(image_features, dtype=np.float32)
        )

    def __len__(self):
        return len(self.arrays.identifiers)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.arrays.relational[index]),
            torch.from_numpy(self.arrays.edges[index]),
            self.image_features[index],
            torch.from_numpy(self.arrays.truth[index]),
            int(index),
        )


class GAEFrozenVisionFusion(nn.Module):
    """Unchanged paper GAE concatenated with a frozen normalized image vector."""

    def __init__(self, relational_dim: int, number_of_edges: int,
                 output_dim: int, image_dim: int = 512):
        super().__init__()
        self.geometry_encoder = GeometryAttentionEncoder(
            relational_dim, number_of_edges, output_dim=64,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(64 + image_dim),
            nn.Linear(64 + image_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim),
        )

    def forward(self, relational, edges, image_features):
        geometry = self.geometry_encoder(relational, edges)
        image = nn.functional.normalize(image_features.float(), dim=-1)
        return self.head(torch.cat([geometry, image], dim=1))


def make_loaders(arrays, features, permutations, condition: str,
                 batch_size: int, workers: int, seed: int,
                 device: torch.device):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    datasets = {}
    for split in SPLITS:
        indices = np.arange(len(features[split]))
        if condition == "shuffled":
            indices = permutations[split]
        datasets[split] = FusionDataset(arrays[split], features[split][indices])
    generator = torch.Generator().manual_seed(seed)
    return {
        split: DataLoader(
            datasets[split], batch_size=batch_size, shuffle=split == "train",
            num_workers=workers, pin_memory=device.type == "cuda",
            generator=generator if split == "train" else None,
        )
        for split in SPLITS
    }


@dataclass(frozen=True)
class TaskAdapter:
    name: str
    benchmark: object
    config: dict
    targets: tuple[str, ...]
    tolerances: dict[str, float]

    def transform(self, arrays):
        return self.benchmark.target_transform(arrays, self.config)

    def metrics(self, truth, prediction):
        return self.benchmark.metric_block(truth, prediction, self.tolerances)


def transform_arrays(transform: dict) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(transform["mean_deg"], dtype=np.float32),
        np.asarray(transform["sd_deg"], dtype=np.float32),
    )


def evaluate(model, loader, device, transform):
    model.eval()
    mean, standard_deviation = transform_arrays(transform)
    predictions, truths, indices = [], [], []
    with torch.inference_mode():
        for relational, edges, image, target, index in loader:
            output = model(
                relational.to(device, non_blocking=True),
                edges.to(device, non_blocking=True),
                image.to(device, non_blocking=True),
            )
            predictions.append(output.float().cpu().numpy() * standard_deviation + mean)
            truths.append(target.numpy())
            indices.append(index.numpy())
    order = np.concatenate(indices).argsort()
    return np.concatenate(truths)[order], np.concatenate(predictions)[order]


def train_trial(adapter: TaskAdapter, arrays, features, permutations,
                condition: str, seed: int, hyperparameters: dict,
                trial_index: int, device: torch.device, checkpoint: Path):
    seed_everything(seed)
    model = GAEFrozenVisionFusion(
        arrays["train"].relational.shape[1], arrays["train"].edges.shape[1],
        len(adapter.targets), features["train"].shape[1],
    ).to(device)
    loaders = make_loaders(
        arrays, features, permutations, condition,
        int(adapter.config["batch_size"]), int(adapter.config["num_workers"]),
        seed, device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(hyperparameters["lr"]),
        weight_decay=float(hyperparameters["weight_decay"]),
    )
    max_epochs = int(adapter.config["max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss_function = nn.HuberLoss(
        delta=float(adapter.config["loss"]["delta"]),
        reduction=adapter.config["loss"]["reduction"],
    )
    amp_enabled = bool(adapter.config["amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    transform = adapter.transform(arrays)
    target_mean, target_sd = transform_arrays(transform)
    target_mean_tensor = torch.from_numpy(target_mean).to(device)
    target_sd_tensor = torch.from_numpy(target_sd).to(device)
    best_score, best_epoch, stale_epochs = float("inf"), 0, 0
    history = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        for relational, edges, image, target, _ in loaders["train"]:
            target = target.to(device, non_blocking=True)
            normalized_target = (target - target_mean_tensor) / target_sd_tensor
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(
                    relational.to(device, non_blocking=True),
                    edges.to(device, non_blocking=True),
                    image.to(device, non_blocking=True),
                )
                loss = loss_function(prediction, normalized_target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), float(adapter.config["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
        scheduler.step()
        truth, prediction = evaluate(model, loaders["val"], device, transform)
        validation = adapter.metrics(truth, prediction)
        score = float(np.mean([validation[target]["mae"] for target in adapter.targets]))
        history.append({
            "epoch": epoch,
            "train_loss_standardized": running_loss / max(1, len(loaders["train"])),
            "validation_mae_deg": {
                target: validation[target]["mae"] for target in adapter.targets
            },
            "validation_selection_score": score,
        })
        if score < best_score:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            torch.save({
                "model": model.state_dict(), "task": adapter.name,
                "condition": condition, "seed": seed, "trial_index": trial_index,
                "hyperparameters": hyperparameters, "best_epoch": best_epoch,
                "validation_selection_score": best_score,
                "target_transform": transform,
            }, checkpoint)
        else:
            stale_epochs += 1
        print(json.dumps({
            "task": adapter.name, "condition": condition, "seed": seed,
            "trial": trial_index, "epoch": epoch,
            "validation_selection_score": score, "best_epoch": best_epoch,
        }), flush=True)
        if stale_epochs >= int(adapter.config["patience"]):
            break
    return {
        "trial_index": trial_index, "hyperparameters": hyperparameters,
        "best_epoch": best_epoch, "validation_selection_score": best_score,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint), "target_transform": transform,
        "epoch_history": history, "elapsed_seconds": time.time() - started,
    }


def evaluate_selected(adapter: TaskAdapter, arrays, features, permutations,
                      condition: str, seed: int, selected: dict,
                      device: torch.device, output: Path):
    checkpoint = torch.load(selected["checkpoint"], map_location=device, weights_only=False)
    model = GAEFrozenVisionFusion(
        arrays["train"].relational.shape[1], arrays["train"].edges.shape[1],
        len(adapter.targets), features["train"].shape[1],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    transform = adapter.transform(arrays)
    if checkpoint["target_transform"] != transform:
        raise ValueError("checkpoint target transform differs from training labels")
    loaders = make_loaders(
        arrays, features, permutations, condition,
        int(adapter.config["batch_size"]), int(adapter.config["num_workers"]),
        seed, device,
    )
    result = {}
    for split in ("val", "test"):
        truth, prediction = evaluate(model, loaders[split], device, transform)
        result[split] = adapter.metrics(truth, prediction)
        result[split]["prediction_diagnostics"] = {
            target: {
                "truth_mean_deg": float(truth[:, index].mean()),
                "truth_sd_deg": float(truth[:, index].std(ddof=1)),
                "prediction_mean_deg": float(prediction[:, index].mean()),
                "prediction_sd_deg": float(prediction[:, index].std(ddof=1)),
            }
            for index, target in enumerate(adapter.targets)
        }
        path = output / adapter.name / condition / f"seed_{seed}_{split}_predictions.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, identifiers=arrays[split].identifiers, truth=truth,
            prediction=prediction, absolute_error=np.abs(prediction - truth),
            image_source_indices=(
                permutations[split] if condition == "shuffled"
                else np.arange(len(arrays[split].identifiers))
            ),
        )
        result[split]["prediction_file"] = str(path)
    return result


def baseline_reference(task: str) -> dict:
    if task == "foot":
        path = REPO_ROOT / "outputs" / "benchmarks" / \
            "joint_geometry_hih_oof_paired_coordstem_standardized" / "results.json"
        key = "test_summary_across_pipeline_seeds"
    else:
        path = REPO_ROOT / "outputs" / "benchmarks" / \
            "joint_geometry_aasce_hih_oof_coordstem_standardized" / "results.json"
        key = "test_summary_across_seeds"
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    result = json.loads(path.read_text())
    return {
        "status": "loaded", "path": str(path), "model": "gae",
        "test_summary": result["models"]["gae"][key],
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_compact_table(results: dict, output: Path) -> tuple[Path, Path]:
    """Write geometry-only, shuffled and matched MAE summaries."""
    rows = []
    definitions = (
        ("GAE only", "baseline_gae_only", None),
        ("GAE + shuffled frozen vision", "conditions", "shuffled"),
        ("GAE + matched frozen vision", "conditions", "matched"),
    )
    for method, source, condition in definitions:
        row = {"method": method}
        for task, targets in (("foot", ("HVA", "IMA")), ("cobb", ("Cobb",))):
            task_result = results.get("tasks", {}).get(task)
            for target in targets:
                mean_key, sd_key = f"{target.lower()}_mae_mean", f"{target.lower()}_mae_sd"
                row[mean_key], row[sd_key] = "", ""
                if task_result is None:
                    continue
                if source == "baseline_gae_only":
                    baseline = task_result[source]
                    if baseline.get("status") != "loaded":
                        continue
                    summary = baseline["test_summary"]
                else:
                    item = task_result[source].get(condition, {})
                    summary = item.get("test_summary_across_pipeline_seeds", {})
                if target in summary:
                    row[mean_key] = summary[target]["mae"]["mean"]
                    row[sd_key] = summary[target]["mae"]["sd"]
        rows.append(row)
    csv_path = output / "paper_table_mae.csv"
    fields = [
        "method", "hva_mae_mean", "hva_mae_sd", "ima_mae_mean", "ima_mae_sd",
        "cobb_mae_mean", "cobb_mae_sd",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path = output / "paper_table_mae.json"
    write_json(json_path, {"rows": rows, "units": "degrees", "seeds": [17, 42, 73]})
    return csv_path, json_path


def load_adapter(task: str, foot_config: Path, cobb_config: Path) -> TaskAdapter:
    if task == "foot":
        config = json.loads(foot_config.read_text())
        benchmark = foot_benchmark
    else:
        config = json.loads(cobb_config.read_text())
        benchmark = cobb_benchmark
    targets = tuple(benchmark.TARGETS)
    tolerances = {
        target: float(config["metrics"]["tolerance_deg"][target])
        for target in targets
    }
    return TaskAdapter(task, benchmark, config, targets, tolerances)


def permutation_audit(task: str, arrays, records, permutations, seed: int) -> dict:
    result = {}
    for split in SPLITS:
        permutation = permutations[split]
        ids = arrays[split].identifiers.astype(str)
        donor_ids = ids[permutation]
        no_self = bool(np.all(donor_ids != ids))
        if not no_self:
            raise ValueError(f"{task}/{split}/seed {seed} shuffled pairing has self matches")
        patient_disjoint = None
        if task == "foot":
            patients = arrays[split].patient_ids
            patient_disjoint = bool(np.all(patients[permutation] != patients))
            if not patient_disjoint:
                raise ValueError("foot shuffled donor shares a patient with its recipient")
        result[split] = {
            "recipient_ids": ids.tolist(), "donor_ids": donor_ids.tolist(),
            "permutation": permutation.tolist(), "no_self_pairing": no_self,
            "different_patient_pairing": patient_disjoint,
        }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=tuple(ENCODERS), default="medimageinsight")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS,
                        default=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output", type=Path,
        help="defaults to an encoder-specific directory under outputs/benchmarks",
    )
    parser.add_argument("--foot-config", type=Path, default=DEFAULT_FOOT_CONFIG)
    parser.add_argument("--cobb-config", type=Path, default=DEFAULT_COBB_CONFIG)
    parser.add_argument("--extract-batch-size", type=int, default=32)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sorted(args.seeds) != [17, 42, 73] or len(set(args.seeds)) != 3:
        raise ValueError("the formal protocol requires paired pipeline seeds 17, 42, 73")
    device = resolve_device(args.device)
    encoder_asset_audit = validate_encoder_assets(args.encoder)
    if args.output is None:
        suffix = args.encoder.replace("-", "_")
        args.output = (
            REPO_ROOT / "outputs" / "benchmarks" /
            f"predicted_landmark_frozen_vision_fusion_{suffix}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.json"
    if results_path.is_file() and not (args.force or args.dry_run or args.extract_only):
        raise FileExistsError(f"{results_path} exists; pass --force for a fresh run")

    records_by_task = {task: records_for_task(task) for task in args.tasks}
    if args.extract_only or not args.dry_run:
        for task in args.tasks:
            extract_features(
                task, args.encoder, args.output, args.extract_batch_size, device
            )
        if args.extract_only:
            print(json.dumps({
                "status": "feature_extraction_complete",
                "paths": {
                    task: str(feature_cache_path(args.output, task, args.encoder))
                    for task in args.tasks
                },
            }, indent=2))
            return

    results = {
        "protocol_version": "predicted-landmark-frozen-vision-attribution-v1",
        "purpose": "matched-vs-shuffled case-specific radiographic evidence",
        "image_encoder": ENCODERS[args.encoder]["model"],
        "image_encoder_short_name": args.encoder,
        "image_encoder_trainable": False,
        "encoder_asset_audit": encoder_asset_audit,
        "image_preprocessing": ENCODERS[args.encoder]["preprocessing"],
        "geometry_encoder": "unchanged paper GeometryAttentionEncoder (GAE)",
        "geometry_inputs": {
            "foot": "predicted L6; train is five-fold patient OOF",
            "cobb": "predicted L68 -> deterministic Auto-EV -> selected L8; train is five-fold image OOF",
        },
        "conditions": {
            "matched": "geometry and frozen image feature from the same case",
            "shuffled": (
                "separately trained identical model; frozen real-image features are "
                "deranged within each split with no self-pairing"
            ),
        },
        "paired_pipeline_seeds": args.seeds,
        "tasks": {},
    }

    for task in args.tasks:
        adapter = load_adapter(task, args.foot_config, args.cobb_config)
        records = records_by_task[task]
        features = (
            None if args.dry_run else
            load_feature_splits(task, args.encoder, args.output, records)
        )
        arrays_by_seed, data_audits, permutation_audits = {}, {}, {}
        for seed in args.seeds:
            arrays, audit = adapter.benchmark.load_data(REPO_ROOT, adapter.config, seed)
            arrays_by_seed[seed], data_audits[str(seed)] = arrays, audit
            permutations = split_permutations(task, arrays, seed)
            permutation_audits[str(seed)] = permutation_audit(
                task, arrays, records, permutations, seed
            )

        sample_arrays = arrays_by_seed[args.seeds[0]]
        image_dim = (
            int(ENCODERS[args.encoder]["feature_dim"])
            if features is None else int(features["train"].shape[1])
        )
        smoke_model = GAEFrozenVisionFusion(
            sample_arrays["train"].relational.shape[1],
            sample_arrays["train"].edges.shape[1], len(adapter.targets), image_dim,
        )
        smoke_output = smoke_model(
            torch.from_numpy(sample_arrays["train"].relational[:2]),
            torch.from_numpy(sample_arrays["train"].edges[:2]),
            torch.randn(2, image_dim),
        )
        if smoke_output.shape != (2, len(adapter.targets)):
            raise ValueError("fusion architecture smoke test failed")
        smoke_output.sum().backward()
        task_result = {
            "targets": list(adapter.targets),
            "baseline_gae_only": baseline_reference(task),
            "data_audit_by_pipeline_seed": data_audits,
            "shuffle_audit_by_pipeline_seed": permutation_audits,
            "architecture": {
                "parameters": sum(p.numel() for p in smoke_model.parameters()),
                "geometry_token_dim": 64, "image_feature_dim": image_dim,
                "output_dim": len(adapter.targets),
                "forward_backward_smoke_test": "passed",
            },
            "training_protocol": {
                key: adapter.config[key] for key in (
                    "batch_size", "num_workers", "amp", "max_epochs", "patience",
                    "optimizer", "scheduler", "target_standardization", "loss",
                    "gradient_clip_norm", "checkpoint_selection", "search_space",
                )
            },
            "conditions": {},
        }
        results["tasks"][task] = task_result

        print(json.dumps({
            "status": "audit_passed", "task": task, "device": str(device),
            "paired_detector_downstream_seeds": [[seed, seed] for seed in args.seeds],
            "paired_predicted_landmark_inputs": {
                str(seed): data_audits[str(seed)]["paths"] for seed in args.seeds
            },
            "encoder": args.encoder,
            "encoder_asset_audit": encoder_asset_audit,
            "feature_cache": str(feature_cache_path(args.output, task, args.encoder)),
            "feature_cache_exists": feature_cache_path(
                args.output, task, args.encoder
            ).is_file(),
            "architecture": task_result["architecture"],
            "shuffle_no_self": True,
            "shuffle_pairing_examples": {
                str(seed): {
                    split: list(zip(
                        permutation_audits[str(seed)][split]["recipient_ids"][:3],
                        permutation_audits[str(seed)][split]["donor_ids"][:3],
                    ))
                    for split in SPLITS
                }
                for seed in args.seeds
            },
        }, indent=2), flush=True)
        if args.dry_run:
            continue

        for condition in args.conditions:
            runs = {}
            for seed in args.seeds:
                arrays = arrays_by_seed[seed]
                permutations = split_permutations(task, arrays, seed)
                trials = []
                for trial_index, hyperparameters in enumerate(adapter.config["search_space"]):
                    checkpoint = (
                        args.output / task / condition / "checkpoints" /
                        f"pipeline_seed_{seed}_trial_{trial_index}.pt"
                    )
                    trials.append(train_trial(
                        adapter, arrays, features, permutations, condition, seed,
                        hyperparameters, trial_index, device, checkpoint,
                    ))
                selected = min(trials, key=lambda row: row["validation_selection_score"])
                evaluation = evaluate_selected(
                    adapter, arrays, features, permutations, condition, seed,
                    selected, device, args.output,
                )
                runs[str(seed)] = {
                    "pipeline_seed": seed, "detector_seed": seed,
                    "downstream_seed": seed, "all_validation_trials": trials,
                    "selected_trial": selected, **evaluation,
                }
                task_result["conditions"].setdefault(condition, {})["runs"] = runs
                write_json(results_path, results)
            task_result["conditions"][condition]["test_summary_across_pipeline_seeds"] = (
                adapter.benchmark.summarize_seeds(runs)
            )
            write_json(results_path, results)

    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_complete", "results_would_be": str(results_path),
            "feature_caches": {
                task: str(feature_cache_path(args.output, task, args.encoder))
                for task in args.tasks
            },
        }, indent=2))
        return
    write_json(results_path, results)
    table_csv, table_json = write_compact_table(results, args.output)
    print(json.dumps({
        "status": "complete", "results": str(results_path),
        "paper_table_csv": str(table_csv), "paper_table_json": str(table_json),
    }, indent=2))


if __name__ == "__main__":
    main()
