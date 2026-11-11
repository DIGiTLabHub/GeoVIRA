#!/usr/bin/env python3
"""MII cross-attention conditioned on predicted measurement geometry.

The experiment changes only the global-concat fusion mechanism used by
``train_predicted_landmark_frozen_vision_fusion.py``.  The unchanged GAE
geometry token is a single query over the frozen 15x15 MedImageInsight spatial
tokens.  Matched and within-split shuffled-image conditions use identical
models, capacity, geometry, and training protocols.  No GT landmarks enter.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "geomed" / "src"))
sys.path.insert(0, str(SCRIPT_ROOT))

os.environ.setdefault(
    "HF_HOME", str(REPO_ROOT / "outputs" / "model_cache" / "huggingface")
)

import train_hvangle_axis_aligned_mii as spatial  # noqa: E402
import train_predicted_landmark_frozen_vision_fusion as global_fusion  # noqa: E402
from geomed.models import GeometryAttentionEncoder  # noqa: E402


TASKS = ("foot", "cobb")
CONDITIONS = ("shuffled", "matched")
SPLITS = ("train", "val", "test")
SEEDS = (17, 42, 73)
PATCH_COUNT = 15 * 15
PATCH_DIM = 2048
ATTENTION_DIM = 512
ATTENTION_HEADS = 8
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs" / "benchmarks" /
    "predicted_landmark_mii_cross_attention"
)


def patch_path(output: Path, task: str, split: str) -> Path:
    return output / "frozen_mii_patches" / task / f"{split}_patches_fp16.npy"


def metadata_path(output: Path, task: str, split: str) -> Path:
    return output / "frozen_mii_patches" / task / f"{split}_metadata.npz"


def expected_identifiers(records: dict, split: str) -> np.ndarray:
    return np.asarray([record.identifier for record in records[split]])


def validate_patch_cache(output: Path, task: str, records: dict) -> bool:
    for split in SPLITS:
        patches_file = patch_path(output, task, split)
        metadata_file = metadata_path(output, task, split)
        if not patches_file.is_file() or not metadata_file.is_file():
            return False
        patches = np.load(patches_file, mmap_mode="r")
        with np.load(metadata_file) as metadata:
            identifiers = metadata["identifiers"].astype(str)
            valid = (
                str(metadata["encoder"].item()) == "MedImageInsight-v1.0.0"
                and tuple(metadata["spatial_shape"].tolist()) == (15, 15, 2048)
                and tuple(metadata["mii_input_size"].tolist()) == (480, 480)
                and int(metadata["letterbox_size"]) == spatial.LETTERBOX_SIZE
                and float(metadata["official_global_max_abs_error"]) <= 2e-5
                and bool(metadata["frozen_encoder"])
            )
        expected_shape = (len(records[split]), PATCH_COUNT, PATCH_DIM)
        if patches.shape != expected_shape or patches.dtype != np.float16:
            raise ValueError(
                f"stale MII patch cache {patches_file}: "
                f"shape={patches.shape}, dtype={patches.dtype}"
            )
        if not valid or not np.array_equal(
            identifiers, expected_identifiers(records, split)
        ):
            raise ValueError(f"stale MII patch metadata: {metadata_file}")
    return True


def extract_mii_patches(
    task: str,
    output: Path,
    records: dict,
    batch_size: int,
    device: torch.device,
) -> dict:
    if validate_patch_cache(output, task, records):
        print(f"[{task}] reusing frozen MII spatial-token cache", flush=True)
        return {
            "status": "reused",
            "paths": {split: str(patch_path(output, task, split)) for split in SPLITS},
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
    if tuple(wrapper.opt["IMAGE_ENCODER"]["IMAGE_SIZE"]) != (480, 480):
        raise ValueError("MedImageInsight input size changed")
    if bool(wrapper.opt["TEST"]["CENTER_CROP"]):
        raise ValueError("MII center crop would invalidate the established mapping")
    wrapper.model.to(device).eval()
    for parameter in wrapper.model.parameters():
        parameter.requires_grad_(False)

    official_error = None
    written_paths = {}
    for split in SPLITS:
        target = patch_path(output, task, split)
        temporary = target.with_suffix(target.suffix + ".tmp")
        target.parent.mkdir(parents=True, exist_ok=True)
        store = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float16,
            shape=(len(records[split]), PATCH_COUNT, PATCH_DIM),
        )
        for start in range(0, len(records[split]), batch_size):
            batch_records = records[split][start:start + batch_size]
            tensors = []
            for record in batch_records:
                with Image.open(record.image_path) as image:
                    canvas, _ = spatial.letterbox_with_mapping(image.convert("RGB"))
                buffer = io.BytesIO()
                canvas.save(buffer, format="JPEG", quality=95)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                tensors.append(wrapper.preprocess(wrapper.decode_base64_image(encoded)))
            pixels = torch.stack(tensors).to(device)
            with torch.inference_mode():
                tokens, shape = spatial.spatial_forward(
                    wrapper.model.image_encoder, pixels
                )
                if shape != (15, 15) or tokens.shape[1:] != (PATCH_COUNT, PATCH_DIM):
                    raise ValueError(
                        f"unexpected MII spatial representation: {tokens.shape}, {shape}"
                    )
                if official_error is None:
                    official_error = spatial.verify_prepool_tokens(
                        wrapper.model, pixels, tokens
                    )
            stop = start + len(batch_records)
            store[start:stop] = tokens.float().cpu().numpy().astype(np.float16)
            print(f"[{task}/{split}] {stop}/{len(records[split])}", flush=True)
        store.flush()
        del store
        temporary.replace(target)
        np.savez_compressed(
            metadata_path(output, task, split),
            identifiers=expected_identifiers(records, split),
            encoder=np.asarray("MedImageInsight-v1.0.0"),
            spatial_shape=np.asarray((15, 15, 2048)),
            mii_input_size=np.asarray((480, 480)),
            letterbox_size=np.asarray(spatial.LETTERBOX_SIZE),
            official_global_max_abs_error=np.asarray(official_error),
            frozen_encoder=np.asarray(True),
            storage_dtype=np.asarray("float16"),
            training_compute_dtype=np.asarray("float32/AMP"),
        )
        written_paths[split] = str(target)
    del wrapper
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not validate_patch_cache(output, task, records):
        raise RuntimeError("MII patch extraction did not produce a complete cache")
    return {
        "status": "extracted",
        "paths": written_paths,
        "official_global_max_abs_error": official_error,
    }


class PatchFusionDataset(Dataset):
    def __init__(self, arrays, patch_file: Path, source_indices: np.ndarray):
        self.arrays = arrays
        self.patch_file = str(patch_file)
        self.source_indices = np.asarray(source_indices, dtype=np.int64)
        if len(self.source_indices) != len(arrays.identifiers):
            raise ValueError("geometry/patch source count mismatch")
        self._patches = None

    @property
    def patches(self):
        if self._patches is None:
            self._patches = np.load(self.patch_file, mmap_mode="r")
        return self._patches

    def __len__(self):
        return len(self.arrays.identifiers)

    def __getitem__(self, index):
        # torch.tensor deliberately copies the read-only memmap row.
        patches = torch.tensor(self.patches[self.source_indices[index]])
        return (
            torch.from_numpy(self.arrays.relational[index]),
            torch.from_numpy(self.arrays.edges[index]),
            patches,
            torch.from_numpy(self.arrays.truth[index]),
            int(index),
        )


class MIICrossAttentionGAE(nn.Module):
    """Exact paper cross-attention: GAE query -> frozen MII patch tokens."""

    def __init__(
        self,
        relational_dim: int,
        number_of_edges: int,
        output_dim: int,
        patch_dim: int = PATCH_DIM,
        attention_dim: int = ATTENTION_DIM,
        heads: int = ATTENTION_HEADS,
    ):
        super().__init__()
        self.geometry_encoder = GeometryAttentionEncoder(
            relational_dim, number_of_edges, output_dim=64
        )
        self.patch_projection = nn.Linear(patch_dim, attention_dim)
        self.query_projection = nn.Linear(64, attention_dim)
        self.cross_attention = nn.MultiheadAttention(
            attention_dim, heads, dropout=0.1, batch_first=True
        )
        # This is Eq. (attn_fusion), not global-image concat fusion: the image
        # contribution is first selected by geometry-conditioned attention.
        self.head = nn.Sequential(
            nn.LayerNorm(attention_dim + 64),
            nn.Linear(attention_dim + 64, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim),
        )

    def forward(
        self,
        relational: torch.Tensor,
        edges: torch.Tensor,
        patches: torch.Tensor,
        return_attention: bool = False,
    ):
        geometry = self.geometry_encoder(relational, edges)
        normalized_patches = nn.functional.normalize(patches.float(), dim=-1)
        keys_values = self.patch_projection(normalized_patches)
        query = self.query_projection(geometry).unsqueeze(1)
        attended, raw_weights = self.cross_attention(
            query,
            keys_values,
            keys_values,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        prediction = self.head(
            torch.cat((attended.squeeze(1), geometry), dim=1)
        )
        if not return_attention:
            return prediction
        weights = raw_weights.mean(dim=1).squeeze(1)
        return prediction, weights


def make_loaders(
    task: str,
    arrays: dict,
    output: Path,
    permutations: dict,
    condition: str,
    batch_size: int,
    workers: int,
    seed: int,
    device: torch.device,
):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    datasets = {}
    for split in SPLITS:
        indices = (
            permutations[split]
            if condition == "shuffled"
            else np.arange(len(arrays[split].identifiers))
        )
        datasets[split] = PatchFusionDataset(
            arrays[split], patch_path(output, task, split), indices
        )
    generator = torch.Generator().manual_seed(seed)
    return {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=device.type == "cuda",
            generator=generator if split == "train" else None,
        )
        for split in SPLITS
    }


def transform_arrays(transform: dict):
    return (
        np.asarray(transform["mean_deg"], dtype=np.float32),
        np.asarray(transform["sd_deg"], dtype=np.float32),
    )


def evaluate(model, loader, device, transform):
    model.eval()
    mean, standard_deviation = transform_arrays(transform)
    predictions, truths, indices = [], [], []
    with torch.inference_mode():
        for relational, edges, patches, target, index in loader:
            output = model(
                relational.to(device, non_blocking=True),
                edges.to(device, non_blocking=True),
                patches.to(device, non_blocking=True),
            )
            predictions.append(
                output.float().cpu().numpy() * standard_deviation + mean
            )
            truths.append(target.numpy())
            indices.append(index.numpy())
    order = np.concatenate(indices).argsort()
    return np.concatenate(truths)[order], np.concatenate(predictions)[order]


def train_trial(
    adapter,
    arrays,
    task: str,
    output: Path,
    permutations,
    condition: str,
    seed: int,
    hyperparameters: dict,
    trial_index: int,
    device: torch.device,
    checkpoint: Path,
):
    global_fusion.seed_everything(seed)
    model = MIICrossAttentionGAE(
        arrays["train"].relational.shape[1],
        arrays["train"].edges.shape[1],
        len(adapter.targets),
    ).to(device)
    loaders = make_loaders(
        task, arrays, output, permutations, condition,
        int(adapter.config["batch_size"]), int(adapter.config["num_workers"]),
        seed, device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hyperparameters["lr"]),
        weight_decay=float(hyperparameters["weight_decay"]),
    )
    max_epochs = int(adapter.config["max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs
    )
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
        for relational, edges, patches, target, _ in loaders["train"]:
            target = target.to(device, non_blocking=True)
            normalized_target = (target - target_mean_tensor) / target_sd_tensor
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                prediction = model(
                    relational.to(device, non_blocking=True),
                    edges.to(device, non_blocking=True),
                    patches.to(device, non_blocking=True),
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
        score = float(np.mean([
            validation[target]["mae"] for target in adapter.targets
        ]))
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
                "model": model.state_dict(),
                "task": task,
                "condition": condition,
                "seed": seed,
                "trial_index": trial_index,
                "hyperparameters": hyperparameters,
                "best_epoch": best_epoch,
                "validation_selection_score": best_score,
                "target_transform": transform,
                "fusion": "paper cross-attention",
            }, checkpoint)
        else:
            stale_epochs += 1
        print(json.dumps({
            "task": task,
            "condition": condition,
            "seed": seed,
            "trial": trial_index,
            "epoch": epoch,
            "validation_selection_score": score,
            "best_epoch": best_epoch,
        }), flush=True)
        if stale_epochs >= int(adapter.config["patience"]):
            break
    return {
        "trial_index": trial_index,
        "hyperparameters": hyperparameters,
        "best_epoch": best_epoch,
        "validation_selection_score": best_score,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint),
        "target_transform": transform,
        "epoch_history": history,
        "elapsed_seconds": time.time() - started,
    }


def evaluate_selected(
    adapter,
    arrays,
    task: str,
    output: Path,
    permutations,
    condition: str,
    seed: int,
    selected: dict,
    device: torch.device,
):
    checkpoint = torch.load(
        selected["checkpoint"], map_location=device, weights_only=False
    )
    model = MIICrossAttentionGAE(
        arrays["train"].relational.shape[1],
        arrays["train"].edges.shape[1],
        len(adapter.targets),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    transform = adapter.transform(arrays)
    if checkpoint["target_transform"] != transform:
        raise ValueError("target transform changed after training")
    loaders = make_loaders(
        task, arrays, output, permutations, condition,
        int(adapter.config["batch_size"]), int(adapter.config["num_workers"]),
        seed, device,
    )
    result = {}
    for split in ("val", "test"):
        truth, prediction = evaluate(model, loaders[split], device, transform)
        result[split] = adapter.metrics(truth, prediction)
        path = output / task / condition / f"seed_{seed}_{split}_predictions.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            identifiers=arrays[split].identifiers,
            truth=truth,
            prediction=prediction,
            absolute_error=np.abs(prediction - truth),
            image_source_indices=(
                permutations[split]
                if condition == "shuffled"
                else np.arange(len(arrays[split].identifiers))
            ),
            fusion=np.asarray("paper cross-attention"),
        )
        result[split]["prediction_file"] = str(path)
    return result


def existing_global_concat(task: str) -> dict:
    path = (
        REPO_ROOT / "outputs" / "benchmarks" /
        "predicted_landmark_frozen_vision_fusion_medimageinsight" /
        "results.json"
    )
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    result = json.loads(path.read_text())
    return {
        "status": "loaded",
        "path": str(path),
        "conditions": {
            condition: result["tasks"][task]["conditions"][condition][
                "test_summary_across_pipeline_seeds"
            ]
            for condition in CONDITIONS
        },
    }


def write_compact(results: dict, output: Path) -> Path:
    compact = {
        "fusion": "paper cross-attention; GAE is one query over 225 frozen MII tokens",
        "conditions": {
            "matched": "same-case image patches",
            "shuffled": "real-image patches deranged within split",
        },
        "tasks": {},
    }
    for task, task_result in results["tasks"].items():
        compact["tasks"][task] = {
            condition: task_result["conditions"][condition][
                "test_summary_across_pipeline_seeds"
            ]
            for condition in CONDITIONS
            if condition in task_result["conditions"]
        }
        compact["tasks"][task]["existing_global_concat"] = (
            task_result["existing_global_concat"]
        )
    path = output / "compact_comparison.json"
    global_fusion.write_json(path, compact)
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--foot-config", type=Path, default=global_fusion.DEFAULT_FOOT_CONFIG
    )
    parser.add_argument(
        "--cobb-config", type=Path, default=global_fusion.DEFAULT_COBB_CONFIG
    )
    parser.add_argument("--extract-batch-size", type=int, default=2)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if tuple(sorted(args.seeds)) != SEEDS or len(set(args.seeds)) != 3:
        raise ValueError("formal paired seeds must be 17, 42, and 73")
    device = global_fusion.resolve_device(args.device)
    assets = global_fusion.validate_encoder_assets("medimageinsight")
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.json"
    if results_path.is_file() and not (args.force or args.dry_run or args.extract_only):
        raise FileExistsError(f"{results_path} exists; pass --force for a fresh run")

    records_by_task = {
        task: global_fusion.records_for_task(task) for task in args.tasks
    }
    extraction = {}
    if not args.dry_run:
        for task in args.tasks:
            extraction[task] = extract_mii_patches(
                task, args.output, records_by_task[task],
                args.extract_batch_size, device,
            )
        if args.extract_only:
            print(json.dumps({
                "status": "feature_extraction_complete",
                "extraction": extraction,
            }, indent=2))
            return

    results = {
        "protocol_version": "predicted-landmark-mii-cross-attention-v1",
        "fusion": {
            "type": "paper cross-attention",
            "query": "unchanged 64-d GAE geometry token",
            "keys_values": "225 frozen MII pre-pooling spatial tokens",
            "patch_dim": PATCH_DIM,
            "attention_dim": ATTENTION_DIM,
            "heads": ATTENTION_HEADS,
            "final_fusion": "[attended visual token; original geometry token], per paper Eq. attn_fusion",
            "not_global_concat": True,
        },
        "image_encoder": "MedImageInsight-v1.0.0",
        "image_encoder_trainable": False,
        "encoder_asset_audit": assets,
        "geometry_encoder": "unchanged GeometryAttentionEncoder",
        "geometry_inputs": {
            "foot": "predicted L6; train is five-fold patient OOF",
            "cobb": "predicted L68 -> Auto-EV predicted L8; train is five-fold image OOF",
        },
        "no_gt_landmarks_or_geometry": True,
        "paired_pipeline_seeds": list(args.seeds),
        "extraction": extraction,
        "tasks": {},
    }

    for task in args.tasks:
        adapter = global_fusion.load_adapter(
            task, args.foot_config, args.cobb_config
        )
        records = records_by_task[task]
        arrays_by_seed, audits, shuffle_audits = {}, {}, {}
        for seed in args.seeds:
            arrays, audit = adapter.benchmark.load_data(
                REPO_ROOT, adapter.config, seed
            )
            arrays_by_seed[seed], audits[str(seed)] = arrays, audit
            permutations = global_fusion.split_permutations(task, arrays, seed)
            shuffle_audits[str(seed)] = global_fusion.permutation_audit(
                task, arrays, records, permutations, seed
            )
        sample = arrays_by_seed[args.seeds[0]]["train"]
        smoke_model = MIICrossAttentionGAE(
            sample.relational.shape[1], sample.edges.shape[1], len(adapter.targets)
        )
        smoke_output, smoke_weights = smoke_model(
            torch.from_numpy(sample.relational[:2]),
            torch.from_numpy(sample.edges[:2]),
            torch.randn(2, PATCH_COUNT, PATCH_DIM),
            return_attention=True,
        )
        if smoke_output.shape != (2, len(adapter.targets)):
            raise ValueError("cross-attention output smoke test failed")
        if smoke_weights.shape != (2, PATCH_COUNT):
            raise ValueError("cross-attention weight smoke test failed")
        smoke_output.sum().backward()
        task_result = {
            "targets": list(adapter.targets),
            "baseline_gae_only": global_fusion.baseline_reference(task),
            "existing_global_concat": existing_global_concat(task),
            "data_audit_by_pipeline_seed": audits,
            "shuffle_audit_by_pipeline_seed": shuffle_audits,
            "architecture": {
                "parameters": sum(p.numel() for p in smoke_model.parameters()),
                "geometry_token_dim": 64,
                "patch_count": PATCH_COUNT,
                "patch_dim": PATCH_DIM,
                "attention_dim": ATTENTION_DIM,
                "attention_heads": ATTENTION_HEADS,
                "output_dim": len(adapter.targets),
                "forward_backward_smoke_test": "passed",
            },
            "training_protocol": {
                key: adapter.config[key] for key in (
                    "batch_size", "num_workers", "amp", "max_epochs",
                    "patience", "optimizer", "scheduler",
                    "target_standardization", "loss", "gradient_clip_norm",
                    "checkpoint_selection", "search_space",
                )
            },
            "conditions": {},
        }
        results["tasks"][task] = task_result
        print(json.dumps({
            "status": "audit_passed",
            "task": task,
            "device": str(device),
            "paired_detector_downstream_seeds": [
                [seed, seed] for seed in args.seeds
            ],
            "predicted_landmark_inputs": {
                str(seed): audits[str(seed)]["paths"] for seed in args.seeds
            },
            "patch_cache": {
                split: str(patch_path(args.output, task, split))
                for split in SPLITS
            },
            "patch_cache_complete": (
                False if args.dry_run
                else validate_patch_cache(args.output, task, records)
            ),
            "architecture": task_result["architecture"],
            "shuffle_no_self": True,
            "no_gt_landmarks_or_geometry": True,
        }, indent=2), flush=True)
        if args.dry_run:
            continue

        for condition in args.conditions:
            runs = {}
            for seed in args.seeds:
                arrays = arrays_by_seed[seed]
                permutations = global_fusion.split_permutations(task, arrays, seed)
                trials = []
                for trial_index, hyperparameters in enumerate(
                    adapter.config["search_space"]
                ):
                    checkpoint = (
                        args.output / task / condition / "checkpoints" /
                        f"pipeline_seed_{seed}_trial_{trial_index}.pt"
                    )
                    trials.append(train_trial(
                        adapter, arrays, task, args.output, permutations,
                        condition, seed, hyperparameters, trial_index,
                        device, checkpoint,
                    ))
                selected = min(
                    trials, key=lambda row: row["validation_selection_score"]
                )
                evaluation = evaluate_selected(
                    adapter, arrays, task, args.output, permutations,
                    condition, seed, selected, device,
                )
                runs[str(seed)] = {
                    "pipeline_seed": seed,
                    "detector_seed": seed,
                    "downstream_seed": seed,
                    "all_validation_trials": trials,
                    "selected_trial": selected,
                    **evaluation,
                }
                task_result["conditions"].setdefault(condition, {})["runs"] = runs
                global_fusion.write_json(results_path, results)
            task_result["conditions"][condition][
                "test_summary_across_pipeline_seeds"
            ] = adapter.benchmark.summarize_seeds(runs)
            global_fusion.write_json(results_path, results)

    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_complete",
            "results_would_be": str(results_path),
        }, indent=2))
        return
    global_fusion.write_json(results_path, results)
    compact_path = write_compact(results, args.output)
    print(json.dumps({
        "status": "complete",
        "results": str(results_path),
        "compact_comparison": str(compact_path),
    }, indent=2))


if __name__ == "__main__":
    main()
