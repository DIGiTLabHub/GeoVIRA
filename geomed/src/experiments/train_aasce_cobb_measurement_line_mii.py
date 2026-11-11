#!/usr/bin/env python3
"""Train measurement-aligned Cobb regression with case-matched visual features.

Auto-EV geometry and frozen MedImageInsight features define the measurement-line
interface. The feature-map shuffle control has a separate training runner.
"""

from __future__ import annotations

import argparse
import base64
import csv
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
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "geomed" / "src"))
sys.path.insert(0, str(SCRIPT_ROOT))

import cobb_geometry as selector  # noqa: E402
import train_hvangle_axis_aligned_mii as spatial  # noqa: E402
import train_joint_geometry_models as benchmark  # noqa: E402
from geomed.data import load_aasce_records  # noqa: E402
from geomed.models import GeometryAttentionEncoder  # noqa: E402


CONDITIONS = ("cobb_axis_aligned_matched",)
SEEDS = (17, 42, 73)
SPLITS = ("train", "val", "test")
LINE_NAMES = ("upper", "lower")
ENDPLATE_EDGE_INDICES = ((0, 13), (22, 27))
AXIS_SAMPLES = 16
ALIGNMENT_DIM = 32
DEFAULT_ALPHA = 0.5
MINIMUM_VALID_SAMPLES = 4
DEFAULT_CONFIG = SCRIPT_ROOT / "aasce_geometry_coordstem_standardized_config.json"
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs" / "benchmarks" /
    "aasce_cobb_measurement_line_mii"
)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def exact_reference_lines(selected: np.ndarray) -> np.ndarray:
    """Return lines exactly equivalent to the repository Cobb directions.

    For corners [p0,p1,p2,p3], the repository uses
    .5*((p1-p0)+(p3-p2)), which is exactly the direction from midpoint(p0,p2)
    to midpoint(p1,p3).  This is not replaced by either individual endplate.
    """
    points = np.asarray(selected, dtype=np.float32)
    if points.shape != (8, 2) or not np.isfinite(points).all():
        raise ValueError("Auto-EV selected geometry must be finite L8")
    vertebrae = points.reshape(2, 4, 2)
    return np.stack([
        np.stack(((v[0] + v[2]) * 0.5, (v[1] + v[3]) * 0.5))
        for v in vertebrae
    ]).astype(np.float32)


def cobb_from_lines(lines: np.ndarray) -> float:
    directions = np.asarray(lines, dtype=np.float64)[:, 1] - lines[:, 0]
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= selector.DEGENERATE_DIRECTION_EPSILON_PX):
        return 0.0
    cosine = float(np.clip(
        np.dot(directions[0], directions[1]) / (norms[0] * norms[1]), -1.0, 1.0
    ))
    return float(np.degrees(np.arccos(cosine)))


def selected_lines_by_seed(arrays_by_seed: dict, audits: dict, records: dict):
    all_lines, verification = {}, {}
    for seed in SEEDS:
        all_lines[seed] = {}
        verification[str(seed)] = {}
        for split in SPLITS:
            archive_path = Path(audits[str(seed)]["paths"][split])
            with np.load(archive_path) as archive:
                identifiers = archive["identifiers"].astype(str)
                predicted = np.asarray(archive["predicted_points"], np.float32)
            if not np.array_equal(identifiers, arrays_by_seed[seed][split].identifiers):
                raise ValueError(f"{seed}/{split} L68 identifier order changed")
            lines, reconstructed = [], []
            for points, record in zip(predicted, records[split]):
                with Image.open(record.image_path) as image:
                    image_size = image.size
                angle, _, _, selected, _ = selector.select_primary_cobb(
                    points, image_size, return_diagnostics=True
                )
                pair = exact_reference_lines(selected)
                rebuilt = cobb_from_lines(pair)
                if abs(rebuilt - angle) > 1e-4:
                    raise RuntimeError(
                        f"{seed}/{split}/{record.identifier}: extracted-line Cobb "
                        f"{rebuilt:.8f} != selector Cobb {angle:.8f}"
                    )
                lines.append(pair)
                reconstructed.append(rebuilt)
            lines = np.asarray(lines, np.float32)
            reconstructed = np.asarray(reconstructed, np.float32)
            error = float(np.max(np.abs(
                reconstructed - arrays_by_seed[seed][split].analytic[:, 0]
            )))
            if error > 1e-4:
                raise RuntimeError(
                    f"{seed}/{split}: line reconstruction differs by {error:.8g}°"
                )
            all_lines[seed][split] = lines
            verification[str(seed)][split] = {
                "n": len(lines), "maximum_absolute_error_deg": error
            }
    return all_lines, verification


def sample_lines(feature_tokens: torch.Tensor, spatial_size: tuple[int, int],
                 lines_original: np.ndarray, mappings: np.ndarray):
    """Mean-pool frozen features along the two predicted reference lines."""
    batch, _, channels = feature_tokens.shape
    height, width = spatial_size
    feature_map = feature_tokens.transpose(1, 2).reshape(batch, channels, height, width)
    lines = torch.as_tensor(lines_original, dtype=torch.float32, device=feature_map.device).clone()
    mapping = torch.as_tensor(mappings, dtype=torch.float32, device=feature_map.device)
    lines[..., 0] = lines[..., 0] * mapping[:, None, None, 0] + mapping[:, None, None, 2]
    lines[..., 1] = lines[..., 1] * mapping[:, None, None, 1] + mapping[:, None, None, 3]
    length = torch.linalg.vector_norm(lines[:, :, 1] - lines[:, :, 0], dim=-1)
    nondegenerate = length > selector.DEGENERATE_DIRECTION_EPSILON_PX
    fractions = torch.linspace(0.0, 1.0, AXIS_SAMPLES, device=feature_map.device)
    samples = (lines[:, :, 0, None] * (1.0 - fractions[None, None, :, None])
               + lines[:, :, 1, None] * fractions[None, None, :, None])
    valid = torch.logical_and(samples >= 0.0, samples < float(spatial.LETTERBOX_SIZE)).all(dim=-1)
    valid_count = valid.sum(dim=-1)
    reliable = nondegenerate & (valid_count >= MINIMUM_VALID_SAMPLES)
    diagnostic_count = torch.where(nondegenerate, valid_count, torch.zeros_like(valid_count))
    safe_samples = torch.where(valid[..., None], samples, torch.zeros_like(samples))
    grid = torch.empty_like(safe_samples)
    grid[..., 0] = 2.0 * (safe_samples[..., 0] + 0.5) / spatial.LETTERBOX_SIZE - 1.0
    grid[..., 1] = 2.0 * (safe_samples[..., 1] + 0.5) / spatial.LETTERBOX_SIZE - 1.0
    sampled = F.grid_sample(feature_map.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    weights = valid[:, None].to(sampled.dtype)
    pooled = (sampled * weights).sum(dim=-1) / valid_count.clamp_min(1)[:, None]
    pooled = torch.where(reliable[..., None], pooled.transpose(1, 2), torch.zeros_like(pooled.transpose(1, 2)))
    diagnostics = {
        "line_length_letterbox_px": length.cpu().numpy(),
        "plus_valid_count": valid_count.cpu().numpy(),
        "minus_valid_count": np.full(tuple(length.shape), -1, dtype=np.int64),
        "selected_direction": torch.zeros_like(length).cpu().numpy(),
        "selected_valid_count": diagnostic_count.cpu().numpy(),
        "selected_valid_fraction": (diagnostic_count.float() / AXIS_SAMPLES).cpu().numpy(),
        "invalid": (~reliable).cpu().numpy(),
    }
    return pooled, diagnostics


def feature_path(output: Path, condition: str, seed: int, split: str) -> Path:
    return output / "line_features" / condition / f"seed_{seed}" / f"{split}.npz"


def write_diagnostics(path: Path, rows: list[dict]) -> None:
    fields = [
        "condition", "detector_seed", "split", "identifier", "line", "alpha",
        "line_length_letterbox_px", "plus_valid_count", "minus_valid_count",
        "selected_direction", "selected_valid_count", "selected_valid_fraction",
        "invalid",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def summarize_diagnostics(rows: list[dict]) -> dict:
    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        for split in SPLITS:
            subset = [r for r in rows if r["condition"] == condition and r["split"] == split]
            fractions = np.asarray([r["selected_valid_fraction"] for r in subset], np.float64)
            invalid = np.asarray([r["invalid"] for r in subset], bool)
            summary[condition][split] = {
                "number_of_lines": len(subset),
                "all_16_valid_percent": float(100.0 * np.mean(fractions == 1.0)),
                "neutral_token_percent": float(100.0 * invalid.mean()),
                "mean_valid_sampling_fraction": float(fractions.mean()),
                "minimum_valid_sampling_fraction": float(fractions.min()),
                "by_detector_seed": {},
            }
            for seed in SEEDS:
                seed_rows = [r for r in subset if int(r["detector_seed"]) == seed]
                seed_fractions = np.asarray(
                    [r["selected_valid_fraction"] for r in seed_rows], np.float64
                )
                seed_invalid = np.asarray([r["invalid"] for r in seed_rows], bool)
                summary[condition][split]["by_detector_seed"][str(seed)] = {
                    "number_of_cases": len(seed_rows) // len(LINE_NAMES),
                    "number_of_lines": len(seed_rows),
                    "all_16_valid_percent": float(
                        100.0 * np.mean(seed_fractions == 1.0)
                    ),
                    "neutral_token_percent": float(100.0 * seed_invalid.mean()),
                    "mean_valid_sampling_fraction": float(seed_fractions.mean()),
                    "minimum_valid_sampling_fraction": float(seed_fractions.min()),
                }
    return summary


def extract_features(output: Path, arrays_by_seed: dict, audits: dict,
                     records: dict, lines_by_seed: dict, batch_size: int,
                     device: torch.device, alpha: float, verification: dict):
    diagnostics_csv = output / "sampling_diagnostics.csv"
    summary_json = output / "sampling_summary.json"
    complete = diagnostics_csv.is_file() and summary_json.is_file() and all(
        feature_path(output, condition, seed, split).is_file()
        for condition in CONDITIONS for seed in SEEDS for split in SPLITS
    )
    if complete:
        for condition in CONDITIONS:
            for seed in SEEDS:
                load_features(output, condition, arrays_by_seed[seed], seed, alpha)
        return {"status": "reused", **json.loads(summary_json.read_text())}

    sys.path.insert(0, str(REPO_ROOT / "MedImageInsights"))
    from medimageinsightmodel import MedImageInsight
    if device.type == "cuda": torch.cuda.set_device(device)
    model_dir = REPO_ROOT / "MedImageInsights" / "2024.09.27"
    wrapper = MedImageInsight(str(model_dir), "medimageinsigt-v1.0.0.pt", "language_model.pth")
    wrapper.load_model()
    if tuple(wrapper.opt["IMAGE_ENCODER"]["IMAGE_SIZE"]) != (480, 480):
        raise ValueError("MedImageInsight input size changed")
    if bool(wrapper.opt["TEST"]["CENTER_CROP"]):
        raise ValueError("MedImageInsight center crop is incompatible with mapping")
    wrapper.model.to(device).eval()
    for parameter in wrapper.model.parameters(): parameter.requires_grad_(False)
    rows, official_error, spatial_shape = [], None, None
    for split in SPLITS:
        collected = {
            condition: {seed: [] for seed in SEEDS} for condition in CONDITIONS
        }
        for start in range(0, len(records[split]), batch_size):
            batch_records = records[split][start:start + batch_size]
            tensors, mappings = [], []
            for record in batch_records:
                with Image.open(record.image_path) as image:
                    canvas, mapping = spatial.letterbox_with_mapping(image.convert("RGB"))
                buffer = io.BytesIO(); canvas.save(buffer, format="JPEG", quality=95)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
                tensors.append(wrapper.preprocess(wrapper.decode_base64_image(encoded)))
                mappings.append(mapping)
            pixels = torch.stack(tensors).to(device)
            with torch.inference_mode():
                frozen, shape = spatial.spatial_forward(wrapper.model.image_encoder, pixels)
                if official_error is None:
                    official_error = spatial.verify_prepool_tokens(wrapper.model, pixels, frozen)
                    spatial_shape = [shape[0], shape[1], frozen.shape[-1]]
                for condition in CONDITIONS:
                    for seed in SEEDS:
                        tokens, diagnostic = sample_lines(
                            frozen, shape,
                            lines_by_seed[seed][split][start:start + len(batch_records)],
                            np.asarray(mappings, np.float32),
                        )
                        collected[condition][seed].append(tokens.cpu().numpy().astype(np.float32))
                        for local_index, record in enumerate(batch_records):
                            for line_index, line_name in enumerate(LINE_NAMES):
                                rows.append({
                                    "condition": condition, "detector_seed": seed,
                                    "split": split, "identifier": record.identifier,
                                    "line": line_name, "alpha": alpha,
                                    "line_length_letterbox_px": float(diagnostic["line_length_letterbox_px"][local_index, line_index]),
                                    "plus_valid_count": int(diagnostic["plus_valid_count"][local_index, line_index]),
                                    "minus_valid_count": int(diagnostic["minus_valid_count"][local_index, line_index]),
                                    "selected_direction": int(diagnostic["selected_direction"][local_index, line_index]),
                                    "selected_valid_count": int(diagnostic["selected_valid_count"][local_index, line_index]),
                                    "selected_valid_fraction": float(diagnostic["selected_valid_fraction"][local_index, line_index]),
                                    "invalid": bool(diagnostic["invalid"][local_index, line_index]),
                                })
            print(f"[{split}] {min(start+batch_size,len(records[split]))}/{len(records[split])}", flush=True)
        identifiers = np.asarray([r.identifier for r in records[split]])
        for condition in CONDITIONS:
            for seed in SEEDS:
                tokens = np.concatenate(collected[condition][seed])
                path = feature_path(output, condition, seed, split)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path, identifiers=identifiers, line_tokens=tokens,
                    condition=np.asarray(condition), detector_seed=np.asarray(seed),
                    alpha=np.asarray(alpha), roles=np.asarray(LINE_NAMES),
                    source=np.asarray("HIH predicted L68 -> deterministic Auto-EV -> exact predicted Cobb reference lines"),
                    spatial_shape=np.asarray(spatial_shape), axis_samples=np.asarray(AXIS_SAMPLES),
                    minimum_valid_samples=np.asarray(MINIMUM_VALID_SAMPLES),
                    official_global_max_abs_error=np.asarray(official_error),
                )
    write_diagnostics(diagnostics_csv, rows)
    summary = {
        "spatial_shape": spatial_shape,
        "official_global_max_abs_error": official_error,
        "no_gt_geometry_used": True,
        "analytic_line_reconstruction": verification,
        "sampling": summarize_diagnostics(rows),
        "diagnostics_csv": str(diagnostics_csv),
    }
    write_json(summary_json, summary)
    del wrapper
    if device.type == "cuda": torch.cuda.empty_cache()
    return {"status": "extracted", **summary}


def load_features(output: Path, condition: str, arrays: dict, seed: int, alpha: float):
    result = {}
    for split in SPLITS:
        path = feature_path(output, condition, seed, split)
        with np.load(path) as archive:
            ids = archive["identifiers"].astype(str)
            tokens = np.asarray(archive["line_tokens"], np.float32)
            metadata_ok = (
                str(archive["condition"].item()) == condition
                and int(archive["detector_seed"]) == seed
                and math.isclose(float(archive["alpha"]), alpha, rel_tol=0, abs_tol=1e-12)
                and tuple(archive["roles"].astype(str)) == LINE_NAMES
                and int(archive["axis_samples"]) == AXIS_SAMPLES
                and int(archive["minimum_valid_samples"]) == MINIMUM_VALID_SAMPLES
                and float(archive["official_global_max_abs_error"]) <= 2e-5
            )
        if not np.array_equal(ids, arrays[split].identifiers) or not metadata_ok:
            raise ValueError(f"stale line feature cache: {path}")
        if tokens.shape != (len(ids), 2, spatial.SPATIAL_CHANNELS):
            raise ValueError(f"invalid line-token shape: {tokens.shape}")
        result[split] = tokens
    return result


class LineDataset(Dataset):
    def __init__(self, arrays, features):
        self.arrays = arrays; self.features = torch.from_numpy(features)
    def __len__(self): return len(self.arrays.identifiers)
    def __getitem__(self, index):
        return (torch.from_numpy(self.arrays.relational[index]),
                torch.from_numpy(self.arrays.edges[index]), self.features[index],
                torch.from_numpy(self.arrays.truth[index]), index)


class CobbLineAlignedGAE(nn.Module):
    def __init__(self, relational_dim: int, number_of_edges: int):
        super().__init__()
        self.gae = GeometryAttentionEncoder(relational_dim, number_of_edges, output_dim=64)
        self.geo_projector = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, ALIGNMENT_DIM), nn.GELU())
        self.visual_projector = nn.Sequential(nn.LayerNorm(spatial.SPATIAL_CHANNELS), nn.Linear(spatial.SPATIAL_CHANNELS, ALIGNMENT_DIM), nn.GELU())
        self.alignment = nn.Sequential(nn.LayerNorm(2*ALIGNMENT_DIM), nn.Linear(2*ALIGNMENT_DIM, ALIGNMENT_DIM), nn.GELU())
        self.head = benchmark.JointRegressionHead(64 + 2*ALIGNMENT_DIM)
    def forward(self, relational, edges, visual):
        geometry = self.gae(relational, edges)
        primitive = torch.stack([
            edges[:, list(pair)].mean(dim=1) for pair in ENDPLATE_EDGE_INDICES
        ], dim=1)
        h = self.alignment(torch.cat((self.geo_projector(primitive), self.visual_projector(visual)), dim=-1))
        return self.head(torch.cat((geometry, h.flatten(1)), dim=1))


def loaders(arrays, features, config, seed, device):
    generator = torch.Generator().manual_seed(seed)
    return {split: DataLoader(LineDataset(arrays[split], features[split]),
        batch_size=int(config["batch_size"]), shuffle=split=="train",
        num_workers=int(config["num_workers"]), pin_memory=device.type=="cuda",
        generator=generator if split=="train" else None) for split in SPLITS}


def evaluate(model, loader, device, transform):
    model.eval(); mean=np.asarray(transform["mean_deg"],np.float32); sd=np.asarray(transform["sd_deg"],np.float32)
    truths=[]; predictions=[]; indices=[]
    with torch.inference_mode():
        for relational,edges,visual,target,index in loader:
            out=model(relational.to(device),edges.to(device),visual.to(device))
            predictions.append(out.float().cpu().numpy()*sd+mean); truths.append(target.numpy()); indices.append(index.numpy())
    order=np.concatenate(indices).argsort()
    return np.concatenate(truths)[order],np.concatenate(predictions)[order]


def train_trial(condition, seed, hp, trial, arrays, features, config, device, checkpoint):
    seed_everything(seed); model=CobbLineAlignedGAE(arrays["train"].relational.shape[1],arrays["train"].edges.shape[1]).to(device)
    data=loaders(arrays,features,config,seed,device); optimizer=torch.optim.AdamW(model.parameters(),lr=float(hp["lr"]),weight_decay=float(hp["weight_decay"]))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=int(config["max_epochs"])); loss_fn=nn.HuberLoss(delta=float(config["loss"]["delta"]),reduction=config["loss"]["reduction"])
    amp=bool(config["amp"] and device.type=="cuda"); scaler=torch.amp.GradScaler("cuda",enabled=amp); transform=benchmark.target_transform(arrays,config)
    mean=torch.tensor(transform["mean_deg"],device=device); sd=torch.tensor(transform["sd_deg"],device=device); tolerance={"Cobb":float(config["metrics"]["tolerance_deg"]["Cobb"])}
    best=float("inf"); best_epoch=0; stale=0; history=[]; started=time.time(); checkpoint.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,int(config["max_epochs"])+1):
        model.train(); running=0.0
        for relational,edges,visual,target,_ in data["train"]:
            target=target.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,enabled=amp):
                prediction=model(relational.to(device),edges.to(device),visual.to(device)); loss=loss_fn(prediction,(target-mean)/sd)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),float(config["gradient_clip_norm"])); scaler.step(optimizer); scaler.update(); running+=float(loss.item())
        scheduler.step(); truth,pred=evaluate(model,data["val"],device,transform); metric=benchmark.metric_block(truth,pred,tolerance); score=float(metric["Cobb"]["mae"])
        history.append({"epoch":epoch,"train_loss_standardized":running/len(data["train"]),"validation_cobb_mae_deg":score})
        if score<best:
            best=score;best_epoch=epoch;stale=0;torch.save({"model":model.state_dict(),"condition":condition,"seed":seed,"trial_index":trial,"hyperparameters":hp,"best_epoch":epoch,"validation_selection_score":score,"target_transform":transform},checkpoint)
        else: stale+=1
        print(json.dumps({"condition":condition,"seed":seed,"trial":trial,"epoch":epoch,"val_cobb_mae":score,"best_epoch":best_epoch}),flush=True)
        if stale>=int(config["patience"]):break
    return {"trial_index":trial,"hyperparameters":hp,"best_epoch":best_epoch,"validation_selection_score":best,"parameter_count":sum(p.numel() for p in model.parameters()),"checkpoint":str(checkpoint),"target_transform":transform,"epoch_history":history,"elapsed_seconds":time.time()-started}


def evaluate_selected(condition,seed,selected,arrays,features,config,device,output):
    ckpt=torch.load(selected["checkpoint"],map_location=device,weights_only=False);model=CobbLineAlignedGAE(arrays["train"].relational.shape[1],arrays["train"].edges.shape[1]).to(device);model.load_state_dict(ckpt["model"])
    transform=benchmark.target_transform(arrays,config)
    if ckpt["target_transform"]!=transform:raise ValueError("target transform mismatch")
    data=loaders(arrays,features,config,seed,device); tolerance={"Cobb":float(config["metrics"]["tolerance_deg"]["Cobb"])};result={}
    for split in ("val","test"):
        truth,pred=evaluate(model,data[split],device,transform);result[split]=benchmark.metric_block(truth,pred,tolerance);path=output/condition/f"seed_{seed}_{split}_predictions.npz";path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,identifiers=arrays[split].identifiers,selected_pairs=arrays[split].selected_pairs,truth=truth,prediction=pred,absolute_error=np.abs(pred-truth));result[split]["prediction_file"]=str(path)
    return result


def existing_results():
    geometry=json.loads((REPO_ROOT/"outputs/benchmarks/joint_geometry_aasce_hih_oof_coordstem_standardized/results.json").read_text())
    global_result=json.loads((REPO_ROOT/"outputs/benchmarks/predicted_landmark_frozen_vision_fusion_medimageinsight/results.json").read_text())
    return {"GAE only":geometry["models"]["gae"]["test_summary_across_seeds"],"MII global shuffled":global_result["tasks"]["cobb"]["conditions"]["shuffled"]["test_summary_across_pipeline_seeds"],"MII global matched":global_result["tasks"]["cobb"]["conditions"]["matched"]["test_summary_across_pipeline_seeds"]}


def parse_args():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=DEFAULT_CONFIG);p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT);p.add_argument("--device",default="auto");p.add_argument("--extract-batch-size",type=int,default=2);p.add_argument("--extract-only",action="store_true");p.add_argument("--dry-run",action="store_true");return p.parse_args()


def main():
    args=parse_args();config=json.loads(args.config.read_text());device=spatial.resolve_device(args.device)
    if tuple(config["seeds"])!=SEEDS:raise ValueError("paired seeds must be 17,42,73")
    records=load_aasce_records(REPO_ROOT);arrays_by_seed={};audits={}
    for seed in SEEDS:arrays_by_seed[seed],audits[str(seed)]=benchmark.load_data(REPO_ROOT,config,seed)
    lines,verification=selected_lines_by_seed(arrays_by_seed,audits,records)
    sample=arrays_by_seed[17]["train"];model=CobbLineAlignedGAE(sample.relational.shape[1],sample.edges.shape[1]);out=model(torch.from_numpy(sample.relational[:2]),torch.from_numpy(sample.edges[:2]),torch.randn(2,2,spatial.SPATIAL_CHANNELS));out.sum().backward()
    audit={"conditions":list(CONDITIONS),"paired_seeds":list(SEEDS),"split_sizes":{s:len(records[s]) for s in SPLITS},"parameter_count":sum(p.numel() for p in model.parameters()),"output_shape":list(out.shape),"line_definition":"midpoint(p0,p2) -> midpoint(p1,p3), exactly matching repository averaged endplate direction","geometry_primitive":"mean of the two corresponding existing GAE endplate edge features","analytic_line_verification":verification,"no_gt_geometry_used":True}
    print(json.dumps({"status":"protocol_line_architecture_audit_passed",**audit},indent=2),flush=True)
    if args.dry_run:return
    result_path=args.output/"results.json";progress=args.output/"training_progress.json"
    if result_path.exists():raise FileExistsError(f"{result_path} exists; formal results are never overwritten")
    args.output.mkdir(parents=True,exist_ok=True);write_json(args.output/"experiment_config.json",{"alpha":DEFAULT_ALPHA,"minimum_valid_samples":MINIMUM_VALID_SAMPLES,"formal_config":config,"audit":audit})
    extraction=extract_features(args.output,arrays_by_seed,audits,records,lines,args.extract_batch_size,device,DEFAULT_ALPHA,verification)
    if args.extract_only:print(json.dumps({"status":"extraction_complete","output":str(args.output),"diagnostics":extraction},indent=2));return
    results={"protocol_version":"aasce-cobb-measurement-line-mii-v1","conditions":{},"architecture_audit":audit,"feature_diagnostics":extraction,"paired_pipeline_seeds":list(SEEDS),"checkpoint_selection":config["checkpoint_selection"],"existing_results":existing_results()};write_json(progress,results)
    for condition in CONDITIONS:
        runs={}
        for seed in SEEDS:
            arrays=arrays_by_seed[seed];features=load_features(args.output,condition,arrays,seed,DEFAULT_ALPHA);trials=[]
            for trial,hp in enumerate(config["search_space"]):trials.append(train_trial(condition,seed,hp,trial,arrays,features,config,device,args.output/condition/"checkpoints"/f"seed_{seed}_trial_{trial}.pt"))
            selected=min(trials,key=lambda x:x["validation_selection_score"]);evaluation=evaluate_selected(condition,seed,selected,arrays,features,config,device,args.output);runs[str(seed)]={"pipeline_seed":seed,"detector_seed":seed,"downstream_seed":seed,"all_validation_trials":trials,"selected_trial":selected,**evaluation};results["conditions"].setdefault(condition,{})["runs"]=runs;write_json(progress,results)
        results["conditions"][condition]["test_summary_across_pipeline_seeds"]=benchmark.summarize_seeds(runs);write_json(progress,results)
    comparison = {
        "GAE only": results["existing_results"]["GAE only"],
        "MII global shuffled": results["existing_results"]["MII global shuffled"],
        "MII global matched": results["existing_results"]["MII global matched"],
        "Cobb measurement-line aligned": results["conditions"][
            "cobb_axis_aligned_matched"
        ]["test_summary_across_pipeline_seeds"],
    }
    results["compact_comparison"] = comparison
    write_json(result_path,results)
    write_json(args.output/"compact_comparison.json",comparison)
    print(json.dumps({"status":"complete","per_seed":{c:{seed:runs["test"]["Cobb"] for seed,runs in results["conditions"][c]["runs"].items()} for c in CONDITIONS},"aggregate":{c:results["conditions"][c]["test_summary_across_pipeline_seeds"] for c in CONDITIONS},"comparison":comparison,"diagnostics":extraction,"paths":{"results":str(result_path),"comparison":str(args.output/"compact_comparison.json"),"sampling_summary":str(args.output/"sampling_summary.json"),"sampling_csv":str(args.output/"sampling_diagnostics.csv")}},indent=2))


if __name__=="__main__":main()
