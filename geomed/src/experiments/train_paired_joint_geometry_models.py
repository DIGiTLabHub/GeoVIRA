#!/usr/bin/env python3
"""Run matched detector/downstream seeds for the predicted-landmark benchmark.

Pipeline seed 17 uses detector-17 OOF/validation/test landmarks and downstream
seed 17; seeds 42 and 73 are paired analogously.  This makes the three-run
summary directly comparable to the existing HIH analytic table, whose repeated
runs vary the upstream detector seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("controlled_geometry_coordstem_standardized_config.json")
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs" / "benchmarks" / "joint_geometry_hih_oof_paired_coordstem_standardized"
)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_hvangle_joint_geometry_models as benchmark  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pipeline-seeds", nargs="+", type=int, default=[17, 42, 73])
    parser.add_argument(
        "--models", nargs="+", choices=benchmark.MODEL_NAMES,
        help="defaults to the model list declared in the selected config",
    )
    parser.add_argument(
        "--device", default="auto",
        help="PyTorch device such as auto, cpu, cuda, or cuda:0",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validation-only", action="store_true",
        help="train/select candidates and evaluate validation only; never evaluate test",
    )
    return parser.parse_args()


def analytic_summary(arrays_by_seed: dict[int, dict], config: dict,
                     split: str = "test") -> dict:
    tolerances = {
        target: float(config["metrics"]["tolerance_deg"][target])
        for target in benchmark.TARGETS
    }
    per_seed = {}
    for seed, arrays in arrays_by_seed.items():
        evaluated = arrays[split]
        per_seed[str(seed)] = benchmark.metric_block(
            evaluated.truth, evaluated.analytic, tolerances
        )
    summary = {}
    for target in benchmark.TARGETS:
        summary[target] = {}
        for metric in (
            "mae", "icc_2_1", "within_tolerance_percent", "bias",
            "loa_lower", "loa_upper",
        ):
            values = np.asarray([
                per_seed[str(seed)][target][metric] for seed in arrays_by_seed
            ], dtype=np.float64)
            summary[target][metric] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "per_seed": values.tolist(),
            }
    return {"runs": per_seed, "summary_across_pipeline_seeds": summary}


def audit_architectures(arrays, model_names, config):
    dataset = benchmark.GeometryDataset(arrays["train"])
    sample = dataset[0]
    inputs = [item.unsqueeze(0) for item in sample[:-1]]
    report = {}
    for model_name in model_names:
        model = benchmark.build_model(
            model_name, arrays["train"].relational.shape[1],
            arrays["train"].edges.shape[1],
        )
        output = model(inputs[0], inputs[1], inputs[2])
        if output.shape != (1, 2):
            raise ValueError(f"{model_name} is not a joint HVA/IMA model")
        transform = benchmark.target_transform(arrays, config)
        mean, standard_deviation = benchmark._transform_arrays(transform)
        normalized_target = (
            inputs[3] - torch.from_numpy(mean).unsqueeze(0)
        ) / torch.from_numpy(standard_deviation).unsqueeze(0)
        loss = nn.HuberLoss(delta=float(config["loss"]["delta"]))(
            output, normalized_target
        )
        loss.backward()
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise ValueError(f"{model_name} produced a non-finite gradient")
        report[model_name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "representation_dim": 64,
            "output_shape": [1, 2],
            "forward_backward_smoke_test": "passed",
        }
        if model_name in benchmark.GAE_FEATURE_ABLATIONS:
            report[model_name]["relational_features"] = dict(
                benchmark.GAE_FEATURE_ABLATIONS[model_name]
            )
    return report


def assert_common_outer_split(arrays_by_seed: dict[int, dict]) -> None:
    reference_seed = next(iter(arrays_by_seed))
    reference = arrays_by_seed[reference_seed]
    for seed, arrays in arrays_by_seed.items():
        for split in ("train", "val", "test"):
            if not np.array_equal(
                arrays[split].identifiers, reference[split].identifiers
            ):
                raise ValueError(
                    f"detector seed {seed} changes the frozen {split} identifiers"
                )
            if not np.array_equal(arrays[split].truth, reference[split].truth):
                raise ValueError(
                    f"detector seed {seed} changes the frozen {split} targets"
                )


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    model_names = args.models or list(config.get("models", benchmark.MODEL_NAMES))
    unknown_models = sorted(set(model_names) - set(benchmark.MODEL_NAMES))
    if unknown_models:
        raise ValueError(f"configuration requests unknown models: {unknown_models}")
    configured_seeds = [int(value) for value in config["seeds"]]
    if args.pipeline_seeds != configured_seeds:
        raise ValueError(
            "paired benchmark must use the predeclared pipeline seeds "
            f"{configured_seeds}, got {args.pipeline_seeds}"
        )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device.index} requested, but only "
                f"{torch.cuda.device_count()} device(s) are visible"
            )

    arrays_by_seed = {}
    data_audits = {}
    for seed in args.pipeline_seeds:
        arrays_by_seed[seed], data_audits[str(seed)] = benchmark.load_data(
            args.repo_root, config, detector_seed=seed
        )
    assert_common_outer_split(arrays_by_seed)
    architecture_audit = audit_architectures(
        arrays_by_seed[args.pipeline_seeds[0]], model_names, config
    )
    evaluation_split = "val" if args.validation_only else "test"
    analytic = analytic_summary(arrays_by_seed, config, split=evaluation_split)
    audit = {
        "status": "paired_data_and_architecture_audit_passed",
        "pairing": [
            {"pipeline_seed": seed, "detector_seed": seed, "downstream_seed": seed}
            for seed in args.pipeline_seeds
        ],
        "data_by_detector_seed": data_audits,
        "analytic": {"split": evaluation_split, **analytic},
        "target_transforms": {
            str(seed): benchmark.target_transform(arrays, config)
            for seed, arrays in arrays_by_seed.items()
        },
        "architectures": architecture_audit,
    }
    print(json.dumps(audit, indent=2), flush=True)
    if args.dry_run:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.json"
    results = {
        "protocol_version": config["protocol_version"],
        "task": "joint HVA/IMA regression from matched HIH predicted landmarks",
        "target_order": list(benchmark.TARGETS),
        "pipeline_seed_pairing": audit["pairing"],
        "training_protocol": {
            key: config[key] for key in (
                "batch_size", "num_workers", "amp", "max_epochs", "patience",
                "optimizer", "scheduler", "loss", "gradient_clip_norm",
                "search_space", "checkpoint_selection", "matched_reference",
            )
        },
        "target_transforms": audit["target_transforms"],
        "data_by_detector_seed": data_audits,
        "analytic": {"split": evaluation_split, **analytic},
        "architecture_audit": architecture_audit,
        "models": {},
    }
    selected_ablation_models = [
        name for name in model_names if name in benchmark.GAE_FEATURE_ABLATIONS
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
                name: benchmark.GAE_FEATURE_ABLATIONS[name]
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
    if "gae_optimization" in config:
        results["training_protocol"]["gae_optimization"] = config[
            "gae_optimization"
        ]
    results["training_protocol"]["evaluation_scope"] = (
        "validation_only" if args.validation_only else "validation_and_test"
    )
    benchmark.write_json(results_path, results)

    for model_name in model_names:
        model_root = args.output / model_name
        model_root.mkdir(parents=True, exist_ok=True)
        runs = {}
        for pipeline_seed in args.pipeline_seeds:
            arrays = arrays_by_seed[pipeline_seed]
            trials = []
            for trial_index, hyperparameters in enumerate(config["search_space"]):
                checkpoint = (
                    model_root /
                    f"pipeline_seed_{pipeline_seed}_trial_{trial_index}.pt"
                )
                trials.append(benchmark.train_trial(
                    model_name, pipeline_seed, hyperparameters, trial_index,
                    arrays, config, device, checkpoint,
                ))
            selected = min(
                trials, key=lambda item: item["validation_selection_score"]
            )
            evaluations = benchmark.evaluate_selected(
                model_name, pipeline_seed, selected, arrays, config,
                device, args.output,
                splits=("val",) if args.validation_only else ("val", "test"),
            )
            runs[str(pipeline_seed)] = {
                "pipeline_seed": pipeline_seed,
                "detector_seed": pipeline_seed,
                "downstream_seed": pipeline_seed,
                "all_validation_trials": trials,
                "selected_trial": selected,
                **evaluations,
            }
            results["models"].setdefault(model_name, {})["runs"] = runs
            benchmark.write_json(results_path, results)
        summary_split = "val" if args.validation_only else "test"
        results["models"][model_name][
            f"{summary_split}_summary_across_pipeline_seeds"
        ] = benchmark.summarize_seeds(runs, split=summary_split)
        benchmark.write_json(results_path, results)

    print(json.dumps({"status": "complete", "results": str(results_path)}, indent=2))


if __name__ == "__main__":
    main()
