# Training and Evaluation

Run all commands from the repository root after [preparing the data](DATA.md). Use `--device cuda:0` to select a GPU, or `--device cpu`. The default is `auto`.

## 1. Geometry-Only Models

Train RAGE and the configured coordinate/set/graph baselines:

```bash
# HVA and IMA: joint prediction
python geomed/src/experiments/train_paired_joint_geometry_models.py --device cuda:0

# Cobb: separate prediction
python geomed/src/experiments/train_joint_geometry_models.py --device cuda:0
```

Pass `--models gae` to train only RAGE. The default configs use standardized targets, Huber loss (δ = 1), AdamW, cosine scheduling, batch size 16 and at most 60 epochs. Learning rate and weight decay are selected on validation MAE. Detector and downstream seeds are paired: 17/17, 42/42, 73/73.

## 2. Geometry–Visual Interfaces

```bash
# GeoVIRA-Global
python geomed/src/experiments/train_predicted_landmark_frozen_vision_fusion.py \
  --encoder medimageinsight --tasks foot cobb --device cuda:0

# GeoVIRA-Latent
python geomed/src/experiments/train_predicted_landmark_mii_cross_attention.py \
  --tasks foot cobb --device cuda:0

# GeoVIRA-Aligned: Foot
python geomed/src/experiments/train_hvangle_axis_aligned_mii.py --device cuda:0

# GeoVIRA-Aligned: Cobb
python geomed/src/experiments/train_aasce_cobb_measurement_line_mii.py --device cuda:0
```

Global and Latent train matched and shuffled conditions separately. The two Aligned runners train the matched conditions. Feature caches and results are saved under `outputs/benchmarks/`.

## 3. Aligned Shuffled Controls

Run after the matched Aligned models and the Latent spatial-feature cache are available:

```bash
python geomed/src/experiments/train_localized_mii_image_shuffled.py \
  --task foot --shuffle-mode feature-map --device cuda:0

python geomed/src/experiments/train_localized_mii_image_shuffled.py \
  --task cobb --shuffle-mode feature-map --device cuda:0
```

The donor feature map is sampled with the recipient's predicted geometry. A fixed within-split derangement is used for each seed; foot donors belong to a different patient. `feature-map` is the default. `token-set` selects a different, auxiliary control.

## 4. Relational Ablations

Run after the geometry-only experiments:

```bash
python geomed/src/experiments/run_rage_feature_ablation.py --task foot --device cuda:0
python geomed/src/experiments/run_rage_feature_ablation.py --task cobb --device cuda:0
python geomed/src/experiments/run_rage_feature_ablation.py --task summarize

python geomed/src/experiments/run_rage_angle_only.py --task foot --device cuda:0
python geomed/src/experiments/run_rage_angle_only.py --task cobb --device cuda:0
python geomed/src/experiments/run_rage_angle_only.py --task summarize
```

These runners mask input descriptors before encoding and reuse the full RAGE results. They retain the full architecture and parameter count. Input masking differs from zeroing the encoded branch representation.

## 5. Attention Correspondence

Run after the matched Latent models:

```bash
python geomed/src/experiments/audit_ordinary_attention_geometry_alignment.py --device cuda:0
python geomed/src/experiments/audit_attention_geometry_repeated_donor_null.py --repetitions 1000
```

The audit keeps model predictions and attention maps fixed, compares self geometry with donor geometry, and uses 10,000 bootstrap resamples per seed. It measures spatial correspondence, not causal dependence.

## Implementation Notes

- Angular descriptors use center-vertex order, followed by neighbor-pair order. The Latent head concatenates visual features before geometry. Preserve these orders when loading checkpoints.
- Early-stopping patience is 10 for Foot and 20 for Cobb.
- `controlled_geometry_config.json` and `aasce_geometry_config.json` contain the original unstandardized δ = 5 baseline settings. They are distinct from the default standardized configs.
- The separate performance-bootstrap table generator and frozen-encoder readout benchmark are not included.
