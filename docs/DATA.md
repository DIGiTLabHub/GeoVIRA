# Data Preparation

## Datasets

Place the released HVAngleEst and AASCE files under the repository root:

```text
GeoVIRA/
├── HVAngleEst/
│   ├── HVAngleEst/
│   │   ├── datasets.csv
│   │   └── images/
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
└── AASCE/data/
    ├── train/
    └── train_txt/
        ├── filenames.csv
        ├── landmarks.csv
        └── angles.csv
```

| Dataset | Measurements | Train / Val / Test | Split |
| --- | --- | --- | --- |
| HVAngleEst | HVA, IMA | 829 / 172 / 176 | Patient-disjoint |
| AASCE | Primary Cobb | 336 / 72 / 73 | Image-level |

HVAngleEst contains 617 / 132 / 133 patients in the three partitions. AASCE does not provide patient IDs; its target is the maximum released Cobb angle.

The foot loader reconstructs the patient split with seed 42. If `outputs/benchmarks/controlled_cross_model_v2/hvangle/hvangle_results.json` exists, its `split_ids` define membership and order instead. Prediction archives must follow the loader's exact identifier order.

## Landmark Predictions

GeoVIRA takes **precomputed detector predictions** as input. Detector training and OOF generation are not included.

Prepare one train/validation/test NPZ set for each detector seed: **17, 42, 73**. Training predictions must be five-fold out-of-fold: by patient for Foot and by image for AASCE. Validation and test predictions must come from a detector trained on the outer training partition.

| NPZ field | Foot | AASCE |
| --- | --- | --- |
| `identifiers` | `(N,)` | `(N,)` |
| `predicted_points` | `(N, 6, 2)` | `(N, 68, 2)` |
| `truth` | `(N, 2)` — HVA, IMA | `(N, 1)` — primary Cobb |
| `analytic` | `(N, 2)` | `(N, 1)` |
| `patient_ids` | `(N,)` | Not required |
| `fold` — training only | `(N,)`, IDs 0–4 | `(N,)`, IDs 0–4 |

Coordinates are in original-image pixels. Foot endpoints follow great-toe, MT1, MT2 order. AASCE corners follow TL, TR, BL, BR order for each of 17 vertebrae; Auto-EV selects the eight measurement landmarks.

Set prediction paths through `landmark_inputs` in the JSON configs under `geomed/src/experiments/`. The default paths retain the original `hih_patient_oof` and `aasce_landmark_oof` cache names. Dataset labels are used for supervision and evaluation, not as geometric model inputs.

## MedImageInsight

Obtain the released MedImageInsight code and checkpoint, and install its dependencies in the same environment:

```text
MedImageInsights/
├── medimageinsightmodel.py
└── 2024.09.27/
    ├── config.yaml
    ├── vision_model/
    │   └── medimageinsigt-v1.0.0.pt
    └── language_model/
        └── clip_tokenizer_4.16.2/
```

The visual encoder remains frozen. GeoVIRA uses its 1024-D global embedding and 15 × 15 × 2048 pre-pooling spatial features. Images, prediction archives and pretrained weights are not distributed in this repository.
