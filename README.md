# CryoRISC

Rotation-invariant feature learning and image-guided candidate refinement for cryo-EM 2D classification.

CryoRISC combines a jointly trained C16 group-equivariant encoder and residual metric adapter with native-image class and pose refinement. Learned features initialize the classes and candidate sets; image-based refinement can subsequently update class assignments.

## Installation

Use Python 3.10 and install a PyTorch build compatible with your CUDA environment. From the repository root, run:

```bash
python -m pip install -e ".[dev]"
```

The tested numerical environment is listed in `requirements-tested.txt`. ESCNN 1.0.13 is required. CUDA is recommended for full-size experiments; the small demo also supports CPU execution.

Run the commands below from the repository root.

## Quick Start: Train and Test on a Tiny Dataset

The example contains 32 clean training projections from EMD-33026 and EMD-70734, plus 32 noisy test particles from EMD-65045. All images are 128 x 128 pixels. Training labels and actual per-particle directions are included.

The upload package already includes the example files. If downloading the example separately, extract `CryoRISC_tiny_example.zip` into the repository root. No pretrained weights are needed: the first command jointly trains the encoder and adapter from random initialization.

The input layout should be:

```text
CryoRISC/
├── examples/tiny65045/
│   ├── particles.mrcs
│   ├── viewdirs.npy
│   ├── source_indices.npy
│   ├── manifest.json
│   ├── train.npz
│   └── training_manifest.json
├── src/
└── README.md
```

Train, classify the test particles using the newly trained weights, and check the outputs:

```bash
python -m cryorisc train-demo
python -m cryorisc demo --encoder runs/tiny_training/encoder.pt --adapter runs/tiny_training/adapter.pt
python tools/check_demo_outputs.py
```

The tiny training run uses two epochs and a batch size of 16: two structures, two view groups per structure and four projections per group. It uses the same network, joint loss and training augmentations as the full trainer. Both modules receive gradients; comparisons across different structures are excluded. These few updates demonstrate training only, not convergence. Add `--device cpu` to `train-demo` to train without CUDA.

The test workflow covers feature extraction, spherical clustering, template initialization, coarse/local alignment, all-class image search, candidate supplementation and weighted class averaging. It uses 8 classes with Top4 initial candidates, Top2 local refinement and Top4 supplemented candidates.

Locally generated weights and the training log are saved to `runs/tiny_training/`. Test outputs, including `class_averages.mrcs`, are saved to `runs/tiny65045/`. A successful check prints `"status": "passed"`. Use new output directories when rerunning with changed settings or weights.

This is a workflow demonstration, not a benchmark or a reproduction of the paper's accuracy. EMD-65045 test images and directions are not used for training or checkpoint selection. Test directions are read only by the output checker, not by classification. The full experiment profiles retain their 200-class settings.

## Joint Training

Training structures: EMD-33026, EMD-70734, EMD-22689, EMD-20841 and EMD-3225. The trainer reserves 10% of the view groups within each structure for validation. All particles belonging to a reserved group remain outside gradient training. EMD-65045 is the independent simulation test structure.

Generate projections with `matlab/generate_training_projections.m`, supplying volume paths in `volumes.emd33026`, `volumes.emd70734`, `volumes.emd22689`, `volumes.emd20841` and `volumes.emd3225`. The generator produces grouped projections for these five structures; the trainer subsequently creates the group-disjoint training/validation split.

Install the data preparation dependency and merge the structure-specific projections:

```bash
python -m pip install -e ".[data]"
python tools/merge_simulated_data.py --grouped-root data/projections --output-dir data/simulated
```

The merger reads grouped particles from the five named structure directories, including `data/projections/emd70734/emd33026_grouped_train.mat`, and creates only `multistructure_v2_train.npz`. No separate validation data file is required. The file contains:

| Key | Shape | Content |
|---|---|---|
| `images` | N x 128 x 128 | Clean simulated projections before training augmentation |
| `viewdirs` | N x 3 | Actual per-particle projection directions |
| `group_ids` | N | View-group labels |
| `structure_ids` | N | Structure labels |

```bash
python -m cryorisc train --data-dir data/simulated --output-dir runs/training --batch-size 128 --epochs 60
```

The encoder and adapter are optimized jointly using a direction-aware contrastive loss and a rotation-consistency loss:

`L = L_con + 0.5 L_rot`

Checkpoint selection uses only the held-out view groups: precision among the ten nearest feature neighbors at an 8-degree threshold is averaged equally across the five structures, then across validation SNRs of 0.03 and 0.10. The held-out particle indices are saved in `split_indices.npz`. Test particles are not used for checkpoint selection.

The batch size is 128: two structures, 16 view groups per structure and four particles per group. Cross-structure pairs are excluded from the contrastive comparison set. Training settings are provided in `configs/joint_training_record.json`.

## Full-Size Classification

```bash
python -m cryorisc classify --input data/particles.mrcs --profile 10028 --encoder checkpoints/best_encoder.pt --adapter checkpoints/best_adapter.pt --output-dir runs/full
```

| Profile | Native image size | Pixel size (Å/pixel) |
|---|---:|---:|
| 10028 | 180 × 180 | 2.68 |
| 10219 | 160 × 160 | 1.71 |
| 65045 | 128 × 128 | 2.8125 |

Input particles must be stored as a float32 MRCS stack. Use encoder and adapter checkpoints from the same joint-training epoch. Only load trusted checkpoint files.

- `--preflight`: check the input and a small batch of embeddings without running the backend.
- `--features-only`: stop after feature extraction and clustering.
- `--allow-cpu`: explicitly allow slow CPU execution for the full-size command.

Repeating the same command resumes completed stages. Use a new output directory when changing inputs, checkpoints or settings.

Main outputs:

| File | Content |
|---|---|
| `class_averages.mrcs` | Native-size 2D class averages; not manually selected or centered |
| `final_dominant_labels.npy` | Final maximum-weight class for each input particle |
| `final_membership_and_poses.npz` | Top4 classes, weights, angles, shifts and effective class mass |
| `features_and_groups.npz` | Embeddings, initial labels, representative anchors and cluster centers |

## Ablations

Set `--arm` in the classification command and use a separate output directory for each variant:

- `full` (default): learned guidance and image-guided candidate supplementation.
- `no-network`: independent random initialization and neutral candidate scores; no checkpoints are required.
- `no-candidate-rescue`: retain learned guidance but replace image-retrieved candidates with feature-ranked candidates. Subsequent refinement and averaging remain enabled.

## Simulation Evaluation: IAC

```bash
python -m cryorisc iac --directions data/viewdirs.npy --labels runs/full/final_dominant_labels.npy --output runs/full/iac.json
```

Intra-class Angular Consistency (IAC) measures whether particles assigned to the same class have nearby true projection axes. The evaluator reports particle-equal consistency at 5°, 8°, 10° and 15°, with antipodal axes treated as equivalent.

Singleton particles contribute zero and remain in the denominator. Use actual per-particle directions and final labels in the same particle order. IAC is not conventional classification accuracy.

## Reconstruction and Reference FSC

The `matlab/` directory provides common-line reconstruction and reference-FSC calling code. Install ASPIRE-Matlab separately and add the required functions, including `ReadMRC`, `WriteMRC` and `FSCorr`, to the MATLAB path.

Class selection, center-of-mass centering and EMAN2 volume alignment are separate operations. Reconstruction-to-reference FSC measures agreement with a reference volume; it is not independent half-map FSC.

## Tests

```bash
python -m pytest
```

