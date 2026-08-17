# AI-LiDAR-Flight-Strategy-Optimizer

A research software prototype for **UAV LiDAR growth-stage recognition, flight-parameter recommendation, and plot-level crop-height estimation**. The supplied system targets DJI Zenmuse L2 point-cloud workflows and combines a PCT-based two-class growth-stage model with rule-based flight recommendations and LiDAR plant-height calculation.

## What the software does

The desktop workflow follows three stages:

1. **Growth-stage detection**
   - Load a LAS point cloud and a plot Shapefile.
   - Split the point cloud by plot polygons.
   - Sample 4096 points per plot and classify each plot with a PCT model.
   - Labels: `Post-tasseling` (class 0) and `Pre-tasseling` (class 1).
   - If more than 50% of valid plots are classified as post-tasseling, the whole area is treated as post-tasseling.

2. **Flight-parameter recommendation**
   - Post-tasseling, accuracy-optimal: 2 m/s, 25 m, 50% overlap, 45° scan angle.
   - Post-tasseling, accuracy/efficiency balanced: 4 m/s, 25 m, 30% overlap, 45°.
   - Pre-tasseling, accuracy-optimal: 2 m/s, 25 m, 30% overlap, 90°.
   - Pre-tasseling, accuracy/efficiency balanced: 4 m/s, 25 m, 30% overlap, 90°.

3. **Plant-height calculation**
   - Split points using LAS classification (`2` as ground, `1` as non-ground/vegetation in this project convention).
   - Estimate plot height at P100, P99.5, and P99.
   - Write a CSV result next to the input point cloud.

These rules reproduce the supplied research software; they are not universal flight recommendations for every crop, sensor, site, or regulatory environment.

## Repository layout

```text
AI-LiDAR-Flight-Strategy-Optimizer/
├── app/                 # Main Tkinter desktop application
├── pct/                 # PCT model architecture and utilities
├── training/            # Corn H5 training/evaluation scripts
├── models/              # Trained checkpoint location
├── legacy/              # Earlier GUI/layout source retained for reference
├── sample_data/
│   ├── las/             # Empty placeholder; large research data not bundled
│   └── shp/             # Empty placeholder
├── tests/               # Configuration and source-hygiene regression tests
├── LiDAR_Optimizer.spec # PyInstaller build specification
├── requirements.txt
├── environment.yml
├── THIRD_PARTY_NOTICES.md
└── LICENSES/
```

## Environment

The original software documentation reports:

- Windows 11
- Python 3.9
- PyTorch 1.13.1 + CUDA 11.7
- NVIDIA RTX 4060 Ti during development

CPU execution is supported by the inference code, but PCT's `pointnet2_ops` extension must still be installed successfully.

## Installation

### 1. Create an environment

```bash
conda env create -f environment.yml
conda activate lidar-flight-optimizer
```

Or create a Python 3.9 environment manually and run:

```bash
pip install -r requirements.txt
```

### 2. Install PyTorch

Install a PyTorch build appropriate for your machine. To reproduce the documented development environment, use a PyTorch 1.13.1 / CUDA 11.7-compatible installation. Follow the official PyTorch installation instructions for the exact command on your platform.

### 3. Install `pointnet2_ops`

The PCT implementation depends on the compiled `pointnet2_ops` extension. The upstream `PCT_Pytorch` repository distributes `pointnet2_ops_lib` and instructs users to install it locally, for example:

```bash
pip install ./pointnet2_ops_lib
```

This repository does **not** vendor that external extension.

### 4. Add the model checkpoint

Place the trained checkpoint at:

```text
models/latest_model-new.t7
```

The supplied checkpoint was trained for two corn growth-stage classes and contains a 200-epoch training state; the app uses its `state_dict` for inference.

## Run the desktop application

From the repository root:

```bash
python -m app.lidar_optimizer_app
```

Select a LiDAR file (`.las`) and the corresponding plot Shapefile. The Shapefile is expected to contain an `Id` field identifying plots.

## Training / evaluation

Training data are expected as:

```text
data/corn/
├── ply_data_train.h5
└── ply_data_test.h5
```

Each H5 file should contain datasets named `data` and `label`. You can also provide a different location with `--data_dir` or the `CORN_DATA_DIR` environment variable.

Example:

```bash
python -m training.main \
  --dataset corn \
  --num_points 4096 \
  --batch_size 8 \
  --epochs 200 \
  --data_dir /path/to/corn_h5
```

Evaluation example:

```bash
python -m training.main \
  --eval True \
  --model_path models/latest_model-new.t7 \
  --data_dir /path/to/corn_h5
```

## Sample data

Large research LAS files are intentionally not committed. The `sample_data/las/` and `sample_data/shp/` folders are placeholders for a future small, redistributable example.

A Shapefile normally consists of several same-name companion files such as `.shp`, `.shx`, `.dbf`, and `.prj`; keep them together.

## PyInstaller

After installing all runtime dependencies and placing the checkpoint under `models/`:

```bash
pyinstaller LiDAR_Optimizer.spec
```

## Upstream PCT attribution

The PCT architecture/utilities are adapted from the MIT-licensed `Strawberry-Eat-Mango/PCT_Pytorch` repository. The original MIT notice is preserved in `LICENSES/PCT_Pytorch-MIT.txt`; see `THIRD_PARTY_NOTICES.md` for attribution and the research citation.

## License status

The repository-specific license has not yet been selected. Public visibility is not the same as granting an open-source license. Before describing the whole repository as MIT/Apache/GPL/etc., the authorized rights holder should select and add that license. Third-party components continue under their original licenses.
