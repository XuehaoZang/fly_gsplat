# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project generates 3D Gaussian Splatting (3DGS) datasets of fruit flies from multi-camera laboratory recordings. The pipeline converts raw EasyWand MATLAB calibration + sparse pixel data into Nerfstudio-compatible training data.

## Environment

**Conda environment:** `fly_gsplat` (Python 3.10 | CUDA 11.8 | PyTorch 2.0.1)

```bash
conda activate fly_gsplat
```

Windows drive data is accessed via WSL mount:
```bash
sudo mount -t drvfs X: /mnt/x
```

## Running the Pipeline

### Step 1: Generate dataset (images + transforms.json)
```bash
python generate_dataset.py
# Edit __main__ block to set data_dir, sparse_dir, target_frame
```

### Step 2: Generate Visual Hull initialization point cloud
```bash
python generate_init_points.py
# Edit __main__ block to set test_data_dir
```

### Step 3: Debug camera configuration (Viser visualization)
```bash
python debug_camera_config.py
# Opens interactive 3D viewer at http://localhost:8080
```

### Step 4: Train with Nerfstudio
```bash
ns-train splatfacto --data ./data \
  --pipeline.model.background-color black \
  --pipeline.datamanager.masks-on-gpu True
# Viser training viewer at http://localhost:7007
```

## Architecture

### Data Flow
```
X:\experiment\Sparse\*.mat + calibration_easyWandData.mat
        ↓ generate_dataset.py
data/images/P{frame}CAM{1-4}.png + data/transforms.json
        ↓ generate_init_points.py
data/init_points.ply
        ↓ ns-train splatfacto
outputs/ (trained 3DGS model)
```

### Key Files

- **`generate_dataset.py`** — Reads EasyWand `.mat` calibration, reconstructs per-frame images from sparse pixel index files (`Camera*_sparse.mat`), and outputs `transforms.json` in Nerfstudio `OPENCV` camera model format.
- **`generate_init_points.py`** — Visual Hull reconstruction: samples random 3D points in a bounding box, projects them into each camera's binary mask, and keeps points that pass a vote threshold (default: ≥2 cameras). Outputs `init_points.ply`.
- **`debug_camera_config.py`** — Loads `transforms.json`, generates per-camera mask frustum point clouds, and launches Viser for interactive 3D inspection. Used for verifying camera geometry before training.
- **`utils.py`** — Shared utilities: `generate_frame_dict` (Nerfstudio frame format), `crop_image`, `binarize_mask`, `dilate_mask`, `compute_target_center` (least-squares ray intersection), `plot_camera_coordinates` / `plot_point_cloud_viser` (Viser wrappers that block until "Continue" is clicked in browser).

### Camera Convention

- **Input (EasyWand)**: world-to-camera rotation `R_w2c`, camera center `X0` (DLT translation vector)
- **Output (Nerfstudio)**: camera-to-world 4×4 `transform_matrix` in OpenGL convention (−Z forward, Y up)
- The code has multiple commented-out calibration approaches (DLT, RQ decomposition, Roni's `.mat`) reflecting active experimentation on the correct axis/sign conventions.

### Data Layout
```
data/  (or data2/)
├── images/           # Reconstructed grayscale PNGs (1280×800)
├── masks/            # Binary masks (fly=white, background=black)
├── debug/            # Centroid and mask verification images
├── transforms.json   # Nerfstudio camera metadata
├── init_points.ply   # Visual Hull initialization point cloud
└── calibration_easyWandData.mat  # EasyWand MATLAB calibration
```

## Dependencies

Key packages beyond requirements.txt: `h5py` (sparse `.mat` reading), `open3d` (PLY I/O), `viser` (3D visualization), `nerfstudio`, `gsplat==1.4.0`.

Install gsplat with pre-built wheel:
```bash
pip install gsplat==1.4.0 --index-url https://docs.gsplat.studio/whl/pt20cu118
```
