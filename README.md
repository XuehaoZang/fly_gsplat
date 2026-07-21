# fly_gsplat

3D Gaussian Splatting pipeline for fruit fly reconstruction from multi-camera view recordings. Converts EasyWand MATLAB calibration and sparse pixel data into Nerfstudio-compatible training datasets.

---

## Environment

```bash
conda activate fly_gsplat   # Python 3.10 | CUDA 11.8 | PyTorch 2.1.2
sudo mount -t drvfs X: /mnt/x   # mount Windows data drive in WSL
```

---

## Pipeline

```
EasyWand .mat + Camera*_sparse.mat
        ↓  generate_dataset.py
data/{name}/images/  +  transforms.json
        ↓  generate_hull.py
data/{name}/init_points.ply
        ↓  ns-train splatfacto
outputs/
```

### Step 1 — Generate dataset
```bash
python generate_dataset.py
# Edit __main__: set data_dir, sparse_dir, target_frame
```
Reads EasyWand calibration, reconstructs images from sparse pixel files, writes `transforms.json` (OpenGL convention, Nerfstudio-compatible).

### Step 2 — Generate Visual Hull
```bash
python generate_hull.py
# Edit __main__: set data_dir
```
Samples 1M points in a 2 mm sphere around the triangulated fly centroid, keeps points visible in all cameras (vote threshold = 4), saves `init_points.ply`.

### Step 3 — Validate (optional)
```bash
python validate_calib.py     # 2D reprojection test; compares rq / roni / native methods
python validate_dataset.py   # 3D Viser check; frustum beams + camera axes at localhost:8080
```

### Step 4 — Train
```bash
ns-train splatfacto \
  --data ./data/ctrl_009_002 \
  --pipeline.model.background-color black
  --pipeline.datamanager.masks-on-gpu True

# Viser training viewer at http://localhost:7007
```

```bash
 ns-train splatfacto-checkpoint   --data ./data/ctrl_009_002_test  --vis tensorboard --max-num-iterations 5000 --pipeline.model.background-color white  --pipeline.model.save_stats True  --pipeline.model.stats_every 1000   --pipeline.model.save_points True --pipeline.model.points_every 5000   --pipeline.model.save_eval_images True --pipeline.model.eval_images_every 1000

tensorboard --logdir outputs/ctrl_009_002
# http://localhost:6006

ns-viewer --load-config outputs/ctrl_009_002_8groups_100frames/G9_sh_degree_0/f0000/splatfacto-checkpoint/2026-07-14_073852/config.yml

ns-viewer --load-config outputs/ctrl_009_002_8groups_100frames/G3_densify_50_1800/f0000/splatfacto-checkpoint/2026-07-13_220525/config.yml
```

```bash
/home/computer0/anaconda3/envs/fly_gsplat/bin/python gpu/schedule/schedule.py --run-name ctrl_009_002_ratio3_sh0_full
```
---

## Data layout

```
data/{experiment_name}/
├── images/                        # Grayscale PNGs (1280×800, fly=visible, background=black)
│   ├── P{frame}CAM1.png
│   └── ...
├── debug/                         # Mask and centroid verification images (auto-generated)
├── transforms.json                # Camera metadata in Nerfstudio/OpenGL format
├── init_points.ply                # Visual Hull point cloud for 3DGS initialisation
└── calibration_easyWandData.mat   # EasyWand MATLAB calibration file
```

---

## Code structure

```
utils/
├── camera.py    # CameraConfig dataclass — central camera representation
├── calib.py     # proj, backproj, triangulate, check_ortho, mask_centroid
├── dataset.py   # generate_frame_dict (CameraConfig -> Nerfstudio frame dict)
├── image.py     # binarize_mask, dilate_mask, crop_image, gray_to_rgba
└── viz.py       # start_viser, add_camera_axes, add_point_cloud, stop_viser,
                 # build_mask_frustum
```

### CameraConfig

All camera data flows through `CameraConfig` (defined in `utils/camera.py`):

| Field | Type | Description |
|-------|------|-------------|
| `K` | (3,3) | Intrinsic matrix |
| `R_w2c` | (3,3) | Rotation world→camera (OpenCV) |
| `X0` | (3,) | Camera centre in world frame |
| `w`, `h` | int | Image dimensions |

Key properties computed on demand: `R_c2w`, `t`, `fx/fy/cx/cy`, `transform_opencv`, `transform_opengl`.

Key factory methods:
- `CameraConfig.easywand_dlt(ew, i)` — production method; RQ decomposition of EasyWand DLT coefs, mirrors Roni's MATLAB `decompose_dlt`
- `CameraConfig.from_opengl(frame)` — reconstruct from a `transforms.json` frame dict

---

## Camera convention

```
EasyWand .mat
  coefs (11 DLT coefficients per camera)
        ↓  RQ decomposition + sign alignment to ew.rotationMatrices
        ↓  vertical flip: cy -> H-cy, fy -> -fy  (EasyWand -> OpenCV image Y-axis)
CameraConfig  [OpenCV: X right, Y down, Z forward]
  projection:  uv ~ K @ R_w2c @ (X - X0)
        ↓  R_c2w_opengl = R_c2w_opencv @ diag(1, -1, -1)
transforms.json  [OpenGL: X right, Y up, Z backward]
  transform_matrix = [R_c2w_opengl | X0]   (Nerfstudio c2w)
```

EasyWand's `focalLengths`, `principalPoints`, and `rotationMatrices` fields are secondary approximations of the DLT fit and are not used in production. The `coefs` field is authoritative.