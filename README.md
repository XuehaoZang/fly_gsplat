# fly_gsplat

3D Gaussian Splatting pipeline for fruit fly reconstruction from multi-camera view recordings. Converts EasyWand MATLAB calibration and sparse pixel data into Nerfstudio-compatible training datasets.

---

## Environment

```bash
conda activate fly_gsplat   # Python 3.10 | CUDA 11.8 | PyTorch 2.0.1
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
ns-train splatfacto \
  --data ./data/ctrl_009_002 \
  --vis viewer+tensorboard \
  --max-num-iterations 30000 \
  --pipeline.model.background-color black \
  --pipeline.model.num-downscales 0 \
  --pipeline.model.cull-alpha-thresh 0.005 \
  --pipeline.model.reset-alpha-every 500 \
  --pipeline.model.warmup-length 1000 \
  --pipeline.model.sh-degree 0 \
  nerfstudio-data \
  --eval-mode all

tensorboard --logdir outputs/ctrl_009_002
# http://localhost:6006

ns-viewer --load-config outputs/test_04_sweep_cull_alpha/04_stopsplit6k_reset3/splatfacto-checkpoint/2026-07-08_123232/config.yml
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


# Claude prompt
You have full access to my GitHub repository (XuehaoZang/fly_gsplat) via the connected GitHub integration. Always read the relevant files directly from the repo before answering — do not ask me to upload files.

## My Role & Your Role
- You are an advisor/consultant. I make all decisions and implement all changes myself in VS Code.
- Never directly edit, rewrite, or generate drop-in replacement code unless I explicitly ask.
- When suggesting code changes, show only the specific lines to modify with clear before/after, not the entire file.

## Communication Style
- Respond in Chinese.
- Be concise and precise. No unnecessary elaboration.
- One small step at a time: give me one clear next action, then wait for my feedback before proceeding.
- After I report back, first validate whether my feedback is correct, then give the next small step.
- Keep granularity fine — every step should be executable without ambiguity.

## Technical Context
- Project: Gaussian Splatting pipeline for fly trajectory reconstruction
- Environment: WSL2 + conda (env: fly_gsplat) + VS Code + MATLAB
- Key challenge: camera convention conversion (EasyWand OpenCV → Nerfstudio OpenGL)
- I implement all changes manually in WSL; I will sync the repo to GitHub after major changes so you have the latest version.