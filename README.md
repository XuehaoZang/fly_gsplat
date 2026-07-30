# fly_gsplat

3D Gaussian Splatting pipeline for fruit fly reconstruction from multi-camera view recordings. 
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
data/{base_name}/f{NNNN}/images/  +  transforms.json
        ↓  generate_hull.py
data/{base_name}/f{NNNN}/init_points.ply
        ↓  ns-train splatfacto  (manual / run/serial / gpu/schedule)
outputs/{name}/{param_set}/f{NNNN}/
```

`sparse_dir` (the EasyWand + sparse pixel source) lives on the cloud/NAS drive mounted at `/mnt/x` (`X:\...` from Windows) — it is never copied wholesale. `generate_dataset.py` pulls only the frames it's asked for and writes the processed result locally under `data/{base_name}/f{NNNN}/`, which (together with `outputs/`) is gitignored.

There are three ways to run the pipeline. Pick one:
- **1. Parallel** (recommended) — full sweeps across both GPUs.
- **2. Serial batch** — single-GPU overnight batches.
- **3. Manual** — one frame, one param set, run each pipeline stage by hand.

### 1. Parallel (recommended) — `gpu/schedule/`

Queue-based scheduler: one config = one full sweep (all `param_sets` × all `frames`), dispatched across `WORKERS_PER_GPU` workers per GPU. Idempotent/resumable — rerunning the same config skips already-completed `(param_set, frame)` tasks. A single run handles everything end-to-end — dataset generation + hull (deduped per frame, skipped if already present), queueing, and training — there's no need to run `generate_dataset.py`/`generate_hull.py` separately first.

```bash
/home/computer0/anaconda3/envs/fly_gsplat/bin/python gpu/schedule/schedule.py --config gpu/schedule/configs/<name>.json
```

Defaults to plain `splatfacto` (no debug dumps). Add `--debug-checkpoint` to train with `splatfacto-checkpoint` instead (dumps stats/points/eval_images mid-training, see `models/splatfacto_checkpoint.py`):
```bash
/home/computer0/anaconda3/envs/fly_gsplat/bin/python gpu/schedule/schedule.py --config gpu/schedule/configs/<name>.json --debug-checkpoint
```

Multiple sweeps run **sequentially**, one `--config` call at a time (no built-in multi-config parallel mode):
```bash
/home/computer0/anaconda3/envs/fly_gsplat/bin/python gpu/schedule/schedule.py --config gpu/schedule/configs/a.json
/home/computer0/anaconda3/envs/fly_gsplat/bin/python gpu/schedule/schedule.py --config gpu/schedule/configs/b.json
```

**Config schema** (`gpu/schedule/configs/*.json`; see `ctrl_009_002_ratio3_sh0_smoketest.json` for a small sample):
```json
{
  "name": "ctrl_009_002_ratio3_sh0_full",
  "sparse_dir": "X:\\antenna\\control\\009_25052026\\Sparse\\Expr_009_mov_002",
  "base_name": "ctrl_009_002",
  "max_iters": 2000,
  "param_sets": {
    "ratio3_sh0": ["--pipeline.model.use-scale-regularization", "True", "--pipeline.model.sh-degree", "0"]
  },
  "frames": {"start": 0, "end": 640}
}
```
- `name` → run/sweep identity, output goes to `outputs/{name}/`.
- `base_name` → input dataset dir, `data/{base_name}/f{NNNN}/`.
- `frames` → Python range semantics (`end` exclusive).
- First run writes `outputs/{name}/sweep_meta.json`; rerunning the same `name` with a **different** config hard-errors (field-level diff printed) instead of silently overwriting.
- ⚠️ `--debug-checkpoint` is a CLI flag using model `splatfacto-checkpoint/`, not part of the config.

### 2. Serial batch — `run/serial/`

One-process, one-GPU loop over a fixed set of param-set "groups" × frames — no queue/worker setup, just `python run/serial/<script>.py`. Used for overnight single-GPU batches (`8groups`/`densify_6groups`) that produce `raw_records.json` + `summary.json` + comparison plots per sweep, written to `outputs/{SWEEP_NAME}/`. Progress is appended to `outputs/{SWEEP_NAME}/batch_progress.log`; a failed frame is recorded and skipped, it doesn't abort the batch. Like `gpu/schedule/`, dataset generation + hull are handled internally (skipped for frames already present) — no manual pre-step needed.

Unlike `gpu/schedule/`, the group definitions (`GROUPS`) and `SPARSE_DIR` are hardcoded at the top of each script rather than read from a JSON config — edit the constants before running:
```bash
python run/serial/batch_8groups_100frames.py            # full 8/10-group x 100-frame batch
python run/serial/batch_8groups_100frames.py smoke       # smoke test: NEW_GROUPS x 3 frames only
python run/serial/batch_densify_6groups_100frames.py
python run/serial/batch_densify_6groups_100frames.py smoke
```

### 3. Manual — single frame, single param set

Run each pipeline stage by hand. Unlike modes 1–2, dataset generation and hull are separate steps you run yourself before training.

#### Step 1 — Generate dataset
```bash
python generate_dataset.py
# Edit __main__: set data_dir, sparse_dir, target_frame
```
Reads EasyWand calibration, reconstructs images from sparse pixel files, writes `transforms.json` (OpenGL convention, Nerfstudio-compatible).

#### Step 2 — Generate Visual Hull
```bash
python generate_hull.py
# Edit __main__: set data_dir
```
Samples 1M points in a 2 mm sphere around the triangulated fly centroid, keeps points visible in all cameras (vote threshold = 4), saves `init_points.ply`.

#### Step 3 — Validate (optional)
```bash
python debug/validate_calib.py     # 2D reprojection test; compares rq / roni / native methods
python debug/validate_dataset.py   # 3D Viser check; frustum beams + camera axes at localhost:8080
```

#### Step 4 — Train
```bash
ns-train splatfacto \
  --data ./data/ctrl_009_002/f0000 \
  --vis tensorboard \
  --pipeline.model.background-color white \
  --pipeline.model.warmup-length 50 --pipeline.model.stop-split-at 1800
# Viser training viewer at http://localhost:7007
```
Swap `splatfacto` → `splatfacto-checkpoint` (+ `--pipeline.model.save_stats True --pipeline.model.stats_every 1000` etc.) for the debug-dump variant.

Trained `splat.ply` feeds straight into **[Postprocessing & Visualization](#postprocessing--visualization)** below (cleaning → labeling → kinematics → viewers).

---

## Postprocessing & Visualization

Per-frame raw `splat.ply` → **T2 cleaning** → **T3 labeling** → **T4 kinematics** → viewers. One-shot orchestrator, auto-resumes from whatever stage is already done:

```bash
python -m postprocessing.calc_kinematics [dataset_root]
# defaults to DEFAULT_DATASET_ROOT set in the file; no other CLI args supported
```
- Already has `kinematics/kinematics_{dataset}.csv` → skip straight to plots/viewer.
- Already has T3 output (`_labeled.csv`) but no kinematics csv → skip T1–T3, run T4 + viewers.
- Only raw per-frame `splat.ply` → run T1 → T2 → T3 → T4 + viewers.

Output lands in `{dataset_root's parent}/kinematics/`: `kinematics_{name}.csv` + debug `.pkl`, `body_angles.png` (yaw/pitch/roll) + `wing_angles.png` (phi/theta/eta, L/R), a `reprojection/` overlay dir (≤5 frames, evenly sampled), then it launches the Viser splat/point viewer (Ctrl+C to exit).

### T1 — Gaussian features
`utils/gaussian_features.py::compute_gaussian_features` computes a per-point feature table from each frame's `splat.ply` → `gaussian_features_f{NNNN}.csv`. Frames that already have the csv are skipped.

### T2 — Cleaning: `postprocessing/cleaning/mark_floaters.py`
Flags isolated floater points, adds an `if_keep` column (no rows/columns removed). Criterion: k-NN connected-component size (k=10, dist_percentile=75) ≤ `min_patch_size`=10 → floater. Locked/validated, not CLI-tunable.
```bash
python -m postprocessing.cleaning.mark_floaters --data-root outputs/<sweep>/<group> --start 0 --end 99
python -m postprocessing.cleaning.mark_floaters --csv path/to/gaussian_features_f0000.csv   # single-frame mode
```

### T3 — Labeling: `postprocessing/labeling/`
Splits kept points into `body` / `wing_L` / `wing_R`, writes `_labeled.csv` (adds `part_label`, `confidence`). Two implementations — `calc_kinematics.py` uses **motion** (current default):
- `labeling/motion/label.py` — cross-frame motion-accumulated density split (T3 v0, current default).
- `labeling/kmeans/kmeans_split.py` (+ `labeling/labeling.py`) — single-frame k-means (v2, aux_weight=1×, finalized); kept for comparison.

```bash
python -m postprocessing.labeling.motion.label --start 0 --end 99 --data-root outputs/<sweep>/<group>
python -m postprocessing.labeling.labeling --start 0 --end 99          # k-means variant
```

### T4 — Kinematics: `postprocessing/kinematics/pipeline.py`
Per-frame `_labeled.csv` → body frame (yaw/pitch/roll) + per-wing angles (phi/theta/eta, L/R) → `kinematics_{name}.csv` + debug `.pkl` (`PipelineConfig` controls `frame_glob`, `min_points`, etc.). A failing frame never aborts the batch — it's recorded in the `status` column instead. Normally invoked via `calc_kinematics.py`, not run standalone.

### Visualization
- **Point cloud (hull, pre-training)**: `generate_hull.py` opens Viser at `http://localhost:8080` by default; or `python debug/debug_splat_ply.py` to compare hull `init_points.ply` vs a trained `splat.ply` side by side with coordinate-space diagnostics (edit `__main__` to set `data_dir`/`splat_dir`).
- **Trained model (live viewer)**:
  ```bash
  tensorboard --logdir outputs/{name}   # loss curves, http://localhost:6006
  ns-viewer --load-config outputs/{name}/{param_set}/f{NNNN}/{method}/{timestamp}/config.yml
  # {method} = splatfacto or splatfacto-checkpoint, depending on how it was trained
  ```
- **Frame-by-frame splat / processed-csv viewer** — `postprocessing/viz/splat_viewer.py` (unified replacement for the old `debug/viz_splat_video.py` + `pointcloud_viewer.py`; run as a module, not a script):
  ```bash
  python -m postprocessing.viz.splat_viewer \
      --config gpu/schedule/configs/ctrl_009_002_ratio3_sh0_dense.json \
      --data-root outputs/ctrl_009_002_ratio3_sh0_dense/ratio3_sh0_dense --start 0 --end 5
  ```
  Opens Viser at `http://localhost:8080` (`--port` to change). All 7 display-mode checkboxes (`points`/`mesh`/`gaussians`/`hull` for splat, `raw`/`cleaned`/`labeled` for processed csv) are always shown in the browser; `--config` and `--data-root` are each independently optional — passing only one greys out the other category's checkboxes instead of hiding them. Use `--group`/`--method`/`--frame-start`/`--frame-end` to narrow the sweep, and `--frame` or `--start`/`--end` to pick processed-csv frames.
- **Reprojection overlay**: `postprocessing/viz/reprojection_viewer.py` projects labeled points back onto the 4 raw camera images; `calc_kinematics.py` calls this automatically for evenly-sampled frames, or run it standalone:
  ```bash
  python -m postprocessing.viz.reprojection_viewer --frame f0061 --data-root outputs/<sweep>/<group>
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