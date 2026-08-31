# fly_gsplat

3D Gaussian Splatting pipeline for fruit fly reconstruction from multi-camera view recordings. 
---

## Methods & Specific Contribution (3D Gaussian Splatting Reconstruction — New)

This is a from-scratch 3D Gaussian Splatting (nerfstudio/gsplat `splatfacto`) reconstruction pipeline built to replace the prior visual-hull + multi-camera-triangulation approach used in the fly antenna mechanosensation project. It runs on the same 4-camera + EasyWand DLT calibration input the old pipeline used, but replaces the reconstruction and part-labeling stage with a trained radiance/Gaussian field.

- Rebuilt the EasyWand→camera-geometry chain around a single `CameraConfig` data structure (`utils/camera.py`) that converts EasyWand DLT calibration into Nerfstudio-compatible OpenGL camera poses (`transforms.json`), via RQ decomposition of the 11 DLT coefficients per camera (`CameraConfig.easywand_dlt`, mirroring the original MATLAB `decompose_dlt`) rather than EasyWand's own `focalLengths`/`principalPoints`/`rotationMatrices` fields, which were found unreliable (notably an unreliable Cam4 principal point, commit `55f9afc`). Verified with a 3-way reprojection comparison (`roni`/`rq`/`native` methods) in `debug/validate_calib.py`.
- Built the dataset + initialization stage: `generate_dataset.py` reconstructs per-frame grayscale images from EasyWand sparse-pixel `.mat` files and emits Nerfstudio `transforms.json`; `generate_hull.py` produces a `splatfacto` initialization point cloud via visual-hull carving (1M points sampled in a 2 mm sphere around the triangulated centroid, kept if visible in ≥4 of 4 cameras).
- Extended nerfstudio's `SplatfactoModel` with a custom `splatfacto-checkpoint` variant (`models/splatfacto_checkpoint.py`) that dumps Gaussian stats/points/eval-images at configurable training-step intervals, plus a synthetic oblique 5th viewpoint for qualitative monitoring beyond the 4 real training cameras — used for mid-training debugging that plain `splatfacto` doesn't expose.
- Built a queue-based, idempotent, multi-GPU/multi-worker parallel training scheduler (`gpu/schedule/`, JSON-config-driven hyperparameter sweeps over `param_sets` × `frames`, commits `0d2e690`, `014300f`) plus a single-GPU serial-batch mode (`run/serial/`), replacing manual per-frame runs.
- Profiled the full per-frame pipeline stage-by-stage (`gpu/timing/`, `TIMING_REPORT.md`, 2026-07-21, n=5 repeats) and used the result to justify the scheduler's concurrency design (see Quantified Results).
- Built a 4-stage postprocessing pipeline (`postprocessing/`) that turns a raw trained `splat.ply` into fly kinematics: **T1** per-point Gaussian feature extraction (`utils/gaussian_features.py`), **T2** floater removal via k-NN connected-component size filtering (`postprocessing/cleaning/mark_floaters.py`, k=10, dist_percentile=75, min_patch_size=10, locked/validated), **T3** body/wing_L/wing_R point labeling (two implementations built and compared: cross-frame motion-accumulated density split — current default — and single-frame k-means, `postprocessing/labeling/`), **T4** body-frame (yaw/pitch/roll) and per-wing (stroke/deviation `phi`/`theta`, chord/pitch `eta`) angle extraction (`postprocessing/kinematics/`).
- Ported the prior MATLAB motion-based body/wing 2D segmentation algorithm (`seg_class`) to Python as an explicit baseline (`postprocessing/reference/seg2d/`, commit `86f1658`), and built a point-to-pixel reprojection cross-check (`seg2d_spec.md` §9) that projects trained 3D Gaussian points back onto the 4 raw camera views and looks up each point's old-pipeline 2D body/wing label — the mechanism for validating the new 3D labels against the old 2D ones, though the comparison itself has not yet been run end-to-end (see Gaps).
- Ran systematic hyperparameter sweeps (scale-regularization ratio, densification schedule) across 8+6 parameter-set groups × 100 frames each, scoring point-cloud quality via `n_gaussians`, `scale_ratio`, `opacity`, `dbscan_floater_frac`, `extent_overshoot` (`outputs/ctrl_009_002_8groups_100frames/summary.json`, `outputs/ctrl_009_002_densify_6groups_100frames/summary.json`).

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

`sparse_dir` (the EasyWand + sparse pixel source) lives on the cloud/NAS drive mounted at `/mnt/x` (`X:\...` from Windows) — it is never copied wholesale. `generate_dataset.py` pulls only the frames it's asked for and writes the processed result locally under `{base_name}/f{NNNN}/`, which (together with `outputs/`) is gitignored.

`base_name` is a full path relative to the repo root, not just a name under a fixed `data/` folder — e.g. `data/ctrl_009_002` (legacy single-video layout) or `ctrl_009/013` (one top-level dir per session, one subdir per video — see [Data layout](#data-layout)). Pick whatever root keeps a given batch's data out of everything else's way; `data_dir_for()` in `gpu/schedule/common.py` just does `REPO / base_name / f"f{NNNN}"`.

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

**Config schema** (`gpu/schedule/configs/sample_config.json` is the only one tracked in git — everything else under `gpu/schedule/configs/` is gitignored, keep real sweep configs organized in one subfolder per batch, e.g. `gpu/schedule/configs/ctrl_009_mid200/`):
```json
{
  "name": "ctrl_009_013_ratio3_sh0_dense_mid200",
  "sparse_dir": "X:\\antenna\\control\\009_25052026\\Sparse\\Expr_009_mov_013",
  "base_name": "ctrl_009/013",
  "max_iters": 2000,
  "param_sets": {
    "ratio3_sh0_dense": ["--pipeline.model.use-scale-regularization", "True",
                         "--pipeline.model.max-gauss-ratio", "3.0",
                         "--pipeline.model.sh-degree", "0",
                         "--pipeline.model.warmup-length", "50",
                         "--pipeline.model.stop-split-at", "1800",
                         "--pipeline.model.densify-grad-thresh", "0.0004",
                         "--pipeline.model.refine-every", "50"]
  },
  "frames": {"start": 1100, "end": 1300}
}
```
- `name` → run/sweep identity, output goes to `outputs/{name}/`.
- `base_name` → input dataset dir (full path relative to repo root, see above), `{base_name}/f{NNNN}/`. It must already contain `calibration_easyWandData.mat` (+ `camera_KRX0.mat`) before the run — `generate_dataset.py` reads calibration from `{base_name}/calibration_easyWandData.mat`, it does not fetch it from `sparse_dir`.
- `frames` → Python range semantics (`end` exclusive).
- `ratio3_sh0` (no densify overrides) vs `ratio3_sh0_dense` (adds `densify-grad-thresh`/`refine-every`, produces denser point clouds) are two different param sets kept side by side for the `ctrl_009_002` video — **`ratio3_sh0_dense` is the current/production one**, use it for any new run unless there's a specific reason not to.
- First run writes `outputs/{name}/sweep_meta.json`; rerunning the same `name` with a **different** config hard-errors (field-level diff printed) instead of silently overwriting.
- ⚠️ `--debug-checkpoint` is a CLI flag using model `splatfacto-checkpoint/`, not part of the config.

**Full run from a sparse path, start to finish** — no manual preprocessing step is needed anywhere in this:
1. Point `sparse_dir` at the video's `Camera*_sparse.mat` folder (`X:\...`, same drive for every video in a session) and pick a `base_name` — a session gets one top-level dir (e.g. `ctrl_009/`), one subdirectory per video (`ctrl_009/013/`) so `data/` doesn't accumulate one folder per video.
2. Drop the calibration `.mat` files into `{base_name}/` once (same EasyWand calibration for every video shot in the same session/rig — no need to recalibrate per video).
3. Write the config (above) and run `python gpu/schedule/schedule.py --config <path>` — Phase A (dataset + hull generation) and Phase B/C (training) all happen inside this one call, per frame, idempotently.
4. Once training is done, run kinematics on the result (`python -m postprocessing.batch_calc_kinematics --sweep-name <name> --group <param_set>`, see [Postprocessing](#postprocessing--visualization)).

For a whole batch of videos (many configs sharing one calibration), see the worked example below.

#### Worked example: 009_25052026 session, valid-window centered 480 frames per video

**Preprocessing/selection step (do this first, no GPU needed).** A video's sparse mat file has a fixed number of frame *slots* (e.g. 2401), but the fly's real tracked signal usually covers only a fraction of that — the rest is empty/degenerate `indIm`. Naively training the middle N frames of the slot count (the old `ctrl_009_mid200` approach below) risks training on frames with no real detections. `select_frame_window.py` fixes this: per video, per camera it finds contiguous runs of real (non-degenerate) detections, trims `MARGIN_FRAMES` off each run's ends (in/out-of-frame transition, unreliable even when a detection exists), intersects across all 4 cameras, keeps the longest surviving run, and — if that run exceeds `MIN_SIGNAL_FRAMES` — centers a fixed `TRAIN_FRAMES` (480) window inside it. Videos whose longest clean run is too short are filtered out entirely rather than trained on marginal/empty frames.

```bash
python select_frame_window.py \
  --sparse-root "X:\antenna\control\009_25052026\Sparse" \
  --out-csv gpu/schedule/configs/ctrl_009_valid480/frame_selection.csv
```
Writes one row per video: raw/trimmed signal length, pass/fail, and (if passed) the exact `[train_start, train_end)` frame range — this csv is the audit trail for which videos were trained and on what frames, and why the rest were dropped. On the 009_25052026 session, **17/29 videos passed** (12 had too little real tracking signal for a 480-frame window — several down to only ~160 raw frame slots total).

Then generate one `schedule.py` config per surviving video and run them:
```bash
python gpu/schedule/generate_configs_from_selection.py \
  --selection-csv gpu/schedule/configs/ctrl_009_valid480/frame_selection.csv \
  --sparse-root "X:\antenna\control\009_25052026\Sparse" \
  --data-root data/ctrl_009 \
  --out-dir gpu/schedule/configs/ctrl_009_valid480 \
  --name-prefix ctrl_009 \
  --param-set-name ratio3_sh0_dense

cd /home/computer0/fly_project/fly_gsplat
nohup ./gpu/schedule/run_valid480_sweep.sh gpu/schedule/configs/ctrl_009_valid480 ratio3_sh0_dense \
  > gpu/schedule/logs/run_valid480_sweep_master.log 2>&1 &
disown
```

`run_valid480_sweep.sh <configs_dir> [param_set_name]` loops over every config in `<configs_dir>` **sequentially** (one `schedule.py --config` call per video, each one using all `WORKERS_PER_GPU`×2 GPU workers until that video's 480 frames are done) and runs `batch_calc_kinematics.py` (T1–T4, including the `eta_unwrap` post-pass on `eta_L`/`eta_R`, see [Postprocessing](#postprocessing--visualization)) for that video immediately after its training finishes, before moving to the next video. `nohup ... & disown` detaches it from the shell so it survives a closed terminal/SSH disconnect. Each video's 480 frames take roughly an hour end-to-end on 2×A5000 — budget accordingly for a full multi-video batch.

Monitor:
```bash
tail -f gpu/schedule/logs/run_valid480_sweep_master.log      # overall progress (which video is running)
tail -f gpu/schedule/logs/ctrl_009_<mov>_ratio3_sh0_dense_valid480.log             # one video's training log
tail -f gpu/schedule/logs/ctrl_009_<mov>_ratio3_sh0_dense_valid480_kinematics.log  # one video's kinematics log
nvidia-smi                                                    # confirm both GPUs are busy
ls outputs/ctrl_009_<mov>_ratio3_sh0_dense_valid480/ratio3_sh0_dense/ | wc -l      # frames trained so far for that video
```

⚠️ T3 (`postprocessing/labeling/motion/`) accumulates motion evidence over a locked ±36-frame window (73 frames total, see `HALF_WINDOW` in `postprocessing/labeling/motion/density.py`) around each target frame — it degrades gracefully with a partially-covered window (validated down to a ~37/73 one-sided window, i.e. right at a video's first/last 36 trained frames) but needs *some* real neighbor context, not near-total absence of it. Don't smoke-test T3/T4 on an isolated handful of frames — train at least ~150 contiguous frames (covering some fully-windowed frames) or it will spuriously fail with `weighted_pca: sum of weights must be positive`.

<details>
<summary>Superseded: middle-200-frames approach (<code>ctrl_009_mid200</code>)</summary>

The first pass at batching this session picked the middle 200 frames of each video's total *slot count* (2401), not its real signal — mistaking "how many frame slots exist" for "how many frames have real tracking". `run_mid200_sweep.sh`/`gpu/schedule/configs/ctrl_009_mid200/` are kept for reference but should not be used for new runs; use `select_frame_window.py` + `run_valid480_sweep.sh` above instead.
</details>

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

For batches (many `outputs/{sweep_name}/` dirs, e.g. one per video in a multi-video session), use `postprocessing/batch_calc_kinematics.py` instead — same T1→T4 logic, but it loops over multiple sweeps and skips the interactive viewer at the end (so it can run unattended after/alongside training), and it resolves each sweep's raw-image dir from that sweep's own config `base_name` instead of a single hardcoded path:
```bash
python -m postprocessing.batch_calc_kinematics --sweep-name ctrl_009_013_ratio3_sh0_dense_mid200 --group ratio3_sh0_dense
python -m postprocessing.batch_calc_kinematics --configs-glob "gpu/schedule/configs/ctrl_009_mid200/*.json" --group ratio3_sh0_dense
```
`--group` is the `param_sets` key (`outputs/{sweep_name}/{group}/` is the trained dataset root). One sweep failing (exception anywhere in T1–T4) is caught, logged, and doesn't stop the rest of the batch.

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
Per-frame `_labeled.csv` → body frame (yaw/pitch/roll) + per-wing angles (phi/theta/eta, L/R) → `kinematics_{name}.csv` + debug `.pkl` (`PipelineConfig` controls `frame_glob`, `min_points`, etc.). A failing frame never aborts the batch — it's recorded in the `status` column instead. `x_body` (body long-axis sign) is sequence-corrected (continuity chain + mandatory anchor-flip safety net, `correct_body_axis/sequence_axis.py`) via `pipeline.run_dataset_with_sequence_correction`, not a per-frame independent guess. Normally invoked via `calc_kinematics.py`, not run standalone.

**T4 diagnostics** (`postprocessing/kinematics/diagnostics.py`): smoothness/periodicity/L-R-symmetry/plausible-range checks on an existing T4 run, no ground truth needed. Writes PNGs + `report.md`:
```bash
python -m postprocessing.kinematics.diagnostics \
    outputs/ctrl_009_002_ratio3_sh0_dense/ratio3_sh0_dense \
    postprocessing/kinematics/diagnostics_output/ratio3_sh0_dense
# args are optional; with none, defaults to the G2b_G9 100-frame dataset
```

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

`{base_name}/` (see `base_name` above) holds calibration at its root and one subdirectory per frame:
```
{base_name}/
├── calibration_easyWandData.mat   # EasyWand MATLAB calibration file (shared by every frame under base_name)
├── camera_KRX0.mat
└── f{NNNN}/
    ├── images/                    # Grayscale PNGs (1280×800, fly=visible, background=black)
    │   ├── P{frame}CAM1.png
    │   └── ...
    ├── transforms.json            # Camera metadata in Nerfstudio/OpenGL format
    └── init_points.ply            # Visual Hull point cloud for 3DGS initialisation
```

Two layouts in active use:
- `data/{video_name}/` — one video per top-level dir (legacy/single-video convention, e.g. `data/ctrl_009_002/`).
- `{session}/{mov}/` — one top-level dir per recording session, one subdir per video, calibration copied into each (all videos in a session share the same EasyWand rig calibration) — e.g. `ctrl_009/013/` for `Expr_009_mov_013`. Use this when a session has many videos, to keep `data/` from accumulating one folder per video.

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