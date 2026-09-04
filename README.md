# fly_gsplat

## Project Overview

`fly_gsplat` reconstructs a 3D point cloud of a fruit fly, per video frame, from synchronized 4-camera lab recordings, and then measures the fly's body and wing kinematics (yaw/pitch/roll, wing stroke/deviation/pitch angles) from that reconstruction.

The pipeline has three stages:

1. **Dataset generation** — turn raw multi-camera EasyWand (MATLAB) calibration + sparse-pixel recordings into a Nerfstudio-compatible dataset (`transforms.json` + per-camera images) and a visual-hull point cloud for initialization.
2. **3D Gaussian Splatting reconstruction** — train a per-frame `splatfacto` (Nerfstudio/`gsplat`) model on the 4 camera views to produce a dense point cloud (`splat.ply`) of the fly.
3. **Postprocessing** — clean the raw point cloud (remove floaters), label points as `body`/`wing_L`/`wing_R`, and extract per-frame kinematic angles from the labeled points.

This README documents how to set up the environment and run the pipeline end to end. It assumes you already know the biological/experimental background; it only covers what the code does and how to run it.

---

## Repository Structure

```
fly_gsplat/
├── generate_dataset.py        # Stage 1: EasyWand calibration + sparse .mat -> transforms.json + images/
├── generate_hull.py           # Stage 1: visual-hull point cloud for 3DGS initialization (init_points.ply)
├── pyproject.toml             # Python package metadata + dependencies (torch/nerfstudio/gsplat excluded, see below)
├── CLAUDE.md                  # Short project/AI-assistant orientation notes (Chinese)
│
├── utils/                     # Core library used by every pipeline stage
│   ├── camera.py               #   CameraConfig: EasyWand DLT <-> OpenCV <-> OpenGL camera conversions
│   ├── calib.py                 #   proj / backproj / triangulate
│   ├── dataset.py                #   transforms.json frame-dict generation, sparse-image reconstruction
│   ├── image.py                   #   mask binarization / dilation / cropping helpers
│   ├── gaussian_features.py        #   T1: per-point feature table from a trained splat.ply
│   ├── ply.py                       #   PLY I/O, rescale/unrescale, connected-component helpers
│   ├── reproject.py                  #   3D point -> 2D pixel reprojection using a trained dataparser transform
│   └── viz.py                         #   shared Viser scene-building helpers
│
├── models/
│   └── splatfacto_checkpoint.py   # Custom SplatfactoModel variant ("splatfacto-checkpoint") that dumps
│                                   # stats/points/eval-images at intervals during training, for debugging
│
├── preprocessing/
│   ├── select_frame_window.py   #   picks a clean, well-tracked frame window per video before training
│   ├── calib/viz_calib.py        #  A/B diagnostic: compare calibration .mat files via reprojection
│   └── hull/viz_hull.py           # A/B diagnostic: build+inspect a visual hull for one calibration/frame combo
│
├── postprocessing/             # Stage 3: raw splat.ply -> cleaned/labeled points -> kinematics
│   ├── calc_kinematics.py       #   single-dataset orchestrator (T1->T2->T3->T4), auto-resumes from
│   │                             #   whatever stage is already done. Header comment is in Chinese.
│   ├── batch_calc_kinematics.py  #   same T1-T4 logic, looped over many sweep/output directories
│   │
│   ├── cleaning/                 #   T2: floater removal
│   │   ├── mark_floaters.py       #     production floater filter (k-NN connected components), CLI-runnable
│   │   └── *.py, CLEAN.md          #     ⚠ EDA / calibration / diagnostic one-offs for mark_floaters.py's
│   │                                #     threshold choice — dense Chinese comments, ask the author before
│   │                                #     relying on or modifying these
│   │
│   ├── labeling/                  #   T3: body / wing_L / wing_R point labeling
│   │   ├── motion/label.py         #     cross-frame motion-accumulated density split — current default
│   │   ├── fusion.py                  #     motion+k-means fusion glue (motion-veto on the k-means clusters,
│   │   │                                #    see .legacy/kmeans/ below)
│   │   └── labeling.py                 #     ⚠ dense Chinese comments. Two roles in one file: (a) shared
│   │                                     #    production helpers (`UP`, `compute_body_axes`,
│   │                                     #    `finalize_part_labels`, ...) imported by `motion/label.py` and
│   │                                     #    others — do not remove; (b) its own `process_frame`/CLI, the
│   │                                     #    superseded k-means-based T3 entry point (imports
│   │                                     #    `.legacy/kmeans/kmeans_split.py`, kept working for comparison)
│   │
│   ├── kinematics/                 #   T4: body-frame + wing-angle extraction from labeled points
│   │   ├── geometry.py              #     stateless geometry primitives (no fly-specific semantics)
│   │   ├── body_frame.py             #     body frame + yaw/pitch/roll
│   │   ├── wing_angles.py             #     wing stroke/deviation (phi/theta)
│   │   ├── chord.py, chord_matlab.py   #     wing pitch (eta) — point-cloud method vs MATLAB-port method
│   │   ├── eta_unwrap.py                #    post-pass: outlier filtering + unwrap for eta_L/eta_R
│   │   ├── pipeline.py                   #    per-dataset T4 driver, incl. sequence-corrected body axis
│   │   ├── robust_body_axis.py            #   head/tail-centroid body axis estimator (MATLAB port)
│   │   ├── correct_body_axis/              #   body-axis sign continuity-correction subsystem
│   │   ├── correct_wing_pitch/              #   ⚠ experimental/numbered scripts (09_.. – 14_..), research notes
│   │   ├── simulate_gt/                      #   synthetic fly renderer, used to validate T4 against known GT
│   │   ├── diagnostics.py                     #  no-GT sanity checks (smoothness/symmetry/plausible range)
│   │   ├── io_schema.py, mock.py               #  I/O contract + synthetic test fixtures
│   │   ├── reference/                           # MATLAB-pipeline ports + design/spec docs
│   │   └── tests/                                # pytest unit tests (T4 stages only)
│   │
│   └── viz/                        #   ⚠ Viser viewers, dense Chinese docstrings
│       ├── splat_viewer.py          #     unified frame-by-frame viewer: raw splat / hull / cleaned / labeled
│       └── reprojection_viewer.py    #    projects labeled 3D points back onto the 4 raw camera images
│
├── run/serial/                 # Single-GPU serial batch runner (alternative to gpu/schedule/)
│                                 # Group definitions and sparse-data path are hardcoded constants at the
│                                 # top of each script — edit before running. ⚠ Chinese comments.
│
├── gpu/                         # Multi-GPU training orchestration + one-off profiling reports
│   ├── schedule/                  #   production sweep scheduler — see Usage below
│   │   ├── schedule.py             #     entry point: one JSON config = one full param_sets x frames sweep
│   │   ├── common.py                #    shared helpers (data_dir_for, etc). ⚠ Chinese comments
│   │   ├── generate_configs_from_selection.py  # turns preprocessing/select_frame_window.py output into schedule.py configs
│   │   ├── configs/                  #   only sample_config.json is tracked in git; real configs are
│   │   │                              #   gitignored — write your own following that template
│   │   └── run_valid480_sweep.sh, run_mid200_sweep.sh   # multi-video batch wrapper shell scripts
│   ├── timing/, parallel/, serial_vs_parallel/, disk/   # one-off profiling/benchmarking studies (reports
│   │                                                     # + scripts), not part of the run path. ⚠ Some
│   │                                                     # scripts (e.g. disk/audit.py) are in Chinese.
│   └── ENV.md                      #   hardware/environment baseline report (Chinese)
│
├── debug/                       # Standalone validation/debug scripts, run manually, not imported elsewhere
│   ├── validate_calib.py          #   2D reprojection check across calibration methods
│   ├── validate_dataset.py         #  3D Viser check of a generated dataset's camera geometry
│   ├── validate_reprojection.py     # ⚠ hardcoded absolute path `/home/computer0/fly_project/fly_gsplat`
│   ├── debug_splat_ply.py            # compares hull init_points.ply vs a trained splat.ply
│   └── debug_checkpoints.py           # ⚠ Chinese docstring; visualizes splatfacto-checkpoint dumps
│
├── env_setup/
│   ├── env_setup.md               #   environment setup instructions — see Environment Setup below
│   ├── Roni_env_setup.md           #  earlier/alternate setup notes, kept for reference
│   ├── dataparser_args.md           # Nerfstudio dataparser CLI argument reference
│   ├── splatfacto_args.md            # Nerfstudio splatfacto CLI argument reference
│   └── test_env/                      # gitignored scratch dir for the Nerfstudio "lego"/"campanile" smoke test
│
├── .legacy/                     # Archived code, kept for reference/comparison only, not on the main run path
│   ├── kmeans/                    #   ⚠ single-frame k-means T3 split (`kmeans_split.py`) — superseded by
│   │                               #   `postprocessing/labeling/motion/`, still imported by
│   │                               #   `postprocessing/labeling/labeling.py` (comparison entry point) and
│   │                               #   `postprocessing/kinematics/simulate_gt/segment.py`. Dense Chinese comments.
│   ├── binary/                     #  earlier rule-based (non-clustering) body/wing two-way split
│   │                               #   (`binary_split.py`), superseded by kmeans then motion; still imported
│   │                               #   by `simulate_gt/segment.py` for comparison. Chinese comments.
│   ├── seg2d/                       # Python port of the legacy MATLAB 2D body/wing segmentation, used as an
│   │                               #   independent baseline to cross-check the new 3D labels — see Known
│   │                               #   Limitations, the cross-check itself has not been run end-to-end
│   ├── debug/                       # archived sphere-dataset debug pipeline, unmaintained
│   └── docs/                        # dated research logs
│
├── data/                        # Gitignored. Per-video input datasets — see Data below
└── outputs/                     # Gitignored. Per-video/sweep training + postprocessing outputs
```

### Notes on the structure

- **Core pipeline** (safe to read as documentation of how the system works): `generate_dataset.py`, `generate_hull.py`, `utils/`, `models/`, `postprocessing/kinematics/{geometry,body_frame,wing_angles,chord,pipeline,io_schema}.py`, `postprocessing/cleaning/mark_floaters.py`, `postprocessing/labeling/motion/label.py`.
- **Experimental / one-off scripts**: most of `gpu/timing/`, `gpu/parallel/`, `gpu/serial_vs_parallel/`, `gpu/disk/`, `postprocessing/kinematics/correct_wing_pitch/`, `postprocessing/cleaning/*.py` besides `mark_floaters.py`, and everything under any `diag/` subfolder. These were written to answer a specific one-time question and are not maintained as production code.
- **Modules flagged with ⚠ above** have dense, logic-heavy Chinese comments/docstrings encoding non-obvious experimental history (why a threshold was chosen, why an approach was abandoned, etc.). If you need to modify one of these, ask the author first rather than relying only on the comments or a machine translation.
- **`.legacy/kmeans/` and `.legacy/binary/` are not a normal importable Python package** — `.legacy` isn't a valid dotted module name, so code that still needs them (`postprocessing/labeling/labeling.py`, `postprocessing/kinematics/simulate_gt/segment.py`, and files inside `.legacy/` itself) adds the `.legacy/` directory to `sys.path` at the top of the file, then imports the sibling packages directly, e.g. `from kmeans.kmeans_split import ...`. Keep this pattern if you add new cross-references into `.legacy/`.

---

## Environment Setup

Full instructions: [`env_setup/env_setup.md`](env_setup/env_setup.md). Summary:

```bash
conda create -n fly_gsplat python=3.10 -y
conda activate fly_gsplat

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install ninja numpy jaxtyping rich
pip install gsplat==1.4.0 --index-url https://docs.gsplat.studio/whl/pt20cu118

pip install fpsample==0.2.0
python -m pip install --upgrade setuptools==69.5.1
pip install nerfstudio==1.1.5

# Install this repo (registers the custom `splatfacto-checkpoint` Nerfstudio method
# declared in pyproject.toml's [project.entry-points] — required even if you never
# use --debug-checkpoint)
pip install -e .
```

`torch`/`nerfstudio`/`gsplat` are CUDA-build-specific and installed manually (see above); everything else in `pyproject.toml`'s `dependencies` installs automatically with `pip install -e .`.

This project was developed on **WSL2 (Ubuntu) + CUDA 11.8 + PyTorch 2.1.2**. Raw recordings live on a lab NAS/Windows drive; on WSL2 this is mounted with:
```bash
sudo mount -t drvfs X: /mnt/x
```
Several scripts hardcode this `X:\...` → `/mnt/x/...` translation (e.g. `generate_dataset.py`, `preprocessing/select_frame_window.py`, `gpu/schedule/generate_configs_from_selection.py`) — if your data source isn't mounted the same way, either replicate this mount point or edit the `.replace("X:", "/mnt/x")` calls in those files.

---

## Data

Raw camera recordings and calibration files are **not checked into git** (`data/` and `outputs/` are gitignored) — they live on a lab NAS/shared drive. Ask the author for access, or for a copy of a sample dataset to get started.

### Expected layout

Each dataset root (`{base_name}/` in the code — a path relative to the repo root, e.g. `data/ctrl_009_002/`) holds one shared calibration and one subdirectory per frame:

```
data/ctrl_009_002/
├── calibration_easyWandData.mat   # EasyWand MATLAB calibration (DLT coefficients), shared by every frame
├── camera_KRX0.mat                # decomposed K/R/X0, secondary/derived — not authoritative (see below)
└── f0000/
    ├── images/
    │   ├── P0CAM1.png              # grayscale, 1280x800, reconstructed from EasyWand sparse-pixel data
    │   ├── P0CAM2.png
    │   ├── P0CAM3.png
    │   └── P0CAM4.png
    ├── transforms.json            # Nerfstudio dataset descriptor: per-camera intrinsics + OpenGL c2w pose
    └── init_points.ply            # visual-hull point cloud, 3DGS training initialization
```

- `calibration_easyWandData.mat`'s `coefs` field (11 DLT coefficients per camera) is authoritative; `CameraConfig.easywand_dlt()` (`utils/camera.py`) derives intrinsics/extrinsics from it via RQ decomposition. EasyWand's own `focalLengths`/`principalPoints`/`rotationMatrices` fields are not used.
- `transforms.json` follows Nerfstudio's OpenGL convention (X right, Y up, Z backward); this is a converted form of the OpenCV-convention camera used internally for triangulation/reprojection (`R_w2c`, `X0`).
- `init_points.ply` is a generic Open3D point cloud (no per-point color/feature data — that's computed later, from the *trained* `splat.ply`, by `utils/gaussian_features.py`).

### Sample dataset

`data/ctrl_009_002/` (frames `f0000`–`f0639`) is the dataset used throughout this README's examples and most of the repo's existing outputs (`outputs/ctrl_009_002_*`, `outputs/baseline/ctrl_009_002/`). It is **not in git** — request a copy, or reproduce it yourself from the raw sparse recordings with `generate_dataset.py` + `generate_hull.py` (see Usage below) if you have NAS access.

The raw input to `generate_dataset.py` (before the layout above exists) is a per-video folder of `Camera{1..4}_sparse.mat` files on the NAS, e.g. `X:\antenna\control\009_25052026\Sparse\Expr_009_mov_002\`.

**What's actually runnable right now, if you have a local copy of the above:**
- `data/ctrl_009_002/` + its already-trained output `outputs/ctrl_009_002_ratio3_sh0_dense/ratio3_sh0_dense/` is the only dataset in the repo that has every stage (dataset → hull → train → T1 → T2 → T3 motion → T4) already run end-to-end — it's the closest thing to a validated fixture. `outputs/ctrl_009_002_ratio3_sh0_dense/kinematics_motion/` already has the final T4 result (`kinematics_ratio3_sh0_dense.csv`, `body_angles.png`, `wing_angles.png`); view any trained frame without retraining anything:
  ```bash
  python -m postprocessing.viz.splat_viewer \
      --data-root outputs/ctrl_009_002_ratio3_sh0_dense/ratio3_sh0_dense --start 0 --end 5
  ```
- To rerun training from scratch on this dataset, `gpu/schedule/configs/sample_config.json` already points `base_name` at `data/ctrl_009_002`, but its `frames` range (0–5) only exercises dataset→hull→train→T1→T2 — it's too short for the production T3 (`postprocessing/labeling/motion/`), which accumulates evidence over a locked ±36-frame window (73 frames) and needs at least ~150 contiguous trained frames or it fails with `weighted_pca: sum of weights must be positive`. For a real T3/T4 test, widen `"frames"` to something like `{"start": 200, "end": 400}` — safely inside this video's fully-windowed `[36, 603]` range.
- Caveat: `postprocessing/labeling/motion/density.py`'s `DATASET_DIR` default and `LAST_VALID_IDX` (`= 639 - HALF_WINDOW`) are hardcoded to this exact 640-frame dataset, not size-generic — a freshly trained subset under a different `name`/`base_name` needs those overridden before T3 motion will run against it.

---

## Usage / How to Run

This walks through the pipeline once, end to end, for a single frame, using the manual (non-scheduled) entry points — the simplest way to understand each stage. For production-scale multi-frame/multi-video runs, see [Scaling up](#scaling-up) below.

**⚠ `generate_dataset.py` and `generate_hull.py` take their parameters from the `if __name__ == "__main__":` block, not CLI args — edit the paths in that block before running (or import and call the function directly from your own script).**

### 1. Generate the dataset (calibration + images → `transforms.json`)

```bash
conda activate fly_gsplat
python generate_dataset.py
# edit __main__ first: data_dir, sparse_dir, target_frame
```
Reads `calibration_easyWandData.mat`, reconstructs grayscale images from the EasyWand sparse-pixel `.mat` files, and writes `{data_dir}/images/` + `{data_dir}/transforms.json`.

### 2. Generate the visual-hull point cloud

```bash
python generate_hull.py
# edit __main__ first: data_dir
```
Triangulates a seed point from the 4 camera masks, samples 1M points in a 2mm sphere around it, keeps points visible in all 4 cameras, and writes `{data_dir}/init_points.ply`. Opens a Viser point-cloud viewer at `http://localhost:8080` by default.

### 3. (Optional) Validate before training

```bash
python debug/validate_calib.py     # 2D reprojection sanity check
python debug/validate_dataset.py   # 3D Viser check: camera frustums + axes, http://localhost:8080
```

### 4. Train (3D Gaussian Splatting)

```bash
ns-train splatfacto \
  --data ./data/ctrl_009_002/f0000 \
  --vis tensorboard \
  --pipeline.model.background-color white \
  --pipeline.model.warmup-length 50 --pipeline.model.stop-split-at 1800
```
Trains a per-frame Gaussian splat model; the trained `splat.ply` (plus a Nerfstudio `config.yml`) lands under `outputs/.../splatfacto/{timestamp}/`. Swap `splatfacto` → `splatfacto-checkpoint` to use the custom debug-dump model variant (`models/splatfacto_checkpoint.py`).

### 5. Postprocess (clean → label → extract kinematics)

```bash
python -m postprocessing.calc_kinematics outputs/<name>/<param_set>
```
Auto-resumes from whatever stage already has output on disk (T1 features → T2 floater cleaning → T3 body/wing labeling → T4 angle extraction), then opens a Viser viewer of the result. If you omit the `dataset_root` argument, it falls back to a hardcoded `DEFAULT_DATASET_ROOT` inside `postprocessing/calc_kinematics.py` — pass the argument explicitly instead of relying on that default.

Output lands in `{dataset_root's parent}/kinematics/`: `kinematics_{name}.csv`, `body_angles.png`, `wing_angles.png`, and a `reprojection/` overlay directory.

### Scaling up

Repeating steps 1–5 by hand per frame doesn't scale past a handful of frames. Two batch entry points wrap the whole pipeline (dataset + hull generation, training, and postprocessing) per config:

- **`gpu/schedule/`** (recommended) — JSON-config-driven, multi-GPU/multi-worker, idempotent/resumable sweeps over `param_sets` × `frames`:
  ```bash
  python gpu/schedule/schedule.py --config gpu/schedule/configs/<your_config>.json
  ```
  Only `gpu/schedule/configs/sample_config.json` is tracked in git (real configs are gitignored) — copy it as a starting template. See its comments and `gpu/schedule/schedule.py`'s docstring for the full config schema.
- **`run/serial/`** — single-GPU, single-process batch loop; group definitions and the sparse-data source path are hardcoded constants at the top of each script (edit before running):
  ```bash
  python run/serial/batch_8groups_100frames.py smoke   # smoke test: reduced groups x 3 frames
  python run/serial/batch_8groups_100frames.py          # full batch
  ```

For batches, use `postprocessing/batch_calc_kinematics.py` instead of `postprocessing.calc_kinematics` — it loops T1–T4 over many `outputs/{sweep_name}/` directories and skips the interactive viewer.

---

## Known Limitations / TODO

- **No CLI args on the entry scripts.** `generate_dataset.py` and `generate_hull.py` require hand-editing their `__main__` block for every new dataset/frame; there's no argparse interface.
- **Hardcoded paths in a few scripts.** `debug/validate_reprojection.py` hardcodes the repo path `/home/computer0/fly_project/fly_gsplat` (both a `sys.path.insert` and a `REPO` constant); `postprocessing/calc_kinematics.py`'s `DEFAULT_DATASET_ROOT` and several docstring example commands (e.g. in `gpu/schedule/schedule.py`) reference the author's own machine/environment paths. Always pass explicit paths/args rather than relying on these defaults on a different machine.
- **NAS path convention assumed.** The `X:\...` → `/mnt/x/...` translation in `generate_dataset.py`, `preprocessing/select_frame_window.py`, and `gpu/schedule/generate_configs_from_selection.py` assumes the WSL2 `drvfs` mount used by the author (`sudo mount -t drvfs X: /mnt/x`). A different mount point requires editing those `.replace(...)` calls.
- **Two parallel T3 (body/wing labeling) implementations coexist.** `postprocessing/labeling/motion/` (cross-frame motion-accumulated density) is the current default used by `calc_kinematics.py`; `.legacy/kmeans/` (single-frame k-means, reached via `postprocessing/labeling/labeling.py`) is kept for comparison but not deleted. There's also an older `.legacy/binary/` two-way split. Check with the author before choosing one for new work.
- **The 2D/3D label cross-check is unfinished.** `.legacy/seg2d/` ports the legacy MATLAB 2D body/wing segmentation as an independent baseline, and a reprojection mechanism exists to compare it against the new 3D point labels — but per the project's own notes this comparison has not yet been run end-to-end.
- **Real-data validation of the segmentation-fusion (motion-veto) pipeline is incomplete.** It passes on synthetic data; real-data metrics (e.g. body-roll jump count) are not yet a clean pass.
- **3-camera support is experimental.** `data/ctrl_119_3cam/` and `outputs/ctrl_3cam_test/` show a 3-camera (instead of the standard 4-camera) variant works mechanically, but it hasn't been validated against ground truth on all test videos.
- **Test coverage is partial.** Only `postprocessing/kinematics/tests/` has pytest unit tests (T4 geometry/pipeline stages). The reconstruction stage (dataset/hull generation, training) and T1–T3 (features, cleaning, labeling) have no automated tests.
- **`.legacy/`** holds archived/superseded code kept for reference or comparison only (the older sphere-dataset debug pipeline, plus the superseded T3 k-means/rule-based splitters and the 2D segmentation baseline, see Repository Structure above) — not a normal package (see the `sys.path` note above) and not guaranteed to stay working as the live code around it evolves.
- **Real sweep configs and training logs are gitignored** (`gpu/schedule/configs/*` except the sample, `gpu/schedule/logs/`) — there is no shared record of exactly which sweeps have been run other than what's under `outputs/` locally.
