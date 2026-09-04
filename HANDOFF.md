# fly_gsplat — Handoff Notes

## System Overview

`fly_gsplat` reconstructs a per-frame 3D point cloud of a fruit fly from synchronized 4-camera lab recordings using 3D Gaussian Splatting (Nerfstudio `splatfacto`), then post-processes that point cloud to extract body/wing pose (yaw/pitch/roll, wing stroke/deviation/pitch angles). The reconstruction and kinematics-extraction stages both run end-to-end on real data today. There is currently no ground-truth point cloud for any real recording — only a purely synthetic/analytic ground truth used to unit-test the angle-estimation math, which is relevant background for the directions below.

---

## Data

A sample fixture ships in this repo (no NAS access needed to look at it):

- `data/ctrl_009_002/` — calibration files (`calibration_easyWandData.mat`, `camera_KRX0.mat`) plus frames `f0200`–`f0399` (200 frames), each with `images/`, `transforms.json`, `init_points.ply`.
- `outputs/ctrl_009_002_ratio3_sh0_dense/` — the corresponding trained/post-processed output for the same frame range: `ratio3_sh0_dense/f02XX–f03XX/splatfacto/<timestamp>/` holds `splat.ply`, `*_labeled.csv`, `config.yml`, `dataparser_transforms.json`; `kinematics_motion/` holds the `body_angles.png` / `wing_angles.png` summary plots for the range.

This is a final-artifacts-only slice — intermediate T1/T2 CSVs and the per-frame kinematics CSV aren't included, and it covers one dataset/one param set out of many the author has run locally. For anything beyond this range (other videos, other sweep configs, raw sparse recordings), ask the author.

---

## Task 1: Baseline Test with Existing PLY

**Goal:** run your own 3D model-fitting / pose-estimation method directly on a `splat.ply` already produced by this pipeline, and estimate body/wing angles from it. This is the cheapest possible check of whether current reconstruction quality is sufficient for downstream fitting — no new infrastructure needed on either side.

**Data:**
- Example: `outputs/ctrl_009_002_ratio3_sh0_dense/ratio3_sh0_dense/f0200/splatfacto/2026-07-23_133234/`, containing:
  - `splat.ply` — the raw 3DGS reconstruction (standard Gaussian-splat PLY: `x,y,z`, `opacity` as a pre-sigmoid logit, `scale_0..2` as pre-exp log-scale, `rot_0..3` as an unnormalized `(w,x,y,z)` quaternion, `f_dc_0..2` as the SH DC/color term). `utils/ply.py::load_ply_with_attrs` loads it with the activation functions already applied to physical units.
  - `gaussian_features_f0200_labeled.csv` — the same points after T1 feature extraction + T2 floater filtering (`if_keep`) + T3 body/wing labeling (`part_label ∈ {body, wing_L, wing_R}`, `confidence`). Part labels are not baked into the PLY — the CSV is where they live.
- Coordinate convention: Nerfstudio/OpenGL world frame (Y up, Z backward), units in meters, fly-scale (coordinates are on the order of 1e-3–1e-2 m).
- Any other frame in `f0200`–`f0399` follows the same layout; `postprocessing/kinematics/io_schema.py` documents the full per-point/per-frame data contract.

**Purpose:** a baseline read on whether the existing reconstruction is good enough as-is for your fitting approach, ahead of investing in any of the directions below.

---

## Task 2: Simulation Benchmark

**Goal:** a controlled, synthetic multi-view benchmark — render synthetic images of a fly model through the existing camera geometry, reconstruct with the real 3DGS pipeline, and compare the reconstruction against exact ground truth, which is known by construction in simulation and needs no separate acquisition step.

**Already reusable:**
- Camera geometry/pose machinery: `utils/camera.py` (`CameraConfig`, EasyWand DLT ↔ OpenCV ↔ OpenGL conversions) and `utils/dataset.py` (`transforms.json` frame-dict generation) — these could drive a synthetic camera rig using the same intrinsics/pose conventions the real pipeline expects, so a synthetic dataset folder could in principle be trained on with the unmodified `ns-train splatfacto` command.
- A simple analytic fly geometry model already exists in `postprocessing/kinematics/simulate_gt/` (`mock.py`, `scene.py`): an ellipsoid body + two flat elliptical wings, with known pose/angle ground truth by construction. Today it's used only to generate 3D points directly in-memory to unit-test the T3 segmentation / T4 angle-estimation math — it has no image-rendering step and is never passed through `generate_dataset.py` / 3DGS training.

**Not there yet:**
- An actual image renderer turning a 3D fly model (this one, or a better one brought in) into synthetic per-camera images/masks in the format `generate_dataset.py` expects (`images/P0CAM{1..4}.png` + `transforms.json`).
- The GT point cloud export itself is straightforward once a model with known geometry exists (sample points from it directly), but the full render → train → compare pipeline doesn't exist yet.

**One possible flow:**
1. Define/extend a fly geometry model with known 3D structure (`mock.py`'s ellipsoid+wing model is one starting point).
2. Render it from several synthetic camera viewpoints using the existing camera-pose utilities.
3. Train with the standard `ns-train splatfacto` command, unchanged, to get a GS point cloud.
4. Export the GT point cloud directly from the model used in step 1.
5. Compare GS vs. GT (e.g. Chamfer distance), varying camera count/placement/trajectory to get a sense of the reconstruction's accuracy ceiling and failure modes under controlled conditions.

Worth noting: the output of this direction — `(synthetic images + camera poses, GT point cloud, GS point cloud)` triples — is the kind of paired data Tasks 3 and 4 would need, which the repo currently has no way to produce for real recordings.

---

## Task 3: Point Cloud Refinement Network

**Goal:** a network that maps a raw 3DGS point cloud to a refined point cloud closer to ground truth.

- Phase (a): geometry-only refinement (point cloud → point cloud), no semantic labels.
- Phase (b), later: extend the network to also predict per-point `body`/`wing_L`/`wing_R` labels.

**Data:**
- Needed: paired (coarse GS point cloud, GT point cloud) — doesn't exist today; this is what Task 2 would produce. Without that (or an equivalent real-world GT source), there's no training signal for this direction yet.
- Optional (phase b) part labels: for synthetic data, exact by construction from Task 2's model; for real data, `part_label` from T3 exists (`postprocessing/labeling/motion/label.py`) but is an estimated label rather than ground truth, and real-data validation of the current labeling method isn't a clean pass yet (see Open Questions).

---

## Task 4: Splatter Image + GT Supervision

**Goal:** a forward network that predicts Gaussian splat parameters directly from images (Splatter-Image-style), with an added GT-point-cloud supervision term during training rather than relying only on the render/photometric loss.

**Data:** paired `(images + camera poses, GT point cloud)` — again, what Task 2 would produce; no such paired data exists today for real recordings.

This is the largest and most exploratory of the four directions — highest data dependency, highest engineering cost, least validated approach for this domain so far. Probably makes more sense as something to revisit once Tasks 1–3 have produced some results, rather than a first move.

---

## Suggested Exploratory Directions

Roughly, Task 1 is the cheapest and needs no new infrastructure, so it's a natural first look. Task 2 is the first direction that requires building something new, but it's the most direct way to get real ground truth into this project, and Tasks 3 and 4 both lean on whatever it produces. Task 3 becomes a fairly self-contained ML problem once that data exists. Task 4 is the biggest bet — worth deciding on with whatever Tasks 1–3 turn up, rather than committing to upfront.

---

## Open Questions / Known Limitations

- **The shipped sample fixture is a narrow slice.** One dataset (`ctrl_009_002`), 200 of its frames, one sweep param set (`ratio3_sh0_dense`), final artifacts only (no T1/T2 intermediate CSVs, no per-frame kinematics CSV). Everything else the author has run locally is not in git — ask for it if needed.
- **No real-recording ground truth exists anywhere in the codebase.** The only "ground truth" present (`postprocessing/kinematics/simulate_gt/`) is a purely analytic synthetic model used to validate segmentation/angle-estimation code, not a rendering+reconstruction pipeline — see Task 2.
- **Body/wing labels live only in CSV, not in the PLY.** `part_label` is a column in `gaussian_features_f{frame}_labeled.csv`, produced by T3; the raw `splat.ply` carries no semantic information.
- **Hardcoded paths in a few scripts**, e.g. `debug/validate_reprojection.py` hardcodes the author's repo path; `postprocessing/calc_kinematics.py`'s default dataset root is also machine-specific. Passing explicit paths sidesteps this.
- **NAS mount convention assumed.** `generate_dataset.py`, `preprocessing/select_frame_window.py`, and `gpu/schedule/generate_configs_from_selection.py` assume a WSL2 `X:` → `/mnt/x` `drvfs` mount.
