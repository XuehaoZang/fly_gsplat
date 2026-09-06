# synth: synthetic fly videos with ground truth

Renders recorded simulator states of the flybody fruit fly through the real
four-camera calibration of this repository and writes the result in the
dataset layout that `ns-train splatfacto` reads in the parent fly_gsplat
repository, together with exact per-frame ground truth (body pose, all joint
angles, wing angles, part-labelled surface points). It is the simulation
benchmark of `../HANDOFF.md` Task 2.

Nothing is trained or simulated in this folder: the fly is posed and rendered
with CPU MuJoCo. The states come from the gs_recon project
(`Physical_AI/gs_recon`, `python -m gs_recon.synth.pipeline`), where a
reinforcement-learning flight policy flies recorded body trajectories; those
rollouts ship here as small `.npz` files (`data/rollouts/`, one per recorded
flight, listed in `data/rollouts/index.csv`). Rendered images are not stored
in the repository: render the rollouts you need.

## Dependencies

Python 3.10 or newer and

```
pip install -r requirements.txt      # numpy, mujoco >= 3.8, h5py, imageio, imageio-ffmpeg
```

Tested with mujoco 3.8 and 3.12. `imageio-ffmpeg` is only needed for the mp4
previews. MuJoCo renders offscreen through GLFW (Windows, desktop Linux) or
EGL/OSMesa (headless Linux: `MUJOCO_GL=egl`). No GPU, no torch, no nerfstudio.

## Usage

All commands are run from this folder (`cd synth`).

```
python render_synth.py --rollout data/rollouts/Expr113_Mov132__tpl_Expr113_Mov023.npz --out out/mov132
```

Options:

- `--calibration PATH`: a `transforms.json`, a frame folder or a dataset root
  containing one; default `../data/ctrl_009_002/f0200/transforms.json`.
- `--fps N`: frame rate of the output, default 16000 (what this repository
  assumes). 5000 is the native state rate; any other value interpolates the
  states (frames are flagged `interpolated` in `gt.h5`).
- `--max-frames N`: cap the number of rendered frames (0 = whole clip).
- `--seed N`: seed of the surface sampling, of the `init_points.ply`
  subsample and of the optional sensor noise (default 0).
- `--config my.json`: overrides of the appearance and arena settings listed
  in `render_synth.py::SynthConfig` (body and wing grey levels, blur, arena,
  surface sample counts, ...).
- `--no-video`, `--no-h5-images` (no image stacks in `gt.h5`),
  `--no-dataset` (no frame folders), `--gt-csv` (a per-frame
  `gt_points.csv` with explicit labels, about 300 kB per frame),
  `--verify-frames N` (self-check after rendering, default 20).

Cost: about 0.3 s per frame with four cameras. A 120 ms rollout at
16000 fps is about 1900 frames, so roughly 10 minutes and a `gt.h5` of a
few hundred MB with image stacks; use `--max-frames`, `--fps 5000` or
`--no-h5-images` for quick looks.

Output of one clip:

```
out/mov132/
  dataset/                       fly_gsplat layout, ready for ns-train
    fNNNN/images/P<N>CAM1.png .. P<N>CAM4.png   1280 x 800, 8-bit grayscale, background exactly 255
    fNNNN/transforms.json                     camera entries identical to the real calibration
    fNNNN/init_points.ply                     up to 2000 ground-truth samples visible in at least one camera
                                              (stand-in for the visual hull), grey 153, chosen with --seed
    fNNNN/gt_points.ply                       every surface sample, colour = body part (part_palette.json)
    frames.csv                                dataset frame number -> clip frame index, time
    part_palette.json                         colour -> part id/name table for gt_points.ply
    camera_KRX0.mat, calibration_easyWandData.mat   copied from the calibration folder
  cam1.mp4 .. cam4.mp4           previews (playback 25 fps, not real time)
  gt.h5                          full ground truth (below)
  calib.json, clip.json          calibrations; scalar metadata plus the full render config
```

`N` in the image names is the dataset frame number without padding
(`f0039/images/P39CAM1.png`), as in the real recordings. Only frames in which
every camera sees the whole fly are written to `dataset/` (the longest such
run, renumbered from 0; `frames.csv` maps them back to the clip).

Frames and units: `dataset/` and `gt.h5` share one world frame, the
calibration frame (+z up, origin at the calibration centre, where your
reconstructions live). `dataset/` files (PLYs, `transforms.json`) are in
metres; `gt.h5` and the PLY written by `export_points` are in centimetres
(divide by 100). `world_offset_cm` in `gt.h5` is the shift that was added to
the raw rollout states to centre the flight in the filming volume:
`raw state position = gt.h5 position - world_offset_cm`.

Check a dataset folder (image format, background, camera entries against the
real calibration, ground-truth points projected onto the silhouettes):

```
python verify_synth.py out/mov132/dataset --reference ../data/ctrl_009_002/f0200/transforms.json [--max-frames N] [--min-hit 0.9]
```

Per camera it prints `hit` (fraction of ground-truth points landing on
non-white pixels, must reach `--min-hit`), `row-flipped` (the same with the
image mirrored vertically, must stay clearly below `hit`), the dark pixel
count and the fraction of exactly-255 background.

Export the labelled surface points of chosen frames as a point PLY with
properties `x y z t part gray nx ny nz frame` (cm, seconds, part id):

```
python -m synthfly.export_points out/mov132/gt.h5 --out points.ply --frames 0 10 20
python -m synthfly.export_points out/mov132/gt.h5 --out points.ply --all --stride 5 [--visible-only]
```

Frame indices must be below `n_frames` (a root attribute of `gt.h5`).

## Ground truth (`gt.h5`)

Units cm, s, rad; `wing_angles_deg` in degrees. F frames, C cameras (4),
P surface samples (5100 by default), H x W = 800 x 1280.

Main root attributes: `trial_id`, `body_trial`, `template_trial` (the recorded
trials whose body path and wing template were flown), `fps`, `n_frames`,
`world_offset_cm`, `base_freq_hz` (the trial's wingbeat frequency),
`frames_fully_visible`, `fully_visible_longest_run`, `arena_side_cm`,
`arena_center_cm`, `control_dt_s`, `ended_by`, `checkpoint_sha256` (the
policy), `rollout_file`, `calibration`, `config` (JSON of the render config).

| dataset | shape, dtype | meaning |
|---|---|---|
| `cameras/<name>/K, R_cv, t_cv, P` | 3x3, 3x3, 3, 3x4 f64 | OpenCV calibration in cm; attrs `width`, `height`, `um_per_px_at_center`, `mujoco_*` |
| `frames/t_s`, `frame_index`, `source_step` | (F,) | time; frame index; the 5 kHz state each frame comes from |
| `frames/fully_visible`, `in_volume`, `interpolated` | (F,) bool | all cameras see the whole fly; inside the arena cube; frame was interpolated |
| `frames/qpos`, `qvel` | (F, 43), (F, 42) f64 | full simulator state (`joints/names`, `qposadr`, `dofadr`) |
| `frames/root_pos_cm`, `root_quat_wxyz`, `com_pos_cm` | (F, 3), (F, 4), (F, 3) | thorax pose (quaternion body to world) and centre of mass |
| `frames/root_lin_vel_cm_s`, `root_ang_vel_body_rad_s` | (F, 3) | world linear velocity, body-frame angular velocity |
| `frames/hinge_rad` | (F, 36) | every hinge angle (`joints/hinge_names`) |
| `frames/wing_joint_rad`, `wing_angles_deg` | (F, 6) | wing joints (yaw, roll, pitch; left then right); stroke, deviation, rotation |
| `frames/head_rad`, `abdomen_rad`, `haltere_rad` | (F, 3), (F, 14), (F, 2) | subsets of `hinge_rad` |
| `frames/body_xpos_cm`, `body_xquat_wxyz` | (F, 68, 3), (F, 68, 4) f32 | pose of every body (`parts/body_names`, part of each in `parts/body_part`) |
| `frames/carrier_phase`, `ctrl_freq_hz` | (F,) | wingbeat carrier phase and frequency in force at the frame |
| `frames/action` | (F, 7) | policy action computed from frame k (produces frame k+1); attr `convention` |
| `frames/ref_root`, `ref_com`, `ref_quat`, `ref_wing_qpos` | (F, 7), (F, 3), (F, 4), (F, 6) | the recorded trajectory the policy was tracking |
| `surface/local_cm`, `geom_id`, `part_id`, `material_gray` | (P, 3), (P,), (P,) u8, (P,) u8 | sample definition: point in its geom frame, geom, part (`parts/names`), material grey |
| `surface/xyz_cm`, `normal` | (F, P, 3) f32, f16 | sample position and surface normal in the world frame |
| `surface/gray`, `visible` | (F, P) u8, (F, P, C) bool | rendered grey at the sample; visible in each camera |
| `images/gray`, `part` | (F, C, H, W) u8 | silhouettes and per-pixel part ids (unless `--no-h5-images`) |

Wing angle mapping (attr `joints/wing_mapping`): `stroke = deg(yaw) + 90`,
`deviation = -deg(roll)`, `rotation = deg(pitch) + 45`. Part ids: 0
background, 1 thorax, 2 head, 3 antenna, 4 abdomen, 5 wing_left, 6 wing_right,
7 haltere_left, 8 haltere_right, 9 leg_left, 10 leg_right. Sample i is the
same material point in every frame.

## The rollouts (`data/rollouts/*.npz`)

Each file is one flight of the `step2` tracking policy of fly_mimic: the body
path of one recorded trial (`body_trial`) flown with the wing template
(baseline wingbeat) of another, randomly chosen trial (`template_trial`),
5 kHz states, 80 to 270 ms. The set covers every recorded flight of the
fly_mimic train and validation splits once (the test split is locked); a
pair was accepted when the policy stayed within 0.3 cm CoM RMS and 30 deg
orientation RMS of the recording, otherwise the template was swapped and the
flight repeated (`gs_recon.synth.pipeline --rollout-only`, seed 0).

`index.csv` lists them: `trial_id` (file name), `body_trial`,
`template_trial`, `split`, `frames` (5 kHz states), `duration_ms`,
`ended_by` (`trajectory_end` = the whole recording was flown), `base_freq_hz`
(wingbeat frequency of the template), `com_rms_cm`, `com_max_cm`,
`orientation_rms_deg`, `orientation_max_deg` (tracking error against the
recording), `size_kb`. The same numbers are in each file's metadata and in
`clip.json` after rendering. To make more rollouts (other pairs, other
policies) use gs_recon's pipeline; this folder only renders.

## Layout of this folder

```
render_synth.py       render one rollout
verify_synth.py       acceptance checks
synthfly/             the renderer (cameras, scene, surface samples, writer, verifier, point export)
synthfly/model/       self-contained fly model: fruitfly_synth.xml + 85 binary STL meshes (13.6 MB)
data/rollouts/        recorded states, one .npz per flight (0.2 to 0.7 MB each), index.csv
out/                  where your renders go (ignored by git)
```

This folder is generated from `Physical_AI/gs_recon` by
`scripts/export_synth_package.py`; edit there, not here.
