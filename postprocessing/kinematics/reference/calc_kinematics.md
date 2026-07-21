# T4 kinematics — reference & convention sheet

**Single source of truth** for how `postprocessing/kinematics/` defines every
body/wing angle, where each definition comes from, and how it differs from the
two legacy implementations. When the code and this document disagree, this
document wins — fix the code.

This file is documentation only. The two legacy code files it refers to live
alongside it and are **reference only, never imported or run**:

- `reference/python_snippets.py` — the stroke-plane angle definitions T4 adopts
  (from `run_pose_estimation.ipynb`, cells 1–3). Contains known bugs, flagged.
- `reference/matlab_snippets.m` — the old voxel/visual-hull definitions
  (from `AbbyLeung/Fly-Hull-Reconstruction`). Source of the chord **baseline**.

---

## 0. Conventions (locked)

| Item | Value |
|---|---|
| Units | **meters** (wing length ≈ 2.5–3 mm ≈ 0.0025–0.003 m) |
| Lab frame | right-handed `x, y, z` |
| Up / gravity | `up = +z`, derived from cam4 extrinsics (cam4 optical axis looks toward world +z; in OpenGL the camera looks down its own −z). Exposed as an overridable `up` parameter; assert `≈ [0,0,1]`. |
| Left / right | the **fly's own** left/right |
| Angle output frame | **stroke-plane frame** (per Python notebook). Lab-frame variants are *not* produced for now. |
| Stroke plane (single frame) | `n_sp` = `x_body` rotated **−45°** about `y_body` (Rodrigues). `stroke_plane_pitch_deg = 45` is a parameter; an externally supplied `stroke_plane_normal` overrides it. |

---

## 1. Input / output contract

**Input** — one per-point CSV per frame (T1 features + T2 `if_keep` + T3
`part_label`). Example path:
`outputs/ctrl_009_002_8groups_100frames/G2b_scale_reg_ratio3/f0090/splatfacto-checkpoint/<ts>/gaussian_features_f0090.csv`
(T2 output adds `if_keep`, suffix `_marked.csv`).

Columns (per point / row):

```
x, y, z,
dist_to_centroid, dist_to_principal_axis,
R, G, B, color_oob,
opacity,
scale_phys_0, scale_phys_1, scale_phys_2, scale_ratio,
linearity, planarity, sphericity,
orientation_x, orientation_y, orientation_z,
local_density,
if_keep,            # bool, from T2
part_label          # {body, wing_L, wing_R}, from T3
```

Core fields T4 relies on: `x,y,z`, `part_label`. Optional (used by the chord
method / filtering): `scale_phys_*`, `planarity`, `orientation_*`, `opacity`,
`if_keep`, `local_density`. `orientation_*` is treated as the per-Gaussian local
surface-normal proxy (verify semantics against T1 before trusting sign).

**Output** — **one row per frame** (the only place T4 breaks from the per-point
table shape). Row contains: body `yaw, pitch, roll`; per side `phi, theta, eta`;
`stroke_plane_normal` (3); chord quality/confidence fields; per side `span_dir`
(3) — the wing PCA span vector `phi`/`theta` were computed from (§4).

---

## 2. Body frame construction (single frame)

1. `x_body` = PCA major axis of `body` points; orient toward head. Sign
   fallback: `x_body · up > 0` (assume head points up-ish).
2. `y_body` = wing-root line `hinge_R → hinge_L`, with the `x_body` component
   removed, normalized. Wing roots taken as the proximal (nearest-body) end of
   each wing along its span. (Simplification acknowledged: T3 likely assigns the
   root region to `body`; we proceed with roots-as-available and keep a fallback
   of "wing-centroid line" behind a flag.)
3. `z_body = x_body × y_body`.
4. `n_sp` (stroke-plane normal) = `x_body` rotated −45° about `y_body`.

Replaces the notebook's multi-frame stroke-plane bootstrap (cell 1) with a
single-frame construction. Interface left open to accept an external
`stroke_plane_normal`.

---

## 3. Body angles

- **yaw** `= atan2(x_body_y, x_body_x)` (deg).
- **pitch** `= 90 − arccos(x_body · ẑ)` (deg) — elevation of body axis above
  horizontal. (Notebook cell 2 form.)
- **roll** — authoritative implementation is cell 2 `calculate_roll(yaw, pitch,
  y_body)`: build the zero-roll frame `(ê_y, ê_z)` from yaw/pitch, then
  `roll = atan2(y_body · ê_z, y_body · ê_y)`. MATLAB `calcBodyRoll.m` is the
  same idea; not duplicated in the snippets to avoid redundancy.

---

## 4. Wing stroke & deviation (phi, theta)

Stroke-plane frame, per notebook cell 3 `calculate_phi`, with the reference
vector revised (see below) to match MATLAB `calcAnglesRaw_Sam.m` (`reference/matlab_snippets.m`
lines ~190-195, `phiRdeg = atan2(rightSpanHat(2), rightSpanHat(1))`,
`thetaRdeg = asin(rightSpanHat(3))`, i.e. driven by `spanHat`, not the leading
edge):

- Project `x_body`, `y_body`, and the wing **span** direction `span_dir` onto
  the stroke plane; `phi = atan2(sign_left · (span_dir·ŷ_sp), span_dir·x̂_sp)`,
  unwrapped. `sign_left = −1` for wing_L, `+1` for wing_R.
- `theta = 90 − arccos(n_sp · span_dir)` (deg) — elevation of the wing span
  out of the stroke plane.

`span_dir` (`wing_angles.estimate_span`) is the wing's own PCA major axis
(root → tip, oriented outward), fit from a RANSAC wing-plane inlier set — this
is the authoritative MATLAB `spanHat` definition, **not** the leading edge.
The leading edge (`estimate_leading_edge`, a RANSAC line fit to the wing's
straight costal-vein edge) is a distinct quantity, kept solely as the source
of `chord.py`'s LE→TE chord-sign disambiguation (§5) — it is no longer used
for `phi`/`theta`. The two vectors are close for a well-formed wing but not
identical; do not conflate them.

---

## 5. Wing chord & pitch (eta) — the core deliverable

**Adopted definition (stroke-plane frame, cell 3):**
with `le_sp_normal = n_sp × le` (right) / `le × n_sp` (left) and
`sp_chord = le × le_sp_normal`,
`eta = atan2(sign_left · (chord · sp_chord), chord · le_sp_normal)`.
The `psi[psi < -100] = −psi` patch in the notebook is **rejected**; chord sign
is fixed instead by physical **LE → TE ordering** (chord always points
leading-edge → trailing-edge), which is unambiguous.

**Baseline to beat** — MATLAB `find_chords_quad.m`: chord = the most distant
pair of voxels inside a thin mid-span strip, disambiguated by wingtip velocity.
Fragile exactly where two wings' hulls merge near stroke reversal.

**New point-cloud chord method (T4-S4):**
1. Fit the wing **plane** (weighted PCA / RANSAC); smallest eigenvector = plane
   normal `n̂_w`. Point cloud is a thin sheet (`planarity` high, λ₂≫λ₃), so
   `n̂_w` is stable — unavailable to the ellipsoidal voxel hull.
2. Fit the **leading edge** (RANSAC line) → span direction.
3. **Segment** along span into bins; per bin take the in-plane extremes
   perpendicular to span → `le_i`, `te_i`; `chord_i = normalize(te_i − le_i)`;
   robust/trimmed weighted average over bins → chord, plus a per-span chord
   distribution (→ wing twist) and a confidence value.
4. Use each Gaussian's own smallest-`scale_phys` axis (via `orientation_*` /
   quaternion) as a local normal; **reject** points whose local normal
   disagrees with `n̂_w` (removes contaminating opposite-wing points).

**Why more accurate near stroke reversal (mechanism):** the voxel hull loses
information irreversibly there — the two wings' visual-hull intersection fuses
into one blob. The Gaussian point cloud keeps each point's xyz **and** a local
orientation, so even when the two wings are spatially close they still belong to
two differently oriented planes. Normal-consistency filtering + two-plane RANSAC
separate them, and the chord is a whole-wing segmented average rather than a
single most-distant pair, so it is far less sensitive to a few outliers.

---

## 6. Old-vs-new summary

| Quantity | Old MATLAB (matlab_snippets.m) | Notebook Python (python_snippets.py) | **T4 (adopted)** |
|---|---|---|---|
| span (phi/theta input) | wingCM→farthest voxel, or voxel PCA (`spanHat`) | RANSAC LE line (`le`) | **wing PCA major axis** (`estimate_span`, matches MATLAB `spanHat`) |
| leading edge (chord sign only) | n/a | n/a | RANSAC LE line (`estimate_leading_edge`, unchanged, §5 only) |
| phi / theta ref frame | lab horizontal (body-frame via 45°) | stroke plane | **stroke plane** |
| eta ref | lab vertical `ẑ` (`calcEta`) | stroke plane | **stroke plane**, sign via LE→TE |
| chord extraction | most-distant voxel pair (`find_chords_quad`) — **baseline** | LE/TE bins + sign patch | **segmented + Gaussian-normal-weighted robust fit** |
| stroke plane | hard 45° tilt in body frame | multi-frame wingtip fit | single-frame: `x_body` −45° about `y_body`; external override |
| roll | `calcBodyRoll.m` | `calculate_roll` (cell 2) | `calculate_roll` (cell 2) |
| frame scope | multi-frame smoothing | multi-frame | **single frame** (multi-frame interface left open) |

---

## 7. Known legacy pitfalls (do not reintroduce)

- Notebook `psi[psi < -100] = −psi` — unprincipled chord sign flip.
- Notebook stroke-plane bootstrap is multi-frame and self-referential; T4 is
  single-frame by design.
- MATLAB `find_chords_quad` most-distant-pair is outlier-dominated near stroke
  reversal — kept only as the baseline to beat.
- `calcEta` reference direction is lab-vertical; do not mix it with the
  stroke-plane `eta` — they are different definitions.
