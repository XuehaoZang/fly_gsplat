"""
debug_calib.py
Compares EasyWand calibration methods by projecting a shared 3D sphere
into each camera view (OpenCV convention throughout).

Convention: (K, R_w2c, X0)  ->  uv ~ K @ R_w2c @ (X - X0)

Methods:
  roni   - camera_KRX0.mat          (ground truth / shared-target basis)
  rq     - RQ decomp of coefs       (production candidate)
  native - focalLengths + rotMats   (diagnostic; Cam4 principal point unreliable)
"""

import io, contextlib
import numpy as np
import scipy.io as sio
from scipy.linalg import rq
from scipy import ndimage
import cv2
from pathlib import Path
from utils import proj, backproj, triangulate, mask_centroid, check_ortho

# ------------------------------------------------------------------ config ---
BASE     = Path("./data/ctrl_009_002")
FRAME    = 10
SPHERE_R = 0.001   # metres, ~half fly body

# --------------------------------------------------------------- adapters ----
def adapt_roni(ew, krx0, i):
    """Load K, R_w2c, X0 directly from Roni's camera_KRX0.mat."""
    K  = krx0[:, 0:3, i].astype(float).copy()
    R  = krx0[:, 3:6, i].astype(float).copy()
    X0 = krx0[:, 6,   i].astype(float).copy()
    return K, R, X0

def adapt_rq(ew, krx0, i):
    """RQ decomposition of coefs, sign-aligned to ew.rotationMatrices.
    Mirrors Roni's MATLAB decompose_dlt exactly.
    Returns K with positive fx/fy, R_w2c with det=+1."""
    coefs      = ew.coefs[:, i]
    ew_rot_w2c = ew.rotationMatrices[:, :, i]

    # build H (3x3) and h (3,) from the 11 DLT coefs
    H = np.array([[coefs[0], coefs[1], coefs[2]],
                  [coefs[4], coefs[5], coefs[6]],
                  [coefs[8], coefs[9], coefs[10]]])
    h = np.array([coefs[3], coefs[7], 1.0])

    X0 = -np.linalg.inv(H) @ h          # camera centre (independent of K/R)

    K_raw, R_raw = rq(H)                 # H = K_raw @ R_raw
    K_raw = K_raw / K_raw[2, 2]

    # sign-align axes to ew.rotationMatrices (resolves RQ ambiguity)
    s = np.sign(np.diag(ew_rot_w2c @ R_raw.T))
    s[s == 0] = 1.0
    Rot_to_ew   = np.diag(s)
    Rot_to_stan = np.diag([1., -1., -1.])   # EasyWand -> OpenCV convention
    S = Rot_to_stan @ Rot_to_ew

    K     = K_raw @ S;  K = K / K[2, 2]
    R_w2c = S @ R_raw

    # vertical flip: cy -> H-cy, fy -> -fy  (image Y-axis orientation)
    h_img    = int(ew.imageHeight[0] if isinstance(ew.imageHeight, np.ndarray) else 800)
    K[1, 2]  = h_img - K[1, 2]
    K[1, 1]  = -K[1, 1]

    return K, R_w2c, X0

def adapt_native(ew, krx0, i):
    """K from focalLengths/principalPoints + R from rotationMatrices.
    NOTE: principalPoints is a placeholder (image centre) for Cam4;
          ew.rotationMatrices is a secondary decomposition of coefs and
          has ~100px inconsistency vs coefs for all cameras. Kept for
          diagnostics only - do not use in production."""
    f  = float(ew.focalLengths[i])
    cx = float(ew.principalPoints[2 * i])
    cy = float(ew.principalPoints[2 * i + 1])
    K  = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])

    S     = np.diag([1., -1., -1.])
    K     = K @ S;  K = K / K[2, 2]
    R_w2c = S @ np.array(ew.rotationMatrices[:, :, i], dtype=float)

    K[1, 1] = -K[1, 1]
    K[1, 2] = 800 - K[1, 2]

    X0 = np.array(ew.DLTtranslationVector[:, i], dtype=float)
    return K, R_w2c, X0

ADAPTERS = {"roni": adapt_roni, "rq": adapt_rq, "native": adapt_native}
METHODS  = ["roni", "rq", "native"]

# ------------------------------------------------------------------- main ----
def main():
    debug_dir = BASE / "debug"; debug_dir.mkdir(exist_ok=True)

    ew   = sio.loadmat(str(BASE / "calibration_easyWandData.mat"),
                       struct_as_record=False, squeeze_me=True)["easyWandData"]
    krx0 = sio.loadmat(str(BASE / "camera_KRX0.mat"),
                       struct_as_record=False, squeeze_me=True)["camera"]
    n = int(ew.nCams)

    grays     = [cv2.imread(str(BASE / f"images/P{FRAME}CAM{j+1}.png"), 0) for j in range(n)]
    centroids = [mask_centroid(g) for g in grays]

    # shared target from roni (ground truth basis, decoupled from tested methods)
    roni_params = [adapt_roni(ew, krx0, j) for j in range(n)]
    rays  = [(X0, backproj(K, R, *centroids[j]))
             for j, (K, R, X0) in enumerate(roni_params)]
    target, tres = triangulate(rays)
    print(f"[shared target]  {target}  residual={tres*1000:.3f} mm\n")

    for name in METHODS:
        fn     = ADAPTERS[name]
        # suppress per-camera debug prints inside adapters during summary
        params = []
        for j in range(n):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                params.append(fn(ew, krx0, j))

        self_rays = [(X0, backproj(K, R, *centroids[j]))
                     for j, (K, R, X0) in enumerate(params)]
        _, self_res = triangulate(self_rays)

        print(f"=== {name} ===  self-residual={self_res*1000:.3f} mm")
        errs = []
        for j, (K, R, X0) in enumerate(params):
            det, ortho, ok = check_ortho(R)
            u, v, z = proj(K, R, X0, target)
            cu, cv  = centroids[j]
            err     = np.hypot(u - cu, v - cv)
            errs.append(err)

            r_px = max(int(round(K[0, 0] * SPHERE_R / z)), 1) if z > 0 else 0
            img  = cv2.imread(str(BASE / f"images/P{FRAME}CAM{j+1}.png"))
            cv2.drawMarker(img, (int(cu), int(cv)), (0, 255, 0), cv2.MARKER_CROSS, 30, 1)
            if z > 0:
                cv2.circle(img, (int(u), int(v)), r_px, (0, 0, 255), 2)
            cv2.imwrite(str(debug_dir / f"{name}_cam{j+1}.png"), img)

            flag = "OK " if ok else "BAD"
            print(f"  Cam{j+1} [{flag}] det={det:+.4f} ortho={ortho:.1e} "
                  f"u={u:7.1f} v={v:7.1f} z={z:.4f} reproj={err:6.2f}px r={r_px}px")
        print(f"  -> mean reproj = {np.mean(errs):.2f} px\n")

if __name__ == "__main__":
    main()