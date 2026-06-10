"""
validate_calib.py
Compares EasyWand calibration methods by projecting a shared 3D sphere
into each camera view (OpenCV convention).

Convention: (K, R_w2c, X0)  ->  uv ~ K @ R_w2c @ (X - X0)

Methods:
  roni   - camera_KRX0.mat          (ground truth / shared-target basis)
  rq     - RQ decomp of coefs       (production candidate)
  native - focalLengths + rotMats   (diagnostic; Cam4 principal point unreliable)
"""

'''
# TODO clean up notes
Calibration from EasyWandData
1. Intrinsics:
    1.1 using focalLengths, ppts=principalPoints(default center)
    1.2 using coefs to decompose K, R, X0, normalize K
    1.3 compare with Roni's code generated camera_KRX0.mat
2. Extrinsics
    2.1 using DLTrotationMatrices.T --> get [R|T] --> gravity flip --> reasonable config, not intersecting beam
    2.2 old setup: using coefs to construct P, use inv(K) from 1.1 to get [R|T]
    2.3 new setup, hull recon: using rotationMatrices to get R - camera direction, DLTtranslationVector to get X0 - camera center
    2.4 Roni's version: using coefs and QR decomposition
'''

import io, contextlib
import numpy as np
import scipy.io as sio
from scipy.linalg import rq
import cv2
from pathlib import Path
from utils.calib import proj, backproj, triangulate, mask_centroid, check_ortho, rq_decompose_dlt

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
    """
    RQ decomposition of coefs, sign-aligned to ew.rotationMatrices.
    Mirrors Roni's MATLAB decompose_dlt exactly.
    Returns K with positive fx/fy, R_w2c with det=+1.
    """
    return rq_decompose_dlt(ew, i)

def adapt_native(ew, krx0, i):
    """
    K from focalLengths/principalPoints + R from rotationMatrices.
    """
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