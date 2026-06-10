"""
debug_calib.py
    Compare EasyWand calibration methods by projecting a shared 3D sphere
    into each camera view (OpenCV convention only).

Standard internal representation: (K, R_w2c, X0)
  projection:    uv ~ K @ R_w2c @ (X - X0)
  backprojection dir(world): R_w2c.T @ inv(K) @ [u,v,1]

Methods compared:
  roni   -> camera_KRX0.mat            (ground truth, used as shared target basis)
  rq     -> RQ decomp of coefs + sign-align to rotationMatrices
  native -> focalLengths/principalPoints + rotationMatrices/DLTtranslationVector
"""

import numpy as np
import scipy.io as sio
from scipy.linalg import rq
from scipy import ndimage
import cv2
from pathlib import Path

# ---------------- config ----------------
BASE     = Path("./data/ctrl_009_002")
FRAME    = 10
SPHERE_R = 0.001          # m, ~half fly body; sphere radius for projection


# ---------------- core math (proven in verify_roni) ----------------
def proj(K, R, X0, X):
    """OpenCV forward projection. Returns (u, v, depth)."""
    xc = R @ (X - X0)
    uv = K @ xc
    return uv[0] / uv[2], uv[1] / uv[2], xc[2]


def backproj_dir(K, R, u, v):
    """Pixel -> world-frame ray direction (unit)."""
    d = R.T @ (np.linalg.inv(K) @ np.array([u, v, 1.0]))
    return d / np.linalg.norm(d)


def triangulate(rays):
    """Least-squares closest point of rays [(C, dir), ...]. Returns (X, residual_m)."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for C, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ C
    X = np.linalg.solve(A, b)
    res = np.mean([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - C)) for C, d in rays])
    return X, res


def gate(R):
    """Sanity check on rotation. Returns (det, ortho_norm, ok)."""
    det   = np.linalg.det(R)
    ortho = np.linalg.norm(R @ R.T - np.eye(3))
    ok    = abs(det - 1.0) < 1e-3 and ortho < 1e-3
    return det, ortho, ok


def mask_centroid(gray):
    v, u = ndimage.center_of_mass(gray > 0)
    return u, v

# ---------------- adapters: easyWand -> (K, R_w2c, X0) ----------------
def adapt_roni(ew, krx0, i):
    K  = krx0[:, 0:3, i].astype(float).copy()
    R  = krx0[:, 3:6, i].astype(float).copy()   # world->cam (verified)
    X0 = krx0[:, 6,  i].astype(float).copy()
    if i == 3:
        print(f"[Roni] fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
        print(f"[Roni] R_w2c={R}")
        print(f"[Roni] X0={X0}")
    return K, R, X0


def adapt_rq(ew, krx0, i):
    """
    精确对应 Roni 的 MATLAB decompose_dlt。
    Returns: K (3x3, fx/fy>0), R_w2c (3x3, det=+1), X0 (3,)
    Projection: K @ R_w2c @ (X - X0)  →  pixel (u,v)
    """
    coefs = ew.coefs[:, i]
    ew_rot_w2c = ew.rotationMatrices[:, :, i]
    # Step 1: 从 coefs 建 H 和 h（对应 MATLAB 的索引 1,2,3 / 5,6,7 / 9,10,11 / 4,8）
    H = np.array([
        [coefs[0], coefs[1], coefs[2]],
        [coefs[4], coefs[5], coefs[6]],
        [coefs[8], coefs[9], coefs[10]]
    ])
    h = np.array([coefs[3], coefs[7], 1.0])

    # Step 2: 光心（不依赖 K/R 分解）
    X0 = -np.linalg.inv(H) @ h

    # Step 3: RQ 分解
    # MATLAB: [Q, R_tri] = QR_Decomposition(inv(H)); K_raw = inv(R_tri); R_w2c_raw = Q'
    # 等价于直接 rq(H) → H = K_raw @ R_w2c_raw
    K_raw, R_w2c_raw = rq(H)
    K_raw = K_raw / K_raw[2, 2]

    # Step 4: 符号对齐
    # MATLAB: change_ax_dir = sign(diag(ew_rotation' * Q_out))
    #   ew_rotation = rotationMatrices^T (c2w),  Q_out = R_c2w_raw = R_w2c_raw.T
    #   → sign(diag(rotationMatrices @ R_w2c_raw.T))
    change_ax_dir = np.sign(np.diag(ew_rot_w2c @ R_w2c_raw.T))
    change_ax_dir[change_ax_dir == 0] = 1.0

    print(f"[DEBUG] change_ax_dir={change_ax_dir}")

    Rot_to_ew   = np.diag(change_ax_dir)
    Rot_to_stan = np.diag([1., -1., -1.])
    S = Rot_to_stan @ Rot_to_ew                 # MATLAB: Rot_to_stan * Rot_to_ew

    # Step 5: 配对翻转，保持 K @ R_w2c = H（差 scale）
    # MATLAB K: K = K*Rot_to_stan*Rot_to_ew/K(3,3); K=K/K(3,3)  →  等价 K_raw@S 再归一
    # MATLAB R: R = Rot_to_stan*Rot_to_ew*R'  →  S @ R_w2c_raw
    K     = K_raw @ S
    K     = K / K[2, 2]
    R_w2c = S @ R_w2c_raw

    # 垂直翻转修正：cy → 800-cy，fy → -fy
    # 对应 MATLAB K(2,2)=-K(2,2); K(2,3)=800-K(2,3)
    h_full = int(ew.imageHeight[0] if isinstance(ew.imageHeight, np.ndarray) else 800)
    K[1, 2] = h_full - K[1, 2]
    K[1, 1] = -K[1, 1]

    if i == 3:
        print(f"[RQ] fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
        print(f"[RQ] R_w2c={R_w2c}")
        print(f"[RQ] X0={X0}")

    return K, R_w2c, X0

def adapt_native(ew, krx0, i):
    X0 = np.array(ew.DLTtranslationVector[:, i], dtype=float)

    f  = float(ew.focalLengths[i])
    cx = float(ew.principalPoints[2 * i])
    cy = float(ew.principalPoints[2 * i + 1])
    K  = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    
    S = np.diag([1., -1., -1.])      # Rot_to_stan，native 无需 change_ax_dir
    K     = K @ S
    K     = K / K[2, 2]

    R_w2c = S @ ew.rotationMatrices[:, :, i]
    # 垂直翻转（图像 Y 轴约定，同 rq）
    K[1, 1] = -K[1, 1]
    K[1, 2] = 800 - K[1, 2]

    if i == 0:
        # K = np.array([[-f, 0, cx], [0, -f, cy], [0, 0, 1.0]])
        # R_w2c = np.diag([-1., -1., 1.]) @ R_w2c
        # print(f"[native cam4] K={K}")
        # print(f"[native cam4] R_w2c={R_w2c}")
        # print(f"[roni  Cam4] K= {krx0[:, 0:3, i]}")
        # print(f"[roni  Cam4] R_w2c={krx0[:, 3:6, i]}")

        # Verify: K_raw @ R_raw should equal H from coefs (before any convention flip)
        K_raw = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        R_raw = np.array(ew.rotationMatrices[:, :, i], dtype=float)
        coefs = ew.coefs[:, i]
        H_coefs = np.array([
            [coefs[0], coefs[1], coefs[2]],
            [coefs[4], coefs[5], coefs[6]],
            [coefs[8], coefs[9], coefs[10]]
        ])
        H_native = K_raw @ R_raw
        H_native_n = H_native / H_native[2, 2]
        H_coefs_n  = H_coefs  / H_coefs[2, 2]
        # print(f"K_raw @ R_raw (normalized) =\n{np.round(H_native_n, 4)}")
        # print(f"H_coefs       (normalized) =\n{np.round(H_coefs_n,  4)}")
        # print(f"diff norm (normalized)     = {np.linalg.norm(H_native_n - H_coefs_n):.4e}\n")

    return K, R_w2c, X0

METHODS  = ["roni", "rq", "native"]
ADAPTERS = {"roni": adapt_roni, "rq": adapt_rq, "native": adapt_native}

# ---------------- main ----------------
def main():
    debug_dir = BASE / "debug"; debug_dir.mkdir(exist_ok=True)

    ew   = sio.loadmat(str(BASE / "calibration_easyWandData.mat"),
                       struct_as_record=False, squeeze_me=True)["easyWandData"]
    krx0 = sio.loadmat(str(BASE / "camera_KRX0.mat"),
                       struct_as_record=False, squeeze_me=True)["camera"]
    n = int(ew.nCams)

    grays     = [cv2.imread(str(BASE / f"images/P{FRAME}CAM{j+1}.png"), 0) for j in range(n)]
    centroids = [mask_centroid(g) for g in grays]

    # shared target: triangulate from roni params (ground truth basis)
    rays = []
    for j in range(n):
        K, R, X0 = adapt_roni(ew, krx0, j)
        rays.append((X0, backproj_dir(K, R, *centroids[j])))
    target, tres = triangulate(rays)
    print(f"[shared target] {target}  residual={tres*1000:.3f} mm\n")

    # ----------------------------------------------------------------
    # Per-method summary (reprojection + rotation sanity)
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PER-METHOD SUMMARY")
    print("=" * 70)
    for name in METHODS:
        fn = ADAPTERS[name]
        # suppress adapt_rq debug prints in the summary pass
        params = []
        for j in range(n):
            if name == "rq":
                import io, contextlib
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    p = fn(ew, krx0, j)
                params.append(p)
            else:
                params.append(fn(ew, krx0, j))

        self_rays = [(X0, backproj_dir(K, R, *centroids[j]))
                     for j, (K, R, X0) in enumerate(params)]
        _, self_res = triangulate(self_rays)

        print(f"\n=== {name} ===  self-residual={self_res*1000:.3f} mm")
        errs = []
        for j, (K, R, X0) in enumerate(params):
            det, ortho, ok = gate(R)
            flag = "OK " if ok else "BAD"
            u, v, z = proj(K, R, X0, target)
            cu, cv = centroids[j]
            err = np.hypot(u - cu, v - cv)
            errs.append(err)

            r_px = max(int(round(K[0, 0] * SPHERE_R / z)), 1) if z > 0 else 0
            img = cv2.imread(str(BASE / f"images/P{FRAME}CAM{j+1}.png"))
            cv2.drawMarker(img, (int(cu), int(cv)), (0, 255, 0),
                           cv2.MARKER_CROSS, 30, 1)
            if z > 0:
                cv2.circle(img, (int(u), int(v)), r_px, (0, 0, 255), 2)
            cv2.imwrite(str(debug_dir / f"{name}_cam{j+1}.png"), img)

            print(f"  Cam{j+1} [{flag}] det={det:+.4f} ortho={ortho:.1e} "
                  f"u={u:7.1f} v={v:7.1f} z={z:.4f} reproj={err:6.2f}px r={r_px}px")
        print(f"  -> mean reproj error = {np.mean(errs):.2f} px")

    for j in range(n):
        if j == 3:  # cam4
                R_raw = np.array(ew.rotationMatrices[:, :, j], dtype=float)
                from scipy.io import loadmat
                krx0_mat = krx0
                R_roni = krx0_mat[:, 3:6, j]
                # print(f"[DEBUG cam4] R_raw (ew.rotationMatrices):\n{R_raw}")
                # print(f"[DEBUG cam4] R_roni (from mat):\n{R_roni}")
                # print(f"[DEBUG cam4] diag(-1,1,-1)@R_raw:\n{np.diag([-1.,1.,-1.]) @ R_raw}")
                # print(f"[DEBUG cam4] diff to R_roni:\n{np.diag([-1.,1.,-1.]) @ R_raw - R_roni}")
        


if __name__ == "__main__":
    main()