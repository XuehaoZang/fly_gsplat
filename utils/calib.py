import numpy as np
from scipy import ndimage
from scipy.linalg import rq as scipy_rq

# camera projection maths
def proj(K, R, X0, X):
    """OpenCV projection -> (u, v, depth)."""
    xc = R @ (X - X0)
    p  = K @ xc
    return p[0] / p[2], p[1] / p[2], xc[2]

def backproj(K, R, u, v):
    """Pixel (u,v) -> unit ray direction in world frame."""
    d = R.T @ (np.linalg.inv(K) @ np.array([u, v, 1.0]))
    return d / np.linalg.norm(d)

def triangulate(rays):
    """Least-squares intersection of rays [(origin, unit_dir), ...].
    Returns (point_3d, mean_residual_metres)."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for C, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ C
    X   = np.linalg.solve(A, b)
    res = np.mean([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - C))
                   for C, d in rays])
    return X, res

def check_ortho(R):
    """Rotation sanity check. Returns (det, ortho_norm, passed)."""
    det   = np.linalg.det(R)
    ortho = np.linalg.norm(R @ R.T - np.eye(3))
    return det, ortho, (abs(det - 1.0) < 1e-3 and ortho < 1e-3)

def mask_centroid(gray):
    """Returns (u, v) centroid of non-zero pixels."""
    v, u = ndimage.center_of_mass(gray > 0)
    return u, v

def rq_decompose_dlt(ew, i: int):
        """
        Decompose EasyWand DLT coefs into (K, R_w2c, X0) in openCV convention
            ew : easyWandData struct (loaded via scipy.io)
            i  : 0-based camera index
        
        Returns K (positive fx/fy), R_w2c (det=+1), X0 (camera centre)."""
        coefs      = ew.coefs[:, i]
        ew_rot_w2c = ew.rotationMatrices[:, :, i]

        H = np.array([[coefs[0], coefs[1], coefs[2]],
                    [coefs[4], coefs[5], coefs[6]],
                    [coefs[8], coefs[9], coefs[10]]])
        h = np.array([coefs[3], coefs[7], 1.0])

        X0    = -np.linalg.inv(H) @ h
        K_raw, R_raw = scipy_rq(H)
        K_raw = K_raw / K_raw[2, 2]

        # resolve RQ sign ambiguity by aligning to ew.rotationMatrices
        s = np.sign(np.diag(ew_rot_w2c @ R_raw.T))
        s[s == 0] = 1.0
        S = np.diag([1., -1., -1.]) @ np.diag(s)   # Rot_to_stan @ Rot_to_ew

        K     = K_raw @ S;  K = K / K[2, 2]
        R_w2c = S @ R_raw

        # vertical flip: image Y-axis orientation (EasyWand -> OpenCV)
        h_img   = int(ew.imageHeight[0] if isinstance(ew.imageHeight, np.ndarray) else 800)
        K[1, 2] = h_img - K[1, 2]
        K[1, 1] = -K[1, 1]

        return K, R_w2c, X0