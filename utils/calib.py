import numpy as np
from scipy import ndimage

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
