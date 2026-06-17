from dataclasses import dataclass
import numpy as np
from scipy.linalg import rq as scipy_rq

@dataclass
class CameraConfig:
    """
    EasyWand calibration data --> Pinhole camera mdoel in OpenCV convention --> OpenGL transforms.json for nerfstudio

    Stored:
      cam_idx   : camera index (1-based)
      K         : 3x3 intrinsic matrix
      R_w2c     : 3x3 rotation, world -> camera
      X0        : (3,) camera centre in world frame

    Derived (properties):
      R_c2w           : 3x3 rotation, camera -> world  (= R_w2c.T)
      t               : (3,) translation in [R|t] form  (= -R_w2c @ X0)
      fx, fy, cx, cy  : intrinsic scalars from K
      transform_opencv: 4x4 camera-to-world, OpenCV convention (Y down, Z forward)
      transform_opengl: 4x4 camera-to-world, OpenGL convention (Y up,   Z backward)
                        used by Nerfstudio and Viser
    """
    cam_idx: int
    K:       np.ndarray   # (3, 3)
    R_w2c:   np.ndarray   # (3, 3)  world -> camera
    X0:      np.ndarray   # (3,)    camera centre, world frame
    w:       int        # image width  (pixels)
    h:       int        # image height (pixels)

    # ---------------------------------------------------------------- derived --
    @property
    def R_c2w(self) -> np.ndarray:
        """3x3 rotation camera -> world."""
        return self.R_w2c.T

    @property
    def t(self) -> np.ndarray:
        """Translation vector in standard [R | t] form: t = -R_w2c @ X0."""
        return -self.R_w2c @ self.X0

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def transform_opencv(self) -> np.ndarray:
        """4x4 camera-to-world in OpenCV convention (Y down, Z forward)."""
        M = np.eye(4)
        M[:3, :3] = self.R_c2w
        M[:3,  3] = self.X0
        return M

    @property
    def transform_opengl(self) -> np.ndarray:
        """4x4 camera-to-world in OpenGL convention (Y up, Z backward).
        Required by Nerfstudio and Viser.
        Derived by flipping Y and Z columns of R_c2w."""
        M = np.eye(4)
        FLIP = np.diag([1., -1., -1.])
        M[:3, :3] = self.R_c2w @ FLIP
        M[:3,  3] = self.X0
        return M

    def apply_crop(self, x_min: int, y_min: int, w_new: int, h_new: int) -> None:
        """
        In-place update of intrinsics after cropping the image.
        x_min, y_min : top-left corner of the crop window (pixels, original image frame)
        w_new, h_new : cropped image dimensions
        Only cx, cy, w, h change; fx, fy, R_w2c, X0 are unaffected by cropping.
        """
        self.K[0, 2] -= x_min
        self.K[1, 2] -= y_min
        self.w = w_new
        self.h = h_new
        
    @classmethod
    def easywand_dlt(cls, ew, i: int)-> "CameraConfig":
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

        w = int(ew.imageWidth[0]  if isinstance(ew.imageWidth,  np.ndarray) else 1280)
        h = int(ew.imageHeight[0] if isinstance(ew.imageHeight, np.ndarray) else 800)
        return cls(cam_idx=i + 1, K=K, R_w2c=R_w2c, X0=X0, w=w, h=h)

    @classmethod
    def from_opengl(cls, frame: dict) -> "CameraConfig":
        """
        Build CameraConfig from a Nerfstudio transforms.json frame dict (OpenGL convention).
        Reverses the OpenCV -> OpenGL conversion applied in transform_opengl.
        frame: single entry from transforms.json['frames']
        """
        M    = np.array(frame["transform_matrix"])
        X0   = M[:3, 3]
        FLIP = np.diag([1., -1., -1.])
        # R_c2w_opengl = R_c2w_opencv @ FLIP  =>  R_c2w_opencv = R_c2w_opengl @ FLIP
        R_w2c = (M[:3, :3] @ FLIP).T

        K = np.array([
            [frame["fl_x"], 0.,           frame["cx"]],
            [0.,            frame["fl_y"], frame["cy"]],
            [0.,            0.,            1.          ]
        ])
        w = int(frame["w"])
        h = int(frame["h"])
        cam_idx = 0   # unknown from json; caller can override after construction
        return cls(cam_idx=cam_idx, K=K, R_w2c=R_w2c, X0=X0, w=w, h=h)
    