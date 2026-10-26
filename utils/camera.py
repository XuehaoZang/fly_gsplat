"""相机：EasyWand DLT -> OpenCV (K, R_w2c, X0) -> OpenGL c2w（nerfstudio transforms.json 用）。

世界坐标系就是 easyWand 标定坐标系（单位米；真实 4 相机装置里 -z 朝上）。
OpenCV 相机系：x 右、y 下、z 朝前；OpenGL 相机系：x 右、y 上、z 朝后。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import rq as scipy_rq

GL_FROM_CV = np.diag([1.0, -1.0, -1.0])


@dataclass
class CameraConfig:
    cam_idx: int
    K: np.ndarray      # (3,3)
    R_w2c: np.ndarray  # (3,3) 世界 -> OpenCV 相机
    X0: np.ndarray     # (3,) 相机中心（世界坐标）
    w: int
    h: int

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
    def transform_opengl(self) -> np.ndarray:
        """4x4 OpenGL c2w：旋转列为相机 x/y/z 轴在世界中的方向，平移为相机中心。"""
        M = np.eye(4)
        M[:3, :3] = self.R_w2c.T @ GL_FROM_CV
        M[:3, 3] = self.X0
        return M

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(N,3) 世界点 -> (u, v, 深度)。"""
        Xc = (self.R_w2c @ (np.asarray(X, dtype=float) - self.X0).T)
        z = Xc[2]
        return self.fx * Xc[0] / z + self.cx, self.fy * Xc[1] / z + self.cy, z

    @classmethod
    def easywand_dlt(cls, ew, i: int) -> "CameraConfig":
        """easyWand 的第 i 台相机（0 起）：11 参数 DLT 做 RQ 分解得到 K、R，再对齐到 OpenCV 约定。

        DLT 的图像 v 轴向上（原点在左下角），这里翻成 OpenCV 的向下（原点在左上角）；
        RQ 分解的符号歧义用 easyWand 自己给出的旋转矩阵来定。
        """
        coefs = ew.coefs[:, i]
        ew_rot_w2c = ew.rotationMatrices[:, :, i]
        H = np.array([[coefs[0], coefs[1], coefs[2]],
                      [coefs[4], coefs[5], coefs[6]],
                      [coefs[8], coefs[9], coefs[10]]])
        h = np.array([coefs[3], coefs[7], 1.0])

        X0 = -np.linalg.inv(H) @ h
        K_raw, R_raw = scipy_rq(H)
        K_raw = K_raw / K_raw[2, 2]

        s = np.sign(np.diag(ew_rot_w2c @ R_raw.T))
        s[s == 0] = 1.0
        S = np.diag([1., -1., -1.]) @ np.diag(s)
        K = K_raw @ S
        K = K / K[2, 2]
        R_w2c = S @ R_raw

        w = int(ew.imageWidth[0] if isinstance(ew.imageWidth, np.ndarray) else 1280)
        h_img = int(ew.imageHeight[0] if isinstance(ew.imageHeight, np.ndarray) else 800)
        K[1, 2] = h_img - K[1, 2]
        K[1, 1] = -K[1, 1]
        if K[0, 0] < 0:
            K[0, 0] = -K[0, 0]
            R_w2c[0, :] = -R_w2c[0, :]
        if K[1, 1] < 0:
            K[1, 1] = -K[1, 1]
            R_w2c[1, :] = -R_w2c[1, :]
        return cls(cam_idx=i + 1, K=K, R_w2c=R_w2c, X0=X0, w=w, h=h_img)

    @classmethod
    def from_opengl(cls, frame: dict, cam_idx: int = 0) -> "CameraConfig":
        """transforms.json 里的一项 -> CameraConfig。"""
        M = np.array(frame["transform_matrix"])
        K = np.array([[frame["fl_x"], 0., frame["cx"]],
                      [0., frame["fl_y"], frame["cy"]],
                      [0., 0., 1.]])
        return cls(cam_idx=cam_idx, K=K, R_w2c=(M[:3, :3] @ GL_FROM_CV).T, X0=M[:3, 3],
                   w=int(frame["w"]), h=int(frame["h"]))
