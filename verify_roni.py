import numpy as np, scipy.io as sio, cv2
from pathlib import Path

base = Path("./data/removed_002_009")
mat = sio.loadmat(str(base / "camera_KRX0.mat"), struct_as_record=False, squeeze_me=True)
cam = mat['camera']   # shape (3,7,4): [K(3x3) | R(3x3) | X0(3x1)]

# 先算 target_center: 用 Roni 的 R,X0 三角化四相机光轴交点
def proj(K, R, X0, X):
    # Roni: x ~ K * R * (X - X0)   (R 是 world->cam)
    xc = R @ (X - X0)
    uv = K @ xc
    return uv[0]/uv[2], uv[1]/uv[2], xc[2]

# 简单取四相机光心平均深度方向交点;先用一个手填的中心试
# 光心 = X0;光轴方向(world->cam 的 R, 第3行是相机 Z 在世界的方向)
from scipy import ndimage
# 用每个相机 mask 质心反投影成射线,三角化果蝇真实 3D 位置
rays = []
for j in range(4):
    K = cam[:, 0:3, j]; R = cam[:, 3:6, j]; X0 = cam[:, 6, j]
    img = cv2.imread(str(base / f"images/P2000CAM{j+1}.png"), 0)
    vc, uc = ndimage.center_of_mass(img > 0)        # mask 质心 (v,u)
    # 反投影: 像素 -> 相机系方向 -> 世界系
    d_cam = np.linalg.inv(K) @ np.array([uc, vc, 1.0])
    d_world = R.T @ d_cam                            # R 是 world->cam, 转置回世界
    d_world /= np.linalg.norm(d_world)
    rays.append((X0, d_world))
    print(f"Cam{j+1} centroid uv=({uc:.0f},{vc:.0f})")

A = np.zeros((3,3)); b = np.zeros(3)
for C, d in rays:
    P = np.eye(3) - np.outer(d, d)
    A += P; b += P @ C
target = np.linalg.solve(A, b)
res = np.mean([np.linalg.norm((np.eye(3)-np.outer(d,d))@(target-C)) for C,d in rays])
print(f"target_center = {target}, residual = {res*1000:.3f} mm")

for j in range(4):
    K = cam[:, 0:3, j]; R = cam[:, 3:6, j]; X0 = cam[:, 6, j]
    u, v, z = proj(K, R, X0, target)
    img = cv2.imread(str(base / f"images/P2000CAM{j+1}.png"))
    cv2.circle(img, (int(u), int(v)), 15, (0,0,255), 2)
    cv2.imwrite(str(base / f"debug/roni_proj_cam{j+1}.png"), img)
    print(f"Cam{j+1}: u={u:.0f} v={v:.0f} depth={z:.4f}")