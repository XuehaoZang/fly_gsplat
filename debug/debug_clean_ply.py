import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud(
    "outputs/ctrl_009_002/splatfacto/2026-06-21_155910/splat.ply")
print(f"原始点数: {len(pcd.points)}")

# 统计离群点去除
pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.05)
print(f"去除后点数: {len(pcd_clean.points)}")

o3d.io.write_point_cloud(
    "outputs/ctrl_009_002/splatfacto/2026-06-21_155910/splat_clean.ply", pcd_clean)
print("保存完成")