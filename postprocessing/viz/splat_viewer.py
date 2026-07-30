"""
统一的 viser 逐帧查看器 —— 合并原 debug/viz_splat_video.py 的"原始 splat.ply 多模式播放"
(points/mesh/gaussians/hull，网页勾选切换) 和原 postprocessing/viz/pointcloud_viewer.py 的
"处理后 T1/T2/T3 阶段csv对比"，用一个 Source 下拉在同一个 slider/Play 下切换：

  Source = Splat       读 outputs/{sweep}/{group}/fXXXX/{method}/{ts}/splat.ply
                        (--config 同 gpu/schedule/schedule.py 的 sweep config schema)，
                        display mode 见 run_viewer() 里 mode_checkboxes 的说明。坐标是
                        dataparser_transforms 反变换回的物理坐标 * display_scale(默认
                        1000，物理单位转mm)，up方向用 "-z"(仅影响viewer轨道相机，不改数据)。
  Source = Processed    读 data_root/fXXXX/.../gaussian_features_fXXXX{_labeled,_marked,}.csv，
                        T1原始/T2清理(if_keep)/T3聚类(part_label)分层对比。坐标是
                        calc_kinematics.md约定的实验室系，单位米，up="+z"。

两份数据的坐标系/单位不同（splat是nerfstudio反归一化后*1000的mm，processed是标定
实验室系的米），不做物理对齐叠加——切换Source只是换渲染哪一层+重置该Source自己的
相机取景，不假设两者数值可比。只提供其中一份数据时不显示Source下拉，跟只用原来
那一个脚本一样。

用法:
  # 只看训练splat(原debug/viz_splat_video.py用法):
  python -m postprocessing.viz.splat_viewer --config gpu/schedule/configs/xxx.json

  # 只看处理后csv(原pointcloud_viewer.py用法):
  python -m postprocessing.viz.splat_viewer --data-root outputs/xxx/G2b_G9 --start 0 --end 5

  # 两份一起看，同一个slider切帧+用Source下拉对比:
  python -m postprocessing.viz.splat_viewer --config gpu/schedule/configs/xxx.json \\
      --data-root outputs/xxx/G2b_G9
"""
import argparse
import dataclasses
import json
import struct
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import trimesh
import viser

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "gpu" / "schedule"))

import common  # noqa: E402  (gpu/schedule/common.py 的 sweep 路径规则)
from postprocessing.cleaning.viz_floater_check import normalize_frame_name  # noqa: E402
from postprocessing.viz._colors import DROP_COLOR, HULL_COLOR, PART_COLORS, RGB_BG  # noqa: E402
from postprocessing.viz._io import load_stage_csv  # noqa: E402
from utils.camera import CameraConfig  # noqa: E402
from utils.ply import load_ply, load_ply_with_attrs, unrescale, unrescale_covariance  # noqa: E402
from utils.viz import add_camera_axes, add_point_cloud, start_viser  # noqa: E402

DATASET_DIR = REPO_ROOT / "outputs" / "ctrl_009_002_8groups_100frames" / "G2b_G9"
RAW_DATA_DIR = REPO_ROOT / "data" / "ctrl_009_002"

_ICOSPHERE = trimesh.creation.icosphere(subdivisions=1)
_ICO_VERTS = np.asarray(_ICOSPHERE.vertices)  # (V,3) 单位球顶点
_ICO_FACES = np.asarray(_ICOSPHERE.faces)     # (F,3)


# ============================================================ Splat source ==

def _make_unlit_glb(mesh: trimesh.Trimesh) -> bytes:
    """trimesh导出vertex-colored mesh时不写任何material，GLTFLoader就用three.js自己的
    默认PBR材质渲染——对着场景灯光走标准光照模型，球面朝光源一侧亮、背光一侧暗，
    看起来像打了高光的实体球（"反光"）。这里给导出的glb手工patch一个KHR_materials_unlit
    材质（trimesh这个版本的导出器不支持写这个扩展），让每个面片只显示自己的顶点颜色、
    不受光照方向影响，视觉上更接近"这个点就是这个rgb"的平色小圆点。"""
    glb = mesh.export(file_type="glb")
    magic, version, total_len = struct.unpack_from("<4sII", glb, 0)
    assert magic == b"glTF"

    offset = 12
    json_len, json_type = struct.unpack_from("<II", glb, offset)
    offset += 8
    doc = json.loads(glb[offset:offset + json_len].decode("utf-8"))
    offset += json_len

    bin_chunk = glb[offset:] if offset < len(glb) else b""
    if bin_chunk:
        bin_len, bin_type = struct.unpack_from("<II", bin_chunk, 0)
        bin_chunk = bin_chunk[8:8 + bin_len]

    doc.setdefault("extensionsUsed", [])
    if "KHR_materials_unlit" not in doc["extensionsUsed"]:
        doc["extensionsUsed"].append("KHR_materials_unlit")
    materials = doc.setdefault("materials", [])
    mat_index = len(materials)
    materials.append({
        "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, 1.0]},
        "extensions": {"KHR_materials_unlit": {}},
    })
    for gltf_mesh in doc.get("meshes", []):
        for prim in gltf_mesh.get("primitives", []):
            prim["material"] = mat_index

    new_json = json.dumps(doc).encode("utf-8")
    new_json += b" " * ((4 - len(new_json) % 4) % 4)  # glTF要求chunk按4字节对齐

    out = struct.pack("<II", len(new_json), 0x4E4F534A) + new_json
    if bin_chunk:
        padded_bin = bin_chunk + b"\x00" * ((4 - len(bin_chunk) % 4) % 4)
        out += struct.pack("<II", len(padded_bin), 0x004E4942) + padded_bin

    return struct.pack("<4sII", b"glTF", 2, 12 + len(out)) + out


def add_gaussian_ellipsoids(server: viser.ViserServer, xyz: np.ndarray, covariances: np.ndarray,
                             colors: np.ndarray, name: str, n_sigma: float = 1.0):
    """把每个高斯画成一个还原了真实旋转+scale的椭球(icosphere变形而来)，所有点合并成一个mesh一次性发送。
    没有用viser自带的 _add_gaussian_splats(高斯渲染器)：实测(headless chrome截图对比，帧10/60像素完全一致)
    确认这个版本(viser 0.2.7)的高斯渲染器只在节点首次挂载时把buffer注册进渲染store，之后无论是覆盖
    同名节点还是先remove再add，画面都不会刷新——是viser本身这个实验性功能的bug，不是这边调用方式的问题。
    普通mesh/GLB走的是标准three.js加载路径，没有这个限制。
    xyz: (N,3) 物理显示坐标；covariances: (N,3,3) 同坐标系下的协方差；
    colors: (N,3) uint8，每个高斯一个颜色（不透明——GLB顶点透明度支持不稳定，
    透明度已经在上游按 rgb*opacity 与背景混合过，参考 points 模式的做法）；
    n_sigma: 椭球半轴 = n_sigma * sqrt(协方差特征值)，即几倍标准差。"""
    n = len(xyz)
    if n == 0:
        return None

    eigvals, eigvecs = np.linalg.eigh(covariances)  # (N,3) 升序, (N,3,3)
    semi_axes = n_sigma * np.sqrt(np.clip(eigvals, 0, None))  # (N,3)

    local = _ICO_VERTS[None, :, :] * semi_axes[:, None, :]  # (N,V,3)，按特征基坐标缩放
    world = np.einsum('nvj,nkj->nvk', local, eigvecs) + xyz[:, None, :]  # 转到世界系+平移

    n_verts = _ICO_VERTS.shape[0]
    all_verts = world.reshape(-1, 3)
    all_faces = (_ICO_FACES[None, :, :] + np.arange(n)[:, None, None] * n_verts).reshape(-1, 3)
    vertex_colors = np.repeat(colors.astype(np.uint8), n_verts, axis=0)

    mesh = trimesh.Trimesh(vertices=all_verts, faces=all_faces,
                            vertex_colors=vertex_colors, process=False)
    return server.scene.add_glb(name=name, glb_data=_make_unlit_glb(mesh))


def add_gaussians_native(server: viser.ViserServer, xyz: np.ndarray, covariances: np.ndarray,
                          rgb: np.ndarray, opacity: np.ndarray, name: str):
    """用 viser 自带的高斯渲染器(实验性API _add_gaussian_splats)画真正的splat效果(柔和的
    alpha混合椭圆，和splatfacto自带viewer一样)，而不是mesh椭球那种硬边实体。
    仅适用于渲染单帧静态画面：实测(headless chrome截图对比)确认这个节点一旦挂载后，
    同名覆盖或remove+add都不会刷新画面(viser 0.2.7这个实验性功能本身的bug)，
    所以不要在逐帧动画里调用它——动画播放请用 add_gaussian_ellipsoids。
    xyz: (N,3) 物理显示坐标；covariances: (N,3,3) 同坐标系下的协方差；
    rgb: (N,3) 0~1；opacity: (N,1) 0~1。"""
    if len(xyz) == 0:
        return None
    return server.scene._add_gaussian_splats(
        name=name,
        centers=np.ascontiguousarray(xyz),
        covariances=np.ascontiguousarray(covariances),
        rgbs=np.ascontiguousarray(rgb),
        opacities=np.ascontiguousarray(opacity),
    )


def _to_colormap(values: np.ndarray) -> np.ndarray:
    """把标量数组归一化并映射成浅灰(低)->深灰(高)的RGB颜色，用于viser point cloud。"""
    v = values.copy()
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)  # 用2~98分位数避免极端离群值压缩色阶
    v = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1)
    gray = ((1 - v) * 200 + 30).astype(np.uint8)  # 高值->深灰(30)，低值->浅灰(230)
    return np.stack([gray, gray, gray], axis=-1)


def blend_rgb_opacity(rgb01: np.ndarray, opacity: np.ndarray) -> np.ndarray:
    """rgb(0~1) 按 opacity 与白色背景(_colors.RGB_BG)混合，模拟透明度，返回uint8。
    splat/processed 两个source的rgb上色都走这一份，避免重复实现。"""
    blended = rgb01 * 255.0 * opacity[:, None] + RGB_BG[None, :] * (1 - opacity[:, None])
    return np.clip(blended, 0, 255).astype(np.uint8)


def _load_dataparser_transform(splat_dir: Path) -> tuple:
    with open(splat_dir / "dataparser_transforms.json") as f:
        dp = json.load(f)
    R_ns = np.array(dp["transform"])[:3, :3]
    t_ns = np.array(dp["transform"])[:3, 3]
    scale = float(dp["scale"])
    return R_ns, t_ns, scale


def load_splat_native(splat_dir: Path, display_scale: float = 1000.0) -> tuple:
    """给"原生高斯渲染(单帧)"按钮用：还原一帧的 xyz/covariances/rgb/opacity 到物理显示坐标系，
    不经过 rgb-opacity 与背景色混合（原生渲染器自己做alpha混合）。"""
    R_ns, t_ns, scale = _load_dataparser_transform(splat_dir)
    attrs = load_ply_with_attrs(splat_dir / "splat.ply")
    xyz = unrescale(attrs["xyz"], R_ns, t_ns, scale) * display_scale
    covariances = unrescale_covariance(attrs["rot"], attrs["scale_xyz"], R_ns, scale) * display_scale ** 2
    return xyz, covariances, attrs["rgb"], attrs["opacity"][:, None]


def load_splat_frame(splat_dir: Path, display_scale: float = 1000.0) -> dict:
    """读取一帧的高斯数据，还原到物理显示坐标系。返回 points/mesh 两种模式共用的原始字段，
    上色留到渲染时按当前 GUI 选择的 display mode / color by 动态计算，这样切换模式或
    color by 不需要重新读盘。"""
    R_ns, t_ns, scale = _load_dataparser_transform(splat_dir)
    attrs = load_ply_with_attrs(splat_dir / "splat.ply")
    xyz = unrescale(attrs["xyz"], R_ns, t_ns, scale) * display_scale
    covariances = unrescale_covariance(attrs["rot"], attrs["scale_xyz"], R_ns, scale) * display_scale ** 2
    return {"xyz": xyz, "covariances": covariances, "rgb": attrs["rgb"],
            "opacity": attrs["opacity"], "scale": attrs["scale"]}


def compute_point_colors(data: dict, color_by: str) -> np.ndarray:
    if color_by == "rgb":
        return blend_rgb_opacity(data["rgb"], data["opacity"])
    if color_by in ("opacity", "scale") and data.get(color_by) is not None:
        return _to_colormap(data[color_by])
    return np.tile(np.uint8([0, 150, 255]), (len(data["xyz"]), 1))


def compute_mesh_colors(data: dict) -> np.ndarray:
    return blend_rgb_opacity(data["rgb"], data["opacity"])


def load_hull_points(hull_path: Path, display_scale: float = 1000.0,
                      color: np.ndarray = HULL_COLOR) -> Optional[dict]:
    """读取一帧 visual hull 点云(data/{base_name}/fXXXX/init_points.ply)，缩放到显示坐标系
    并统一上色。路径不存在时返回 None，调用方按需缓存以避免重复探测磁盘。"""
    if not hull_path.exists():
        return None
    pts = load_ply(hull_path) * display_scale
    return {"xyz": pts, "colors": np.tile(color, (len(pts), 1))}


def load_splat_config(config_path: str) -> dict:
    required = ["name", "base_name", "param_sets", "frames"]
    with open(config_path) as f:
        cfg = json.load(f)
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config {config_path} missing required keys: {missing}")
    return cfg


def load_splat_source(cfg: dict, group: Optional[str], method: str,
                       frame_start: Optional[int], frame_end: Optional[int],
                       display_scale: float = 1000.0):
    """按 gpu/schedule/schedule.py 的sweep config，逐帧读取一个param_set的splat.ply。
    返回 (clouds, hull_path_for, find_splat_dir_for, scene_center)，clouds为空dict时
    调用方应视为该source不可用。"""
    sweep_name = cfg["name"]
    base_name = cfg["base_name"]
    param_sets = cfg["param_sets"]
    group = group or next(iter(param_sets))
    if group not in param_sets:
        raise ValueError(f"group={group!r} not in config param_sets: {list(param_sets)}")

    f_start = frame_start if frame_start is not None else cfg["frames"]["start"]
    f_end = frame_end if frame_end is not None else cfg["frames"]["end"]

    print(f"[splat] Loading splat.ply: outputs/{sweep_name}/{group}/fXXXX/{method}/...")
    clouds = {}
    for f in range(f_start, f_end):
        exp_name = common.exp_name_for(sweep_name, group, f)
        splat_dir = common.find_splat_dir(exp_name, method)
        if splat_dir is None or not (splat_dir / "splat.ply").exists():
            continue
        data = load_splat_frame(splat_dir, display_scale)
        clouds[f] = data
        print(f"  frame {f}: {len(data['xyz'])} points")

    if not clouds:
        print("[splat] No splat.ply found for any frame.")
        return {}, None, None, None

    def hull_path_for(frame_idx: int) -> Path:
        return common.data_dir_for(base_name, frame_idx) / "init_points.ply"

    def find_splat_dir_for(frame_idx: int):
        return common.find_splat_dir(common.exp_name_for(sweep_name, group, frame_idx), method)

    valid_frames = sorted(clouds.keys())
    scene_center = load_ply(hull_path_for(valid_frames[0])).mean(axis=0) * display_scale
    print(f"[splat] Scene center: {scene_center}")
    return clouds, hull_path_for, find_splat_dir_for, scene_center


# ========================================================= Processed source ==

def load_frame_stages(frame: str, data_root: Path) -> dict:
    """加载一帧csv，返回渲染各图层所需的numpy数组，按实际列决定哪些阶段可画。"""
    csv_path, df = load_stage_csv(frame, data_root)

    xyz = df[["x", "y", "z"]].to_numpy()
    rgb01 = df[["R", "G", "B"]].to_numpy(dtype=float)
    opacity = df["opacity"].to_numpy(dtype=float) if "opacity" in df.columns else np.ones(len(df))
    colors_orig = blend_rgb_opacity(rgb01, opacity)

    has_keep = "if_keep" in df.columns
    has_label = "part_label" in df.columns
    kept = df["if_keep"].astype(bool).to_numpy() if has_keep else np.ones(len(df), dtype=bool)
    part_label = df["part_label"].to_numpy() if has_label else None

    return {
        "csv_path": csv_path, "n_total": len(df),
        "xyz": xyz, "colors_orig": colors_orig, "kept": kept,
        "has_keep": has_keep, "has_label": has_label, "part_label": part_label,
    }


def load_frame_cameras(frame: str, raw_data_dir: Path) -> list:
    with open(raw_data_dir / frame / "transforms.json") as f:
        cam_frames = json.load(f)["frames"]
    cams = []
    for idx, cam_frame in enumerate(cam_frames):
        cam = CameraConfig.from_opengl(cam_frame)
        cam.cam_idx = idx + 1
        cams.append(cam)
    return cams


def load_processed_source(frames: list, data_root: Path):
    """按frame列表('f0061'形式)读取T1/T2/T3阶段csv，key用int帧号跟splat source对齐。
    单帧读取失败(csv缺失)会打印跳过，不中断整批。"""
    stages = {}
    for frame in frames:
        try:
            d = load_frame_stages(frame, data_root)
            stages[int(frame[1:])] = d
            print(f"[processed] [{frame}] n_total={d['n_total']}  csv={d['csv_path'].name}  "
                  f"has_if_keep={d['has_keep']}  has_part_label={d['has_label']}")
        except FileNotFoundError as e:
            print(f"[processed] [{frame}] SKIP: {e}")
    if not stages:
        print("[processed] No stage csv found for any frame.")
    return stages


def render_processed_layers(server: viser.ViserServer, data: dict, point_size: float,
                             display_scale: float = 1.0) -> list:
    """画T1(全部)/T2(kept/floaters)/T3(body/wing_L/wing_R, kept/dropped)分层点云，
    返回所有创建的handle列表，供调用方在切帧/切source时统一clear。

    display_scale: 跟splat source的物理坐标->mm显示坐标是同一个参数，只影响这里点的
    世界坐标(不改CSV本身单位)。CSV原始单位是米、整只果蝇bbox只有几mm(见
    calc_kinematics.md §0)——viser前端相机near clip固定在0.05世界单位(源码写死，
    python API不能改)，不做这个缩放的话相机稍微靠近一点果蝇就直接被near平面裁掉，
    实测(headless chrome截图对比)确认了这个坑：不缩放时processed source切换过去后
    画面完全是空的。"""
    xyz = data["xyz"] * display_scale
    colors_orig, kept = data["colors_orig"], data["kept"]
    handles = []

    handles.append(add_point_cloud(server, xyz, colors_orig, name="/T1_raw/all", point_size=point_size))

    if data["has_keep"]:
        handles.append(add_point_cloud(server, xyz[kept], colors_orig[kept],
                                        name="/T2_cleaned/kept", point_size=point_size))
        n_drop = int((~kept).sum())
        if n_drop > 0:
            handles.append(server.scene.add_point_cloud(
                name="/T2_cleaned/floaters", points=xyz[~kept],
                colors=np.tile(DROP_COLOR, (n_drop, 1)),
                point_size=point_size * 2.0, point_shape="diamond",
            ))

    if data["has_label"]:
        part_label = data["part_label"]
        for lab in ("body", "wing_L", "wing_R"):
            color_u8 = (np.array(PART_COLORS[lab]) * 255).astype(np.uint8)
            lab_mask = part_label == lab
            kept_mask = lab_mask & kept
            drop_mask = lab_mask & ~kept
            if kept_mask.any():
                handles.append(add_point_cloud(server, xyz[kept_mask], np.tile(color_u8, (int(kept_mask.sum()), 1)),
                                                name=f"/T3_labeled/{lab}_kept", point_size=point_size))
            if drop_mask.any():
                handles.append(server.scene.add_point_cloud(
                    name=f"/T3_labeled/{lab}_dropped", points=xyz[drop_mask],
                    colors=np.tile(color_u8, (int(drop_mask.sum()), 1)),
                    point_size=point_size * 2.0, point_shape="diamond",
                ))

    return [h for h in handles if h is not None]


# ================================================================ Unified UI ==

def run_viewer(
    splat_clouds: dict = None,
    hull_path_for: Optional[Callable[[int], Optional[Path]]] = None,
    splat_scene_center: Optional[np.ndarray] = None,
    find_splat_dir_for: Optional[Callable[[int], Optional[Path]]] = None,
    processed_frames: dict = None,
    processed_cameras: Optional[list] = None,
    display_scale: float = 1000.0,
    fps: int = 4,
    port: int = 8080,
    default_mode: str = "mesh",
    default_source: Optional[str] = None,
    ellipsoid_sigma: float = 1.0,
    hull_color: np.ndarray = HULL_COLOR,
    point_size: float = 0.00008,
) -> None:
    """在 viser 里做逐帧查看，Source 下拉在"splat"(原始splat.ply多模式播放)和
    "processed"(T1/T2/T3阶段csv对比)之间切换；只传其中一份数据时不显示下拉，
    行为等价于只用原来那一个脚本。

    splat_clouds/hull_path_for/splat_scene_center/find_splat_dir_for : splat source，
        语义同原 debug/viz_splat_video.py::run_splat_video_viewer。
    processed_frames  : frame_idx(int) -> load_frame_stages() 结果，processed source。
    processed_cameras : 可选，画一次相机坐标轴(实验室系，取自任意一帧的transforms.json)。

    Display mode(仅Source=splat时生效，网页GUI里打钩切换，互斥):
      points     — 每个高斯只画质心一个点，可实时切换 Color by（rgb/opacity/scale/none）
      mesh       — 还原每个高斯真实的旋转+scale，合并成一个mesh画成椭球（颜色固定用rgb+opacity blend）
      gaussians  — viser 原生高斯渲染器，真正的柔和alpha splat效果，但只能单帧静态渲染
                   （挂载后无法刷新，是viser 0.2.7的bug，见 add_gaussians_native），
                   勾选后需要点击 "Render" 按钮才会画出当前帧
      hull       — 对应帧 init_points.ply 的 visual hull 点云（固定绿色）
    """
    splat_clouds = splat_clouds or {}
    processed_frames = processed_frames or {}
    have_splat = bool(splat_clouds)
    have_processed = bool(processed_frames)
    if not have_splat and not have_processed:
        print("[Error] No frames to visualize (neither splat nor processed source provided).")
        return

    valid_frames = sorted(set(splat_clouds) | set(processed_frames))
    print(f"Loaded {len(valid_frames)} frame(s) total "
          f"({len(splat_clouds)} splat, {len(processed_frames)} processed).")

    source = default_source or ("splat" if have_splat else "processed")

    hull_cache = {}

    def get_hull_frame(frame_idx: int) -> Optional[dict]:
        if hull_path_for is None:
            return None
        if frame_idx not in hull_cache:
            hull_path = hull_path_for(frame_idx)
            hull_cache[frame_idx] = (load_hull_points(hull_path, display_scale, hull_color)
                                      if hull_path is not None else None)
        return hull_cache[frame_idx]

    server = start_viser(port=port)
    server.scene.set_up_direction("-z" if source == "splat" else "+z")

    # processed source 的CSV是实验室系下的真实物理坐标(米，整只果蝇bbox只有几mm，见
    # calc_kinematics.md §0)。viser前端相机near clip固定在0.05世界单位(写死在js bundle里，
    # python API不能改)——按原始米制坐标摆相机，取景距离必然小于0.05就会被near平面整个裁掉，
    # 实测(headless chrome截图对比)踩过这个坑：不缩放时切到processed画面完全是空的。
    # 这里跟splat source用同一个display_scale把processed的点云+相机坐标轴一起放大到mm量级
    # 显示（只影响这里的渲染坐标，不改CSV/analysis用的原始米制数据），两个source视觉尺度
    # 才可比，near-plane也不会咬到。
    if have_processed and processed_cameras:
        scaled_cameras = [dataclasses.replace(c, X0=c.X0 * display_scale) for c in processed_cameras]
        add_camera_axes(server, scaled_cameras)

    processed_scene_center, processed_extent = None, None
    if have_processed:
        any_xyz = next(iter(processed_frames.values()))["xyz"] * display_scale
        processed_scene_center = any_xyz.mean(axis=0)
        processed_extent = max(float(np.linalg.norm(any_xyz.max(0) - any_xyz.min(0))), 1e-6)

    connected_clients = []

    def apply_camera_for_splat(client):
        if splat_scene_center is None:
            return
        cam_offset = np.array([0.0, -15.0, 10.0])
        client.camera.position = splat_scene_center + cam_offset
        client.camera.look_at = splat_scene_center
        client.camera.up_direction = (0.0, 0.0, -1.0)

    def apply_camera_for_processed(client):
        if processed_scene_center is None:
            return
        d = processed_extent * 2.5  # 经验值：略大于bbox对角线，保证整只果蝇进入视野
        cam_offset = np.array([0.0, -d, d * 0.6])
        client.camera.position = processed_scene_center + cam_offset
        client.camera.look_at = processed_scene_center
        client.camera.up_direction = (0.0, 0.0, 1.0)

    def apply_camera_for_source(client, src: str):
        if src == "splat":
            apply_camera_for_splat(client)
        else:
            apply_camera_for_processed(client)

    @server.on_client_connect
    def _(client):
        connected_clients.append(client)
        apply_camera_for_source(client, source)

    @server.on_client_disconnect
    def _(client):
        if client in connected_clients:
            connected_clients.remove(client)

    slider = server.gui.add_slider(
        "Frame", min=0, max=len(valid_frames) - 1, step=1, initial_value=0
    )
    play_checkbox = server.gui.add_checkbox("Play", initial_value=False)

    source_dropdown = None
    if have_splat and have_processed:
        source_dropdown = server.gui.add_dropdown("Source", options=["splat", "processed"], initial_value=source)

    # Display mode(仅Source=splat时用): 网页上用打钩选择，互斥（勾一个自动取消其它），
    # 选中即立刻切换渲染，避免旧模式的图层叠加残留。
    mode_checkboxes = {
        "points":    server.gui.add_checkbox("Points", initial_value=(default_mode == "points")),
        "mesh":      server.gui.add_checkbox("Ellipsoids", initial_value=(default_mode == "mesh")),
        "gaussians": server.gui.add_checkbox("Gaussians", initial_value=(default_mode == "gaussians")),
        "hull":      server.gui.add_checkbox("Hull", initial_value=(default_mode == "hull")),
    }
    color_by_dropdown = server.gui.add_dropdown("Color by", options=["rgb", "opacity", "scale", "none"], initial_value="rgb")
    native_button = server.gui.add_button("Render")

    mode_state = {"mode": default_mode}
    source_state = {"source": source, "updating": False}
    current_layers = []
    native_state = {"counter": 0}

    def clear_layers():
        for h in current_layers:
            h.remove()
        current_layers.clear()

    def update_controls_visibility():
        is_splat = source_state["source"] == "splat"
        for cb in mode_checkboxes.values():
            cb.visible = is_splat
        color_by_dropdown.visible = is_splat and (mode_state["mode"] == "points")
        native_button.visible = is_splat and (mode_state["mode"] == "gaussians")

    def render_splat_frame(idx: int):
        mode = mode_state["mode"]
        if mode is None or mode == "gaussians":
            return  # gaussians 模式是单帧静态渲染，靠按钮触发，不跟着slider自动刷新

        f = valid_frames[idx]
        if mode == "hull":
            hull = get_hull_frame(f)
            if hull is None:
                return
            h = add_point_cloud(server, hull["xyz"], hull["colors"], name="/anim/splat", point_size=0.00005)
            if h is not None:
                current_layers.append(h)
            return

        if f not in splat_clouds:
            return
        data = splat_clouds[f]
        if mode == "mesh":
            colors = compute_mesh_colors(data)
            h = add_gaussian_ellipsoids(server, data["xyz"], data["covariances"], colors,
                                         name="/anim/splat", n_sigma=ellipsoid_sigma)
        elif mode == "points":
            colors = compute_point_colors(data, color_by_dropdown.value)
            h = add_point_cloud(server, data["xyz"], colors, name="/anim/splat", point_size=0.03)
        else:
            h = None
        if h is not None:
            current_layers.append(h)

    def render_processed_frame(idx: int):
        f = valid_frames[idx]
        data = processed_frames.get(f)
        if data is None:
            return
        current_layers.extend(render_processed_layers(server, data, point_size, display_scale))

    def render_current():
        clear_layers()
        if source_state["source"] == "splat":
            render_splat_frame(slider.value)
        else:
            render_processed_frame(slider.value)

    def set_source(new_source: str):
        source_state["source"] = new_source
        server.scene.set_up_direction("-z" if new_source == "splat" else "+z")
        for c in connected_clients:
            apply_camera_for_source(c, new_source)
        update_controls_visibility()
        render_current()

    if source_dropdown is not None:
        @source_dropdown.on_update
        def _(_):
            if source_state["updating"]:
                return
            set_source(source_dropdown.value)

    def make_mode_callback(name: str):
        def _(_):
            if source_state["source"] != "splat":
                return
            if mode_checkboxes[name].value:
                for other, cb in mode_checkboxes.items():
                    if other != name:
                        cb.value = False
                mode_state["mode"] = name
                update_controls_visibility()
                render_current()
            elif mode_state["mode"] == name:
                # 取消了当前唯一勾选的模式：不再画任何图层
                mode_state["mode"] = None
                update_controls_visibility()
                clear_layers()
        return _

    for name, cb in mode_checkboxes.items():
        cb.on_update(make_mode_callback(name))

    @color_by_dropdown.on_update
    def _(_):
        if source_state["source"] == "splat" and mode_state["mode"] == "points":
            render_current()

    update_controls_visibility()

    @native_button.on_click
    def _(_):
        if source_state["source"] != "splat" or mode_state["mode"] != "gaussians" or find_splat_dir_for is None:
            return
        f = valid_frames[slider.value]
        splat_dir = find_splat_dir_for(f)
        if splat_dir is None:
            return
        xyz, covariances, rgb, opacity = load_splat_native(splat_dir, display_scale)
        clear_layers()
        native_state["counter"] += 1
        name = f"/anim/splat_native_{native_state['counter']}"
        # 每次都换新节点名强制前端重新挂载：同名覆盖/remove+add都无法刷新画面，
        # 是viser 0.2.7原生高斯渲染器实验性功能自身的bug，不是调用方式问题。
        h = add_gaussians_native(server, xyz, covariances, rgb, opacity, name=name)
        if h is not None:
            current_layers.append(h)
        print(f"[native] rendered frame {f} at {name} (静态快照，改变slider后需再次点击按钮才会刷新)")

    @slider.on_update
    def _(_):
        render_current()

    render_current()

    print(f"Viser running — drag slider or check 'Play' to animate over {len(valid_frames)} frames.")
    if source_dropdown is not None:
        print("用 Source 下拉在 splat(原始splat.ply)/processed(T1/T2/T3阶段csv)之间切换。")
    try:
        while True:
            if play_checkbox.value:
                next_idx = (slider.value + 1) % len(valid_frames)
                slider.value = next_idx  # 触发 on_update，自动渲染
                time.sleep(1.0 / fps)
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass


# =================================================================== CLI ==

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None,
                         help="splat source: sweep config json路径，schema同 gpu/schedule/schedule.py")
    parser.add_argument("--group", type=str, default=None, help="splat source: param_set名，不传取config第一个")
    parser.add_argument("--method", choices=["splatfacto", "splatfacto-checkpoint"], default="splatfacto",
                         help="splat source: 训练用的method(对应schedule.py的--debug-checkpoint)")
    parser.add_argument("--frame-start", type=int, default=None, help="splat source: 覆盖config[frames][start]")
    parser.add_argument("--frame-end", type=int, default=None, help="splat source: 覆盖config[frames][end]")

    parser.add_argument("--data-root", type=str, default=None,
                         help="processed source: 存放各帧gaussian_features_*.csv的数据集根目录")
    parser.add_argument("--raw-data-dir", type=str, default=str(RAW_DATA_DIR),
                         help="processed source: 存放各帧transforms.json(相机位姿)的原始数据目录")
    parser.add_argument("--frame", type=str, default=None, help="processed source单帧模式，如 f0061 或 61")
    parser.add_argument("--start", type=int, default=None, help="processed source多帧模式起始帧号(含)")
    parser.add_argument("--end", type=int, default=None, help="processed source多帧模式结束帧号(含)")
    parser.add_argument("--no-cameras", action="store_true", help="processed source: 不画相机坐标轴")
    parser.add_argument("--point-size", type=float, default=0.00008, help="processed source点大小(米)")

    parser.add_argument("--fps", type=int, default=8, help="播放帧率(回放速度，不必等于拍摄fps)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mode", choices=["points", "mesh", "gaussians", "hull"], default="mesh",
                         help="splat source初始display mode")
    args = parser.parse_args()

    if not args.config and not args.data_root:
        parser.error("需要指定 --config(splat source) 和/或 --data-root(processed source) 至少一个")

    splat_clouds, hull_path_for, find_splat_dir_for, splat_scene_center = {}, None, None, None
    if args.config:
        cfg = load_splat_config(args.config)
        splat_clouds, hull_path_for, find_splat_dir_for, splat_scene_center = load_splat_source(
            cfg, args.group, args.method, args.frame_start, args.frame_end,
        )

    processed_frames, processed_cameras = {}, None
    if args.data_root:
        data_root = Path(args.data_root)
        raw_data_dir = Path(args.raw_data_dir)
        if args.frame is not None:
            frames = [normalize_frame_name(args.frame)]
        elif args.start is not None and args.end is not None:
            frames = [f"f{i:04d}" for i in range(args.start, args.end + 1)]
        else:
            parser.error("使用 --data-root 时需要指定 --frame，或者 --start/--end")
            return
        print(f"[processed] Loading {len(frames)} frame(s) from {data_root} ...")
        processed_frames = load_processed_source(frames, data_root)
        if processed_frames and not args.no_cameras:
            try:
                any_frame = f"f{next(iter(processed_frames)):04d}"
                processed_cameras = load_frame_cameras(any_frame, raw_data_dir)
            except FileNotFoundError as e:
                print(f"[警告] 找不到相机位姿，跳过相机坐标轴: {e}")

    if not splat_clouds and not processed_frames:
        sys.exit("[Error] No data loaded from either source.")

    run_viewer(
        splat_clouds=splat_clouds,
        hull_path_for=hull_path_for,
        splat_scene_center=splat_scene_center,
        find_splat_dir_for=find_splat_dir_for,
        processed_frames=processed_frames,
        processed_cameras=processed_cameras,
        fps=args.fps,
        port=args.port,
        default_mode=args.mode,
        point_size=args.point_size,
    )


if __name__ == "__main__":
    main()
