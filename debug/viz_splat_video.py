"""
读取 outputs/{BASE_NAME}/{GROUP_NAME}/f{N:04d}/splatfacto-checkpoint/{timestamp}/splat.ply
（每帧取该帧目录下最新 timestamp），在 viser 里做逐帧播放/拖动查看。
"""
import time
import json
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.camera import CameraConfig
from utils.ply import load_ply, load_ply_with_attrs, unrescale
from utils.viz import start_viser, add_point_cloud

BASE_NAME   = "ctrl_009_002_ratio3_sh0_full"
GROUP_NAME  = "ratio3_sh0"
DATA_NAME   = "ctrl_009_002"  
FRAME_RANGE = range(0, 640)
FPS         = 16  # 播放帧率（不必等于拍摄fps，这里是回放速度）
DISPLAY_SCALE = 1000.0  # 米 -> 毫米，让点云在 viser 默认场景尺度下显示为"正常大小"

COLOR_BY = "rgb"  # "opacity" | "scale" | "rgb" | None（不上色，用统一蓝色）
RGB_BG   = np.array([255, 255, 255], dtype=np.float64)  # rgb 模式下，opacity 越低越接近该背景色（模拟透明度）

def _to_colormap(values: np.ndarray) -> np.ndarray:
    """把标量数组归一化并映射成浅灰(低)->深灰(高)的RGB颜色，用于viser point cloud。"""
    v = values.copy()
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)  # 用2~98分位数避免极端离群值压缩色阶
    v = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1)
    gray = ((1 - v) * 200 + 30).astype(np.uint8)  # 高值->深灰(30)，低值->浅灰(230)
    return np.stack([gray, gray, gray], axis=-1)

def load_splat_physical(splat_dir: Path) -> tuple:
    """返回 (points, colors)，colors 按 COLOR_BY 指定的属性上色。"""
    with open(splat_dir / "dataparser_transforms.json") as f:
        dp = json.load(f)
    R_ns = np.array(dp["transform"])[:3, :3]
    t_ns = np.array(dp["transform"])[:3, 3]
    scale = float(dp["scale"])

    attrs = load_ply_with_attrs(splat_dir / "splat.ply")
    pts_physical = unrescale(attrs["xyz"], R_ns, t_ns, scale) * DISPLAY_SCALE

    if COLOR_BY == "rgb":
        rgb = attrs["rgb"]  # (N, 3) float, 0~1
        opacity = attrs["opacity"][:, None]  # (N, 1) float, 0~1
        blended = rgb * 255.0 * opacity + RGB_BG[None, :] * (1 - opacity)
        colors = np.clip(blended, 0, 255).astype(np.uint8)
    else:
        color_vals = attrs.get(COLOR_BY) if COLOR_BY else None
        if color_vals is not None:
            colors = _to_colormap(color_vals)
        else:
            colors = np.tile(np.uint8([0, 150, 255]), (len(pts_physical), 1))

    return pts_physical, colors

def find_latest_splat_dir(frame_idx: int) -> Path | None:
    frame_dir = Path("outputs") / BASE_NAME / GROUP_NAME / f"f{frame_idx:04d}" / "splatfacto-checkpoint"
    if not frame_dir.exists():
        return None
    timestamps = sorted(frame_dir.iterdir())
    if not timestamps:
        return None
    splat_dir = timestamps[-1]
    return splat_dir if (splat_dir / "splat.ply").exists() else None

def get_scene_center(frame_idx: int) -> np.ndarray:
    data_dir = Path("data") / DATA_NAME / f"f{frame_idx:04d}"
    hull_pts = load_ply(data_dir / "init_points.ply")
    return hull_pts.mean(axis=0) * DISPLAY_SCALE


def main():
    print("Loading point clouds...")
    clouds = {}
    for f in FRAME_RANGE:
        splat_dir = find_latest_splat_dir(f)
        if splat_dir is None:
            continue
        pts, colors = load_splat_physical(splat_dir)
        clouds[f] = (pts, colors)
        print(f"  frame {f}: {len(pts)} points")

    if not clouds:
        print("[Error] No splat.ply found for any frame.")
        return

    valid_frames = sorted(clouds.keys())
    print(f"Loaded {len(valid_frames)} frames.")

    scene_center = get_scene_center(valid_frames[0])
    print(f"Scene center (from cameras): {scene_center}")

    server = start_viser()
    server.scene.set_up_direction("-z")

    @server.on_client_connect
    def _(client):
        # 相机放在中心正上方一点、往下看，距离按场景尺度（毫米级）设一个近距离
        cam_offset = np.array([0.0, -15.0, 10.0])
        client.camera.position = scene_center + cam_offset
        client.camera.look_at = scene_center
        client.camera.up_direction = (0.0, 0.0, -1.0)

    slider = server.gui.add_slider(
        "Frame", min=0, max=len(valid_frames) - 1, step=1, initial_value=0
    )
    play_checkbox = server.gui.add_checkbox("Play", initial_value=False)

    def render_frame(idx: int):
        f = valid_frames[idx]
        pts, colors = clouds[f]
        add_point_cloud(server, pts, colors,
                         name="/anim/splat", point_size=0.03)

    @slider.on_update
    def _(_):
        render_frame(slider.value)

    render_frame(0)

    print(f"Viser running — drag slider or check 'Play' to animate over {len(valid_frames)} frames.")
    try:
        while True:
            if play_checkbox.value:
                next_idx = (slider.value + 1) % len(valid_frames)
                slider.value = next_idx  # 触发 on_update，自动渲染
                time.sleep(1.0 / FPS)
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()