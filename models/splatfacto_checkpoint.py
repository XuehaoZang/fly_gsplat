# models/splatfacto_checkpoint.py
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib
import imageio.v2 as imageio
from PIL import Image
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.cameras.cameras import Cameras, CameraType
from dataclasses import dataclass, field
from typing import Type, List, Optional


# 第5视角（oblique）相机：斜上方偏移方向 + 距离系数，可调
OBLIQUE_DIR = np.array([1.0, 1.0, 1.2])   # 世界坐标下的斜上方偏移方向（会被归一化）
OBLIQUE_RADIUS_SCALE = 1.0                 # eye 到场景中心距离 = 系数 * 参考训练相机到场景中心的真实距离


@dataclass
class SplatfactoCheckpointConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoCheckpointModel)

    save_stats: bool = True
    stats_every: int = 1000

    save_points: bool = False
    points_every: int = 2000

    save_eval_images: bool = False
    eval_images_every: int = 5000
    eval_camera_indices: List[int] = field(default_factory=lambda: [0, 1, 2, 3])  # 4个训练相机全渲

    checkpoint_dir: str = "./debug_checkpoints"


class SplatfactoCheckpointModel(SplatfactoModel):
    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        cbs = super().get_training_callbacks(training_callback_attributes)

        trainer = training_callback_attributes.trainer
        pipeline = training_callback_attributes.pipeline
        if trainer is not None:
            ckpt_dir = trainer.base_dir / "debug_checkpoints"
            final_step = trainer.config.max_num_iterations - 1
        else:
            ckpt_dir = Path(self.config.checkpoint_dir)
            final_step = -1  # 无 trainer（例如单元测试）时不触发 final 强制 dump

        # 只注册一个统一回调，每步触发，内部按各自周期/开关判断
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self._on_iteration,
                update_every_num_iters=1,
                args=[ckpt_dir, final_step, pipeline],
            )
        )
        return cbs

    def _on_iteration(self, ckpt_dir: Path, final_step: int, pipeline, step: int):
        cfg = self.config
        is_final = (step == final_step)

        # stats/points 是纯数值 dump，整体包在 no_grad 里
        with torch.no_grad():
            if cfg.save_stats and (is_final or (cfg.stats_every > 0 and step % cfg.stats_every == 0)):
                self._dump_stats(ckpt_dir, step)
            if cfg.save_points and (is_final or (cfg.points_every > 0 and step % cfg.points_every == 0)):
                self._dump_points(ckpt_dir, step)

        # eval 渲染涉及 train/eval 切换，交给子函数自己用 try/finally 管理
        if cfg.save_eval_images and (is_final or (cfg.eval_images_every > 0 and step % cfg.eval_images_every == 0)):
            self._dump_eval_images(ckpt_dir, step, pipeline)

    # ------------------------------------------------------------------ stats --
    def _dump_stats(self, ckpt_dir: Path, step: int):
        out_dir = ckpt_dir / "stats"
        out_dir.mkdir(parents=True, exist_ok=True)

        means_np = self.means.detach().cpu().numpy()
        n = int(means_np.shape[0])

        stats = {"step": int(step), "n_gaussians": n}

        if n > 0:
            # scale_ratio：复用 utils/ply.py::analyze_scale_ratio 的算法，直接从 raw scales 算
            scales = torch.exp(self.scales.detach()).cpu().numpy()
            ratios = scales.max(axis=-1) / np.clip(scales.min(axis=-1), 1e-12, None)
            # opacity：sigmoid 还原真实透明度
            opac = torch.sigmoid(self.opacities.detach()).squeeze(-1).cpu().numpy()

            bbox_min = means_np.min(axis=0)
            bbox_max = means_np.max(axis=0)
            stats.update({
                "scale_ratio": {
                    "median": float(np.median(ratios)),
                    "p90": float(np.percentile(ratios, 90)),
                    "p95": float(np.percentile(ratios, 95)),
                    "max": float(ratios.max()),
                    "frac_over_10": float((ratios > 10).mean()),
                },
                "opacity": {
                    "mean": float(opac.mean()),
                    "median": float(np.median(opac)),
                    "p10": float(np.percentile(opac, 10)),
                },
                "bbox_min": bbox_min.tolist(),
                "bbox_max": bbox_max.tolist(),
                "bbox_extent": (bbox_max - bbox_min).tolist(),
            })

        with open(out_dir / f"step_{step:05d}_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    # ----------------------------------------------------------------- points --
    def _dump_points(self, ckpt_dir: Path, step: int):
        out_dir = ckpt_dir / "points"
        out_dir.mkdir(parents=True, exist_ok=True)

        keys = ["means", "scales", "quats", "opacities", "features_dc", "features_rest"]
        data = {k: getattr(self, k).detach().cpu().numpy().astype(np.float32) for k in keys}
        np.savez(out_dir / f"step_{step:05d}_gaussians.npz", **data)

        # 兼容旧脚本 debug/debug_checkpoints.py：顶层单独存一份 means.npy（路径/格式不变）
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        np.save(ckpt_dir / f"step_{step:05d}_means.npy", data["means"])

    # ------------------------------------------------------------ eval images --
    def _save_depth_png(self, depth, cmap, path: Path):
        """depth [H,W,1] -> 按本帧 min/max 归一化 -> turbo colormap 8bit png。"""
        d = depth.squeeze(-1).detach().cpu().numpy().astype(np.float32)
        dmin, dmax = float(d.min()), float(d.max())
        norm = np.zeros_like(d) if (dmax - dmin) < 1e-8 else (d - dmin) / (dmax - dmin)
        rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
        imageio.imwrite(path, rgb)

    def _dump_eval_images(self, ckpt_dir: Path, step: int, pipeline):
        if pipeline is None:
            print(f"[eval images] step {step}: pipeline is None, skip")
            return
        out_dir = ckpt_dir / "eval_images"
        out_dir.mkdir(parents=True, exist_ok=True)

        dm = pipeline.datamanager
        cmap = matplotlib.colormaps["turbo"]

        was_training = self.training
        try:
            if was_training:
                self.eval()   # 干净可复现渲染：关掉 camera_opt / 训练降采样 / 随机背景
            with torch.no_grad():
                for idx in self.config.eval_camera_indices:
                    cam = dm.train_cameras[idx:idx + 1].to(self.device)
                    out = self.get_outputs(cam)

                    rgb = out["rgb"].clamp(0, 1).detach().cpu().numpy()
                    rgb_u8 = (rgb * 255).astype(np.uint8)

                    # GT 图：uint8 -> /255；有 alpha 只取前3通道
                    gt = dm.cached_train[idx]["image"]
                    gt = gt.detach().cpu().numpy() if torch.is_tensor(gt) else np.asarray(gt)
                    gt = gt.astype(np.float32) / 255.0 if gt.dtype == np.uint8 else gt.astype(np.float32)
                    gt = gt[..., :3]
                    gt_u8 = (np.clip(gt, 0, 1) * 255).astype(np.uint8)

                    # side-by-side（左 GT，右渲染），分辨率不一致时把 GT resize 到渲染尺寸
                    if gt_u8.shape[:2] != rgb_u8.shape[:2]:
                        gt_u8 = np.asarray(
                            Image.fromarray(gt_u8).resize((rgb_u8.shape[1], rgb_u8.shape[0]))
                        )
                    sep = np.full((rgb_u8.shape[0], 4, 3), 255, np.uint8)  # 4px 白色分隔线
                    sbs = np.concatenate([gt_u8, sep, rgb_u8], axis=1)
                    imageio.imwrite(out_dir / f"step_{step:05d}_cam{idx}_rgb_vs_gt.png", sbs)

                    depth = out.get("depth")
                    if depth is not None:
                        self._save_depth_png(depth, cmap, out_dir / f"step_{step:05d}_cam{idx}_depth.png")

                    acc = out.get("accumulation")
                    if acc is not None:
                        acc_np = acc.squeeze(-1).clamp(0, 1).detach().cpu().numpy()
                        imageio.imwrite(out_dir / f"step_{step:05d}_cam{idx}_acc.png",
                                        (acc_np * 255).astype(np.uint8))

                self._dump_oblique(out_dir, step, dm, cmap)
        finally:
            if was_training:
                self.train()  # 保证异常也切回训练模式

    def _dump_oblique(self, out_dir: Path, step: int, dm, cmap):
        """不在训练集里的斜上方第5视角，无 GT 对比，单张渲染。"""
        means_np = self.means.detach().cpu().numpy()
        if len(means_np) == 0:
            return
        center = means_np.mean(axis=0)

        # 内参/分辨率直接复用 eval_camera_indices[0] 那个训练相机
        ref_idx = self.config.eval_camera_indices[0]
        ref = dm.train_cameras[ref_idx:ref_idx + 1]
        ref_pos = ref.camera_to_worlds[0, :, 3].detach().cpu().numpy()

        # 距离复用参考训练相机到场景中心的真实距离，而不是点云 bbox extent：
        # 训练收敛后 extent 相对相机-物体距离可能小两个数量级（数据集本身 fly 很小、
        # 相机远距长焦拍摄），用 extent 推距离会把虚拟相机怼进点云内部，画面变成
        # 近距离模糊团块、看不到白背景。参考相机的距离是标定好的真实取景尺度，
        # 配合复用的内参能保证视场角匹配、物体完整入画。
        radius = OBLIQUE_RADIUS_SCALE * float(np.linalg.norm(ref_pos - center))
        if radius <= 1e-8:
            extent = means_np.max(axis=0) - means_np.min(axis=0)
            radius = 3.0 * float(extent.max())  # 退化兜底（例如参考相机与中心重合）
        if radius <= 0:
            return

        direction = OBLIQUE_DIR / np.linalg.norm(OBLIQUE_DIR)
        eye = center + radius * direction

        # look-at：nerfstudio 相机看向 -Z（见 utils/camera.py from_opengl 的 FLIP），
        # 故 camera_to_world 的 +Z 列指向 eye-center（背离物体），否则会渲染成背面/全黑。
        z = eye - center
        z = z / np.linalg.norm(z)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(world_up, z))) > 0.999:   # up 与视线近似平行时换一个 up 避免退化
            world_up = np.array([0.0, 1.0, 0.0])
        x = np.cross(world_up, z); x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z], axis=1)  # 各向量作为列 -> camera-to-world 旋转
        c2w = np.concatenate([R, eye[:, None]], axis=1).astype(np.float32)  # (3,4)

        cam = Cameras(
            camera_to_worlds=torch.from_numpy(c2w)[None],
            fx=ref.fx.reshape(1, 1).clone(),
            fy=ref.fy.reshape(1, 1).clone(),
            cx=ref.cx.reshape(1, 1).clone(),
            cy=ref.cy.reshape(1, 1).clone(),
            width=ref.width.reshape(1, 1).clone(),
            height=ref.height.reshape(1, 1).clone(),
            camera_type=CameraType.PERSPECTIVE,
        ).to(self.device)

        out = self.get_outputs(cam)
        rgb = (out["rgb"].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(out_dir / f"step_{step:05d}_oblique_rgb.png", rgb)
        depth = out.get("depth")
        if depth is not None:
            self._save_depth_png(depth, cmap, out_dir / f"step_{step:05d}_oblique_depth.png")

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig

splatfacto_checkpoint_method = MethodSpecification(
    config=TrainerConfig(
        method_name="splatfacto-checkpoint",
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=SplatfactoCheckpointConfig(),
        ),
        optimizers={
            "means": {"optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15), "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=1e-8, lr_final=1.6e-6, warmup_steps=0, max_steps=30000, ramp="cosine")},
            "features_dc": {"optimizer": AdamOptimizerConfig(lr=2.5e-3, eps=1e-15), "scheduler": None},
            "features_rest": {"optimizer": AdamOptimizerConfig(lr=1.25e-4, eps=1e-15), "scheduler": None},
            "opacities": {"optimizer": AdamOptimizerConfig(lr=5e-2, eps=1e-15), "scheduler": None},
            "scales": {"optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15), "scheduler": None},
            "quats": {"optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15), "scheduler": None},
            "camera_opt": {"optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15), "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=0, lr_final=5e-7, warmup_steps=1000, max_steps=30000, ramp="cosine")},
            "bilateral_grid": {"optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15), "scheduler": ExponentialDecaySchedulerConfig(lr_pre_warmup=0, lr_final=1e-4, warmup_steps=1000, max_steps=30000, ramp="cosine")},
        },
    ),
    description="Splatfacto with periodic mean checkpoint dumps for debugging.",
)