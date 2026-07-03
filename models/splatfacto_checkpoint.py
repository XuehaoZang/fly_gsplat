# models/splatfacto_checkpoint.py
from pathlib import Path
import numpy as np
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from dataclasses import dataclass, field
from typing import Type, List


@dataclass
class SplatfactoCheckpointConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SplatfactoCheckpointModel)
    checkpoint_every: int = 1000
    checkpoint_dir: str = "./debug_checkpoints"


class SplatfactoCheckpointModel(SplatfactoModel):
    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        cbs = super().get_training_callbacks(training_callback_attributes)

        trainer = training_callback_attributes.trainer
        if trainer is not None:
            ckpt_dir = trainer.base_dir / "debug_checkpoints"
        else:
            ckpt_dir = Path(self.config.checkpoint_dir)

        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.dump_means,
                update_every_num_iters=self.config.checkpoint_every,
                args=[ckpt_dir],
            )
        )
        return cbs

    def dump_means(self, checkpoint_dir: Path, step: int):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        means = self.means.detach().cpu().numpy()
        np.save(checkpoint_dir / f"step_{step:05d}_means.npy", means)

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