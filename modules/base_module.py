import logging
import os
import torch
import lightning
import numpy as np
import tifffile as tiff

from torch.nn.functional import interpolate

logger = logging.getLogger(__name__)


class TrainModuleBase(lightning.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss,
        optimizer_factory,
        loss_cfg=None,
        loss_weights=None,
        scheduler_configs=None,
        evaluator=None,
        save_pred_dir=None
    ):
        super().__init__()
        self.model = model
        self.optimizer_factory = optimizer_factory
        self.scheduler_configs = scheduler_configs
        self.save_pred_dir = save_pred_dir
        if loss_cfg:
            assert len(loss_cfg.keys()) == len(loss_weights)
            self.losses = {
                name: {"fn": cfg, "weight": loss_weights[i]}
                for i, (name, cfg) in enumerate(loss_cfg.items())
            }
        else:
            self.losses = {"main": {"fn": loss, "weight": 1.0}}
        self.rank = 0 if "LOCAL_RANK" not in os.environ else os.environ["LOCAL_RANK"]
        self.evaluator = evaluator

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(params=self.parameters())

        if self.scheduler_configs is not None:
            schedulers = []
            logger.info(f"Initializing schedulers: {self.scheduler_configs}")
            for scheduler_name, scheduler_config in self.scheduler_configs.items():
                if scheduler_config is None:
                    continue  # skip empty configs during finetuning

                logger.info(f"Initializing scheduler: {scheduler_name}")
                scheduler_config["scheduler"] = scheduler_config["scheduler"](
                    optimizer=optimizer
                )
                scheduler_config = dict(scheduler_config)
                schedulers.append(scheduler_config)
            return [optimizer], schedulers
        return optimizer

    def _validation_step(self, batch, batch_idx, stage):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        self._validation_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._validation_step(batch, batch_idx, "test")
