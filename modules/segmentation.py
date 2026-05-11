import logging
import os
import torch
import lightning
import numpy as np
import tifffile as tiff
from collections.abc import Mapping

from monai.inferers.inferer import SlidingWindowInfererAdapt


logger = logging.getLogger(__name__)


class TrainModule(lightning.LightningModule):

    def __init__(
        self,
        model: torch.nn.Module,
        loss,
        optimizer_factory,
        prediction_threshold: float,
        supervise_both_decoders: bool = False,
        loss_weights=None,
        scheduler_configs=None,
        evaluator=None,
        input_size: tuple = None,
        batch_size: int = None,
        save_pred_dir: str = None,
        infer_scale_intensity_in_window: bool = False,
        infer_scale_eps: float = 1e-8,
    ):
        super().__init__()
        self.model = model
        self.optimizer_factory = optimizer_factory
        self.scheduler_configs = scheduler_configs
        self.prediction_threshold = prediction_threshold
        self.supervise_both_decoders = supervise_both_decoders
        self.save_pred_dir = save_pred_dir
        self.infer_scale_intensity_in_window = infer_scale_intensity_in_window
        self.infer_scale_eps = infer_scale_eps
        if save_pred_dir:
            os.makedirs(save_pred_dir, exist_ok=True)
        if isinstance(loss, Mapping):
            weights = loss.get("weights", None)
            assert weights is not None
            losses, loss_names = {}, []
            for i, (name, cfg) in enumerate(loss.items()):
                if name in ["weights", "start_steps"]:
                    continue
                loss_names.append(name)
                losses[name] = {"fn": cfg, "weight": weights[i]}
            self.losses = losses
            start_steps = loss.get("start_steps", None)
            if start_steps is not None:
                assert len(start_steps) == len(loss_names)
                for name, start_step in zip(loss_names, start_steps):
                    self.losses[name]["start_step"] = start_step
        elif isinstance(loss, torch.nn.Module):
            self.losses = {"main": {"fn": loss, "weight": 1.0}}
        else:
            raise NotImplementedError
        self.evaluator = evaluator
        self.sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=input_size, sw_batch_size=batch_size, overlap=0.5, progress=False
        )

    def _window_predictor(self, image_window: torch.Tensor):
        if self.infer_scale_intensity_in_window:
            # Scale each sliding window independently to [0, 1].
            # Keep batch/channel dims and reduce only over spatial dims.
            reduce_dims = tuple(range(2, image_window.ndim))
            vmin = image_window.amin(dim=reduce_dims, keepdim=True)
            vmax = image_window.amax(dim=reduce_dims, keepdim=True)
            image_window = (image_window - vmin) / (
                torch.clamp(vmax - vmin, min=self.infer_scale_eps)
            )
        return self.model(image_window)

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

    def training_step(self, batch, batch_idx):
        image, mask = batch["image"], batch["mask"]
        snr_enabled = True
        if "snr_classify" in self.losses:
            snr_cfg = self.losses["snr_classify"]
            snr_start_step = snr_cfg.get("start_step", 0)
            snr_enabled = self.global_step >= snr_start_step
        if hasattr(self.model, "use_snr_classify"):
            self.model.use_snr_classify = snr_enabled
        pred = self.model(image)
        if isinstance(pred, dict):
            pred_mask = pred["mask"]
        else:
            pred_mask = pred
        total_loss = 0.0
        log_dict = {}
        for name, obj in self.losses.items():
            fn, w = obj["fn"], obj["weight"]
            if "start_step" in obj and self.global_step < obj["start_step"]:
                continue
            if name == "snr_classify":
                loss_val = fn(pred["pred_label"], batch["snr_label"])
            elif hasattr(fn, "allow_delay_start") and fn.allow_delay_start:
                loss_val = fn(pred_mask, mask, global_step=self.global_step)
            elif (
                (not snr_enabled or self.supervise_both_decoders)
                and isinstance(pred, dict)
                and "mask_sn" in pred
                and "mask_ws" in pred
            ):
                loss_sn = fn(pred["mask_sn"], mask)
                loss_ws = fn(pred["mask_ws"], mask)
                loss_val = 0.5 * (loss_sn + loss_ws)
                log_dict[f"train_loss_{name}_sn"] = loss_sn.detach()
                log_dict[f"train_loss_{name}_ws"] = loss_ws.detach()
            else:
                loss_val = fn(pred_mask, mask)
            total_loss += w * loss_val
            log_dict[f"train_loss_{name}"] = loss_val.detach()
            log_dict[f"train_loss_{name}_w"] = (w * loss_val).detach()
        assert isinstance(total_loss, torch.Tensor), "Total loss must be a torch.Tensor"
        log_dict["train_loss_total"] = total_loss.detach()
        self.log_dict(log_dict)
        return total_loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch["image"], batch["mask"]
        with torch.inference_mode():
            pred_mask = self.sliding_window_inferer(image, self._window_predictor)
            total_loss = 0.0
            log_dict = {}
            for name, obj in self.losses.items():
                if name == "snr_classify":
                    continue
                fn, w = obj["fn"], obj["weight"]
                loss_val = fn(pred_mask, mask)
                total_loss += w * loss_val
                log_dict[f"val_loss_{name}"] = loss_val
                log_dict[f"val_loss_{name}_w"] = w * loss_val
            log_dict["val_loss_total"] = total_loss
            self.log_dict(log_dict, sync_dist=True, on_step=False, on_epoch=True)

            pred_prob_gpu = torch.sigmoid(pred_mask.detach())
            metrics = self.evaluator.estimate_metrics(
                pred_prob_gpu,
                mask.detach(),
                threshold=self.prediction_threshold,
                fast=True,
                return_tensor=True,
            )
            batch_size = image.shape[0]
            self.log(
                f"val_cnt",
                batch_size,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                reduce_fx="sum",
            )
            for name, value in metrics.items():
                self.log(
                    f"valmetric_{name}",
                    value,
                    sync_dist=True,
                    on_step=False,
                    on_epoch=True,
                    reduce_fx="sum",
                )

            # del pred_mask, pred_prob_gpu, image, mask

    def on_validation_epoch_end(self):
        cm = self.trainer.callback_metrics
        cnt = cm.get("val_cnt", 0)
        assert cnt > 0, "No samples were processed during validation epoch."
        for k in list(cm.keys()):
            if k == "cnt" or not k.startswith("valmetric_"):
                continue
            s = cm.get(k)
            mean = (s / (cnt + 1e-6)).detach()
            self.log(k, mean, on_step=False, on_epoch=True, sync_dist=False)

    def test_step(self, batch, batch_idx):
        image, names = (
            batch["image"],
            batch["cube_name"],
        )
        with torch.inference_mode():
            tif_path = os.path.join(self.save_pred_dir, f"{names[0]}.tif")
            if os.path.exists(tif_path):
                return
            pred_mask = self.sliding_window_inferer(image, self._window_predictor)
            if self.save_pred_dir:
                tiff.imwrite(
                    tif_path,
                    (pred_mask.sigmoid()[0] * 255)
                    .squeeze()
                    .cpu()
                    .numpy()
                    .astype(np.uint8),
                )

            # del pred_mask, pred_prob_cpu, image, mask

    def on_test_epoch_end(self):
        logger.info(
            "Test inference finished. Metrics are deferred to offline CPU evaluation."
        )
