import logging
import sys
import warnings
import os
import gc
import hydra
import torch
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only
import time

from utils.utils import set_seed, count_parameters, get_run_timestamp
from utils.offline_metrics import run_offline_seg_metrics

warnings.filterwarnings("ignore")

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)


@rank_zero_only
def make_dir(path):
    assert not os.path.isdir(path) or not os.listdir(
        path
    ), "The output_dir has existed and not empty!"
    os.makedirs(path, exist_ok=True)


def setup_logging(output_dir: str):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(
        os.path.join(output_dir, "train.log"), mode="a", encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)

    root.propagate = False


@hydra.main(config_path="configs", config_name="train", version_base="1.3.2")
def main(cfg):
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    resume_ckpt_path = getattr(cfg, "resume_ckpt_path", None)

    if resume_ckpt_path is not None:
        assert os.path.exists(resume_ckpt_path), f"Checkpoint path {resume_ckpt_path} does not exist!"
        ckpt_dir = os.path.dirname(resume_ckpt_path)
        output_dir = os.path.dirname(ckpt_dir)
    else:
        timestamp = get_run_timestamp()
        output_dir = (
            cfg.log_dir + "/" + cfg.project_name + "/" + cfg.run_name + "/" + timestamp
        )
        make_dir(output_dir)

    setup_logging(output_dir)

    logger = logging.getLogger(__name__)

    yaml_path = f"{output_dir}/config.yaml"
    OmegaConf.save(cfg, yaml_path)
    wnb_logger = TensorBoardLogger(save_dir=output_dir)

    # callbacks
    lr_monitor = LearningRateMonitor()
    monitor_metric = cfg.monitor_metric
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir + "/ckpts",
        monitor=monitor_metric,
        save_top_k=1,
        mode="max",
        filename=(f"{cfg.run_name}_" + "{step}_{" + monitor_metric + ":.4f}"),
        auto_insert_metric_name=True,
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "-"
    checkpoint_callback.CHECKPOINT_NAME_LAST = cfg.run_name + "_last"

    # init trainer
    trainer = hydra.utils.instantiate(cfg.trainer.lightning_trainer)
    trainer_additional_kwargs = {
        "logger": wnb_logger,
        "callbacks": [lr_monitor, checkpoint_callback],
        "devices": cfg.devices,
    }
    trainer = trainer(**trainer_additional_kwargs)

    # init dataloader
    train_dataset = hydra.utils.instantiate(cfg.data.train)
    train_loader = hydra.utils.instantiate(cfg.dataloader)(dataset=train_dataset)
    logger.info(f"Train dataset size mapped to {len(train_dataset)} samples")

    val_dataset = hydra.utils.instantiate(cfg.data.val)
    val_loader = hydra.utils.instantiate(cfg.dataloader)(
        dataset=val_dataset, batch_size=1, shuffle=False
    )
    logger.info(f"Validation dataset size mapped to {len(val_dataset)} samples")

    test_dataset = hydra.utils.instantiate(cfg.data.test)
    test_loader = hydra.utils.instantiate(cfg.dataloader)(
        dataset=test_dataset, batch_size=1, shuffle=False
    )
    logger.info(f"Test dataset size mapped to {len(test_dataset)} samples")

    # init model
    model = hydra.utils.instantiate(cfg.model)
    if resume_ckpt_path is None and cfg.path_to_chkpt is not None:
        logger.info(f"Loading pretrained weights from: {cfg.path_to_chkpt}")
        chkpt = torch.load(cfg.path_to_chkpt, map_location=f"cuda:{cfg.devices[0]}")
        model_chkpt = {
            k.replace("model.", ""): v
            for k, v in chkpt["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(model_chkpt, strict=True)

    params, params_trainable = count_parameters(model)
    logger.info(f"Total: {params/1e6:.2f} M, Trainable: {params_trainable/1e6:.2f} M")

    # init lightning module
    evaluator = hydra.utils.instantiate(cfg.evaluator)
    save_pred_dir = os.path.join(output_dir, "preds")
    lightning_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
        model=model, evaluator=evaluator, save_pred_dir=save_pred_dir
    )

    # train loop
    logger.info("Starting training")
    if resume_ckpt_path is not None:
        logger.info(f"Resuming training from checkpoint: {resume_ckpt_path}")
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resume_ckpt_path,
        )
    else:
        # trainer.validate(lightning_module, val_loader)
        trainer.fit(
            lightning_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
    logger.info("Finished training")
    

    logger.info("Starting testing")
    set_seed(cfg.seed)
    best_ckpt_path = checkpoint_callback.best_model_path
    if not best_ckpt_path:
        best_ckpt_path = checkpoint_callback.last_model_path
    if not best_ckpt_path:
        raise RuntimeError(
            "No checkpoint available for testing (best/last checkpoint path is empty)."
        )

    logger.info(f"Testing checkpoint: {best_ckpt_path}")
    trainer.test(lightning_module, test_loader, ckpt_path=best_ckpt_path)

    del lightning_module
    del model
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = run_offline_seg_metrics(
        dataset=test_dataset,
        save_pred_dir=save_pred_dir,
        evaluator=evaluator,
        prediction_threshold=float(cfg.trainer.lightning_module.prediction_threshold),
        summary_path=os.path.join(output_dir, "summary.json"),
    )
    logger.info(f"Offline CPU test metrics: {summary}")


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    main()
