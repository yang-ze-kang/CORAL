import logging
import sys
import warnings
import os
import glob
from typing import List, Tuple
import hydra
import torch
import gc

from utils.utils import set_seed
from utils.offline_metrics import run_offline_seg_metrics

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def _resolve_checkpoint_and_pred_dir(path_to_chkpt: str) -> Tuple[str, str]:
    if not path_to_chkpt:
        raise ValueError("cfg.path_to_chkpt is required for test-time prediction directory.")

    if os.path.isdir(path_to_chkpt):
        paths = glob.glob(os.path.join(path_to_chkpt, "*_last.ckpt"))
        if not paths:
            raise FileNotFoundError(f"No '*_last.ckpt' found in {path_to_chkpt}")
        ckpt = torch.load(paths[0], weights_only=False)
        best_ckpt_path = ckpt["callbacks"][
            "ModelCheckpoint{'monitor': 'val_clDiceMetric', 'mode': 'max', 'every_n_train_steps': 0, 'every_n_epochs': 1, 'train_time_interval': None}"
        ]["best_model_path"]
    else:
        best_ckpt_path = path_to_chkpt

    save_pred_dir = os.path.dirname(best_ckpt_path).replace("ckpts", "preds")
    return best_ckpt_path, save_pred_dir


def _find_missing_prediction_cubes(dataset, save_pred_dir: str) -> List[str]:
    cube_names: List[str] = []

    if hasattr(dataset, "annos"):
        # CubeDataset keeps file metadata in memory; use this to avoid costly __getitem__ I/O.
        for anno in dataset.annos:
            img_path = anno.get("img_path")
            mask_path = anno.get("mask_path")
            if not img_path or not mask_path:
                continue
            cube_names.append(os.path.splitext(os.path.basename(img_path))[0])
    elif hasattr(dataset, "cube_names"):
        cube_names = [str(x) for x in getattr(dataset, "cube_names")]
    else:
        raise ValueError(
            "Unable to collect cube names without dataset[idx]. "
            "Expected dataset.annos or dataset.cube_names."
        )

    missing = []
    for cube_name in cube_names:
        tif_path = os.path.join(save_pred_dir, f"{cube_name}.tif")
        if not os.path.exists(tif_path):
            missing.append(cube_name)
    return missing


@hydra.main(config_path="configs", config_name="train", version_base="1.3.2")
def main(cfg):
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    # init dataloader
    test_dataset = hydra.utils.instantiate(cfg.data.test)
    test_loader = hydra.utils.instantiate(cfg.dataloader)(
        dataset=test_dataset, batch_size=1, shuffle=False
    )
    logger.info(f"Test dataset size mapped to {len(test_dataset)} samples")

    best_ckpt_path, save_pred_dir = _resolve_checkpoint_and_pred_dir(cfg.path_to_chkpt)
    os.makedirs(save_pred_dir, exist_ok=True)

    missing_cubes = _find_missing_prediction_cubes(test_dataset, save_pred_dir)
    if missing_cubes:
        logger.info(
            "Found %d/%d missing prediction tif files. Will run model inference. Examples: %s",
            len(missing_cubes),
            len(test_dataset),
            ", ".join(missing_cubes[:10]),
        )
    else:
        logger.info(
            "All prediction tif files already exist in %s. Skip model inference.",
            save_pred_dir,
        )

    evaluator = hydra.utils.instantiate(cfg.evaluator)
    if missing_cubes:
        # init trainer
        trainer = hydra.utils.instantiate(cfg.trainer.lightning_trainer)
        trainer_additional_kwargs = {"devices": cfg.devices}
        trainer = trainer(**trainer_additional_kwargs)

        # init model
        model = hydra.utils.instantiate(cfg.model)
        chkpt = torch.load(best_ckpt_path, map_location="cpu")
        model_chkpt = {
            k.replace("model.", ""): e
            for k, e in chkpt["state_dict"].items()
            if "model" in k
        }
        model.load_state_dict(model_chkpt)

        # init lightning module
        lightning_module = hydra.utils.instantiate(cfg.trainer.lightning_module)(
            model=model, evaluator=evaluator, save_pred_dir=save_pred_dir
        )
        trainer.test(lightning_module, test_loader)

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
        summary_path=os.path.join(os.path.dirname(save_pred_dir), "summary.json"),
    )
    logger.info("Offline CPU test metrics: %s", summary)


if __name__ == "__main__":
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
    main()
