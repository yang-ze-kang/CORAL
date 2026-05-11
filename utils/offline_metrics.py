import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from multiprocessing import get_context
from typing import Optional

import numpy as np
import tifffile as tiff
import torch
from tqdm.auto import tqdm


logger = logging.getLogger(__name__)
_OFFLINE_DATASET = None
_OFFLINE_EVALUATOR = None


def _safe_metric_value(metric_name: str, value) -> float:
    v = float(value)
    if metric_name.lower() == "cldice" and np.isnan(v):
        return 0.0
    return v


def _try_get_cube_name_without_loading(dataset, idx: int) -> Optional[str]:
    # Try common metadata containers first to avoid triggering heavy __getitem__ loading.
    for attr in ("data", "samples", "items"):
        container = getattr(dataset, attr, None)
        if container is None:
            continue
        try:
            item = container[idx]
        except Exception:
            continue
        if isinstance(item, dict) and item.get("cube_name") is not None:
            return str(item["cube_name"])
    return None


def _load_cached_case_metrics(case_json_path: str):
    try:
        with open(case_json_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if not isinstance(cached, dict):
            return None
        return cached
    except Exception:
        return None


def _to_b1c1dhw(mask) -> torch.Tensor:
    if isinstance(mask, np.ndarray):
        mask_t = torch.from_numpy(mask)
    elif torch.is_tensor(mask):
        mask_t = mask.detach().cpu()
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)}")

    if mask_t.ndim == 3:
        mask_t = mask_t.unsqueeze(0).unsqueeze(0)
    elif mask_t.ndim == 4:
        # C,D,H,W
        mask_t = mask_t.unsqueeze(0)
    elif mask_t.ndim != 5:
        raise ValueError(f"Unsupported mask shape: {tuple(mask_t.shape)}")
    return mask_t.float()


def _to_b1c1dhw_pred(pred_prob_np: np.ndarray) -> torch.Tensor:
    pred_t = torch.from_numpy(pred_prob_np.astype(np.float32))
    if pred_t.ndim == 3:
        pred_t = pred_t.unsqueeze(0).unsqueeze(0)
    elif pred_t.ndim == 4:
        pred_t = pred_t.unsqueeze(0)
    elif pred_t.ndim != 5:
        raise ValueError(f"Unsupported prediction shape: {tuple(pred_t.shape)}")
    return pred_t


def _init_offline_metric_worker(dataset, evaluator):
    global _OFFLINE_DATASET, _OFFLINE_EVALUATOR
    _OFFLINE_DATASET = dataset
    _OFFLINE_EVALUATOR = evaluator


def _process_offline_metric_case(
    idx: int,
    save_pred_dir: str,
    prediction_threshold: float,
    per_case_json: bool,
):
    cube_name = _try_get_cube_name_without_loading(_OFFLINE_DATASET, idx) or str(idx)
    case_json_path = os.path.join(save_pred_dir, f"{cube_name}.json")
    cached_metrics = _load_cached_case_metrics(case_json_path)
    if cached_metrics is not None:
        return {"status": "ok", "cube_name": cube_name, "metrics": cached_metrics}

    data = _OFFLINE_DATASET[idx]
    cube_name = str(data.get("cube_name", cube_name))
    case_json_path = os.path.join(save_pred_dir, f"{cube_name}.json")

    # If the guessed cube_name changed after loading metadata, check cache once again.
    cached_metrics = _load_cached_case_metrics(case_json_path)
    if cached_metrics is not None:
        return {"status": "ok", "cube_name": cube_name, "metrics": cached_metrics}

    mask = data.get("mask", None)

    if mask is None:
        return {"status": "skipped_no_mask", "cube_name": cube_name}

    tif_path = os.path.join(save_pred_dir, f"{cube_name}.tif")
    if not os.path.exists(tif_path):
        return {"status": "missing_pred", "cube_name": cube_name}

    pred_prob = tiff.imread(tif_path).astype(np.float32) / 255.0
    pred_prob_t = _to_b1c1dhw_pred(pred_prob)
    gt_t = _to_b1c1dhw(mask)

    save_case_json_path = case_json_path if per_case_json else None
    metrics = _OFFLINE_EVALUATOR.estimate_metrics(
        pred_prob_t,
        gt_t,
        threshold=prediction_threshold,
        save_path=save_case_json_path,
    )
    return {"status": "ok", "cube_name": cube_name, "metrics": metrics}


def run_offline_seg_metrics(
    dataset,
    save_pred_dir: str,
    evaluator,
    prediction_threshold: float = 0.5,
    summary_path: Optional[str] = None,
    per_case_json: bool = True,
    num_workers: Optional[int] = None,
):
    if not save_pred_dir:
        raise ValueError("save_pred_dir must be provided for offline metric evaluation.")
    if not os.path.isdir(save_pred_dir):
        raise FileNotFoundError(f"Prediction directory not found: {save_pred_dir}")

    total_samples = len(dataset)
    if num_workers is None:
        num_workers = min(16, max(1, total_samples))
    else:
        num_workers = max(1, int(num_workers))

    total_cases = 0
    metric_sums = {}
    missing_preds = []
    skipped_no_mask = []

    results = []
    if num_workers == 1 or total_samples <= 1:
        _init_offline_metric_worker(dataset, evaluator)
        iterator = (
            _process_offline_metric_case(
                idx=idx,
                save_pred_dir=save_pred_dir,
                prediction_threshold=prediction_threshold,
                per_case_json=per_case_json,
            )
            for idx in range(total_samples)
        )
        results = list(
            tqdm(
                iterator,
                total=total_samples,
                desc="Offline metric eval",
            )
        )
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=get_context("fork"),
            initializer=_init_offline_metric_worker,
            initargs=(dataset, evaluator),
        ) as executor:
            iterator = executor.map(
                _process_offline_metric_case,
                range(total_samples),
                repeat(save_pred_dir),
                repeat(prediction_threshold),
                repeat(per_case_json),
            )
            results = list(
                tqdm(
                    iterator,
                    total=total_samples,
                    desc="Offline metric eval",
                )
            )

    for result in results:
        status = result["status"]
        cube_name = result["cube_name"]
        if status == "ok":
            metrics = result["metrics"]
            total_cases += 1
            for k, v in metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + _safe_metric_value(k, v)
        elif status == "missing_pred":
            missing_preds.append(cube_name)
        elif status == "skipped_no_mask":
            skipped_no_mask.append(cube_name)

    if missing_preds:
        preview = ", ".join(missing_preds[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_preds)} prediction tif files in {save_pred_dir}. "
            f"Examples: {preview}"
        )

    if total_cases <= 0:
        raise RuntimeError("No valid test cases found for offline metric evaluation.")

    summary = {"count": float(total_cases)}
    for k in sorted(metric_sums.keys()):
        summary[f"testmetric_{k}"] = metric_sums[k] / (total_cases + 1e-6)

    if summary_path is None:
        summary_path = os.path.join(os.path.dirname(save_pred_dir), "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    if skipped_no_mask:
        logger.warning(
            "Skipped %d samples without mask during offline metric evaluation.",
            len(skipped_no_mask),
        )

    logger.info("Offline CPU metrics written to %s", summary_path)
    return summary
