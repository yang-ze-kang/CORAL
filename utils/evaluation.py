import numpy as np
import torch
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.measure import euler_number, label
import json
from monai.transforms import AsDiscrete
from monai.metrics import SurfaceDiceMetric, HausdorffDistanceMetric


def to_json_serializable(d: dict):
    """
    Convert a dict containing torch.Tensor or MetaTensor to JSON-serializable types.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            # 0-dim tensor -> float
            if v.numel() == 1:
                out[k] = v.item()
            else:
                out[k] = v.detach().cpu().tolist()
        else:
            out[k] = v
    return out


def nsd_3d_binary(
    pred: torch.Tensor,  # (B,1,D,H,W) 0/1
    label: torch.Tensor, # (B,1,D,H,W) 0/1
    spacing=(1.0,1.0,1.0),
    tolerances=(1,2,3,5),
    return_tensor: bool = False,
):
    out = {}
    for t in tolerances:
        metric = SurfaceDiceMetric(
            class_thresholds=[t],
            include_background=True,
            reduction="mean",
        )
        v = metric(pred, label, spacing=spacing).mean()
        out[f"NSD@{t}"] = v if return_tensor else float(v.item())
    return out


def hd_and_hd95_3d_binary(
    pred: torch.Tensor,       # (B,1,D,H,W) values 0/1
    label: torch.Tensor,         # (B,1,D,H,W) values 0/1
    spacing=(1.0, 1.0, 1.0),      # (dz,dy,dx) matches (D,H,W)
    return_tensor: bool = False,
):
    # HD (max Hausdorff)
    hd_metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=None,      # None -> max Hausdorff distance
        reduction="none",     # keep per-sample then we average ourselves
    )
    hd = hd_metric(pred, label, spacing=spacing)   # shape (B,1) usually
    hd_mean_t = hd.mean()

    # HD95
    hd95_metric = HausdorffDistanceMetric(
        include_background=True,
        percentile=95,        # 95th percentile Hausdorff
        reduction="none",
    )
    hd95 = hd95_metric(pred, label, spacing=spacing)
    hd95_mean_t = hd95.mean()

    if return_tensor:
        return hd_mean_t, hd95_mean_t
    return float(hd_mean_t.item()), float(hd95_mean_t.item())

class Evaluator:
    def extract_labels(self, gt_array, pred_array):
        """
        Adapted from https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py#L18.
        """
        labels_gt = np.unique(gt_array)
        labels_pred = np.unique(pred_array)
        labels = list(set().union(labels_gt, labels_pred))
        labels = [int(x) for x in labels]
        return labels

    def betti_number_error(self, gt, pred):
        """
        Adapted from https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py#L250.
        """
        labels = self.extract_labels(gt_array=gt, pred_array=pred)
        labels.remove(0)

        if len(labels) == 0:
            return 0, 0
        assert len(labels) == 1 and 1 in labels, "Invalid binary segmentatio.n"

        gt_betti_numbers = self.betti_number(gt)
        pred_betti_numbers = self.betti_number(pred)
        betti_0_error = abs(pred_betti_numbers[0] - gt_betti_numbers[0])
        betti_1_error = abs(pred_betti_numbers[1] - gt_betti_numbers[1])
        return (
            betti_0_error,
            betti_1_error,
            pred_betti_numbers[0],
            pred_betti_numbers[1],
            pred_betti_numbers[2],
            gt_betti_numbers[0],
            gt_betti_numbers[1],
            gt_betti_numbers[2],
        )

    def betti_number(self, img):
        """
        Adapted from https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py#L186.
        """
        assert img.ndim == 3
        N6 = 1
        N26 = 3

        padded = np.pad(img, pad_width=1)
        assert set(np.unique(padded)).issubset({0, 1})

        _, b0 = label(padded, return_num=True, connectivity=N26)
        euler_char_num = euler_number(padded, connectivity=N26)
        _, b2 = label(1 - padded, return_num=True, connectivity=N6)

        b2 -= 1
        b1 = b0 + b2 - euler_char_num
        return [b0, b1, b2]

    def cl_dice(self, v_p, v_l):
        """
        Adapted from https://github.com/jocpae/clDice/blob/master/cldice_metric/cldice.py.
        """

        def cl_score(v, s):
            if np.sum(v) == 0:
                return 0.0
            return np.sum(v * s) / np.sum(s)

        if len(v_p.shape) == 2:
            tprec = cl_score(v_p, skeletonize(v_l))
            tsens = cl_score(v_l, skeletonize(v_p))
        elif len(v_p.shape) == 3:
            tprec = cl_score(v_p, skeletonize_3d(v_l))
            tsens = cl_score(v_l, skeletonize_3d(v_p))
        else:
            raise ValueError(f"Invalid shape for cl_dice: {v_p.shape}")
        return 2 * tprec * tsens / (tprec + tsens + np.finfo(float).eps)

    def estimate_metrics(
        self,
        pred_seg,
        gt_seg,
        threshold=0.5,
        fast=False,
        eps=1e-6,
        save_path=None,
        return_tensor=False,
    ):
        """
        Args:
            pred_seg: torch.Tensor of shape (B, D, H, W), predicted segmentation
            gt_seg: torch.Tensor of shape (B, D, H, W), ground truth segmentation
        """
        if not torch.is_tensor(pred_seg) or not torch.is_tensor(gt_seg):
            raise TypeError("pred_seg and gt_seg must be torch.Tensor.")
        if pred_seg.device != gt_seg.device:
            raise ValueError(
                f"pred_seg and gt_seg must be on the same device, got "
                f"{pred_seg.device} vs {gt_seg.device}."
            )
        if fast and pred_seg.device.type != "cuda":
            raise RuntimeError(
                f"Fast evaluation only supports GPU tensors. Got device={pred_seg.device}."
            )

        metrics = {}

        pred_bin = (pred_seg >= threshold).float()
        gt_bin = gt_seg.float()
        reduce_dims = tuple(range(1, pred_bin.dim()))

        tp = (pred_bin * gt_bin).sum(dim=reduce_dims)
        fp = (pred_bin * (1 - gt_bin)).sum(dim=reduce_dims)
        fn = ((1 - pred_bin) * gt_bin).sum(dim=reduce_dims)

        dice_per_sample = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        dice_mean = dice_per_sample.mean()
        metrics["dice"] = dice_mean if return_tensor else float(dice_mean.item())

        if fast:
            # Normed Surface Dice
            nsd = nsd_3d_binary(
                pred_bin, gt_bin, tolerances=(1,), return_tensor=return_tensor
            )
            metrics.update(nsd)
            return metrics
        
        # Normed Surface Dice
        nsd = nsd_3d_binary(
            pred_bin, gt_bin, tolerances=(1, 2, 3, 5), return_tensor=return_tensor
        )
        metrics.update(nsd)

        # Hausdorff Distance and HD95
        hd, hd95 = hd_and_hd95_3d_binary(
            pred_bin, gt_bin, return_tensor=return_tensor
        )
        metrics['HD'] = hd
        metrics['HD95'] = hd95

        # clDice
        pred_bin_np = pred_bin.cpu().squeeze().clone().detach().byte().numpy()
        gt_seg_np = gt_seg.squeeze().cpu().clone().detach().byte().numpy()
        cldice = self.cl_dice(pred_bin_np, gt_seg_np)
        metrics["cldice"] = (
            torch.tensor(cldice, device=pred_bin.device)
            if return_tensor
            else cldice
        )

        # Betti number error
        (
            betti_0_error,
            betti_1_error,
            pred_betti0,
            pred_betti1,
            pred_betti2,
            gt_betti0,
            gt_betti1,
            gt_betti2,
        ) = self.betti_number_error(gt_seg_np, pred_bin_np)
        if return_tensor:
            metrics["betti_0_error"] = torch.tensor(betti_0_error, device=pred_bin.device)
            metrics["betti_1_error"] = torch.tensor(betti_1_error, device=pred_bin.device)
            metrics["pred_betti_0"] = torch.tensor(pred_betti0, device=pred_bin.device)
            metrics["pred_betti_1"] = torch.tensor(pred_betti1, device=pred_bin.device)
            metrics["pred_betti_2"] = torch.tensor(pred_betti2, device=pred_bin.device)
            metrics["gt_betti_0"] = torch.tensor(gt_betti0, device=pred_bin.device)
            metrics["gt_betti_1"] = torch.tensor(gt_betti1, device=pred_bin.device)
            metrics["gt_betti_2"] = torch.tensor(gt_betti2, device=pred_bin.device)
        else:
            metrics["betti_0_error"] = betti_0_error
            metrics["betti_1_error"] = betti_1_error
            metrics["pred_betti_0"] = pred_betti0
            metrics["pred_betti_1"] = pred_betti1
            metrics["pred_betti_2"] = pred_betti2
            metrics["gt_betti_0"] = gt_betti0
            metrics["gt_betti_1"] = gt_betti1
            metrics["gt_betti_2"] = gt_betti2

        if save_path:
            with open(save_path, "w") as f:
                json.dump(to_json_serializable(metrics), f, indent="\t")
        return metrics
