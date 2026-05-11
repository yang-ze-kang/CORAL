import click
import glob
import os
import json
import statistics
from typing import List, Tuple, Dict, Any
from collections import defaultdict

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from swclib.metrics.manager import METRIC_MAP, default_metric_params, NumpyEncoder


def load_skeleton_seconds(pred_swc_path: str) -> float:
    log_path = pred_swc_path.replace(".swc", "_log.json")
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            ds = json.load(f)
        seconds = ds.get("seconds")
        if isinstance(seconds, (int, float)):
            return float(seconds)
    except Exception:
        return None
    return None


def cal_metric(pred_swc_path: str, gt_swc_path: str, result_path: str, metrics: tuple):
    result = {"gt_path": str(gt_swc_path), "pred_path": str(pred_swc_path)}
    for name, metric in metrics.items():
        result[name] = metric.run(gt_swc_path, pred_swc_path)

    # Use tracing time recorded by skeleton.py (from *_log.json), not metric compute time.
    skeleton_seconds = load_skeleton_seconds(pred_swc_path)
    result["skeleton_seconds"] = skeleton_seconds

    with open(result_path, "w") as f:
        json.dump(result, f, indent="\t", cls=NumpyEncoder)
    return result


def _match_by_basename(
    pred_paths: List[str], gt_dir: str, results_dir: str
) -> List[Tuple[str, str, str]]:
    """Match pred and gt by file basename (e.g., xxx.swc)."""
    pairs: List[Tuple[str, str, str]] = []
    for p in pred_paths:
        base = os.path.basename(p)
        g = os.path.join(gt_dir, base)
        if os.path.isfile(g):
            r = os.path.join(
                results_dir, f"{os.path.splitext(os.path.basename(p))[0]}.json"
            )
            pairs.append((p, g, r))
    return pairs


def collect_results(
    results_dir: str, metric_names: str, check_total_num: int
) -> Dict[str, Any]:
    """
    Collect JSON files under results_dir and write:
      - results_dir/results.jsonl
      - results_dir/summary.json
    Return summary dict.
    """
    paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    assert (
        len(paths) == check_total_num
    ), f"Expected {check_total_num} per-pair json files under {results_dir}, but found {len(paths)}. Please check your results_dir and matching criteria."
    if not paths:
        raise RuntimeError(f"No per-pair json found under: {results_dir}")

    summary_result = {"sample_num": len(paths)}
    skeleton_seconds_list: List[float] = []

    metric_acc: Dict[str, Dict[str, Any]] = {
        name: {
            "confusions": defaultdict(lambda: {"TP": 0.0, "FP": 0.0, "FN": 0.0}),
            "confusion_labels_seen": defaultdict(set),
            "scalars": defaultdict(lambda: {"sum": 0.0, "count": 0}),
        }
        for name in metric_names
    }

    def _is_number(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    def _add_scalar(name: str, key: str, value: Any):
        if _is_number(value):
            metric_acc[name]["scalars"][key]["sum"] += float(value)
            metric_acc[name]["scalars"][key]["count"] += 1

    def _add_confusion(name: str, key: str, payload: Dict[str, Any]):
        if not all(_is_number(payload.get(k)) for k in ("TP", "FP", "FN")):
            return
        metric_acc[name]["confusions"][key]["TP"] += float(payload["TP"])
        metric_acc[name]["confusions"][key]["FP"] += float(payload["FP"])
        metric_acc[name]["confusions"][key]["FN"] += float(payload["FN"])
        metric_acc[name]["confusion_labels_seen"][key].update({"TP", "FP", "FN"})

    def _parse_prefixed_confusion_key(k: Any) -> Tuple[str, str]:
        """
        Support flattened confusion keys like:
          - leaf_TP / leaf_FP / leaf_FN
          - TP_leaf / FP_leaf / FN_leaf
        Return (group_key, label) or (None, None) when not matched.
        """
        if not isinstance(k, str):
            return None, None
        parts = k.split("_")
        if len(parts) < 2:
            return None, None
        labels = {"TP", "FP", "FN"}
        if parts[-1] in labels:
            return "_".join(parts[:-1]), parts[-1]
        if parts[0] in labels:
            return "_".join(parts[1:]), parts[0]
        return None, None

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            ds = json.load(f)
            skeleton_seconds = ds.get("skeleton_seconds")
            if isinstance(skeleton_seconds, (int, float)):
                skeleton_seconds_list.append(float(skeleton_seconds))
            for name in metric_names:
                metric_payload = ds.get(name)
                if not isinstance(metric_payload, dict):
                    continue

                # Case 1: metric itself is TP/FP/FN structure.
                if all(k in metric_payload for k in ("TP", "FP", "FN")):
                    _add_confusion(name, "_root", metric_payload)

                # Case 2: aggregate each sub-metric key.
                for key, value in metric_payload.items():
                    if isinstance(value, dict) and all(k in value for k in ("TP", "FP", "FN")):
                        _add_confusion(name, key, value)
                        continue
                    if key in ("TP", "FP", "FN"):
                        continue
                    group_key, conf_label = _parse_prefixed_confusion_key(key)
                    if group_key and conf_label and _is_number(value):
                        metric_acc[name]["confusions"][group_key][conf_label] += float(value)
                        metric_acc[name]["confusion_labels_seen"][group_key].add(
                            conf_label
                        )
                        continue
                    _add_scalar(name, key, value)

    def _calc_prf(tp: float, fp: float, fn: float) -> Dict[str, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
        }

    for name in metric_names:
        summary_result[name] = {}

        # Root TP/FP/FN compatibility: keep old output shape when present.
        root_conf = metric_acc[name]["confusions"].get("_root")
        if root_conf:
            summary_result[name].update(
                _calc_prf(root_conf["TP"], root_conf["FP"], root_conf["FN"])
            )

        # Sub-metric TP/FP/FN outputs.
        for key, conf in metric_acc[name]["confusions"].items():
            if key == "_root":
                continue
            labels_seen = metric_acc[name]["confusion_labels_seen"].get(key, set())
            if not {"TP", "FP", "FN"}.issubset(labels_seen):
                # Ignore incomplete groups like TP_pred/TP_gold (no FP/FN).
                continue
            summary_result[name][key] = _calc_prf(conf["TP"], conf["FP"], conf["FN"])

        # Scalar outputs: sum + mean.
        mean_prf_keys = {"precision", "recall", "f1"}
        for key, stat in metric_acc[name]["scalars"].items():
            if stat["count"] == 0:
                # continue
                raise RuntimeError(f"No valid scalar value found for metric '{name}' key '{key}'.")
            avg = stat["sum"] / stat["count"]
            # Keep only mean for scalar values.
            # Distinguish sample-mean PRF from TP/FP/FN-derived PRF.
            if key in mean_prf_keys:
                out_key = f"macro_{key}"
                summary_result[name][out_key] = avg
            else:
                summary_result[name][key] = avg

    if skeleton_seconds_list:
        summary_result["skeleton_timing"] = {
            "count": len(skeleton_seconds_list),
            "total_seconds": sum(skeleton_seconds_list),
            "avg_seconds": statistics.mean(skeleton_seconds_list),
            "median_seconds": statistics.median(skeleton_seconds_list),
            "min_seconds": min(skeleton_seconds_list),
            "max_seconds": max(skeleton_seconds_list),
        }
        # Keep old key name for compatibility.
        summary_result["timing"] = dict(summary_result["skeleton_timing"])
    else:
        summary_result["skeleton_timing"] = {"count": 0}
        summary_result["timing"] = {"count": 0}

    return summary_result


def save_summary(summary: Dict[str, Any], results_dir: str) -> str:
    results_dir = os.path.normpath(results_dir)
    summary_json_path = f"{results_dir}.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    return summary_json_path


def _print_summary_from_summary(summary: Dict[str, Any]):
    for name in summary.keys():
        if isinstance(summary[name], dict):
            print(f"===={name}====")
            for key, val in summary[name].items():
                print(f"{key}:{val}")
            print()


@click.command()
@click.option(
    "--pred_swc_dir", type=str, required=True, help="Directory of predicted SWC files."
)
@click.option(
    "--gt_swc_dir", type=str, required=True, help="Directory of ground-truth SWC files."
)
@click.option(
    "--results_dir",
    type=str,
    required=True,
    help="Directory to save per-pair results and summary.",
)
@click.option(
    "--metric_names",
    multiple=True,
    type=click.Choice(
        ["ssd", "point", "length", "keypoints", "fiber"], case_sensitive=False
    ),
    default=("ssd", "point", "length", "keypoints", "fiber"),
    show_default=True,
    help="Which metrics to compute.",
)
@click.option("--check-total-num", default=694, type=int, show_default=True)
@click.option(
    "--workers",
    default=None,
    type=int,
    show_default=True,
    help="Number of processes. Default: os.cpu_count().",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Run single-process serially; raise exceptions for debugging.",
)
@click.option("--verbose", is_flag=True)
@click.option(
    "--skip_existing",
    is_flag=True,
    help="Skip pairs whose per-pair json already exists.",
)
def main(
    pred_swc_dir,
    gt_swc_dir,
    results_dir,
    metric_names,
    check_total_num,
    workers,
    debug,
    verbose,
    skip_existing,
):
    pred_paths = sorted(glob.glob(os.path.join(pred_swc_dir, "*.swc")))
    if not pred_paths:
        raise click.ClickException(f"No .swc files found in: {pred_swc_dir}")
    if not os.path.isdir(gt_swc_dir):
        raise click.ClickException(f"gt_swc_dir is not a directory: {gt_swc_dir}")

    os.makedirs(results_dir, exist_ok=True)
    pairs = _match_by_basename(pred_paths, gt_swc_dir, results_dir)
    assert len(pairs) == check_total_num, f"Expected {check_total_num} matched pairs, but found {len(pairs)}. Please check your pred/gt directories and matching criteria."
    if not pairs:
        raise click.ClickException("No matched pred/gt pairs (matching by basename).")

    # Optional skip existing outputs
    if skip_existing:
        new_pairs = []
        for p, g, r in pairs:
            if not os.path.isfile(r):
                new_pairs.append((p, g, r))
        pairs = new_pairs

    n = len(pairs)
    click.echo(f"Matched {n} pairs (by basename).")
    click.echo(f"Results will be saved under: {results_dir}")

    # Metrics
    metrics = {}
    for name in metric_names:
        metric_class = METRIC_MAP[name]
        metric = metric_class(**default_metric_params[name])
        metrics[name] = metric

    # ---- DEBUG: single-process ----
    if debug:
        click.echo("[DEBUG] Single-process mode (exceptions will be raised).")
        for pred_path, gt_path, result_path in tqdm(
            pairs, total=n, desc="Metric(debug)", unit="pair"
        ):
            print(
                f"Processing pair:\n  pred: {pred_path}\n  gt: {gt_path}\n  result: {result_path}"
            )
            cal_metric(pred_path, gt_path, result_path, metrics)  # raise on error
        summary = collect_results(results_dir, metric_names, check_total_num)
        summary_path = save_summary(summary, results_dir)
        click.echo(f"Saved summary json: {summary_path}")
        _print_summary_from_summary(summary)
        return

    # ---- MULTI-PROCESS ----
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1
    workers = min(workers, n) if n > 0 else 1

    click.echo(f"Multi-process mode. workers={workers}")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(cal_metric, p, g, r, metrics) for p, g, r in pairs]
        # We only collect tiny dicts for progress; actual outputs are already on disk.
        for _ in tqdm(
            as_completed(futures), total=len(futures), desc="Metric", unit="pair"
        ):
            pass

    summary = collect_results(results_dir, metric_names, check_total_num)
    summary_path = save_summary(summary, results_dir)
    click.echo(f"Saved summary json: {summary_path}")
    _print_summary_from_summary(summary)


if __name__ == "__main__":
    main()
