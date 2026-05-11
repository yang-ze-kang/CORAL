import click
import glob
import os
import time
import json
import statistics
import traceback
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from collections import defaultdict

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from swclib.metrics.manager import build_metrics, default_metric_params_wb, NumpyEncoder
from swclib.data.swc import Swc


def find_pred_swc_paths(pred_swc_dir: str) -> List[str]:
    """
    Find predicted SWCs stored as:
      pred_swc_dir/neuron-04/neuron-04.swc
    """
    paths = []
    for p in glob.glob(os.path.join(pred_swc_dir, "*", "*.swc")):
        neuron_name = os.path.basename(os.path.dirname(p))
        if os.path.basename(p) == f"{neuron_name}.swc":
            paths.append(p)
    return sorted(paths)


# TODO: change this import to your real metric function location
# from your_pkg.metrics import cal_metric
def cal_metric(pred_swc_path: str, gt_swc_path: str, result_path: str, metrics: tuple):
    result = {"gt_path": str(gt_swc_path), "pred_path": str(pred_swc_path)}
    for name, metric in metrics.items():
        result[name] = metric.run(gt_swc_path, pred_swc_path)
    log_path = os.path.join(os.path.dirname(pred_swc_path), "trace_log.jsonl")
    ts = 0
    with open(log_path, "r") as f:
        logs = [json.loads(line) for line in f]
        for log in logs:
            ts += log["trace_time"] + log["merge_time"] + log["search_time"]
    result["trace_time"] = ts
    swc = Swc(pred_swc_path)
    result['total_length'] = swc.length
    swc = Swc(gt_swc_path)
    result['total_length_gt'] = swc.length
    with open(result_path, "w") as f:
        json.dump(result, f, indent="\t", cls=NumpyEncoder)
    return result


@dataclass
class JobResult:
    pred_path: str
    gt_path: str
    ok: bool
    seconds: float
    metric: Optional[object] = None
    error: Optional[str] = None


def _match_by_basename(
    pred_paths: List[str], gt_dir: str, results_dir: str
) -> List[Tuple[str, str]]:
    """Match pred and gt by file basename (e.g., xxx.swc)."""
    pairs: List[Tuple[str, str, str]] = []
    for p in pred_paths:
        base = os.path.basename(os.path.dirname(p))
        base = base + ".swc"
        g = os.path.join(gt_dir, base)
        if os.path.isfile(g):
            r = os.path.join(results_dir, f"{base.replace('.swc', '.json')}")
            pairs.append((p, g, r))
    return pairs


def _neuron_name_from_pred_path(pred_path: str) -> str:
    return f"{os.path.basename(os.path.dirname(pred_path))}.swc"


def collect_neuron_counts(
    pred_paths: List[str], gt_dir: str, pairs: List[Tuple[str, str, str]]
) -> Dict[str, Any]:
    pred_names = {_neuron_name_from_pred_path(p) for p in pred_paths}
    gt_names = {
        os.path.basename(p)
        for p in glob.glob(os.path.join(gt_dir, "*.swc"))
        if os.path.isfile(p)
    }
    matched_names = {os.path.basename(gt_path) for _, gt_path, _ in pairs}
    return {
        "pred_neuron_num": len(pred_names),
        "gt_neuron_num": len(gt_names),
        "matched_neuron_num": len(matched_names),
    }


def _run_one_and_save(
    pred_path: str,
    gt_path: str,
    result_path: str,
    metrics: tuple,
    results_dir: str,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Worker task: compute metric and save per-pair JSON.
    Return a tiny dict for progress/summary (avoid shipping large metric objects back).
    """
    t0 = time.perf_counter()
    try:
        m = cal_metric(pred_path, gt_path, result_path, metrics)
        r = JobResult(
            pred_path=pred_path,
            gt_path=gt_path,
            ok=True,
            seconds=time.perf_counter() - t0,
            metric=m,
        )
    except Exception:
        r = JobResult(
            pred_path=pred_path,
            gt_path=gt_path,
            ok=False,
            seconds=time.perf_counter() - t0,
            metric=None,
            error=traceback.format_exc(),
        )
    return {
        "pred_path": r.pred_path,
        "gt_path": r.gt_path,
        "ok": r.ok,
        "seconds": r.seconds,
        "error": r.error,
    }


def collect_results(
    results_dir: str,
    metric_names: Tuple[str, ...],
    check_total_num: Optional[int] = None,
    neuron_counts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Collect JSON files under results_dir and write:
      - results_dir/results.jsonl
      - results_dir/summary.json
    Return summary dict.
    """
    paths = [
        p
        for p in sorted(glob.glob(os.path.join(results_dir, "*.json")))
        if os.path.basename(p) not in {"summary.json", "results.json"}
    ]
    if check_total_num is not None and len(paths) != check_total_num:
        raise RuntimeError(
            f"Expected {check_total_num} per-pair json files under {results_dir}, "
            f"but found {len(paths)}. Please check your results_dir and matching criteria."
        )
    if not paths:
        raise RuntimeError(f"No per-pair json found under: {results_dir}")

    summary_result = {"sample_num": len(paths)}
    if neuron_counts is not None:
        summary_result["neuron_counts"] = neuron_counts

    trace_time_list: List[float] = []
    total_length = 0.0
    total_length_gt = 0.0

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

    def _parse_prefixed_confusion_key(k: Any) -> Tuple[Optional[str], Optional[str]]:
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
            trace_time = ds.get("trace_time")
            if _is_number(trace_time):
                trace_time_list.append(float(trace_time))
            if _is_number(ds.get("total_length")):
                total_length += float(ds["total_length"])
            if _is_number(ds.get("total_length_gt")):
                total_length_gt += float(ds["total_length_gt"])
            for name in metric_names:
                metric_payload = ds.get(name)
                if not isinstance(metric_payload, dict):
                    continue

                if all(k in metric_payload for k in ("TP", "FP", "FN")):
                    _add_confusion(name, "_root", metric_payload)

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

        root_conf = metric_acc[name]["confusions"].get("_root")
        if root_conf:
            summary_result[name].update(
                _calc_prf(root_conf["TP"], root_conf["FP"], root_conf["FN"])
            )

        for key, conf in metric_acc[name]["confusions"].items():
            if key == "_root":
                continue
            labels_seen = metric_acc[name]["confusion_labels_seen"].get(key, set())
            if not {"TP", "FP", "FN"}.issubset(labels_seen):
                continue
            summary_result[name][key] = _calc_prf(conf["TP"], conf["FP"], conf["FN"])

        mean_prf_keys = {"precision", "recall", "f1"}
        for key, stat in metric_acc[name]["scalars"].items():
            if stat["count"] == 0:
                raise RuntimeError(
                    f"No valid scalar value found for metric '{name}' key '{key}'."
                )
            avg = stat["sum"] / stat["count"]
            if key in mean_prf_keys:
                summary_result[name][f"macro_{key}"] = avg
            else:
                summary_result[name][key] = avg

    if trace_time_list:
        trace_time_total = sum(trace_time_list)
        summary_result["trace_timing"] = {
            "count": len(trace_time_list),
            "total_seconds": trace_time_total,
            "avg_seconds": statistics.mean(trace_time_list),
            "median_seconds": statistics.median(trace_time_list),
            "min_seconds": min(trace_time_list),
            "max_seconds": max(trace_time_list),
        }
        summary_result["trace_time"] = trace_time_total
    if total_length > 0:
        summary_result["total_length"] = total_length
    if total_length_gt > 0:
        summary_result["total_length_gt"] = total_length_gt
    if total_length > 0 and trace_time_list and sum(trace_time_list) > 0:
        summary_result["trace_voxel_per_second"] = total_length / sum(trace_time_list)
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
        else:
            print(f"{name}:{summary[name]}")


@click.command()
@click.option(
    "--pred-swc-dir",
    "--pred_swc_dir",
    type=str,
    required=True,
    help="Directory of predicted SWC files.",
)
@click.option(
    "--gt-swc-dir",
    "--gt_swc_dir",
    type=str,
    required=True,
    help="Directory of ground-truth SWC files.",
)
@click.option(
    "--results-dir",
    "--results_dir",
    type=str,
    required=True,
    help="Directory to save per-pair results and summary.",
)
@click.option(
    "--metric-names",
    "--metric_names",
    multiple=True,
    type=click.Choice(["ssd", "point", "length", "keypoints", "fiber"], case_sensitive=False),
    default=("ssd", "point", "length", "keypoints", "fiber"),
    show_default=True,
    help="Which metrics to compute.",
)
@click.option(
    "--check-total-num",
    "--check_total_num",
    default=None,
    type=int,
    help="Expected matched neuron count and per-pair json count.",
)
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
    "--skip-existing",
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
    pred_paths = find_pred_swc_paths(pred_swc_dir)
    if not pred_paths:
        raise click.ClickException(
            f"No predicted SWC files found under {pred_swc_dir}. "
            "Expected files like: pred_swc_dir/neuron-04/neuron-04.swc"
        )
    if not os.path.isdir(gt_swc_dir):
        raise click.ClickException(f"gt_swc_dir is not a directory: {gt_swc_dir}")

    os.makedirs(results_dir, exist_ok=True)
    pairs = _match_by_basename(pred_paths, gt_swc_dir, results_dir)
    neuron_counts = collect_neuron_counts(pred_paths, gt_swc_dir, pairs)
    click.echo(
        "Neuron counts: "
        f"pred={neuron_counts['pred_neuron_num']}, "
        f"gt={neuron_counts['gt_neuron_num']}, "
        f"matched={neuron_counts['matched_neuron_num']}"
    )
    if check_total_num is not None and len(pairs) != check_total_num:
        raise click.ClickException(
            f"Expected {check_total_num} matched pairs, but found {len(pairs)}. "
            "Please check your pred/gt directories and matching criteria."
        )
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
    metrics = build_metrics(metric_names, default_metric_params_wb)

    # ---- DEBUG: single-process ----
    if debug:
        click.echo("[DEBUG] Single-process mode (exceptions will be raised).")
        for pred_path, gt_path, result_path in tqdm(
            pairs, total=n, desc="Metric(debug)", unit="pair"
        ):
            t0 = time.perf_counter()
            m = cal_metric(pred_path, gt_path, result_path, metrics)  # raise on error
            r = JobResult(pred_path, gt_path, True, time.perf_counter() - t0, metric=m)
        summary = collect_results(
            results_dir, metric_names, check_total_num, neuron_counts
        )
        summary_path = save_summary(summary, results_dir)
        click.echo(f"Saved summary json: {summary_path}")
        _print_summary_from_summary(summary)
        return

    # ---- MULTI-PROCESS ----
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1
    workers = min(workers, n) if n > 0 else 1

    click.echo(f"Multi-process mode. workers={workers}")

    failed_jobs = []
    if pairs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_run_one_and_save, p, g, r, metrics, results_dir, verbose)
                for p, g, r in pairs
            ]
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Metric", unit="pair"
            ):
                job = fut.result()
                if not job["ok"]:
                    failed_jobs.append(job)
    if failed_jobs:
        first = failed_jobs[0]
        raise click.ClickException(
            f"{len(failed_jobs)} metric jobs failed. First failure:\n"
            f"pred: {first['pred_path']}\n"
            f"gt: {first['gt_path']}\n"
            f"{first['error']}"
        )

    summary = collect_results(results_dir, metric_names, check_total_num, neuron_counts)
    summary_path = save_summary(summary, results_dir)
    click.echo(f"Saved summary json: {summary_path}")
    _print_summary_from_summary(summary)


if __name__ == "__main__":
    main()
