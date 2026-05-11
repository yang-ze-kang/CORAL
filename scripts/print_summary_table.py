#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def get_by_path(obj: Dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for key in path.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def format_value(v: Any, precision: int, scale: float) -> str:
    if v is None:
        return "NA"
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "NA"
        return f"{(v * scale):.{precision}f}"
    if isinstance(v, int):
        return f"{(v * scale):.{precision}f}"
    return str(v)


def print_aligned_table(header: List[str], rows: List[List[str]], add_group_separator: bool) -> None:
    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: List[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    line = "-" * (sum(widths) + 2 * (len(widths) - 1))
    print(line)
    print(fmt_row(header))
    print(line)
    prev_seg = None
    for row in rows:
        if add_group_separator and prev_seg is not None and row[0] != prev_seg:
            print(line)
        print(fmt_row(row))
        prev_seg = row[0]
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print selected metrics from summary_metrics.json as a table."
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        required=True,
        help="Path to summary_metrics.json",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help='Comma-separated metric paths, e.g. "length.f1_score,ssd.ssd_percent"',
    )
    parser.add_argument(
        "--sep",
        type=str,
        default="",
        help="Column separator. If set, use separated output; if empty, print aligned table.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Float precision, default 2.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="",
        help="Metric path to sort by (descending). Only used when --order=metric.",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="seg_trace",
        choices=["seg_trace", "metric"],
        help="Row order: seg_trace (default) or metric.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Scale factor for numeric metric values (default: 100).",
    )
    parser.add_argument(
        "--no-scale-metrics",
        type=str,
        default="ssd.sd,ssd.ssd",
        help='Comma-separated metric paths that should NOT be scaled (default: "ssd.sd,ssd.ssd").',
    )

    args = parser.parse_args()

    summary_path = Path(args.summary_json).expanduser().resolve()
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    metric_paths: List[str] = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metric_paths:
        raise SystemExit("No metrics provided.")
    no_scale_metrics = {m.strip() for m in args.no_scale_metrics.split(",") if m.strip()}

    rows: List[List[str]] = []
    sort_values: List[Any] = []

    for seg_name in sorted(data.keys()):
        traces = data[seg_name]
        if not isinstance(traces, dict):
            continue
        for trace_name in sorted(traces.keys()):
            metric_obj = traces[trace_name]
            if not isinstance(metric_obj, dict):
                continue

            seg_method = seg_name
            trace_method = trace_name

            vals = [get_by_path(metric_obj, p) for p in metric_paths]
            scaled_cells: List[str] = []
            for metric_path, v in zip(metric_paths, vals):
                scale = 1.0 if metric_path in no_scale_metrics else args.scale
                scaled_cells.append(format_value(v, args.precision, scale))
            rows.append([seg_method, trace_method] + scaled_cells)

            if args.sort_by:
                sort_values.append(get_by_path(metric_obj, args.sort_by))
            else:
                sort_values.append(None)

    if args.order == "metric":
        if not args.sort_by:
            raise SystemExit("When --order=metric, --sort-by is required.")
        idx = list(range(len(rows)))

        def key_func(i: int):
            v = sort_values[i]
            if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return (0, float(v))
            return (1, float('-inf'))

        idx.sort(key=key_func, reverse=True)
        rows = [rows[i] for i in idx]

    header = ["seg_method", "trace_method"] + metric_paths
    if args.sep:
        print(args.sep.join(header))
        for row in rows:
            print(args.sep.join(row))
    else:
        print_aligned_table(header, rows, add_group_separator=(args.order == "seg_trace"))


if __name__ == "__main__":
    main()
