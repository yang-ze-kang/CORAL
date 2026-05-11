#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


DEFAULT_INPUT_DIR = "outputs/whole_brain_trace_metrics"
DEFAULT_OUTPUT = (
    "outputs/results-whole-brain-trace-metrics.json"
)
DEFAULT_METHODS = ("APP2", "Kimimaro", "neuTube")


def parse_csv(arg: str) -> Set[str]:
    items = [x.strip() for x in arg.split(",") if x.strip()]
    return {x.lower() for x in items}


def parse_metric_filename(filename: str, method_names: List[str]) -> Tuple[str, str]:
    # Example: dynunet-cldice_neuTube_bfs-metrics.json
    #   -> ("dynunet-cldice", "neuTube_bfs")
    name = filename
    if name.endswith("-metrics.json"):
        name = name[: -len("-metrics.json")]

    parts = name.split("_")
    method_lut = {m.lower() for m in method_names}
    for idx, part in enumerate(parts):
        if part.lower() in method_lut and idx > 0:
            seg_name = "_".join(parts[:idx])
            trace_name = "_".join(parts[idx:])
            return seg_name, trace_name

    if "_" not in name:
        raise ValueError(f"Cannot parse metric filename without '_': {filename}")

    seg_name, trace_name = name.rsplit("_", 1)
    return seg_name, trace_name


def trace_method(trace_name: str) -> str:
    return trace_name.split("_", 1)[0]


def collect_metrics(
    input_dir: Path,
    method_names: List[str],
    seg_filter: Set[str],
    method_filter: Set[str],
) -> Dict[str, Dict[str, dict]]:
    merged: Dict[str, Dict[str, dict]] = {}

    metric_files = sorted(input_dir.glob("*-metrics.json"))
    if not metric_files:
        print(f"[WARN] No '*-metrics.json' found in: {input_dir}")

    for metric_file in metric_files:
        seg_name, trace_name = parse_metric_filename(metric_file.name, method_names)
        method_name = trace_method(trace_name)

        if seg_filter and seg_name.lower() not in seg_filter:
            continue
        if method_filter and method_name.lower() not in method_filter:
            continue

        with metric_file.open("r", encoding="utf-8") as f:
            metric_obj = json.load(f)

        merged.setdefault(seg_name, {})[trace_name] = metric_obj

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate whole-brain tracing metrics into "
            '{"seg_name": {"trace_name": {...}}}.'
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing '*-metrics.json' files (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--method-names",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated method tokens used to split filenames "
            f'(default: {",".join(DEFAULT_METHODS)}).'
        ),
    )
    parser.add_argument(
        "--segs",
        type=str,
        default="",
        help='Comma-separated segmentation names to keep, e.g. "dynunet,vnet".',
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help='Comma-separated trace methods to keep, e.g. "APP2,Kimimaro,neuTube".',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}). Use empty string to print.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent (default: 2).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    method_names = [m.strip() for m in args.method_names.split(",") if m.strip()]
    if not method_names:
        raise SystemExit("No method names provided. Use --method-names.")

    seg_filter = parse_csv(args.segs) if args.segs else set()
    method_filter = parse_csv(args.methods) if args.methods else set()
    merged = collect_metrics(input_dir, method_names, seg_filter, method_filter)

    out = json.dumps(merged, ensure_ascii=False, indent=args.indent)
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out + "\n", encoding="utf-8")
        print(f"[OK] Wrote: {out_path}")
    else:
        print(out)


if __name__ == "__main__":
    main()
