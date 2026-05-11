#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


def parse_methods(methods_arg: str) -> Set[str]:
    methods = [m.strip() for m in methods_arg.split(",") if m.strip()]
    return {m.lower() for m in methods}


def load_dirs_from_file(file_path: Path) -> List[str]:
    lines = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def normalize_trace_name(filename: str) -> str:
    # Example: preds-APP2-metrics.json -> APP2
    name = filename
    if name.endswith("-metrics.json"):
        name = name[: -len("-metrics.json")]
    if name.startswith("preds-"):
        name = name[len("preds-") :]
    return name


def collect_metrics(run_dirs: List[Path], method_filter: Set[str]) -> Dict[str, Dict[str, dict]]:
    merged: Dict[str, Dict[str, dict]] = {}

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"[WARN] Skip non-directory path: {run_dir}")
            continue

        seg_name = run_dir.parent.name
        merged.setdefault(seg_name, {})

        metric_files = sorted(run_dir.glob("*-metrics.json"))
        if not metric_files:
            print(f"[WARN] No '*-metrics.json' found in: {run_dir}")
            continue

        for metric_file in metric_files:
            trace_name = normalize_trace_name(metric_file.name)
            if method_filter and trace_name.lower() not in method_filter:
                continue

            with metric_file.open("r", encoding="utf-8") as f:
                metric_obj = json.load(f)

            merged[seg_name][trace_name] = metric_obj

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate tracing metrics from multiple run dirs into "
            '{"seg_name": {"trace_name": {...}}}.'
        )
    )
    parser.add_argument(
        "--run-dirs",
        nargs="*",
        default=[],
        help="Run directories (e.g., .../dscnet-dice/2026-04-25-16-17-35).",
    )
    parser.add_argument(
        "--run-dirs-file",
        type=str,
        default="",
        help="Text file listing run directories, one path per line.",
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
        default="",
        help="Output JSON path. If empty, print to stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent (default: 2).",
    )

    args = parser.parse_args()

    run_dirs = [Path(p).expanduser().resolve() for p in args.run_dirs]
    if args.run_dirs_file:
        run_dirs_file = Path(args.run_dirs_file).expanduser().resolve()
        run_dirs.extend(Path(p).expanduser().resolve() for p in load_dirs_from_file(run_dirs_file))

    if not run_dirs:
        raise SystemExit("No run directories provided. Use --run-dirs or --run-dirs-file.")

    method_filter = parse_methods(args.methods) if args.methods else set()
    merged = collect_metrics(run_dirs, method_filter)

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
