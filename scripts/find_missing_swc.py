#!/usr/bin/env python3
"""Find SWC files missing from one directory compared with another."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_REFERENCE_DIR = Path(
    "/gpfs-flash/hulab/yangzekang/neuron/neuron-trace/outputs/"
    "C2-cubes1937-iter10000/dscnet-dice/2026-04-25-16-17-35/preds-Kimimaro"
)
DEFAULT_TARGET_DIR = Path(
    "/gpfs-flash/hulab/yangzekang/neuron/neuron-trace/outputs/"
    "C2-cubes1937-iter10000/dscnet-dice/2026-04-25-16-17-35/preds-smartTrace"
)


def collect_swc_ids(swc_dir: Path) -> dict[str, Path]:
    if not swc_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {swc_dir}")

    swc_files = sorted(swc_dir.glob("*.swc"))
    return {swc_file.stem: swc_file for swc_file in swc_files}


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two SWC directories and print ids that are present in the "
            "reference directory but missing from the target directory."
        )
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Directory whose .swc ids are treated as complete/reference.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Directory checked for missing .swc ids.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save missing ids, one id per line.",
    )
    parser.add_argument(
        "--show-extra",
        action="store_true",
        help="Also print ids that exist in target but not in reference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference = collect_swc_ids(args.reference_dir)
    target = collect_swc_ids(args.target_dir)

    missing_ids = sorted(set(reference) - set(target))
    extra_ids = sorted(set(target) - set(reference))

    print(f"reference_dir: {args.reference_dir}")
    print(f"target_dir:    {args.target_dir}")
    print(f"reference .swc count: {len(reference)}")
    print(f"target .swc count:    {len(target)}")
    print(f"missing in target:    {len(missing_ids)}")

    if missing_ids:
        print("\nMissing SWC ids:")
        for swc_id in missing_ids:
            print(swc_id)
    else:
        print("\nNo missing SWC ids.")

    if args.output is not None:
        write_lines(args.output, missing_ids)
        print(f"\nSaved missing ids to: {args.output}")

    if args.show_extra:
        print(f"\nextra in target:      {len(extra_ids)}")
        if extra_ids:
            print("\nExtra SWC ids:")
            for swc_id in extra_ids:
                print(swc_id)


if __name__ == "__main__":
    main()
