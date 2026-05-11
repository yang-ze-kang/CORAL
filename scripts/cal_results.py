from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize(values: List[float]) -> Dict[str, float]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(var)
    return {
        "count": float(n),
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def main(dir_path: str, pattern: str = "*.json") -> None:
    root = Path(dir_path)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern {pattern} in {root}")

    metric_values: Dict[str, List[float]] = {}
    bad_files: List[Tuple[str, str]] = []

    for fp in files:
        try:
            data = load_json(fp)
            if not isinstance(data, dict):
                bad_files.append((str(fp), "JSON root is not an object"))
                continue

            for k, v in data.items():
                if is_number(v):
                    metric_values.setdefault(k, []).append(float(v))
                # 非数值字段直接跳过
        except Exception as e:
            bad_files.append((str(fp), repr(e)))

    # 统计输出
    print(f"Found {len(files)} JSON files.")
    if bad_files:
        print(f"Warning: {len(bad_files)} files failed to parse or had invalid format:")
        for p, err in bad_files[:10]:
            print(f"  - {p}: {err}")
        if len(bad_files) > 10:
            print("  ...")

    # 对齐：有些文件可能缺某些key，这里默认“按出现的值”计算均值
    # 如果你希望“缺失就当0”或“必须所有文件都有该key”，可以再改一版
    rows = []
    for k in sorted(metric_values.keys()):
        vals = metric_values[k]
        stats = summarize(vals)
        rows.append((k, int(stats["count"]), stats["mean"], stats["std"], stats["min"], stats["max"]))

    # 美观打印
    header = ("metric", "n", "mean", "std", "min", "max")
    colw = [max(len(header[i]), max((len(str(r[i])) for r in rows), default=0)) for i in range(len(header))]

    def fmt_row(r):
        return "  ".join(str(r[i]).ljust(colw[i]) for i in range(len(r)))

    print(fmt_row(header))
    print("-" * (sum(colw) + 2 * (len(colw) - 1)))
    for k, n, mean, std, mn, mx in rows:
        print(f"{k.ljust(colw[0])}  {str(n).ljust(colw[1])}  {mean: .6f}  {std: .6f}  {mn: .6f}  {mx: .6f}")


if __name__ == "__main__":
    # json_dir = "/data1/yangzekang/neuron/neuron-trace/outputs/neuron-seg/CH1-iter10000/dynunet-cldice-iter10/2026-01-30-18-21-26/preds"
    json_dir = '/data1/yangzekang/neuron/neuron-trace/outputs/neuron-seg/CH1-iter10000/adtlnet-dice/2026-02-04-00-26-34/preds'
    main(json_dir)