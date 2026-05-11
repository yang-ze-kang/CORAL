import click
import glob
import os
import time
import statistics
import traceback
from dataclasses import dataclass, asdict
from typing import Optional, List
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json
import requests
from pathlib import Path

from postprocess.trace_kimimaro import trace_kimimaro
from postprocess.trace_skel import trace_skel
from postprocess.neutube import trace_neutube
from postprocess.spe_dnr import trace_spe_dnr
from postprocess.netracer import trace_netracer


@dataclass
class JobResult:
    pred_path: str
    out_swc_path: str
    ok: bool
    seconds: float
    error: Optional[str] = None


def trace_vaa3d(url, tif_file, swc_file):
    assert isinstance(tif_file, str) and isinstance(swc_file, str)
    try:
        with open(tif_file, "rb") as f:
            files = {"file": f}
            r = requests.post(url, files=files, timeout=3600)

        if r.status_code != 200:
            print(f"[FAIL] {tif_file} -> status {r.status_code}")
            return False
        else:
            with open(swc_file, "wb") as out:
                out.write(r.content)
            return True
    except Exception as e:
        print(f"[ERROR] {tif_file}: {e}")
        return False


def run_trace(
    pred: str,
    out_swc_path: str,
    method: str,
    raw: str = None,
    verbose: bool = False,
    max_try: int = 3,
) -> tuple[bool, float]:
    """
    Run tracing with retry support.

    Args:
        pred: Input prediction file path.
        out_swc_path: Output SWC file path.
        method: Tracing method name.
        raw: Raw image path, required by some methods.
        verbose: Whether to enable verbose mode for supported methods.
        max_try: Maximum number of attempts.

    Returns:
        (stat, elapsed_sec)
        stat: Whether tracing succeeded.
        elapsed_sec: Total elapsed time in seconds.
    """
    if max_try < 1:
        raise ValueError(f"max_try must be >= 1, but got {max_try}")

    stat = False
    last_error = None
    for attempt in range(1, max_try + 1):
        try:
            start_time = time.perf_counter()
            stat = True
            if method.lower() == "kimimaro":
                trace_kimimaro(pred, out_swc_path)
            elif method == "skel":
                trace_skel(pred, out_swc_path)
            elif method == "neuTube":
                trace_neutube(pred, out_swc_path)
            elif method == "APP2":
                url = "http://127.0.0.1:8000/trace_vaa3d_app2"
                stat = trace_vaa3d(url, pred, out_swc_path)
            elif method == "smartTrace":
                url = "http://127.0.0.1:8000/trace_vaa3d_smartTrace"
                stat = trace_vaa3d(url, pred, out_swc_path)
            elif method == "SPE-DNR":
                # assert (
                #     raw is not None
                # ), "SPE-DNR method requires --raw_dir to be specified"
                if raw is None:
                    raw = pred 
                trace_spe_dnr(raw, pred, out_swc_path)
            elif method == "NETracer":
                # assert (
                #     raw is not None
                # ), "NETracer method requires --raw_dir to be specified"
                if raw is None:
                    raw = pred
                trace_netracer(raw, pred, out_swc_path, verbose)
            else:
                raise NotImplementedError(f"Unknown method: {method}")

            if not stat:
                raise RuntimeError(
                    f"Tracing backend returned stat=False for method {method}"
                )

            if not Path(out_swc_path).exists():
                raise FileNotFoundError(
                    f"Tracing method {method} did not produce output file {os.path.basename(out_swc_path)}"
                )

            elapsed_sec = time.perf_counter() - start_time
            return True, elapsed_sec

        except Exception as e:
            stat = False
            last_error = e
            print(f"[WARN] Attempt {attempt}/{max_try} failed for {method}: {e}")
            if attempt == max_try:
                break

    if not Path(out_swc_path).exists():
        print(
            f"[ERROR] Tracing method {method} failed after {max_try} attempts, "
            # f"and I generate a none file"
        )
        # with open(out_swc_path, "w") as f:
        #     f.write(f"# ERROR: SWC file not created by tracing method {method}.\n")
        #     if last_error is not None:
        #         f.write(f"# Last error: {repr(last_error)}\n")

    elapsed_sec = time.perf_counter() - start_time
    return False, elapsed_sec


def _run_one(
    pred_path: str, out_swc_path: str, method: str, raw_path: str = None
) -> JobResult:
    """Worker function. Must be top-level for multiprocessing pickling."""
    try:
        stat, elapsed_sec = run_trace(pred_path, out_swc_path, method, raw_path)
        res = JobResult(pred_path, out_swc_path, stat, elapsed_sec)
    except Exception:
        res = JobResult(
            pred_path=pred_path,
            out_swc_path=out_swc_path,
            ok=False,
            seconds=elapsed_sec,
            error=traceback.format_exc(),
        )
        with open(out_swc_path, "w") as f:
            f.write(f"# ERROR during tracing:\n# {res.error}\n")
    log_path = out_swc_path.replace(".swc", "_log.json")
    with open(log_path, "w") as f:
        json.dump(asdict(res), f, indent=2)
    return res


def _print_and_save_summary(
    results: List[JobResult], total_seconds: float, time_csv: Optional[str]
) -> None:
    ok_results = [r for r in results if r.ok]
    failures = [r for r in results if not r.ok]
    ok_times = [r.seconds for r in ok_results]

    click.echo("\n=== Timing summary ===")
    click.echo(f"Total wall time: {total_seconds:.3f} s")
    click.echo(
        f"Success: {len(ok_results)}/{len(results)}, Failed: {len(failures)}/{len(results)}"
    )

    if ok_times:
        click.echo(f"Avg per-file (success): {statistics.mean(ok_times):.3f} s")
        click.echo(f"Median per-file (success): {statistics.median(ok_times):.3f} s")
        click.echo(
            f"Min/Max per-file (success): {min(ok_times):.3f} / {max(ok_times):.3f} s"
        )

    click.echo("\n=== Slowest files (success) ===")
    for r in sorted(ok_results, key=lambda x: x.seconds, reverse=True)[:10]:
        click.echo(
            f"{r.seconds:9.3f} s  {os.path.basename(r.pred_path)} -> {os.path.basename(r.out_swc_path)}"
        )

    if failures:
        click.echo("\n=== Failures ===")
        for r in failures:
            click.echo(f"[FAILED] {os.path.basename(r.pred_path)} ({r.seconds:.3f} s)")
            # Print last line for brevity; full traceback is in CSV if enabled.
            click.echo(
                r.error.rstrip().splitlines()[-1] if r.error else "Unknown error"
            )

    if time_csv:
        import csv

        os.makedirs(os.path.dirname(time_csv) or ".", exist_ok=True)
        with open(time_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pred_path", "out_swc_path", "ok", "seconds", "error"])
            for r in sorted(results, key=lambda x: x.pred_path):
                w.writerow(
                    [
                        r.pred_path,
                        r.out_swc_path,
                        int(r.ok),
                        f"{r.seconds:.6f}",
                        r.error or "",
                    ]
                )
        click.echo(f"\nWrote per-file timing CSV: {time_csv}")


@click.command()
@click.option("--pred_dir", type=str, required=True)
@click.option("--out_swc_dir", type=str, required=True)
@click.option("--raw_dir", type=str, required=False)
@click.option("--method", default="Kimimaro", type=str, show_default=True)
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
@click.option("--verbose", is_flag=True, help="Verbose logging inside tracing.")
@click.option(
    "--time_csv",
    default=None,
    type=str,
    help="Optional: write per-file timing CSV to this path.",
)
@click.option(
    "--skip_existing", is_flag=True, help="Skip files where output SWC already exists."
)
def main(
    pred_dir,
    out_swc_dir,
    raw_dir,
    method,
    workers,
    debug,
    verbose,
    time_csv,
    skip_existing,
):
    os.makedirs(out_swc_dir, exist_ok=True)

    preds = sorted(glob.glob(os.path.join(pred_dir, "*.tif")))
    if not preds:
        raise click.ClickException(f"No .tif files found in: {pred_dir}")

    raw_paths = (
        [os.path.join(raw_dir, os.path.basename(p).replace("_mask", "")) for p in preds]
        if raw_dir is not None
        else [None] * len(preds)
    )

    out_swc_paths = [
        os.path.join(out_swc_dir, os.path.basename(p).replace(".tif", ".swc"))
        for p in preds
    ]
    print(
        f"Found {len(preds)} prediction files. Method: {method}. Workers: {workers or 'auto'}"
    )

    if skip_existing:
        filtered = []
        for pred, out_swc, raw_path in zip(preds, out_swc_paths, raw_paths):
            if os.path.exists(out_swc):
                # click.echo(f"[SKIP] Output exists: {out_swc}")
                continue
            else:
                filtered.append((pred, out_swc, raw_path))
        if not filtered:
            click.echo("All files already processed. Nothing to do.")
            return
        preds, out_swc_paths, raw_paths = zip(*filtered)

    n = len(preds)

    # ---- DEBUG: force single-process, raise exceptions ----
    if debug:
        click.echo(f"[DEBUG] Single-process mode. Files={n}, method={method}")
        t_all0 = time.perf_counter()
        results: List[JobResult] = []

        for pred, out_swc, raw_path in tqdm(
            list(zip(preds, out_swc_paths, raw_paths)),
            total=n,
            desc="Tracing(debug)",
            unit="file",
        ):
            print(
                f"\n[DEBUG] Tracing {os.path.basename(pred)} -> {os.path.basename(out_swc)} with method {method}"
            )
            t0 = time.perf_counter()
            # In debug mode: raise immediately on error to see full stacktrace
            run_trace(pred, out_swc, method, raw_path, verbose=True)
            results.append(JobResult(pred, out_swc, True, time.perf_counter() - t0))

        total_seconds = time.perf_counter() - t_all0
        _print_and_save_summary(results, total_seconds, time_csv)
        return

    # ---- MULTI-PROCESS ----
    if workers is None or workers <= 0:
        workers = os.cpu_count() or 1
    workers = min(workers, n)

    click.echo(f"Multi-process mode. Files={n}, method={method}, workers={workers}")
    t_all0 = time.perf_counter()

    results: List[JobResult] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = []
        for pred, out_swc, raw_path in zip(preds, out_swc_paths, raw_paths):
            time.sleep(0.5)
            futures.append(ex.submit(_run_one, pred, out_swc, method, raw_path))

        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Tracing", unit="file"
        ):
            results.append(fut.result())

    total_seconds = time.perf_counter() - t_all0
    _print_and_save_summary(results, total_seconds, time_csv)


if __name__ == "__main__":
    main()
