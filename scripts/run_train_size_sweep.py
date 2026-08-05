"""
Trains the model at several training-set sizes and summarizes the effect on
accuracy, for either or both of the two train/val split options:

  original : the get_query_splits path (train/val/test derived from
             data.query_dir + data.val_size/test_size).
  csv      : the --train_csv/--eval_csv path (subquery-level CSVs, split
             into train/val/test by load_train_pool_from_csv).

Only the TRAINING split varies -- val/test/eval are identical across every
run of a mode, so the q-errors are comparable size-to-size (see
main.subsample_train_split).

Each run is a fresh `python main.py` process writing into its own
--result_dir, so nothing leaks between sizes. Afterwards this script reads
each run's QError.csv + train_split_sizes.json and writes one summary CSV.

Examples
--------
Both modes, default size grid:

  python scripts/run_train_size_sweep.py \
      --orig_config configs/config-joblight-robust.yaml \
      --csv_config configs/config-grasp-csv-robust.yaml \
      --train_csv path/to/train_subqueries.csv \
      --eval_csv path/to/eval_subqueries.csv

Only the original split, custom sizes, subquery-level counting:

  python scripts/run_train_size_sweep.py --modes original \
      --orig_config configs/config-joblight-robust.yaml \
      --sizes 0.1,0.5,1.0 --train_size_level subquery

Re-summarize finished runs without retraining:

  python scripts/run_train_size_sweep.py --summarize_only ...

Anything after a bare `--` is forwarded verbatim to every main.py call,
e.g. `-- --remove_duplicate_subqueries 1 --learn_residual 0`.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SIZES = "0.1,0.25,0.5,0.75,1.0"


def parse_sizes(sizes_str):
    sizes = []
    for tok in sizes_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        val = float(tok)
        if val <= 0:
            raise ValueError(f"--sizes entries must be > 0, got {tok}")
        sizes.append(val)
    if not sizes:
        raise ValueError("--sizes did not contain any values")
    return sizes


def size_tag(size):
    """Filesystem-safe, sortable directory name for a training size."""
    if size <= 1.0:
        return "frac_" + ("%g" % size).replace(".", "p")
    return "n_%d" % int(round(size))


def build_cmd(args, mode, size, run_dir, passthrough):
    config = args.orig_config if mode == "original" else args.csv_config
    cmd = [
        sys.executable, "main.py",
        "--config", config,
        "--alg", args.alg,
        "--eval_fns", args.eval_fns,
        "--result_dir", run_dir,
        "--train_size", str(size),
        "--train_size_level", args.train_size_level,
        "--train_size_seed", str(args.train_size_seed),
        "--random_seed", str(args.random_seed),
    ]
    if mode == "csv":
        cmd += ["--train_csv", args.train_csv, "--eval_csv", args.eval_csv,
                "--use_csv_split_cache", str(args.use_csv_split_cache)]
    cmd += passthrough
    return cmd


def run_one(cmd, run_dir):
    """
    Runs main.py, streaming its output to this console and to
    <run_dir>/run.log. Returns the process exit code.
    """
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
    return proc.returncode


def summarize_run(run_dir, mode, size):
    """
    Turns one finished run into summary rows -- one per samples_type
    (train/test/<eval dir name>) found in that run's QError.csv.
    """
    qerr_path = os.path.join(run_dir, "QError.csv")
    if not os.path.exists(qerr_path):
        print(f"  no QError.csv in {run_dir}, skipping in summary")
        return []

    sizes_path = os.path.join(run_dir, "train_split_sizes.json")
    split_sizes = {}
    if os.path.exists(sizes_path):
        with open(sizes_path) as f:
            split_sizes = json.load(f)

    df = pd.read_csv(qerr_path)
    rows = []
    for samples_type, group in df.groupby("samples_type"):
        errors = group["errors"].astype(float).values
        errors = errors[np.isfinite(errors)]
        if len(errors) == 0:
            continue
        rows.append({
            "mode": mode,
            "train_size": size,
            "train_size_level": split_sizes.get("train_size_level"),
            "num_train_queries": split_sizes.get("num_train_queries"),
            "num_train_subqueries": split_sizes.get("num_train_subqueries"),
            "num_val_subqueries": split_sizes.get("num_val_subqueries"),
            "num_test_subqueries": split_sizes.get("num_test_subqueries"),
            "num_eval_subqueries": split_sizes.get("num_eval_subqueries"),
            "samples_type": samples_type,
            "num_samples": len(errors),
            "qerror_mean": np.mean(errors),
            "qerror_median": np.median(errors),
            "qerror_90p": np.percentile(errors, 90),
            "qerror_99p": np.percentile(errors, 99),
            "qerror_max": np.max(errors),
            "result_dir": run_dir,
        })
    return rows


def print_summary(df):
    if df.empty:
        print("No results to summarize.")
        return
    cols = ["mode", "train_size", "num_train_subqueries", "samples_type",
            "num_samples", "qerror_mean", "qerror_median", "qerror_90p",
            "qerror_99p"]
    view = df[cols].copy()
    for c in ["qerror_mean", "qerror_median", "qerror_90p", "qerror_99p"]:
        view[c] = view[c].round(3)
    print()
    print("=" * 100)
    print("Training-size sweep summary")
    print("=" * 100)
    print(view.to_string(index=False))


def main():
    args, passthrough = read_flags()
    # argparse may or may not swallow the separator depending on where it
    # lands; drop any stray ones so they don't reach main.py.
    passthrough = [tok for tok in passthrough if tok != "--"]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in ("original", "csv"):
            raise ValueError(f"unknown mode {mode!r} (expected original/csv)")
        if mode == "csv" and not (args.train_csv and args.eval_csv):
            raise ValueError("mode 'csv' needs both --train_csv and --eval_csv")
        config = args.orig_config if mode == "original" else args.csv_config
        if not os.path.exists(os.path.join(REPO_ROOT, config)):
            raise ValueError(f"config for mode {mode!r} not found: {config}")

    sizes = parse_sizes(args.sizes)
    os.makedirs(args.result_root, exist_ok=True)

    all_rows = []
    failures = []
    for mode in modes:
        for size in sizes:
            run_dir = os.path.join(args.result_root, mode, size_tag(size))
            cmd = build_cmd(args, mode, size, run_dir, passthrough)

            print()
            print("-" * 100)
            print(f"[{mode}] train_size={size} -> {run_dir}")
            print(" ".join(cmd))
            print("-" * 100)

            if args.dry_run:
                continue

            already_done = os.path.exists(os.path.join(run_dir, "QError.csv"))
            if args.summarize_only:
                pass
            elif args.skip_existing and already_done:
                print("  QError.csv already present, skipping (--skip_existing)")
            else:
                rc = run_one(cmd, run_dir)
                if rc != 0:
                    print(f"  RUN FAILED (exit {rc}); see {os.path.join(run_dir, 'run.log')}")
                    failures.append((mode, size, rc))
                    continue

            all_rows += summarize_run(run_dir, mode, size)

    if args.dry_run:
        return

    summary = pd.DataFrame(all_rows)
    if not summary.empty:
        summary = summary.sort_values(["mode", "samples_type", "train_size"])
        out_path = os.path.join(args.result_root, "train_size_sweep_summary.csv")
        summary.to_csv(out_path, index=False)
        print_summary(summary)
        print(f"\nSaved summary to {out_path}")

    if failures:
        print("\nFailed runs:")
        for mode, size, rc in failures:
            print(f"  {mode} train_size={size}: exit {rc}")
        sys.exit(1)


def read_flags():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--modes", type=str, default="original,csv",
            help="Comma separated subset of original,csv (default: both)")
    parser.add_argument("--sizes", type=str, default=DEFAULT_SIZES,
            help="Comma separated training sizes. Values in (0,1] are "
                 "fractions of the training pool, values > 1 are absolute "
                 f"sample counts. Default: {DEFAULT_SIZES}")

    parser.add_argument("--orig_config", type=str,
            default="configs/config-joblight-robust.yaml",
            help="Config used for the get_query_splits (original) mode")
    parser.add_argument("--csv_config", type=str,
            default="configs/config-grasp-csv-robust.yaml",
            help="Config used for the --train_csv/--eval_csv mode")
    parser.add_argument("--train_csv", type=str, default=None,
            help="Subquery-level train-pool CSV (required for mode csv)")
    parser.add_argument("--eval_csv", type=str, default=None,
            help="Subquery-level held-out eval CSV (required for mode csv)")
    parser.add_argument("--use_csv_split_cache", type=int, default=1,
            help="Passed through to main.py; the cache makes repeated runs "
                 "over the same CSVs much faster, so keep it at 1")

    parser.add_argument("--alg", type=str, default="mscn")
    parser.add_argument("--eval_fns", type=str, default="qerr",
            help="Passed to main.py. Default qerr only: plan-cost eval "
                 "needs a live postgres and dominates sweep runtime.")
    parser.add_argument("--train_size_level", type=str, default="auto",
            choices=["auto", "query", "subquery"],
            help="Passed to main.py --train_size_level. Use 'subquery' when "
                 "comparing the two modes against each other.")
    parser.add_argument("--train_size_seed", type=int, default=42)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--result_root", type=str,
            default="./results/train_size_sweep",
            help="Per-run result dirs go under <result_root>/<mode>/<size>/")
    parser.add_argument("--skip_existing", type=int, default=0,
            help="1 to skip runs whose result dir already has a QError.csv "
                 "(resume an interrupted sweep)")
    parser.add_argument("--summarize_only", type=int, default=0,
            help="1 to skip training entirely and just rebuild the summary "
                 "CSV from result dirs that already exist")
    parser.add_argument("--dry_run", type=int, default=0,
            help="1 to print the main.py commands without running them")

    return parser.parse_known_args()


if __name__ == "__main__":
    main()
