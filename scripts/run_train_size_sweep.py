""" TODO: Fix the summary,  Error: in ./results/train_size_sweep/original/frac_1/seed1, skipping in summary
it should look in the model subdir and for cardinality_distributions_joblight.csv"""

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

Each size is run once per --train_size_seeds value (default 3 seeds), so
every point on the size curve gets an error bar: the spread tells you how
much of a size-to-size difference is a real size effect and how much is
just which subsample you happened to draw. --train_seed is held FIXED
across all of them, so the spread is attributable to the subsample rather
than to the initialization.

train_size=1.0 is run only once, at the first seed: it keeps the whole
training pool, so the subsample seed is never used and repeats would be
byte-identical runs.

Each run is a fresh `python main.py` process writing into its own
--result_dir (<result_root>/<mode>/<size>/seed<n>), so nothing leaks
between runs. Afterwards this script reads each run's QError.csv +
train_split_sizes.json and writes two CSVs: one row per run, and one
aggregated across seeds per (mode, size, samples_type).

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

With duplicate subqueries and cross-split leakage removed first (both are
applied before the subsample, so they shrink the pool a fractional
--train_size is measured against):

  python scripts/run_train_size_sweep.py \
      --remove_duplicate_subqueries 1 --remove_leakage 1

To re-summarize finished runs without retraining, the arguments should be passed exactly as they were for the original sweep, 
except for --summarize_only 1. For example:

  python scripts/run_train_size_sweep.py --summarize_only 1 --modes original \
          --remove_duplicate_subqueries 1 --remove_leakage 1

Anything after a bare `--` is forwarded verbatim to every main.py call,
e.g. `-- --learn_residual 1`. Flags the sweep sets itself (see
MANAGED_FLAGS) are rejected there -- use the matching sweep option.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_SIZES = "0.01,0.1,0.5,1.0"
DEFAULT_SEEDS = "1,2,3"

# main.py flags this script sets itself. Passing one after `--` would
# silently fight with the sweep's own value (argparse keeps the last
# occurrence), so they're rejected there in favour of the sweep option.
MANAGED_FLAGS = {
    "--config", "--alg", "--eval_fns", "--result_dir",
    "--train_size", "--train_size_level", "--train_size_seed",
    "--random_seed", "--train_seed",
    "--train_csv", "--eval_csv",
    "--remove_duplicate_subqueries", "--detect_leakage", "--remove_leakage",
}


def flag_given(flag):
    """
    Was @flag actually typed on the command line? Only looks before a bare
    `--`, so args forwarded to main.py don't count as sweep flags.
    """
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[:argv.index("--")]
    return any(a == flag or a.startswith(flag + "=") for a in argv)


# Flags that used to exist and no longer do. parse_known_args() would sweep
# an old command line's copy into `passthrough` and forward it to main.py,
# which dies with a bare "unrecognized arguments" -- explain it here instead.
REMOVED_FLAGS = {
    "--use_csv_split_cache":
        "the CSV split cache was removed from main.py; the split is now "
        "always computed fresh, so there is nothing to toggle. Drop this "
        "flag from your command line.",
}


def check_passthrough(passthrough):
    gone = sorted({tok for tok in passthrough if tok in REMOVED_FLAGS})
    if gone:
        raise ValueError("; ".join(
            [f"{flag}: {REMOVED_FLAGS[flag]}" for flag in gone]))

    clashes = sorted({tok for tok in passthrough if tok in MANAGED_FLAGS})
    if clashes:
        raise ValueError(
            "these main.py flags are set by the sweep itself; use the "
            "matching sweep option instead of passing them after `--`: "
            + ", ".join(clashes))


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


def parse_seeds(seeds_str):
    seeds = []
    for tok in seeds_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        seeds.append(int(tok))
    if not seeds:
        raise ValueError("--train_size_seeds did not contain any values")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--train_size_seeds has duplicates: {seeds}")
    return seeds


def seeds_for_size(size, seeds):
    """
    Which subsample seeds to run at this training size.

    At train_size == 1.0 the whole pool is kept -- main._resolve_train_size
    returns None and never touches the RNG -- so repeats would be identical
    runs. One seed is enough there.

    (An absolute --sizes entry that happens to exceed the pool is the same
    no-op, but the pool size is not known until main.py has loaded the
    queries, so those still get the full set of repeats.)
    """
    return seeds[:1] if size == 1.0 else seeds


def size_tag(size):
    """Filesystem-safe, sortable directory name for a training size."""
    if size <= 1.0:
        return "frac_" + ("%g" % size).replace(".", "p")
    return "n_%d" % int(round(size))


def cleaning_tag(args):
    """
    Run-dir suffix for the dedup/leakage cleaning in effect, so two sweeps
    that differ only in cleaning can share a --result_root instead of one
    silently overwriting the other. Empty when no cleaning is on, which
    keeps the plain <mode>/<size> layout for the default sweep.

    --detect_leakage is deliberately not part of this: it only logs the
    overlap, it doesn't change the data, so runs with and without it are
    interchangeable.
    """
    parts = []
    if args.remove_duplicate_subqueries:
        parts.append("dedup")
    if args.remove_leakage:
        parts.append("noleak")
    return ("_" + "_".join(parts)) if parts else ""


def build_cmd(args, mode, size, train_size_seed, run_dir, passthrough):
    config = args.orig_config if mode == "original" else args.csv_config
    cmd = [
        sys.executable, "main.py",
        "--config", config,
        "--alg", args.alg,
        "--eval_fns", args.eval_fns,
        "--result_dir", run_dir,
        "--train_size", str(size),
        "--train_size_level", args.train_size_level,
        # The one knob that varies across repeats: which subsample of the
        # training pool this run draws.
        "--train_size_seed", str(train_size_seed),
        "--random_seed", str(args.random_seed),
        # Pass explicitly rather than letting each run fall back to
        # model.train_seed: the two modes use DIFFERENT config files, so an
        # implicit default would let 'original' and 'csv' train from different
        # initializations and quietly confound the mode comparison.
        "--train_seed", str(args.train_seed),
        # Applied before the subsample, so these decide what pool a given
        # --train_size is a fraction OF. Held constant across the sweep.
        "--remove_duplicate_subqueries", str(args.remove_duplicate_subqueries),
        "--detect_leakage", str(args.detect_leakage),
        "--remove_leakage", str(args.remove_leakage),
    ]
    if mode == "csv":
        cmd += ["--train_csv", args.train_csv, "--eval_csv", args.eval_csv]
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


def summarize_run(run_dir, mode, size, train_size_seed):
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
            # The repeat index for this (mode, size): the seed the subsample
            # was drawn with. Taken from this script rather than the run's
            # JSON so the aggregation below still groups correctly if an old
            # run predates save_split_sizes() recording it.
            "train_size_seed": train_size_seed,
            "train_size_level": split_sizes.get("train_size_level"),
            # Recorded by main.save_split_sizes(); carried into the summary so
            # a row can always be traced back to the seeds that produced it.
            "train_seed": split_sizes.get("train_seed"),
            "random_seed": split_sizes.get("random_seed"),
            "data_seed": split_sizes.get("data_seed"),
            # Read back from the run rather than echoed from this script's
            # flags, so --summarize_only over a directory of older runs
            # reports the cleaning each of them actually used.
            "remove_duplicate_subqueries": split_sizes.get("remove_duplicate_subqueries"),
            "detect_leakage": split_sizes.get("detect_leakage"),
            "remove_leakage": split_sizes.get("remove_leakage"),
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


def aggregate_over_seeds(df):
    """
    Collapses the per-run rows into one row per (mode, cleaning, size,
    samples_type), with mean/std across the --train_size_seeds repeats.

    std is the population-vs-sample question that matters here: with 3 seeds
    ddof=1 (pandas' default) is the right estimator of the underlying spread,
    and it is NaN at train_size=1.0 where there is only one run. That NaN is
    meaningful -- it says "no repeats here" -- so it is left in rather than
    filled with 0, which would read as "no variance".
    """
    if df.empty:
        return df

    keys = ["mode", "remove_duplicate_subqueries", "remove_leakage",
            "train_size", "samples_type"]
    agg = df.groupby(keys, dropna=False).agg(
        n_seeds=("train_size_seed", "nunique"),
        num_train_subqueries=("num_train_subqueries", "mean"),
        qerror_mean_avg=("qerror_mean", "mean"),
        qerror_mean_std=("qerror_mean", "std"),
        qerror_mean_min=("qerror_mean", "min"),
        qerror_mean_max=("qerror_mean", "max"),
        qerror_median_avg=("qerror_median", "mean"),
        qerror_median_std=("qerror_median", "std"),
        qerror_90p_avg=("qerror_90p", "mean"),
        qerror_90p_std=("qerror_90p", "std"),
        qerror_99p_avg=("qerror_99p", "mean"),
        qerror_99p_std=("qerror_99p", "std"),
    ).reset_index()
    return agg.sort_values(["mode", "samples_type", "train_size"])


def print_summary(df):
    if df.empty:
        print("No results to summarize.")
        return
    cols = ["mode", "train_size", "train_size_seed",
            "remove_duplicate_subqueries", "remove_leakage",
            "num_train_subqueries", "samples_type",
            "num_samples", "qerror_mean", "qerror_median", "qerror_90p",
            "qerror_99p"]
    view = df[cols].copy()
    for c in ["qerror_mean", "qerror_median", "qerror_90p", "qerror_99p"]:
        view[c] = view[c].round(3)
    # short headers; the full names live in the CSV
    view = view.rename(columns={"remove_duplicate_subqueries": "dedup",
                                "remove_leakage": "noleak",
                                "train_size_seed": "seed"})
    print()
    print("=" * 100)
    print("Training-size sweep -- individual runs")
    print("=" * 100)
    print(view.to_string(index=False))


def print_aggregate(df):
    if df.empty:
        return
    cols = ["mode", "train_size", "n_seeds", "remove_duplicate_subqueries",
            "remove_leakage", "num_train_subqueries", "samples_type",
            "qerror_mean_avg", "qerror_mean_std",
            "qerror_median_avg", "qerror_median_std",
            "qerror_90p_avg", "qerror_90p_std"]
    view = df[cols].copy()
    for c in cols:
        if c.startswith("qerror") or c == "num_train_subqueries":
            view[c] = view[c].astype(float).round(3)
    view = view.rename(columns={"remove_duplicate_subqueries": "dedup",
                                "remove_leakage": "noleak"})
    print()
    print("=" * 100)
    print("Training-size sweep -- mean +/- std over train_size_seeds")
    print("=" * 100)
    print(view.to_string(index=False))
    print()
    print("std is NaN where only one seed was run (train_size=1.0 keeps the "
          "whole pool, so repeats would be identical).")


def main():
    args, passthrough = read_flags()
    # argparse may or may not swallow the separator depending on where it
    # lands; drop any stray ones so they don't reach main.py.
    passthrough = [tok for tok in passthrough if tok != "--"]
    check_passthrough(passthrough)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in ("original", "csv"):
            raise ValueError(f"unknown mode {mode!r} (expected original/csv)")
        if mode == "csv" and not (args.train_csv and args.eval_csv):
            raise ValueError("mode 'csv' needs both --train_csv and --eval_csv")
        config = args.orig_config if mode == "original" else args.csv_config
        if not os.path.exists(os.path.join(REPO_ROOT, config)):
            raise ValueError(f"config for mode {mode!r} not found: {config}")

    # --modes decides WHICH config flag is read (orig_config vs csv_config),
    # so a config passed for a mode that is not selected is silently ignored,
    # and the selected mode quietly falls back to its own default config --
    # swapping query_dir/max_epochs out from under the experiment without a
    # word. Say what each mode resolved to, and refuse the ignored-config case.
    for mode in modes:
        chosen = args.orig_config if mode == "original" else args.csv_config
        print(f"[{mode}] config: {chosen}")

    for unused_mode, flag, dest in (("csv", "--csv_config", "csv_config"),
                                    ("original", "--orig_config", "orig_config")):
        if unused_mode in modes:
            continue
        # Scan argv rather than comparing against the parser default: the
        # whole point is to catch someone passing a config for the mode they
        # didn't select, and that config is very often exactly the default
        # value, which a default-comparison can't see.
        if not flag_given(flag):
            continue
        other = "--orig_config" if unused_mode == "csv" else "--csv_config"
        raise ValueError(
            f"{flag}={getattr(args, dest)!r} was given, but mode "
            f"{unused_mode!r} is not in --modes={args.modes!r}. That config "
            f"would be silently ignored and the selected mode(s) would run "
            f"with a different config (different query_dir / max_epochs / "
            f"eval set). Either add {unused_mode!r} to --modes, or pass the "
            f"config you actually want as {other}.")

    sizes = parse_sizes(args.sizes)
    seeds = parse_seeds(args.train_size_seeds)
    os.makedirs(args.result_root, exist_ok=True)

    all_rows = []
    failures = []
    for mode in modes:
        for size in sizes:
            cur_seeds = seeds_for_size(size, seeds)
            if len(cur_seeds) < len(seeds):
                print(f"\n[{mode}] train_size={size} keeps the whole training "
                      f"pool, so --train_size_seed is never used: running once "
                      f"at seed {cur_seeds[0]} instead of {len(seeds)} "
                      f"identical repeats")

            for seed in cur_seeds:
                run_dir = os.path.join(args.result_root, mode,
                        size_tag(size) + cleaning_tag(args), f"seed{seed}")
                cmd = build_cmd(args, mode, size, seed, run_dir, passthrough)

                print()
                print("-" * 100)
                print(f"[{mode}] train_size={size} train_size_seed={seed}"
                      f" -> {run_dir}")
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
                        failures.append((mode, size, seed, rc))
                        continue

                all_rows += summarize_run(run_dir, mode, size, seed)

    if args.dry_run:
        return

    summary = pd.DataFrame(all_rows)
    if not summary.empty:
        summary = summary.sort_values(
                ["mode", "samples_type", "train_size", "train_size_seed"])
        # Tagged like the run dirs, so a cleaned sweep doesn't overwrite the
        # uncleaned one's summary. The two CSVs share a schema (both carry
        # remove_duplicate_subqueries/remove_leakage), so they concatenate.
        tag = cleaning_tag(args)
        out_path = os.path.join(args.result_root,
                f"train_size_sweep_summary{tag}.csv")
        summary.to_csv(out_path, index=False)
        print_summary(summary)
        print(f"\nSaved per-run summary to {out_path}")

        agg = aggregate_over_seeds(summary)
        agg_path = os.path.join(args.result_root,
                f"train_size_sweep_by_seed{tag}.csv")
        agg.to_csv(agg_path, index=False)
        print_aggregate(agg)
        print(f"\nSaved seed-aggregated summary to {agg_path}")

    if failures:
        print("\nFailed runs:")
        for mode, size, seed, rc in failures:
            print(f"  {mode} train_size={size} train_size_seed={seed}: exit {rc}")
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
    parser.add_argument("--alg", type=str, default="mscn")
    parser.add_argument("--eval_fns", type=str, default="qerr",
            help="Passed to main.py. Default qerr only: plan-cost eval "
                 "needs a live postgres and dominates sweep runtime.")
    parser.add_argument("--train_size_level", type=str, default="auto",
            choices=["auto", "query", "subquery"],
            help="Passed to main.py --train_size_level. Use 'subquery' when "
                 "comparing the two modes against each other.")
    parser.add_argument("--train_size_seeds", type=str, default=DEFAULT_SEEDS,
            help="Comma separated seeds for main.py --train_size_seed. Each "
                 "size is run once per seed, giving repeats that differ only "
                 "in WHICH subsample of the training pool was drawn -- so the "
                 "spread at a size is the subsample effect, not training "
                 "noise. train_size=1.0 keeps the whole pool and is run once "
                 f"regardless. Default: {DEFAULT_SEEDS}. Pass a single value "
                 "for the old one-run-per-size behaviour.")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--train_seed", type=int, default=42,
            help="Passed to main.py --train_seed: weight init, batch order, "
                 "dropout, feature masking. Held FIXED across every size and "
                 "seed in the sweep, so the repeats isolate the subsample "
                 "effect. Re-run the whole sweep at 2-3 values of this to "
                 "measure training noise instead.")

    # Subquery cleaning. Both run BEFORE the training-size subsample, so
    # they change the pool a fractional --train_size is measured against
    # (and shrink it); the summary reports num_train_subqueries so the
    # actual size behind each row is never in doubt. Held constant across
    # every run of a sweep -- to compare cleaned vs uncleaned, run the
    # sweep twice; the run dirs are suffixed so they won't collide.
    parser.add_argument("--remove_duplicate_subqueries", type=int, default=0,
            help="Passed to main.py: 1 drops repeated subqueries (same "
                 "tables/joins/predicates) within each of train/val/test, "
                 "keeping the first occurrence. Without it a subquery that "
                 "appears in many query files is effectively upweighted, "
                 "and the same rows can be counted in both train and eval.")
    parser.add_argument("--remove_leakage", type=int, default=0,
            help="Passed to main.py: 1 drops subqueries that appear in more "
                 "than one split, keeping each in the highest-priority one "
                 "(train > val > test). Use this when the test/val q-errors "
                 "look implausibly good at small training sizes.")
    parser.add_argument("--detect_leakage", type=int, default=0,
            help="Passed to main.py: 1 logs the train/val/test subquery "
                 "overlap per run without changing the data. Cheap way to "
                 "see how much overlap a mode has before deciding whether "
                 "--remove_leakage is worth the rerun.")

    parser.add_argument("--result_root", type=str,
            default="./results/train_size_sweep",
            help="Per-run result dirs go under <result_root>/<mode>/<size>/, "
                 "with a _dedup/_noleak suffix when those options are on")
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
