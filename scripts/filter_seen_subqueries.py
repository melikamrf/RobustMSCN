"""
Flag which subqueries of a target workload only use what a source (training)
workload has, and report q-errors of model results on those subsets.

The vocabulary of the source workload is:
  tables       real table names its queries use
  pred_cols    predicate columns its queries filter on, keyed the way the
               featurizer keys them (alias + column, digits removed:
               mi1.info_type_id -> mi.info_type_id)
  ops          comparison operators its predicates use

Every subquery of each target workload then gets three flags:
  seen_tables  all its tables are source tables
  seen_preds   all its predicates are on a source predicate column with a
               source operator (no predicates counts as seen)
  seen_both    both

The vocabulary is built once per source and cached; the flags are built once
per source and target. Both are rebuilt with --rebuild.

    # JOB-light training data -> CEB and JOB; q-errors for the runs under
    # results/saved_model_eval
    python scripts/filter_seen_subqueries.py

    # any source and targets, and any result folders
    python scripts/filter_seen_subqueries.py --source queries/ceb-imdb \
        --targets queries/job,queries/joblight --runs results/<run>/<exp id>

    # several runs that scored different subqueries: compare on the shared ones
    python scripts/filter_seen_subqueries.py --common --runs <folder> <folder>

Outputs go to results/seen_subqueries/<source>/:
  vocab_<source>.json                        the source vocabulary
  subquery_filters_<target>_vs_<source>.csv  one row per target subquery with its flags
  filtered_per_run_<source>.csv              q-error stats per run, target and filter
  filtered_summary_<source>.csv              for the PG sweep runs: mean / std across seeds
A result folder is one holding cardinality_distributions_<target>.csv files.
"""
import argparse
import ast
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

FILTERS = ["all", "seen_tables", "seen_preds", "seen_both"]
SWEEP_EVAL_ROOT = os.path.join(REPO_ROOT, "results", "saved_model_eval")


def workload_name(qdir):
    return os.path.basename(os.path.normpath(qdir))


def strip_digits(col):
    # the featurizer's column key (feat_separate_alias: 0)
    return "".join(ch for ch in col if not ch.isdigit())


def query_files(qdir, runtime_subset=True):
    fns = sorted(glob.glob(os.path.join(REPO_ROOT, qdir, "*", "*.pkl")))
    if runtime_subset and "imdb" in workload_name(qdir):
        # main.py only evaluates the CEB queries listed here
        with open(os.path.join(REPO_ROOT, "queries", "ceb_runtime_qnames.txt")) as f:
            keep = set(f.read().split("\n"))
        fns = [fn for fn in fns if os.path.basename(fn) in keep]
    if not fns:
        sys.exit(f"No query files under {os.path.join(REPO_ROOT, qdir)}")
    return fns


def load_queries(fns):
    # main.py's loader: same parsing (JOB fix-ups) and the same skipped queries
    from query_representation.utils import load_qdata
    return load_qdata(fns)


def build_vocab(source_dirs, source, out_dir, rebuild, runtime_subset):
    path = os.path.join(out_dir, f"vocab_{source}.json")
    if os.path.exists(path) and not rebuild:
        with open(path, encoding="utf-8-sig") as f:
            saved = json.load(f)
        print(f"Using the saved vocabulary {path}")
        return {k: set(saved[k]) for k in ["tables", "pred_cols", "ops"]}
    tables, pred_cols, ops = set(), set(), set()
    n_queries = 0
    for qdir in source_dirs:
        fns = query_files(qdir, runtime_subset)
        print(f"Reading {len(fns):,} {workload_name(qdir)} queries for the vocabulary "
              f"(once; saved afterwards) ...", flush=True)
        for qrep in load_queries(fns):
            n_queries += 1
            for _, info in qrep["join_graph"].nodes(data=True):
                tables.add(info["real_name"])
                for col, op in zip(info.get("pred_cols", []), info.get("pred_types", [])):
                    pred_cols.add(strip_digits(col))
                    ops.add(op)
    vocab = {"tables": tables, "pred_cols": pred_cols, "ops": ops}
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"source": source_dirs, "queries": n_queries,
                   **{k: sorted(v) for k, v in vocab.items()}}, f, indent=1)
    return vocab


def build_flags(target_dir, vocab, source, out_dir, rebuild, runtime_subset):
    path = os.path.join(out_dir, f"subquery_filters_{workload_name(target_dir)}_vs_{source}.csv")
    if os.path.exists(path) and not rebuild:
        return pd.read_csv(path)
    fns = query_files(target_dir, runtime_subset)
    print(f"Flagging the subqueries of {len(fns):,} {workload_name(target_dir)} queries ...", flush=True)
    rows = []
    for qrep in load_queries(fns):
        table_ok, preds_ok = {}, {}
        for alias, info in qrep["join_graph"].nodes(data=True):
            table_ok[alias] = info["real_name"] in vocab["tables"]
            preds_ok[alias] = all(strip_digits(col) in vocab["pred_cols"] and op in vocab["ops"]
                                  for col, op in zip(info.get("pred_cols", []),
                                                     info.get("pred_types", [])))
        for node in qrep["subset_graph"].nodes():
            node = tuple(node)
            if any(a not in table_ok for a in node):
                continue  # the artificial source node
            rows.append(dict(name=qrep["name"], node=str(node), n_tables=len(node),
                             seen_tables=all(table_ok[a] for a in node),
                             seen_preds=all(preds_ok[a] for a in node)))
    flags = pd.DataFrame(rows)
    flags["seen_both"] = flags["seen_tables"] & flags["seen_preds"]
    flags.to_csv(path, index=False)
    return flags


def read_results(res_dir, target):
    path = os.path.join(res_dir, f"cardinality_distributions_{target}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).dropna(subset=["model_estimated"])
    df["node"] = df["node"].map(lambda s: str(tuple(ast.literal_eval(s))))
    return df


def qerror_stats(q):
    if len(q) == 0:
        return dict(n=0)
    return dict(n=len(q), mean=q.mean(), median=np.median(q), p90=np.percentile(q, 90),
                p99=np.percentile(q, 99), max=q.max())


def score(df, target, flags):
    df = df.merge(flags, on=["name", "node"], how="left")
    missing = df["seen_tables"].isna().sum()
    if missing:
        print(f"  {target}: {missing} result rows not in the flags (other queries?), left out")
        df = df.dropna(subset=["seen_tables"])
    true, est = df["true_cardinality"].astype(float), df["model_estimated"].astype(float)
    q = np.maximum(true / est, est / true)
    out = []
    for flt in FILTERS:
        keep = np.ones(len(df), bool) if flt == "all" else df[flt].astype(bool).values
        out.append(dict(target=target, filter=flt, queries=df.loc[keep, "name"].nunique(),
                        **qerror_stats(q.values[keep])))
    return out


def sweep_result_dirs():
    """results/saved_model_eval/<model>/<exp id>/, the newest exp per model."""
    out = {}
    for model_dir in sorted(glob.glob(os.path.join(SWEEP_EVAL_ROOT, "MSCN*"))):
        d = resolve_result_dir(model_dir)
        if d is not None:
            out[os.path.basename(model_dir)] = d
    return out


def resolve_result_dir(d):
    """d itself if it holds result CSVs, else its newest subfolder that does."""
    has_csvs = lambda x: glob.glob(os.path.join(x, "cardinality_distributions_*.csv"))
    if has_csvs(d):
        return d
    subs = [s for s in glob.glob(os.path.join(d, "*")) if os.path.isdir(s) and has_csvs(s)]
    if subs:
        return max(subs, key=os.path.getmtime)
    print(f"No cardinality_distributions_*.csv in {d} or its subfolders, skipping it")
    return None


def run_label(res_dir):
    # result folders are named by experiment id; for the sweep re-scoring
    # (saved_model_eval/<model>/<exp id>) the model is the parent folder
    parent = os.path.basename(os.path.dirname(os.path.normpath(res_dir)))
    return parent if parent.startswith("MSCN") else os.path.basename(os.path.normpath(res_dir))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="queries/joblight_train",
                   help="training workload(s) the vocabulary comes from, comma-separated "
                        "(default: queries/joblight_train)")
    p.add_argument("--targets", default="queries/ceb-imdb,queries/job",
                   help="workloads to flag, comma-separated (default: queries/ceb-imdb,queries/job)")
    p.add_argument("--runs", nargs="*", default=None,
                   help="result folders to score (default: every run under results/saved_model_eval)")
    p.add_argument("--common", action="store_true",
                   help="score only the subqueries every run has (runs differ when some "
                        "deduplicated their eval sets and others did not)")
    p.add_argument("--all_ceb_queries", action="store_true",
                   help="use every CEB query, not only the runtime subset main.py evaluates on")
    p.add_argument("--out_dir", default=None,
                   help="default: results/seen_subqueries/<source name>")
    p.add_argument("--rebuild", action="store_true", help="rebuild the vocabulary and the flags")
    opts = p.parse_args()

    source_dirs = [s for s in opts.source.split(",") if s]
    target_dirs = [t for t in opts.targets.split(",") if t]
    runtime_subset = not opts.all_ceb_queries
    source = "+".join(workload_name(s) for s in source_dirs)
    out_dir = opts.out_dir or os.path.join(REPO_ROOT, "results", "seen_subqueries", source)
    os.makedirs(out_dir, exist_ok=True)

    vocab = build_vocab(source_dirs, source, out_dir, opts.rebuild, runtime_subset)
    print(f"Source {source}:")
    print(f"  tables ({len(vocab['tables'])}): {', '.join(sorted(vocab['tables']))}")
    print(f"  predicate columns ({len(vocab['pred_cols'])}): {', '.join(sorted(vocab['pred_cols']))}")
    print(f"  operators: {', '.join(sorted(vocab['ops']))}")

    flags = {}
    for t in target_dirs:
        fl = build_flags(t, vocab, source, out_dir, opts.rebuild, runtime_subset)
        flags[workload_name(t)] = fl
        n = len(fl)
        print(f"{workload_name(t)}: {fl['name'].nunique():,} queries, {n:,} subqueries; "
              + ", ".join(f"{f.replace('_', ' ')} {fl[f].sum():,} ({fl[f].mean():.1%})"
                          for f in FILTERS[1:]))

    if opts.runs is None:
        dirs = sweep_result_dirs()
    else:
        resolved = [resolve_result_dir(d) for d in opts.runs]
        dirs = {run_label(d): d for d in resolved if d is not None}
    if not dirs:
        print("\nNo result folders given or found; wrote only the flags to " + out_dir)
        return

    try:
        from eval_saved_pg_sweep import RUNS
        meta = {m: dict(pg=pg, train_subqueries=n, seed=seed) for m, pg, n, _, seed, _ in RUNS}
    except ImportError:
        meta = {}

    rows = []
    for target, fl in flags.items():
        results = {run: read_results(d, target) for run, d in dirs.items()}
        results = {run: df for run, df in results.items() if df is not None}
        if not results:
            continue
        if opts.common:
            common = set.intersection(*(set(zip(df["name"], df["node"])) for df in results.values()))
            print(f"{target}: {len(common):,} subqueries scored by all {len(results)} runs")
            results = {run: df[[k in common for k in zip(df["name"], df["node"])]]
                       for run, df in results.items()}
        for run, df in results.items():
            for r in score(df, target, fl):
                rows.append(dict(run=run, **meta.get(run, {}), **r))
    if not rows:
        print("\nNone of the result folders has results for "
              + ", ".join(flags) + "; wrote only the flags to " + out_dir)
        return

    per_run = pd.DataFrame(rows)
    per_run.insert(0, "source", source)
    per_run.to_csv(os.path.join(out_dir, f"filtered_per_run_{source}.csv"), index=False)
    pd.set_option("display.width", 250)
    fmt = lambda v: f"{v:,.2f}"
    stats = ["mean", "median", "p90", "p99", "max"]
    for target in flags:
        t = per_run[per_run["target"] == target]
        if len(t):
            print(f"\n== {target} ==")
            print(t[["run", "filter", "queries", "n"] + stats].to_string(index=False, float_format=fmt))

    # PG sweep runs: runs of the same seed averaged first, then mean / std across seeds
    if "pg" in per_run and per_run["pg"].notna().any():
        sw = per_run.dropna(subset=["pg"]).copy()
        sw["seed_key"] = sw["seed"].astype(str) + "|" + sw["run"].where(sw["seed"].isna(), "")
        by_seed = sw.groupby(["target", "filter", "pg", "train_subqueries", "seed_key"],
                             as_index=False)[stats + ["n"]].mean()
        summary = by_seed.groupby(["target", "filter", "pg", "train_subqueries"]).agg(
            seeds=("seed_key", "nunique"), n=("n", "max"),
            **{f"{s} mean": (s, "mean") for s in stats},
            **{f"{s} std": (s, "std") for s in stats}).reset_index()
        summary.insert(0, "source", source)
        summary.to_csv(os.path.join(out_dir, f"filtered_summary_{source}.csv"), index=False)
        print("\n== PG sweep runs, mean across seeds ==")
        print(summary[["target", "filter", "pg", "train_subqueries", "seeds", "n", "median mean",
                       "median std", "p99 mean", "p99 std", "max mean"]].to_string(index=False, float_format=fmt))
    print("\nOutputs in " + out_dir)


if __name__ == "__main__":
    main()
