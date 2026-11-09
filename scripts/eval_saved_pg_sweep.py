"""
Score the saved models of the PG=0 / PG=1 train-size sweep (trained on
joblight_train, 100 epochs, dedup, leakage removed) on other workloads,
by default CEB-IMDb and JOB.

Each run is re-created with main.py --load_model: same config (the run's own
cfg.json), same train subsample, dedup and leakage removal, so the featurizer
is the one the model was trained with. The new workloads come in through
--extra_eval_query_dir, which loads them only after the featurizer is built.
Anything in them the model has no input slot for (tables, joins, columns,
operators that JOB-light never had) is left out of their features.

As a check, every run also re-scores joblight and test; --collect compares
those estimates with the ones the original run saved.

    # 1. list the runs, where their weights and configs are, and the commands
    python scripts/eval_saved_pg_sweep.py

    # 2. score them (all of them, one model, or one per Slurm array task)
    python scripts/eval_saved_pg_sweep.py --run
    python scripts/eval_saved_pg_sweep.py --run --only MSCN2109954879
    python scripts/eval_saved_pg_sweep.py --run --index $SLURM_ARRAY_TASK_ID

    # 3. summarize: per run, and mean / std across seeds per PG and size
    python scripts/eval_saved_pg_sweep.py --collect

Outputs go to results/saved_model_eval/<model>/ (main.py adds one more
folder, named by its new experiment id, with the cardinality_distributions
CSVs in it).
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join("results", "saved_model_eval")
# the config every sweep run was started with; only used when a run's own
# cfg.json cannot be found
SWEEP_CONFIG = "configs/config-robust-original.yaml"

# (model, PG = featurizer.heuristic_features, train subqueries, train_size,
#  train_size_seed, the run's result folder relative to --runs_root)
# Left out: MSCN792203518 (PG=0) and MSCN2988871891 (PG=1), the full-size runs
# that hit the time limit after epoch 100 -- they never saved weights, only
# epoch-100 predictions.
RUNS = [
    ("MSCN2109954879", 0, 889, 0.01, 1, "results/train_size_sweep/original/frac_0p01_dedup_noleak/seed1/MSCN2109954879"),
    ("MSCN319512952", 0, 889, 0.01, 2, "results/train_size_sweep/original/frac_0p01_dedup_noleak/seed2/MSCN319512952"),
    ("MSCN3293835870", 0, 889, 0.01, 3, "results/train_size_sweep/original/frac_0p01_dedup_noleak/seed3/MSCN3293835870"),
    ("MSCN4210010208", 1, 889, 0.01, 1, "results/train_size_sweep/original/frac_0p01_dedup_noleak/seed1/MSCN4210010208"),
    ("MSCN415384146", 1, 889, 0.01, 2, "results/train_size_sweep/original/frac_0p01_dedup_noleak/seed2/MSCN415384146"),
    ("MSCN3993024008", 1, 889, 0.01, 3, "results/train_size_sweep/original/frac_0p01_dedup_noleak/seed3/MSCN3993024008"),
    ("MSCN1240401878", 0, 8893, 0.1, 1, "results/train_size_sweep/original/frac_0p1_dedup_noleak/seed1/MSCN1240401878"),
    ("MSCN1751951973", 0, 8893, 0.1, 1, "results/train_size_sweep/original/frac_0p1_dedup_noleak/seed1/MSCN1751951973"),
    ("MSCN3608827853", 0, 8893, 0.1, 2, "results/train_size_sweep/original/frac_0p1_dedup_noleak/seed2/MSCN3608827853"),
    ("MSCN3119513022", 0, 8893, 0.1, 3, "results/train_size_sweep/original/frac_0p1_dedup_noleak/seed3/MSCN3119513022"),
    ("MSCN3179920740", 1, 8893, 0.1, 1, "results/train_size_sweep/original/frac_0p1_dedup_noleak/seed1/MSCN3179920740"),
    ("MSCN2804571446", 1, 8893, 0.1, 2, "results/train_size_sweep/original/frac_0p1_dedup_noleak/seed2/MSCN2804571446"),
    ("MSCN1484622032", 1, 8893, 0.1, 3, "results/train_size_sweep/original/frac_0p1_dedup_noleak/seed3/MSCN1484622032"),
    ("MSCN1840276947", 0, 44463, 0.5, 1, "results/train_size_sweep/original/frac_0p5_dedup_noleak/seed1/MSCN1840276947"),
    ("MSCN264975786", 0, 44463, 0.5, 2, "results/train_size_sweep/original/frac_0p5_dedup_noleak/seed2/MSCN264975786"),
    ("MSCN4278625527", 0, 44463, 0.5, 3, "results/train_size_sweep/original/frac_0p5_dedup_noleak/seed3/MSCN4278625527"),
    ("MSCN3343127029", 1, 44463, 0.5, 1, "results/train_size_sweep/original/frac_0p5_dedup_noleak/seed1/MSCN3343127029"),
    ("MSCN3458090404", 1, 44463, 0.5, 2, "results/train_size_sweep/original/frac_0p5_dedup_noleak/seed2/MSCN3458090404"),
    ("MSCN939926319", 1, 44463, 0.5, 3, "results/train_size_sweep/original/frac_0p5_dedup_noleak/seed3/MSCN939926319"),
    # slurm 1424, before the sweep script: whole train pool, no --train_size
    ("MSCN1587650273", 0, 88926, None, None, "results/MSCN1587650273"),
    ("MSCN2515561509", 1, 88926, 1.0, 1, "results/train_size_sweep/original/frac_1_dedup_noleak/seed1/MSCN2515561509"),
]

SPLITS = ["joblight", "test", "ceb-imdb", "job"]
# the original run saved these too: re-scoring them must give the same numbers
CHECK_SPLITS = ["joblight", "test"]


def train_args(train_size, train_size_seed):
    """main.py arguments that decide the run's training data, as the sweep set them."""
    args = ["--alg", "mscn"]
    if train_size is not None:
        args += ["--train_size", str(train_size), "--train_size_level", "subquery",
                 "--train_size_seed", str(train_size_seed),
                 "--random_seed", "42", "--train_seed", "42"]
    args += ["--remove_duplicate_subqueries", "1", "--detect_leakage", "0",
             "--remove_leakage", "1"]
    return args


def find_weights(model, weights_dirs):
    """The model's last checkpoint: <model>[_frac_X]__epochN.pt with the largest N."""
    found = []
    for d in weights_dirs:
        for path in glob.glob(os.path.join(d, model + "_*.pt")):
            m = re.search(r"__epoch(\d+)\.pt$", path)
            if m:
                found.append((int(m.group(1)), path))
    if not found:
        return None, None
    epoch, path = max(found)
    return path, epoch


def load_run_cfg(run_dir):
    path = os.path.join(run_dir, "cfg.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f), path
    return None, None


def plan(opts):
    rows = []
    for model, pg, n_train, train_size, seed, run_path in RUNS:
        run_dir = os.path.join(opts.runs_root, run_path)
        weights, epoch = find_weights(model, opts.weights_dirs)
        cfg, cfg_path = load_run_cfg(run_dir)
        problems = []
        if weights is None:
            problems.append("no weights")
        elif epoch != 99:
            problems.append(f"last checkpoint is epoch {epoch}, expected 99")
        if cfg is None:
            problems.append(f"no cfg.json, will use {SWEEP_CONFIG}")
        elif int(cfg["featurizer"]["heuristic_features"]) != pg:
            problems.append(f"cfg.json has heuristic_features="
                            f"{cfg['featurizer']['heuristic_features']}, expected {pg}")
        out_dir = os.path.join(OUT_ROOT, model)
        cmd = [sys.executable, "main.py",
               "--config", os.path.join(out_dir, "eval_config.yaml"),
               *train_args(train_size, seed),
               "--eval_fns", "qerr",
               "--result_dir", out_dir,
               "--load_model", weights or f"<{model} weights>",
               "--extra_eval_query_dir", opts.extra_eval_query_dir,
               "--skip_train_eval", "1"]
        rows.append(dict(model=model, pg=pg, n_train=n_train, train_size=train_size,
                         seed=seed, run_dir=run_dir, cfg=cfg, cfg_path=cfg_path,
                         weights=weights, out_dir=out_dir, cmd=cmd, problems=problems))
    return rows


def write_eval_config(row):
    """The run's own config, with per-query prediction pickles and wandb off."""
    if row["cfg"] is not None:
        cfg = json.loads(json.dumps(row["cfg"]))
    else:
        import yaml
        with open(os.path.join(REPO_ROOT, SWEEP_CONFIG)) as f:
            cfg = yaml.safe_load(f)
    cfg["eval"]["save_test_preds"] = 0
    cfg["eval"]["use_wandb"] = 0
    os.makedirs(os.path.join(REPO_ROOT, row["out_dir"]), exist_ok=True)
    # JSON is valid YAML, which is what main.py reads
    with open(os.path.join(REPO_ROOT, row["out_dir"], "eval_config.yaml"), "w") as f:
        json.dump(cfg, f, indent=2)


def run_one(row):
    write_eval_config(row)
    log_path = os.path.join(REPO_ROOT, row["out_dir"], "run.log")
    print("\n$ " + " ".join(row["cmd"]), flush=True)
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(row["cmd"], cwd=REPO_ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
    print(f"{row['model']}: exit code {proc.returncode}, log in {log_path}")
    return proc.returncode


def list_runs(rows):
    print(f"{len(rows)} runs (PG x train size sweep, joblight_train, 100 epochs, dedup, no leakage)\n")
    print(f"{'#':>2}  {'model':<15} {'PG':>2} {'train subq':>10} {'seed':>4}  {'weights':<5} {'cfg':<4}  notes")
    for i, r in enumerate(rows):
        print(f"{i:>2}  {r['model']:<15} {r['pg']:>2} {r['n_train']:>10,} {str(r['seed'] or '-'):>4}  "
              f"{'yes' if r['weights'] else 'NO':<5} {'yes' if r['cfg'] else 'NO':<4}  "
              f"{'; '.join(r['problems'])}")
    ready = [r for r in rows if r["weights"]]
    print(f"\n{len(ready)} of {len(rows)} have weights.")
    if ready:
        print("Command for the first one:\n  " + " ".join(ready[0]["cmd"]))
    if len(ready) < len(rows):
        print("Looked for weights in: " + ", ".join(os.path.abspath(d) for d in OPTS.weights_dirs)
              + "\n(pass --weights_dirs to look elsewhere)")


# ---------------------------------------------------------------- collect

def qerror_stats(q):
    import numpy as np
    q = np.asarray(q, dtype=float)
    return dict(n=len(q), mean=q.mean(), median=np.median(q), p90=np.percentile(q, 90),
                p99=np.percentile(q, 99), max=q.max())


def read_estimates(csv_path):
    import numpy as np
    import pandas as pd
    df = pd.read_csv(csv_path).dropna(subset=["model_estimated"])
    est = df["model_estimated"].astype(float)
    true = df["true_cardinality"].astype(float)
    df["q"] = np.maximum(true / est, est / true)
    return df


def newest_result_dir(out_dir):
    dirs = [d for d in glob.glob(os.path.join(REPO_ROOT, out_dir, "*"))
            if os.path.isdir(d) and glob.glob(os.path.join(d, "cardinality_distributions_*.csv"))]
    return max(dirs, key=os.path.getmtime) if dirs else None


def collect(rows):
    import numpy as np
    import pandas as pd

    per_run = []
    for r in rows:
        res_dir = newest_result_dir(r["out_dir"])
        if res_dir is None:
            continue
        for split in SPLITS:
            path = os.path.join(res_dir, f"cardinality_distributions_{split}.csv")
            if not os.path.exists(path):
                continue
            df = read_estimates(path)
            row = dict(model=r["model"], pg=r["pg"], train_subqueries=r["n_train"],
                       seed=r["seed"], split=split, **qerror_stats(df["q"]))
            # Same estimates as the original run? Only then is the rebuilt
            # featurizer the one the model was trained with.
            orig = os.path.join(r["run_dir"], f"cardinality_distributions_{split}.csv")
            if split in CHECK_SPLITS and os.path.exists(orig):
                o = read_estimates(orig)
                both = df.merge(o, on=["name", "node"], suffixes=("", "_orig"))
                rel = (both["model_estimated"] - both["model_estimated_orig"]).abs() \
                    / both["model_estimated_orig"].abs()
                row["matched_rows"] = f"{len(both)}/{len(o)}"
                row["max_rel_diff_vs_original"] = rel.max()
            per_run.append(row)

    if not per_run:
        print("No results yet under " + os.path.join(REPO_ROOT, OUT_ROOT))
        return
    per_run = pd.DataFrame(per_run)
    os.makedirs(os.path.join(REPO_ROOT, OUT_ROOT), exist_ok=True)
    per_run.to_csv(os.path.join(REPO_ROOT, OUT_ROOT, "per_run.csv"), index=False)

    stats = ["mean", "median", "p90", "p99", "max"]
    # Runs of the same seed are averaged first, then mean / std across seeds.
    # The full-size PG=0 run has no seed; it counts as its own.
    per_run["seed_key"] = per_run["seed"].fillna(-1).astype(str) + "|" + \
        per_run["model"].where(per_run["seed"].isna(), "")
    by_seed = per_run.groupby(["split", "pg", "train_subqueries", "seed_key"], as_index=False)[stats + ["n"]].mean()
    summary = by_seed.groupby(["split", "pg", "train_subqueries"]).agg(
        seeds=("seed_key", "nunique"), n=("n", "max"),
        **{f"{s} mean": (s, "mean") for s in stats},
        **{f"{s} std": (s, "std") for s in stats}).reset_index()
    summary["split"] = pd.Categorical(summary["split"], SPLITS, ordered=True)
    summary = summary.sort_values(["split", "train_subqueries", "pg"])
    summary.to_csv(os.path.join(REPO_ROOT, OUT_ROOT, "summary.csv"), index=False)

    pd.set_option("display.width", 250)
    if "max_rel_diff_vs_original" in per_run:
        chk = per_run.dropna(subset=["max_rel_diff_vs_original"])
        bad = chk[chk["max_rel_diff_vs_original"] > 1e-3]
        print(f"Check against the original runs ({', '.join(CHECK_SPLITS)}): "
              f"{len(chk) - len(bad)} of {len(chk)} match (max relative difference <= 1e-3).")
        if len(bad):
            print("These do NOT match, so their featurizer differs from training "
                  "(e.g. code changed since); treat their CEB/JOB numbers with care:")
            print(bad[["model", "split", "matched_rows", "max_rel_diff_vs_original"]].to_string(index=False))
    fmt = lambda v: f"{v:,.2f}"
    for split in SPLITS:
        s = summary[summary["split"] == split]
        if len(s) == 0:
            continue
        print(f"\n== {split} ==")
        cols = ["pg", "train_subqueries", "seeds", "n"] + [c for st in stats for c in (f"{st} mean", f"{st} std")]
        print(s[cols].to_string(index=False, float_format=fmt))
    print("\nWrote " + os.path.join(OUT_ROOT, "per_run.csv") + " and " + os.path.join(OUT_ROOT, "summary.csv"))


def main():
    global OPTS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true", help="score the models (default: only list them)")
    p.add_argument("--collect", action="store_true", help="summarize finished runs")
    p.add_argument("--only", nargs="+", default=None, help="model names to run")
    p.add_argument("--index", type=int, default=None, help="run only row N of the list (Slurm array)")
    p.add_argument("--weights_dirs", nargs="+", default=["saved_models"],
                   help="where to look for <model>_*__epochN.pt (default: saved_models)")
    p.add_argument("--runs_root", default=".",
                   help="folder the run paths (results/train_size_sweep/...) are relative to, "
                        "for cfg.json and the original CSVs (default: repo root)")
    p.add_argument("--extra_eval_query_dir", default="queries/ceb-imdb,queries/job")
    p.add_argument("--rerun", action="store_true", help="run again even if results exist")
    OPTS = p.parse_args()
    os.chdir(REPO_ROOT)

    rows = plan(OPTS)
    if OPTS.only:
        rows = [r for r in rows if r["model"] in OPTS.only]
    if OPTS.index is not None:
        rows = [rows[OPTS.index]]

    if OPTS.collect:
        collect(rows)
        return
    if not OPTS.run:
        list_runs(rows)
        return

    failed = []
    for r in rows:
        if r["weights"] is None:
            print(f"{r['model']}: no weights, skipping")
            continue
        if not OPTS.rerun and newest_result_dir(r["out_dir"]) is not None \
                and all(os.path.exists(os.path.join(newest_result_dir(r["out_dir"]),
                        f"cardinality_distributions_{os.path.basename(os.path.normpath(d))}.csv"))
                        for d in OPTS.extra_eval_query_dir.split(",")):
            print(f"{r['model']}: already scored, skipping (--rerun to redo)")
            continue
        if run_one(r) != 0:
            failed.append(r["model"])
    if failed:
        print("Failed: " + " ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
