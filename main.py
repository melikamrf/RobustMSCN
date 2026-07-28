import sys
sys.path.append(".")
# from query_representation.query import *
from compute_dist_table import compute_distributions_for_qreps, compute_jsd, extract_subquery_rows, build_distribution
from query_representation.utils import get_query_splits, SOURCE_NODE

import os
# This forces the output to be unbuffered at the binary level
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from cardinality_estimation.featurizer import *
from cardinality_estimation.dataset import QueryDataset, load_qdata
from cardinality_estimation import get_alg
from evaluation.eval_fns import get_eval_fn
# Distribution table utilities
#from compute_dist_table import compute_distributions_for_files, compute_distributions_for_qreps
# import glob
import argparse
# import random
import json

import pdb
import copy
import pickle
import os
import yaml
from collections import defaultdict, Counter

import wandb
import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

# Setup main logger
main_logger = logging.getLogger(__name__)
main_logger.setLevel(logging.INFO)
if not main_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    main_logger.addHandler(handler)

import time

import pandas as pd
import numpy as np

def _pattern_to_string(pattern):
    if isinstance(pattern, tuple) and len(pattern) == 4:
        return f"{pattern[0]}.{pattern[1]} = {pattern[2]}.{pattern[3]}"
    if isinstance(pattern, list):
        return " | ".join(_pattern_to_string(item) for item in pattern)
    if isinstance(pattern, tuple):
        return " | ".join(_pattern_to_string(item) for item in pattern)
    return str(pattern)


def _build_frequency_dataframe(items):
    total, counts, _ = build_distribution(items)
    rows = []
    for pattern, count in sorted(counts.items(), key=lambda x: (-x[1], str(x[0]))):
        rows.append({
            "pattern": _pattern_to_string(pattern),
            "count": int(count),
            "probability": (count / total if total else 0.0),
        })
    return pd.DataFrame(rows, columns=["pattern", "count", "probability"])


def _save_distribution_artifacts(split_name, dist, workload_df, output_dir, save_format="pkl"):
    if output_dir is None:
        return None

    os.makedirs(output_dir, exist_ok=True)
    tables_items = workload_df["tables"].tolist() if not workload_df.empty else []
    joins_items = [item for item in (workload_df["joins"].tolist() if not workload_df.empty else []) if item]
    predicate_items = []
    for preds in (workload_df["predicates"].tolist() if not workload_df.empty else []):
        if not preds:
            continue
        for pred in preds:
            if pred is not None:
                predicate_items.append([pred])

    tables_df = _build_frequency_dataframe(tables_items)
    joins_df = _build_frequency_dataframe(joins_items)
    predicates_df = _build_frequency_dataframe(predicate_items)

    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    if save_format.lower() == "pkl":
        payload = {
            "split_name": split_name,
            "distribution": dist,
            "workload_df": workload_df,
            "frequency_tables": {
                "tables": tables_df,
                "joins": joins_df,
                "predicates": predicates_df,
            },
        }
        out_path = os.path.join(split_dir, f"{split_name}_dist_table.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)
        return out_path

    tables_path = os.path.join(split_dir, f"{split_name}_tables_frequency.csv")
    joins_path = os.path.join(split_dir, f"{split_name}_joins_frequency.csv")
    predicates_path = os.path.join(split_dir, f"{split_name}_predicates_frequency.csv")
    workload_path = os.path.join(split_dir, f"{split_name}_subqueries.csv")

    tables_df.to_csv(tables_path, index=False)
    joins_df.to_csv(joins_path, index=False)
    predicates_df.to_csv(predicates_path, index=False)
    workload_df.to_csv(workload_path, index=False)
    return {
        "tables": tables_path,
        "joins": joins_path,
        "predicates": predicates_path,
        "workload": workload_path,
    }


def calc_datasets_jsd(trainqs, valqs, testqs, evalqs, eval_qdirs,
        result_dir=None, save_dist_tables=True, dist_table_format="pkl"):
    """
    Compute JSD between training and eval query distributions.
    Uses the whole dataset only; it does not stratify by join count.
    """
    split_results = {}

    split_results["train"] = compute_distributions_for_qreps(
        "train_all", trainqs, return_df=True, num_joins=None
    )
    split_results["val"] = compute_distributions_for_qreps(
        "val_all", valqs, return_df=True, num_joins=None
    )
    split_results["test"] = compute_distributions_for_qreps(
        "test_all", testqs, return_df=True, num_joins=None
    )

    for idx, evalq in enumerate(evalqs):
        raw_label = eval_qdirs[idx] if idx < len(eval_qdirs) else f"eval_{idx}"
        label = os.path.basename(os.path.normpath(raw_label)) or f"eval_{idx}"
        split_results[label] = compute_distributions_for_qreps(
            f"{label}_all", evalq, return_df=True, num_joins=None
        )

    if save_dist_tables:
        dist_root = os.path.join(result_dir or "./results", "dataset_jsd_tables")
        for split_name, (dist, workload_df) in split_results.items():
            _save_distribution_artifacts(
                split_name,
                dist,
                workload_df,
                dist_root,
                save_format=dist_table_format,
            )

    def _collect_jsd_rows(rows, source_label, target_label):
        source_dist, _ = split_results[source_label]
        target_dist, _ = split_results[target_label]

        rows.append({
            "source_split": source_label,
            "target_split": target_label,
            "num_joins": None,
            "jsd_joins": compute_jsd(source_dist["joins"], target_dist["joins"]),
            "jsd_predicates": compute_jsd(source_dist["predicates"], target_dist["predicates"]),
            "jsd_tables": compute_jsd(source_dist["tables"], target_dist["tables"]),
        })

    print("Calculating dataset distribution similarities (JSD) between train, val, test, and eval query sets...")

    rows = []
    _collect_jsd_rows(rows, "train", "val")
    _collect_jsd_rows(rows, "train", "test")

    for idx, _ in enumerate(evalqs):
        raw_label = eval_qdirs[idx] if idx < len(eval_qdirs) else f"eval_{idx}"
        label = os.path.basename(os.path.normpath(raw_label)) or f"eval_{idx}"
        _collect_jsd_rows(rows, "train", label)

    jsd_df = pd.DataFrame(rows)
    for _, row in jsd_df.iterrows():
        join_scope = "all joins" if pd.isna(row["num_joins"]) else f"join count {int(row['num_joins'])}"
        print(
            f"{row['source_split']} vs {row['target_split']} ({join_scope}): "
            f"joins={row['jsd_joins']:.6f}, predicates={row['jsd_predicates']:.6f}, tables={row['jsd_tables']:.6f}"
        )

    return jsd_df


def _format_subquery_tables(subquery_id):
    table_entries = [str(alias) for alias in subquery_id]
    table_entries.sort()
    return " | ".join(table_entries)


def _build_subquery_level_dataframe(qreps):
    rows = []
    for qrep in qreps:
        subset_graph = qrep["subset_graph"]
        for subquery_row in extract_subquery_rows(qrep, num_joins=None):
            subquery_id = subquery_row["subquery_id"]
            if not subquery_id:
                continue
            card_info = subset_graph.nodes()[subquery_id].get("cardinality", {})
            rows.append({
                "tables": _format_subquery_tables(subquery_id),
                "joins": " | ".join(str(join) for join in subquery_row["joins"]),
                "predicates": " | ".join(str(pred) for pred in subquery_row["predicates"]),
                "true_cardinality": card_info.get("actual"),
                "postgres_cardinality": card_info.get("expected"),
                "source_pickle": qrep.get("name"),
            })

    return pd.DataFrame(rows, columns=[
        "tables",
        "joins",
        "predicates",
        "true_cardinality",
        "postgres_cardinality",
        "source_pickle",
    ])


def _dataset_file_prefix(query_dir):
    base_name = os.path.basename(os.path.normpath(query_dir))
    if base_name.endswith("_train"):
        return base_name[:-6]
    return base_name


def save_subquery_split_dataframes(trainqs, valqs, testqs, evalqs, eval_qdirs, result_dir, query_dir):
    if result_dir is None:
        return {}

    export_dir = os.path.join(result_dir, "query_level_dataframes")
    os.makedirs(export_dir, exist_ok=True)

    train_prefix = _dataset_file_prefix(query_dir)

    split_to_qreps = {
        "train": (trainqs, train_prefix),
        "val": (valqs, train_prefix),
        "test": (testqs, train_prefix),
    }
    for idx, evalq in enumerate(evalqs):
        raw_label = eval_qdirs[idx] if idx < len(eval_qdirs) else f"eval_{idx}"
        split_name = os.path.basename(os.path.normpath(raw_label)) or f"eval_{idx}"
        split_to_qreps[split_name] = (evalq, split_name)

    saved_paths = {}
    for split_name, value in split_to_qreps.items():
        qreps, prefix = value
        if qreps is None:
            continue
        df = _build_subquery_level_dataframe(qreps)
        if split_name == "train":
            file_stem = f"{prefix}_train"
        elif split_name == "test":
            file_stem = f"{prefix}_test"
        elif split_name == "val":
            file_stem = f"{prefix}_val"
        else:
            file_stem = f"{prefix}_{split_name}"
        csv_path = os.path.join(export_dir, f"{file_stem}.csv")
        df.to_csv(csv_path, index=False)
        saved_paths[split_name] = csv_path
        print(f"Saved query-level dataframe for {split_name} to: {csv_path}")

    return saved_paths


def _qrep_subquery_signatures(qrep):
    """
    Maps each subquery node (subset_graph node) in qrep to a hashable
    signature of its (tables, joins, predicates) -- two subqueries with the
    same signature are the same subquery, regardless of which top-level
    query/pkl file they came from.
    """
    sigs = {}
    for row in extract_subquery_rows(qrep, num_joins=None):
        sigs[row["subquery_id"]] = (
            tuple(row["tables"]),
            tuple(row["joins"]),
            tuple(str(pred) for pred in row["predicates"]),
        )
    return sigs


def _build_keep_mask(qreps, excluded_by_qrep_idx):
    """
    @excluded_by_qrep_idx: {qrep_idx: set(subquery_id)} -- nodes to leave
    out of the mask (i.e. not used as training/eval samples).

    Returns a subplan_mask in the format QueryDataset expects: a list,
    same length/order as qreps, of lists of `list(node)` for every
    subset_graph node to KEEP as a sample. subset_graph itself is never
    read destructively here -- every node's data stays put, so any other
    (kept) node that depends on looking it up (e.g. a multi-table
    subplan's featurizer reading its single-table components' cardinality
    estimates) is unaffected by what this mask excludes.
    """
    mask = []
    for qi, qrep in enumerate(qreps):
        excluded = excluded_by_qrep_idx.get(qi, set())
        keep = [list(node) for node in qrep["subset_graph"].nodes()
                if node not in excluded]
        mask.append(keep)
    return mask


def _intersect_masks(qreps, mask_a, mask_b):
    """
    Combines two subplan_masks (either may be None, meaning "no
    restriction") into one that keeps only nodes present in both.
    """
    if mask_a is None:
        return mask_b
    if mask_b is None:
        return mask_a
    combined = []
    for qi in range(len(qreps)):
        keep_a = {tuple(node) for node in mask_a[qi]}
        keep_b = {tuple(node) for node in mask_b[qi]}
        combined.append([list(node) for node in (keep_a & keep_b)])
    return combined


def remove_duplicate_subqueries(qreps, split_name="split"):
    """
    Computes a subplan_mask (see QueryDataset/cardinality_estimation's
    train() methods) that keeps only the first occurrence -- by qrep
    order -- of each subquery signature (tables/joins/predicates) within
    this split; later occurrences of the same signature, in other qreps,
    are excluded from becoming training/eval samples.

    subset_graph is never modified: every qrep keeps its full node set,
    so featurizer lookups that a kept subplan makes against its own
    qrep's graph (e.g. a multi-table subplan reading the postgres
    estimate of one of its single-table components) always succeed,
    regardless of what this mask excludes elsewhere in the same graph.
    Only which nodes get turned into dataset rows is affected.

    Returns (mask, num_excluded): mask is a list, same length/order as
    qreps, of lists of `list(node)` -- pass it as `subplan_mask` (train),
    `val_subplan_mask`, or `test_subplan_mask` to alg.train()/
    train_with_dann()/train_with_new_discriminator().
    """
    seen = {}
    excluded_by_qrep = defaultdict(set)
    num_excluded = 0
    for qi, qrep in enumerate(qreps):
        for subquery_id, sig in _qrep_subquery_signatures(qrep).items():
            if sig in seen:
                excluded_by_qrep[qi].add(subquery_id)
                num_excluded += 1
            else:
                seen[sig] = (qi, subquery_id)

    mask = _build_keep_mask(qreps, excluded_by_qrep)
    main_logger.info(
        f"[{split_name}] remove_duplicate_subqueries: excluding {num_excluded} "
        f"duplicate subquery nodes from training/eval samples "
        f"({len(seen)} unique subqueries kept)"
    )
    return mask, num_excluded


def detect_leakage(named_splits, remove=False, priority=None):
    """
    @named_splits: [(split_name, qreps), ...], e.g.
        [("train", trainqs), ("val", valqs), ("test", testqs)]
    Finds subquery signatures (tables/joins/predicates) that appear in
    more than one split, and logs the pairwise overlap counts.

    @remove: if True, also builds a subplan_mask per split that excludes
    leaked subplans from becoming training/eval samples in every split
    except the highest-priority one that contains them (priority
    defaults to the order of named_splits, i.e. train wins over val, val
    wins over test). subset_graph is never modified, same as
    remove_duplicate_subqueries -- only which nodes get turned into
    dataset rows is affected.

    Returns (report, masks): report is a dict with leaked signature
    counts and pairwise overlap counts; masks is {split_name: mask or
    None} (None meaning "no filtering needed for this split").
    """
    split_names = [name for name, _ in named_splits]
    if priority is None:
        priority = split_names
    priority_rank = {name: i for i, name in enumerate(priority)}

    # signature -> {split_name: [(qrep_idx, subquery_id), ...]}
    sig_to_splits = defaultdict(lambda: defaultdict(list))
    for split_name, qreps in named_splits:
        for qi, qrep in enumerate(qreps):
            for subquery_id, sig in _qrep_subquery_signatures(qrep).items():
                sig_to_splits[sig][split_name].append((qi, subquery_id))

    overlap_counts = Counter()
    excluded_by_split_qrep = {name: defaultdict(set) for name in split_names}
    leaked_signatures = 0

    for sig, per_split in sig_to_splits.items():
        present_in = [s for s in split_names if s in per_split]
        if len(present_in) <= 1:
            continue

        leaked_signatures += 1
        for i in range(len(present_in)):
            for j in range(i + 1, len(present_in)):
                overlap_counts[(present_in[i], present_in[j])] += 1

        if remove:
            keep_split = min(present_in,
                    key=lambda s: priority_rank.get(s, len(priority)))
            for split_name in present_in:
                if split_name == keep_split:
                    continue
                for qi, subquery_id in per_split[split_name]:
                    excluded_by_split_qrep[split_name][qi].add(subquery_id)

    main_logger.info(
        f"detect_leakage: {leaked_signatures} subquery signatures found in "
        f"more than one split"
    )
    for (split_a, split_b), count in overlap_counts.items():
        main_logger.info(f"  overlap {split_a} <-> {split_b}: {count} shared subqueries")

    masks = {}
    removed_counts = {}
    for split_name, qreps in named_splits:
        if remove:
            masks[split_name] = _build_keep_mask(qreps, excluded_by_split_qrep[split_name])
            removed_counts[split_name] = sum(
                    len(v) for v in excluded_by_split_qrep[split_name].values())
        else:
            masks[split_name] = None

    if remove:
        for split_name, count in removed_counts.items():
            main_logger.info(f"  excluding {count} leaked subquery nodes from {split_name}")

    return {
        "leaked_signatures": leaked_signatures,
        "overlap_counts": dict(overlap_counts),
        "removed_counts": removed_counts,
    }, masks


def split_queries_for_discriminator(target_queries, holdout_fraction=0.5, random_state=42):
    """
    Split target queries at the query level into:
      1. discriminator adaptation target set
      2. held-out target eval set for later MSCN evaluation
    """
    if len(target_queries) < 2 or holdout_fraction <= 0.0:
        return list(target_queries), []

    from sklearn.model_selection import train_test_split

    disc_targetqs, heldout_evalqs = train_test_split(
        list(target_queries),
        test_size=holdout_fraction,
        random_state=random_state,
        shuffle=True,
    )
    return disc_targetqs, heldout_evalqs


def prepare_discriminator_weights(trainqs, evalqs, eval_qdirs, featurizer):
    """
    Optionally train discriminator and return:
      1. adversarial source weights for training
      2. eval query sets to use for MSCN evaluation
      3. eval query directories aligned with returned eval query sets
    """
    disc_weights = None
    mscn_evalqs = evalqs
    mscn_eval_qdirs = eval_qdirs

    if not args.use_discriminator:
        main_logger.info(
            "Discriminator disabled via --use_discriminator=0. "
            "Proceeding with uniform training weights."
        )
        return disc_weights, mscn_evalqs, mscn_eval_qdirs

    if len(evalqs) == 0 or len(evalqs[0]) == 0:
        main_logger.info("Skipping discriminator training because no eval queries were loaded.")
        return disc_weights, [], []

    disc_targetqs, heldout_evalqs = split_queries_for_discriminator(
        evalqs[0],
        holdout_fraction=args.disc_holdout_frac,
        random_state=args.random_seed,
    )
    main_logger.info(
        "Target workload split for discriminator/MSCN eval: "
        f"{len(disc_targetqs)} queries for discriminator training, "
        f"{len(heldout_evalqs)} held out for MSCN evaluation"
    )
    if len(heldout_evalqs) > 0:
        mscn_evalqs = [heldout_evalqs]
        mscn_eval_qdirs = [eval_qdirs[0]]
    else:
        mscn_evalqs = []
        mscn_eval_qdirs = []

    from discriminator import train_discriminator, save_discriminator_outputs

    main_logger.info("Starting domain discriminator training...")
    start_disc_time = time.time()
    disc_result = train_discriminator(
        source_queries=trainqs,
        target_queries=disc_targetqs,
        featurizer=featurizer,
        batch_size=args.disc_batch_size,
        epochs=args.disc_epochs,
        lr=args.disc_lr,
    )
    disc_time = time.time() - start_disc_time
    main_logger.info(f"Domain discriminator training completed in {disc_time:.2f} seconds")
    main_logger.info(f"Discriminator loss plot saved to: {disc_result['plot_path']}")
    main_logger.info(
        f"Discriminator stats - "
        f"Source predictions mean: {disc_result['source_predictions'].mean():.4f}, "
        f"Target predictions mean: {disc_result['target_predictions'].mean():.4f}"
    )

    disc_dir = os.path.join(args.result_dir, "domain_discriminator")
    artifact_paths = save_discriminator_outputs(disc_result, disc_dir)
    main_logger.info(
        f"Saved discriminator artifacts: source_predictions={artifact_paths['source_predictions']}, "
        f"source_density_ratios={artifact_paths['source_density_ratios']}, "
        f"source_weight_map={artifact_paths['source_weight_map']}, "
        f"model_state={artifact_paths['model_state']}"
    )
    disc_weights = disc_result["source_density_ratios"]
    return disc_weights, mscn_evalqs, mscn_eval_qdirs

def extract_cardinalities(qreps, ests):
    """
    Extract true, postgres estimated, model estimated cardinalities, and q-error.
    """
    card_data = []
    
    for idx, (qrep, est) in enumerate(zip(qreps, ests)):
        # Access subset_graph nodes (this is where cardinalities are stored)
        if "subset_graph" in qrep:
            sg = qrep["subset_graph"]
            for node, data in sg.nodes(data=True):
            
                true_card = data["cardinality"]["actual"]
                postgres_card = data["cardinality"]["expected"]
                model_card = est[node] if node in est else None

                if model_card is not None and true_card not in (None, 0) and model_card != 0:
                    qerror = np.maximum((true_card / model_card), (model_card / true_card))
                else:
                    qerror = np.nan
                
                # Calculate the ratio to verify update_labels was applied
                # If update_labels was applied: new_actual = old_actual / old_expected
                # So the ratio should reflect the updated value
                ratio = true_card / postgres_card if postgres_card > 0 else None
                
                card_data.append({
                    "name": qrep.get("name", f"query_{idx}"),
                    "node": node,
                    "true_cardinality": true_card,
                    "postgres_estimated": postgres_card,
                    "model_estimated": model_card,  # Model estimates the root query
                    "qerror": qerror,
                    "actual_expected_ratio": ratio,
                    # Add original values before update if available
                    "is_ratio_close_to_1": abs(ratio - 1.0) < 0.01 if ratio else False
                })
    
    return pd.DataFrame(card_data)

def undersample_train_queries(trainqs, eqs, seed=None):
    """
    Undersample source (trainqs) to match target (evalqs) size.
    """
    total_evalqs_count = sum(eqs)
    if len(trainqs) > total_evalqs_count:
        seed = args.random_seed if seed is None else seed
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(trainqs), size=total_evalqs_count, replace=False)
        trainqs = [trainqs[i] for i in sorted(indices)]
        print(f"Undersampled trainqs to {len(trainqs)} to match evalqs size ({total_evalqs_count})")
    elif len(trainqs) < total_evalqs_count:
        print(f"WARNING: trainqs size ({len(trainqs)}) is already smaller than evalqs size ({total_evalqs_count}). No undersampling performed.")
    
    return trainqs

def undersample_train_queries_by_subquery_count(trainqs, evalqs, seed=None):
    """
    Undersample source (trainqs) to match target (evalqs) by total subquery count.
    
    Args:
        trainqs: List of training queries
        evalqs: List of lists of eval queries
        
    Returns:
        Undersampled trainqs list with approximately same subquery count as evalqs
    """
    # Count total subqueries in evalqs
    total_eval_subqueries = 0
    for evalq_set in evalqs:
        for qrep in evalq_set:
            if "subset_graph" in qrep:
                total_eval_subqueries += len(qrep["subset_graph"].nodes())
    
    # Count subqueries per training query
    train_subquery_counts = []
    for qrep in trainqs:
        if "subset_graph" in qrep:
            count = len(qrep["subset_graph"].nodes())
        else:
            count = 1
        train_subquery_counts.append(count)
    
    # Undersample trainqs to match eval subquery count
    seed = args.random_seed if seed is None else seed
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(trainqs))
    selected_indices = []
    total_train_subqueries = 0
    
    for idx in indices:
        subquery_count = train_subquery_counts[idx]
        if total_train_subqueries + subquery_count <= total_eval_subqueries:
            selected_indices.append(idx)
            total_train_subqueries += subquery_count
    
    # Sort to preserve original order
    selected_indices = sorted(selected_indices)
    trainqs = [trainqs[i] for i in selected_indices]
    
    print(f"Undersampled trainqs: {len(trainqs)} queries with {total_train_subqueries} subqueries to match evalqs ({total_eval_subqueries} subqueries)")
    
    return trainqs

def update_labels(qreps):
    """
    Add residual labels to qreps without overwriting true cardinalities.
    residual = actual - expected.
    """
    for qrep in qreps:
        sg = qrep["subset_graph"]
        for node, data in sg.nodes(data=True):
            if "cardinality" not in data:
                continue
            card = data["cardinality"]
            card["changed"] = False
            if "actual" in card and "expected" in card:
                card["actual"] = card["actual"] / card["expected"]
                #card["changed"] = True

        for _, _, data in sg.edges(data=True):
            if "join_key_cardinality" not in data:
                continue
            for _, jcard in data["join_key_cardinality"].items():
                if "actual" in jcard and "expected" in jcard:
                    jcard["actual"] = jcard["actual"] / jcard["expected"]

    return qreps

# eval funcs whose eval()/save_logs() only look at preds[i].keys() (never
# at qrep["subset_graph"].nodes() directly to decide what to score), so
# it's safe to hand them a subset of a qrep's subquery predictions.
# Plan-cost style funcs (ppc/simple plan cost) need every subplan's
# estimate to build a full query plan, so they're deliberately excluded
# here and always get the complete, unfiltered predictions.
_PER_SUBQUERY_EVAL_FUNCS = {"QError", "AbsError", "MeanSquaredError", "RelativeError"}

def _filter_ests_by_mask(ests, subplan_mask):
    """
    @ests: [{node: prediction}, ...], one dict per qrep, as returned by
    alg.test() -- always covers every subset_graph node.
    @subplan_mask: subplan_mask format (list, same order as the qreps
    ests came from, of lists of `list(node)` to keep).

    Returns a same-shape copy of ests with entries for excluded nodes
    dropped, so per-subquery metrics computed from it only reflect kept
    (e.g. deduped) subqueries.
    """
    filtered = []
    for qi, qrep_ests in enumerate(ests):
        if qi >= len(subplan_mask):
            filtered.append(qrep_ests)
            continue
        keep = {tuple(node) for node in subplan_mask[qi]}
        filtered.append({k: v for k, v in qrep_ests.items() if k in keep})
    return filtered

def eval_alg(alg, eval_funcs, qreps, cfg,
        samples_type,
        featurizer=None,
        subplan_mask=None):
    '''
    '''
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    start = time.time()
    alg_name = alg.__str__()
    exp_name = alg.get_exp_name()
    samples_label = str(samples_type).strip() if samples_type is not None else ""
    if not samples_label:
        samples_label = "eval"

    ests = alg.test(qreps)
    rdir = None
    if args.result_dir is not None:
        rdir = os.path.join(args.result_dir, exp_name)
        make_dir(rdir)
        # print("Going to store results at: ", rdir)
        args_fn = os.path.join(rdir, "cfg.json")
        with open(args_fn, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)
            
    df = extract_cardinalities(qreps, ests)
    card_path = os.path.join(args.result_dir, exp_name, f"cardinality_distributions_{samples_label}.csv")
    df.to_csv(card_path, index=False)
    print(f"Saved {len(df)} query cardinalities to CSV")

    if cfg["eval"]["save_test_preds"]:
        preds_dir = os.path.join(rdir, samples_label + "-preds")
        make_dir(preds_dir)
        for i,qrep in enumerate(qreps):
            newfn = os.path.basename(qrep["name"])
            predfn = os.path.join(preds_dir, qrep["name"])
            cur_ests = ests[i]
            with open(predfn, "wb") as f:
                pickle.dump(cur_ests, f)

    if subplan_mask is not None:
        masked_ests = _filter_ests_by_mask(ests, subplan_mask)
    else:
        masked_ests = ests

    for efunc in eval_funcs:
        if "plan" in str(efunc).lower() and "train" in qreps[0]["template_name"]:
            print("skipping _train_ workload plan cost eval")
            continue

        cur_ests = masked_ests if type(efunc).__name__ in _PER_SUBQUERY_EVAL_FUNCS else ests

        errors = efunc.eval(qreps, cur_ests,
                user = cfg["db"]["user"], pwd = cfg["db"]["pwd"],
                port = cfg["db"]["port"], db_name = cfg["db"]["db_name"],
                db_host = cfg["db"]["db_host"],
                samples_type = samples_label,
                num_processes = cfg["eval"]["num_processes"],
                alg_name = alg_name,
                save_pdf_plans= cfg["eval"]["save_pdf_plans"],
                query_dir = cfg["data"]["query_dir"],
                result_dir = args.result_dir,
                use_wandb = cfg["eval"]["use_wandb"],
                featurizer = featurizer, alg=alg)

        print("{}, {}, {}, #samples: {}, {}: mean: {}, 25p: {}, median: {}, 75p: {}, 99p: {}, max: {}"\
                .format(cfg["db"]["db_name"], samples_label, alg, len(errors),
                    efunc.__str__(), np.round(np.mean(errors),3),
                    np.round(np.percentile(errors,25),3),
                    np.round(np.median(errors),3),
                    np.round(np.percentile(errors,75),3),
                    np.round(np.percentile(errors,99),3),
                    np.round(np.max(errors))))

        if cfg["eval"]["use_wandb"]:
            loss_key = "Final-{}-{}-{}".format(str(efunc), samples_label,
                    "mean")
            wandb.run.summary[loss_key] = np.round(np.mean(errors),3)

    print("All loss computations took: ", time.time()-start)

def get_featurizer(trainqs, valqs, testqs, eval_qs):

    featurizer = Featurizer(**cfg["db"])
    featdata_fn = os.path.join(cfg["data"]["query_dir"],
            "dbdata.json")

    all_evalqs = []
    for e0 in eval_qs:
        all_evalqs += e0

    if args.regen_featstats or not os.path.exists(featdata_fn):
        # we can assume that we have db stats for any column in the db
        featurizer.update_column_stats(trainqs+valqs+testqs+all_evalqs)
        ATTRS_TO_SAVE = ['aliases', 'cmp_ops', 'column_stats', 'joins',
                'max_in_degree', 'max_joins', 'max_out_degree', 'max_preds',
                'max_tables', 'regex_cols', 'tables', 'join_key_stats',
                'primary_join_keys', 'join_key_normalizers',
                'join_key_stat_names', 'join_key_stat_tmps'
                'max_tables', 'regex_cols', 'tables',
                'mcvs']

        featdata = {}
        for k in dir(featurizer):
            if k not in ATTRS_TO_SAVE:
                continue
            attrvals = getattr(featurizer, k)
            if isinstance(attrvals, set):
                attrvals = list(attrvals)
            featdata[k] = attrvals

        if args.save_featstats:
            f = open(featdata_fn, "w")
            json.dumps(featdata, f)
            f.close()
    else:
        f = open(featdata_fn, "r")
        featdata = json.load(f)
        f.close()
        featurizer.update_using_saved_stats(featdata)

    if args.alg in ["mscn", "mscn_joinkey", "mstn"]:
        feat_type = "set"
    else:
        feat_type = "combined"

    card_type = "subplan"

    # Look at the various keyword arguments to setup() to change the
    # featurization behavior; e.g., include certain features etc.
    # these configuration properties do not influence the basic statistics
    # collected in the featurizer.update_column_stats call; Therefore, we don't
    # include this in the cached version

    qdir_name = os.path.basename(cfg["data"]["query_dir"])
    bitmap_dir = cfg["data"]["bitmap_dir"]
    # ** converts the dictionary into keyword args
    featurizer.setup(
            **cfg["featurizer"],
            loss_func = cfg["model"]["loss_func_name"],
            featurization_type = feat_type,
            bitmap_dir = cfg["data"]["bitmap_dir"],
            card_type = card_type
            )

    # just updates stuff like max-num-tables etc. for some implementation
    # things
    featurizer.update_max_sets(trainqs+valqs+testqs+all_evalqs)
    featurizer.update_workload_stats(trainqs+valqs+testqs+all_evalqs)

    featurizer.init_feature_mapping()

    if cfg["featurizer"]["feat_onlyseen_maxy"]:
        featurizer.update_ystats(trainqs,
                max_num_tables=cfg["model"]["max_num_tables"])
    else:
        featurizer.update_ystats(trainqs+valqs+testqs+all_evalqs,
                max_num_tables = cfg["model"]["max_num_tables"])

    featurizer.update_seen_preds(trainqs)

    return featurizer

def main():
    global args,cfg

    with open(args.config) as f:
        cfg = yaml.safe_load(f.read())

    print(yaml.dump(cfg, default_flow_style=False))

    # set up wandb logging metrics
    if cfg["eval"]["use_wandb"]:
        wandbcfg = {}
        for k,v in cfg.items():
            if isinstance(v, dict):
                for k2,v2 in v.items():
                    wandbcfg.update({k+"-"+k2:v2})
            else:
                wandbcfg.update({k:v})

        wandbcfg.update(vars(args))
        # additional config tags
        wandb_tags = ["1a"]
        if args.wandb_tags is not None:
            wandb_tags += args.wandb_tags.split(",")
        wandb.init(project="ceb", config=wandbcfg,
                tags=wandb_tags)

    train_qfns, test_qfns, val_qfns, eval_qfns = get_query_splits(cfg["data"])
    trainqs = load_qdata(train_qfns)
    # Note: can be quite memory intensive to load them all; might want to just
    # keep around the qfns and load them as needed
    valqs = load_qdata(val_qfns)
    testqs = load_qdata(test_qfns)

    # subplan_mask passed to alg.train()/train_with_dann()/
    # train_with_new_discriminator(): which subset_graph nodes to
    # actually turn into training/eval samples. None means "no
    # filtering" (the original, possibly-duplicated/leaky data).
    train_subplan_mask = None
    val_subplan_mask = None
    test_subplan_mask = None

    if args.remove_duplicate_subqueries:
        train_subplan_mask, _ = remove_duplicate_subqueries(trainqs, split_name="train")
        val_subplan_mask, _ = remove_duplicate_subqueries(valqs, split_name="val")
        test_subplan_mask, _ = remove_duplicate_subqueries(testqs, split_name="test")

    if args.remove_leakage or args.detect_leakage:
        _, leakage_masks = detect_leakage(
                [("train", trainqs), ("val", valqs), ("test", testqs)],
                remove=bool(args.remove_leakage))
        if args.remove_leakage:
            train_subplan_mask = _intersect_masks(trainqs, train_subplan_mask,
                    leakage_masks.get("train"))
            val_subplan_mask = _intersect_masks(valqs, val_subplan_mask,
                    leakage_masks.get("val"))
            test_subplan_mask = _intersect_masks(testqs, test_subplan_mask,
                    leakage_masks.get("test"))

    if args.learn_residual:
        trainqs = update_labels(trainqs)
        valqs = update_labels(valqs)
        testqs = update_labels(testqs)
    
    eval_qdirs = cfg["data"]["eval_query_dir"].split(",")

    evalqs = []
    for eval_qfn in eval_qfns:
        temp_evalqs = load_qdata(eval_qfn)
        if args.learn_residual:
            temp_evalqs = update_labels(temp_evalqs)
            
        evalqs.append(temp_evalqs)
        
    eqs = [len(eq) for eq in evalqs]
    print("""Selected Queries: {} train, {} test, {} val, {} eval"""\
            .format(len(trainqs), len(testqs), len(valqs), sum(eqs)))

    if args.result_dir is not None:
        save_subquery_split_dataframes(
            trainqs,
            valqs,
            testqs,
            evalqs,
            eval_qdirs,
            args.result_dir,
            cfg["data"]["query_dir"],
        )
   

    # Undersample source (trainqs) to match target (evalqs) size
    # Use undersample_train_queries() to match query count
    # OR use undersample_train_queries_by_subquery_count() to match total subquery count
    # trainqs = undersample_train_queries_by_subquery_count(trainqs, evalqs, seed=args.undersample_seed)
    #trainqs = undersample_train_queries(trainqs, eqs, seed=args.undersample_seed)

    dataset_jsd_df = calc_datasets_jsd(trainqs, valqs, testqs, evalqs, eval_qdirs, args.result_dir)
    if args.result_dir is not None:
        os.makedirs(args.result_dir, exist_ok=True)
        train_dataset = cfg["data"]["query_dir"].rstrip("/").split("/")[-1]
        dataset_jsd_path = os.path.join(args.result_dir, f"dataset_jsd_{train_dataset}.csv")
        dataset_jsd_df.to_csv(dataset_jsd_path, index=False)
        print(f"Saved dataset JSD summary to {dataset_jsd_path}")

        
    # only needs featurizer for learned models
    if args.alg in ["xgb", "fcnn", "mscn", "mscn_joinkey", "mstn"]:
        featurizer = get_featurizer(trainqs, valqs, testqs, evalqs)
    else:
        featurizer = None

    alg = get_alg(args.alg, cfg)

    eval_fns = []
    for efn in args.eval_fns.split(","):
        eval_fns.append(get_eval_fn(efn))

    # from Adversarial_weight import learn_weights

    # main_logger.info("Starting adversarial weight learning...")
    # start_weight_time = time.time()
    # result = learn_weights(
    #     source_queries=trainqs,
    #     target_queries=valqs,   # or testqs / one eval workload
    #     featurizer=featurizer,
    #     batch_size=128,
    #     epochs=100,
    #     lr=1e-3,
    #     n_disc_steps=5,
    # )
    # weight_time = time.time() - start_weight_time
    # main_logger.info(f"Adversarial weight learning completed in {weight_time:.2f} seconds")

    # weights = result["weights"]
    # main_logger.info(f"Result feature stats: {result['feature_stats']}")

    use_new_discriminator_train = bool(cfg["model"].get("use_new_discriminator_train", 0))
    use_generator_adversarial_train = bool(cfg["model"].get("use_generator_adversarial_train", 0))
    use_dann_train = bool(cfg["model"].get("use_dann_train", 0))
    selected_adv_modes = (
        int(use_new_discriminator_train)
        + int(use_generator_adversarial_train)
        + int(use_dann_train)
    )
    if selected_adv_modes > 1:
        raise ValueError(
            "Only one adversarial training mode can be enabled at a time: "
            "use_new_discriminator_train, use_generator_adversarial_train, use_dann_train."
        )

    use_alt_adversarial_train = (
        use_new_discriminator_train or use_generator_adversarial_train or use_dann_train
    )

    if use_alt_adversarial_train:
        main_logger.info(
            "Using in-model adversarial training path: skipping standalone "
            "feature-space discriminator weighting."
        )
        disc_weights = None
        mscn_evalqs = evalqs
        mscn_eval_qdirs = eval_qdirs
    else:
        disc_weights, mscn_evalqs, mscn_eval_qdirs = prepare_discriminator_weights(
            trainqs=trainqs,
            evalqs=evalqs,
            eval_qdirs=eval_qdirs,
            featurizer=featurizer,
        )

    if args.load_model is not None:
        alg.featurizer = featurizer
        dummy_ds = QueryDataset(testqs[:1], featurizer,
                    load_query_together=alg.load_query_together,
                    load_padded_mscn_feats=getattr(alg, 'load_padded_mscn_feats', False),
                    max_num_tables=-1)
        alg.load_model(args.load_model, sample=dummy_ds[0])

    elif cfg["model"]["eval_epoch"] < cfg["model"]["max_epochs"]:
        if use_generator_adversarial_train:
            if not hasattr(alg, "train_with_latent_generator"):
                raise RuntimeError(
                    f"{args.alg} does not support train_with_latent_generator"
                )
            alg.train_with_latent_generator(
                trainqs,
                valqs=valqs,
                testqs=testqs,
                evalqs=mscn_evalqs,
                eval_qdirs=mscn_eval_qdirs,
                featurizer=featurizer,
                result_dir=args.result_dir,
                adv_weights=disc_weights,
                adv_weight_level="dataset",
                subplan_mask=train_subplan_mask,
                val_subplan_mask=val_subplan_mask,
                test_subplan_mask=test_subplan_mask,
            )
        elif use_dann_train:
            if not hasattr(alg, "train_with_dann"):
                raise RuntimeError(
                    f"{args.alg} does not support train_with_dann"
                )
            alg.train_with_dann(
                trainqs,
                valqs=valqs,
                testqs=testqs,
                evalqs=mscn_evalqs,
                eval_qdirs=mscn_eval_qdirs,
                featurizer=featurizer,
                result_dir=args.result_dir,
                adv_weights=disc_weights,
                adv_weight_level="dataset",
                subplan_mask=train_subplan_mask,
                val_subplan_mask=val_subplan_mask,
                test_subplan_mask=test_subplan_mask,
            )
        elif use_new_discriminator_train:
            if not hasattr(alg, "train_with_new_discriminator"):
                raise RuntimeError(
                    f"{args.alg} does not support train_with_new_discriminator"
                )
            alg.train_with_new_discriminator(
                trainqs,
                valqs=valqs,
                testqs=testqs,
                evalqs=mscn_evalqs,
                eval_qdirs=mscn_eval_qdirs,
                featurizer=featurizer,
                result_dir=args.result_dir,
                adv_weights=disc_weights,
                adv_weight_level="dataset",
                subplan_mask=train_subplan_mask,
                val_subplan_mask=val_subplan_mask,
                test_subplan_mask=test_subplan_mask,
            )
        else:
            alg.train(trainqs, valqs=valqs, testqs=testqs, evalqs = mscn_evalqs,
                    eval_qdirs = mscn_eval_qdirs, featurizer=featurizer,
                    result_dir=args.result_dir,
                    adv_weights=disc_weights, adv_weight_level="dataset",
                    subplan_mask=train_subplan_mask,
                    val_subplan_mask=val_subplan_mask,
                    test_subplan_mask=test_subplan_mask)
    else:
        if use_generator_adversarial_train:
            if not hasattr(alg, "train_with_latent_generator"):
                raise RuntimeError(
                    f"{args.alg} does not support train_with_latent_generator"
                )
            alg.train_with_latent_generator(
                trainqs,
                valqs=valqs,
                testqs=None,
                evalqs=mscn_evalqs,
                eval_qdirs=mscn_eval_qdirs,
                featurizer=featurizer,
                result_dir=args.result_dir,
                adv_weights=disc_weights,
                adv_weight_level="dataset",
                subplan_mask=train_subplan_mask,
                val_subplan_mask=val_subplan_mask,
            )
        elif use_dann_train:
            if not hasattr(alg, "train_with_dann"):
                raise RuntimeError(
                    f"{args.alg} does not support train_with_dann"
                )
            alg.train_with_dann(
                trainqs,
                valqs=valqs,
                testqs=None,
                evalqs=mscn_evalqs,
                eval_qdirs=mscn_eval_qdirs,
                featurizer=featurizer,
                result_dir=args.result_dir,
                adv_weights=disc_weights,
                adv_weight_level="dataset",
                subplan_mask=train_subplan_mask,
                val_subplan_mask=val_subplan_mask,
            )
        elif use_new_discriminator_train:
            if not hasattr(alg, "train_with_new_discriminator"):
                raise RuntimeError(
                    f"{args.alg} does not support train_with_new_discriminator"
                )
            alg.train_with_new_discriminator(
                trainqs,
                valqs=valqs,
                testqs=None,
                evalqs=mscn_evalqs,
                eval_qdirs=mscn_eval_qdirs,
                featurizer=featurizer,
                result_dir=args.result_dir,
                adv_weights=disc_weights,
                adv_weight_level="dataset",
                subplan_mask=train_subplan_mask,
                val_subplan_mask=val_subplan_mask,
            )
        else:
            alg.train(trainqs, valqs=valqs, testqs=None, evalqs = mscn_evalqs,
                    eval_qdirs = mscn_eval_qdirs, featurizer=featurizer,
                    result_dir=args.result_dir,
                    adv_weights=disc_weights, adv_weight_level="dataset",
                    subplan_mask=train_subplan_mask,
                    val_subplan_mask=val_subplan_mask)

    start_time = time.time()
    # subplan_mask here is the same mask actually used for training, so
    # the reported train QError reflects the same (deduped/leakage-
    # cleaned) subqueries the model was trained on, not the full,
    # possibly-duplicated set.
    eval_alg(alg, eval_fns, trainqs, cfg, "train", featurizer=featurizer,
            subplan_mask=train_subplan_mask)
    execution_time = time.time() - start_time
    print(f"{args.alg} Evaluation time on train set: {execution_time:.2f} seconds")

    # if len(valqs) > 0:
    #     start_time = time.time()
    #     eval_alg(alg, eval_fns, valqs, cfg, "val", featurizer=featurizer,
    #             subplan_mask=val_subplan_mask)
    #     execution_time = time.time() - start_time
    #     print(f"{args.alg} Evaluation time on val set: {execution_time:.2f} seconds")

    if len(testqs) > 0:
        start_time = time.time()
        print(' ----------- Evaluation time on test set starts -----------')
        eval_alg(alg, eval_fns, testqs, cfg, "test", featurizer=featurizer,
                subplan_mask=test_subplan_mask)
        execution_time = time.time() - start_time
        print(f"{args.alg} Evaluation time on test set: {execution_time:.2f} seconds")
        print(' ----------- Evaluation time on test set ends -----------')

    if len(mscn_evalqs) > 0 and len(mscn_evalqs[0]) > 0:
        for ei, evalq in enumerate(mscn_evalqs):
            start_time = time.time()
            eval_dir_name = os.path.basename(os.path.normpath(mscn_eval_qdirs[ei]))
            if not eval_dir_name:
                eval_dir_name = f"eval_{ei}"

            evalq_mask = None
            if args.remove_duplicate_subqueries:
                evalq_mask, _ = remove_duplicate_subqueries(evalq, split_name=eval_dir_name)

            eval_alg(alg, eval_fns, evalq, cfg, eval_dir_name, featurizer=featurizer,
                    subplan_mask=evalq_mask)
            execution_time = time.time() - start_time
            print(f"Evaluation time on held-out eval set {ei}: {execution_time:.2f} seconds")
            del evalq[:]

def read_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False,
            default="configs/config.yaml")
    parser.add_argument("--alg", type=str, required=False,
            default="mscn")

    parser.add_argument("--regen_featstats", type=int, required=False,
            default=0)
    parser.add_argument("--save_featstats", type=int, required=False,
            default=0)
    parser.add_argument("--use_saved_feats", type=int, required=False,
            default=1)

    # logging arguments
    parser.add_argument("--wandb_tags", type=str, required=False,
        default=None, help="additional tags for wandb logs")

    parser.add_argument("--result_dir", type=str, required=False,
            default="./results")
    parser.add_argument("--eval_fns", type=str, required=False,
            default="ppc,qerr")

    parser.add_argument("--learn_residual", type=int, required=False,
                        default=0)
    parser.add_argument("--load_model", type=str, required=False,
            default=None,
            help="If specified, load the model from the given path")
    parser.add_argument("--disc_holdout_frac", type=float, required=False,
            default=0.5,
            help="Fraction of target/eval queries held out for MSCN evaluation")
    parser.add_argument("--use_discriminator", type=int, required=False,
            default=0,
            help="Set to 1 to train/use discriminator weights, 0 for uniform weights")
    parser.add_argument("--disc_epochs", type=int, required=False,
            default=10)
    parser.add_argument("--disc_batch_size", type=int, required=False,
            default=128)
    parser.add_argument("--disc_lr", type=float, required=False,
            default=1e-3)
    parser.add_argument("--random_seed", type=int, required=False,
            default=42)
    parser.add_argument("--undersample_seed", type=int, required=False,
            default=42,
            help="Optional seed for undersampling (defaults to --random_seed)")

    parser.add_argument("--remove_duplicate_subqueries", type=int, required=False,
            default=0,
            help="Set to 1 to remove duplicate subquery nodes (same tables/joins/"
                 "predicates, seen in more than one qrep) within each of train/val/"
                 "test independently. Default 0 keeps the original duplicated data.")
    parser.add_argument("--detect_leakage", type=int, required=False,
            default=0,
            help="Set to 1 to log subquery overlap between train/val/test without "
                 "modifying the data.")
    parser.add_argument("--remove_leakage", type=int, required=False,
            default=0,
            help="Set to 1 to remove subqueries that leak across train/val/test, "
                 "keeping each one only in the highest-priority split (train > val "
                 "> test). Implies --detect_leakage reporting.")
    return parser.parse_args()

if __name__ == "__main__":
    args = read_flags()
    main()
