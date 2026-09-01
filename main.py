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
from cardinality_estimation.seeding import DEFAULT_TRAIN_SEED, seed_everything
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
import ast
import re
import glob
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


def _mask_to_allowed_sets(subplan_mask):
    """
    Converts a subplan_mask (list, aligned with a qreps list, of lists of
    `list(node)` to keep) into the list-of-sets-of-node-tuples form the
    reporting/distribution helpers expect. None (no mask) stays None, meaning
    "use every subset_graph node".
    """
    if subplan_mask is None:
        return None
    return [{tuple(node) for node in per_qrep} for per_qrep in subplan_mask]


def calc_datasets_jsd(trainqs, valqs, testqs, evalqs, eval_qdirs,
        result_dir=None, save_dist_tables=True, dist_table_format="pkl",
        train_mask=None, val_mask=None, test_mask=None, eval_masks=None):
    """
    Compute JSD between training and eval query distributions.
    Uses the whole dataset only; it does not stratify by join count.

    train_mask/val_mask/test_mask/eval_masks are optional per-split
    subplan_masks; when given, only the subqueries each mask keeps are
    counted (so the JSD reflects the actual split, and -- critically for
    subquery-level --train_csv splits -- we don't materialize every
    template's full subset_graph for every split, which OOMs).
    """
    split_results = {}

    split_results["train"] = compute_distributions_for_qreps(
        "train_all", trainqs, return_df=True, num_joins=None,
        allowed_nodes_per_qrep=_mask_to_allowed_sets(train_mask)
    )
    split_results["val"] = compute_distributions_for_qreps(
        "val_all", valqs, return_df=True, num_joins=None,
        allowed_nodes_per_qrep=_mask_to_allowed_sets(val_mask)
    )
    split_results["test"] = compute_distributions_for_qreps(
        "test_all", testqs, return_df=True, num_joins=None,
        allowed_nodes_per_qrep=_mask_to_allowed_sets(test_mask)
    )

    for idx, evalq in enumerate(evalqs):
        raw_label = eval_qdirs[idx] if idx < len(eval_qdirs) else f"eval_{idx}"
        label = os.path.basename(os.path.normpath(raw_label)) or f"eval_{idx}"
        eval_mask = eval_masks[idx] if (eval_masks is not None and idx < len(eval_masks)) else None
        split_results[label] = compute_distributions_for_qreps(
            f"{label}_all", evalq, return_df=True, num_joins=None,
            allowed_nodes_per_qrep=_mask_to_allowed_sets(eval_mask)
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


def _build_subquery_level_dataframe(qreps, allowed_nodes_per_qrep=None):
    rows = []
    for qi, qrep in enumerate(qreps):
        subset_graph = qrep["subset_graph"]
        allowed = None
        if allowed_nodes_per_qrep is not None and qi < len(allowed_nodes_per_qrep):
            allowed = allowed_nodes_per_qrep[qi]
        for subquery_row in extract_subquery_rows(qrep, num_joins=None, restrict_to=allowed):
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


def save_subquery_split_dataframes(trainqs, valqs, testqs, evalqs, eval_qdirs, result_dir, query_dir,
        train_mask=None, val_mask=None, test_mask=None, eval_masks=None):
    if result_dir is None:
        return {}

    export_dir = os.path.join(result_dir, "query_level_dataframes")
    os.makedirs(export_dir, exist_ok=True)

    train_prefix = _dataset_file_prefix(query_dir)

    # (qreps, file-prefix, subplan_mask): the mask restricts the exported
    # rows to the subqueries actually in that split (see calc_datasets_jsd
    # for why this matters in subquery-level --train_csv mode).
    split_to_qreps = {
        "train": (trainqs, train_prefix, train_mask),
        "val": (valqs, train_prefix, val_mask),
        "test": (testqs, train_prefix, test_mask),
    }
    for idx, evalq in enumerate(evalqs):
        raw_label = eval_qdirs[idx] if idx < len(eval_qdirs) else f"eval_{idx}"
        split_name = os.path.basename(os.path.normpath(raw_label)) or f"eval_{idx}"
        eval_mask = eval_masks[idx] if (eval_masks is not None and idx < len(eval_masks)) else None
        split_to_qreps[split_name] = (evalq, split_name, eval_mask)

    saved_paths = {}
    for split_name, value in split_to_qreps.items():
        qreps, prefix, mask = value
        if qreps is None:
            continue
        df = _build_subquery_level_dataframe(qreps, _mask_to_allowed_sets(mask))
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


def _qrep_subquery_signatures(qrep, allowed_nodes=None):
    """
    Maps each subquery node (subset_graph node) in qrep to a hashable
    signature of its (tables, joins, predicates) -- two subqueries with the
    same signature are the same subquery, regardless of which top-level
    query/pkl file they came from.

    @allowed_nodes: optional set of node tuples to restrict to (see
    extract_subquery_rows's restrict_to) -- pass the split's existing
    subplan_mask (converted via _mask_to_allowed_sets) whenever qrep's
    subset_graph can contain more nodes than are actually selected for
    this split, e.g. --train_csv/--eval_csv row-level splitting. Without
    it, duplicate/leakage detection would scan every combinatorial
    subplan of the template instead of just the ones in use, wildly
    over-counting both.
    """
    sigs = {}
    for row in extract_subquery_rows(qrep, num_joins=None, restrict_to=allowed_nodes):
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


def _aliases_from_csv_row(join_tables_str, predicates_str):
    """
    Reconstructs the set of table aliases a subquery-level CSV row (as
    produced by GRASP's read_query_file_batched, e.g.
    join_tables="(('ci','movie_id','t','id'), ...)") touches, so the row can
    be matched back to a subset_graph node (itself just a tuple of sorted
    aliases).
    """
    joins = ast.literal_eval(join_tables_str) if isinstance(join_tables_str, str) and join_tables_str else ()
    preds = ast.literal_eval(predicates_str) if isinstance(predicates_str, str) and predicates_str else ()
    aliases = set()
    for jt in joins:
        aliases.add(jt[0])
        aliases.add(jt[2])
    for pred in preds:
        aliases.add(pred[0])
    return tuple(sorted(aliases))


def _qrep_alias_canonical_map(qrep):
    """
    Mirrors the self-join alias collapsing GRASP's CEB utilities apply
    (drop a trailing digit from an alias, e.g. 't1' -> 't', only when that
    base table has just one alias in the whole query) so CSV rows produced
    by that pipeline can still be matched against this qrep's (uncollapsed)
    subset_graph node keys.
    """
    aliases = list(qrep["join_graph"].nodes())
    groups = defaultdict(list)
    for alias in aliases:
        groups[re.sub(r'\d+$', '', alias)].append(alias)
    return {alias: (base if len(members) == 1 else alias)
            for base, members in groups.items() for alias in members}


def _match_csv_rows_to_nodes(rows_df, path_index):
    """
    Groups subquery-level CSV rows (template,join_tables,predicates,...) by
    their `template` (.pkl basename), loads each distinct template's qrep
    exactly once (via load_qdata, so it gets the same JOB-parsing fixes/
    cardinality filtering as every other code path), and resolves every row
    to its subset_graph node.

    Returns {template_basename: (qrep, [row_node, ...])} -- one node per
    matched row, in row order (duplicate rows produce duplicate entries;
    callers that need unique nodes should dedupe). Templates not found under
    query_dir, filtered out entirely by load_qdata, or rows that can't be
    matched to any node, are skipped with a printed count.
    """
    matches = {}
    num_unmatched = 0
    num_missing_templates = 0

    for template, group in rows_df.groupby("template"):
        fullpath = path_index.get(template)
        if fullpath is None:
            num_missing_templates += 1
            continue

        loaded = load_qdata([fullpath])
        if len(loaded) == 0:
            num_missing_templates += 1
            continue
        qrep = loaded[0]

        node_set = set(qrep["subset_graph"].nodes())
        node_set.discard(SOURCE_NODE)
        canonical_to_node = None

        row_nodes = []
        for _, row in group.iterrows():
            row_aliases = _aliases_from_csv_row(row["join_tables"], row["predicates"])

            if row_aliases in node_set:
                row_nodes.append(row_aliases)
                continue

            if canonical_to_node is None:
                canonical_map = _qrep_alias_canonical_map(qrep)
                canonical_to_node = {
                    tuple(sorted(canonical_map.get(a, a) for a in node)): node
                    for node in node_set
                }

            matched_node = canonical_to_node.get(row_aliases)
            if matched_node is not None:
                row_nodes.append(matched_node)
            else:
                num_unmatched += 1

        if row_nodes:
            matches[template] = (qrep, row_nodes)

    if num_unmatched:
        print(f"_match_csv_rows_to_nodes: {num_unmatched} CSV rows did not "
              f"match any subset_graph node, skipped")
    if num_missing_templates:
        print(f"_match_csv_rows_to_nodes: {num_missing_templates} CSV "
              f"template(s) not found under query_dir (or filtered out by "
              f"load_qdata), skipped")

    return matches


def _index_query_dir_by_basename(query_dir):
    fns = glob.glob(os.path.join(query_dir, "*", "*.pkl"))
    return {os.path.basename(fn): fn for fn in fns}


def _assert_csv_matched_something(csv_path, query_dir, rows_df, path_index, matches):
    """
    Fails fast with an actionable message instead of letting an empty
    `matches` silently propagate into a confusing sklearn train_test_split
    error ("n_samples=0") several calls downstream. The most common cause is
    query_dir pointing at a different workload's directory than whatever
    CEB-imdb tree the CSV's `template` filenames were generated from.
    """
    if matches:
        return
    csv_templates = set(rows_df["template"].unique())
    sample_csv = sorted(csv_templates)[:5]
    sample_dir = sorted(path_index.keys())[:5]
    raise ValueError(
        f"No rows in {csv_path} could be matched to any file under "
        f"query_dir={query_dir!r}: {len(csv_templates)} distinct CSV "
        f"template(s) referenced, {len(path_index)} .pkl files found "
        f"under query_dir. Sample CSV templates: {sample_csv}. Sample "
        f"filenames found under query_dir: {sample_dir}. This almost "
        f"always means query_dir points at a different workload directory "
        f"than the one the CSV was generated from -- check data.query_dir "
        f"in your config."
    )


def _matches_to_qreps_mask(matches):
    qreps = []
    subplan_mask = []
    for qrep, nodes in matches.values():
        qreps.append(qrep)
        subplan_mask.append([list(node) for node in dict.fromkeys(nodes)])
    return qreps, subplan_mask


def load_eval_set_from_csv(csv_path, query_dir):
    """
    Loads a subquery-level eval CSV (template,join_tables,predicates,...) as
    held-out eval group(s) -- no further train/val/test split.

    If the CSV has a `seen_or_unseen` column (GRASP's test CSV does), its rows
    are split into two independent groups, "eval_seen" and "eval_unseen", so
    each gets its own per-epoch q-error curve (the label is per-subquery, not
    per-file: the same .pkl can contribute seen subplans to one group and
    unseen subplans to the other). Otherwise a single "eval" group is used.

    Returns [(label, qreps, subplan_mask), ...].
    """
    rows_df = pd.read_csv(csv_path)
    path_index = _index_query_dir_by_basename(query_dir)

    groups = []
    if "seen_or_unseen" in rows_df.columns:
        for label in ("seen", "unseen"):
            sub_df = rows_df[rows_df["seen_or_unseen"] == label]
            if len(sub_df) == 0:
                continue
            matches = _match_csv_rows_to_nodes(sub_df, path_index)
            qreps, mask = _matches_to_qreps_mask(matches)
            if qreps:
                groups.append(("eval_" + label, qreps, mask))
        if not groups:
            # nothing matched at all -> raise the actionable query_dir error
            _assert_csv_matched_something(csv_path, query_dir, rows_df, path_index, {})
    else:
        matches = _match_csv_rows_to_nodes(rows_df, path_index)
        _assert_csv_matched_something(csv_path, query_dir, rows_df, path_index, matches)
        qreps, mask = _matches_to_qreps_mask(matches)
        groups.append(("eval", qreps, mask))

    return groups


def load_train_pool_from_csv(csv_path, query_dir, val_size, test_size, seed):
    """
    Loads a subquery-level train-pool CSV and splits its *rows* (not whole
    query files -- the same .pkl template can legitimately contribute nodes
    to more than one split) into train/val/test, mirroring the val-then-test
    order get_query_splits uses for train_test_split_kind == "query"
    (query_representation/utils.py:282-296).

    Each distinct template is loaded once for matching, then deep-copied
    once per split it contributes to, so trainqs/valqs/testqs never share a
    mutable qrep object (relevant for --learn_residual, which mutates
    subset_graph cardinalities in place per split).

    Returns (trainqs, train_mask, valqs, val_mask, testqs, test_mask).
    """
    from sklearn.model_selection import train_test_split

    rows_df = pd.read_csv(csv_path)
    path_index = _index_query_dir_by_basename(query_dir)
    matches = _match_csv_rows_to_nodes(rows_df, path_index)
    _assert_csv_matched_something(csv_path, query_dir, rows_df, path_index, matches)

    pairs = sorted({
        (template, node)
        for template, (_, nodes) in matches.items()
        for node in nodes
    })

    if val_size and val_size > 0.0:
        val_pairs, pool_pairs = train_test_split(pairs, test_size=1 - val_size,
                random_state=seed)
    else:
        val_pairs, pool_pairs = [], pairs

    if test_size and test_size > 0.0:
        train_pairs, test_pairs = train_test_split(pool_pairs,
                test_size=test_size, random_state=seed)
    else:
        train_pairs, test_pairs = pool_pairs, []

    def _build_split(split_pairs):
        by_template = defaultdict(list)
        for template, node in split_pairs:
            by_template[template].append(node)

        qreps = []
        mask = []
        for template, nodes in by_template.items():
            base_qrep, _ = matches[template]
            qreps.append(copy.deepcopy(base_qrep))
            mask.append([list(node) for node in nodes])
        return qreps, mask

    trainqs, train_mask = _build_split(train_pairs)
    valqs, val_mask = _build_split(val_pairs)
    testqs, test_mask = _build_split(test_pairs)

    return trainqs, train_mask, valqs, val_mask, testqs, test_mask


def load_csv_split_data(train_csv, eval_csv, query_dir, val_size, test_size, seed):
    """
    Matches the train/eval CSVs to subplan nodes and builds the split.
    Returns (trainqs, train_mask, valqs, val_mask, testqs, test_mask,
    eval_groups), where eval_groups is [(label, qreps, mask), ...] (split into
    eval_seen/eval_unseen when the eval CSV carries a seen_or_unseen column).
    """
    trainqs, train_mask, valqs, val_mask, testqs, test_mask = \
            load_train_pool_from_csv(train_csv, query_dir, val_size, test_size, seed)
    eval_groups = load_eval_set_from_csv(eval_csv, query_dir)

    return (trainqs, train_mask, valqs, val_mask, testqs, test_mask, eval_groups)


def remove_duplicate_subqueries(qreps, split_name="split", allowed_mask=None):
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

    @allowed_mask: optional pre-existing subplan_mask restricting which
    nodes of each qrep are actually "in" this split already (e.g. from
    --train_csv/--eval_csv row-level loading, where the same template's
    complete, un-split subset_graph is shared across splits and only
    specific rows/nodes belong to any given split). When given, duplicate
    detection only considers those nodes instead of a qrep's full
    subset_graph -- without it, this would scan every combinatorial
    subplan of the template, wildly over-counting duplicates that were
    never actually selected into any split to begin with.

    Returns (mask, num_excluded): mask is a list, same length/order as
    qreps, of lists of `list(node)` -- pass it as `subplan_mask` (train),
    `val_subplan_mask`, or `test_subplan_mask` to alg.train()/
    train_with_dann()/train_with_new_discriminator().
    """
    allowed_sets = _mask_to_allowed_sets(allowed_mask)
    seen = {}
    excluded_by_qrep = defaultdict(set)
    num_excluded = 0
    for qi, qrep in enumerate(qreps):
        allowed = allowed_sets[qi] if allowed_sets is not None and qi < len(allowed_sets) else None
        for subquery_id, sig in _qrep_subquery_signatures(qrep, allowed_nodes=allowed).items():
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


def detect_leakage(named_splits, remove=False, priority=None, allowed_masks=None):
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

    @allowed_masks: optional {split_name: subplan_mask or None}
    restricting which nodes of each split's qreps are actually eligible
    already (see remove_duplicate_subqueries's allowed_mask for why this
    matters, e.g. --train_csv/--eval_csv row-level splitting). Without
    it, splits that share the same underlying template qreps (just
    different rows/nodes selected per split) would appear to leak
    almost entirely into each other, since their *complete* subset_graphs
    are identical regardless of which specific rows each split actually
    uses.

    Returns (report, masks): report is a dict with leaked signature
    counts and pairwise overlap counts; masks is {split_name: mask or
    None} (None meaning "no filtering needed for this split").
    """
    split_names = [name for name, _ in named_splits]
    if priority is None:
        priority = split_names
    priority_rank = {name: i for i, name in enumerate(priority)}
    allowed_masks = allowed_masks or {}

    # signature -> {split_name: [(qrep_idx, subquery_id), ...]}
    sig_to_splits = defaultdict(lambda: defaultdict(list))
    for split_name, qreps in named_splits:
        allowed_sets = _mask_to_allowed_sets(allowed_masks.get(split_name))
        for qi, qrep in enumerate(qreps):
            allowed = allowed_sets[qi] if allowed_sets is not None and qi < len(allowed_sets) else None
            for subquery_id, sig in _qrep_subquery_signatures(qrep, allowed_nodes=allowed).items():
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


def prepare_discriminator_weights(trainqs, evalqs, eval_qdirs, featurizer,
        train_seed=DEFAULT_TRAIN_SEED):
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
        train_seed=train_seed,
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

def extract_cardinalities(qreps, ests, subplan_mask=None):
    """
    Extract cardinalities for the subplans that were actually evaluated.

    ``alg.test(..., subplan_mask=...)`` returns estimates only for the masked
    nodes.  The qreps themselves still retain their complete subset graphs, so
    iterating those graphs here would emit one row per *unselected* subplan
    with an empty model estimate.  Use the same mask for reporting; without a
    mask, use the estimate keys, which also excludes SOURCE_NODE.
    """
    card_data = []

    for idx, (qrep, est) in enumerate(zip(qreps, ests)):
        if "subset_graph" not in qrep:
            continue

        sg = qrep["subset_graph"]
        if subplan_mask is not None and idx < len(subplan_mask):
            nodes = (tuple(node) for node in subplan_mask[idx])
        else:
            nodes = est.keys()

        for node in nodes:
            if node == SOURCE_NODE or node not in sg:
                continue

            data = sg.nodes[node]
            true_card = data["cardinality"]["actual"]
            postgres_card = data["cardinality"]["expected"]
            model_card = est.get(node)

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
                "model_estimated": model_card,
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

def _mask_sample_count(qreps, mask):
    """
    How many subplans would actually become dataset rows: every
    subset_graph node when mask is None, else only the kept ones.
    SOURCE_NODE is never a training sample (QueryDataset._get_query_
    features_nodes drops it), so it's excluded either way.
    """
    if mask is None:
        return sum(len([n for n in q["subset_graph"].nodes() if n != SOURCE_NODE])
                   for q in qreps)
    return sum(len([n for n in per_qrep if tuple(n) != SOURCE_NODE])
               for per_qrep in mask)


def _materialize_full_mask(qreps):
    """
    The explicit "keep every node" subplan_mask, so subquery-level
    subsampling has something concrete to shrink when a split had no mask
    yet (None == unrestricted).
    """
    return [[list(node) for node in q["subset_graph"].nodes() if node != SOURCE_NODE]
            for q in qreps]


def _resolve_train_size(train_size, pool_size):
    """
    @train_size: a fraction of the pool when in (0, 1], an absolute sample
    count when > 1. Note 1 therefore means "all of it", not "one sample".

    Returns the target size, or None when no subsampling is needed
    (train_size unset/<=0, or >= the whole pool).
    """
    if train_size is None or train_size <= 0:
        return None
    if train_size <= 1.0:
        target = int(round(train_size * pool_size))
    else:
        target = int(round(train_size))
    target = max(1, min(target, pool_size))
    return None if target >= pool_size else target


def resolve_train_size_level(level, from_csv):
    """
    "auto" -> "subquery" in --train_csv mode (that split is already
    subquery-level), "query" otherwise (the get_query_splits path splits
    over .pkl files).
    """
    if level == "auto":
        return "subquery" if from_csv else "query"
    if level not in ("query", "subquery"):
        raise ValueError(f"unknown train_size level: {level!r} "
                         f"(expected auto/query/subquery)")
    return level


def subsample_train_split(trainqs, train_mask, train_size,
        level="auto", seed=42, from_csv=False):
    """
    Shrinks ONLY the training split to @train_size; val/test/eval are left
    exactly as they were, so a sweep over training sizes varies one thing
    and the resulting numbers stay comparable across runs.

    @train_size: fraction of the current training pool in (0, 1], or an
    absolute number of samples when > 1. None/<=0/1.0 keeps everything.
    @level: what a "sample" is when counting/drawing.
      "query":    whole query files (qreps) are dropped. The natural unit
                  for the get_query_splits path, whose split is itself
                  over .pkl files.
      "subquery": individual subplans are dropped, spread over all qreps.
                  The natural unit for --train_csv (that split is already
                  subquery-level), and the apples-to-apples unit when
                  comparing the two split options, since it makes a given
                  train_size mean the same number of training rows in both.
      "auto":     "subquery" in --train_csv mode, "query" otherwise.
    @train_mask: the split's current subplan_mask (None == unrestricted).
    It is subsampled alongside trainqs so the two stay positionally
    aligned, which is what QueryDataset indexes it by.

    Call this AFTER --remove_duplicate_subqueries/--remove_leakage have
    narrowed train_mask, so the requested size counts samples that are
    actually trained on.

    Returns (trainqs, train_mask).
    """
    level = resolve_train_size_level(level, from_csv)

    num_queries = len(trainqs)
    num_subqueries = _mask_sample_count(trainqs, train_mask)

    pool_size = num_queries if level == "query" else num_subqueries
    target = _resolve_train_size(train_size, pool_size)
    if target is None:
        main_logger.info(
            f"subsample_train_split(train_size={train_size}): using the full "
            f"training split ({num_queries} queries, {num_subqueries} "
            f"subquery samples)"
        )
        return trainqs, train_mask

    rng = np.random.RandomState(seed)

    if level == "query":
        idxs = sorted(rng.choice(num_queries, size=target, replace=False))
        new_trainqs = [trainqs[i] for i in idxs]
        new_mask = [train_mask[i] for i in idxs] if train_mask is not None else None
    else:
        mask = train_mask if train_mask is not None else _materialize_full_mask(trainqs)
        flat = [(qi, node) for qi, per_qrep in enumerate(mask)
                for node in per_qrep if tuple(node) != SOURCE_NODE]
        # Sort before drawing: we sample INDICES into this list, so its order
        # decides which subqueries get picked. _intersect_masks builds its
        # lists by iterating a set of alias tuples, and CPython randomizes
        # string hashing per process (PYTHONHASHSEED), so without this the
        # same seed picks a different training subset in every process
        # whenever --remove_duplicate_subqueries/--remove_leakage are on.
        flat.sort(key=lambda qi_node: (qi_node[0], tuple(qi_node[1])))
        idxs = rng.choice(len(flat), size=target, replace=False)
        kept = defaultdict(list)
        for i in sorted(idxs):
            qi, node = flat[i]
            kept[qi].append(list(node))

        # qreps that kept no samples are dropped outright -- they would only
        # cost featurizer/bitmap work for rows that no longer exist. Every
        # retained qrep keeps its full subset_graph, so featurizer lookups a
        # kept subplan makes against its own graph still resolve.
        new_trainqs = []
        new_mask = []
        for qi, qrep in enumerate(trainqs):
            if qi in kept:
                new_trainqs.append(qrep)
                new_mask.append(kept[qi])

    main_logger.info(
        f"subsample_train_split(level={level}, train_size={train_size}, "
        f"seed={seed}): {num_queries} -> {len(new_trainqs)} train queries, "
        f"{num_subqueries} -> {_mask_sample_count(new_trainqs, new_mask)} "
        f"train subquery samples (val/test/eval unchanged)"
    )

    return new_trainqs, new_mask


def save_split_sizes(result_dir, trainqs, train_mask, valqs, val_mask,
        testqs, test_mask, evalqs, eval_masks, extra=None):
    """
    Dumps the per-split query/subquery counts actually used by this run to
    <result_dir>/train_split_sizes.json, so a training-size sweep can
    report "x training subqueries -> y q-error" without re-deriving the
    sizes or scraping stdout.
    """
    if result_dir is None:
        return None

    os.makedirs(result_dir, exist_ok=True)
    sizes = dict(extra or {})
    sizes.update({
        "num_train_queries": len(trainqs),
        "num_train_subqueries": _mask_sample_count(trainqs, train_mask),
        "num_val_queries": len(valqs),
        "num_val_subqueries": _mask_sample_count(valqs, val_mask),
        "num_test_queries": len(testqs),
        "num_test_subqueries": _mask_sample_count(testqs, test_mask),
        "num_eval_queries": sum(len(eq) for eq in evalqs),
        "num_eval_subqueries": sum(
            _mask_sample_count(
                eq,
                eval_masks[i] if (eval_masks is not None and i < len(eval_masks)) else None)
            for i, eq in enumerate(evalqs)),
    })

    out_path = os.path.join(result_dir, "train_split_sizes.json")
    with open(out_path, "w") as f:
        json.dump(sizes, f, indent=2)
    print(f"Saved split sizes to {out_path}")
    return out_path


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

    # Pass the mask into test() so featurization/prediction is restricted to
    # the subplans actually in this split, instead of every node of every
    # query's full graph. Without this, evaluating train/test/eval qreps
    # (which carry full graphs in subquery-level --train_csv/--eval_csv mode)
    # featurizes ~all subplans -> OOM. The later _filter_ests_by_mask call
    # then becomes a no-op (ests already only contain masked nodes).
    # Note: this returns estimates only for masked subplans, which is what
    # per-subquery eval fns (e.g. qerr) need; plan-cost (ppc) eval, which
    # needs every node, is not compatible with a masked eval set.
    ests = alg.test(qreps, subplan_mask=subplan_mask)
    rdir = None
    if args.result_dir is not None:
        rdir = os.path.join(args.result_dir, exp_name)
        make_dir(rdir)
        # print("Going to store results at: ", rdir)
        args_fn = os.path.join(rdir, "cfg.json")
        with open(args_fn, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=4)
            
    df = extract_cardinalities(qreps, ests, subplan_mask=subplan_mask)
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
                'join_key_stat_names', 'join_key_stat_tmps',
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
            tmp_fn = featdata_fn + ".tmp"
            with open(tmp_fn, "w") as f:
                json.dump(featdata, f)
            os.replace(tmp_fn, featdata_fn)
            print("Saved featurizer db stats to:", featdata_fn)
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

def resolve_train_seed(args, cfg):
    """
    --train_seed wins; otherwise model.train_seed from the config if the
    config pins one; otherwise the default. Lets a config fix a seed for
    "the canonical run of this experiment" while a sweep still overrides it
    per run from the CLI.
    """
    if args.train_seed is not None:
        return int(args.train_seed)

    cfg_seed = cfg.get("model", {}).get("train_seed", None)
    if cfg_seed is not None:
        return int(cfg_seed)

    return DEFAULT_TRAIN_SEED

def main():
    global args,cfg

    with open(args.config) as f:
        cfg = yaml.safe_load(f.read())

    print(yaml.dump(cfg, default_flow_style=False))

    # --train_seed governs everything downstream of the splits: weight init,
    # DataLoader shuffling, dropout. Data seeds (data.seed,
    # data.diff_templates_seed, --train_size_seed, --random_seed) are separate
    # on purpose -- fix those and vary --train_seed to measure training noise
    # for one architecture; see cardinality_estimation/seeding.py.
    train_seed = resolve_train_seed(args, cfg)
    seed_everything(train_seed, strict=bool(args.strict_determinism))
    main_logger.info(
        f"Seeded run: train_seed={train_seed}, "
        f"strict_determinism={bool(args.strict_determinism)}, "
        f"random_seed={args.random_seed}, "
        f"data.seed={cfg['data']['seed']}, "
        f"data.diff_templates_seed={cfg['data']['diff_templates_seed']}")

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
        # vars(args)["train_seed"] is None when the seed came from the config,
        # so log the resolved value -- runs are grouped/compared by it.
        wandbcfg.update({"train_seed": train_seed})
        # additional config tags
        wandb_tags = ["1a"]
        if args.wandb_tags is not None:
            wandb_tags += args.wandb_tags.split(",")
        wandb.init(project="ceb", config=wandbcfg,
                tags=wandb_tags)

    # subplan_mask passed to alg.train()/train_with_dann()/
    # train_with_new_discriminator(): which subset_graph nodes to
    # actually turn into training/eval samples. None means "no
    # filtering" (the original, possibly-duplicated/leaky data).
    if args.train_csv and args.eval_csv:
        trainqs, train_subplan_mask, valqs, val_subplan_mask, \
                testqs, test_subplan_mask, csv_eval_groups = \
                load_csv_split_data(
                        args.train_csv, args.eval_csv, cfg["data"]["query_dir"],
                        cfg["data"]["val_size"], cfg["data"]["test_size"],
                        cfg["data"]["diff_templates_seed"])
        eval_qfns = None
    else:
        train_subplan_mask = None
        val_subplan_mask = None
        test_subplan_mask = None
        train_qfns, test_qfns, val_qfns, eval_qfns = get_query_splits(cfg["data"])
        trainqs = load_qdata(train_qfns)
        # Note: can be quite memory intensive to load them all; might want to just
        # keep around the qfns and load them as needed
        valqs = load_qdata(val_qfns)
        testqs = load_qdata(test_qfns)

    if args.remove_duplicate_subqueries:
        # Compose onto (not overwrite) whatever train/val/test_subplan_mask
        # were initialized to above -- in --train_csv/--eval_csv mode those
        # already restrict samples to the CSV-selected subqueries, and
        # deduping must narrow that further rather than replace it. Passing
        # them in as allowed_mask also makes duplicate detection itself only
        # consider those already-selected nodes, not every combinatorial
        # subplan of the underlying template (see remove_duplicate_subqueries
        # docstring -- without this, CSV row-level splitting massively
        # over-counts duplicates since a qrep's subset_graph is the
        # template's full, un-split one).
        dedup_train_mask, _ = remove_duplicate_subqueries(trainqs, split_name="train",
                allowed_mask=train_subplan_mask)
        dedup_val_mask, _ = remove_duplicate_subqueries(valqs, split_name="val",
                allowed_mask=val_subplan_mask)
        dedup_test_mask, _ = remove_duplicate_subqueries(testqs, split_name="test",
                allowed_mask=test_subplan_mask)
        train_subplan_mask = _intersect_masks(trainqs, train_subplan_mask, dedup_train_mask)
        val_subplan_mask = _intersect_masks(valqs, val_subplan_mask, dedup_val_mask)
        test_subplan_mask = _intersect_masks(testqs, test_subplan_mask, dedup_test_mask)

    if args.remove_leakage or args.detect_leakage:
        # Same reasoning as above: restrict leakage scanning to whatever
        # each split's mask already narrowed it to (CSV selection, and/or
        # the dedup pass just above), not each qrep's full subset_graph.
        _, leakage_masks = detect_leakage(
                [("train", trainqs), ("val", valqs), ("test", testqs)],
                remove=bool(args.remove_leakage),
                allowed_masks={
                    "train": train_subplan_mask,
                    "val": val_subplan_mask,
                    "test": test_subplan_mask,
                })
        if args.remove_leakage:
            train_subplan_mask = _intersect_masks(trainqs, train_subplan_mask,
                    leakage_masks.get("train"))
            val_subplan_mask = _intersect_masks(valqs, val_subplan_mask,
                    leakage_masks.get("val"))
            test_subplan_mask = _intersect_masks(testqs, test_subplan_mask,
                    leakage_masks.get("test"))

    # Training-size experiments: shrink train only (val/test/eval keep the
    # sizes data.val_size/test_size gave them). Done after the dedup/
    # leakage masks above so --train_size counts samples actually trained on.
    train_size_seed = args.train_size_seed if args.train_size_seed is not None \
            else args.random_seed
    train_size_level = resolve_train_size_level(args.train_size_level,
            from_csv=bool(args.train_csv and args.eval_csv))
    trainqs, train_subplan_mask = subsample_train_split(
            trainqs, train_subplan_mask, args.train_size,
            level=train_size_level,
            seed=train_size_seed)

    if args.learn_residual:
        trainqs = update_labels(trainqs)
        valqs = update_labels(valqs)
        testqs = update_labels(testqs)
    
    if args.train_csv and args.eval_csv:
        # One eval group per (label, qreps, mask) from the eval CSV. With a
        # seen_or_unseen column this is [eval_seen, eval_unseen], so each gets
        # its own per-epoch q-error curve; the short labels also keep
        # save_subquery_split_dataframes' f"{label}_{label}.csv" filenames
        # well within Windows' path length limit.
        eval_qdirs = [label for label, _, _ in csv_eval_groups]
        evalqs = []
        eval_masks = []
        for _, group_qreps, group_mask in csv_eval_groups:
            if args.learn_residual:
                group_qreps = update_labels(group_qreps)
            evalqs.append(group_qreps)
            eval_masks.append(group_mask)
    else:
        eval_masks = None
        eval_qdirs = cfg["data"]["eval_query_dir"].split(",")

        evalqs = []
        for eval_qfn in eval_qfns:
            temp_evalqs = load_qdata(eval_qfn)
            if args.learn_residual:
                temp_evalqs = update_labels(temp_evalqs)

            evalqs.append(temp_evalqs)

    eqs = [len(eq) for eq in evalqs]
    eval_summary = ", ".join("{}={}".format(name or "eval", n)
            for name, n in zip(eval_qdirs, eqs)) or "none"
    print("Selected Queries: {} train, {} test, {} val | eval: {}".format(
            len(trainqs), len(testqs), len(valqs), eval_summary))

    save_split_sizes(
        args.result_dir,
        trainqs, train_subplan_mask,
        valqs, val_subplan_mask,
        testqs, test_subplan_mask,
        evalqs, eval_masks,
        extra={
            "split_source": "csv" if (args.train_csv and args.eval_csv) else "query_splits",
            "train_size": args.train_size,
            "train_size_level": train_size_level,
            "train_size_seed": train_size_seed,
            "train_seed": train_seed,
            "random_seed": args.random_seed,
            "data_seed": cfg["data"]["seed"],
            "diff_templates_seed": cfg["data"]["diff_templates_seed"],
            "strict_determinism": bool(args.strict_determinism),
            "val_size": cfg["data"]["val_size"],
            "test_size": cfg["data"]["test_size"],
            # these run before the subsample, so they change what the
            # training pool a given --train_size is a fraction OF
            "remove_duplicate_subqueries": args.remove_duplicate_subqueries,
            "detect_leakage": args.detect_leakage,
            "remove_leakage": args.remove_leakage,
            "alg": args.alg,
            "config": args.config,
        },
    )

    if args.result_dir is not None:
        save_subquery_split_dataframes(
            trainqs,
            valqs,
            testqs,
            evalqs,
            eval_qdirs,
            args.result_dir,
            cfg["data"]["query_dir"],
            train_mask=train_subplan_mask,
            val_mask=val_subplan_mask,
            test_mask=test_subplan_mask,
            eval_masks=eval_masks,
        )


    # Undersample source (trainqs) to match target (evalqs) size
    # Use undersample_train_queries() to match query count
    # OR use undersample_train_queries_by_subquery_count() to match total subquery count
    # trainqs = undersample_train_queries_by_subquery_count(trainqs, evalqs, seed=args.undersample_seed)
    #trainqs = undersample_train_queries(trainqs, eqs, seed=args.undersample_seed)

    dataset_jsd_df = calc_datasets_jsd(trainqs, valqs, testqs, evalqs, eval_qdirs, args.result_dir,
            train_mask=train_subplan_mask, val_mask=val_subplan_mask,
            test_mask=test_subplan_mask, eval_masks=eval_masks)
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

    alg = get_alg(args.alg, cfg, train_seed=train_seed)

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
            train_seed=train_seed,
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
                train_size=args.train_size,
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
                train_size=args.train_size,
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
                train_size=args.train_size,
                subplan_mask=train_subplan_mask,
                val_subplan_mask=val_subplan_mask,
                test_subplan_mask=test_subplan_mask,
            )
        else:
            # eval_masks aligns with evalqs; only pass it through when
            # mscn_evalqs is still that same list (the discriminator-holdout
            # path can resample it, which would misalign the masks).
            train_eval_masks = eval_masks if mscn_evalqs is evalqs else None
            alg.train(trainqs, valqs=valqs, testqs=testqs, evalqs = mscn_evalqs,
                    eval_qdirs = mscn_eval_qdirs, featurizer=featurizer,
                    result_dir=args.result_dir,
                    adv_weights=disc_weights, adv_weight_level="dataset",
                    train_size=args.train_size,
                    subplan_mask=train_subplan_mask,
                    val_subplan_mask=val_subplan_mask,
                    test_subplan_mask=test_subplan_mask,
                    eval_subplan_masks=train_eval_masks)
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
                train_size=args.train_size,
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
                train_size=args.train_size,
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
                train_size=args.train_size,
                subplan_mask=train_subplan_mask,
                val_subplan_mask=val_subplan_mask,
            )
        else:
            alg.train(trainqs, valqs=valqs, testqs=None, evalqs = mscn_evalqs,
                    eval_qdirs = mscn_eval_qdirs, featurizer=featurizer,
                    result_dir=args.result_dir,
                    adv_weights=disc_weights, adv_weight_level="dataset",
                    train_size=args.train_size,
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

            # eval_masks[ei] (from --eval_csv) is only positionally valid
            # against evalqs itself; if the discriminator holdout path
            # (prepare_discriminator_weights) resampled it into
            # heldout_evalqs, mscn_evalqs is no longer evalqs and we can't
            # line the masks back up, so they're skipped in that case.
            if mscn_evalqs is evalqs and eval_masks is not None and ei < len(eval_masks):
                evalq_mask = eval_masks[ei]
            else:
                evalq_mask = None
            if args.remove_duplicate_subqueries:
                # allowed_mask=evalq_mask: restrict duplicate detection to
                # whatever this eval group's rows already are (from
                # --eval_csv), not the underlying template's full
                # subset_graph -- same reasoning as the train/val/test
                # dedup calls above.
                dedup_mask, _ = remove_duplicate_subqueries(evalq, split_name=eval_dir_name,
                        allowed_mask=evalq_mask)
                evalq_mask = _intersect_masks(evalq, evalq_mask, dedup_mask)

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
            default=42,
            help="DATA seed: discriminator target holdout split, and the "
                 "default for --train_size_seed/--undersample_seed. Decides "
                 "which queries land in which split, so hold it FIXED when "
                 "comparing architectures. See --train_seed for the knob to "
                 "vary across repeats.")
    parser.add_argument("--undersample_seed", type=int, required=False,
            default=42,
            help="Optional seed for undersampling (defaults to --random_seed)")
    parser.add_argument("--train_seed", type=int, required=False,
            default=None,
            help="TRAINING seed: weight init, DataLoader shuffle order, "
                 "dropout -- everything downstream of the data splits. This "
                 "is the seed to vary (e.g. 1,2,3,4,5) to get repeats of the "
                 "same architecture on identical data, so architectures can "
                 "be compared as mean +/- std instead of single runs. "
                 "Defaults to model.train_seed from the config, else "
                 f"{DEFAULT_TRAIN_SEED}.")
    parser.add_argument("--strict_determinism", type=int, required=False,
            default=0,
            help="Set to 1 to force deterministic CUDA kernels "
                 "(cudnn.deterministic, use_deterministic_algorithms). Costs "
                 "throughput and warns on ops with no deterministic "
                 "implementation; --train_seed alone already gives "
                 "run-to-run reproducibility on the same hardware and "
                 "library versions.")

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

    parser.add_argument("--train_size", type=float, required=False,
            default=None,
            help="Shrink the TRAINING split only, leaving val/test/eval at "
                 "the sizes data.val_size/test_size produced (so runs at "
                 "different training sizes stay comparable). A value in "
                 "(0, 1] is a fraction of the training pool, a value > 1 is "
                 "an absolute number of samples. Default (unset) or 1.0 "
                 "trains on everything. Works in both split modes: the "
                 "get_query_splits path and --train_csv/--eval_csv.")
    parser.add_argument("--train_size_level", type=str, required=False,
            default="auto", choices=["auto", "query", "subquery"],
            help="What --train_size counts. 'query': whole query files. "
                 "'subquery': individual subplans across all query files. "
                 "'auto' (default) uses subquery in --train_csv mode and "
                 "query otherwise. Force 'subquery' for both modes when "
                 "comparing them against each other, so a given "
                 "--train_size means the same number of training rows.")
    parser.add_argument("--train_size_seed", type=int, required=False,
            default=None,
            help="Seed for the --train_size subsample (defaults to "
                 "--random_seed). Vary it to get repeats at the same size.")

    parser.add_argument("--train_csv", type=str, required=False,
            default=None,
            help="Subquery-level CSV (template,join_tables,predicates,pgcard,"
                 "true_card,model_card) defining the train pool. If set "
                 "together with --eval_csv, this replaces get_query_splits: "
                 "the pool's rows are split into train/val/test (using "
                 "data.val_size/test_size/diff_templates_seed) at the "
                 "individual-subquery level, so the same .pkl file can "
                 "contribute different subqueries to different splits.")
    parser.add_argument("--eval_csv", type=str, required=False,
            default=None,
            help="Subquery-level CSV (same schema as --train_csv) used "
                 "entirely as the held-out eval/target set, with no further "
                 "split. Must be set together with --train_csv.")
    return parser.parse_args()

if __name__ == "__main__":
    args = read_flags()
    main()
