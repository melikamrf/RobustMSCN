from pathlib import Path
import pickle as pkl
import re
import ast
from collections import Counter
import pandas as pd
import argparse
import os
import yaml

from query_representation.utils import get_query_splits

TYPE_TO_OP = {
    "eq": "=",
    "lt": "<",
    "gt": ">",
    "lte": "<=",
    "gte": ">=",
    "neq": "!=",
}

JOIN_RE = re.compile(r"^\s*([A-Za-z_][\w]*)\.(\w+)\s*=\s*([A-Za-z_][\w]*)\.(\w+)\s*$")

def parse_join_condition(cond):
    if not cond:
        return None
    match = JOIN_RE.match(cond)
    if match:
        return (match.group(1), match.group(2), match.group(3), match.group(4))
    return (cond.strip(),)

def extract_op_from_pred_str(pred_str):
    if not pred_str:
        return None
    for op in ["<=", ">=", "!=", "=", "<", ">"]:
        if op in pred_str:
            return op
    match = re.search(r"\s+(LIKE|IN|BETWEEN)\s+", pred_str, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def normalize_pred_value(pred_val, pred_str, op):
    if isinstance(pred_val, (list, tuple)):
        if len(pred_val) == 0:
            return None
        if len(pred_val) == 1:
            return pred_val[0]
        if len(pred_val) == 2:
            left, right = pred_val[0], pred_val[1]
            if left is None and right is None:
                return None
            if left is None and right is not None:
                return right
            if right is None and left is not None:
                return left
        return tuple(pred_val)
    if pred_val is not None:
        return pred_val
    if pred_str and op:
        match = re.search(rf"\s*{re.escape(op)}\s*(.+)$", pred_str)
        if match:
            raw = match.group(1).strip()
            try:
                return ast.literal_eval(raw)
            except Exception:
                return raw
    return None

def build_predicate_tuple(alias_hint, pred_col, pred_type, pred_val, pred_str):
    alias = alias_hint
    col = pred_col
    if pred_col and "." in pred_col:
        alias, col = pred_col.split(".", 1)
    elif pred_str:
        match = re.match(r"\s*([A-Za-z_][\w]*)\.(\w+)", pred_str)
        if match:
            alias, col = match.group(1), match.group(2)
    op = extract_op_from_pred_str(pred_str) or TYPE_TO_OP.get(pred_type, pred_type or "?")
    value = normalize_pred_value(pred_val, pred_str, op)
    return (alias, col, op, value)

def _is_nx_graph(obj):
    return callable(getattr(obj, "nodes", None)) and callable(getattr(obj, "edges", None))

def _get_join_nodes(join_graph):
    if _is_nx_graph(join_graph):
        return list(join_graph.nodes(data=True))
    nodes = join_graph.get("nodes", [])
    return [(node.get("id"), node) for node in nodes if node.get("id")]

def _get_subset_ids(subset_graph):
    if _is_nx_graph(subset_graph):
        return list(subset_graph.nodes())
    nodes = subset_graph.get("nodes", [])
    return [node.get("id") for node in nodes if node.get("id")]

def extract_subquery_rows(query_dict):
    join_graph = query_dict.get("join_graph", {})
    subset_graph = query_dict.get("subset_graph", {})
    join_nodes = _get_join_nodes(join_graph)
    alias_to_node = {alias: data for alias, data in join_nodes if alias}
    adjacency = join_graph.get("adjacency", []) if not _is_nx_graph(join_graph) else None

    rows = []
    for subquery_id in _get_subset_ids(subset_graph):
        if not subquery_id:
            continue
        if isinstance(subquery_id, (list, tuple)):
            subquery_aliases = set(subquery_id)
        else:
            subquery_aliases = {subquery_id}
        tables = sorted(subquery_aliases)

        joins_set = set()
        if _is_nx_graph(join_graph):
            for left_alias, right_alias, data in join_graph.edges(data=True):
                if left_alias not in subquery_aliases or right_alias not in subquery_aliases:
                    continue
                join_tuple = parse_join_condition(data.get("join_condition"))
                if join_tuple:
                    joins_set.add(join_tuple)
        else:
            for source_node, neighbors in zip(join_graph.get("nodes", []), adjacency):
                source_alias = source_node.get("id")
                if source_alias not in subquery_aliases:
                    continue
                for neighbor in neighbors:
                    target_alias = neighbor.get("id")
                    if target_alias not in subquery_aliases:
                        continue
                    join_cond = neighbor.get("join_condition")
                    join_tuple = parse_join_condition(join_cond)
                    if join_tuple:
                        joins_set.add(join_tuple)
        joins = sorted(joins_set)

        predicates = []
        for alias in sorted(subquery_aliases):
            node = alias_to_node.get(alias)
            if not node:
                continue
            pred_strs = node.get("predicates", [])
            pred_cols = node.get("pred_cols", [])
            pred_types = node.get("pred_types", [])
            pred_vals = node.get("pred_vals", [])
            for idx, pred_str in enumerate(pred_strs):
                pred_col = pred_cols[idx] if idx < len(pred_cols) else ""
                pred_type = pred_types[idx] if idx < len(pred_types) else ""
                pred_val = pred_vals[idx] if idx < len(pred_vals) else None
                predicates.append(build_predicate_tuple(alias, pred_col, pred_type, pred_val, pred_str))

        rows.append({
            "tables": tables,
            "joins": joins,
            "predicates": predicates,
            "subquery_id": subquery_id,
        })
    return rows

def freeze(value):
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return tuple((k, freeze(v)) for k, v in items)
    if isinstance(value, set):
        return tuple(sorted((freeze(v) for v in value), key=str))
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(freeze(v) for v in value)
    try:
        hash(value)
        return value
    except TypeError:
        return repr(value)
    return value

def print_distribution(label, items):
    total = len(items)
    counts = Counter(freeze(item) for item in items)
    print(f"{label.upper()}: ")
    print(f"  Total unique values: {len(counts)}")
    print("  Distribution:")
    for key, count in sorted(counts.items(), key=lambda x: (-x[1], str(x[0]))):
        prob = count / total if total else 0.0
        display_key = [] if len(key) == 0 else list(key)
        print(f"    {display_key}: {prob:.4f} ({prob*100:.2f}%)")

def compute_distributions_for_files(label, qfns, max_files=None, verbose=False):
    if max_files:
        qfns = qfns[:max_files]

    subquery_rows = []
    for qfn in qfns:
        if verbose:
            print(f"Processing file: {qfn}")
        with open(qfn, "rb") as f:
            query_dict = pkl.load(f)
        subquery_rows.extend(extract_subquery_rows(query_dict))

    workload_df = pd.DataFrame(subquery_rows)
    print("\n" + "=" * 80)
    print(f"{label.upper()} SPLIT")
    print("=" * 80)
    print(f"Loaded {len(qfns)} files, {len(workload_df)} subqueries")

    tables_entries = workload_df["tables"].tolist() if not workload_df.empty else []
    joins_entries = workload_df["joins"].tolist() if not workload_df.empty else []

    # For predicate distribution, emit one predicate per list (or [] when none).
    predicate_entries = []
    for preds in (workload_df["predicates"].tolist() if not workload_df.empty else []):
        if preds:
            for pred in preds:
                predicate_entries.append([pred])
        else:
            predicate_entries.append([])

    print_distribution("TABLES", tables_entries)
    print_distribution("JOINS", joins_entries)
    print_distribution("PREDICATES", predicate_entries)

    return workload_df

def compute_distributions_for_qreps(label, qreps, max_qreps=None):
    if max_qreps:
        qreps = qreps[:max_qreps]

    subquery_rows = []
    for qrep in qreps:
        subquery_rows.extend(extract_subquery_rows(qrep))

    workload_df = pd.DataFrame(subquery_rows)
    print("\n" + "=" * 80)
    print(f"{label.upper()} SPLIT")
    print("=" * 80)
    print(f"Loaded {len(qreps)} queries, {len(workload_df)} subqueries")

    tables_entries = workload_df["tables"].tolist() if not workload_df.empty else []
    joins_entries = workload_df["joins"].tolist() if not workload_df.empty else []

    predicate_entries = []
    for preds in (workload_df["predicates"].tolist() if not workload_df.empty else []):
        if preds:
            for pred in preds:
                predicate_entries.append([pred])
        else:
            predicate_entries.append([])

    print_distribution("TABLES", tables_entries)
    print_distribution("JOINS", joins_entries)
    print_distribution("PREDICATES", predicate_entries)

    return workload_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="configs/config.yaml",
        help="Path to YAML config used for query splits",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        required=False,
        default=None,
        help="Optional limit per split",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each pickle file path while processing",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_qfns, test_qfns, val_qfns, eval_qfns = get_query_splits(cfg["data"])
    train_df = compute_distributions_for_files("train", train_qfns, 10, args.verbose)
    # compute_distributions_for_files("val", val_qfns, args.max_files, args.verbose)
    # compute_distributions_for_files("test", test_qfns, args.max_files, args.verbose)
    train_df.to_csv("train_subquery_distribution.csv", index=False)
    eval_dirs = cfg["data"].get("eval_query_dir", "").split(",")
    for idx, qfns in enumerate(eval_qfns):
        raw_label = eval_dirs[idx] if idx < len(eval_dirs) else f"eval_{idx}"
        label = os.path.basename(os.path.normpath(raw_label)) or f"eval_{idx}"
        compute_distributions_for_files(f"eval_{label}", qfns, args.max_files, args.verbose)

if __name__ == "__main__":
    main()