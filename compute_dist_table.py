"""
Join support is limited to simple equi-joins of the form "alias1.col1 = alias2.col2"

instead of [] for single table queries, we can either only add the table name, 
or nothing since probably the element is in the predicate list (but not always). 
so probably should add single table and use empty string for left table in the join.
we assume there is no left or right join, so the order of joins does not matter.

This is a simple version where the table assumes joins and predicates are independent, which is not always the case. 
#TODO: we can handle this later.

"""
from pathlib import Path
import pickle as pkl
import re
import ast
from collections import Counter
import pandas as pd
import argparse
import os
import yaml
import numpy as np
from collections import defaultdict

from query_representation.query import load_qrep
from query_representation.utils import get_query_splits
from cardinality_estimation.dataset import load_qdata

TYPE_TO_OP = {
    "eq": "=",
    "lt": "<",
    "gt": ">",
    "lte": "<=",
    "gte": ">=",
    "neq": "!=",
}

# match simple join conditions of the form "alias1.col1 = alias2.col2"
# TODO: Is there any joins that is not supported by this regex? We can expand it as needed.

JOIN_RE = re.compile(r"^\s*([A-Za-z_][\w]*)\.(\w+)\s*=\s*([A-Za-z_][\w]*)\.(\w+)\s*$")

def parse_join_condition(cond):
    "returns (alias1, col1, alias2, col2) if cond is a simple equi-join, else returns (cond,) or None if empty"
    if not cond:
        return None
    match = JOIN_RE.match(cond)
    if match:
        return (match.group(1), match.group(2), match.group(3), match.group(4))
    return (cond.strip(),)

def canonicalize_join_tuple(join_tuple):
    if not join_tuple:
        return join_tuple
    if len(join_tuple) != 4:
        return join_tuple
    left = (join_tuple[0], join_tuple[1])
    right = (join_tuple[2], join_tuple[3])
    if right < left:
        left, right = right, left
    return (left[0], left[1], right[0], right[1])

def canonicalize_tables(subquery_id):
    if isinstance(subquery_id, (list, tuple, set, frozenset)):
        return sorted(subquery_id, key=str)
    return [subquery_id]

# def get_spj_qrep(qfns):
#     spj_qreps = []
    
#     for qfn in qfns:

#         with open(qfn, "rb") as f:
#             qrep = pkl.load(f)
        
#         if len(qrep['join_graph']['graph']) > 0:
#             print(f"SPJ query: {qrep['sql']}")
#             if qrep.get("aggr_cmd"):
#                 spj_qreps.append(qrep)
#                 print(f"SPJ query: {qrep['sql']}")
#             else:
#                 print(f"Skipping non-SPJ query: {qrep['sql']}")
#     return spj_qreps


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


def extract_subquery_rows(qrep, num_joins=None):

    join_graph = qrep["join_graph"]
    subset_graph = qrep["subset_graph"]

    rows = []
    joins = []
    for subquery_id in subset_graph.nodes():
        if not subquery_id:
            continue
        tables = canonicalize_tables(subquery_id)

        subplan_graph = join_graph.subgraph(subquery_id)

        joins_set = set()
        for left_alias, right_alias, data in subplan_graph.edges(data=True):
            join_tuple = parse_join_condition(data.get("join_condition"))
            if join_tuple:
                joins_set.add(canonicalize_join_tuple(join_tuple))

        if len(joins_set) == num_joins or num_joins is None:
            joins = sorted(joins_set)

            predicates = []
            for alias, node in subplan_graph.nodes(data=True):
                pred_strs = node.get("predicates", [])
                pred_cols = node.get("pred_cols", [])
                pred_types = node.get("pred_types", [])
                pred_vals = node.get("pred_vals", [])
                for idx, pred_str in enumerate(pred_strs):
                    pred_col = pred_cols[idx] if idx < len(pred_cols) else ""
                    pred_type = pred_types[idx] if idx < len(pred_types) else ""
                    pred_val = pred_vals[idx] if idx < len(pred_vals) else None
                    predicates.append(build_predicate_tuple(alias, pred_col, pred_type, pred_val, pred_str))

            predicates.sort(key=str)

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

def build_distribution(items):
    total = len(items)
    counts = Counter(freeze(item) for item in items)
    probs = {k: (v / total if total else 0.0) for k, v in counts.items()}
    return total, counts, probs

def print_distribution_from_counts(label, total, counts):
    print(f"{label.upper()}: ")
    print(f"  Total unique values: {len(counts)}")
    print("  Distribution:")
    for key, count in sorted(counts.items(), key=lambda x: (-x[1], str(x[0]))):
        prob = count / total if total else 0.0
        display_key = [] if len(key) == 0 else list(key)
        print(f"    {display_key}: {prob:.4f} ({prob*100:.2f}%)")

def print_distribution(label, items):
    total, counts, _ = build_distribution(items)
    print_distribution_from_counts(label, total, counts)


def compute_distributions_for_qreps(label, qreps, max_qreps=None, return_df=False, num_joins=None):
    if max_qreps:
        qreps = qreps[:max_qreps]

    subquery_rows = []
    for qrep in qreps:
        subquery_rows.extend(extract_subquery_rows(qrep, num_joins))

    workload_df = pd.DataFrame(subquery_rows)
    print("\n" + "=" * 80)
    print(f"{label.upper()} SPLIT")
    print("=" * 80)
    print(f"Loaded {len(qreps)} queries, {len(workload_df)} subqueries")

    tables_entries = workload_df["tables"].tolist() if not workload_df.empty else []
    joins_entries = [
        joins for joins in (workload_df["joins"].tolist() if not workload_df.empty else [])
        if joins
    ]

    predicate_entries = []
    for preds in (workload_df["predicates"].tolist() if not workload_df.empty else []):
        if not preds:
            continue
        for pred in preds:
            if pred is not None:
                predicate_entries.append([pred])

    tables_total, tables_counts, tables_probs = build_distribution(tables_entries)
    joins_total, joins_counts, joins_probs = build_distribution(joins_entries)
    preds_total, preds_counts, preds_probs = build_distribution(predicate_entries)

    # print_distribution_from_counts("TABLES", tables_total, tables_counts)
    # print_distribution_from_counts("JOINS", joins_total, joins_counts)
    # print_distribution_from_counts("PREDICATES", preds_total, preds_counts)

    dist = {
        "tables": tables_probs,
        "joins": joins_probs,
        "predicates": preds_probs,
    }
    if return_df:
        return dist, workload_df
    return dist


def compute_jsd(source_probs: dict, target_probs: dict, k: float = 1e-10) -> float:
    """
    Compute Jensen-Shannon Divergence between source and target distributions.
    
    Args:
        source_probs: dict mapping pattern (as tuple or string) -> probability
        target_probs: dict mapping pattern (as tuple or string) -> probability
        k: smoothing constant for zero probabilities
    
    Returns:
        JSD value in [0, 1] (using log base 2)
    """
    # Step 1: Get union of all patterns
    all_patterns = set(source_probs.keys()) | set(target_probs.keys())
    V = len(all_patterns)
    
    # Step 2: Build aligned vectors with smoothing
    P = np.array([source_probs.get(p, 0) + k for p in all_patterns])
    Q = np.array([target_probs.get(p, 0) + k for p in all_patterns])
    
    # Step 3: Renormalize after smoothing
    P = P / P.sum()
    Q = Q / Q.sum()
    
    # Step 4: Compute mixture
    M = 0.5 * (P + Q)
    
    # Step 5: KL divergences
    def kl(A, B):
        # Only compute where A > 0
        mask = A > 0
        return np.sum(A[mask] * np.log2(A[mask] / B[mask]))
    
    jsd = 0.5 * kl(P, M) + 0.5 * kl(Q, M)
    return jsd


def _build_distribution_from_df(workload_df, field):
    if workload_df.empty:
        return {}

    if field == "predicates":
        predicate_entries = []
        for preds in workload_df["predicates"].tolist():
            if not preds:
                continue
            for pred in preds:
                if pred is not None:
                    predicate_entries.append([pred])
        _, _, probs = build_distribution(predicate_entries)
        return probs

    items = workload_df[field].tolist() if field in workload_df.columns else []
    if field == "joins":
        items = [item for item in items if item]
    _, _, probs = build_distribution(items)
    return probs

def compute_jsd_contributions(source_probs: dict, target_probs: dict, k: float = 1e-10, top_n: int = 20):
    """
    Returns per-pattern JSD contributions, sorted by most divergent.
    Useful for identifying which patterns drive the drift.
    """
    all_patterns = set(source_probs.keys()) | set(target_probs.keys())
    
    P_raw = {p: source_probs.get(p, 0) + k for p in all_patterns}
    Q_raw = {p: target_probs.get(p, 0) + k for p in all_patterns}
    
    P_sum = sum(P_raw.values())
    Q_sum = sum(Q_raw.values())
    
    contributions = []
    for p in all_patterns:
        pi = P_raw[p] / P_sum
        qi = Q_raw[p] / Q_sum
        mi = 0.5 * (pi + qi)
        
        kl_p = pi * np.log2(pi / mi) if pi > 0 else 0
        kl_q = qi * np.log2(qi / mi) if qi > 0 else 0
        contrib = 0.5 * (kl_p + kl_q)
        
        contributions.append({
            'pattern': p,
            'source_prob': pi,
            'target_prob': qi,
            'jsd_contribution': contrib,
            'only_in': 'source' if source_probs.get(p, 0) > 0 and target_probs.get(p, 0) == 0
                       else 'target' if target_probs.get(p, 0) > 0 and source_probs.get(p, 0) == 0
                       else 'both'
        })
    
    contributions.sort(key=lambda x: x['jsd_contribution'], reverse=True)
    return contributions[:top_n]

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
    trainqs = load_qdata(train_qfns)
    valqs = load_qdata(val_qfns)
    testqs = load_qdata(test_qfns)

    train_dist, train_df = compute_distributions_for_qreps(
        "train", trainqs, 1, return_df=True
    )
   
    val_dist, val_df = compute_distributions_for_qreps(
        "val", valqs, None, return_df=True
    )
    test_dist, test_df = compute_distributions_for_qreps(
        "test", testqs, None, return_df=True
    )

    train_df.to_csv("train_subquery_distribution.csv", index=False)
    
    eval_dirs = cfg["data"].get("eval_query_dir", "").split(",")
    
    eval_dists = []
    for idx, qfns in enumerate(eval_qfns):
        raw_label = eval_dirs[idx] if idx < len(eval_dirs) else f"eval_{idx}"
        label = os.path.basename(os.path.normpath(raw_label)) or f"eval_{idx}"

        evalqs = load_qdata(qfns)
        print(len(evalqs))
        eval_dist, eval_df = compute_distributions_for_qreps(
            f"eval_{label}", evalqs, None, return_df=True
        )
        eval_dists.append(eval_dist)
    
    print(compute_jsd(train_dist["joins"], test_dist["joins"]))
    # print(compute_jsd(train_dist["joins"], val_dist["joins"]))
    # if eval_dists:
    #     print(compute_jsd(train_dist["joins"], eval_dists[0]["joins"]))

    # print(compute_jsd(train_dist["predicates"], test_dist["predicates"]))
    # print(compute_jsd(train_dist["predicates"], val_dist["predicates"]))
    # if eval_dists:
    #     print(compute_jsd(train_dist["predicates"], eval_dists[0]["predicates"]))

if __name__ == "__main__":
    main()