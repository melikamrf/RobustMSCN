"""
Compute JSD (joins and predicates) between two workload dataframes of the form:
    template, join_tables, predicates, pgcard, true_card, model_card
where join_tables/predicates hold tuples, e.g.
    join_tables: (('ci', 'movie_id', 't', 'id'), ...)
    predicates:  (('t', 'production_year', 'lt', (1950, 1990)), ...)
If loaded from CSV, those columns come in stringified; they're parsed automatically.

Reuses build_distribution / compute_jsd from compute_dist_table.py.
"""
import argparse
import ast

import pandas as pd

from compute_dist_table import build_distribution, compute_jsd, compute_jsd_contributions


def _parse_tuple_cell(value):
    if isinstance(value, str):
        value = value.strip()
        return ast.literal_eval(value) if value else ()
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ()
    return value


def _prepare_workload_df(df):
    df = df.copy()
    df["join_tables"] = df["join_tables"].apply(_parse_tuple_cell)
    df["predicates"] = df["predicates"].apply(_parse_tuple_cell)
    return df


def build_join_distribution(df):
    joins_entries = [jt for jt in df["join_tables"].tolist() if jt]
    _, _, probs = build_distribution(joins_entries)
    return probs


def build_predicate_distribution(df):
    predicate_entries = []
    for preds in df["predicates"].tolist():
        for pred in preds:
            predicate_entries.append([pred])
    _, _, probs = build_distribution(predicate_entries)
    return probs


def compute_workload_jsd(df_a, df_b, top_n=15):
    df_a = _prepare_workload_df(df_a)
    df_b = _prepare_workload_df(df_b)

    join_probs_a = build_join_distribution(df_a)
    join_probs_b = build_join_distribution(df_b)
    jsd_joins = compute_jsd(join_probs_a, join_probs_b)

    pred_probs_a = build_predicate_distribution(df_a)
    pred_probs_b = build_predicate_distribution(df_b)
    jsd_predicates = compute_jsd(pred_probs_a, pred_probs_b)

    print(f"Rows: A={len(df_a)}, B={len(df_b)}")
    print(f"JSD (join patterns):  {jsd_joins:.6f}")
    print(f"JSD (predicates):     {jsd_predicates:.6f}")

    if top_n:
        print(f"\nTop {top_n} join patterns driving divergence:")
        for c in compute_jsd_contributions(join_probs_a, join_probs_b, top_n=top_n):
            print(f"  [{c['only_in']:>6}] contrib={c['jsd_contribution']:.5f}  "
                  f"A={c['source_prob']:.4f} B={c['target_prob']:.4f}  {c['pattern']}")

        print(f"\nTop {top_n} predicates driving divergence:")
        for c in compute_jsd_contributions(pred_probs_a, pred_probs_b, top_n=top_n):
            print(f"  [{c['only_in']:>6}] contrib={c['jsd_contribution']:.5f}  "
                  f"A={c['source_prob']:.4f} B={c['target_prob']:.4f}  {c['pattern']}")

    return jsd_joins, jsd_predicates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_a", type=str, help="First workload CSV (e.g. train)")
    parser.add_argument("path_b", type=str, help="Second workload CSV (e.g. test)")
    parser.add_argument("--top-n", type=int, default=15, help="Show top-N divergence contributors (0 to skip)")
    args = parser.parse_args()

    df_a = pd.read_csv(args.path_a)
    df_b = pd.read_csv(args.path_b)
    
    compute_workload_jsd(df_a, df_b, top_n=args.top_n)


if __name__ == "__main__":
    main()
