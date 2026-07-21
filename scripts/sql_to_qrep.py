
import os
import pdb

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from networkx.readwrite import json_graph
from query_representation.utils import *
from query_representation.query import *

OUTPUT_DIR="./queries/joblight/all_joblight/"
INPUT_FN = "./queries/joblight.sql"
OUTPUT_FN_TMP = "{i}.sql"

make_dir(OUTPUT_DIR)

with open(INPUT_FN, "r") as f:
    data = f.read()
queries = data.split("\n")

for i, sql in enumerate(queries):
    output_fn = OUTPUT_DIR + str(i+1) + ".pkl"
    if "select" not in sql.lower():
        continue
    sql, true_card, pg_card, num = sql.split("||")
    sql = sql.strip(';')

    qrep = parse_sql(sql, None, None, None, None, None,
            compute_ground_truth=False)

    qrep["subset_graph"] = \
            nx.DiGraph(json_graph.adjacency_graph(qrep["subset_graph"]))
    qrep["join_graph"] = json_graph.adjacency_graph(qrep["join_graph"])

    import re
    for node in qrep["join_graph"].nodes():
        qrep["join_graph"].nodes()[node]["pred_cols"] = []
        qrep["join_graph"].nodes()[node]["pred_types"] = []
        qrep["join_graph"].nodes()[node]["pred_vals"] = []

        if "predicates" not in qrep["join_graph"].nodes()[node]:
            continue

        preds = qrep["join_graph"].nodes()[node]["predicates"]
        bounds = {}
        
        def parse_single(pred_str):
            pred_str = pred_str.strip().strip("(").strip(")")
            if " OR " in pred_str.upper():
                parts = re.split(r'\s+OR\s+', pred_str, flags=re.IGNORECASE)
                for part in parts:
                    parse_single(part)
                return
                
            if ">=" in pred_str:
                col, val = pred_str.split(">=")
                col = col.strip()
                val = int(val.strip()) if val.strip().isdigit() else val.strip().strip("'")
                if col not in bounds: bounds[col] = [None, None]
                bounds[col][0] = val
            elif "<=" in pred_str:
                col, val = pred_str.split("<=")
                col = col.strip()
                val = int(val.strip()) if val.strip().isdigit() else val.strip().strip("'")
                if col not in bounds: bounds[col] = [None, None]
                bounds[col][1] = val
            elif ">" in pred_str:
                col, val = pred_str.split(">")
                col = col.strip()
                val = int(val.strip()) if val.strip().isdigit() else val.strip().strip("'")
                if col not in bounds: bounds[col] = [None, None]
                bounds[col][0] = val
            elif "<" in pred_str:
                col, val = pred_str.split("<")
                col = col.strip()
                val = int(val.strip()) if val.strip().isdigit() else val.strip().strip("'")
                if col not in bounds: bounds[col] = [None, None]
                bounds[col][1] = val
            elif "=" in pred_str and "!=" not in pred_str:
                col, val = pred_str.split("=")
                col = col.strip()
                val = int(val.strip()) if val.strip().isdigit() else val.strip().strip("'")
                qrep["join_graph"].nodes()[node]["pred_cols"].append(col)
                qrep["join_graph"].nodes()[node]["pred_types"].append("eq")
                qrep["join_graph"].nodes()[node]["pred_vals"].append({'literal': val})
            elif "!=" in pred_str:
                col, val = pred_str.split("!=")
                col = col.strip()
                val = int(val.strip()) if val.strip().isdigit() else val.strip().strip("'")
                qrep["join_graph"].nodes()[node]["pred_cols"].append(col)
                qrep["join_graph"].nodes()[node]["pred_types"].append("not eq")
                qrep["join_graph"].nodes()[node]["pred_vals"].append({'literal': val})
            elif " IN " in pred_str.upper():
                col, val = re.split(r'\s+IN\s+', pred_str, flags=re.IGNORECASE)
                col = col.strip()
                val = val.strip().strip("()")
                vals = [int(v.strip()) if v.strip().isdigit() else v.strip().strip("'").replace('\n', '').strip() for v in val.split(",")]
                qrep["join_graph"].nodes()[node]["pred_cols"].append(col)
                qrep["join_graph"].nodes()[node]["pred_types"].append("in")
                qrep["join_graph"].nodes()[node]["pred_vals"].append(vals)
            elif " BETWEEN " in pred_str.upper():
                col, vals = re.split(r'\s+BETWEEN\s+', pred_str, flags=re.IGNORECASE)
                v1, v2 = vals.split(" AND ")
                col = col.strip()
                qrep["join_graph"].nodes()[node]["pred_cols"].append(col)
                qrep["join_graph"].nodes()[node]["pred_types"].append("lt")
                qrep["join_graph"].nodes()[node]["pred_vals"].append([int(v1.strip()), int(v2.strip())])
            elif " NOT LIKE " in pred_str.upper():
                col, val = re.split(r'\s+NOT LIKE\s+', pred_str, flags=re.IGNORECASE)
                col = col.strip()
                val = val.strip().strip("'")
                qrep["join_graph"].nodes()[node]["pred_cols"].append(col)
                qrep["join_graph"].nodes()[node]["pred_types"].append("not_like")
                qrep["join_graph"].nodes()[node]["pred_vals"].append([val])
            elif " LIKE " in pred_str.upper():
                col, val = re.split(r'\s+LIKE\s+', pred_str, flags=re.IGNORECASE)
                col = col.strip()
                val = val.strip().strip("'")
                qrep["join_graph"].nodes()[node]["pred_cols"].append(col)
                qrep["join_graph"].nodes()[node]["pred_types"].append("like")
                qrep["join_graph"].nodes()[node]["pred_vals"].append([val])

        for pred_str in preds:
            parse_single(pred_str)
        
        for col, bound in bounds.items():
            qrep["join_graph"].nodes()[node]["pred_cols"].append(col)
            qrep["join_graph"].nodes()[node]["pred_types"].append("lt")
            qrep["join_graph"].nodes()[node]["pred_vals"].append(bound)

    save_qrep(output_fn, qrep)