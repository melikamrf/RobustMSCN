"""
join_bitmap_generator.py

Generates "join bitmaps" as described in Negi et al., "Robust Query Driven
Cardinality Estimation under Changing Workloads" (VLDB 2023), Section 4.3.

WHY THIS DIFFERS FROM A PLAIN "SAMPLE BITMAP" (§2.2 of the paper):
A sample bitmap draws an INDEPENDENT fixed sample per table and applies each
table's filter to its own sample -- it cannot capture correlations across a
join (a classical result: joining two independent 1% samples gives you an
effective ~0.01% sample of the join). A join bitmap instead:
  1. Draws ONE fixed sample of primary-key values (e.g. title.id).
  2. For every foreign table that references that primary key, pulls the
     CORRELATED rows (i.e. WHERE fk_col = ANY(pk_sample)) rather than an
     independent sample of that table.
  3. Applies that table's own filter to its correlated rows.
  4. Intersects the "passing" primary-key sets across all foreign tables
     that hang off the same primary key, producing ONE bitvector per
     primary key group (Fig. 8 / Fig. 6d in the paper; also see §4.5.2,
     "there is no easy way to ... capture correlations across equi-joins
     involving more than one primary key in a single bitmap" -- which is
     why grouping is done PER primary key, not per query).

SCHEMA ASSUMPTION: every table's own primary key column is literally named
"id" (true for the standard IMDb/JOB/CEB schema used in the paper). A join
edge "a.col1 = b.col2" is treated as (primary side = whichever of col1/col2
is literally "id", foreign side = the other). If your schema differs,
adjust `PK_COLUMN_NAMES` below.

CAVEAT: I could not execute this end-to-end against a live Postgres instance
(no DB network access from this environment), so treat this as carefully
reasoned but untested code. Please run it against a small workload first and
sanity-check a few bitmaps (e.g. against a `SELECT COUNT(*)` on the raw
query) before trusting it at scale.

Requires: psycopg2-binary, numpy
"""

import hashlib
import os
import pickle
import re
from dataclasses import dataclass, field

import numpy as np
import psycopg2

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

PK_COLUMN_NAMES = {"id"}  # column names treated as a table's own primary key

DEFAULT_SAMPLE_SIZE = 1000       # size of the fixed primary-key sample
DEFAULT_SEED = 42                # for reproducible sampling
DEFAULT_MAX_FK_ROWS = 30000      # cap on rows pulled per foreign table
                                  # (paper: "at most 1% of the table, which
                                  # is at most 30K on IMDb", §6.1)

CACHE_DIR = "./bitmap_cache"


# --------------------------------------------------------------------------
# Minimal SQL parsing for CEB/JOB-style conjunctive queries
#   SELECT COUNT(*) FROM t1 AS a1, t2 AS a2, ... WHERE c1 AND c2 AND ...
# Not a general SQL parser -- handles equi-join and single-table filter
# conditions (=, !=, <, <=, >, >=, IN (...), LIKE), no OR / no subqueries.
# --------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    tables: dict = field(default_factory=dict)   # alias -> table_name
    joins: list = field(default_factory=list)    # [(alias1, col1, alias2, col2)]
    filters: dict = field(default_factory=dict)  # alias -> [(col, op, value)]


def _split_top_level_and(where_clause: str):
    """Split on ' AND ' that is not inside parens or quotes."""
    conditions, buf, depth, in_quote = [], "", 0, False
    i, n = 0, len(where_clause)
    while i < n:
        c = where_clause[i]
        if c == "'":
            in_quote = not in_quote
            buf += c
            i += 1
            continue
        if not in_quote:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            if depth == 0 and where_clause[i:i + 5].upper() == " AND ":
                conditions.append(buf.strip())
                buf = ""
                i += 5
                continue
        buf += c
        i += 1
    if buf.strip():
        conditions.append(buf.strip())
    return conditions


def _cast_value(raw: str):
    raw = raw.strip()
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


_JOIN_RE = re.compile(r"^\s*(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)\s*$")
_IN_RE = re.compile(r"^\s*(\w+)\.(\w+)\s+IN\s*\((.*)\)\s*$", re.IGNORECASE)
_LIKE_RE = re.compile(r"^\s*(\w+)\.(\w+)\s+(NOT\s+)?LIKE\s+(.+?)\s*$", re.IGNORECASE)
_CMP_RE = re.compile(r"^\s*(\w+)\.(\w+)\s*(=|!=|<>|>=|<=|>|<)\s*(.+?)\s*$")


def _split_top_level_or(clause: str):
    """Same idea as _split_top_level_and, but splits on ' OR '."""
    parts, buf, depth, in_quote = [], "", 0, False
    i, n = 0, len(clause)
    while i < n:
        c = clause[i]
        if c == "'":
            in_quote = not in_quote
            buf += c
            i += 1
            continue
        if not in_quote:
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            if depth == 0 and clause[i:i + 4].upper() == " OR ":
                parts.append(buf.strip())
                buf = ""
                i += 4
                continue
        buf += c
        i += 1
    if buf.strip():
        parts.append(buf.strip())
    return parts


def _classify_single_condition(cond: str):
    """Classifies one non-join, non-OR condition. Returns (alias, col, op, value)."""
    im = _IN_RE.match(cond)
    if im:
        alias, col, values_raw = im.groups()
        return alias, col, "IN", [_cast_value(v) for v in values_raw.split(",")]
    lm = _LIKE_RE.match(cond)
    if lm:
        alias, col, neg, pattern = lm.groups()
        return alias, col, ("NOT LIKE" if neg else "LIKE"), _cast_value(pattern)
    cm = _CMP_RE.match(cond)
    if cm:
        alias, col, op, value = cm.groups()
        return alias, col, op, _cast_value(value)
    return None


def parse_query(sql: str) -> ParsedQuery:
    sql = re.sub(r"\s+", " ", sql.strip().rstrip(";"))
    m = re.search(r"FROM\s+(.*?)\s+WHERE\s+(.*)$", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        raise ValueError(f"Could not find FROM/WHERE in query: {sql!r}")
    from_clause, where_clause = m.group(1), m.group(2)

    parsed = ParsedQuery()
    for table_expr in from_clause.split(","):
        table_expr = table_expr.strip()
        tm = re.match(r"^(\w+)\s+(?:AS\s+)?(\w+)$", table_expr, re.IGNORECASE)
        if not tm:
            raise ValueError(f"Could not parse table expression: {table_expr!r}")
        table, alias = tm.group(1), tm.group(2)
        parsed.tables[alias] = table
        parsed.filters.setdefault(alias, [])

    for cond in _split_top_level_and(where_clause):
        jm = _JOIN_RE.match(cond)
        if jm:
            a1, c1, a2, c2 = jm.groups()
            parsed.joins.append((a1, c1, a2, c2))
            continue

        # Parenthesized OR-group, e.g. "(mc.note LIKE '%A%' OR mc.note LIKE '%B%')"
        paren_m = re.match(r"^\((.*)\)$", cond)
        if paren_m and " OR " in f" {paren_m.group(1)} ":
            sub_conds = _split_top_level_or(paren_m.group(1))
            classified = [_classify_single_condition(c) for c in sub_conds]
            if all(classified) and len({(a, c) for a, c, _, _ in classified}) == 1:
                alias, col = classified[0][0], classified[0][1]
                alternatives = [(op, val) for _, _, op, val in classified]
                parsed.filters.setdefault(alias, []).append((col, "OR", alternatives))
                continue
            raise ValueError(
                f"Unsupported OR-group (must be same alias.col on every branch): {cond!r}"
            )

        single = _classify_single_condition(cond)
        if single:
            alias, col, op, value = single
            parsed.filters.setdefault(alias, []).append((col, op, value))
            continue

        raise ValueError(f"Could not classify condition: {cond!r}")

    return parsed


def group_by_primary_key(parsed: ParsedQuery):
    """
    Groups join edges by primary key, using the heuristic that whichever
    side of an equi-join is literally named 'id' is the primary-key side.

    Returns: {pk_alias: {'table': pk_table, 'col': pk_col,
                          'members': [(fk_alias, fk_table, fk_col), ...]}}
    """
    groups = {}
    for a1, c1, a2, c2 in parsed.joins:
        if c1 in PK_COLUMN_NAMES and c2 not in PK_COLUMN_NAMES:
            pk_alias, pk_col, fk_alias, fk_col = a1, c1, a2, c2
        elif c2 in PK_COLUMN_NAMES and c1 not in PK_COLUMN_NAMES:
            pk_alias, pk_col, fk_alias, fk_col = a2, c2, a1, c1
        else:
            # both or neither side is 'id' -- ambiguous / self join; skip
            # (extend here if your workload needs this case handled)
            continue
        g = groups.setdefault(
            pk_alias, {"table": parsed.tables[pk_alias], "col": pk_col, "members": []}
        )
        g["members"].append((fk_alias, parsed.tables[fk_alias], fk_col))
    return groups


# --------------------------------------------------------------------------
# DB access + caching
# --------------------------------------------------------------------------

def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name + ".pkl")


def get_primary_key_sample(conn, table: str, col: str, sample_size=DEFAULT_SAMPLE_SIZE,
                            seed=DEFAULT_SEED):
    """
    Returns a FIXED list of `sample_size` primary-key values for `table.col`.
    Cached to disk so the same sample is reused for every query in the
    workload (required for the bitmaps to be comparable across queries).
    """
    key = f"pk_{table}_{col}_{sample_size}_{seed}"
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    with conn.cursor() as cur:
        cur.execute("SELECT setseed(%s)", (((seed % 1000) / 1000.0) - 0.5,))
        cur.execute(
            f"SELECT {col} FROM {table} ORDER BY random() LIMIT %s", (sample_size,)
        )
        ids = sorted(row[0] for row in cur.fetchall())

    with open(path, "wb") as f:
        pickle.dump(ids, f)
    return ids


def _filters_to_sql(alias_placeholder: str, filters, params):
    """Builds a SQL WHERE fragment (using the raw column names, no alias
    prefix, since we query the base table directly) for a list of
    (col, op, value) filters. Appends bind params to `params` in place."""
    clauses = []
    for col, op, value in filters:
        if op == "IN":
            placeholders = ",".join(["%s"] * len(value))
            clauses.append(f"{col} IN ({placeholders})")
            params.extend(value)
        elif op in ("LIKE", "NOT LIKE"):
            clauses.append(f"{col} {op} %s")
            params.append(value)
        else:
            clauses.append(f"{col} {op} %s")
            params.append(value)
    return " AND ".join(clauses)


def get_fk_correlated_rows(conn, fk_table: str, fk_col: str, pk_sample,
                            extra_cols=(), max_rows=DEFAULT_MAX_FK_ROWS,
                            pk_sample_key: str = ""):
    """
    Pulls the rows of `fk_table` whose `fk_col` is in `pk_sample` -- i.e. a
    sample CORRELATED with the primary-key sample, not an independent
    sample of fk_table. Caches the raw rows (fk_col + extra_cols) per
    (fk_table, pk_sample) so repeated queries against the same fk_table in
    a workload don't re-hit the DB. Filters are applied later, in Python,
    against this cached raw data.
    """
    cols = tuple(sorted(set(extra_cols)))
    key = f"fk_{fk_table}_{fk_col}_{pk_sample_key}_{cols}"
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    select_cols = [fk_col] + [c for c in cols if c != fk_col]
    with conn.cursor() as cur:
        cur.execute("SELECT setseed(%s)", (((DEFAULT_SEED % 1000) / 1000.0) - 0.5,))
        cur.execute(
            f"SELECT {', '.join(select_cols)} FROM {fk_table} "
            f"WHERE {fk_col} = ANY(%s) "
            f"ORDER BY random() LIMIT %s",
            (list(pk_sample), max_rows),
        )
        rows = cur.fetchall()

    result = [dict(zip(select_cols, row)) for row in rows]
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


def _eval_single_op(v, op, value):
    """Evaluates one (op, value) condition against a single row's column value."""
    if v is None:
        return False
    if op == "=":
        return v == value
    if op in ("!=", "<>"):
        return v != value
    if op == ">":
        return v > value
    if op == ">=":
        return v >= value
    if op == "<":
        return v < value
    if op == "<=":
        return v <= value
    if op == "IN":
        return v in value
    if op == "LIKE":
        # re.escape does NOT put a backslash before % or _ (they aren't
        # regex-special chars), so they survive re.escape as literal % / _
        # and can be swapped for their SQL-wildcard regex equivalents here.
        pattern = re.escape(value).replace("%", ".*").replace("_", ".")
        return re.match(pattern + "$", str(v), re.DOTALL) is not None
    if op == "NOT LIKE":
        pattern = re.escape(value).replace("%", ".*").replace("_", ".")
        return re.match(pattern + "$", str(v), re.DOTALL) is None
    if op == "OR":
        # value is a list of (sub_op, sub_val) alternatives; any may pass
        return any(_eval_single_op(v, sub_op, sub_val) for sub_op, sub_val in value)
    raise ValueError(f"Unsupported op: {op}")


def _rows_pass_filters(rows, filters):
    """Given cached raw rows (list of dicts) and a list of (col, op, value)
    filters, returns the set of fk_col values (first key in each row dict)
    for rows that satisfy ALL filters."""
    fk_col = None
    passing = set()
    for row in rows:
        if fk_col is None:
            fk_col = next(iter(row))
        if all(_eval_single_op(row.get(col), op, value) for col, op, value in filters):
            passing.add(row[fk_col])
    return passing


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def compute_join_bitmaps(conn, sql: str, sample_size=DEFAULT_SAMPLE_SIZE,
                          seed=DEFAULT_SEED, max_fk_rows=DEFAULT_MAX_FK_ROWS):
    """
    Returns: {pk_alias: {'table': str, 'col': str, 'pk_sample': [ids...],
                          'bitmap': np.ndarray of 0/1 aligned to pk_sample}}
    """
    parsed = parse_query(sql)
    groups = group_by_primary_key(parsed)
    if not groups:
        raise ValueError(
            "No primary-key join detected (no join edge had an 'id' column "
            "on either side) -- nothing to build a join bitmap for."
        )

    results = {}
    for pk_alias, g in groups.items():
        pk_table, pk_col = g["table"], g["col"]
        pk_sample = get_primary_key_sample(conn, pk_table, pk_col, sample_size, seed)
        pk_sample_key = f"{pk_table}_{pk_col}_{sample_size}_{seed}"
        pk_sample_set = set(pk_sample)

        matched = set(pk_sample)  # start with everything, then intersect down

        # incorporate the primary table's own filters (if any), e.g. a
        # filter directly on t.production_year
        own_filters = parsed.filters.get(pk_alias, [])
        if own_filters:
            rows = get_fk_correlated_rows(
                conn, pk_table, pk_col, pk_sample,
                extra_cols=[c for c, _, _ in own_filters],
                max_rows=max_fk_rows, pk_sample_key=pk_sample_key,
            )
            matched &= _rows_pass_filters(rows, own_filters)

        for fk_alias, fk_table, fk_col in g["members"]:
            fk_filters = parsed.filters.get(fk_alias, [])
            rows = get_fk_correlated_rows(
                conn, fk_table, fk_col, pk_sample,
                extra_cols=[c for c, _, _ in fk_filters],
                max_rows=max_fk_rows, pk_sample_key=pk_sample_key,
            )
            passing = _rows_pass_filters(rows, fk_filters) if fk_filters else {
                r[fk_col] for r in rows
            }
            matched &= passing

        bitmap = np.array([1 if pid in matched else 0 for pid in pk_sample], dtype=np.uint8)
        results[pk_alias] = {
            "table": pk_table,
            "col": pk_col,
            "pk_sample": pk_sample,
            "bitmap": bitmap,
        }
    return results


def compute_join_bitmaps_for_workload(conn, sql_list, **kwargs):
    """Convenience wrapper: computes join bitmaps for every query in a list,
    reusing the on-disk cache so repeated primary keys / foreign tables
    across the workload don't re-hit the DB."""
    return [compute_join_bitmaps(conn, sql, **kwargs) for sql in sql_list]


def run_on_workload_dir(conn, pkl_dir: str, out_path: str, **kwargs):
    """
    Batch mode for CEB-style workloads: reads every *.pkl query file in
    `pkl_dir` (each expected to have a top-level 'sql' key, as in
    job_queries/all_job/1a.pkl), computes join bitmaps for each, and
    pickles the results to `out_path` as {query_filename: bitmaps_dict}.
    Skips (and reports) any query the parser can't handle rather than
    aborting the whole batch.
    """
    import glob

    results, failures = {}, []
    for path in sorted(glob.glob(os.path.join(pkl_dir, "*.pkl"))):
        with open(path, "rb") as f:
            query_data = pickle.load(f)
        sql = query_data["sql"] if isinstance(query_data, dict) else query_data
        try:
            results[os.path.basename(path)] = compute_join_bitmaps(conn, sql, **kwargs)
        except Exception as e:  # noqa: BLE001 - report and keep going
            failures.append((os.path.basename(path), str(e)))

    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Wrote {len(results)} bitmap sets to {out_path}")
    if failures:
        print(f"{len(failures)} queries failed to parse/compute:")
        for name, err in failures:
            print(f"  {name}: {err}")
    return results, failures


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _connect():
    return psycopg2.connect(
        dbname=os.environ.get("PGDATABASE", "imdb"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "password"),
        host=os.environ.get("PGHOST", "127.0.0.1"),
        port=os.environ.get("PGPORT", "5432"),
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--workload-dir",
        help="Directory of CEB-style query .pkl files (each with a 'sql' key). "
             "If given, processes every query in the directory instead of the demo query.",
    )
    ap.add_argument(
        "--out", default="join_bitmaps.pkl",
        help="Output path for --workload-dir mode (default: join_bitmaps.pkl)",
    )
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    conn = _connect()
    try:
        if args.workload_dir:
            run_on_workload_dir(
                conn, args.workload_dir, args.out,
                sample_size=args.sample_size, seed=args.seed,
            )
        else:
            # Demo: JOB query 1a (SELECT MIN(mc.note), ... company_type/info_type/
            # movie_companies/movie_info_idx/title, with an OR-combined LIKE
            # predicate on mc.note) -- this is the exact query this script's
            # parser and filter logic were tested against above.
            QUERY = """
                SELECT MIN(mc.note) AS production_note,
                       MIN(t.title) AS movie_title,
                       MIN(t.production_year) AS movie_year
                FROM company_type AS ct, info_type AS it, movie_companies AS mc,
                     movie_info_idx AS mi_idx, title AS t
                WHERE ct.kind = 'production companies'
                  AND it.info = 'top 250 rank'
                  AND mc.note NOT LIKE '%(as Metro-Goldwyn-Mayer Pictures)%'
                  AND (mc.note LIKE '%(co-production)%' OR mc.note LIKE '%(presents)%')
                  AND ct.id = mc.company_type_id
                  AND t.id = mc.movie_id
                  AND t.id = mi_idx.movie_id
                  AND mc.movie_id = mi_idx.movie_id
                  AND it.id = mi_idx.info_type_id;
            """
            bitmaps = compute_join_bitmaps(
                conn, QUERY, sample_size=args.sample_size, seed=args.seed
            )
            for pk_alias, info in bitmaps.items():
                print(f"Primary key group: {pk_alias} ({info['table']}.{info['col']})")
                print(f"  sample size: {len(info['pk_sample'])}")
                print(f"  bits set:    {info['bitmap'].sum()} / {len(info['bitmap'])}")
                print(f"  bitmap[:20]: {info['bitmap'][:20]}")
           
    finally:
        conn.close()
