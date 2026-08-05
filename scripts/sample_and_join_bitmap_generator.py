"""
sample_and_join_bitmap_generator.py

Generates bitmaps matching the structure of your actual 1a-sample-bitmap.pkl
/ 1a-join-bitmap.pkl files -- NOT the paper's correlated join-bitmap
technique (that's in join_bitmap_generator.py, kept separately). This is
the simpler, independent-per-table-sample approach that your files show:

For every table alias in the query:
  1. Draw a FIXED sample of that table's own rows (up to `sample_size`,
     cached to disk so the SAME rows are reused across the whole workload
     -- this reuses get_primary_key_sample()/get_fk_correlated_rows() from
     join_bitmap_generator.py, which already do exactly this).
  2. Apply that table's OWN filter predicate(s) from the query to the
     sampled rows (independently -- no correlation with any other table's
     sample).
  3. sample bitmap = the set of that table's own 'id' values among the
     rows that passed the filter.
  4. join bitmap = one entry PER FOREIGN-KEY COLUMN defined on that table
     (from SCHEMA_FK_TO_PK below, not just columns used by this query's
     joins), each holding the set of DISTINCT values of that column among
     the SAME passing rows.

Evidence this is the right structure (from your uploaded files):
  - ('ct',) and ('it',) are EMPTY in the join-bitmap file -- company_type
    and info_type are pure dimension tables with no outgoing FK columns,
    which is exactly what SCHEMA_FK_TO_PK predicts.
  - mc's join bitmap has a 'company_id' entry even though this query never
    joins to company_name -- confirming it's ALL of a table's FK columns,
    not just the ones the current query happens to use.
  - mc's sample-bitmap count and its join-bitmap 'company_id' count were
    identical (9 in both) in your file -- consistent with both being
    projections of the exact same passing-row set.

CAVEAT: your files use shortened join-key names in some cases
('info_type_id' -> 'info_id', 'company_type_id' -> 'company_type') rather
than the raw column name. I could only confirm these two from your
examples (see JOIN_KEY_NAME_OVERRIDES) -- I don't know the naming
convention for columns I haven't seen a real example of, so other columns
default to a simple "strip trailing _id" rule. Verify/extend that dict if
exact key-name compatibility with your loader matters.

Also note: exact per-run VALUES won't match your files bit-for-bit unless
you use the same seed AND the same underlying table-sampling method your
original pipeline used (TABLESAMPLE vs ORDER BY random() behave
differently) -- your own two uploaded files disagree with each other on
mc's sample size (20 in the embedded query pkl vs 9 in the standalone
sample-bitmap.pkl), which suggests the seed wasn't pinned across those
runs either. Matching STRUCTURE is achievable here; matching exact numbers
isn't, without your original generation seed/method.

Requires: psycopg2-binary (via join_bitmap_generator.py)
"""

import os
import pickle

from join_bitmap_generator import (
    DEFAULT_MAX_FK_ROWS,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    _eval_single_op,
    get_fk_correlated_rows,
    get_primary_key_sample,
    parse_query,
)

# --------------------------------------------------------------------------
# Schema: every outgoing foreign-key column per table, for the standard
# IMDb/JOB/CEB schema. Extend this if your workload touches tables not
# listed here (aka_name, complete_cast, movie_link, person_info, etc. are
# included since they're part of the standard schema even though they
# don't appear in query 1a).
# --------------------------------------------------------------------------
SCHEMA_FK_TO_PK = {
    ("aka_name", "person_id"): ("name", "id"),
    ("aka_title", "movie_id"): ("title", "id"),
    ("cast_info", "movie_id"): ("title", "id"),
    ("cast_info", "person_id"): ("name", "id"),
    ("cast_info", "person_role_id"): ("char_name", "id"),
    ("cast_info", "role_id"): ("role_type", "id"),
    ("complete_cast", "movie_id"): ("title", "id"),
    ("complete_cast", "subject_id"): ("comp_cast_type", "id"),
    ("complete_cast", "status_id"): ("comp_cast_type", "id"),
    ("movie_companies", "movie_id"): ("title", "id"),
    ("movie_companies", "company_id"): ("company_name", "id"),
    ("movie_companies", "company_type_id"): ("company_type", "id"),
    ("movie_info", "movie_id"): ("title", "id"),
    ("movie_info", "info_type_id"): ("info_type", "id"),
    ("movie_info_idx", "movie_id"): ("title", "id"),
    ("movie_info_idx", "info_type_id"): ("info_type", "id"),
    ("movie_keyword", "movie_id"): ("title", "id"),
    ("movie_keyword", "keyword_id"): ("keyword", "id"),
    ("movie_link", "movie_id"): ("title", "id"),
    ("movie_link", "linked_movie_id"): ("title", "id"),
    ("movie_link", "link_type_id"): ("link_type", "id"),
    ("person_info", "person_id"): ("name", "id"),
    ("person_info", "info_type_id"): ("info_type", "id"),
    ("title", "kind_id"): ("kind_type", "id"),
}

# Confirmed (from your uploaded files) or best-guess shortened key names.
# Only the two marked CONFIRMED were actually verified against your data.
JOIN_KEY_NAME_OVERRIDES = {
    ("movie_companies", "company_type_id"): "company_type",   # CONFIRMED
    ("movie_info_idx", "info_type_id"): "info_id",             # CONFIRMED
    ("movie_info", "info_type_id"): "info_id",                 # guess, by analogy
    ("person_info", "info_type_id"): "info_id",                # guess, by analogy
}


def _short_name(table: str, col: str) -> str:
    # NOTE: there's no consistent transformation rule here -- movie_id and
    # company_id keep their full column name in your files, while
    # company_type_id and info_type_id get shortened in two DIFFERENT,
    # non-uniform ways. Defaulting to the raw column name is the safest
    # choice for anything not explicitly confirmed below.
    return JOIN_KEY_NAME_OVERRIDES.get((table, col), col)


def _rows_that_pass(rows, filters):
    """Returns the list of row dicts (not projected to one column) that
    satisfy ALL (col, op, value) filters."""
    return [
        row for row in rows
        if all(_eval_single_op(row.get(col), op, value) for col, op, value in filters)
    ]


def compute_sample_and_join_bitmaps(conn, sql: str, sample_size=DEFAULT_SAMPLE_SIZE,
                                     seed=DEFAULT_SEED, max_rows=DEFAULT_MAX_FK_ROWS):
    """
    Returns (sample_bitmap, join_bitmap), each keyed by (alias,) tuples,
    matching the structure of your uploaded files:

      sample_bitmap[(alias,)] = {f'sb{sample_size}': {matching own-id values}}
      join_bitmap[(alias,)]   = {f'{table}_{short_col}_sb{sample_size}':
                                  {matching fk-column values}, ...}
                                 (empty dict for tables with no outgoing FKs)
    """
    parsed = parse_query(sql)
    sample_bitmap, join_bitmap = {}, {}

    for alias, table in parsed.tables.items():
        own_filters = parsed.filters.get(alias, [])
        fk_cols = sorted(c for (t, c) in SCHEMA_FK_TO_PK if t == table)

        # Fixed, cached, independent sample of this table's own row ids.
        own_id_sample = get_primary_key_sample(conn, table, "id", sample_size, seed)
        pk_sample_key = f"{table}_id_{sample_size}_{seed}"

        if not own_id_sample:
            sample_bitmap[(alias,)] = {f"sb{sample_size}": set()}
            join_bitmap[(alias,)] = {
                f"{table}_{_short_name(table, c)}_sb{sample_size}": set() for c in fk_cols
            }
            continue

        needed_cols = sorted({c for c, _, _ in own_filters} | set(fk_cols))
        rows = get_fk_correlated_rows(
            conn, table, "id", own_id_sample,
            extra_cols=needed_cols, max_rows=max_rows, pk_sample_key=pk_sample_key,
        )
        passing_rows = _rows_that_pass(rows, own_filters) if own_filters else rows

        sample_bitmap[(alias,)] = {
            f"sb{sample_size}": {r["id"] for r in passing_rows if r.get("id") is not None}
        }
        join_bitmap[(alias,)] = {
            f"{table}_{_short_name(table, c)}_sb{sample_size}": {
                r[c] for r in passing_rows if r.get(c) is not None
            }
            for c in fk_cols
        }

    return sample_bitmap, join_bitmap


def run_on_workload_dir(conn, pkl_dir: str, sample_bitmap_out: str, join_bitmap_out: str,
                         **kwargs):
    """Batch mode: reads every *.pkl query file in `pkl_dir` (each expected
    to have a top-level 'sql' key), computes sample+join bitmaps for each,
    and writes two pickles keyed by query filename."""
    import glob

    sample_results, join_results, failures = {}, {}, []
    for path in sorted(glob.glob(os.path.join(pkl_dir, "*.pkl"))):
        with open(path, "rb") as f:
            query_data = pickle.load(f)
        sql = query_data["sql"] if isinstance(query_data, dict) else query_data
        name = os.path.basename(path)
        try:
            sb, jb = compute_sample_and_join_bitmaps(conn, sql, **kwargs)
            sample_results[name] = sb
            join_results[name] = jb
        except Exception as e:  # noqa: BLE001 - report and keep going
            failures.append((name, str(e)))

    with open(sample_bitmap_out, "wb") as f:
        pickle.dump(sample_results, f)
    with open(join_bitmap_out, "wb") as f:
        pickle.dump(join_results, f)

    print(f"Wrote {len(sample_results)} sample bitmaps to {sample_bitmap_out}")
    print(f"Wrote {len(join_results)} join bitmaps to {join_bitmap_out}")
    if failures:
        print(f"{len(failures)} queries failed:")
        for name, err in failures:
            print(f"  {name}: {err}")
    return sample_results, join_results, failures


if __name__ == "__main__":
    import argparse

    import psycopg2

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload-dir")
    ap.add_argument("--sample-bitmap-out", default="sample_bitmaps.pkl")
    ap.add_argument("--join-bitmap-out", default="join_bitmaps.pkl")
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    conn = psycopg2.connect(
        dbname=os.environ.get("PGDATABASE", "imdb"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "password"),
        host=os.environ.get("PGHOST", "127.0.0.1"),
        port=os.environ.get("PGPORT", "5432"),
    )

    try:
        if args.workload_dir:
            run_on_workload_dir(
                conn, args.workload_dir, args.sample_bitmap_out, args.join_bitmap_out,
                sample_size=args.sample_size, seed=args.seed,
            )
        else:
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
            sb, jb = compute_sample_and_join_bitmaps(
                conn, QUERY, sample_size=args.sample_size, seed=args.seed
            )
            with open('./queries/allbitmaps/job/join_bitmap/1a.pkl', 'rb') as f:
                    true_jb = pickle.load(f)
            print(true_jb.items())
            
          
            for alias in sb:
                print(f"{alias}:")
                for k, v in sb[alias].items():
                    print(f"  sample_bitmap[{k!r}]: {len(v)} values, e.g. {list(v)[:5]}")
                for k, v in jb[alias].items():
                    print(f"  join_bitmap[{k!r}]: {len(v)} values, e.g. {list(v)[:5]}")
                
    finally:
        conn.close()


# the first sample_and_join_bitmap is tables and columns are correct. 