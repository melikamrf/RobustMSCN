"""
extract_bitmaps.py

Stage 3 of the bitmap pipeline. Reads the sample tables that were already
materialized in Postgres by

    create_sampling_tables_bitmaps.py   ->  {table}_sb{N}                (uniform sample)
    create_sampling_join_bitmaps.py     ->  {table}_{joinkey}_sb{N}      (correlated sample)

and turns them into the per-query bitmap pickles that featurizer.py reads.
This script does NO sampling of its own -- run those two first.

How it works (mirrors scripts/get_query_join_bitmaps.py in the original
learned-cardinalities repo):

  * load the qrep, restore join_graph with networkx
  * for each single-table node, take the 1-node subgraph and build its SQL
    with query_representation.utils.nx_graph_to_query -- exactly the SQL the
    original pipeline used, so the filter semantics are Postgres', not a
    Python re-implementation
  * rewrite  FROM <table> AS <alias>  ->  FROM "<sample_table>" AS <alias>
    and       COUNT(*)                ->  "<join_key_column>"
  * the surviving values of that column are the bitmap

SMALL_FIDS
----------
A join key gets a dedicated correlated table only when its defining primary
table is NOT in SMALL_PRIMARY_TABS. But the original script uses a separate,
broader hand-tuned list -- SMALL_FIDS -- to decide which FK columns read from
the plain uniform sample instead. Verified against the reference bitmaps in
queries/allbitmaps/{job,joblight_train,ceb-imdb}: only `movie_id` and
`person_id` are read from correlated tables; every other universe (keyword,
company_id, char_id, info_id, kind_id, role_id, company_type, link_id,
subject) is projected from the table's own uniform sample -- an exact match
for SMALL_FIDS. Do not "fix" this to use SMALL_PRIMARY_TABS; it would silently
diverge from the reference data.

Usage
-----
    python scripts/extract_bitmaps.py \
        --query_dir queries/job/all_job \
        --out_dir   queries/allbitmaps/job \
        --sample_num 1000 --port 5432 --db_name imdb --user postgres

    # regenerate and diff against the existing reference pickles instead of
    # writing anything:
    python scripts/extract_bitmaps.py --query_dir queries/job/all_job \
        --out_dir queries/allbitmaps/job --verify

Writes  <out_dir>/sample_bitmap/<query>.pkl  and  <out_dir>/join_bitmap/<query>.pkl
    sample_bitmap[(alias,)] = {"sb1000": {id, ...}}
    join_bitmap[(alias,)]   = {"<table>_<universe>_sb1000": {val, ...}, ...}

Requires: psycopg2, networkx (uses the repo's query_representation package)
"""

import argparse
import glob
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psycopg2 as pg
from networkx.readwrite import json_graph

from query_representation.utils import nx_graph_to_query

NEW_TABLE_TEMPLATE = "{TABLE}_{SS}{NUM}"
NEW_JOIN_TABLE_TEMPLATE = "{TABLE}_{JOINKEY}_{SS}{NUM}"

# (query_dir, out_dir) relative to the repo root, for --all.
# ceb-imdb is nested one level deep by template; the glob is recursive so that
# works, and output names stay flat (CEB query files are uniquely hashed).
WORKLOADS = [
    ("queries/job/all_job",                       "queries/allbitmaps/job"),
    ("queries/joblight/all_joblight",             "queries/allbitmaps/joblight"),
    ("queries/ceb-imdb",                          "queries/allbitmaps/ceb-imdb"),
    ("queries/joblight_train/joblight-train-all", "queries/allbitmaps/joblight_train"),
]

# --------------------------------------------------------------------------
# Copied verbatim from scripts/get_query_join_bitmaps.py
# --------------------------------------------------------------------------

SMALL_FIDS = ["kind_id", "role_id", "person_role_id", "info_type_id",
              "company_type_id", "company_id", "link_id",
              "keyword_id", "linked_movie_id", "link_type_id",
              "status_id", "subject_id"]

JOIN_COL_MAP_IMDB = {}
JOIN_COL_MAP_IMDB["title.id"] = "movie_id"
JOIN_COL_MAP_IMDB["movie_info.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["cast_info.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["movie_keyword.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["movie_companies.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["movie_link.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["movie_info_idx.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["movie_link.linked_movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["aka_title.movie_id"] = "movie_id"
JOIN_COL_MAP_IMDB["complete_cast.movie_id"] = "movie_id"

JOIN_COL_MAP_IMDB["movie_keyword.keyword_id"] = "keyword"
JOIN_COL_MAP_IMDB["keyword.id"] = "keyword"

JOIN_COL_MAP_IMDB["name.id"] = "person_id"
JOIN_COL_MAP_IMDB["person_info.person_id"] = "person_id"
JOIN_COL_MAP_IMDB["cast_info.person_id"] = "person_id"
JOIN_COL_MAP_IMDB["aka_name.person_id"] = "person_id"

JOIN_COL_MAP_IMDB["title.kind_id"] = "kind_id"
JOIN_COL_MAP_IMDB["kind_type.id"] = "kind_id"

JOIN_COL_MAP_IMDB["cast_info.role_id"] = "role_id"
JOIN_COL_MAP_IMDB["role_type.id"] = "role_id"

JOIN_COL_MAP_IMDB["cast_info.person_role_id"] = "char_id"
JOIN_COL_MAP_IMDB["char_name.id"] = "char_id"

JOIN_COL_MAP_IMDB["movie_info.info_type_id"] = "info_id"
JOIN_COL_MAP_IMDB["movie_info_idx.info_type_id"] = "info_id"
JOIN_COL_MAP_IMDB["person_info.info_type_id"] = "info_id"
JOIN_COL_MAP_IMDB["info_type.id"] = "info_id"

JOIN_COL_MAP_IMDB["movie_companies.company_type_id"] = "company_type"
JOIN_COL_MAP_IMDB["company_type.id"] = "company_type"

JOIN_COL_MAP_IMDB["movie_companies.company_id"] = "company_id"
JOIN_COL_MAP_IMDB["company_name.id"] = "company_id"

JOIN_COL_MAP_IMDB["movie_link.link_type_id"] = "link_id"
JOIN_COL_MAP_IMDB["link_type.id"] = "link_id"

JOIN_COL_MAP_IMDB["complete_cast.status_id"] = "subject"
JOIN_COL_MAP_IMDB["complete_cast.subject_id"] = "subject"
JOIN_COL_MAP_IMDB["comp_cast_type.id"] = "subject"

JOIN_COL_MAP = JOIN_COL_MAP_IMDB


# --------------------------------------------------------------------------
# SQL
# --------------------------------------------------------------------------

def build_exec_sql(subsql, table, alias, exec_table, col):
    """
    Rewrites the single-table subplan SQL to run against a sample table and
    project one column instead of counting.

        SELECT COUNT(*) FROM movie_companies AS mc WHERE mc.note LIKE '%x%'
     -> SELECT "movie_id" FROM "movie_companies_movie_id_sb1000" AS mc
                                                   WHERE mc.note LIKE '%x%'

    The alias is preserved, so the predicates still resolve. Anchoring the
    replacement on the whole "FROM <table> AS <alias>" phrase (rather than the
    original's space-padded bare table name) means a table name appearing
    inside a predicate literal can't be corrupted.
    """
    frm = "FROM {} AS {}".format(table, alias)
    if frm in subsql:
        execsql = subsql.replace(frm, 'FROM "{}" AS {}'.format(exec_table, alias), 1)
    else:
        # fall back to the original script's rewrite
        execsql = subsql.replace(" " + table + " ", ' "' + exec_table + '" ')
    return execsql.replace("COUNT(*)", '"{}"'.format(col))


def _coerce(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


class SampleReader:
    """Runs the rewritten SQL, memoising by the exact SQL string. Workloads
    like JOBLight-train repeat predicates heavily across 43k queries, so the
    cache removes most of the round trips."""

    def __init__(self, conn_args):
        self.conn_args = conn_args
        self.con = pg.connect(**conn_args)
        self.cache = {}
        self.missing_tables = set()
        self.errors = []

    def _reconnect(self):
        try:
            self.con.close()
        except Exception:
            pass
        self.con = pg.connect(**self.conn_args)

    def values(self, execsql):
        if execsql in self.cache:
            return self.cache[execsql]
        try:
            cur = self.con.cursor()
            cur.execute(execsql)
            out = cur.fetchall()
            cur.close()
            vals = {_coerce(o[0]) for o in out if o[0] is not None}
        except pg.errors.UndefinedTable:
            # sample table was never materialized -- record once, not per query
            tab = execsql.split('FROM "')[-1].split('"')[0] if 'FROM "' in execsql else "?"
            self.missing_tables.add(tab)
            self._reconnect()
            self.cache[execsql] = None
            return None
        except Exception as e:
            self.errors.append((execsql, str(e)))
            self._reconnect()
            self.cache[execsql] = None
            return None
        self.cache[execsql] = vals
        return vals

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# Bitmap extraction
# --------------------------------------------------------------------------

def bitmaps_for_qrep(qrep, reader, sample_num):
    """Returns (sample_bitmap, join_bitmap), both keyed by (alias,) tuples."""
    jg = json_graph.adjacency_graph(qrep["join_graph"])

    sample_bitmap, join_bitmap = {}, {}

    for alias in jg.nodes():
        data = jg.nodes()[alias]
        if "real_name" not in data:
            continue
        table = data["real_name"]

        # networkx: the 1-node induced subgraph is exactly this table plus its
        # own predicates, and carries no join edges.
        #
        # .copy() rather than the subgraph view, because the view shares
        # .graph with the parent and we must clear aggr_cmd on it. JOB qreps
        # carry aggr_cmd = "SELECT MIN(mc.note), MIN(t.title), ..."; with it
        # set, nx_graph_to_query emits that SELECT list instead of COUNT(*),
        # the COUNT(*) rewrite below silently does nothing, and Postgres then
        # errors with "missing FROM-clause entry for table mc". Clearing it
        # forces COUNT_SIZE_TEMPLATE, which is what we rewrite.
        sg = jg.subgraph([alias]).copy()
        sg.graph.pop("aggr_cmd", None)
        sg.graph.pop("groupby_cmd", None)
        subsql = nx_graph_to_query(sg)

        uniform_tab = NEW_TABLE_TEMPLATE.format(TABLE=table, SS="sb", NUM=sample_num)

        # ---- sample bitmap: surviving own ids from the uniform sample ----
        vals = reader.values(build_exec_sql(subsql, table, alias, uniform_tab, "id"))
        sample_bitmap[(alias,)] = {"sb{}".format(sample_num): vals if vals is not None else set()}

        # ---- join bitmaps: one entry per join universe this table sits in ----
        jb = {}
        for join_key, join_real in JOIN_COL_MAP.items():
            if ".id" in join_key.lower():
                continue
            if "posttypeid" in join_key.lower():
                continue

            join_tab, join_col = join_key.split(".", 1)
            if join_tab != table:
                continue

            store_key = NEW_JOIN_TABLE_TEMPLATE.format(
                TABLE=table, JOINKEY=join_real, SS="sb", NUM=sample_num)

            # First-wins, matching the original's
            #   `if sample_tab in info["join_bitmap"]: continue`.
            # Only bites on movie_link, where movie_id and linked_movie_id
            # both map to the movie_id universe and so collide on this key.
            if store_key in jb:
                continue

            # SMALL_FIDS: no correlated table exists (or the original chose
            # not to use one) -- project the column from the uniform sample.
            exec_tab = uniform_tab if join_col in SMALL_FIDS else store_key

            vals = reader.values(
                build_exec_sql(subsql, table, alias, exec_tab, join_col))
            if vals is None:
                continue
            jb[store_key] = vals

        join_bitmap[(alias,)] = jb

    return sample_bitmap, join_bitmap


def _diff(gen, ref, label, name, problems):
    if set(gen.keys()) != set(ref.keys()):
        problems.append("{} {}: alias sets differ gen={} ref={}".format(
            name, label, sorted(gen), sorted(ref)))
        return
    for alias in ref:
        g, r = gen[alias], ref[alias]
        if set(g.keys()) != set(r.keys()):
            problems.append("{} {} {}: keys differ gen={} ref={}".format(
                name, label, alias, sorted(g), sorted(r)))
            continue
        for k in r:
            if set(g[k]) != set(r[k]):
                gs, rs = set(g[k]), set(r[k])
                problems.append(
                    "{} {} {} [{}]: {} vs {} values (+{} / -{})".format(
                        name, label, alias, k, len(gs), len(rs),
                        len(gs - rs), len(rs - gs)))


def run_workload(reader, query_dir, out_dir, sample_num,
                 verify=False, resume=False, limit=-1):
    """Regenerates (or verifies) every query in one workload directory.
    Returns (done, skipped, failures, problems)."""
    sbdir = os.path.join(out_dir, "sample_bitmap")
    jbdir = os.path.join(out_dir, "join_bitmap")
    if not verify:
        os.makedirs(sbdir, exist_ok=True)
        os.makedirs(jbdir, exist_ok=True)

    # recursive: ceb-imdb nests query files one level deep, by template
    fns = sorted(glob.glob(os.path.join(query_dir, "**", "*.pkl"), recursive=True))
    if limit != -1:
        fns = fns[:limit]

    seen_names = {}
    for fn in fns:
        seen_names.setdefault(os.path.basename(fn), []).append(fn)
    dupes = {n: p for n, p in seen_names.items() if len(p) > 1}
    if dupes:
        print("  WARNING: {} duplicate basenames across subdirs; outputs would "
              "collide. First: {}".format(len(dupes), list(dupes.items())[0]))

    print("  {} query files".format(len(fns)))

    problems, failures, done, skipped = [], [], 0, 0
    t0 = time.time()
    for i, fn in enumerate(fns):
        name = os.path.basename(fn)
        sout, jout = os.path.join(sbdir, name), os.path.join(jbdir, name)

        if resume and not verify and os.path.exists(sout) and os.path.exists(jout):
            skipped += 1
            continue

        try:
            with open(fn, "rb") as f:
                qrep = pickle.load(f)
            sb, jb = bitmaps_for_qrep(qrep, reader, sample_num)
        except Exception as e:
            failures.append((name, str(e)))
            continue

        if verify:
            if not (os.path.exists(sout) and os.path.exists(jout)):
                problems.append("{}: no reference pickle to compare".format(name))
                continue
            with open(sout, "rb") as f:
                _diff(sb, pickle.load(f), "sample_bitmap", name, problems)
            with open(jout, "rb") as f:
                _diff(jb, pickle.load(f), "join_bitmap", name, problems)
        else:
            # write to a temp name then replace, so an interrupted run never
            # leaves a half-written pickle that --resume would then skip
            for path, obj in ((sout, sb), (jout, jb)):
                tmp = path + ".tmp"
                with open(tmp, "wb") as f:
                    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
        done += 1

        if (i + 1) % 500 == 0:
            el = time.time() - t0
            rate = done / el if el > 0 else 0
            left = (len(fns) - i - 1) / rate / 60 if rate > 0 else 0
            print("    {}/{}  cache={}  {:.0f} q/s  ~{:.0f} min left".format(
                i + 1, len(fns), len(reader.cache), rate, left))

    return done, skipped, failures, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="run every workload in WORKLOADS, sharing one connection "
                         "and one SQL cache (paths are relative to the repo root)")
    ap.add_argument("--query_dir", help="directory of qrep .pkl files (searched recursively)")
    ap.add_argument("--out_dir", help="writes <out_dir>/sample_bitmap/ and <out_dir>/join_bitmap/")
    ap.add_argument("--sample_num", type=int, default=1000)
    ap.add_argument("-n", "--num_queries", type=int, default=-1)
    ap.add_argument("--verify", action="store_true",
                    help="compare against existing pickles instead of writing")
    ap.add_argument("--resume", action="store_true",
                    help="skip queries whose two output pickles already exist")
    ap.add_argument("--db_name", default=os.environ.get("PGDATABASE", "imdb"))
    ap.add_argument("--db_host", default=os.environ.get("PGHOST", "localhost"))
    ap.add_argument("--user", default=os.environ.get("PGUSER", "postgres"))
    ap.add_argument("--pwd", default=os.environ.get("PGPASSWORD", "password"))
    ap.add_argument("--port", default=os.environ.get("PGPORT", "5432"))
    args = ap.parse_args()

    if args.all:
        jobs = [(str(ROOT / q), str(ROOT / o)) for q, o in WORKLOADS]
    else:
        if not (args.query_dir and args.out_dir):
            ap.error("give --all, or both --query_dir and --out_dir")
        jobs = [(args.query_dir, args.out_dir)]

    reader = SampleReader(dict(dbname=args.db_name, user=args.user, password=args.pwd,
                               host=args.db_host, port=args.port))

    totals, start = [], time.time()
    try:
        for qdir, odir in jobs:
            print("\n=== {} -> {}".format(qdir, odir))
            if not os.path.isdir(qdir):
                print("  SKIP: query dir does not exist")
                continue
            done, skipped, failures, problems = run_workload(
                reader, qdir, odir, args.sample_num,
                verify=args.verify, resume=args.resume, limit=args.num_queries)
            totals.append((qdir, done, skipped, failures, problems))
            print("  done={} skipped={} failed={} mismatches={}".format(
                done, skipped, len(failures), len(problems)))
            for n, e in failures[:3]:
                print("    FAIL {}: {}".format(n, e))
            for p in problems[:5]:
                print("    DIFF " + p)
    finally:
        reader.close()

    print("\n{}".format("=" * 60))
    print("total time: {:.1f} min   distinct SQL executed: {}".format(
        (time.time() - start) / 60.0, len(reader.cache)))
    for qdir, done, skipped, failures, problems in totals:
        print("  {:52s} done={:6d} skip={:6d} fail={:4d} diff={}".format(
            os.path.basename(qdir), done, skipped, len(failures), len(problems)))
    if reader.missing_tables:
        print("MISSING sample tables (run the two create_sampling_* scripts first): {}".format(
            sorted(reader.missing_tables)))
    if reader.errors:
        print("{} distinct SQL errors, first 3:".format(len(reader.errors)))
        for sql, err in reader.errors[:3]:
            print("   {}\n     {}".format(sql[:160], err))


if __name__ == "__main__":
    main()
