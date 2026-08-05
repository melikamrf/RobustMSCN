"""
bitmap_generator.py

Generates sample_bitmap and join_bitmap pkl files from pre-parsed query pkl
files. No hardcoded schema -- FK relationships are derived from the join
graph in each query pkl (any join where one side is '.id' means the other
side is a FK column).

Usage:
    from bitmap_generator import compute_bitmaps_from_qrep, generate_for_workload
    conn = psycopg2.connect(...)
    
    # Single query
    sb, jb = compute_bitmaps_from_qrep(conn, qrep)
    
    # Whole workload (scans all queries first to discover all FK columns)
    generate_for_workload(conn, pkl_dir, out_sb_dir, out_jb_dir)
"""
import os
import pickle
import re
import glob
import psycopg2
from collections import defaultdict

DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SEED = 42
CACHE_DIR = './bitmap_cache'


# ---------------------------------------------------------------------------
# Derive FK relationships from a join graph
# ---------------------------------------------------------------------------

def _parse_join_condition(jc_str):
    """Parse 'a.col1 = b.col2' -> (alias1, col1, alias2, col2)."""
    m = re.match(r'(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', jc_str.strip())
    if m:
        return m.group(1), m.group(2), m.group(3), m.group(4)
    return None


def extract_fk_columns(join_graph):
    """From a join graph, extract per-table FK columns.
    
    For each join edge where one side is '.id', the other side is a FK.
    Both sides of the join map to the same 'join_real' name (the FK col name).
    
    Returns:
        table_fks: {real_table_name: set of fk_col_names}
        join_real_map: {(table, col): short_name} mapping for bitmap key naming
    """
    table_fks = defaultdict(set)
    join_real_map = {}
    
    # node id -> real_name mapping
    alias_to_table = {}
    for node in join_graph['nodes']:
        alias_to_table[node['id']] = node['real_name']
    
    seen_edges = set()
    for adj_list in join_graph['adjacency']:
        for edge in adj_list:
            jc = edge.get('join_condition', '')
            if not jc:
                continue
            parsed = _parse_join_condition(jc)
            if not parsed:
                continue
            a1, c1, a2, c2 = parsed
            edge_key = tuple(sorted([(a1, c1), (a2, c2)]))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            
            t1 = alias_to_table.get(a1)
            t2 = alias_to_table.get(a2)
            if t1 is None or t2 is None:
                continue
            
            if c1 == 'id' and c2 != 'id':
                # t1.id is PK, t2.c2 is FK
                table_fks[t2].add(c2)
                join_real_map[(t2, c2)] = c2
                join_real_map[(t1, c1)] = c2
            elif c2 == 'id' and c1 != 'id':
                # t2.id is PK, t1.c1 is FK
                table_fks[t1].add(c1)
                join_real_map[(t1, c1)] = c1
                join_real_map[(t2, c2)] = c1
            # else: neither or both are 'id' -- skip (self-join or ambiguous)
    
    return table_fks, join_real_map


def extract_all_fk_columns(pkl_paths):
    """Scan all query pkl files to discover ALL FK columns per table.
    
    This ensures every query's bitmap includes the full set of FK columns
    for each table, even if that particular query doesn't join on all of them.
    """
    all_table_fks = defaultdict(set)
    all_join_real_map = {}
    
    for path in pkl_paths:
        with open(path, 'rb') as f:
            qrep = pickle.load(f)
        table_fks, join_real_map = extract_fk_columns(qrep['join_graph'])
        for t, cols in table_fks.items():
            all_table_fks[t].update(cols)
        all_join_real_map.update(join_real_map)
    
    return dict(all_table_fks), all_join_real_map


# ---------------------------------------------------------------------------
# Table sampling (cached to disk)
# ---------------------------------------------------------------------------

def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = re.sub(r'[^\w\-.]', '_', name)
    return os.path.join(CACHE_DIR, safe + '.pkl')


def get_table_sample(conn, table, sample_size=DEFAULT_SAMPLE_SIZE, seed=DEFAULT_SEED):
    key = f'tablesample_{table}_{sample_size}_{seed}'
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    with conn.cursor() as cur:
        cur.execute('SELECT setseed(%s)', (((seed % 1000) / 1000.0) - 0.5,))
        cur.execute(f'SELECT * FROM {table} ORDER BY random() LIMIT %s',
                    (sample_size,))
        col_names = [desc[0] for desc in cur.description]
        raw_rows = cur.fetchall()

    rows = [dict(zip(col_names, row)) for row in raw_rows]
    with open(path, 'wb') as f:
        pickle.dump(rows, f)
    return rows


# ---------------------------------------------------------------------------
# Filter evaluation
# ---------------------------------------------------------------------------

def _eval_predicate(row, pred_col, pred_type, pred_val):
    col_name = pred_col.split('.')[1] if '.' in pred_col else pred_col
    v = row.get(col_name)
    if v is None:
        if pred_type == 'is_null':
            return True
        if pred_type == 'is_not_null':
            return False
        return False

    if pred_type == 'eq':
        target = pred_val['literal'] if isinstance(pred_val, dict) and 'literal' in pred_val else pred_val
        try:
            if isinstance(v, int): target = int(target)
            elif isinstance(v, float): target = float(target)
        except (ValueError, TypeError):
            pass
        return v == target

    elif pred_type == 'neq':
        target = pred_val['literal'] if isinstance(pred_val, dict) and 'literal' in pred_val else pred_val
        try:
            if isinstance(v, int): target = int(target)
            elif isinstance(v, float): target = float(target)
        except (ValueError, TypeError):
            pass
        return v != target

    elif pred_type == 'in':
        return v in pred_val

    elif pred_type == 'like':
        if isinstance(pred_val, list):
            return any(
                re.match(re.escape(str(p)).replace('%', '.*').replace('_', '.') + '$',
                         str(v), re.DOTALL) is not None
                for p in pred_val)
        pattern = re.escape(str(pred_val)).replace('%', '.*').replace('_', '.')
        return re.match(pattern + '$', str(v), re.DOTALL) is not None

    elif pred_type == 'not_like':
        if isinstance(pred_val, list):
            return all(
                re.match(re.escape(str(p)).replace('%', '.*').replace('_', '.') + '$',
                         str(v), re.DOTALL) is None
                for p in pred_val)
        pattern = re.escape(str(pred_val)).replace('%', '.*').replace('_', '.')
        return re.match(pattern + '$', str(v), re.DOTALL) is None

    elif pred_type == 'lt':
        lo, hi = pred_val[0], pred_val[1]
        if lo is not None and v <= lo: return False
        if hi is not None and v >= hi: return False
        return True

    elif pred_type == 'lte':
        lo, hi = pred_val[0], pred_val[1]
        if lo is not None and v < lo: return False
        if hi is not None and v > hi: return False
        return True

    elif pred_type == 'is_null':
        return False
    elif pred_type == 'is_not_null':
        return True
    else:
        print(f'WARNING: unknown pred_type={pred_type}, skipping')
        return True


def apply_filters(rows, node):
    pred_cols = node.get('pred_cols', [])
    pred_types = node.get('pred_types', [])
    pred_vals = node.get('pred_vals', [])
    if not pred_cols:
        return rows
    passing = []
    for row in rows:
        ok = True
        for i, (col, ptype) in enumerate(zip(pred_cols, pred_types)):
            pval = pred_vals[i] if i < len(pred_vals) else None
            if not _eval_predicate(row, col, ptype, pval):
                ok = False
                break
        if ok:
            passing.append(row)
    return passing


# ---------------------------------------------------------------------------
# Bitmap generation
# ---------------------------------------------------------------------------

def compute_bitmaps_from_qrep(conn, qrep, sample_size=DEFAULT_SAMPLE_SIZE,
                               seed=DEFAULT_SEED, table_fks=None,
                               join_real_map=None):
    """Generate sample_bitmap and join_bitmap for one query.
    
    Args:
        conn: psycopg2 connection
        qrep: parsed query dict (from pkl file)
        sample_size: number of rows to sample per table
        seed: random seed for reproducibility
        table_fks: optional pre-computed {table: set(fk_cols)} from 
                   extract_all_fk_columns(). If None, derived from this 
                   query's join graph only.
        join_real_map: optional pre-computed {(table, col): short_name}.
                       If None, derived from this query's join graph.
    """
    join_graph = qrep['join_graph']
    nodes = join_graph['nodes']

    # Derive FK columns from this query's join graph if not pre-computed
    if table_fks is None or join_real_map is None:
        table_fks, join_real_map = extract_fk_columns(join_graph)

    sample_bitmap = {}
    join_bitmap = {}

    for node in nodes:
        alias = node['id']
        table = node['real_name']
        fk_cols = sorted(table_fks.get(table, set()))

        all_rows = get_table_sample(conn, table, sample_size, seed)
        passing_rows = apply_filters(all_rows, node)

        # sample_bitmap: set of surviving own .id values
        sample_bitmap[(alias,)] = {
            f'sb{sample_size}': {r['id'] for r in passing_rows
                                 if r.get('id') is not None}
        }

        # join_bitmap: per FK column, set of surviving FK values
        jb = {}
        for fk_col in fk_cols:
            short = join_real_map.get((table, fk_col), fk_col)
            key_name = f'{table}_{short}_sb{sample_size}'
            jb[key_name] = {r[fk_col] for r in passing_rows
                            if r.get(fk_col) is not None}
        join_bitmap[(alias,)] = jb

    return sample_bitmap, join_bitmap


def generate_for_workload(conn, pkl_dir, out_sb_dir, out_jb_dir,
                          sample_size=DEFAULT_SAMPLE_SIZE, seed=DEFAULT_SEED):
    """Generate bitmaps for an entire workload directory.
    
    First scans all query pkl files to discover every FK column per table,
    then generates bitmaps for each query using the full FK column set.
    """
    os.makedirs(out_sb_dir, exist_ok=True)
    os.makedirs(out_jb_dir, exist_ok=True)

    pkl_paths = sorted(glob.glob(os.path.join(pkl_dir, '*.pkl')))
    if not pkl_paths:
        print(f'No pkl files found in {pkl_dir}')
        return

    # First pass: discover all FK columns across the workload
    print(f'Scanning {len(pkl_paths)} queries to discover FK columns...')
    table_fks, join_real_map = extract_all_fk_columns(pkl_paths)
    print(f'Found FK columns: { {t: sorted(cols) for t, cols in table_fks.items()} }')

    # Second pass: generate bitmaps
    count = 0
    for path in pkl_paths:
        fn = os.path.basename(path)
        with open(path, 'rb') as f:
            qrep = pickle.load(f)
        try:
            sb, jb = compute_bitmaps_from_qrep(
                conn, qrep, sample_size, seed, table_fks, join_real_map)
            with open(os.path.join(out_sb_dir, fn), 'wb') as f:
                pickle.dump(sb, f)
            with open(os.path.join(out_jb_dir, fn), 'wb') as f:
                pickle.dump(jb, f)
            count += 1
        except Exception as e:
            print(f'  FAILED {fn}: {e}')
    print(f'Generated {count} bitmap pairs in {out_sb_dir} and {out_jb_dir}.')



def main():
    import psycopg2
    from bitmap_generator import generate_for_workload

    conn = psycopg2.connect(
        dbname=os.environ.get("PGDATABASE", "imdb"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "password"),
        host=os.environ.get("PGHOST", "127.0.0.1"),
        port=os.environ.get("PGPORT", "5432"),
    )
    generate_for_workload(
        conn,
        pkl_dir='./queries/joblight/all_joblight/',
        out_sb_dir='./queries/allbitmaps/joblight/sample_bitmap/',
        out_jb_dir='./queries/allbitmaps/joblight/join_bitmap/',
    )

# def parse_args():
#     import argparse
   
#     parser = argparse.ArgumentParser(description='Generate sample and join bitmaps for a workload.')
#     parser.add_argument('--pkl_dir', type=str, required=True, default='./queries/joblight/all_joblight/',
#                       help='Directory containing query pkl files')
#     parser.add_argument('--out_sb_dir', type=str, required=True, default='./queries/allbitmaps/joblight/sample_bitmap/',
#                       help='Output directory for sample_bitmap pkl files')
#     parser.add_argument('--out_jb_dir', type=str, required=True, default='./queries/allbitmaps/joblight/join_bitmap/',
#                       help='Output directory for join_bitmap pkl files')
#     args = parser.parse_args(namespace=argparse.Namespace(**vars(parser.parse_args())))
#     return args

if __name__ == '__main__':
    # args = parse_args()
    main()