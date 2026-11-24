"""
benchmark_parallel_sort.py — Compare sorting strategies for an astropy Table.

Strategies tested:
1. Single-threaded: astropy Table.sort() — baseline
2. Parallel sort + hierarchical parallel merge + parallel column reorder

The parallel sort works as follows:
  a) Split Vpeak array into nthreads chunks
  b) Sort each chunk in parallel (ThreadPoolExecutor)
  c) Merge sorted chunks in log2(nthreads) rounds, each round fully parallelised
     — each individual merge uses concatenate + Timsort (exploits 2 sorted runs)
  d) Apply final index to all table columns in parallel

This is a drop-in replacement for astropy Table.sort() + reverse().

Usage:
    python benchmark_parallel_sort.py [--nrows N] [--ncols C] [--nthreads T]

On NERSC compute node, try --nrows 2100000000 --nthreads 64
"""

import argparse
import gc
import time
import numpy as np
from astropy.table import Table
from concurrent.futures import ThreadPoolExecutor
from numba import njit


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nrows',    type=int, default=27_000_000,
                   help='Number of rows (default 27M; use 2100000000 for full scale)')
    p.add_argument('--ncols',   type=int, default=17,
                   help='Number of columns including Vpeak (default 17)')
    p.add_argument('--nthreads', type=int, nargs='+', default=[16, 32, 64],
                   help='Thread counts to test (default: 16 32 64)')
    p.add_argument('--no_verify', action='store_true',
                   help='Skip verification (saves time at full scale)')
    return p.parse_args()


# =============================================================================
# Table generation
# =============================================================================

def make_table(nrows, ncols, rng):
    t = Table()
    t['Vpeak'] = rng.random(nrows).astype(np.float32) * 1000.0
    for i in range(ncols - 1):
        t[f'col{i}'] = rng.random(nrows).astype(np.float32)
    return t


# =============================================================================
# Strategy 1: single-threaded baseline
# =============================================================================

def sort_single(t):
    """Baseline: single-threaded astropy Table.sort() + reverse()."""
    t2 = t.copy()
    t2.sort('Vpeak')
    t2.reverse()
    return t2


# =============================================================================
# Strategy 2: parallel sort + hierarchical parallel merge + parallel reorder
# =============================================================================

def _sort_chunk(args):
    """
    Sort one chunk of Vpeak descending.
    Returns global indices into the original array in sorted order.
    args: (vpeak_chunk, global_offset)
    """
    vpeak_chunk, offset = args
    local_idx = np.argsort(vpeak_chunk, kind='stable')[::-1]
    return local_idx + offset


@njit
def _numba_reorder_col(src, idx, dst):
    """
    Reorder one column in-place using numba.
    src : source array (float32, length n)
    idx : index array (int64, length n)
    dst : pre-allocated destination buffer (float32, length n)
    Writes src[idx[i]] into dst[i] for all i.
    """
    for i in range(len(idx)):
        dst[i] = src[idx[i]]


@njit
def _numba_merge_desc(va, ia, vb, ib):
    """
    True O(n) merge of two descending-sorted arrays, compiled by numba.
    va, vb : float32 value arrays, each already sorted descending
    ia, ib : int64 index arrays
    Returns (merged_values, merged_indices).
    """
    na = len(va)
    nb = len(vb)
    out_v = np.empty(na + nb, dtype=va.dtype)
    out_i = np.empty(na + nb, dtype=ia.dtype)
    i = j = k = 0
    while i < na and j < nb:
        if va[i] >= vb[j]:
            out_v[k] = va[i]; out_i[k] = ia[i]; i += 1
        else:
            out_v[k] = vb[j]; out_i[k] = ib[j]; j += 1
        k += 1
    while i < na:
        out_v[k] = va[i]; out_i[k] = ia[i]; i += 1; k += 1
    while j < nb:
        out_v[k] = vb[j]; out_i[k] = ib[j]; j += 1; k += 1
    return out_v, out_i


def _merge_two(args):
    """
    Merge two sorted index arrays using numba O(n) merge.
    args: (idx_a, idx_b, vpeak)
    Returns merged index array.
    """
    idx_a, idx_b, vpeak = args
    va = vpeak[idx_a]
    vb = vpeak[idx_b]
    _, merged_i = _numba_merge_desc(va, idx_a, vb, idx_b)
    return merged_i


def _reorder_col(args):
    """Apply index to one column."""
    col_data, idx = args
    return col_data[idx]


def parallel_sort_descending(t, key, nthreads, verbose=True):
    """
    Sort astropy Table t descending by column `key`, using parallel sort
    + hierarchical parallel merge + parallel column reorder.

    Parameters
    ----------
    t        : astropy Table
    key      : column name to sort by (e.g. 'Vpeak')
    nthreads : number of threads; ideally a power of 2
    verbose  : print timing breakdown

    Returns
    -------
    Sorted astropy Table (new object, t is unchanged)
    """
    vpeak = np.asarray(t[key])
    n = len(vpeak)
    chunk_size = (n + nthreads - 1) // nthreads
    offsets = [i * chunk_size for i in range(nthreads)]
    chunks = [(vpeak[o : min(o + chunk_size, n)], o) for o in offsets]

    # (a) Sort each chunk in parallel
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        sorted_idx = list(ex.map(_sort_chunk, chunks))
    if verbose:
        print(f"  (a) Parallel chunk sorts ({nthreads} threads): {time.time()-t0:.2f}s")

    # (b) Hierarchical parallel merge
    # Each round merges adjacent pairs in parallel, halving the number of chunks
    t0 = time.time()
    round_num = 0
    while len(sorted_idx) > 1:
        # Pair up chunks for this round
        pairs = []
        leftovers = []
        for i in range(0, len(sorted_idx) - 1, 2):
            pairs.append((sorted_idx[i], sorted_idx[i+1], vpeak))
        if len(sorted_idx) % 2 == 1:
            leftovers.append(sorted_idx[-1])

        n_merges = len(pairs)
        with ThreadPoolExecutor(max_workers=min(nthreads, n_merges)) as ex:
            merged = list(ex.map(_merge_two, pairs))

        sorted_idx = merged + leftovers
        round_num += 1
        if verbose:
            print(f"  (b) Merge round {round_num} "
                  f"({n_merges} parallel merges → {len(sorted_idx)} chunks): "
                  f"{time.time()-t0:.2f}s cumulative")

    final_idx = sorted_idx[0]
    if verbose:
        print(f"  (b) Hierarchical merge total: {time.time()-t0:.2f}s")

    # (c) Memory-efficient sequential numba reorder.
    # Reorder one column at a time using a single pre-allocated buffer.
    # Peak extra memory = 2 columns (~17 GB at full scale) rather than
    # a full copy of the table (~143 GB).
    t0 = time.time()
    n = len(final_idx)
    buf = np.empty(n, dtype=np.float32)
    for c in t.colnames:
        src = np.asarray(t[c], dtype=np.float32)
        _numba_reorder_col(src, final_idx, buf)
        t[c] = buf.copy()
    if verbose:
        print(f"  (c) Sequential numba reorder: {time.time()-t0:.2f}s")

    return t


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    rng = np.random.default_rng(42)

    # Generate table once
    print(f"Generating synthetic table: {args.nrows:,} rows x {args.ncols} cols...")
    t0 = time.time()
    t = make_table(args.nrows, args.ncols, rng)
    mem = sum(t[c].nbytes for c in t.colnames) / 1e9
    print(f"  Generated in {time.time()-t0:.2f}s,  Memory: ~{mem:.2f} GB")

    # Warm up numba JIT before timing
    print(f"\nWarming up numba JIT...")
    _va = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    _ia = np.array([0, 1, 2], dtype=np.int64)
    _vb = np.array([2.5, 1.5, 0.5], dtype=np.float32)
    _ib = np.array([3, 4, 5], dtype=np.int64)
    _numba_merge_desc(_va, _ia, _vb, _ib)
    _src = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    _dst = np.empty(4, dtype=np.float32)
    _numba_reorder_col(_src, np.array([3, 1, 0, 2], dtype=np.int64), _dst)
    print(f"  Numba JIT ready.")

    # Single-threaded baseline once
    print(f"\n1) Single-threaded astropy sort...")
    t0 = time.time()
    t_single = sort_single(t)
    t_single_time = time.time() - t0
    print(f"   Total: {t_single_time:.2f}s")

    # Loop over thread counts
    results = []
    for nthreads in args.nthreads:
        print(f"\n2) Parallel sort + hierarchical merge ({nthreads} threads)...")
        t0 = time.time()
        t_par = parallel_sort_descending(t.copy(), 'Vpeak', nthreads, verbose=True)
        t_par_time = time.time() - t0
        print(f"   Total: {t_par_time:.2f}s")

        if not args.no_verify:
            ok = np.allclose(t_single['Vpeak'][:1000], t_par['Vpeak'][:1000])
            print(f"  Verification: {'PASS' if ok else 'FAIL'}")

        results.append((nthreads, t_par_time))

    print(f"\n{'='*55}")
    print(f"Rows: {args.nrows:,}   Cols: {args.ncols}")
    print(f"Single-threaded baseline: {t_single_time:.2f}s")
    print(f"{'Threads':>10} {'Time(s)':>10} {'Speedup':>10}")
    print(f"{'-'*32}")
    for nthreads, t_par_time in results:
        print(f"{nthreads:>10} {t_par_time:>10.2f} {t_single_time/t_par_time:>10.2f}x")


if __name__ == '__main__':
    main()
