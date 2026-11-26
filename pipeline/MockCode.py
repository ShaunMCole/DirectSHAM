#Functions used in the mock creation pipeline MockVpeakZpeak.ipynb
from numba import njit,prange
import multiprocessing
import logging
import os
import h5py
import numpy as np
import time
import gc
from astropy.table import Table
from kcorrections  import DESI_KCorrection
import catalogue_analysis as ca
import fitsio
from desimodel.footprint import is_point_in_desi
import matplotlib.pyplot as plt

log = logging.getLogger()  # root logger — inherits handlers from make_checkpoint.py

from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# Parallel argsort descending — used in pre_sort_subhalos to avoid full-table
# physical sort.  Only the Vpeak array is sorted; the table is reordered only
# at write time when it has been trimmed to ~8-18M rows.
# =============================================================================

@njit
def _numba_merge_desc(va, ia, vb, ib):
    """O(n) merge of two descending-sorted index arrays using numba."""
    na, nb = len(va), len(vb)
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


def _sort_chunk_indices(args):
    """Sort one chunk of Vpeak descending; return global indices."""
    vpeak_chunk, offset = args
    local_idx = np.argsort(vpeak_chunk, kind='stable')[::-1]
    return local_idx + offset


def _merge_two(args):
    """Merge two sorted index arrays using numba O(n) merge."""
    idx_a, idx_b, vpeak = args
    _, merged_i = _numba_merge_desc(vpeak[idx_a], idx_a, vpeak[idx_b], idx_b)
    return merged_i


def parallel_argsort_desc(vpeak, nthreads=16):
    """
    Parallel argsort descending on a 1-D float array.

    1. Split into nthreads chunks; sort each chunk in parallel.
    2. Merge sorted index arrays hierarchically using numba O(n) merge.

    Returns index array such that vpeak[result] is sorted descending.
    """
    vpeak = np.asarray(vpeak)
    n = len(vpeak)
    chunk_size = (n + nthreads - 1) // nthreads
    offsets = [i * chunk_size for i in range(nthreads)]
    chunks = [(vpeak[o : min(o + chunk_size, n)], o) for o in offsets]

    # Step 1: sort each chunk in parallel
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        sorted_idx = list(ex.map(_sort_chunk_indices, chunks))

    # Step 2: hierarchical parallel merge
    while len(sorted_idx) > 1:
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

    return sorted_idx[0]


# Warm up numba JIT on first import (small arrays, negligible cost)
_wv = np.array([3.0, 2.0, 1.0], dtype=np.float32)
_wi = np.array([0, 1, 2], dtype=np.int64)
_numba_merge_desc(_wv, _wi, _wv, _wi)
del _wv, _wi




# Add gaussian scatter to log10(Vpeak) before ranking if required   
def apply_log_scatter(x, sigma=None, sig_lgv_ref=None, sig_lgv_slope=None,
                      lgv_pivot=2.6, alpha_v=0.0, f_col_proxy=None):
    """
    Apply log-normal scatter to x (e.g. Vpeak).

    Two modes:
      - scalar sigma:  apply_log_scatter(x, sigma=0.1)
      - Vpeak-dependent: apply_log_scatter(x, sig_lgv_ref=0.14,
                                            sig_lgv_slope=0.24, lgv_pivot=2.6)
        In this mode sigma(x) = max(0, sig_lgv_ref + sig_lgv_slope*(log10(x) - lgv_pivot))
        and no intermediate sigma array is retained after the scatter is drawn.

    Optional assembly-history scatter with luminosity-dependent suppression:
      When alpha_v > 0 and f_col_proxy is provided:

          x_col = sqrt(3) * (1 - 2*f_col_proxy)   [zero mean, unit variance]
          eta   ~ N(0, 1)
          eps   = alpha_v * SIGMA_MIN / max(sigma_v, SIGMA_MIN) * x_col
                  + sigma_v * eta

      alpha_v is the maximum boost in log10(Vpeak), attained at the faint
      end where sigma_v <= SIGMA_MIN. At the bright end (large sigma_v) the
      boost is suppressed as SIGMA_MIN/sigma_v, so luminous samples are
      minimally affected. SIGMA_MIN = 0.035 is hardwired to the median
      sigma_v of the faintest sample (Mr < -17.5).
      Early-forming haloes (small f_col_proxy, large Mpeak_Scale) have
      x_col > 0 and therefore receive a positive boost to their scattered
      Vpeak, so they are preferentially matched to brighter galaxies.
    """
    if sig_lgv_ref is not None:
        log_x     = np.log10(x)
        sigma_arr = sig_lgv_ref + sig_lgv_slope * (log_x - lgv_pivot)
        np.clip(sigma_arr, 0.0, None, out=sigma_arr)
        eta = np.random.normal(loc=0.0, scale=1.0, size=len(x))
        if alpha_v != 0.0 and f_col_proxy is not None:
            # Assembly-history boost with luminosity-dependent suppression:
            #   epsilon = alpha_v * sigma_min / max(sigma_v, sigma_min) * x_col
            #             + sigma_v * eta
            # alpha_v is the maximum boost (attained at the faint end where
            # sigma_v <= sigma_min). The boost is suppressed at the bright end
            # (large sigma_v) so that luminous samples are minimally affected.
            # sigma_min is hardwired to the median sigma_v of the faintest
            # sample (Mr < -17.5), so alpha_v retains a physical interpretation.
            SIGMA_MIN = 0.035
            x_col   = np.sqrt(3.0) * (1.0 - 2.0 * np.asarray(f_col_proxy))
            boost   = alpha_v * SIGMA_MIN / np.maximum(sigma_arr, SIGMA_MIN)
            scatter = boost * x_col + sigma_arr * eta
            del x_col, boost
        else:
            scatter = sigma_arr * eta
        del sigma_arr, eta
        return x * 10.0**scatter
    elif sigma is None or sigma == 0.0:
        return x
    else:
        eta = np.random.normal(loc=0.0, scale=sigma, size=len(x))
        return x * 10.0**eta

@njit(parallel=True)
def _apply_scatter_parallel(x, sig_lgv_ref, sig_lgv_slope, lgv_pivot,
                             alpha_v, f_col_proxy, sigma_min, seed):
    """
    Parallel version of apply_log_scatter for the Vpeak-dependent case
    with optional assembly-history boost (alpha_v).
    Uses numba's built-in random.gauss for thread-safe per-element RNG.
    """
    n   = len(x)
    out = np.empty(n, dtype=np.float32)
    np.random.seed(seed)
    for i in prange(n):
        log_xi  = np.log10(np.float64(x[i]))
        sigma_i = sig_lgv_ref + sig_lgv_slope * (log_xi - lgv_pivot)
        if sigma_i < 0.0:
            sigma_i = 0.0
        eta_i = np.random.normal(0.0, 1.0)
        if alpha_v != 0.0:
            x_col_i = np.sqrt(3.0) * (1.0 - 2.0 * np.float64(f_col_proxy[i]))
            denom   = sigma_i if sigma_i > sigma_min else sigma_min
            boost_i = alpha_v * sigma_min / denom
            scatter = boost_i * x_col_i + sigma_i * eta_i
        else:
            scatter = sigma_i * eta_i
        out[i] = np.float32(np.float64(x[i]) * 10.0**scatter)
    return out


@njit(parallel=True)
def _rotate_coords_parallel(x, y, z, R):
    """
    Apply a 3x3 rotation matrix R to coordinate arrays in parallel.
    Returns (xr, yr, zr).
    """
    n  = len(x)
    xr = np.empty(n, dtype=x.dtype)
    yr = np.empty(n, dtype=y.dtype)
    zr = np.empty(n, dtype=z.dtype)
    for i in prange(n):
        xi = x[i]; yi = y[i]; zi = z[i]
        xr[i] = R[0,0]*xi + R[0,1]*yi + R[0,2]*zi
        yr[i] = R[1,0]*xi + R[1,1]*yi + R[1,2]*zi
        zr[i] = R[2,0]*xi + R[2,1]*yi + R[2,2]*zi
    return xr, yr, zr


# Warmup
_wn = np.array([100.0, 200.0], dtype=np.float32)
_wf = np.zeros(2, dtype=np.float32)
_apply_scatter_parallel(_wn, 0.18, 0.30, 2.6, 0.02, _wf, 0.035, np.int64(42))
_wR = np.eye(3, dtype=np.float64)
_wd = np.array([1.0, 2.0], dtype=np.float64)
_rotate_coords_parallel(_wd, _wd, _wd, _wR)
del _wn, _wf, _wR, _wd


#Generate a random set of Euler angles
def random_euler_angles(seed=None):

    if (seed==0):
        #Align with simulation axes
        phi=0.0
        theta=0.0
        psi=0.0
    else:
        #Chosen random orientation isotropically
        rng = np.random.default_rng(seed)

        phi = 2.0 * np.pi * rng.random()

        costheta = 2.0 * rng.random() - 1.0
        theta = np.arccos(costheta)

        psi = 2.0 * np.pi * rng.random()

    return phi, theta, psi

#Compute the rotation matrix descrbed by a set of Euler angles
#(For phi,theta,psi= (0,0,0) this is just the identity matrix)   
def euler_rotation_matrix(phi, theta, psi):

    c1 = np.cos(phi)
    s1 = np.sin(phi)

    c2 = np.cos(theta)
    s2 = np.sin(theta)

    c3 = np.cos(psi)
    s3 = np.sin(psi)

    Rz1 = np.array([
        [ c1, -s1, 0],
        [ s1,  c1, 0],
        [  0,   0, 1]
    ])

    Ry = np.array([
        [ c2, 0, s2],
        [  0, 1,  0],
        [-s2, 0, c2]
    ])

    Rz2 = np.array([
        [ c3, -s3, 0],
        [ s3,  c3, 0],
        [  0,   0, 1]
    ])

    return Rz1 @ Ry @ Rz2

# Uuchu routine adapted from those provided by Elena Fernandez
#
#   Reads one of the Uchu files and makes appends 7 periodic replicas
#   lbox is the simulation box size. coordinates assumed to be [0,lbox] 
#   frac_box < 1 cuts out a sphere of radius frac_box*lbox around the observer
#
def read_halolist_file_reps(file_path, randseed, columns, min_logMvir=10.4, min_Vpeak=0.0, lbox=2000.0, frac_box=0.8, xobs=0.0, yobs=0.0, zobs=0.0):

    shortname = os.path.basename(file_path)
    results = dict()

    #List of columns that we can recast as 32bit rather than native 64bit and hence save memory   
    float32_cols = {'vx', 'vy', 'vz','Mpeak_Scale','Rvir','rs','Vpeak'}

    with h5py.File(file_path, 'r') as f:
        for col in columns:
            arr = f[col][:]

            if col in float32_cols:
                arr = arr.astype(np.float32)

            results[col] = arr



    # --- Vpeak cut ---
    ntot_vpeak = len(results['Vpeak'])
    keep = (results['Vpeak'] > min_Vpeak)
    nkeep_vpeak = int(np.sum(keep))
    for col in results:
        results[col] = results[col][keep]

    # --- mass cut ---
    ntot_mvir = len(results['Mvir_all'])
    keep = (results['Mvir_all'] > 10**min_logMvir)
    nkeep_mvir = int(np.sum(keep))
    for col in results:
        results[col] = results[col][keep]

    print(f'{shortname}: Vpeak ({100*nkeep_vpeak/ntot_vpeak:.1f}%)  '
          f'Mvir ({100*nkeep_mvir/ntot_mvir:.1f}%)  '
          f'final={nkeep_mvir:,}')



    
    # ------------------------------------------------------------
    # Compute Mhost: host halo mass for each subhalo.
    # For centrals (upid == -1), Mhost = Mvir_all.
    # For satellites, Mhost = Mvir_all of the top-level host (upid).
    # This must be done before replication because IDs are made unique
    # per replica afterwards, breaking the upid -> id lookup.
    # We use upid (ultimate parent) rather than pid (immediate parent)
    # so that sub-subhalos also get the correct top-level host mass.
    # The completeness check confirms all upid values are found within
    # the same file (fraction = 1.000000, 0 orphans).
    # ------------------------------------------------------------
    id_arr   = results['id']
    upid_arr = results['upid']
    mvir_arr = results['Mvir_all']

    # Build id -> Mvir_all map for centrals (upid == -1)
    is_central  = (upid_arr == -1)
    id_to_mvir  = dict(zip(id_arr[is_central], mvir_arr[is_central]))

    # Assign Mhost
    mhost = np.where(
        is_central,
        mvir_arr,
        np.array([id_to_mvir.get(u, np.nan) for u in upid_arr], dtype=np.float32)
    )
    results['Mhost'] = mhost.astype(np.float32)
    del id_to_mvir, id_arr, upid_arr, mvir_arr, is_central, mhost

    # ------------------------------------------------------------
    # Reposition the box about the chosen observer
    # ------------------------------------------------------------


    results['x'] = (results['x'] - xobs) % lbox
    results['y'] = (results['y'] - yobs) % lbox
    results['z'] = (results['z'] - zobs) % lbox

 # ------------------------------------------------------------------
    # Periodic replication with immediate radial cut.
    #
    # Instead of constructing the full 8-replica catalogue and then
    # throwing most of it away, process each replica separately and
    # retain only objects inside the target sphere.
    # ------------------------------------------------------------------

    shifts = np.array(
        np.meshgrid(
            [0, -lbox],
            [0, -lbox],
            [0, -lbox]
        )
    ).T.reshape(-1, 3)

    nrep = len(shifts)  # should be 8

    r2_max = (frac_box * lbox)**2

    out = {k: [] for k in results}

    for irep, (dx, dy, dz) in enumerate(shifts):

        # Shift coordinates for this replica only
        xrep = results['x'] + dx
        yrep = results['y'] + dy
        zrep = results['z'] + dz

        # Radial cut for this replica
        r2 = xrep*xrep + yrep*yrep + zrep*zrep
        keep = (r2 < r2_max)
            
        
        # Store coordinate columns
        out['x'].append(xrep[keep])
        out['y'].append(yrep[keep])
        out['z'].append(zrep[keep])

        # Store all remaining columns
        for k in results:

            if k in ('x', 'y', 'z'):
                continue

            elif k == 'id':
                # Make IDs unique across replicas
                out[k].append((results[k] * nrep + irep)[keep])

            elif k in ('pid', 'upid'):
                # Transform pid/upid consistently with id so that parent
                # lookups remain valid after replication, while preserving
                # the sentinel value -1 that identifies centrals.
                arr = results[k]
                transformed = np.where(arr == -1, -1, arr * nrep + irep)
                out[k].append(transformed[keep])

            else:
                out[k].append(results[k][keep])

    # Concatenate surviving halos from all replicas
    for k in out:
        out[k] = np.concatenate(out[k])


    return out






# Correspondence between snapshot and redshift
redshift_table = {
    33: 1.03, 
    34: 0.94,
    35: 0.86,
    36: 0.78,
    37: 0.70,
    38: 0.63,
    39: 0.56,
    40: 0.49,
    41: 0.43,
    43: 0.30,
    45: 0.19,
    47: 0.09,
    50: 0.00
}

# Onion shell boundaries used previously by Elena

#zmin zmax snapshot
#0.0 0.05 50
#0.05 0.10 47
#0.10 0.25 45
#0.25 0.40 43
#0.40 0.45 41
#0.45 0.50 40

# Make all columns in a astro-py table contiguous in memory which might help numba code to be more efficent
def make_all_columns_contiguous(tab):
    for name in tab.colnames:
        col = tab[name]
        # Convert column to contiguous array **without changing dtype**
        arr_contig = np.ascontiguousarray(col)
        tab[name] = arr_contig
    return tab

# Parallel routine to read all Uchuu subhalo files (starmap version — default)
def read_halolist_parallel(file_paths, randseed, columns, min_logMvir=10.4, min_Vpeak=0.0, lbox=2000.0, frac_box=0.8, xobs=0.0, yobs=0.0, zobs=0.0, use_tmpfiles=False, tmpdir=None):
    """
    Read all Uchuu subhalo files and return an astropy Table.

    use_tmpfiles : bool (default False)
        If False (default): original pipe-based parallel method using
        pool.starmap.  Fast and works well for low-z snapshots where
        total data volume is manageable (~50-100 GB).

        If True: sequential checkpointed method.  Each file is read one
        at a time and its result written to a .npy file in tmpdir.
        Already-written files are skipped, so the run is restartable if
        it times out.  Slower (no parallelism during the read) but
        reliable for high-z snapshots (snap 40-43) where the parallel
        pool deadlocks due to the large data volume (~160 GB).

    tmpdir : str or None
        Directory for .npy checkpoint files when use_tmpfiles=True.
        Should be on $SCRATCH for best performance.  If None, a
        temporary directory is created automatically (not restartable).
        Pass a fixed path (e.g. '$SCRATCH/uchuu_snap043_tmp') to enable
        restart after timeout: existing .npy files are skipped.
        The directory is NOT deleted after completion so that reruns
        can reuse it — clean it up manually once the checkpoint is done.
    """
    nfiles = len(file_paths)

    args = []
    for i, filename in enumerate(file_paths):
        args.append([filename,randseed*nfiles+i,columns,min_logMvir,min_Vpeak,lbox,frac_box,xobs,yobs,zobs])
        
    log.info(f"Start method: {multiprocessing.get_start_method()}")
    ncpus = multiprocessing.cpu_count()
    nworkers = min(nfiles, ncpus)
    log.info(f"ncpus={ncpus}  nworkers={nworkers}")

    if use_tmpfiles:
        # ------------------------------------------------------------------
        # Sequential checkpointed method.
        # Each halo file is read one at a time and saved to a .npy file.
        # If tmpdir already contains output from a previous (timed-out)
        # run, those files are reused and the corresponding reads skipped.
        # Concatenation is also sequential so peak RAM is ~2x one file.
        # ------------------------------------------------------------------
        import tempfile

        if tmpdir is None:
            tmpdir = tempfile.mkdtemp(dir=os.environ.get('SCRATCH', '/tmp'))
            print(f"use_tmpfiles=True  tmpdir (auto): {tmpdir}")
        else:
            os.makedirs(tmpdir, exist_ok=True)
            print(f"use_tmpfiles=True  tmpdir (fixed): {tmpdir}")

        # Read each file sequentially, skipping those already on disk
        outfiles = []
        for i, arg in enumerate(args):
            outfile = os.path.join(tmpdir, f'tmp_{i:04d}.npy')
            outfiles.append(outfile)
            if os.path.exists(outfile):
                print(f"  [{i+1}/{nfiles}] Skipping (cached): {os.path.basename(arg[0])}")
                continue
            print(f"  [{i+1}/{nfiles}] Reading: {os.path.basename(arg[0])}", flush=True)
            result = read_halolist_file_reps(*arg)
            np.save(outfile, result)
            del result
            gc.collect()

        # Concatenate sequentially — only one file's data in RAM at a time
        print("All files read. Concatenating...")
        results = {}
        tot_bytes = 0
        for outfile in outfiles:
            d = np.load(outfile, allow_pickle=True).item()
            for k, v in d.items():
                tot_bytes += v.nbytes
                if k in results:
                    results[k] = np.concatenate([results[k], v])
                else:
                    results[k] = v.copy()
            del d
            gc.collect()
        print(f"All file_results = {tot_bytes/1024**3:.1f} GB")
        print(f"Checkpoint files retained in {tmpdir} — delete manually when done.")

    else:
        # ------------------------------------------------------------------
        # Original pipe-based method (default).  Works well for low-z
        # snapshots where total data volume is manageable.
        # ------------------------------------------------------------------
        t_read = time.time()
        with multiprocessing.Pool(nworkers) as pool:
            file_results = pool.starmap(read_halolist_file_reps, args)
        log.info(f'Parallel read complete in {time.time()-t_read:.1f}s')

        tot_bytes = 0
        for d in file_results:
            for v in d.values():
                tot_bytes += v.nbytes
        log.info(f'All file_results = {tot_bytes/1024**3:.1f} GB')

        # concatenate across files without doubling memory usage
        t_concat = time.time()
        results = {}
        cols = list(file_results[0].keys())
        for col in cols:
            results[col] = np.concatenate([x[col] for x in file_results])
            # immediately release this column from file_results
            for x in file_results:
                del x[col]
        log.info(f'Concatenation done in {time.time()-t_concat:.1f}s')
        # free up memory
        del file_results
        gc.collect()

    t_table = time.time()
    tab = Table(results)
    log.info(f'Table construction done in {time.time()-t_table:.1f}s')
    return tab


def _read_halolist_file_reps_star(args):
    """Wrapper for pool.imap — unpacks args list."""
    return read_halolist_file_reps(*args)


def read_halolist_parallel_imap(file_paths, randseed, columns, min_logMvir=10.4, min_Vpeak=0.0, lbox=2000.0, frac_box=0.8, xobs=0.0, yobs=0.0, zobs=0.0, use_tmpfiles=False, tmpdir=None):
    """
    Experimental imap_unordered version — kept for reference but slower than
    starmap in practice due to per-result IPC overhead. Do not use in production.

    Uses pool.imap_unordered so that each worker's result is concatenated
    and freed as soon as it arrives, rather than holding all 100 results
    (~156 GB) simultaneously in memory before concatenating.
    """
    if use_tmpfiles:
        return read_halolist_parallel(
            file_paths, randseed, columns,
            min_logMvir=min_logMvir, min_Vpeak=min_Vpeak,
            lbox=lbox, frac_box=frac_box,
            xobs=xobs, yobs=yobs, zobs=zobs,
            use_tmpfiles=True, tmpdir=tmpdir,
        )

    nfiles = len(file_paths)
    args = [
        [filename, randseed*nfiles+i, columns, min_logMvir, min_Vpeak,
         lbox, frac_box, xobs, yobs, zobs]
        for i, filename in enumerate(file_paths)
    ]

    log.info(f"Start method: {multiprocessing.get_start_method()}")
    ncpus = multiprocessing.cpu_count()
    nworkers = min(nfiles, ncpus)
    log.info(f"ncpus={ncpus}  nworkers={nworkers}")

    # Accumulate results as a list of arrays per column, then concatenate once.
    # This avoids O(n^2) repeated concatenation and minimises peak memory —
    # each worker's dict is freed as soon as its columns are appended.
    t_read = time.time()
    col_lists = {}   # col_name -> list of arrays, one per file
    tot_bytes = 0
    with multiprocessing.Pool(nworkers) as pool:
        for d in pool.imap_unordered(_read_halolist_file_reps_star, args):
            for col, arr in d.items():
                tot_bytes += arr.nbytes
                if col in col_lists:
                    col_lists[col].append(arr)
                else:
                    col_lists[col] = [arr]
            del d
            gc.collect()
    log.info(f'Parallel read + streaming collect complete in {time.time()-t_read:.1f}s')
    log.info(f'All file_results = {tot_bytes/1024**3:.1f} GB')

    # Final concatenation — one np.concatenate call per column
    t_concat = time.time()
    results = {}
    for col, arrays in col_lists.items():
        results[col] = np.concatenate(arrays)
        del col_lists[col]  # free as we go
    log.info(f'Final concatenation done in {time.time()-t_concat:.1f}s')
    del col_lists
    gc.collect()

    return Table(results)


# Three routines that are just-in-time compiled using numba:
    
# compute the rms scatter in quantity z among the N_neigh neighbours of each element in the array
@njit(parallel=True)
def compute_sky_coords(x, y, z, vx, vy, vz, Dc_grid, z_grid):
    """
    Compute sky coordinates and redshifts for all subhalos in one parallel pass.

    Parameters
    ----------
    x, y, z   : float64 arrays, comoving position relative to observer [Mpc/h]
    vx, vy, vz: float32 arrays, peculiar velocity [km/s]
    Dc_grid   : float64 array, comoving distance grid for z interpolation
    z_grid    : float64 array, redshift grid corresponding to Dc_grid

    Returns
    -------
    dist  : float32  — comoving distance [Mpc/h]
    ra    : float32  — right ascension [degrees]
    dec   : float32  — declination [degrees]
    zu    : float32  — real-space redshift
    zobs  : float32  — redshift-space redshift
    """
    n = len(x)
    c_light = np.float64(299792.458)
    pi      = np.float64(3.141592653589793)
    r2d     = np.float64(180.0 / pi)
    ng      = len(Dc_grid)

    dist = np.empty(n, dtype=np.float64)
    ra   = np.empty(n, dtype=np.float32)
    dec  = np.empty(n, dtype=np.float32)
    zu   = np.empty(n, dtype=np.float32)
    zobs = np.empty(n, dtype=np.float32)

    for i in prange(n):
        xi = x[i]; yi = y[i]; zi = z[i]
        d  = np.sqrt(xi*xi + yi*yi + zi*zi)
        dist[i] = d

        # RA and DEC
        dec[i] = np.float32(r2d * np.arcsin(zi / d))
        ra_val = r2d * np.arctan2(yi, xi)
        if ra_val < 0.0:
            ra_val += 360.0
        ra[i] = np.float32(ra_val)

        # Radial velocity / c
        vr_c = (xi * np.float64(vx[i]) +
                yi * np.float64(vy[i]) +
                zi * np.float64(vz[i])) / (c_light * d)

        # Real-space redshift via linear interpolation on Dc_grid
        # Binary search for d in Dc_grid
        lo = np.int64(0)
        hi = np.int64(ng - 2)
        while lo < hi:
            mid = (lo + hi) // 2
            if Dc_grid[mid + 1] <= d:
                lo = mid + 1
            else:
                hi = mid
        t    = (d - Dc_grid[lo]) / (Dc_grid[lo + 1] - Dc_grid[lo])
        zu_i = z_grid[lo] + t * (z_grid[lo + 1] - z_grid[lo])
        zu[i] = np.float32(zu_i)

        # Redshift-space redshift
        zobs[i] = np.float32((1.0 + zu_i) * (1.0 + vr_c) - 1.0)

    return dist, ra, dec, zu, zobs


_wx = np.array([100.0, 200.0], dtype=np.float64)
_wDc = np.linspace(0.0, 3000.0, 100)
_wz  = np.linspace(0.0, 1.0,   100)
compute_sky_coords(_wx, _wx, _wx,
                   np.ones(2, dtype=np.float32),
                   np.ones(2, dtype=np.float32),
                   np.ones(2, dtype=np.float32),
                   _wDc, _wz)
del _wx, _wDc, _wz


@njit(parallel=True)
def compute_scatter(z, N_neigh):
    zloc=z #make a temporary local copy which might help cache usage and numba efficency
    n = len(z)
    sigma = np.zeros(n, dtype=np.float32)
    half = N_neigh // 2

    for i in prange(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        total=hi-lo-1  #total in range minus self-count

        zi = zloc[i]
        sumsq = 0

        for j in range(lo, hi):
            sumsq=sumsq+(zloc[j]-zi)**2


        if total > 0:
            sigma[i] = sumsq / total
        else:
            sigma[i] = np.nan

    return sigma

@njit(parallel=True)
def compute_frac_local_mpeak(mpeak_scale, N_neigh):

    n = len(mpeak_scale)
    frac = np.zeros(n, dtype=np.float32)
    half = N_neigh // 2

    for i in prange(n):

        lo = max(0, i - half)
        hi = min(n, i + half + 1)

        total = hi - lo - 1

        ai = mpeak_scale[i]

        count = 0

        for j in range(lo, hi):

            # Equivalent to zpeak[j] > zpeak[i]
            if mpeak_scale[j] < ai:
                count += 1

        if total > 0:
            frac[i] = count / total
        else:
            frac[i] = np.nan

    return frac


@njit(parallel=True)
def compute_frac_local(z, N_neigh):
    zloc=z #make a temporary local copy which might help cache usage and numba efficency
    n = len(z)
    frac = np.zeros(n, dtype=np.float32)
    half = N_neigh // 2

    for i in prange(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        total=hi-lo-1  #total in range minus self-count

        zi = zloc[i]
        count = 0

        for j in range(lo, hi):
            if zloc[j] > zi:
                count += 1

        if total > 0:
            frac[i] = count / total
        else:
            frac[i] = np.nan

    return frac




# i) Sort the subhalos into decreasing order of Vpeak (index only — no physical reorder)
# ii) Compute fraction of neighbours with higher zpeak
#iii) Add scatter to Vpeak and sort again (index only)
# iv) Trim the catalogue by applying a tailored Vpeak(z) threshold
# v) Store sort index in subhalos['sort_idx'] for use at write time
def pre_sort_subhalos(subhalos, N_neigh=100, sig_lgv_ref=0.0, sig_lgv_slope=0.0,
                      lgv_pivot=2.6, f_colour=0.0, seed_fcolour=42,
                      alpha_v=0.0, plot=False, nthreads=16):

    nsub = len(subhalos)

    # ------------------------------------------------------------------
    # Sort 1: parallel argsort of Vpeak only — no full-table reorder
    # ------------------------------------------------------------------
    start_time = time.time()
    log.info("Starting argsort 1 (index only, no table reorder)")
    idx1 = parallel_argsort_desc(subhalos['Vpeak'], nthreads=nthreads)
    end_time = time.time()
    log.info(f"Argsort 1 complete in {end_time - start_time:.1f}s")


  
    # ------------------------------------------------------------------
    # compute FRAC (local rank statistic)
    # compute_frac_local_mpeak requires Mpeak_Scale in Vpeak-sorted order,
    # so we apply idx1 to Mpeak_Scale only (~8 GB reorder — cheap).
    # ------------------------------------------------------------------
    start_time = time.time()
    mps_raw = np.asarray(subhalos['Mpeak_Scale'])[idx1]  # reorder Mpeak_Scale by idx1
    rng_fc = np.random.default_rng(seed_fcolour)
    if f_colour > 0.0:
        noise = rng_fc.standard_normal(len(mps_raw)).astype(np.float32) * np.float32(f_colour)
    else:
        noise = (rng_fc.random(len(mps_raw)).astype(np.float32) - 0.5) * np.float32(0.01)
    np.log10(mps_raw, out=mps_raw)
    mps_raw += noise
    del noise
    frac_sorted = compute_frac_local_mpeak(mps_raw, N_neigh=N_neigh)
    del mps_raw
    # Map FracZpeak back to original (unsorted) order using inverse of idx1
    inv_idx1 = np.empty(nsub, dtype=np.int64)
    inv_idx1[idx1] = np.arange(nsub, dtype=np.int64)
    subhalos['FracZpeak'] = frac_sorted[inv_idx1]
    del frac_sorted, inv_idx1
    end_time = time.time()
    log.info(f"FracZpeak computed in {end_time - start_time:.1f}s")

    if sig_lgv_ref > 0.0 or sig_lgv_slope != 0.0:
        #Add scatter to Vpeak
        start_time = time.time()
        # Capture before/after (insert BEFORE the scatter line)
        if plot:
            # Sample a small fraction BEFORE copying anything
            nsub = len(subhalos)
            sample_frac = 0.02  # 2% — adjust as needed
            idx = np.random.choice(nsub, size=int(nsub * sample_frac), replace=False)
            vpeak_old = subhalos['Vpeak'][idx].copy()  # small sampled copy only
            log_mpeak_s = np.log10(subhalos['Mvir_all'][idx])  # grab Mpeak at same indices

        SIGMA_MIN = np.float64(0.035)
        f_proxy = np.asarray(subhalos['FracZpeak'], dtype=np.float32) \
                  if alpha_v != 0.0 else np.zeros(nsub, dtype=np.float32)
        subhalos['Vpeak'] = _apply_scatter_parallel(
            np.asarray(subhalos['Vpeak'], dtype=np.float32),
            np.float64(sig_lgv_ref),
            np.float64(sig_lgv_slope),
            np.float64(lgv_pivot),
            np.float64(alpha_v),
            f_proxy,
            SIGMA_MIN,
            np.int64(seed_fcolour),
        )
        del f_proxy
        if plot:
            vpeak_new = subhalos['Vpeak'][idx].copy()  # grab sample BEFORE sort

        end_time = time.time()
        log.info(f"Vpeak scatter added in {end_time - start_time:.1f}s")
        start_time = time.time()
        log.info("Starting argsort 2 (index only, no table reorder)")
        idx2 = parallel_argsort_desc(subhalos['Vpeak'], nthreads=nthreads)
        end_time = time.time()
        log.info(f"Argsort 2 complete in {end_time - start_time:.1f}s")
        if plot:
           
            log_old = np.log10(vpeak_old)
            log_new = np.log10(vpeak_new)
            delta   = log_new - log_old
        
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
            ax = axes[0]
            ax.scatter(log_old, log_new, s=1.0, alpha=0.3, rasterized=True)
            lims = [log_old.min(), log_old.max()]
            ax.plot(lims, lims, 'r--', lw=1, label='1:1')
            ax.set_xlabel(r'$\log_{10}(V_\mathrm{peak,\,old})$')
            ax.set_ylabel(r'$\log_{10}(V_\mathrm{peak,\,new})$')
            ax.set_title(rf'$\sigma_{{ref}}={sig_lgv_ref}$, slope$={sig_lgv_slope}$ (2% sample)')
            ax.legend()
        
            ax = axes[1]
            ax.hist(delta, bins=200, density=True, histtype='step', color='steelblue')
            mu, std = delta.mean(), delta.std()
            x = np.linspace(delta.min(), delta.max(), 300)
            ax.plot(x, np.exp(-0.5*((x-mu)/std)**2) / (std*np.sqrt(2*np.pi)),
                    'r-', lw=1.5, label=rf'Gaussian fit: $\sigma={std:.4f}$')
            ax.set_xlabel(r'$\Delta\log_{10}(V_\mathrm{peak})$')
            ax.set_ylabel('Density')
            ax.set_title(rf'Measured scatter: $\sigma={std:.4f}$,  $\mu={mu:.4f}$')
            ax.legend()
        
            fig.tight_layout()
            plt.savefig('vpeak_scatter_check.png', dpi=150)
            plt.show()
            print(f"Measured log10 scatter: sigma={std:.4f}, mu={mu:.4f}")
        
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(log_mpeak_s, log_old, s=1.0, alpha=0.3, rasterized=True)
            ax.set_xlabel(r'$\log_{10}(M_\mathrm{peak})$')
            ax.set_ylabel(r'$\log_{10}(V_\mathrm{peak,\,old})$')
            ax.set_title(r'$V_\mathrm{peak}$ vs $M_\mathrm{peak}$ (2% sample)')
            fig.tight_layout()
            plt.savefig('vpeak_vs_mpeak.png', dpi=150)
            plt.show()

    
    # ------------------------------------------------------------------
    # Post-scatter FracZpeak recomputation
    # FracZpeak was computed above on the pre-scatter Vpeak ordering (idx1)
    # and used in the alpha term of the scatter formula.  With alpha>0 the
    # scatter direction is correlated with FracZpeak, so at a given
    # post-scatter Vpeak rank FracZpeak is no longer uniform.  Recomputing
    # on the post-scatter ordering (idx2) restores uniformity at each
    # post-scatter Vpeak and decouples the colour assignment from the
    # scatter, ensuring the colour distribution is independent of alpha.
    # ------------------------------------------------------------------
    if sig_lgv_ref > 0.0 or sig_lgv_slope != 0.0:
        start_time = time.time()
        mps_raw2 = np.asarray(subhalos['Mpeak_Scale'])[idx2]
        rng_fc2 = np.random.default_rng(seed_fcolour + 1)
        if f_colour > 0.0:
            noise2 = rng_fc2.standard_normal(len(mps_raw2)).astype(np.float32) * np.float32(f_colour)
        else:
            noise2 = (rng_fc2.random(len(mps_raw2)).astype(np.float32) - 0.5) * np.float32(0.01)
        np.log10(mps_raw2, out=mps_raw2)
        mps_raw2 += noise2
        del noise2
        frac_sorted2 = compute_frac_local_mpeak(mps_raw2, N_neigh=N_neigh)
        del mps_raw2
        inv_idx2 = np.empty(nsub, dtype=np.int64)
        inv_idx2[idx2] = np.arange(nsub, dtype=np.int64)
        subhalos['FracZpeak'] = frac_sorted2[inv_idx2]
        del frac_sorted2, inv_idx2
        log.info(f"Post-scatter FracZpeak recomputed in {time.time() - start_time:.1f}s")

    # ------------------------------------------------------------------
    # Final sort index: idx2 if scatter was applied, idx1 otherwise
    # ------------------------------------------------------------------
    final_idx = idx2 if (sig_lgv_ref > 0.0 or sig_lgv_slope != 0.0) else idx1

    # Rank index: position in final sorted order (1 = highest Vpeak)
    # Stored in original (unsorted) row order via inverse index.
    start_time = time.time()
    i1_sorted = np.arange(1, nsub + 1, dtype=np.int64)
    inv_final = np.empty(nsub, dtype=np.int64)
    inv_final[final_idx] = np.arange(nsub, dtype=np.int64)
    subhalos['i1'] = i1_sorted[inv_final]
    del i1_sorted, inv_final
    log.info(f'Rank index: nsub={nsub:,} max(i1)={np.max(subhalos["i1"]):,} min(i1)={np.min(subhalos["i1"]):,}')
    end_time = time.time()
    log.info(f"Rank index defined in {end_time - start_time:.1f}s")

    # ------------------------------------------------------------------
    # Distance dependent trimming — element-wise, no sort needed
    # Piecewise: original cubic polynomial below 1300 Mpc/h; above 1300
    # a quadratic extension matching value and slope at the join point,
    # rising more steeply to trim unusable high-distance subhalos.
    # ------------------------------------------------------------------
    start_time = time.time()
    dist  = subhalos['dist']
    _d0, _v0, _slope0, _B = 1300.0, 545.5, 1.436, 0.01
    orig  = ((6.5e-07 * dist - 9.02e-04) * dist + 0.486) * dist + 10.0
    limit = np.where(dist <= _d0,
                     orig,
                     _v0 + _slope0 * (dist - _d0) + _B * (dist - _d0)**2)
    mask  = subhalos['Vpeak'] > limit
    subhalos['keep'] = mask
    end_time = time.time()
    log.info(f"Distance trimming mask defined in {end_time - start_time:.1f}s")

    # Store final sort index for use at write time (applied after trimming)
    subhalos['sort_idx'] = final_idx.astype(np.int64)

    return subhalos

# Memory efficient way of shrinking a table keeping only the values with column'keep' True
def filter_table_staged(tab, mask, drop_mask=True, mask_name='keep'):

    new_cols = {}
    
    # Process columns one by one
    for col in list(tab.colnames):
        if col == mask_name:
            continue
        
        # Create filtered column
        new_cols[col] = tab[col][mask]
        
        # Immediately delete original column to free memory
        tab.remove_column(col)

    # Optionally drop mask column
    if drop_mask and mask_name in tab.colnames:
        tab.remove_column(mask_name)

    # Now rebuild table from filtered columns
    new_tab = Table(new_cols)

    return new_tab

# Main slice by slice abundance matching code
#    
def abundance_match_subhalos_to_randoms(
    subhalos,
    ran,
    lbox,
    rsphere,
    fsky,
    nmult,
    z_bins,
    scatter_sigma=None,
    DeltaZ=0.03,
    reg=None,
    N_neigh=50,
    Rejection=False,
    diagnostics_path='./sdata/Mocks/diagnostics.fits'
):
    """
    NumPy-based abundance matching with minimal memory overhead.
    """
    kcorr_r  = DESI_KCorrection(band='R', file='jmext', photsys=reg) 
    Ntot_expected = 0.0
    Ntot_actual = 0
    Ntot_ran_vis = 0.0
    print(rf"fsky={fsky}  ")
    # ------------------------------------------------------------
    # Extract NumPy arrays (no copies yet)
    # ------------------------------------------------------------
    sub_id = subhalos['ID']
    sub_vpeak = subhalos['Vpeak']
    sub_z = subhalos['Zu']
    i1= subhalos['i1']

    ran_id = ran['ID']
    ran_mag=ran['ABSMAG_R']
    ran_gp = ran['ABSMAG_GP1']
    ran_rp = ran['ABSMAG_RP1']
    ran_rmag = ran['rmag']
    ran_z = ran['Z']
    #ran_zmin = ran['zmin']
    ran_zmax = ran['zmax']


    # ------------------------------------------------------------
    # Sort randoms once by luminosity (brightest first)
    # ------------------------------------------------------------
    ran_order = np.argsort(ran_mag)  # more negative mag = brighter
    ran_id = ran_id[ran_order]
    ran_z = ran_z[ran_order]
    #ran_zmin = ran_zmin[ran_order]
    ran_zmax = ran_zmax[ran_order]
    ran_gp = ran_gp[ran_order]
    ran_rp = ran_rp[ran_order]
    ran_mag =ran_mag[ran_order]

    # Get global zmin of the catalogue
    Sel=ca.selection(reg)
    zmin_cat=Sel['zmin']  #but this is already z_bins[0]

    V_sim = (4.0*np.pi/3.0)* (rsphere**3)  #Volume cut from simulation box within which ranking was performed. Needed for scaling.
    print('Global subhalo density in full sphere: ',len(subhalos)/V_sim)

    # ------------------------------------------------------------
    # Output containers (column-wise, memory-efficient)
    # ------------------------------------------------------------
    out_sub = []
    out_gal = []

    #Set up Table of diagnostics
    diagnostics = Table(
        names=(
        'zmin',
        'zmax',
        'Nsubs',
        'Nran_shell',
        'Nran_slice',
        'Nran_vis_slice',
        'Nactual',
        'Nexpected',
        'ratio_actual_expected',
        'ratio_vis_actual',
        'R',
        'frac_rejected',
        'V_shell',
        'V_slice'    
        ),
        dtype=(
        float, float,
        int, int, int, int,
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        float
        )
    )

    diagnostics.meta['FSKY'] = fsky
    diagnostics.meta['NMULT'] = nmult
    diagnostics.meta['DELTAZ'] = DeltaZ
    diagnostics.meta['VSIM'] = V_sim
    

    
    # ------------------------------------------------------------
    # Loop over redshift slices
    # ------------------------------------------------------------
    zmax_subhalo_insufficient = None
    for zmin, zmax in zip(z_bins[:-1], z_bins[1:]):

       

        zmin_ex=np.maximum(zmin_cat,zmax-DeltaZ) # The lower redshift of the shell of randoms used for finding matches
        # Boolean masks (cheap views)
        #   All sub haloes in he narrow slice we are populating
        sub_m = (sub_z >= zmin) & (sub_z < zmax)
        #   All randoms in the wider shell that would be in the catalogue if moved to zmax of the slice
        if Rejection: #keep all galaxies that might be visible at some point in the slice (and later reject those that aren't visible at their assigned redhsift)
            ran_m = (ran_z >= zmin_ex) & (ran_z < zmax) & (ran_zmax>=zmin)
        else: #only keep galaxies visible throughout the slice
            ran_m = (ran_z >= zmin_ex) & (ran_z < zmax) & (ran_zmax>=zmax)

        Nran_slice=np.count_nonzero((ran_z >= zmin) & (ran_z < zmax)) #used to determine the random count to get a 2nd check on the normalization
        Nran_vis_slice=np.count_nonzero((ran_z >= zmin) & (ran_z < zmax) & (ran_zmax >= zmax))
        
       
        if not sub_m.any() or not ran_m.any(): #If nothing in one or other slice move to next slice
            continue

        # Indices
        sub_idx = np.nonzero(sub_m)[0] #returns as a numpy array the indices of the subhaloes in the redshift slice i.e. for which the mask is True
        ran_idx = np.nonzero(ran_m)[0] #returns as a numpy array the indices of the random galaxies in the redshift slice i.e. for which the mask is True
        Nran=len(ran_idx) #number of random galaxies in the shell

        # volume of the slice we are populating
        V_slice = fsky * (4.0*np.pi/3.0)*( (ca.cosmo.comoving_distance(zmax).value)**3 -(ca.cosmo.comoving_distance(zmin).value)**3  ) 
        # volume of shell in random catalogue
        V_shell = fsky* nmult * (4.0*np.pi/3.0)*( (ca.cosmo.comoving_distance(zmax).value)**3 -(ca.cosmo.comoving_distance(zmin_ex).value)**3  ) 
        # volume ratio needed for scaling when making matches (so that we are matching in number density)
        R = V_sim / V_shell 

        ## Diagnostic to check that on average we match the expected number of galaxies
        Nexpected_slice = V_slice*(Nran/V_shell)
        Ntot_expected += Nexpected_slice
        Ntot_ran_vis += Nran_vis_slice / nmult
        ##
        
        

        # ------------------------------------------------------------
        # Pure luminosity SHAM mapping
        # ------------------------------------------------------------
        target_rank = np.rint(i1[sub_idx] / R).astype(np.int64) - 1
        
        
        # objects that can actually be matched
        valid = (target_rank >= 0) & (target_rank < Nran)
        
        
        if not np.any(valid):
            continue
        
        sub_idx_valid = sub_idx[valid]
        match_local = target_rank[valid]

        if Rejection:
            matched_local = match_local.copy()
            sub_z_match = sub_z[sub_idx_valid] #redshifts of the matched subhalos
            matched_zmax = ran_zmax[ran_idx[matched_local]] #zmax values of the corresponding random galaxies

            good = matched_zmax >= sub_z_match #only keep those that would be visible at their assigned redshift
            print("Tested=", len(good),"Matched=", np.sum(good),"Rejected=", np.sum(~good),"Frac rejected=", np.mean(~good))

            sub_idx_valid = sub_idx_valid[good] #update the matches to keep only the good ones
            match_local = matched_local[good]
            frac_rejected = np.mean(~good)
        else:
            frac_rejected = 0.0

               
        #compute diagnostics
        Nactual_slice=len(sub_idx_valid)
   

        ratio_actual_expected = (
            Nactual_slice / Nexpected_slice
            if Nexpected_slice > 0 else np.nan
        )

        ratio_vis_actual = (
            (Nran_vis_slice / nmult) / Nactual_slice
            if Nactual_slice > 0 else np.nan
        )

        

        
       
        Ntot_actual += Nactual_slice
        Nsubs_shell=sub_m.sum()
        diagnostics.add_row(
            (
                zmin,
                zmax,
                Nsubs_shell,
                Nran,
                Nran_slice,
                Nran_vis_slice,
                Nactual_slice,
                Nexpected_slice,
                ratio_actual_expected,
                ratio_vis_actual,
                R,
                frac_rejected,
                V_shell,
                V_slice
            )
        )
                
        if Nactual_slice > 0: print("Visible/Mock ratio =",(Nran_vis_slice / nmult) / Nactual_slice)
        if (Nactual_slice==Nsubs_shell): 
            print('**WARNING: Subhalo density is insufficient to match the faintest galaxies in the shell resulting Ratio<1.')
            zmax_subhalo_insufficient = zmax
        print(rf"Processing slice:{zmin:.3f}<z<{zmax:.3f} Nsubs={Nsubs_shell} Nrans={Nran} R={R:.2f} Nactual={Nactual_slice} Nexp={Nexpected_slice:.2f}  Ratio={(Nactual_slice/Nexpected_slice):.2f} nden_sub={(Nsubs_shell/V_slice):.5f} nden_gal={(Nran/V_shell):.6f} Nran_slice/nmult={(Nran_slice/nmult):.2f} Nran_vis_slice/nmult={(Nran_vis_slice/nmult):.2f}")

        
        #  code to do secondary matching

        rng = np.random.default_rng()
        f_test = rng.random(len(sub_idx_valid))
        
        colour = (ran_gp - ran_rp).astype(np.float32)
        fracz = subhalos['FracZpeak'][sub_idx_valid]


        matched_gal_ids = np.empty(len(sub_idx_valid), dtype=ran_id.dtype)

        if (N_neigh != 0): # Do secondary matching over N_neigh neighbours

            half = N_neigh // 2

            for i, (j0, f) in enumerate(zip(match_local, fracz)):

                lo = max(0, j0 - half)
                hi = min(len(ran_idx), j0 + half + 1)

                # candidate galaxies in this luminosity neighbourhood
                neigh = np.arange(lo, hi)

                # sort by colour (bluest -> reddest)
                colour_order = np.argsort(colour[ran_idx[neigh]])

                nloc = len(colour_order)

                # percentile position
                k = min(nloc - 1,max(0,int(np.floor((1-f) * (nloc - 1)))))  

        

                chosen = neigh[colour_order[k]]

                matched_gal_ids[i] = ran_id[ran_idx[chosen]]

        else: #only do the primary matching
                matched_gal_ids = ran_id[ran_idx[match_local]]  
                
              
        # ------------------------------------------------------------
        # Store results
        # ------------------------------------------------------------
        out_sub.extend(sub_id[sub_idx_valid])
        out_gal.extend(matched_gal_ids)

    ##   
    print()
    print("="*80)
    print("FINAL SUMMARY")
    print("="*80)
    if zmax_subhalo_insufficient is not None:
        print(f"Subhalo density insufficient up to z = {zmax_subhalo_insufficient:.3f}")
    else:
        print("Subhalo density was sufficient at all redshifts")
    print(f"Matched subhalos = {len(out_sub)}")
    print(f"Matched galaxies = {len(out_gal)}")
    print("=")
    print()
    print("Expected from shell abundances =", Ntot_expected)
    print("Actually matched              =", Ntot_actual)
    print("Ratio                         =", Ntot_actual/Ntot_expected)
    print("Sum visible randoms =", Ntot_ran_vis)
    print("Matched galaxies    =", Ntot_actual)
    print("Ratio               =", Ntot_actual/Ntot_ran_vis)


    #Save diagnostics to table from which we will make diagnostic plots
    diagnostics.meta['N_MATCHED'] = len(out_sub)
    diagnostics.meta['N_EXPECTED'] = Ntot_expected
    diagnostics.meta['N_VIS_TOTAL'] = Ntot_ran_vis
    diagnostics.write(diagnostics_path,overwrite=True)
    ##
    # ------------------------------------------------------------
    # When all slices are processed return the complete matched list
    # as two corresponding numpy arrays
    # ------------------------------------------------------------
    return np.asarray(out_sub), np.asarray(out_gal), zmax_subhalo_insufficient
    
 
  







# Returns the fraction of sky in a defined mask and optionally a set of random RAs and DECs within it
def generate_randoms_in_mask(usemask=True,fsky_only=False,nr=10_000_000,tiles_file=None,reg=None,RAmin=None,RAmax=None,DECmin=None,DECmax=None):

    if usemask:
        tiles = fitsio.read(tiles_file) #read the official tiles/mask file
        rng = np.random.default_rng()

        ra_chunks = []
        dec_chunks = []

        nacc = 0
        ntried = 0

        while nacc < nr:

            # estimate how many more points we need
            nneed = nr - nacc

            # sample until correct number successful
            #ntry = max(100000, int(2.0 * nneed))
            ntry=500000
            ntried += ntry  #keep track of many randoms tried
    
            ra = 360.0 * rng.random(ntry)
    
            sindec = -1.0 +2.0*rng.random(ntry)          
    
            dec = np.degrees(np.arcsin(sindec))
    
            if (reg=='N'):
                mask = is_point_in_desi(tiles, ra, dec) # in whole of official DESI DR2 mask
                mask = mask & (ra>85.0) & (ra<305.0) &  (dec>32.375) #cut to portion in region North
            elif (reg=='S'):    
                mask = (ra>85.0) & (ra<305.0) &  (dec>32.375) # region that includes DESI North and none of DESI South
                mask = ~mask & is_point_in_desi(tiles, ra, dec) # in whole of official DESI DR2 mask but with ~mask excludes the North
            else:
                print('Invalid value for reg:',reg)
        
    
            ra_keep = ra[mask]
            dec_keep = dec[mask]
    
            ra_chunks.append(ra_keep)
            dec_chunks.append(dec_keep)
    
            nacc += len(ra_keep)
            print(f"Accepted {nacc:,}/{nr:,}  fsky={nacc/ntried:.4f}", end="\r")
    
        print(f"Accepted {nacc:,}/{nr:,}  fsky={nacc/ntried:.4f}")
        print('nr=',nr)
        ra_out = np.concatenate(ra_chunks)[:nr]
        dec_out = np.concatenate(dec_chunks)[:nr]
        fsky=nacc/ntried
        if fsky_only:
            del ra_out,dec_out
    else:
        if ~fsky_only:
            # Generate RA and DEC that are randomly uniformly distributed within the specified RA and DEC limits
            rng = np.random.default_rng()
            ra_out=RAmin+(RAmax-RAmin)*rng.random(nr) #uniform in RA
            sindecmin= np.sin(np.deg2rad(DECmin))
            sindecmax= np.sin(np.deg2rad(DECmax))
            sindec=sindecmin+(sindecmax-sindecmin)*rng.random(nr)
            dec_out=np.rad2deg(np.arcsin(sindec)) #uniform in sin(DEC) 
        fsky= (RAmax-RAmin)*(np.pi/180.)*(sindecmax-sindecmin)/(4.0*np.pi)

    # If fsky_only=True, we return just its estimate and no random points within it
    if fsky_only:
        return fsky
    else:    
        return ra_out,dec_out,fsky
