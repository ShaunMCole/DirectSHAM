#Functions used in the mock creation pipeline MockVpeakZpeak.ipynb
from numba import njit,prange
import multiprocessing
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

    Optional correlated scatter (alpha_v != 0):
      When alpha_v > 0 and f_col_proxy is provided, the scatter epsilon is
      drawn as a linear combination of an assembly-history term and an
      independent Gaussian (equation 7 of Cole et al. 2026):

          x_col = sqrt(3) * (2*f_col_proxy - 1)   [zero mean, unit variance]
          eta   ~ N(0, 1)
          eps   = sigma_v * [alpha_v * x_col + sqrt(1 - alpha_v^2) * eta]

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
            # Rescale f_col_proxy to zero mean, unit variance
            x_col = np.sqrt(3.0) * (1.0 - 2.0 * np.asarray(f_col_proxy))
            eta   = alpha_v * x_col + np.sqrt(1.0 - alpha_v**2) * eta
            del x_col
        scatter = sigma_arr * eta
        del sigma_arr, eta
        return x * 10.0**scatter
    elif sigma is None or sigma == 0.0:
        return x
    else:
        eta = np.random.normal(loc=0.0, scale=sigma, size=len(x))
        return x * 10.0**eta

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
    keep = (results['Vpeak'] >  min_Vpeak)
    nkeep = np.sum(keep)
    pkeep = 100 * nkeep / len(keep)
    print(f'{shortname} Global Vpeak cut: keeping {nkeep}/{len(keep)} ({pkeep:.1f}%) entries')

    for col in results:
        results[col] = results[col][keep]
        

    # --- mass cut ---
    keep = (results['Mvir_all'] > 10**min_logMvir)
    nkeep = np.sum(keep)
    pkeep = 100 * nkeep / len(keep)
    print(f'{shortname} Mvir_all cut: keeping {nkeep}/{len(keep)} ({pkeep:.1f}%) entries')

    for col in results:
        results[col] = results[col][keep]

    
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

# Parallel routine to read all Uchuu subhalo files
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
        
    print("Start method:", multiprocessing.get_start_method())
    ncpus= multiprocessing.cpu_count()
    nworkers = min(nfiles,ncpus)
    print("ncpus=",ncpus,"nworkers",nworkers)

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
        with multiprocessing.Pool(nworkers) as pool:
            file_results = pool.starmap(read_halolist_file_reps, args)
            
        tot_bytes = 0
        for d in file_results:
            for v in d.values():
                tot_bytes += v.nbytes
        print(f"All file_results = {tot_bytes/1024**3:.1f} GB")

        # concatenate across files without doubling memory usage
        results = {}
        cols = list(file_results[0].keys())
        for col in cols:
            results[col] = np.concatenate([x[col] for x in file_results])
            # immediately release this column from file_results
            for x in file_results:
                del x[col]
        #free up memory        
        del file_results
        gc.collect()

    return Table(results)


# Three routines that are just-in-time compiled using numba:
    
# compute the rms scatter in quantity z among the N_neigh neighbours of each element in the array
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




# i) Sort the subhalos into decreasing order of Vpeak 
# ii) Compute fraction of neighbours with higher zpeak
#iii) Add scatter to Vpeak and sort again
# iv) Trim the catalogue by applying a tailored Vpeak(z) threshold
def pre_sort_subhalos(subhalos, N_neigh=100, sig_lgv_ref=0.0, sig_lgv_slope=0.0,
                      lgv_pivot=2.6, f_colour=0.0, seed_fcolour=42,
                      alpha_v=0.0, plot=False):
    
    start_time = time.time()
    print("Starting sort")
    subhalos.sort('Vpeak')
    print("Sort complete")
    subhalos.reverse()
    print("Reversal complete")
    end_time = time.time()
    print(f"time to sort subhalos: {end_time - start_time:.4f} seconds")

    nsub = len(subhalos)


  
    # ------------------------------------------------------------------
    # compute FRAC (local rank statistic)
    # ------------------------------------------------------------------
    start_time = time.time()
    # Convert Mpeak_Scale to log10 in-place, add noise in log space, compute FracZpeak.
    # f_colour > 0: Gaussian scatter with sigma=f_colour in log10(Mpeak_Scale) units,
    #               large enough to mix quantization bands (~0.02-0.03 apart in log).
    # f_colour = 0: uniform +-0.005 tie-breaking noise only (same effect as before).
    # All operations are in-place or on a single temporary noise array to minimise
    # memory usage — Mpeak_Scale is not needed again after FracZpeak is computed.
    mps = subhalos['Mpeak_Scale']  # pointer to existing array — writes go back to table
    rng_fc = np.random.default_rng(seed_fcolour)
    if f_colour > 0.0:
        noise = rng_fc.standard_normal(len(mps)).astype(np.float32) * np.float32(f_colour)
    else:
        noise = (rng_fc.random(len(mps)).astype(np.float32) - 0.5) * np.float32(0.01)
    np.log10(mps, out=mps)  # in-place log10; Mpeak_Scale not used again after FracZpeak
    mps += noise             # add noise in log space
    del noise
    subhalos['FracZpeak'] = compute_frac_local_mpeak(subhalos['Mpeak_Scale'], N_neigh=N_neigh)
    end_time = time.time()
    print(f"time to compute FracZpeak: {end_time - start_time:.4f} seconds")

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

        subhalos['Vpeak'] = apply_log_scatter(
            subhalos['Vpeak'],
            sig_lgv_ref=sig_lgv_ref,
            sig_lgv_slope=sig_lgv_slope,
            lgv_pivot=lgv_pivot,
            alpha_v=alpha_v,
            f_col_proxy=subhalos['FracZpeak'] if alpha_v != 0.0 else None,
        )
        if plot:
            vpeak_new = subhalos['Vpeak'][idx].copy()  # grab sample BEFORE sort

        end_time = time.time()
        print(f"time to add Vpeak scatter: {end_time - start_time:.4f} seconds")
        start_time = time.time()
        print("Starting 2nd sort")
        subhalos.sort('Vpeak')
        print("Sort complete")
        subhalos.reverse()
        print("Reversal complete")
        end_time = time.time()
        print(f"time to sort subhalos: {end_time - start_time:.4f} seconds")
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

    
    # Rank index
    start_time = time.time()
    #First check if int32 is sufficient to rank all the subhalos
    #if nsub > np.iinfo(np.int32).max:
    #        raise RuntimeError(
    #       f"{nsub=} exceeds int32 limit "
    #        f"{np.iinfo(np.int32).max}"
    #    )
    subhalos['i1'] = np.arange(1, nsub + 1, dtype=np.int64)
    print('These numbers should be the same if ranking done as expected. nsub=',len(subhalos),'?=?',np.max(subhalos["i1"]))
    print('and  1 ?=?', np.min(subhalos["i1"]))
    end_time = time.time()
    print(f"time to define rank index: {end_time - start_time:.4f} seconds")
    
    # ------------------------------------------------------------------
    # Distance dependent trimming
    # ------------------------------------------------------------------
    start_time = time.time()
    dist=subhalos['dist']
    limit = ((6.5e-07 * dist - 9.02e-04) * dist + 0.486) * dist + 10.0
    mask = subhalos['Vpeak'] >limit
    subhalos['keep'] = mask
    end_time = time.time()
    print(f"time to define distance dependent trimming mask: {end_time - start_time:.4f} seconds")

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
