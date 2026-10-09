"""
make_checkpoint.py — Read the full Uchuu halo catalogue, apply scatter to
Vpeak, sort, mask to the survey footprint, and write a compact checkpoint
file containing only the subhalos needed for the SHAM step.

This is the expensive step: it requires an exclusive CPU node at NERSC
(~128 GB RAM) and takes ~30 minutes.  Everything downstream (make_mock.py)
can run on a login node using the checkpoint file.

The checkpoint file is keyed by reg, obs_seed, sig_lgv and snapshot_index,
so different scatter values each produce a distinct file and can be run in
parallel as separate SLURM jobs.

Usage examples
--------------
# North, default scatter and seed:
python make_checkpoint.py --reg N

# South, sig_lgv=0.15, non-default seed:
python make_checkpoint.py --reg S --sig_lgv 0.15 --obs_seed 1

# North, different snapshot:
python make_checkpoint.py --reg N --snapshot_index 43

SLURM example
-------------
salloc --nodes=1 --qos=interactive --constraint=cpu \\
       --account=desi --mem=128GB --time=02:00:00
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
python make_checkpoint.py --reg N --sig_lgv 0.10
"""

# --- multiprocessing start method must be set before any other imports ---
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import argparse
import gc
import glob
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table
from scipy.interpolate import interp1d
import fitsio
from desimodel.footprint import is_point_in_desi

# Pipeline modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DEFAULT_OBS_SEED,
    DEFAULT_SIG_LGV,
    MIN_VPEAK,
    N_NEIGH_VPEAK,
    LBOX,
    FRAC_BOX,
    UCHUU_COLUMNS,
    MIN_LOG_MVIR,
    SNAPSHOT_INDEX,
    UCHUU_HALODIR_TEMPLATE,
    TILES_FILE,
    NORTH_RA_MIN,
    NORTH_RA_MAX,
    NORTH_DEC_MIN,
    cosmo_uchuu,  # Uchuu cosmology used ONLY for distance→redshift conversion
)
from pipeline_utils import make_paths, write_fits_meta, setup_logging

# MockCode is imported after argument parsing so that the fork start method
# is already in effect before any multiprocessing objects are created.


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--reg", required=True, choices=["N", "S"],
        help="Photometric region to process: N (North) or S (South)",
    )
    p.add_argument(
        "--obs_seed", type=int, default=DEFAULT_OBS_SEED,
        help=(
            f"Observer position seed. 0 places the observer at the box "
            f"centre; any other value places it randomly. (default: {DEFAULT_OBS_SEED})"
        ),
    )
    p.add_argument(
        "--sig_lgv", type=float, default=DEFAULT_SIG_LGV,
        help=(
            f"Gaussian scatter in log10(Vpeak) in dex applied before "
            f"ranking subhalos. 0.0 means no scatter. (default: {DEFAULT_SIG_LGV})"
        ),
    )
    p.add_argument(
        "--min_vpeak", type=float, default=MIN_VPEAK,
        help=(
            f"Discard subhalos with Vpeak below this value [km/s]. "
            f"Sets the minimum redshift at which the mock is complete. "
            f"(default: {MIN_VPEAK})"
        ),
    )
    p.add_argument(
        "--n_neigh_vpeak", type=int, default=N_NEIGH_VPEAK,
        help=(
            f"Number of Vpeak neighbours used to compute the secondary "
            f"matching statistic FracZpeak. (default: {N_NEIGH_VPEAK})"
        ),
    )
    p.add_argument(
        "--snapshot_index", type=int, default=SNAPSHOT_INDEX,
        help=(
            f"Uchuu snapshot index to use. Determines the simulation "
            f"redshift. (default: {SNAPSHOT_INDEX}, z≈0.19)"
        ),
    )
    p.add_argument(
        "--plot", action="store_true",
        help="Produce diagnostic plots of the Vpeak scatter (requires matplotlib).",
    )
    return p.parse_args()


# =============================================================================
# Observer position
# =============================================================================

def observer_position(obs_seed, lbox, log):
    """
    Return the (x, y, z) position of the observer in the simulation box.

    seed=0  → centre of the box.
    seed>0  → random position drawn with a fixed offset from obs_seed so
              that it differs from the direction seed used elsewhere.
    """
    if obs_seed == 0:
        xobs = yobs = zobs = lbox / 2.0
        log.info(f"Observer at box centre: ({xobs}, {yobs}, {zobs})")
    else:
        offset = 58
        rng  = np.random.default_rng(obs_seed + offset)
        xobs = lbox * rng.random()
        yobs = lbox * rng.random()
        zobs = lbox * rng.random()
        log.info(f"Observer position (seed={obs_seed}): ({xobs:.1f}, {yobs:.1f}, {zobs:.1f})")
    return xobs, yobs, zobs


# =============================================================================
# Coordinate computation
# =============================================================================

def add_sky_coords(subhalos, obs_seed, log):
    """
    Rotate positions and velocities by a random Euler rotation, then compute
    RA, DEC, cosmological redshift (Zu) and observed redshift (Zobs) for
    each subhalo.  Concentration (Rvir/rs) is also computed here and the
    raw Rvir, rs, and velocity columns are dropped to save memory.

    Modifies subhalos in place and returns it.
    """
    import MockCode as mc

    t0 = time.time()
    phi, theta, psi = mc.random_euler_angles(seed=obs_seed)
    log.info(f"Euler angles: phi={phi:.4f}  theta={theta:.4f}  psi={psi:.4f}")
    R = mc.euler_rotation_matrix(phi, theta, psi)

    # Rotate velocities
    vxr = R[0,0]*subhalos['vx'] + R[0,1]*subhalos['vy'] + R[0,2]*subhalos['vz']
    vyr = R[1,0]*subhalos['vx'] + R[1,1]*subhalos['vy'] + R[1,2]*subhalos['vz']
    vzr = R[2,0]*subhalos['vx'] + R[2,1]*subhalos['vy'] + R[2,2]*subhalos['vz']
    subhalos['vx'] = vxr;  subhalos['vy'] = vyr;  subhalos['vz'] = vzr
    del vxr, vyr, vzr

    # Rotate positions
    xr = R[0,0]*subhalos['x'] + R[0,1]*subhalos['y'] + R[0,2]*subhalos['z']
    yr = R[1,0]*subhalos['x'] + R[1,1]*subhalos['y'] + R[1,2]*subhalos['z']
    zr = R[2,0]*subhalos['x'] + R[2,1]*subhalos['y'] + R[2,2]*subhalos['z']
    subhalos['x'] = xr;  subhalos['y'] = yr;  subhalos['z'] = zr
    del xr, yr, zr
    log.info(f"Rotation done in {time.time()-t0:.1f}s")

    t0 = time.time()
    dist = np.sqrt(subhalos['x']**2 + subhalos['y']**2 + subhalos['z']**2)
    subhalos['dist'] = dist
    subhalos['DEC']  = np.degrees(np.arcsin(subhalos['z'] / dist))
    subhalos['RA']   = np.degrees(np.arctan2(subhalos['y'], subhalos['x'])) % 360

    # Radial velocity / c
    c_light = 299792.458  # km/s
    vr_c = (
        subhalos['x']*subhalos['vx'] +
        subhalos['y']*subhalos['vy'] +
        subhalos['z']*subhalos['vz']
    ) / (c_light * dist)

    # Drop velocities — no longer needed
    for col in ('vx', 'vy', 'vz'):
        subhalos.remove_column(col)
    gc.collect()

    # Concentration
    subhalos['conc'] = subhalos['Rvir'] / subhalos['rs']
    subhalos.remove_column('Rvir')
    subhalos.remove_column('rs')
    gc.collect()

    # Redshift lookup table: comoving distance → redshift.
    # Uses the Uchuu cosmology (Om0=0.3089) because Uchuu particle positions
    # are defined in that cosmology.  Downstream scripts treat the resulting
    # redshifts as observed quantities and convert back to distances using the
    # standard analysis cosmology (Om0=0.313), exactly as is done for real data.
    z_grid  = np.linspace(0.0, 1.0, 10000)
    Dc_grid = cosmo_uchuu.comoving_distance(z_grid).value  # Mpc/h
    z_of_Dc = interp1d(Dc_grid, z_grid, bounds_error=False, fill_value="extrapolate")

    subhalos['Zu']   = z_of_Dc(dist)
    subhalos['Zobs'] = (1.0 + subhalos['Zu']) * (1.0 + vr_c) - 1.0
    del z_grid, Dc_grid, z_of_Dc, vr_c, dist

    log.info(f"RA/DEC/redshift computation done in {time.time()-t0:.1f}s")
    return subhalos


# =============================================================================
# Sky mask
# =============================================================================

def apply_sky_mask(subhalos, reg, log):
    """
    Restrict subhalos to the DESI DR2 survey footprint for the given region.

    North: inside the official DESI mask AND RA∈[85,305] AND DEC>32.375
    South: inside the official DESI mask AND NOT North
    """
    t0 = time.time()
    log.info(f"Applying DESI DR2 sky mask for region {reg}")
    tiles = fitsio.read(str(TILES_FILE))

    in_desi = is_point_in_desi(tiles, subhalos['RA'], subhalos['DEC'])
    north   = (
        (subhalos['RA']  > NORTH_RA_MIN) &
        (subhalos['RA']  < NORTH_RA_MAX) &
        (subhalos['DEC'] > NORTH_DEC_MIN)
    )

    if reg == 'N':
        mask = in_desi & north
    else:
        mask = in_desi & ~north

    del in_desi, north, tiles
    subhalos = subhalos[mask]
    log.info(f"After mask: {len(subhalos):,} subhalos  ({time.time()-t0:.1f}s)")
    return subhalos


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Import MockCode here (after fork start method is set and args parsed).
    import MockCode as mc

    log = setup_logging("make_checkpoint")
    log.info(
        f"reg={args.reg}  obs_seed={args.obs_seed}  sig_lgv={args.sig_lgv}  "
        f"min_vpeak={args.min_vpeak}  n_neigh_vpeak={args.n_neigh_vpeak}  "
        f"snapshot_index={args.snapshot_index}"
    )

    # Check snapshot index is valid
    if args.snapshot_index not in mc.redshift_table:
        log.error(
            f"snapshot_index={args.snapshot_index} not in redshift_table. "
            f"Valid values: {sorted(mc.redshift_table.keys())}"
        )
        sys.exit(1)
    redshift = mc.redshift_table[args.snapshot_index]
    log.info(f"Snapshot {args.snapshot_index} → z={redshift}")

    # Output path
    paths = make_paths(
        reg=args.reg,
        obs_seed=args.obs_seed,
        sig_lgv=args.sig_lgv,
        min_vpeak=args.min_vpeak,
        n_neigh_vpeak=args.n_neigh_vpeak,
    )
    out_path = paths["subhalo_checkpoint"]
    log.info(f"Output checkpoint: {out_path}")

    # -------------------------------------------------------------------------
    # 1. Read full Uchuu catalogue in parallel
    # -------------------------------------------------------------------------
    halodir   = UCHUU_HALODIR_TEMPLATE.format(index=args.snapshot_index)
    halofiles = sorted(glob.glob(f"{halodir}/halolist_*.h5"))
    if not halofiles:
        log.error(f"No halo files found in {halodir}")
        sys.exit(1)
    log.info(f"Found {len(halofiles)} halo files in {halodir}")

    xobs, yobs, zobs = observer_position(args.obs_seed, LBOX, log)

    t0 = time.time()
    subhalos = mc.read_halolist_parallel(
        halofiles,
        randseed=0,
        columns=UCHUU_COLUMNS,
        min_logMvir=MIN_LOG_MVIR,
        min_Vpeak=args.min_vpeak,
        lbox=LBOX,
        frac_box=FRAC_BOX,
        xobs=xobs,
        yobs=yobs,
        zobs=zobs,
    )
    log.info(f"Read {len(subhalos):,} subhalos in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------------------
    # 2. Rotate coordinates; compute RA, DEC, redshift
    # -------------------------------------------------------------------------
    subhalos = add_sky_coords(subhalos, args.obs_seed, log)

    # -------------------------------------------------------------------------
    # 3. Sort by Vpeak, compute FracZpeak, apply scatter, re-sort, trim
    # -------------------------------------------------------------------------
    t0 = time.time()
    subhalos = mc.make_all_columns_contiguous(subhalos)
    log.info(f"Made contiguous in {time.time()-t0:.1f}s")

    subhalos = mc.pre_sort_subhalos(
        subhalos,
        N_neigh=args.n_neigh_vpeak,
        sig_lgv=args.sig_lgv,
        plot=args.plot,
    )

    t0 = time.time()
    subhalos = mc.filter_table_staged(
        subhalos, subhalos['keep'], drop_mask=True, mask_name='keep'
    )
    gc.collect()
    log.info(f"Distance-dependent Vpeak cut: {len(subhalos):,} remain ({time.time()-t0:.1f}s)")

    # -------------------------------------------------------------------------
    # 4. Apply sky mask
    # -------------------------------------------------------------------------
    subhalos = apply_sky_mask(subhalos, args.reg, log)

    # Rename 'id' → 'ID' for consistency with downstream scripts
    if 'id' in subhalos.colnames:
        subhalos.rename_column('id', 'ID')

    # -------------------------------------------------------------------------
    # 5. Write checkpoint file with metadata
    # -------------------------------------------------------------------------
    write_fits_meta(subhalos.meta, {
        "STAGE":      "make_checkpoint",
        "REG":        args.reg,
        "OBS_SEED":   args.obs_seed,
        "SIG_LGV":    args.sig_lgv,
        "MIN_VPEAK":  args.min_vpeak,
        "NNVPEAK":    args.n_neigh_vpeak,
        "SNAPIDX":    args.snapshot_index,
        "SNAPZ":      redshift,
        "LBOX":       LBOX,
        "FRAC_BOX":   FRAC_BOX,
        # Store mask boundaries so downstream scripts can verify compatibility
        "RAMIN":      NORTH_RA_MIN,
        "RAMAX":      NORTH_RA_MAX,
        "DECMIN":     NORTH_DEC_MIN,
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subhalos.write(str(out_path), overwrite=True)
    log.info(f"Written checkpoint ({len(subhalos):,} subhalos) in {time.time()-t0:.1f}s")
    log.info(f"→ {out_path}")
    log.info("make_checkpoint.py complete.")


if __name__ == "__main__":
    main()
