"""
make_checkpoint_snap.py
-----------------------
Read one Uchuu snapshot, sort subhaloes by scattered Vpeak, compute
FracZpeak, apply the distance-dependent trimming mask, and write the
checkpoint FITS file to the lightcone model directory.

This is a lightcone-only companion to make_checkpoint.py.
The cut-sky pipeline is untouched.

Key differences from make_checkpoint.py
----------------------------------------
* --snapshot argument selects which Uchuu snapshot (and hence redshift)
  to process.  The halo directory is resolved via UCHUU_HALODIR_TEMPLATE.
* obs_seed is fixed to 0 (DEFAULT_OBS_SEED) so that all snapshots share
  the same observer position.  Different observers per snapshot would
  corrupt the lightcone geometry.
* Output goes to the lightcone model directory rather than the cut-sky
  model directory.

Usage
-----
Run on an exclusive SLURM node (same requirements as make_checkpoint.py):

    python make_checkpoint_snap.py --snapshot 47 --reg NS \
        --sig_lgv_ref 0.18 --sig_lgv_slope 0.30 \
        --f_colour 0.00

The --reg NS option writes both N and S checkpoint files in one Uchuu
read (recommended to save I/O time).
"""

import multiprocessing
multiprocessing.set_start_method('fork', force=True)  # must be before other imports

import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table

import MockCode as mc
from pipeline_utils import make_lightcone_paths, write_fits_meta, setup_logging
from config import (
    cosmo_uchuu,
    UCHUU_HALODIR_TEMPLATE,
    UCHUU_COLUMNS,
    MIN_LOG_MVIR,
    TILES_FILE,
    LBOX,
    FRAC_BOX,
    MIN_VPEAK,
    N_NEIGH_VPEAK,
    DEFAULT_SIG_LGV_REF,
    DEFAULT_SIG_LGV_SLOPE,
    DEFAULT_LGV_PIVOT,
    DEFAULT_F_COLOUR,
    DEFAULT_ALPHA,
    DEFAULT_OBS_SEED,
    DEFAULT_SEED_FCOLOUR,
    NORTH_RA_MIN,
    NORTH_RA_MAX,
    NORTH_DEC_MIN,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Build lightcone checkpoint for one Uchuu snapshot.'
    )
    p.add_argument('--snapshot',      type=int,   required=True,
                   help='Uchuu snapshot index (e.g. 45 for z=0.19).')
    p.add_argument('--reg',           type=str,   default='NS',
                   help='Region(s): N, S, or NS (default NS).')
    p.add_argument('--sig_lgv_ref',   type=float, default=DEFAULT_SIG_LGV_REF)
    p.add_argument('--sig_lgv_slope', type=float, default=DEFAULT_SIG_LGV_SLOPE)
    p.add_argument('--lgv_pivot',     type=float, default=DEFAULT_LGV_PIVOT)
    p.add_argument('--f_colour',      type=float, default=DEFAULT_F_COLOUR)
    p.add_argument('--seed_fcolour',  type=int,   default=DEFAULT_SEED_FCOLOUR)
    p.add_argument('--n_neigh_vpeak', type=int,   default=N_NEIGH_VPEAK)
    p.add_argument('--min_vpeak',     type=float, default=MIN_VPEAK)
    return p.parse_args()


def get_observer_position(obs_seed=DEFAULT_OBS_SEED, lbox=LBOX):
    """Same logic as cut-sky pipeline: seed + offset 58."""
    offset = 58
    rng  = np.random.default_rng(obs_seed + offset)
    xobs = lbox * rng.random()
    yobs = lbox * rng.random()
    zobs = lbox * rng.random()
    return xobs, yobs, zobs


def compute_subhalo_redshifts(subhalos):
    """
    Compute 3-D comoving distance and invert to cosmological redshift
    using the Uchuu cosmology (cosmo_uchuu from config.py).
    """
    dist = np.sqrt(
        subhalos['x']**2 + subhalos['y']**2 + subhalos['z']**2
    ).astype(np.float32)
    subhalos['dist'] = dist

    # Invert d_comoving(z) via interpolation on a fine grid
    z_grid = np.linspace(0.0, 0.65, 10_000)
    d_grid = cosmo_uchuu.comoving_distance(z_grid).value  # Mpc/h (H0=100)
    zu = np.interp(dist, d_grid, z_grid).astype(np.float32)
    subhalos['Zu'] = zu
    return subhalos


def add_sky_coords_and_velocities(subhalos, obs_seed=DEFAULT_OBS_SEED):
    """
    Add RA, Dec, and line-of-sight peculiar velocity.
    Uses a random Euler rotation with the same seed as the cut-sky pipeline.
    For obs_seed=0 the rotation is the identity (phi=theta=psi=0).
    """
    phi, theta, psi = mc.random_euler_angles(seed=obs_seed)
    R = mc.euler_rotation_matrix(phi, theta, psi)

    xyz = np.stack(
        [subhalos['x'], subhalos['y'], subhalos['z']], axis=1
    )
    xyz_rot = xyz @ R.T

    dist = subhalos['dist']
    safe = dist > 0

    ra  = np.zeros(len(subhalos), dtype=np.float32)
    dec = np.zeros(len(subhalos), dtype=np.float32)
    ra[safe]  = (np.degrees(
        np.arctan2(xyz_rot[safe, 1], xyz_rot[safe, 0])
    ) % 360.0).astype(np.float32)
    dec[safe] = np.degrees(
        np.arcsin(np.clip(xyz_rot[safe, 2] / dist[safe], -1.0, 1.0))
    ).astype(np.float32)

    subhalos['RA']  = ra
    subhalos['DEC'] = dec

    # Line-of-sight peculiar velocity
    v_xyz = np.stack(
        [subhalos['vx'], subhalos['vy'], subhalos['vz']], axis=1
    ) @ R.T
    los_hat = np.zeros_like(xyz_rot)
    los_hat[safe] = xyz_rot[safe] / dist[safe, None]
    v_los = np.einsum('ij,ij->i', v_xyz, los_hat).astype(np.float32)
    subhalos['Vlos'] = v_los

    return subhalos


def mask_to_footprint(subhalos, reg):
    """Apply DESI DR2 footprint mask for region reg (N or S)."""
    from desimodel.footprint import is_point_in_desi
    import fitsio
    tiles = fitsio.read(str(TILES_FILE))
    ra    = subhalos['RA']
    dec   = subhalos['DEC']

    in_desi = is_point_in_desi(tiles, ra, dec)
    is_north = (ra > NORTH_RA_MIN) & (ra < NORTH_RA_MAX) & (dec > NORTH_DEC_MIN)

    if reg == 'N':
        mask = in_desi & is_north
    elif reg == 'S':
        mask = in_desi & ~is_north
    else:
        raise ValueError(f'Unknown reg: {reg}')

    return subhalos[mask]


def process_region(subhalos_full, reg, snapshot, redshift_snap, args, logger):
    """Mask → pre-sort → write checkpoint for one region."""
    logger.info(f'[snap{snapshot:02d}/{reg}] Masking to footprint...')
    subhalos = mask_to_footprint(subhalos_full, reg)
    logger.info(f'[snap{snapshot:02d}/{reg}] {len(subhalos):,} subhaloes after footprint mask.')

    logger.info(f'[snap{snapshot:02d}/{reg}] Pre-sorting (scatter + FracZpeak)...')
    subhalos = mc.pre_sort_subhalos(
        subhalos,
        N_neigh=args.n_neigh_vpeak,
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        lgv_pivot=args.lgv_pivot,
        f_colour=args.f_colour,
        seed_fcolour=args.seed_fcolour,
    )

    # Apply distance-dependent trimming (keep=True rows only)
    keep_mask = subhalos['keep']
    subhalos  = mc.filter_table_staged(subhalos, keep_mask)
    logger.info(f'[snap{snapshot:02d}/{reg}] {len(subhalos):,} subhaloes after Vpeak trim.')

    outpath = make_lightcone_paths(
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        f_colour=args.f_colour,
        alpha=DEFAULT_ALPHA,
        snapshot=snapshot,
        reg=reg,
    )['subhalo_fits']

    meta = {
        'SNAP':      snapshot,
        'ZSNAP':     redshift_snap,
        'REG':       reg,
        'OBS_SEED':  DEFAULT_OBS_SEED,
        'SIGLGVREF': args.sig_lgv_ref,
        'SIGLGVSLP': args.sig_lgv_slope,
        'LGVPIVOT':  args.lgv_pivot,
        'FCOLOUR':   args.f_colour,
        'SEEDFC':    args.seed_fcolour,
        'NNEIGHVPK': args.n_neigh_vpeak,
        'MINVPEAK':  args.min_vpeak,
    }

    logger.info(f'[snap{snapshot:02d}/{reg}] Writing checkpoint to {outpath}')
    subhalos.meta.update(meta)
    write_fits_meta(subhalos.meta, meta)   # attaches PIPE_VER, CREATED
    subhalos.write(str(outpath), overwrite=True)
    logger.info(f'[snap{snapshot:02d}/{reg}] Done. {len(subhalos):,} subhaloes written.')


def main():
    args     = parse_args()
    snapshot = args.snapshot
    regions  = list(args.reg)   # 'NS' → ['N','S']

    if snapshot not in mc.redshift_table:
        raise ValueError(
            f'Snapshot {snapshot} not in redshift_table. '
            f'Valid: {sorted(mc.redshift_table.keys())}'
        )
    redshift_snap = mc.redshift_table[snapshot]

    # Create lightcone directory
    lc_paths      = make_lightcone_paths(
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        f_colour=args.f_colour,
        alpha=DEFAULT_ALPHA,
    )
    lightcone_dir = lc_paths['lightcone_dir']
    lightcone_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(
        f'make_checkpoint_snap{snapshot:02d}',
        log_dir=lightcone_dir / 'logs',
    )
    logger.info(f'Snapshot {snapshot}  z={redshift_snap}  regions={regions}')
    logger.info(f'Lightcone dir: {lightcone_dir}')

    # Observer position — always obs_seed=0 for lightcones
    xobs, yobs, zobs = get_observer_position()
    logger.info(f'Observer (seed=0): ({xobs:.2f}, {yobs:.2f}, {zobs:.2f}) Mpc/h')

    # Locate halo files
    halodir    = UCHUU_HALODIR_TEMPLATE.format(index=snapshot)
    file_paths = sorted(glob.glob(os.path.join(halodir, '*.hdf5')))
    if not file_paths:
        raise FileNotFoundError(f'No hdf5 files in {halodir}')
    logger.info(f'Reading {len(file_paths)} halo files from {halodir}')

    # Read all halo files in parallel
    t0 = time.time()
    subhalos = mc.read_halolist_parallel(
        file_paths,
        randseed=0,
        columns=list(UCHUU_COLUMNS),
        min_logMvir=MIN_LOG_MVIR,
        min_Vpeak=args.min_vpeak,
        lbox=LBOX,
        frac_box=FRAC_BOX,
        xobs=xobs,
        yobs=yobs,
        zobs=zobs,
    )
    logger.info(f'Read {len(subhalos):,} subhaloes in {time.time()-t0:.1f}s')

    # Rename id → ID for consistency with cut-sky pipeline
    subhalos.rename_column('id', 'ID')

    # Compute distances, redshifts, sky coords and velocities
    subhalos = compute_subhalo_redshifts(subhalos)
    subhalos = add_sky_coords_and_velocities(subhalos, obs_seed=DEFAULT_OBS_SEED)

    # Process each region
    for reg in regions:
        process_region(subhalos, reg, snapshot, redshift_snap, args, logger)

    logger.info('make_checkpoint_snap.py complete.')


if __name__ == '__main__':
    main()
