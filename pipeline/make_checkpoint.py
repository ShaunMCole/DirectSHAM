"""
make_checkpoint_snap.py
-----------------------
Read one Uchuu snapshot, sort subhaloes by scattered Vpeak, compute
FracZpeak, apply the distance-dependent trimming mask, and write the
checkpoint FITS file to the lightcone model directory.

Lightcone-only companion to make_checkpoint.py. Cut-sky pipeline untouched.

Key differences from make_checkpoint.py
----------------------------------------
* --snapshot selects the Uchuu snapshot.  Halo directory resolved via
  UCHUU_HALODIR_TEMPLATE.
* obs_seed is fixed to 0 (box centre observer, no rotation).  All six
  snapshots must share the same observer position or the lightcone
  geometry is corrupted.
* Output goes to the lightcone model directory.

Usage
-----
Run on an exclusive SLURM node (~30-60 min per snapshot):

    python make_checkpoint_snap.py --snapshot 47 --reg NS \\
        --sig_lgv_ref 0.18 --sig_lgv_slope 0.30

--reg NS writes both N and S in one Uchuu read (recommended).
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import MockCode as mc
from pipeline_utils import make_lightcone_paths, write_fits_meta, setup_logging
from config import (
    DEFAULT_OBS_SEED,
    DEFAULT_SIG_LGV_REF,
    DEFAULT_SIG_LGV_SLOPE,
    DEFAULT_LGV_PIVOT,
    DEFAULT_F_COLOUR,
    DEFAULT_ALPHA,
    DEFAULT_SEED_FCOLOUR,
    MIN_VPEAK,
    N_NEIGH_VPEAK,
    LBOX,
    FRAC_BOX,
    UCHUU_COLUMNS,
    MIN_LOG_MVIR,
    UCHUU_HALODIR_TEMPLATE,
    TILES_FILE,
    NORTH_RA_MIN,
    NORTH_RA_MAX,
    NORTH_DEC_MIN,
    cosmo_uchuu,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Build lightcone checkpoint for one Uchuu snapshot.'
    )
    p.add_argument('--snapshot',      type=int,   required=True)
    p.add_argument('--reg',           type=str,   default='NS',
                   choices=['N', 'S', 'NS'])
    p.add_argument('--sig_lgv_ref',   type=float, default=DEFAULT_SIG_LGV_REF)
    p.add_argument('--sig_lgv_slope', type=float, default=DEFAULT_SIG_LGV_SLOPE)
    p.add_argument('--lgv_pivot',     type=float, default=DEFAULT_LGV_PIVOT)
    p.add_argument('--f_colour',      type=float, default=DEFAULT_F_COLOUR)
    p.add_argument('--seed_fcolour',  type=int,   default=DEFAULT_SEED_FCOLOUR)
    p.add_argument('--n_neigh_vpeak', type=int,   default=N_NEIGH_VPEAK)
    p.add_argument('--min_vpeak',     type=float, default=MIN_VPEAK)
    p.add_argument('--plot',          action='store_true')
    return p.parse_args()


def observer_position(log):
    """obs_seed=0: observer at box centre (matches make_checkpoint.py)."""
    xobs = yobs = zobs = LBOX / 2.0
    log.info(f'Observer at box centre: ({xobs}, {yobs}, {zobs})')
    return xobs, yobs, zobs


def add_sky_coords(subhalos, log):
    """
    Compute RA, DEC, dist, Zu (cosmological redshift), Zobs.
    obs_seed=0: no rotation applied (identity, same as make_checkpoint.py).
    vx/vy/vz removed after computing vr_c.
    """
    t0 = time.time()
    log.info('obs_seed=0: no rotation applied.')

    dist = np.sqrt(subhalos['x']**2 + subhalos['y']**2 + subhalos['z']**2)
    subhalos['dist'] = dist
    subhalos['DEC']  = np.degrees(np.arcsin(subhalos['z'] / dist))
    subhalos['RA']   = np.degrees(np.arctan2(subhalos['y'], subhalos['x'])) % 360.0

    c_light = 299792.458
    vr_c = (
        subhalos['x'] * subhalos['vx'] +
        subhalos['y'] * subhalos['vy'] +
        subhalos['z'] * subhalos['vz']
    ) / (c_light * dist)

    for col in ('vx', 'vy', 'vz'):
        subhalos.remove_column(col)
    gc.collect()

    # conc (= Rvir/rs) is not computed here as Rvir and rs are not read
    # from the Uchuu files (not in UCHUU_COLUMNS) to save memory.
    gc.collect()

    z_grid  = np.linspace(0.0, 1.0, 10000)
    Dc_grid = cosmo_uchuu.comoving_distance(z_grid).value
    z_of_Dc = interp1d(Dc_grid, z_grid, bounds_error=False,
                       fill_value='extrapolate')
    subhalos['Zu']   = z_of_Dc(dist).astype(np.float32)
    subhalos['Zobs'] = ((1.0 + subhalos['Zu']) * (1.0 + vr_c) - 1.0
                        ).astype(np.float32)
    del z_grid, Dc_grid, z_of_Dc, vr_c, dist

    log.info(f'Sky coords + redshifts done in {time.time()-t0:.1f}s')
    return subhalos


def apply_sky_mask(subhalos, reg, log):
    t0 = time.time()
    log.info(f'Applying DESI DR2 sky mask for region {reg}')
    tiles   = fitsio.read(str(TILES_FILE))
    in_desi = is_point_in_desi(tiles, subhalos['RA'], subhalos['DEC'])
    north   = (
        (subhalos['RA']  > NORTH_RA_MIN) &
        (subhalos['RA']  < NORTH_RA_MAX) &
        (subhalos['DEC'] > NORTH_DEC_MIN)
    )
    mask   = in_desi & north if reg == 'N' else in_desi & ~north
    result = subhalos[mask]
    log.info(f'After mask: {len(result):,} subhalos ({time.time()-t0:.1f}s)')
    return result


def write_checkpoint(subhalos_reg, reg, snapshot, redshift, args, log):
    if 'id' in subhalos_reg.colnames:
        subhalos_reg.rename_column('id', 'ID')

    out_path = make_lightcone_paths(
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        f_colour=args.f_colour,
        alpha=DEFAULT_ALPHA,
        snapshot=snapshot,
        reg=reg,
    )['subhalo_fits']
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        'STAGE':     'make_checkpoint_snap',
        'SNAP':      snapshot,
        'ZSNAP':     redshift,
        'REG':       reg,
        'OBS_SEED':  DEFAULT_OBS_SEED,
        'SIGLGVREF': args.sig_lgv_ref,
        'SIGLGVSLP': args.sig_lgv_slope,
        'LGVPIVOT':  args.lgv_pivot,
        'FCOLOUR':   args.f_colour,
        'SEEDFC':    args.seed_fcolour,
        'MIN_VPEAK': args.min_vpeak,
        'NNVPEAK':   args.n_neigh_vpeak,
        'LBOX':      LBOX,
        'FRAC_BOX':  FRAC_BOX,
    }
    write_fits_meta(subhalos_reg.meta, meta)

    t0 = time.time()
    subhalos_reg.write(str(out_path), overwrite=True)
    log.info(f'Written {len(subhalos_reg):,} subhalos in {time.time()-t0:.1f}s')
    log.info(f'→ {out_path}')


def main():
    args     = parse_args()
    snapshot = args.snapshot
    regs     = ['N', 'S'] if args.reg == 'NS' else [args.reg]

    if snapshot not in mc.redshift_table:
        raise ValueError(
            f'Snapshot {snapshot} not in redshift_table. '
            f'Valid: {sorted(mc.redshift_table.keys())}'
        )
    redshift = mc.redshift_table[snapshot]

    lc_paths      = make_lightcone_paths(
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        f_colour=args.f_colour,
        alpha=DEFAULT_ALPHA,
    )
    lightcone_dir = lc_paths['lightcone_dir']
    lightcone_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logging(
        f'make_checkpoint_snap{snapshot:02d}',
        log_dir=lightcone_dir / 'logs',
    )
    log.info(f'Snapshot {snapshot}  z={redshift}  regions={regs}')
    log.info(f'Lightcone dir: {lightcone_dir}')

    xobs, yobs, zobs = observer_position(log)

    halodir   = UCHUU_HALODIR_TEMPLATE.format(index=snapshot)
    halofiles = sorted(glob.glob(f'{halodir}/halolist_*.h5'))
    if not halofiles:
        raise FileNotFoundError(f'No halo files found in {halodir}')
    log.info(f'Found {len(halofiles)} halo files')

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
    log.info(f'Read {len(subhalos):,} subhalos in {time.time()-t0:.1f}s')

    subhalos = add_sky_coords(subhalos, log)

    subhalos = mc.make_all_columns_contiguous(subhalos)

    subhalos = mc.pre_sort_subhalos(
        subhalos,
        N_neigh=args.n_neigh_vpeak,
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        lgv_pivot=args.lgv_pivot,
        f_colour=args.f_colour,
        seed_fcolour=args.seed_fcolour,
        plot=args.plot,
    )

    subhalos = mc.filter_table_staged(
        subhalos, subhalos['keep'], drop_mask=True, mask_name='keep'
    )
    gc.collect()
    log.info(f'After distance trim: {len(subhalos):,} subhalos')

    for reg in regs:
        log.info(f'=== Region {reg} ===')
        subhalos_reg = apply_sky_mask(subhalos, reg, log)
        write_checkpoint(subhalos_reg, reg, snapshot, redshift, args, log)
        del subhalos_reg
        gc.collect()

    log.info('make_checkpoint_snap.py complete.')


if __name__ == '__main__':
    main()
