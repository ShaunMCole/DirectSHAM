"""
make_checkpoint.py — Read the full Uchuu halo catalogue, apply scatter to
Vpeak, sort, mask to the survey footprint, and write a compact checkpoint
file containing only the subhalos needed for the SHAM step.

This is the expensive step: it requires an exclusive CPU node at NERSC
(~128 GB RAM) and takes ~30 minutes.  Everything downstream (make_mock.py)
can run on a login node using the checkpoint file.

Usage examples
--------------
# North and South in one pass (recommended — reads Uchuu only once):
python make_checkpoint.py --reg NS

# Both regions, non-default scatter parametrisation:
python make_checkpoint.py --reg NS --sig_lgv_ref 0.16 --sig_lgv_slope 0.30 --obs_seed 1
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
from config import (
    DEFAULT_OBS_SEED,
    DEFAULT_SIG_LGV_REF,
    DEFAULT_SIG_LGV_SLOPE,
    DEFAULT_LGV_PIVOT,
    DEFAULT_F_COLOUR,
    DEFAULT_SEED_FCOLOUR,
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
    cosmo_uchuu,
)
from pipeline_utils import make_paths, write_fits_meta, setup_logging


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reg', required=True, choices=['N', 'S', 'NS'])
    p.add_argument('--obs_seed',      type=int,   default=DEFAULT_OBS_SEED)
    p.add_argument('--sig_lgv_ref',   type=float, default=DEFAULT_SIG_LGV_REF)
    p.add_argument('--sig_lgv_slope', type=float, default=DEFAULT_SIG_LGV_SLOPE)
    p.add_argument('--min_vpeak',     type=float, default=MIN_VPEAK)
    p.add_argument('--n_neigh_vpeak', type=int,   default=N_NEIGH_VPEAK)
    p.add_argument('--snapshot_index',type=int,   default=SNAPSHOT_INDEX)
    p.add_argument('--f_colour',      type=float, default=DEFAULT_F_COLOUR)
    p.add_argument('--seed_fcolour',  type=int,   default=DEFAULT_SEED_FCOLOUR)
    p.add_argument('--alpha',         type=float, default=0.0,
                   help='Correlated scatter parameter alpha_v (default: 0.0)')
    p.add_argument('--plot',          action='store_true')
    return p.parse_args()


def observer_position(obs_seed, lbox, log):
    if obs_seed == 0:
        xobs = yobs = zobs = lbox / 2.0
        log.info(f'Observer at box centre: ({xobs}, {yobs}, {zobs})')
    else:
        rng  = np.random.default_rng(obs_seed + 58)
        xobs = lbox * rng.random()
        yobs = lbox * rng.random()
        zobs = lbox * rng.random()
        log.info(f'Observer position [Mpc/h] (seed={obs_seed}): ({xobs:.1f}, {yobs:.1f}, {zobs:.1f})')
    return xobs, yobs, zobs


def add_sky_coords(subhalos, obs_seed, log):
    import MockCode as mc

    if obs_seed != 0:
        t0 = time.time()
        phi, theta, psi = mc.random_euler_angles(seed=obs_seed)
        log.info(f'Euler angles[radians]: phi={phi:.4f}  theta={theta:.4f}  psi={psi:.4f}')
        R = mc.euler_rotation_matrix(phi, theta, psi)
        vxr = R[0,0]*subhalos['vx'] + R[0,1]*subhalos['vy'] + R[0,2]*subhalos['vz']
        vyr = R[1,0]*subhalos['vx'] + R[1,1]*subhalos['vy'] + R[1,2]*subhalos['vz']
        vzr = R[2,0]*subhalos['vx'] + R[2,1]*subhalos['vy'] + R[2,2]*subhalos['vz']
        subhalos['vx'] = vxr; subhalos['vy'] = vyr; subhalos['vz'] = vzr
        del vxr, vyr, vzr
        xr = R[0,0]*subhalos['x'] + R[0,1]*subhalos['y'] + R[0,2]*subhalos['z']
        yr = R[1,0]*subhalos['x'] + R[1,1]*subhalos['y'] + R[1,2]*subhalos['z']
        zr = R[2,0]*subhalos['x'] + R[2,1]*subhalos['y'] + R[2,2]*subhalos['z']
        subhalos['x'] = xr; subhalos['y'] = yr; subhalos['z'] = zr
        del xr, yr, zr
        log.info(f'Rotation done in {time.time()-t0:.1f}s')
    else:
        log.info('obs_seed=0: no rotation applied.')

    t0 = time.time()
    dist = np.sqrt(subhalos['x']**2 + subhalos['y']**2 + subhalos['z']**2)
    subhalos['dist'] = dist
    subhalos['DEC']  = np.degrees(np.arcsin(subhalos['z'] / dist))
    subhalos['RA']   = np.degrees(np.arctan2(subhalos['y'], subhalos['x'])) % 360.0

    c_light = 299792.458
    vr_c = (
        subhalos['x']*subhalos['vx'] +
        subhalos['y']*subhalos['vy'] +
        subhalos['z']*subhalos['vz']
    ) / (c_light * dist)

    for col in ('vx', 'vy', 'vz'):
        subhalos.remove_column(col)
    gc.collect()

    # conc (= Rvir/rs) is not computed as Rvir and rs are not in UCHUU_COLUMNS.

    z_grid  = np.linspace(0.0, 1.0, 10000)
    Dc_grid = cosmo_uchuu.comoving_distance(z_grid).value
    z_of_Dc = interp1d(Dc_grid, z_grid, bounds_error=False, fill_value='extrapolate')
    subhalos['Zu']   = z_of_Dc(dist).astype(np.float32)
    subhalos['Zobs'] = ((1.0 + subhalos['Zu']) * (1.0 + vr_c) - 1.0).astype(np.float32)
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


def write_checkpoint(subhalos_reg, reg, args, redshift, out_path, log):
    if 'id' in subhalos_reg.colnames:
        subhalos_reg.rename_column('id', 'ID')

    write_fits_meta(subhalos_reg.meta, {
        'STAGE':     'make_checkpoint',
        'REG':       reg,
        'OBS_SEED':  args.obs_seed,
        'SIGLGVREF': args.sig_lgv_ref,
        'SIGLGVSLP': args.sig_lgv_slope,
        'LGVPIVOT':  DEFAULT_LGV_PIVOT,
        'FCOLOUR':   args.f_colour,
        'ALPHA':     args.alpha,
        'SEEDFC':    args.seed_fcolour,
        'MIN_VPEAK': args.min_vpeak,
        'NNVPEAK':   args.n_neigh_vpeak,
        'SNAPIDX':   args.snapshot_index,
        'SNAPZ':     redshift,
        'LBOX':      LBOX,
        'FRAC_BOX':  FRAC_BOX,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subhalos_reg.write(str(out_path), overwrite=True)
    log.info(f'Written {len(subhalos_reg):,} subhalos in {time.time()-t0:.1f}s')
    log.info(f'→ {out_path}')


def main():
    args = parse_args()
    import MockCode as mc

    log = setup_logging('make_checkpoint')
    log.info(
        f'reg={args.reg}  obs_seed={args.obs_seed}  '
        f'sig_lgv_ref={args.sig_lgv_ref}  sig_lgv_slope={args.sig_lgv_slope}  '
        f'f_colour={args.f_colour}  alpha={args.alpha}  '
        f'min_vpeak={args.min_vpeak}  '
        f'n_neigh_vpeak={args.n_neigh_vpeak}  snapshot_index={args.snapshot_index}'
    )

    if args.snapshot_index not in mc.redshift_table:
        log.error(f'snapshot_index={args.snapshot_index} not in redshift_table.')
        sys.exit(1)
    redshift = mc.redshift_table[args.snapshot_index]
    log.info(f'Snapshot {args.snapshot_index} → z={redshift}')

    regs = ['N', 'S'] if args.reg == 'NS' else [args.reg]
    out_paths = {}
    for reg in regs:
        paths = make_paths(
            reg=reg,
            obs_seed=args.obs_seed,
            sig_lgv_ref=args.sig_lgv_ref,
            sig_lgv_slope=args.sig_lgv_slope,
            min_vpeak=args.min_vpeak,
            n_neigh_vpeak=args.n_neigh_vpeak,
            f_colour=args.f_colour,
            alpha=args.alpha,
        )
        out_paths[reg] = paths['subhalo_checkpoint']
        log.info(f'Output checkpoint ({reg}): {out_paths[reg]}')

    halodir   = UCHUU_HALODIR_TEMPLATE.format(index=args.snapshot_index)
    halofiles = sorted(glob.glob(f'{halodir}/halolist_*.h5'))
    if not halofiles:
        log.error(f'No halo files found in {halodir}')
        sys.exit(1)
    log.info(f'Found {len(halofiles)} halo files in {halodir}')

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
    log.info(f'Read {len(subhalos):,} subhalos in {time.time()-t0:.1f}s')

    subhalos = add_sky_coords(subhalos, args.obs_seed, log)
    subhalos = mc.make_all_columns_contiguous(subhalos)

    subhalos = mc.pre_sort_subhalos(
        subhalos,
        N_neigh=args.n_neigh_vpeak,
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        lgv_pivot=DEFAULT_LGV_PIVOT,
        f_colour=args.f_colour,
        seed_fcolour=args.seed_fcolour,
        alpha_v=args.alpha,
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
        write_checkpoint(subhalos_reg, reg, args, redshift, out_paths[reg], log)
        del subhalos_reg
        gc.collect()

    log.info('make_checkpoint.py complete.')


if __name__ == '__main__':
    main()
