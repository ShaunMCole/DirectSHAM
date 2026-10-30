"""
make_mock_snap.py
-----------------
SHAM abundance-matching for one Uchuu snapshot (lightcone pipeline).
Lightcone-only companion to make_mock.py. Cut-sky pipeline untouched.

Key differences from make_mock.py
-----------------------------------
* --snapshot selects the checkpoint from the lightcone directory.
* Full redshift range is matched; shell slicing done by make_lightcone.py.
* Zr is NOT written (debug-only column).
* Zobs computed as (1+Zu)*(1+vr_c)-1, consistent with make_checkpoint_snap.py.
* obs_seed always 0.

Usage (login node)
-------------------------------------------
    python make_mock_snap.py --snapshot 47 --reg N \\
        --sig_lgv_ref 0.18 --sig_lgv_slope 0.30
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
import MockCode as mc
import catalogue_analysis as ca
from kcorrections import DESI_KCorrection
from pipeline_utils import (
    make_lightcone_paths,
    make_paths,
    write_fits_meta,
    setup_logging,
)
from config import (
    cosmo,
    LBOX,
    FRAC_BOX,
    DELTA_Z,
    NMULT,
    N_ZBINS,
    N_NEIGH_LUM,
    DEFAULT_SIG_LGV_REF,
    DEFAULT_SIG_LGV_SLOPE,
    DEFAULT_LGV_PIVOT,
    DEFAULT_F_COLOUR,
    DEFAULT_ALPHA,
    DEFAULT_OBS_SEED,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='SHAM matching for one Uchuu snapshot (lightcone).'
    )
    p.add_argument('--snapshot',      type=int,   required=True)
    p.add_argument('--reg',           type=str,   required=True,
                   choices=['N', 'S'])
    p.add_argument('--sig_lgv_ref',   type=float, default=DEFAULT_SIG_LGV_REF)
    p.add_argument('--sig_lgv_slope', type=float, default=DEFAULT_SIG_LGV_SLOPE)
    p.add_argument('--lgv_pivot',     type=float, default=DEFAULT_LGV_PIVOT)
    p.add_argument('--f_colour',      type=float, default=DEFAULT_F_COLOUR)
    p.add_argument('--alpha',         type=float, default=DEFAULT_ALPHA)
    p.add_argument('--n_neigh',       type=int,   default=N_NEIGH_LUM)
    p.add_argument('--nclones',       type=int,   default=NMULT)
    p.add_argument('--deltaz',        type=float, default=DELTA_Z)
    p.add_argument('--nzbins',        type=int,   default=N_ZBINS)
    return p.parse_args()


def main():
    args     = parse_args()
    snapshot = args.snapshot
    reg      = args.reg

    if snapshot not in mc.redshift_table:
        raise ValueError(
            f'Snapshot {snapshot} not in redshift_table. '
            f'Valid: {sorted(mc.redshift_table.keys())}'
        )
    redshift_snap = mc.redshift_table[snapshot]

    # ── Paths ─────────────────────────────────────────────────────────────────
    lc_paths = make_lightcone_paths(
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        f_colour=args.f_colour,
        alpha=args.alpha,
        snapshot=snapshot,
        reg=reg,
    )
    lightcone_dir   = lc_paths['lightcone_dir']
    subhalo_path    = lc_paths['subhalo_fits']
    shell_mock_path = lc_paths['shell_mock']

    cs_paths    = make_paths(
        reg=reg,
        obs_seed=DEFAULT_OBS_SEED,
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        f_colour=args.f_colour,
        alpha=args.alpha,
    )
    random_path = cs_paths[f'fordata_ran_{reg}']

    diag_dir = lightcone_dir / f'diagnostics_snap{snapshot:02d}_{reg}'
    diag_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = diag_dir / 'diagnostics.fits'

    log = setup_logging(
        f'make_mock_snap{snapshot:02d}_{reg}',
        log_dir=lightcone_dir / 'logs',
    )
    log.info(f'Snapshot {snapshot}  z={redshift_snap}  reg={reg}')
    log.info(f'Checkpoint  : {subhalo_path}')
    log.info(f'Randoms     : {random_path}')
    log.info(f'Output      : {shell_mock_path}')

    # ── Read inputs ───────────────────────────────────────────────────────────
    log.info('Reading checkpoint...')
    t0 = time.time()
    subhalos = Table.read(str(subhalo_path))
    log.info(f'  {len(subhalos):,} subhalos in {time.time()-t0:.1f}s')

    log.info('Reading fordata randoms...')
    t0 = time.time()
    ran = Table.read(str(random_path))
    log.info(f'  {len(ran):,} randoms in {time.time()-t0:.1f}s')

    # ── Survey parameters ─────────────────────────────────────────────────────
    Sel      = ca.selection(reg)
    fsky     = Sel['fsky']
    zmin_cat = Sel['zmin']
    zmax_cat = Sel['zmax']
    z_bins   = np.linspace(zmin_cat, zmax_cat, args.nzbins + 1)
    rsphere  = FRAC_BOX * LBOX

    # ── Abundance matching ────────────────────────────────────────────────────
    log.info('Running abundance matching...')
    t0 = time.time()
    out_sub, out_gal, z_insufficient = mc.abundance_match_subhalos_to_randoms(
        subhalos=subhalos,
        ran=ran,
        lbox=LBOX,
        rsphere=rsphere,
        fsky=fsky,
        nmult=args.nclones,
        z_bins=z_bins,
        DeltaZ=args.deltaz,
        reg=reg,
        N_neigh=args.n_neigh,
        diagnostics_path=str(diagnostics_path),
    )
    log.info(
        f'Done in {time.time()-t0:.1f}s  ({len(out_sub):,} matched)'
    )
    if z_insufficient is not None:
        log.warning(f'Subhalo density insufficient below z={z_insufficient:.3f}')

    # ── Build output catalogue ────────────────────────────────────────────────
    log.info('Building output catalogue...')

    sub_id_arr = np.array(subhalos['ID'])
    ran_id_arr = np.array(ran['ID'])
    sub_sorter = np.argsort(sub_id_arr)
    ran_sorter = np.argsort(ran_id_arr)
    sub_rows   = sub_sorter[np.searchsorted(sub_id_arr, out_sub, sorter=sub_sorter)]
    gal_rows   = ran_sorter[np.searchsorted(ran_id_arr, out_gal, sorter=ran_sorter)]

    mock = Table()

    # Positional columns from subhalo checkpoint
    mock['RA']     = subhalos['RA'][sub_rows]
    mock['DEC']    = subhalos['DEC'][sub_rows]
    mock['Z']      = subhalos['Zu'][sub_rows]      # real-space cosmological redshift
    mock['Zobs']   = subhalos['Zobs'][sub_rows]    # (1+Zu)*(1+vr_c)-1, from checkpoint
    mock['Vpeak']  = subhalos['Vpeak'][sub_rows]
    mock['Mvir']   = subhalos['Mvir_all'][sub_rows]
    mock['conc']   = subhalos['conc'][sub_rows]
    mock['sub_ID'] = subhalos['ID'][sub_rows]

    # All galaxy properties from matched random (Zr excluded — debug only)
    skip_ran_cols = {'ID', 'Z', 'RA', 'DEC'}
    for col in ran.colnames:
        if col in skip_ran_cols:
            continue
        mock[col] = ran[col][gal_rows]

    # Recompute apparent r magnitude at subhalo redshift
    kcorr_r  = DESI_KCorrection(band='R', file='jmext', photsys=reg)
    rest_gmr = np.array(ran['REST_GMR_0P1'][gal_rows])
    absmag_r = np.array(ran['ABSMAG_RP1'][gal_rows])
    z_sub    = np.array(mock['Z'])
    kc       = kcorr_r.k(z_sub, rest_gmr)
    dmod     = cosmo.distmod(z_sub).value
    mock['rmag'] = (absmag_r + dmod + kc).astype(np.float32)

    # ── Write output ──────────────────────────────────────────────────────────
    meta = {
        'STAGE':     'make_mock_snap',
        'SNAP':      snapshot,
        'ZSNAP':     redshift_snap,
        'REG':       reg,
        'OBS_SEED':  DEFAULT_OBS_SEED,
        'SIGLGVREF': args.sig_lgv_ref,
        'SIGLGVSLP': args.sig_lgv_slope,
        'LGVPIVOT':  args.lgv_pivot,
        'FCOLOUR':   args.f_colour,
        'ALPHA':     args.alpha,
        'N_NEIGH':   args.n_neigh,
    }
    if z_insufficient is not None:
        meta['ZINSUFFY'] = z_insufficient

    write_fits_meta(mock.meta, meta)
    log.info(f'Writing {len(mock):,} rows to {shell_mock_path}')
    mock.write(str(shell_mock_path), overwrite=True)
    log.info('make_mock_snap.py complete.')


if __name__ == '__main__':
    main()
