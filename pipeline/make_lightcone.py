"""
make_lightcone.py
-----------------
Assemble the per-snapshot shell mock files produced by make_mock_snap.py
into final lightcone catalogues.

For each shell:
  * Read the full-range mock for that snapshot and region.
  * Cut on real-space redshift:  z_min <= Z < z_max.
  * Append to the output catalogue.

After all shells are stacked, split into NGC / SGC by galactic latitude
(b > 0 = NGC) and write:

    lightcone_dir/mock_lightcone_N.fits
    lightcone_dir/mock_lightcone_S.fits
    lightcone_dir/mock_lightcone_NGC.fits
    lightcone_dir/mock_lightcone_SGC.fits

Usage (login node)
------------------
    python make_lightcone.py \
        --sig_lgv_ref 0.18 --sig_lgv_slope 0.30 --f_colour 0.00
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

from pipeline_utils import make_lightcone_paths, write_fits_meta, setup_logging
from config import (
    DEFAULT_SIG_LGV_REF,
    DEFAULT_SIG_LGV_SLOPE,
    DEFAULT_F_COLOUR,
    DEFAULT_ALPHA,
    DEFAULT_OBS_SEED,
    LIGHTCONE_SHELLS,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Assemble lightcone catalogues from per-snapshot shell mocks.'
    )
    p.add_argument('--sig_lgv_ref',   type=float, default=DEFAULT_SIG_LGV_REF)
    p.add_argument('--sig_lgv_slope', type=float, default=DEFAULT_SIG_LGV_SLOPE)
    p.add_argument('--f_colour',      type=float, default=DEFAULT_F_COLOUR)
    p.add_argument('--alpha',         type=float, default=DEFAULT_ALPHA)
    return p.parse_args()


def ngc_sgc_mask(ra, dec):
    """True = NGC (galactic latitude b > 0), False = SGC."""
    coords = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame='icrs')
    return coords.galactic.b.degree > 0


def assemble_region(reg, args, logger):
    """Read and slice all shells for one region; return stacked Table."""
    shells = []
    for snap, z_snap, z_min, z_max in LIGHTCONE_SHELLS:
        shell_path = make_lightcone_paths(
            sig_lgv_ref=args.sig_lgv_ref,
            sig_lgv_slope=args.sig_lgv_slope,
            f_colour=args.f_colour,
            alpha=args.alpha,
            snapshot=snap,
            reg=reg,
        )['shell_mock']

        if not shell_path.exists():
            logger.error(f'Missing shell mock: {shell_path}')
            sys.exit(1)

        logger.info(
            f'  snap{snap:02d} (z_snap={z_snap})  '
            f'shell [{z_min}, {z_max})  {shell_path.name}'
        )
        t0    = time.time()
        shell = Table.read(str(shell_path))
        logger.info(f'    {len(shell):,} rows read in {time.time()-t0:.1f}s')

        # Cut on real-space redshift (Z column, not Zobs)
        mask  = (shell['Z'] >= z_min) & (shell['Z'] < z_max)
        shell = shell[mask]
        logger.info(f'    {len(shell):,} rows after Z in [{z_min}, {z_max})')

        shells.append(shell)

    logger.info(f'Stacking {len(shells)} shells for region {reg}...')
    cat = vstack(shells)
    logger.info(f'Total {reg}: {len(cat):,} rows')
    return cat


def main():
    args = parse_args()

    lc_paths      = make_lightcone_paths(
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        f_colour=args.f_colour,
        alpha=args.alpha,
    )
    lightcone_dir = lc_paths['lightcone_dir']
    lightcone_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging('make_lightcone', log_dir=lightcone_dir / 'logs')
    logger.info(f'Lightcone dir: {lightcone_dir}')
    logger.info(f'Shells: {LIGHTCONE_SHELLS}')

    meta_base = {
        'SIGLGVREF': args.sig_lgv_ref,
        'SIGLGVSLP': args.sig_lgv_slope,
        'FCOLOUR':   args.f_colour,
        'ALPHA':     args.alpha,
        'OBS_SEED':  DEFAULT_OBS_SEED,
    }

    ngc_parts = []
    sgc_parts = []

    for reg in ('N', 'S'):
        logger.info(f'=== Region {reg} ===')
        cat = assemble_region(reg, args, logger)

        # Write N / S file
        out_ns = lightcone_dir / f'mock_lightcone_{reg}.fits'
        meta   = dict(meta_base, REG=reg)
        cat.meta.update(meta)
        write_fits_meta(cat.meta, meta)
        logger.info(f'Writing {out_ns.name}  ({len(cat):,} rows)')
        cat.write(str(out_ns), overwrite=True)

        # NGC / SGC split
        is_ngc = ngc_sgc_mask(np.array(cat['RA']), np.array(cat['DEC']))
        ngc_parts.append(cat[is_ngc])
        sgc_parts.append(cat[~is_ngc])
        logger.info(
            f'  {reg} → NGC: {np.sum(is_ngc):,}  SGC: {np.sum(~is_ngc):,}'
        )

    # Write combined NGC / SGC catalogues
    for label, parts, outpath in [
        ('NGC', ngc_parts, lc_paths['mock_lightcone_NGC']),
        ('SGC', sgc_parts, lc_paths['mock_lightcone_SGC']),
    ]:
        combined = vstack(parts)
        meta     = dict(meta_base, GCAP=label)
        combined.meta.update(meta)
        write_fits_meta(combined.meta, meta)
        logger.info(f'Writing {outpath.name}  ({len(combined):,} rows)')
        combined.write(str(outpath), overwrite=True)

    logger.info('make_lightcone.py complete.')


if __name__ == '__main__':
    main()
