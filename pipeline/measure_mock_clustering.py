"""
measure_mock_clustering.py — Apply sample selection cuts to a mock galaxy
catalogue and its corresponding random catalogue, then measure the two-point
correlation function multipoles ξ₀(s) and ξ₂(s) using pycorr directly.

Unlike measure_clustering.py (which handles real data), this script uses:
  - Unit weights throughout (no PIP, no angular upweighting)
  - No parent catalogue symlinks
  - pycorr called directly in Python rather than via xirunpc.py subprocess

Must be run on a compute node (not a login node):
    salloc --nodes=1 --qos=interactive --constraint=cpu \\
           --account=desi --time=01:00:00

Usage examples
--------------
# Default sample, default mock parameters:
python measure_mock_clustering.py --obs_seed 0 --sig_lgv 0.10

# Custom magnitude and redshift cuts:
python measure_mock_clustering.py --obs_seed 0 --sig_lgv 0.10 \\
    --mr_cut -21.0 --zmin 0.1 --zmax 0.4

# Measure both caps separately (default is to combine NGC+SGC):
python measure_mock_clustering.py --obs_seed 0 --sig_lgv 0.10 --cap NGC
python measure_mock_clustering.py --obs_seed 0 --sig_lgv 0.10 --cap SGC
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table
import fitsio

# Pipeline modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, DEFAULT_OBS_SEED, DEFAULT_SIG_LGV
from pipeline_utils import make_paths, setup_logging


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--obs_seed", type=int, default=DEFAULT_OBS_SEED,
        help=f"Observer position seed (default: {DEFAULT_OBS_SEED})",
    )
    p.add_argument(
        "--sig_lgv", type=float, default=DEFAULT_SIG_LGV,
        help=f"log10(Vpeak) scatter in dex (default: {DEFAULT_SIG_LGV})",
    )
    p.add_argument(
        "--mr_cut", type=float, default=-21.5,
        help="Absolute magnitude cut in 0.1Mr (default: -21.5)",
    )
    p.add_argument(
        "--zmin", type=float, default=0.02,
        help="Minimum redshift (default: 0.02)",
    )
    p.add_argument(
        "--zmax", type=float, default=0.42,
        help="Maximum redshift (default: 0.42)",
    )
    p.add_argument(
        "--cap", type=str, default="NGC+SGC",
        choices=["NGC", "SGC", "NGC+SGC"],
        help=(
            "Galactic cap(s) to measure. NGC+SGC combines both "
            "before measuring (default: NGC+SGC)"
        ),
    )
    p.add_argument(
        "--nthreads", type=int, default=64,
        help="Number of threads for pycorr (default: 64)",
    )
    p.add_argument(
        "--outdir", type=str, default=None,
        help=(
            "Output directory. Defaults to "
            "DATA_DIR/clustering/mocks/seed{obs_seed}_sig_lgv{sig_lgv}/"
            "Mr{mr_cut}_z{zmin}-{zmax}/"
        ),
    )
    return p.parse_args()


# =============================================================================
# Sample selection
# =============================================================================

def load_and_cut(path, mr_cut, zmin, zmax, log):
    """
    Read a catalogue, apply redshift and absolute magnitude cuts, and
    return the trimmed astropy Table.
    """
    log.info(f"Reading {path}")
    data  = Table(fitsio.read(str(path)))
    z_col = 'Z' if 'Z' in data.colnames else 'Zobs'
    mask  = (data[z_col] >= zmin) & (data[z_col] <= zmax)
    if 'ABSMAG_RP1' in data.colnames:
        mask &= (data['ABSMAG_RP1'] < mr_cut)
    data_cut = data[mask]
    log.info(f"  {len(data):,} → {len(data_cut):,} rows after cuts")
    return data_cut


def stack_caps(table_ngc, table_sgc, log):
    """Vertically stack NGC and SGC catalogues."""
    from astropy.table import vstack
    combined = vstack([table_ngc, table_sgc], metadata_conflicts='silent')
    log.info(f"  Combined NGC+SGC: {len(combined):,} rows")
    return combined


# =============================================================================
# Pycorr measurement
# =============================================================================

def measure_xi(dat, ran, nthreads, log):
    """
    Measure ξ(s, μ) using pycorr with unit weights and return the
    TwoPointCorrelationFunction result object.

    Parameters
    ----------
    dat, ran : astropy Table
        Must contain columns: Z (or Zobs), RA, DEC.
    nthreads : int
    log : logging.Logger
    """
    from pycorr import TwoPointCorrelationFunction
    from cosmodesi_tools.catalog import CatalogMesh  # for FKP weights if needed

    # Redshift column.
    z_col_dat = 'Z'   if 'Z'   in dat.colnames else 'Zobs'
    z_col_ran = 'Z'   if 'Z'   in ran.colnames else 'Zobs'

    # Positions: (RA, DEC, Z) — pycorr converts to comoving using its own
    # cosmology setting which should match Om0=0.313.
    dat_pos = [dat['RA'], dat['DEC'], dat[z_col_dat]]
    ran_pos = [ran['RA'], ran['DEC'], ran[z_col_ran]]

    # Mock weights: unit for galaxies, 1/NMULT for randoms (already stored
    # as WEIGHT in the formock random catalogue).
    dat_weights = np.ones(len(dat))
    ran_weights = ran['WEIGHT'] if 'WEIGHT' in ran.colnames else np.ones(len(ran))

    # Logarithmic s bins matching those used in measure_clustering.py.
    edges = np.logspace(np.log10(0.1), np.log10(150.0), 49)

    log.info(
        f"Running pycorr: {len(dat):,} galaxies  {len(ran):,} randoms  "
        f"nthreads={nthreads}"
    )
    t0 = time.time()

    result = TwoPointCorrelationFunction(
        mode='smu',
        edges=(edges, np.linspace(-1, 1, 201)),
        data_positions1=dat_pos,
        randoms_positions1=ran_pos,
        data_weights1=dat_weights,
        randoms_weights1=ran_weights,
        position_type='rdd',   # (RA, DEC, distance/redshift)
        dtype=np.float32,
        nthreads=nthreads,
        los='midpoint',
        engine='corrfunc',
    )
    log.info(f"pycorr done in {time.time()-t0:.1f}s")
    return result


# =============================================================================
# Output
# =============================================================================

def save_result(result, outpath, log):
    """Save the pycorr result in its native format (.npy)."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(outpath))
    log.info(f"Saved: {outpath}")

    # Log the multipoles as a quick sanity check.
    xi0 = result(ells=0, return_sep=False)
    xi2 = result(ells=2, return_sep=False)
    s   = result.sep
    log.info("  s      xi0       xi2")
    for i in range(0, len(s), max(1, len(s)//8)):
        log.info(f"  {s[i]:6.2f}  {xi0[i]:+.4f}  {xi2[i]:+.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    log  = setup_logging("measure_mock_clustering")
    log.info(
        f"obs_seed={args.obs_seed}  sig_lgv={args.sig_lgv}  "
        f"mr_cut={args.mr_cut}  zmin={args.zmin}  zmax={args.zmax}  "
        f"cap={args.cap}  nthreads={args.nthreads}"
    )

    # Resolve mock catalogue paths.
    paths = make_paths(obs_seed=args.obs_seed, sig_lgv=args.sig_lgv)

    # Output directory.
    sample_tag = f"Mr{args.mr_cut}_z{args.zmin}-{args.zmax}"
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        mock_tag = f"seed{args.obs_seed:03d}_sig_lgv{args.sig_lgv:.2f}"
        outdir   = DATA_DIR / "clustering" / "mocks" / mock_tag / sample_tag
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Determine which caps to load
    # -------------------------------------------------------------------------
    caps_to_load = ["NGC", "SGC"] if args.cap == "NGC+SGC" else [args.cap]

    # -------------------------------------------------------------------------
    # Load and cut mock galaxy catalogues
    # -------------------------------------------------------------------------
    log.info("=== Loading mock catalogues ===")
    dat_tables = []
    for cap in caps_to_load:
        path = paths[f"mock_{cap}"]
        if not path.exists():
            log.error(
                f"Mock catalogue not found: {path}\n"
                "Run make_mock.py and split_ngcsgc.py first."
            )
            sys.exit(1)
        dat_tables.append(load_and_cut(path, args.mr_cut, args.zmin, args.zmax, log))

    # -------------------------------------------------------------------------
    # Load and cut formock random catalogues
    # -------------------------------------------------------------------------
    log.info("=== Loading formock random catalogues ===")
    ran_tables = []
    for cap in caps_to_load:
        path = paths[f"formock_ran_{cap}"]
        if not path.exists():
            log.error(
                f"Formock random catalogue not found: {path}\n"
                "Run make_mock.py and split_ngcsgc.py first."
            )
            sys.exit(1)
        ran_tables.append(load_and_cut(path, args.mr_cut, args.zmin, args.zmax, log))

    # -------------------------------------------------------------------------
    # Stack caps if measuring NGC+SGC combined
    # -------------------------------------------------------------------------
    if args.cap == "NGC+SGC":
        log.info("=== Stacking NGC + SGC ===")
        dat = stack_caps(dat_tables[0], dat_tables[1], log)
        ran = stack_caps(ran_tables[0], ran_tables[1], log)
    else:
        dat = dat_tables[0]
        ran = ran_tables[0]

    # -------------------------------------------------------------------------
    # Measure clustering
    # -------------------------------------------------------------------------
    log.info("=== Measuring clustering ===")
    result = measure_xi(dat, ran, args.nthreads, log)

    # -------------------------------------------------------------------------
    # Save result
    # -------------------------------------------------------------------------
    outfile = outdir / f"xi_mock_{args.cap}_seed{args.obs_seed:03d}_sig_lgv{args.sig_lgv:.2f}.npy"
    save_result(result, outfile, log)

    log.info(f"measure_mock_clustering.py complete.  Results in: {outdir}")


if __name__ == "__main__":
    main()
