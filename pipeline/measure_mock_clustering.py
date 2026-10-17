"""
measure_mock_clustering.py — Apply sample selection cuts to a mock galaxy
catalogue and its corresponding random catalogue, then measure the two-point
correlation function multipoles ξ₀(s) and ξ₂(s) using xirunpc.py from the
DESI LSS pipeline.

Unlike measure_clustering.py (which handles real data), this script uses:
  - Unit weights throughout (no PIP, no angular upweighting)
  - No parent catalogue symlinks needed
  - Same xirunpc.py subprocess approach as measure_clustering.py

Must be run on a compute node (not a login node):
    salloc --nodes=1 --qos=interactive --constraint=cpu \\
           --account=desi --time=01:00:00

Usage examples
--------------
# Default sample, default mock parameters:
python measure_mock_clustering.py --obs_seed 0

# Custom magnitude and redshift cuts:
python measure_mock_clustering.py --obs_seed 0 \\
    --mr_cut -21.0 --zmin 0.1 --zmax 0.4

# Measure NGC and SGC separately:
python measure_mock_clustering.py --obs_seed 0 --regions NGC
python measure_mock_clustering.py --obs_seed 0 --regions SGC
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table
import fitsio

# Pipeline modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DEFAULT_OBS_SEED, DEFAULT_SIG_LGV_REF, DEFAULT_SIG_LGV_SLOPE
from pipeline_utils import make_paths, setup_logging

# LSS repo location
LSS_SCRIPT = Path(os.environ.get("SCRATCH", "")) / "LSS/scripts/xirunpc.py"
LSS_PY     = Path(os.environ.get("SCRATCH", "")) / "LSS/py"


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
        "--sig_lgv_ref", type=float, default=DEFAULT_SIG_LGV_REF,
        help=f"log10(Vpeak) scatter at pivot Vpeak (default: {DEFAULT_SIG_LGV_REF})",
    )
    p.add_argument(
        "--sig_lgv_slope", type=float, default=DEFAULT_SIG_LGV_SLOPE,
        help=f"Gradient of scatter with log10(Vpeak) (default: {DEFAULT_SIG_LGV_SLOPE})",
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
        "--regions", nargs="+", default=["NGC", "SGC"],
        choices=["NGC", "SGC"],
        help="Galactic caps to measure and combine (default: NGC SGC)",
    )
    p.add_argument(
        "--outdir", type=str, default=None,
        help=(
            "Output directory for clustering results. "
            "Defaults to DATA_DIR/clustering/mocks/seed{obs_seed}_sig_lgv{sig_lgv}/"
            "Mr{mr_cut}_z{zmin}-{zmax}/"
        ),
    )
    p.add_argument(
        "--lss_script", type=str, default=None,
        help=f"Path to xirunpc.py (default: {LSS_SCRIPT})",
    )
    return p.parse_args()


# =============================================================================
# Sample selection
# =============================================================================

def apply_cuts(infile, outfile, mr_cut, zmin, zmax, log):
    """
    Apply redshift and absolute magnitude cuts to a catalogue and write
    the trimmed version to outfile in the format xirunpc.py expects.
    """
    log.info(f"Reading {infile}")
    data  = Table.read(str(infile))
    log.info(f"  {len(data):,} rows")
    log.info(f"  Columns with Z: {[c for c in data.colnames if 'Z' in c or 'z' in c]}")

    # Use Zobs (redshift space) for mocks; fall back to Z if not present.
    # Never use Z for mocks even though it exists — that would be real space.
    z_col = 'Zobs' if 'Zobs' in data.colnames else 'Z'
    mask  = (data[z_col] >= zmin) & (data[z_col] <= zmax)
    if 'ABSMAG_RP1' in data.colnames:
        mask &= (data['ABSMAG_RP1'] < mr_cut)

    # xirunpc.py expects a WEIGHT column; add unit weights if not present.
    data_cut = data[mask]
    if 'WEIGHT' not in data_cut.colnames:
        data_cut['WEIGHT'] = np.ones(len(data_cut), dtype=np.float32)

    # xirunpc.py expects the redshift column to be named 'Z'.
    # For mocks, Zobs is the redshift-space redshift; rename it to Z
    # so that xirunpc.py uses redshift-space positions.
    if 'Zobs' in data_cut.colnames:
        if 'Z' in data_cut.colnames:
            data_cut.remove_column('Z')   # remove real-space Z
        data_cut.rename_column('Zobs', 'Z')

    log.info(f"  After cuts: {len(data_cut):,} rows")
    data_cut.write(str(outfile), format='fits', overwrite=True)
    log.info(f"  Written: {outfile}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    log  = setup_logging("measure_mock_clustering")
    log.info(
        f"obs_seed={args.obs_seed}  sig_lgv_ref={args.sig_lgv_ref}  "
        f"sig_lgv_slope={args.sig_lgv_slope}  "
        f"mr_cut={args.mr_cut}  zmin={args.zmin}  zmax={args.zmax}  "
        f"regions={args.regions}"
    )

    lss_script = Path(args.lss_script) if args.lss_script else LSS_SCRIPT
    if not lss_script.exists():
        log.error(
            f"xirunpc.py not found at {lss_script}. "
            "Check that $SCRATCH/LSS is set up correctly."
        )
        sys.exit(1)

    # Resolve mock catalogue paths.
    paths = make_paths(obs_seed=args.obs_seed, sig_lgv_ref=args.sig_lgv_ref,
                       sig_lgv_slope=args.sig_lgv_slope)

    # Output directories.
    sample_tag = f"Mr{args.mr_cut}_z{args.zmin}-{args.zmax}"
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = paths["model_dir"] / "clustering" / sample_tag

    basedir = outdir / "catalogues"
    basedir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Apply sample selection cuts
    # -------------------------------------------------------------------------
    log.info("=== Step 1: Applying sample selection cuts ===")

    for cap in args.regions:
        # Mock galaxy catalogue
        mock_path = paths[f"mock_{cap}"]
        if not mock_path.exists():
            log.error(
                f"Mock catalogue not found: {mock_path}\n"
                "Run make_mock.py and split_ngcsgc.py first."
            )
            sys.exit(1)
        apply_cuts(
            mock_path,
            basedir / f"BGS_BRIGHT_{cap}_clustering.dat.fits",
            args.mr_cut, args.zmin, args.zmax,
            log,
        )

        # Formock random catalogue
        ran_path = paths[f"formock_ran_{cap}"]
        if not ran_path.exists():
            log.error(
                f"Formock random catalogue not found: {ran_path}\n"
                "Run make_mock.py and split_ngcsgc.py first."
            )
            sys.exit(1)
        apply_cuts(
            ran_path,
            basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran.fits",
            args.mr_cut, args.zmin, args.zmax,
            log,
        )

    # -------------------------------------------------------------------------
    # Step 2: Run xirunpc.py with unit weights and no angular upweighting
    # -------------------------------------------------------------------------
    log.info("=== Step 2: Running xirunpc.py ===")

    env = os.environ.copy()
    env["PYTHONPATH"] = f":{LSS_PY}:{env.get('PYTHONPATH', '')}"

    cmd = [
        "python", str(lss_script),
        "--tracer",      "BGS_BRIGHT",
        "--basedir",     str(basedir),
        "--region",      *args.regions,
        "--corr_type",   "smu",
        "--njack",       "0",
        "--weight_type", "default",   # unit weights, no PIP/angular upweighting
        "--bin_type",    "log",
        "--zlim",        str(args.zmin), str(args.zmax),
        "--nran",        "1",
        "--outdir",      str(outdir),
    ]

    log.info("Command: " + " ".join(cmd))
    t0 = time.time()

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        log.error(f"xirunpc.py failed with return code {result.returncode}")
        sys.exit(result.returncode)

    log.info(f"xirunpc.py completed in {time.time()-t0:.1f}s")

    # -------------------------------------------------------------------------
    # Step 3: Report output files
    # -------------------------------------------------------------------------
    log.info("=== Output files ===")
    for f in sorted(outdir.glob("*.npy")):
        log.info(f"  {f}")

    log.info(f"measure_mock_clustering.py complete.  Results in: {outdir}")


if __name__ == "__main__":
    main()
