"""
measure_clustering.py — Apply sample selection cuts to the BGS data and
random catalogues and measure the two-point correlation function multipoles
ξ₀(s) and ξ₂(s) using xirunpc.py from the DESI LSS pipeline.

This script is the Python equivalent of ~/Elena/xi_script.sh, restructured
to integrate with the DirectSHAM pipeline.  It calls xirunpc.py as a
subprocess, keeping the native pycorr output format intact.

Must be run on a compute node (not a login node):
    salloc --nodes=1 --qos=interactive --constraint=cpu \\
           --account=desi --time=02:00:00

Usage examples
--------------
# Default sample (Mr < -21.5, 0.02 < z < 0.42):
python measure_clustering.py

# Custom cuts:
python measure_clustering.py --mr_cut -21.0 --zmin 0.1 --zmax 0.4

# With jackknife errors (slower):
python measure_clustering.py --njack 20
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
from config import DATA_DIR, DATA_DATE
from pipeline_utils import setup_logging


# =============================================================================
# Fixed paths — data-side inputs that don't change between runs
# =============================================================================

# LSS repo location (must be on PYTHONPATH before calling xirunpc.py).
LSS_SCRIPT = Path(os.environ.get("SCRATCH", "")) / "LSS/scripts/xirunpc.py"
LSS_PY     = Path(os.environ.get("SCRATCH", "")) / "LSS/py"

# Parent catalogues needed by xirunpc.py for angular upweighting.
LSS_CATS = Path(
    "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2.1"
)
PARENT_DAT = LSS_CATS / "BGS_BRIGHT_full_HPmapcut.dat.fits"
PARENT_RAN = LSS_CATS / "BGS_BRIGHT_0_full_HPmapcut.ran.fits"

# Data and random catalogues split by galactic cap.
_DATDIR  = DATA_DIR
DAT_NGC  = _DATDIR / f"BGS_BRIGHT_{DATA_DATE}_NGC.fits"
DAT_SGC  = _DATDIR / f"BGS_BRIGHT_{DATA_DATE}_SGC.fits"
RAN_NGC  = _DATDIR / f"rancat{DATA_DATE}_fordata_NGC.fits"
RAN_SGC  = _DATDIR / f"rancat{DATA_DATE}_fordata_SGC.fits"


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        "--njack", type=int, default=0,
        help=(
            "Number of jackknife regions for error estimation. "
            "0 = no jackknife (default: 0)"
        ),
    )
    p.add_argument(
        "--outdir", type=str, default=None,
        help=(
            "Output directory for clustering results. "
            "Defaults to DATA_DIR/clustering/data/mr{mr_cut}_z{zmin}-{zmax}/"
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

def apply_cuts(infile, outfile, mr_cut, zmin, zmax, is_dat, log):
    """
    Apply redshift and absolute magnitude cuts to a catalogue and write
    the trimmed version to outfile in the format xirunpc.py expects.
    """
    log.info(f"Reading {infile}")
    data = Table(fitsio.read(str(infile)))
    log.info(f"  {len(data):,} rows")

    # Detect redshift column.
    z_col = 'Z' 
    mask  = (data[z_col] >= zmin) & (data[z_col] <= zmax)

    if 'ABSMAG_RP1' in data.colnames:
        mask &= (data['ABSMAG_RP1'] < mr_cut)

    data_cut = data[mask]
    log.info(f"  After cuts: {len(data_cut):,} rows")
    data_cut.write(str(outfile), format='fits', overwrite=True)
    log.info(f"  Written: {outfile}")


# =============================================================================
# Main
# =============================================================================

def main():
    args   = parse_args()
    log    = setup_logging("measure_clustering")

    log.info(
        f"mr_cut={args.mr_cut}  zmin={args.zmin}  zmax={args.zmax}  "
        f"njack={args.njack}"
    )

    # Resolve paths.
    lss_script = Path(args.lss_script) if args.lss_script else LSS_SCRIPT
    if not lss_script.exists():
        log.error(
            f"xirunpc.py not found at {lss_script}. "
            "Check that $SCRATCH/LSS is set up correctly."
        )
        sys.exit(1)

    for f, label in [
        (DAT_NGC, "DAT_NGC"), (DAT_SGC, "DAT_SGC"),
        (RAN_NGC, "RAN_NGC"), (RAN_SGC, "RAN_SGC"),
    ]:
        if not f.exists():
            log.error(f"{label} not found: {f}")
            sys.exit(1)

    # Output directories.
    sample_tag = f"Mr{args.mr_cut}_z{args.zmin}-{args.zmax}"
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = DATA_DIR / "clustering" / "data" / sample_tag

    basedir = outdir / "catalogues"
    basedir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Apply sample selection cuts
    # -------------------------------------------------------------------------
    log.info("=== Step 1: Applying sample selection cuts ===")

    pairs = [
        (DAT_NGC, basedir / "BGS_BRIGHT_NGC_clustering.dat.fits",     True),
        (DAT_SGC, basedir / "BGS_BRIGHT_SGC_clustering.dat.fits",     True),
        (RAN_NGC, basedir / "BGS_BRIGHT_NGC_0_clustering.ran.fits",   False),
        (RAN_SGC, basedir / "BGS_BRIGHT_SGC_0_clustering.ran.fits",   False),
    ]
    for infile, outfile, is_dat in pairs:
        apply_cuts(infile, outfile, args.mr_cut, args.zmin, args.zmax,
                   is_dat, log)

    # -------------------------------------------------------------------------
    # Step 2: Symlink parent catalogues for angular upweighting
    # -------------------------------------------------------------------------
    log.info("=== Step 2: Symlinking parent catalogues ===")
    for src, name in [
        (PARENT_DAT, "BGS_BRIGHT_full_HPmapcut.dat.fits"),
        (PARENT_RAN, "BGS_BRIGHT_0_full_HPmapcut.ran.fits"),
    ]:
        dst = basedir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        log.info(f"  {dst.name} → {src}")

    # -------------------------------------------------------------------------
    # Step 3: Run xirunpc.py
    # -------------------------------------------------------------------------
    log.info("=== Step 3: Running xirunpc.py ===")

    env = os.environ.copy()
    env["PYTHONPATH"] = f":{LSS_PY}:{env.get('PYTHONPATH', '')}"

    cmd = [
        "python", str(lss_script),
        "--tracer",      "BGS_BRIGHT",
        "--basedir",     str(basedir),
        "--region",      "NGC", "SGC",
        "--corr_type",   "smu",
        "--njack",       str(args.njack),
        "--weight_type", "pip_angular_bitwise",
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
    # Step 4: Report output files
    # -------------------------------------------------------------------------
    log.info("=== Output files ===")
    for f in sorted(outdir.glob("*.npy")):
        log.info(f"  {f}")

    log.info(f"measure_clustering.py complete.  Results in: {outdir}")


if __name__ == "__main__":
    main()
