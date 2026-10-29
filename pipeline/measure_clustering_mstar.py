"""
measure_clustering_mstar.py — Apply stellar mass threshold sample selection
to BGS data or mock catalogues and measure ξ₀(s) and ξ₂(s).

Supports three sample types:
  mr     : ABSMAG_RP1 < mr_cut           (luminosity threshold, for reference)
  cigale : LOGM > mstar_cut              (CIGALE stellar mass threshold)
  fsf    : LOGMSTAR > mstar_cut          (FSF stellar mass threshold)

For data, uses pip_angular_bitwise weights + jackknife.
For mocks, uses unit weights, no jackknife.

Must be run on a compute node:
    salloc --nodes=1 --qos=interactive --constraint=cpu \\
           --account=desi --mem=128GB --time=02:00:00

Usage examples
--------------
# CIGALE stellar mass threshold, data:
python measure_clustering_mstar.py --mode data \\
    --sample_type cigale --mstar_cut 11.10 --zmin 0.02 --zmax 0.42

# FSF stellar mass threshold, data:
python measure_clustering_mstar.py --mode data \\
    --sample_type fsf --mstar_cut 11.45 --zmin 0.02 --zmax 0.42

# CIGALE threshold, mock:
python measure_clustering_mstar.py --mode mock \\
    --sample_type cigale --mstar_cut 11.10 --zmin 0.02 --zmax 0.42 \\
    --sig_lgv_ref 0.18 --sig_lgv_slope 0.30

# Mr threshold, mock (same as measure_mock_clustering.py):
python measure_clustering_mstar.py --mode mock \\
    --sample_type mr --mr_cut -21.5 --zmin 0.02 --zmax 0.42 \\
    --sig_lgv_ref 0.18 --sig_lgv_slope 0.30
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table

# Pipeline modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DATA_DIR, DATA_DATE, DATA_DATE_DIR,
                    DEFAULT_OBS_SEED, DEFAULT_SIG_LGV_REF,
                    DEFAULT_SIG_LGV_SLOPE, DEFAULT_F_COLOUR, DEFAULT_ALPHA)
from pipeline_utils import make_paths, setup_logging

# LSS repo location
LSS_SCRIPT = Path(os.environ.get("SCRATCH", "")) / "LSS/scripts/xirunpc.py"
LSS_PY     = Path(os.environ.get("SCRATCH", "")) / "LSS/py"

# Parent catalogues for angular upweighting (data mode only)
LSS_CATS   = Path(
    "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2.1"
)
PARENT_DAT = LSS_CATS / "BGS_BRIGHT_full_HPmapcut.dat.fits"
PARENT_RAN = LSS_CATS / "BGS_BRIGHT_0_full_HPmapcut.ran.fits"

# Data and random catalogues
DAT_NGC = DATA_DATE_DIR / f"BGS_BRIGHT_{DATA_DATE}_NGC.fits"
DAT_SGC = DATA_DATE_DIR / f"BGS_BRIGHT_{DATA_DATE}_SGC.fits"
RAN_NGC = DATA_DATE_DIR / "rancat_fordata_NGC.fits"
RAN_SGC = DATA_DATE_DIR / "rancat_fordata_SGC.fits"

# Sample type configuration
SAMPLE_CFG = {
    "mr":     {"col": "ABSMAG_RP1", "sense": "lt",  "label": "Mr"},
    "cigale": {"col": "LOGM",       "sense": "gt",  "label": "Mcig"},
    "fsf":    {"col": "LOGMSTAR",   "sense": "gt",  "label": "Mfsf"},
}


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Mode
    p.add_argument("--mode", choices=["data", "mock"], required=True,
                   help="Run on data or mock catalogues")
    # Sample type
    p.add_argument("--sample_type", choices=["mr", "cigale", "fsf"],
                   default="cigale",
                   help="Selection variable: mr, cigale, or fsf (default: cigale)")
    # Cut values
    p.add_argument("--mstar_cut", type=float, default=None,
                   help="Stellar mass threshold log M* (for cigale or fsf mode)")
    p.add_argument("--mr_cut", type=float, default=-21.5,
                   help="Absolute magnitude cut (for mr mode, default: -21.5)")
    # Redshift limits
    p.add_argument("--zmin", type=float, default=0.02,
                   help="Minimum redshift (default: 0.02)")
    p.add_argument("--zmax", type=float, default=0.42,
                   help="Maximum redshift (default: 0.42)")
    # Jackknife (data only)
    p.add_argument("--njack", type=int, default=60,
                   help="Jackknife regions for data (default: 60, 0=none)")
    # Mock parameters (mock mode only)
    p.add_argument("--obs_seed", type=int, default=DEFAULT_OBS_SEED)
    p.add_argument("--sig_lgv_ref", type=float, default=DEFAULT_SIG_LGV_REF)
    p.add_argument("--sig_lgv_slope", type=float, default=DEFAULT_SIG_LGV_SLOPE)
    p.add_argument("--f_colour", type=float, default=DEFAULT_F_COLOUR)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    # Regions
    p.add_argument("--regions", nargs="+", default=["NGC", "SGC"],
                   choices=["NGC", "SGC"])
    # Misc
    p.add_argument("--outdir", type=str, default=None,
                   help="Override output directory")
    p.add_argument("--lss_script", type=str, default=None)
    return p.parse_args()


# =============================================================================
# Sample selection
# =============================================================================

def apply_cuts(infile, outfile, sample_type, cut_value, zmin, zmax,
               is_mock, log):
    """
    Apply redshift and sample selection cuts; write trimmed catalogue.
    For mocks: rename Zobs→Z for xirunpc.py.
    """
    log.info(f"Reading {Path(infile).name}")
    cat = Table.read(str(infile))
    log.info(f"  {len(cat):,} rows")

    cfg  = SAMPLE_CFG[sample_type]
    col  = cfg["col"]

    # Redshift column
    z_col = "Zobs" if ("Zobs" in cat.colnames and is_mock) else "Z"
    mask  = (cat[z_col] >= zmin) & (cat[z_col] <= zmax)

    # Primary selection cut — only apply to galaxy catalogues, not randoms
    if col in cat.colnames:
        if cfg["sense"] == "lt":
            mask &= cat[col] < cut_value
        else:
            mask &= cat[col] > cut_value
    else:
        # Randoms carry the same stellar mass columns — apply the same cut
        # so that the random catalogue matches the galaxy selection function
        log.info(f"  Column {col!r} not found — no selection cut applied")

    cat_cut = cat[mask]

    # Unit weights for mocks if WEIGHT not present
    if "WEIGHT" not in cat_cut.colnames:
        cat_cut["WEIGHT"] = np.ones(len(cat_cut), dtype=np.float32)

    # Rename Zobs→Z for mocks (xirunpc.py always uses Z)
    if is_mock and "Zobs" in cat_cut.colnames:
        if "Z" in cat_cut.colnames:
            cat_cut.remove_column("Z")
        cat_cut.rename_column("Zobs", "Z")

    log.info(f"  After cuts: {len(cat_cut):,} rows")
    cat_cut.write(str(outfile), format="fits", overwrite=True)
    log.info(f"  Written: {Path(outfile).name}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    log  = setup_logging("measure_clustering_mstar")

    # Validate cut value
    if args.sample_type in ("cigale", "fsf") and args.mstar_cut is None:
        log.error("--mstar_cut is required for --sample_type cigale or fsf")
        sys.exit(1)
    cut_value = args.mstar_cut if args.sample_type in ("cigale", "fsf") \
                else args.mr_cut

    cfg       = SAMPLE_CFG[args.sample_type]
    label     = cfg["label"]
    is_mock   = (args.mode == "mock")

    log.info(f"mode={args.mode}  sample_type={args.sample_type}  "
             f"cut={cut_value}  zmin={args.zmin}  zmax={args.zmax}")

    lss_script = Path(args.lss_script) if args.lss_script else LSS_SCRIPT
    if not lss_script.exists():
        log.error(f"xirunpc.py not found: {lss_script}")
        sys.exit(1)

    # Resolve paths
    paths = make_paths(obs_seed=args.obs_seed, sig_lgv_ref=args.sig_lgv_ref,
                       sig_lgv_slope=args.sig_lgv_slope,
                       f_colour=args.f_colour, alpha=args.alpha)

    # Output directory
    sample_tag = f"{label}{cut_value}_z{args.zmin:.2f}-{args.zmax:.2f}"
    if args.outdir:
        outdir = Path(args.outdir)
    elif is_mock:
        outdir = paths["model_dir"] / "clustering" / sample_tag
    else:
        outdir = DATA_DATE_DIR / "clustering" / "data" / sample_tag

    basedir = outdir / "catalogues"
    basedir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Apply cuts
    # -------------------------------------------------------------------------
    log.info("=== Step 1: Applying sample selection cuts ===")

    if is_mock:
        pairs = []
        for cap in args.regions:
            mock_path = paths[f"mock_{cap}"]
            ran_path  = paths[f"formock_ran_{cap}"]
            for src, name in [
                (mock_path, f"BGS_BRIGHT_{cap}_clustering.dat.fits"),
                (ran_path,  f"BGS_BRIGHT_{cap}_0_clustering.ran.fits"),
            ]:
                if not src.exists():
                    log.error(f"Input not found: {src}")
                    sys.exit(1)
                pairs.append((src, basedir / name))
    else:
        pairs = [
            (DAT_NGC, basedir / "BGS_BRIGHT_NGC_clustering.dat.fits"),
            (DAT_SGC, basedir / "BGS_BRIGHT_SGC_clustering.dat.fits"),
            (RAN_NGC, basedir / "BGS_BRIGHT_NGC_0_clustering.ran.fits"),
            (RAN_SGC, basedir / "BGS_BRIGHT_SGC_0_clustering.ran.fits"),
        ]
        for f, _ in pairs:
            if not Path(f).exists():
                log.error(f"Input not found: {f}")
                sys.exit(1)

    for src, dst in pairs:
        apply_cuts(src, dst, args.sample_type, cut_value,
                   args.zmin, args.zmax, is_mock, log)

    # -------------------------------------------------------------------------
    # Step 2: Symlink parent catalogues (data mode only)
    # -------------------------------------------------------------------------
    if not is_mock:
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

    weight_type = "default" if is_mock else "pip_angular_bitwise"
    njack       = "0"       if is_mock else str(args.njack)

    cmd = [
        "python", str(lss_script),
        "--tracer",      "BGS_BRIGHT",
        "--basedir",     str(basedir),
        "--region",      *args.regions,
        "--corr_type",   "smu",
        "--njack",       njack,
        "--weight_type", weight_type,
        "--bin_type",    "log",
        "--zlim",        f"{args.zmin:.2f}", f"{args.zmax:.2f}",
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
    # Step 4: Report
    # -------------------------------------------------------------------------
    log.info("=== Output files ===")
    for f in sorted(outdir.glob("*.npy")):
        log.info(f"  {f}")
    log.info(f"Done. Results in: {outdir}")


if __name__ == "__main__":
    main()
