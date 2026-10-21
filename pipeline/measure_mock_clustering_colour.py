"""
measure_mock_clustering_colour.py — Apply colour-split sample selection cuts
to a mock galaxy catalogue and measure the two-point correlation function
multipoles ξ₀(s) and ξ₂(s) separately for red and blue galaxies.

The colour split follows:
    Red:  REST_GMR_0P1 >= 0.21 - 0.03 * ABSMAG_RP1
    Blue: REST_GMR_0P1  < 0.21 - 0.03 * ABSMAG_RP1

Uses unit weights throughout (no PIP, no angular upweighting).
Results are written into separate per-colour subdirectories:
    clustering/Mr{mr_cut}_z{zmin}-{zmax}_red/
    clustering/Mr{mr_cut}_z{zmin}-{zmax}_blue/

Must be run on a compute node (not a login node):
    salloc --nodes=1 --qos=interactive --constraint=cpu \\
           --account=desi --time=01:00:00

Usage examples
--------------
# Default sample, default mock parameters:
python measure_mock_clustering_colour.py --sig_lgv_ref 0.18 --sig_lgv_slope 0.30 \\
    --f_colour 0.03 --alpha 0.0 --mr_cut -21.0 --zmin 0.02 --zmax 0.34

# Measure a single colour only:
python measure_mock_clustering_colour.py --sig_lgv_ref 0.18 --sig_lgv_slope 0.30 \\
    --f_colour 0.03 --alpha 0.0 --mr_cut -21.0 --zmin 0.02 --zmax 0.34 \\
    --colours red
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
from config import (
    DEFAULT_OBS_SEED, DEFAULT_SIG_LGV_REF, DEFAULT_SIG_LGV_SLOPE,
    DEFAULT_F_COLOUR, DEFAULT_ALPHA,
)
from pipeline_utils import make_paths, setup_logging

# LSS repo location
LSS_SCRIPT = Path(os.environ.get("SCRATCH", "")) / "LSS/scripts/xirunpc.py"
LSS_PY     = Path(os.environ.get("SCRATCH", "")) / "LSS/py"


# =============================================================================
# Colour cut
# =============================================================================

def colour_mask(data, colour):
    """
    Return a boolean mask selecting red or blue galaxies.

    Red:  REST_GMR_0P1 >= 0.21 - 0.03 * ABSMAG_RP1
    Blue: REST_GMR_0P1  < 0.21 - 0.03 * ABSMAG_RP1
    """
    threshold = 0.21 - 0.03 * data['ABSMAG_RP1']
    if colour == 'red':
        return data['REST_GMR_0P1'] >= threshold
    else:
        return data['REST_GMR_0P1'] < threshold


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
        "--f_colour", type=float, default=DEFAULT_F_COLOUR,
        help=f"f_colour used when making the checkpoint (default: {DEFAULT_F_COLOUR})",
    )
    p.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"alpha used when making the mock (default: {DEFAULT_ALPHA})",
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
        "--colours", nargs="+", default=["red", "blue"],
        choices=["red", "blue"],
        help="Which colour subsets to measure (default: red blue)",
    )
    p.add_argument(
        "--regions", nargs="+", default=["NGC", "SGC"],
        choices=["NGC", "SGC"],
        help="Galactic caps to measure and combine (default: NGC SGC)",
    )
    p.add_argument(
        "--data_only", action="store_true",
        help="Measure data clustering only (skip mock).",
    )
    p.add_argument(
        "--lss_script", type=str, default=None,
        help=f"Path to xirunpc.py (default: {LSS_SCRIPT})",
    )
    return p.parse_args()


# =============================================================================
# Sample selection + colour cut
# =============================================================================

def apply_cuts(infile, outfile, mr_cut, zmin, zmax, colour, log, is_random=False):
    """
    Apply redshift, magnitude and (for data files) colour cuts, then write
    the trimmed catalogue in the format xirunpc.py expects.
    """
    log.info(f"Reading {infile}")
    data = Table.read(str(infile))
    log.info(f"  {len(data):,} rows")

    # Redshift cut — use Zobs for mocks, Z for randoms
    z_col = 'Zobs' if 'Zobs' in data.colnames else 'Z'
    mask  = (data[z_col] >= zmin) & (data[z_col] <= zmax)

    # Magnitude cut
    if 'ABSMAG_RP1' in data.colnames:
        mask &= (data['ABSMAG_RP1'] < mr_cut)

    # Colour cut — applied to mock galaxies only, not randoms
    if not is_random and 'REST_GMR_0P1' in data.colnames:
        mask &= colour_mask(data, colour)

    data_cut = data[mask]

    # Ensure WEIGHT column exists
    if 'WEIGHT' not in data_cut.colnames:
        data_cut['WEIGHT'] = np.ones(len(data_cut), dtype=np.float32)

    # xirunpc.py expects redshift column named 'Z'
    if 'Zobs' in data_cut.colnames:
        if 'Z' in data_cut.colnames:
            data_cut.remove_column('Z')
        data_cut.rename_column('Zobs', 'Z')

    log.info(f"  After cuts ({colour}): {len(data_cut):,} rows")
    data_cut.write(str(outfile), format='fits', overwrite=True)
    log.info(f"  Written: {outfile}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    log  = setup_logging("measure_mock_clustering_colour")
    log.info(
        f"obs_seed={args.obs_seed}  sig_lgv_ref={args.sig_lgv_ref}  "
        f"sig_lgv_slope={args.sig_lgv_slope}  f_colour={args.f_colour}  "
        f"alpha={args.alpha}  "
        f"mr_cut={args.mr_cut}  zmin={args.zmin}  zmax={args.zmax}  "
        f"colours={args.colours}  regions={args.regions}"
    )

    lss_script = Path(args.lss_script) if args.lss_script else LSS_SCRIPT
    if not lss_script.exists():
        log.error(f"xirunpc.py not found at {lss_script}.")
        sys.exit(1)

    paths = make_paths(
        obs_seed=args.obs_seed,
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        f_colour=args.f_colour,
        alpha=args.alpha,
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f":{LSS_PY}:{env.get('PYTHONPATH', '')}"

    # -------------------------------------------------------------------------
    # Mock clustering — one pass per colour
    # -------------------------------------------------------------------------
    if not args.data_only:
        log.info("=== Mock colour clustering ===")

        for colour in args.colours:
            log.info(f"--- Colour: {colour} ---")

            # Each colour gets its own output directory
            sample_tag = (
                f"Mr{args.mr_cut}_z{args.zmin:.2f}-{args.zmax:.2f}_{colour}"
            )
            outdir  = paths["model_dir"] / "clustering" / sample_tag
            basedir = outdir / "catalogues"
            basedir.mkdir(parents=True, exist_ok=True)
            outdir.mkdir(parents=True, exist_ok=True)

            for cap in args.regions:
                # Mock galaxy catalogue — apply colour cut
                mock_path = paths[f"mock_{cap}"]
                if not mock_path.exists():
                    log.error(f"Mock catalogue not found: {mock_path}")
                    sys.exit(1)
                apply_cuts(
                    mock_path,
                    basedir / f"BGS_BRIGHT_{cap}_clustering.dat.fits",
                    args.mr_cut, args.zmin, args.zmax, colour, log,
                    is_random=False,
                )

                # Formock random catalogue — no colour cut on randoms
                ran_path = paths[f"formock_ran_{cap}"]
                if not ran_path.exists():
                    log.error(f"Formock random catalogue not found: {ran_path}")
                    sys.exit(1)
                apply_cuts(
                    ran_path,
                    basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran.fits",
                    args.mr_cut, args.zmin, args.zmax, colour, log,
                    is_random=True,
                )

            cmd = [
                "python", str(lss_script),
                "--tracer",      "BGS_BRIGHT",
                "--basedir",     str(basedir),
                "--region",      *args.regions,
                "--corr_type",   "smu",
                "--njack",       "0",
                "--weight_type", "default",
                "--bin_type",    "log",
                "--zlim",        str(args.zmin), str(args.zmax),
                "--nran",        "1",
                "--outdir",      str(outdir),
            ]
            log.info("Command: " + " ".join(cmd))
            t0 = time.time()
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                log.error(f"xirunpc.py failed (return code {result.returncode})")
                sys.exit(result.returncode)
            log.info(f"xirunpc.py ({colour}) completed in {time.time()-t0:.1f}s")

            log.info(f"Output files ({colour}):")
            for f in sorted(outdir.glob("*.npy")):
                log.info(f"  {f}")

    log.info("measure_mock_clustering_colour.py complete.")


if __name__ == "__main__":
    main()
