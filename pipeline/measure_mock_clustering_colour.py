"""
measure_mock_clustering_colour.py — Apply sample selection and red/blue colour
splits to mock and BGS data catalogues, then measure the two-point correlation
function multipoles ξ₀(s) and ξ₂(s) for each colour subset.

The colour split follows:
    Red:  REST_GMR_0P1 >= 0.21 - 0.03 * ABSMAG_RP1
    Blue: REST_GMR_0P1  < 0.21 - 0.03 * ABSMAG_RP1

The same cut is applied to the random catalogues so that they share the same
selection function as each colour subset.

Mock clustering uses unit weights throughout.
Data clustering uses PIP inverse-bitwise weights + angular upweighting,
exactly as measure_clustering.py does for the monochrome case.

Must be run on a compute node (not a login node):
    salloc --nodes=1 --qos=interactive --constraint=cpu \\
           --account=desi --time=02:00:00

Usage examples
--------------
# Default parameters, both colours:
python measure_mock_clustering_colour.py --obs_seed 0

# Custom magnitude and redshift cuts:
python measure_mock_clustering_colour.py --obs_seed 0 \\
    --mr_cut -21.0 --zmin 0.02 --zmax 0.34

# Red only:
python measure_mock_clustering_colour.py --obs_seed 0 --colours red

# Blue only:
python measure_mock_clustering_colour.py --obs_seed 0 --colours blue

# Data only (skip mock):
python measure_mock_clustering_colour.py --data_only
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
    DATA_DIR,
    DATA_DATE,
    DATA_DATE_DIR,
    DEFAULT_OBS_SEED,
    DEFAULT_SIG_LGV_REF,
    DEFAULT_SIG_LGV_SLOPE,
)
from pipeline_utils import make_paths, setup_logging

# LSS repo location
LSS_SCRIPT = Path(os.environ.get("SCRATCH", "")) / "LSS/scripts/xirunpc.py"
LSS_PY     = Path(os.environ.get("SCRATCH", "")) / "LSS/py"

# Colour split parameters
COLOUR_INTERCEPT = 0.21
COLOUR_SLOPE     = 0.03   # sign: red if REST_GMR_0P1 >= intercept - slope * Mr


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
        help=f"Vpeak scatter at pivot Vpeak (default: {DEFAULT_SIG_LGV_REF})",
    )
    p.add_argument(
        "--sig_lgv_slope", type=float, default=DEFAULT_SIG_LGV_SLOPE,
        help=f"Gradient of scatter with log10(Vpeak) (default: {DEFAULT_SIG_LGV_SLOPE})",
    )
    p.add_argument(
        "--f_colour", type=float, default=1.0,
        help=(
            "Colour-assembly bias strength parameter (default: 1.0 = fully "
            "correlated). Reserved for future use when make_mock.py implements "
            "tunable colour-assembly bias; has no effect on the colour cut itself."
        ),
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
        help="Galactic caps to measure (default: NGC SGC)",
    )
    p.add_argument(
        "--data_only", action="store_true",
        help="Measure data clustering only, skip mock.",
    )
    p.add_argument(
        "--mock_only", action="store_true",
        help="Measure mock clustering only, skip data.",
    )
    p.add_argument(
        "--lss_script", type=str, default=None,
        help=f"Path to xirunpc.py (default: {LSS_SCRIPT})",
    )
    p.add_argument(
        "--data_njack", type=int, default=60,
        help="njack value for data clustering jackknife errors (default: 60)",
    )
    return p.parse_args()


# =============================================================================
# Colour split
# =============================================================================

def colour_mask(table, colour):
    """
    Return a boolean mask selecting red or blue galaxies.

    Red:  REST_GMR_0P1 >= COLOUR_INTERCEPT - COLOUR_SLOPE * ABSMAG_RP1
    Blue: REST_GMR_0P1  < COLOUR_INTERCEPT - COLOUR_SLOPE * ABSMAG_RP1
    """
    threshold = COLOUR_INTERCEPT - COLOUR_SLOPE * table['ABSMAG_RP1']
    if colour == 'red':
        return table['REST_GMR_0P1'] >= threshold
    else:
        return table['REST_GMR_0P1'] < threshold


# =============================================================================
# Sample selection and colour split — mocks
# =============================================================================

def apply_cuts_mock(infile, outfile_red, outfile_blue, mr_cut, zmin, zmax, log):
    """
    Apply redshift, magnitude and colour cuts to a mock catalogue.
    Writes red and blue subsets separately.
    Returns (n_red, n_blue).
    """
    log.info(f"Reading mock: {infile}")
    data = Table.read(str(infile))
    log.info(f"  {len(data):,} rows")

    # Use Zobs (redshift space) for mocks.
    z_col = 'Zobs' if 'Zobs' in data.colnames else 'Z'
    base_mask = (data[z_col] >= zmin) & (data[z_col] <= zmax)
    if 'ABSMAG_RP1' in data.colnames:
        base_mask &= (data['ABSMAG_RP1'] < mr_cut)

    data_cut = data[base_mask]

    # Rename Zobs → Z for xirunpc.py
    if 'Zobs' in data_cut.colnames:
        if 'Z' in data_cut.colnames:
            data_cut.remove_column('Z')
        data_cut.rename_column('Zobs', 'Z')

    if 'WEIGHT' not in data_cut.colnames:
        data_cut['WEIGHT'] = np.ones(len(data_cut), dtype=np.float32)

    for colour, outfile in [('red', outfile_red), ('blue', outfile_blue)]:
        mask = colour_mask(data_cut, colour)
        subset = data_cut[mask]
        subset.write(str(outfile), format='fits', overwrite=True)
        log.info(f"  {colour}: {len(subset):,} rows → {outfile.name}")

    return int(base_mask.sum())


# =============================================================================
# Sample selection and colour split — data
# =============================================================================

def apply_cuts_data(infile, outfile_red, outfile_blue, mr_cut, zmin, zmax, log):
    """
    Apply redshift, magnitude and colour cuts to a BGS data catalogue.
    Writes red and blue subsets separately.
    PIP weights and BITWEIGHTS are preserved for xirunpc.py.
    """
    log.info(f"Reading data: {infile}")
    data = Table.read(str(infile))
    log.info(f"  {len(data):,} rows")

    base_mask = (data['Z'] >= zmin) & (data['Z'] <= zmax)
    if 'ABSMAG_RP1' in data.colnames:
        base_mask &= (data['ABSMAG_RP1'] < mr_cut)

    data_cut = data[base_mask]

    for colour, outfile in [('red', outfile_red), ('blue', outfile_blue)]:
        mask = colour_mask(data_cut, colour)
        subset = data_cut[mask]
        subset.write(str(outfile), format='fits', overwrite=True)
        log.info(f"  {colour}: {len(subset):,} rows → {outfile.name}")


# =============================================================================
# Sample selection and colour split — randoms
# =============================================================================

def apply_cuts_randoms(infile, outfile_red, outfile_blue, mr_cut, zmin, zmax,
                       z_col, log):
    """
    Apply redshift, magnitude and colour cuts to a random catalogue.
    Randoms must be colour-split to match the selection function of each
    colour subset.
    z_col: 'Z' for fordata randoms, 'Z' for formock randoms.
    """
    log.info(f"Reading randoms: {infile}")
    ran = Table.read(str(infile))
    log.info(f"  {len(ran):,} rows")

    base_mask = (ran[z_col] >= zmin) & (ran[z_col] <= zmax)
    if 'ABSMAG_RP1' in ran.colnames:
        base_mask &= (ran['ABSMAG_RP1'] < mr_cut)

    ran_cut = ran[base_mask]

    for colour, outfile in [('red', outfile_red), ('blue', outfile_blue)]:
        mask = colour_mask(ran_cut, colour)
        subset = ran_cut[mask]
        if 'WEIGHT' not in subset.colnames:
            subset['WEIGHT'] = np.ones(len(subset), dtype=np.float32)
        subset.write(str(outfile), format='fits', overwrite=True)
        log.info(f"  randoms {colour}: {len(subset):,} rows → {outfile.name}")


# =============================================================================
# Run xirunpc.py
# =============================================================================

def run_xirunpc(lss_script, basedir, regions, zmin, zmax, outdir,
                weight_type, njack, log):
    """Run xirunpc.py for a prepared set of catalogues."""
    env = os.environ.copy()
    env["PYTHONPATH"] = f":{LSS_PY}:{env.get('PYTHONPATH', '')}"

    cmd = [
        "python", str(lss_script),
        "--tracer",      "BGS_BRIGHT",
        "--basedir",     str(basedir),
        "--region",      *regions,
        "--corr_type",   "smu",
        "--njack",       str(njack),
        "--weight_type", weight_type,
        "--bin_type",    "log",
        "--zlim",        str(zmin), str(zmax),
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


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    log  = setup_logging("measure_mock_clustering_colour")
    log.info(
        f"obs_seed={args.obs_seed}  sig_lgv_ref={args.sig_lgv_ref}  "
        f"sig_lgv_slope={args.sig_lgv_slope}  f_colour={args.f_colour}  "
        f"mr_cut={args.mr_cut}  zmin={args.zmin}  zmax={args.zmax}  "
        f"colours={args.colours}  regions={args.regions}"
    )

    lss_script = Path(args.lss_script) if args.lss_script else LSS_SCRIPT
    if not lss_script.exists():
        log.error(f"xirunpc.py not found at {lss_script}.")
        sys.exit(1)

    paths    = make_paths(
        obs_seed=args.obs_seed,
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
    )
    sample_tag = f"Mr{args.mr_cut}_z{args.zmin}-{args.zmax}"
    mock_tag   = (
        f"seed{args.obs_seed:03d}_sig_lgv{args.sig_lgv_ref:.2f}"
        f"_{args.sig_lgv_slope:.2f}"
    )

    # -------------------------------------------------------------------------
    # Mock clustering
    # -------------------------------------------------------------------------
    if not args.data_only:
        log.info("=== Mock colour clustering ===")
        for colour in args.colours:
            log.info(f"--- Mock {colour} ---")
            basedir = (paths["model_dir"] / "clustering"
                       / f"{sample_tag}_{colour}" / "catalogues")
            outdir  = basedir.parent
            basedir.mkdir(parents=True, exist_ok=True)

            for cap in args.regions:
                mock_path = paths[f"mock_{cap}"]
                if not mock_path.exists():
                    log.error(f"Mock not found: {mock_path}")
                    sys.exit(1)
                apply_cuts_mock(
                    mock_path,
                    basedir / f"BGS_BRIGHT_{cap}_clustering.dat_red.fits",
                    basedir / f"BGS_BRIGHT_{cap}_clustering.dat_blue.fits",
                    args.mr_cut, args.zmin, args.zmax, log,
                )
                # Write the colour subset as the dat file xirunpc expects
                colour_dat = basedir / f"BGS_BRIGHT_{cap}_clustering.dat_red.fits" \
                    if colour == 'red' \
                    else basedir / f"BGS_BRIGHT_{cap}_clustering.dat_blue.fits"
                dest = basedir / f"BGS_BRIGHT_{cap}_clustering.dat.fits"
                if dest.exists():
                    dest.unlink()
                dest.symlink_to(colour_dat.name)

                # Formock randoms — colour split
                ran_path = paths[f"formock_ran_{cap}"]
                if not ran_path.exists():
                    log.error(f"Formock randoms not found: {ran_path}")
                    sys.exit(1)
                apply_cuts_randoms(
                    ran_path,
                    basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran_red.fits",
                    basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran_blue.fits",
                    args.mr_cut, args.zmin, args.zmax, 'Z', log,
                )
                colour_ran = basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran_red.fits" \
                    if colour == 'red' \
                    else basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran_blue.fits"
                dest_ran = basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran.fits"
                if dest_ran.exists():
                    dest_ran.unlink()
                dest_ran.symlink_to(colour_ran.name)

            run_xirunpc(
                lss_script, basedir, args.regions,
                args.zmin, args.zmax, outdir,
                weight_type="default", njack=0, log=log,
            )

    # -------------------------------------------------------------------------
    # Data clustering
    # -------------------------------------------------------------------------
    if not args.mock_only:
        log.info("=== Data colour clustering ===")

        # Data files live in DATA_DATE_DIR
        data_files = {
            cap: DATA_DATE_DIR / f"BGS_BRIGHT_{DATA_DATE}_{cap}.fits"
            for cap in args.regions
        }
        ran_files = {
            cap: DATA_DATE_DIR / f"rancat_fordata_{cap}.fits"
            for cap in args.regions
        }

        for colour in args.colours:
            log.info(f"--- Data {colour} ---")
            basedir = (DATA_DATE_DIR / "clustering" / "data"
                       / f"{sample_tag}_{colour}" / "catalogues")
            outdir  = basedir.parent
            basedir.mkdir(parents=True, exist_ok=True)

            for cap in args.regions:
                if not data_files[cap].exists():
                    log.error(f"Data file not found: {data_files[cap]}")
                    sys.exit(1)
                apply_cuts_data(
                    data_files[cap],
                    basedir / f"BGS_BRIGHT_{cap}_clustering.dat_red.fits",
                    basedir / f"BGS_BRIGHT_{cap}_clustering.dat_blue.fits",
                    args.mr_cut, args.zmin, args.zmax, log,
                )
                colour_dat = basedir / f"BGS_BRIGHT_{cap}_clustering.dat_red.fits" \
                    if colour == 'red' \
                    else basedir / f"BGS_BRIGHT_{cap}_clustering.dat_blue.fits"
                dest = basedir / f"BGS_BRIGHT_{cap}_clustering.dat.fits"
                if dest.exists():
                    dest.unlink()
                dest.symlink_to(colour_dat.name)

                if not ran_files[cap].exists():
                    log.error(f"Fordata randoms not found: {ran_files[cap]}")
                    sys.exit(1)
                apply_cuts_randoms(
                    ran_files[cap],
                    basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran_red.fits",
                    basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran_blue.fits",
                    args.mr_cut, args.zmin, args.zmax, 'Z', log,
                )
                colour_ran = basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran_red.fits" \
                    if colour == 'red' \
                    else basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran_blue.fits"
                dest_ran = basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran.fits"
                if dest_ran.exists():
                    dest_ran.unlink()
                dest_ran.symlink_to(colour_ran.name)

            # Symlink parent catalogues needed for angular upweighting
            LSS_CATS = Path(
                "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS"
                "/loa-v1/LSScats/v2.1"
            )
            for src, name in [
                (LSS_CATS / "BGS_BRIGHT_full_HPmapcut.dat.fits",
                 "BGS_BRIGHT_full_HPmapcut.dat.fits"),
                (LSS_CATS / "BGS_BRIGHT_0_full_HPmapcut.ran.fits",
                 "BGS_BRIGHT_0_full_HPmapcut.ran.fits"),
            ]:
                dst = basedir / name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src)
                log.info(f"  {dst.name} → {src}")

            run_xirunpc(
                lss_script, basedir, args.regions,
                args.zmin, args.zmax, outdir,
                weight_type="pip_angular_bitwise",
                njack=args.data_njack, log=log,
            )

    log.info("measure_mock_clustering_colour.py complete.")


if __name__ == "__main__":
    main()
