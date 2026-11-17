"""
measure_wp_cigale_bins.py — Measure the projected correlation function w(r_p)
for the KP CIGALE stellar mass bin samples defined in Table I of the KP paper.

Loops over all redshift × stellar mass bins in the table, applying the
per-bin π_max.  Skips the z=0.010–0.027 row (below our z=0.02 completeness
limit).

Bin definitions (Table I, CIGALE):
  z=0.027–0.043  log M*: 8.5–9.0, 9.0–9.5, 9.5–10.0, 10.0–10.5, 10.5–11.0
                 π_max = 20 Mpc/h
  z=0.043–0.068  log M*: 9.0–9.5, 9.5–10.0, 10.0–10.5, 10.5–11.0
                 π_max = 40 Mpc/h
  z=0.068–0.107  log M*: 9.5–10.0, 10.0–10.5, 10.5–11.0, 11.0–12.0
                 π_max = 40 Mpc/h
  z=0.107–0.162  log M*: 10.0–10.5, 10.5–11.0, 11.0–12.0
                 π_max = 40 Mpc/h
  z=0.162–0.240  log M*: 10.5–11.0, 11.0–12.0
                 π_max = 40 Mpc/h
  z=0.240–0.344  log M*: 11.0–12.0
                 π_max = 40 Mpc/h

Bin edges for xirunpc.py
------------------------
For bins with π_max = 40 Mpc/h, we pass --bin_type log and xirunpc.py uses
its built-in rppi log edges: rp in [0.01, 100) Mpc/h (48 bins) and
π in [-40, 40] Mpc/h (80 bins, 1 Mpc/h steps).

For the π_max = 20 Mpc/h bin we cannot use --bin_type log (which hardcodes
π_max = 40).  Instead we write a temporary .npz file with custom edges
(same rp as above but π in [-20, 20] Mpc/h) and pass its path as --bin_type.

r_p binning: 25 log-spaced bins from 0.1 to 50 Mpc/h (placeholder — update
when KP confirms the official binning).  For the standard (pimax=40) bins this
overrides the default rp edges via the custom .npz route too so that all bins
use the same rp grid regardless of pimax.

Output layout
-------------
Catalogues and results share the same directory tree as the multipole script
so it is clear the samples are identical:

  clustering/<sample_tag>/catalogues/   ← shared catalogue files
  clustering/<sample_tag>/rppi/         ← w(rp) pair counts (.npy)
  clustering/<sample_tag>/smu/          ← multipole pair counts (.npy, from
                                           measure_clustering_mstar.py)

xirunpc.py automatically writes rppi output into an rppi/ subdirectory of
--outdir, so there is no filename clash with the smu results.

If the catalogue files already exist from a previous multipole run they are
reused; the script checks for their presence and skips the cut step if they
are all present.

For data, uses pip_angular_bitwise weights + jackknife.
For mocks, uses unit weights, no jackknife.

Must be run on a compute node:
    salloc --nodes=1 --qos=interactive --constraint=cpu \\
           --account=desi --mem=512GB --time=04:00:00

Usage examples
--------------
# All bins, data:
python measure_wp_cigale_bins.py --mode data

# All bins, mock:
python measure_wp_cigale_bins.py --mode mock \\
    --sig_lgv_ref 0.18 --sig_lgv_slope 0.30

# Single bin (for testing):
python measure_wp_cigale_bins.py --mode data --zmin 0.043 --zmax 0.068 \\
    --mstar_lo 9.0 --mstar_hi 9.5

# Force recreation of catalogue files even if they exist:
python measure_wp_cigale_bins.py --mode data --force_cuts
"""

import argparse
import os
import subprocess
import sys
import tempfile
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

# =============================================================================
# KP Table I — CIGALE stellar mass bins
# Each entry: (zmin, zmax, mstar_lo, mstar_hi, pimax)
# z=0.010–0.027 row omitted (below our z=0.02 completeness limit).
# Dashes in Table I for pimax are taken as 40 Mpc/h.
# =============================================================================
KP_BINS = [
    # z=0.027–0.043, pimax=20
    (0.027, 0.043,  8.5,  9.0, 20.0),
    (0.027, 0.043,  9.0,  9.5, 20.0),
    (0.027, 0.043,  9.5, 10.0, 20.0),
    (0.027, 0.043, 10.0, 10.5, 20.0),
    (0.027, 0.043, 10.5, 11.0, 20.0),
    # z=0.043–0.068, pimax=40
    (0.043, 0.068,  9.0,  9.5, 40.0),
    (0.043, 0.068,  9.5, 10.0, 40.0),
    (0.043, 0.068, 10.0, 10.5, 40.0),
    (0.043, 0.068, 10.5, 11.0, 40.0),
    # z=0.068–0.107, pimax=40
    (0.068, 0.107,  9.5, 10.0, 40.0),
    (0.068, 0.107, 10.0, 10.5, 40.0),
    (0.068, 0.107, 10.5, 11.0, 40.0),
    (0.068, 0.107, 11.0, 12.0, 40.0),
    # z=0.107–0.162, pimax=40
    (0.107, 0.162, 10.0, 10.5, 40.0),
    (0.107, 0.162, 10.5, 11.0, 40.0),
    (0.107, 0.162, 11.0, 12.0, 40.0),
    # z=0.162–0.240, pimax=40
    (0.162, 0.240, 10.5, 11.0, 40.0),
    (0.162, 0.240, 11.0, 12.0, 40.0),
    # z=0.240–0.344, pimax=40
    (0.240, 0.344, 11.0, 12.0, 40.0),
]

# r_p binning: 25 log-spaced bins from 0.1 to 50 Mpc/h
# TODO: replace with official KP binning when confirmed by Andres
RP_EDGES = np.logspace(np.log10(0.1), np.log10(50.0), 26)  # 26 edges → 25 bins


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
    # Optional single-bin override (for testing)
    p.add_argument("--zmin", type=float, default=None,
                   help="Run only the bin with this zmin (must also set --zmax, "
                        "--mstar_lo, --mstar_hi)")
    p.add_argument("--zmax", type=float, default=None,
                   help="Run only the bin with this zmax")
    p.add_argument("--mstar_lo", type=float, default=None,
                   help="Run only the bin with this lower log M* edge")
    p.add_argument("--mstar_hi", type=float, default=None,
                   help="Run only the bin with this upper log M* edge")
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
    # Catalogue reuse
    p.add_argument("--force_cuts", action="store_true",
                   help="Recreate catalogue files even if they already exist "
                        "(default: reuse existing files from a prior multipole run)")
    # Misc
    p.add_argument("--outdir", type=str, default=None,
                   help="Override base output directory (bin subdirs still created "
                        "beneath it)")
    p.add_argument("--lss_script", type=str, default=None,
                   help=f"Path to xirunpc.py (default: {LSS_SCRIPT})")
    return p.parse_args()


# =============================================================================
# Sample selection
# =============================================================================

def apply_cuts(infile, outfile, zmin, zmax, mstar_lo, mstar_hi, is_mock, log):
    """
    Apply redshift and CIGALE stellar mass bin cuts (using LOGM column);
    write trimmed catalogue. For mocks: rename Zobs→Z for xirunpc.py.
    """
    log.info(f"  Reading {Path(infile).name}")
    cat = Table.read(str(infile))
    log.info(f"    {len(cat):,} rows")

    z_col = "Zobs" if ("Zobs" in cat.colnames and is_mock) else "Z"
    mask  = (cat[z_col] >= zmin) & (cat[z_col] <= zmax)

    if "LOGM" in cat.colnames:
        mask &= (cat["LOGM"] > mstar_lo) & (cat["LOGM"] <= mstar_hi)
    else:
        log.warning("    Column 'LOGM' not found — no stellar mass cut applied")

    cat_cut = cat[mask]

    # Unit weights for mocks if WEIGHT not present
    if "WEIGHT" not in cat_cut.colnames:
        cat_cut["WEIGHT"] = np.ones(len(cat_cut), dtype=np.float32)

    # Rename Zobs→Z for mocks (xirunpc.py always uses Z)
    if is_mock and "Zobs" in cat_cut.colnames:
        if "Z" in cat_cut.colnames:
            cat_cut.remove_column("Z")
        cat_cut.rename_column("Zobs", "Z")

    log.info(f"    After cuts: {len(cat_cut):,} rows")
    cat_cut.write(str(outfile), format="fits", overwrite=True)
    log.info(f"    Written: {Path(outfile).name}")


def catalogues_exist(basedir, regions):
    """Return True if all expected catalogue files are present in basedir."""
    expected = []
    for cap in regions:
        expected.append(basedir / f"BGS_BRIGHT_{cap}_clustering.dat.fits")
        expected.append(basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran.fits")
    return all(f.exists() for f in expected)


def output_exists(outdir, zmin, zmax, is_mock, njack):
    """
    Return True if the GCcomb rppi output .npy already exists for this bin.
    This is the combined NGC+SGC file written by xirunpc.py at the end of its
    run — if it is present the pair counting completed successfully.
    """
    weight_type = "default" if is_mock else "pip_angular_bitwise"
    njack_str   = "0" if is_mock else str(njack)
    fname = (f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.3f}_{zmax:.3f}"
             f"_{weight_type}_custom_njack{njack_str}_nran1_split20.npy")
    return (outdir / "rppi" / fname).exists()


# =============================================================================
# Edge file helpers
# =============================================================================

def make_edges_file(pimax, tmpdir):
    """
    Write a temporary .npz file with custom rp and pi edges for xirunpc.py.
    Uses the global RP_EDGES and 1 Mpc/h pi steps from -pimax to +pimax.
    Returns the path to the .npz file.
    """
    pi_edges = np.linspace(-pimax, pimax, int(2 * pimax) + 1)
    path = Path(tmpdir) / f"edges_pimax{pimax:.0f}.npz"
    np.savez(str(path), rp=RP_EDGES, pi=pi_edges)
    return path


# =============================================================================
# Single-bin measurement
# =============================================================================

def run_bin(zmin, zmax, mstar_lo, mstar_hi, pimax, args, paths,
            lss_script, is_mock, log, tmpdir):
    """Prepare catalogues and run xirunpc.py for one (z, M*) bin."""

    # Sample tag matches what measure_clustering_mstar.py would produce for
    # the same bin so that the clustering/ directory is shared.
    bin_tag = (f"Mcig{mstar_lo:.1f}-{mstar_hi:.1f}"
               f"_z{zmin:.3f}-{zmax:.3f}")
    log.info(f"\n{'='*60}")
    log.info(f"Bin: {bin_tag}  pimax={pimax:.0f} Mpc/h")
    log.info(f"{'='*60}")

    # Output directory — same root as the multipole script
    if args.outdir:
        outdir = Path(args.outdir) / bin_tag
    elif is_mock:
        outdir = paths["model_dir"] / "clustering" / bin_tag
    else:
        outdir = DATA_DATE_DIR / "clustering" / "data" / bin_tag

    # Skip entirely if the GCcomb output file already exists
    if not args.force_cuts and output_exists(outdir, zmin, zmax, is_mock,
                                             args.njack):
        log.info("  rppi output already exists — skipping bin")
        return True

    basedir = outdir / "catalogues"
    basedir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Apply cuts (skip if catalogues already exist)
    # ------------------------------------------------------------------
    if not args.force_cuts and catalogues_exist(basedir, args.regions):
        log.info("  Step 1: Catalogue files already present — skipping cut step")
    else:
        log.info("  Step 1: Applying sample selection cuts")

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
                        log.error(f"  Input not found: {src}")
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
                    log.error(f"  Input not found: {f}")
                    sys.exit(1)

        for src, dst in pairs:
            apply_cuts(src, dst, zmin, zmax, mstar_lo, mstar_hi, is_mock, log)

    # ------------------------------------------------------------------
    # Step 2: Symlink parent catalogues (data mode only)
    # ------------------------------------------------------------------
    if not is_mock:
        log.info("  Step 2: Symlinking parent catalogues")
        for src, name in [
            (PARENT_DAT, "BGS_BRIGHT_full_HPmapcut.dat.fits"),
            (PARENT_RAN, "BGS_BRIGHT_0_full_HPmapcut.ran.fits"),
        ]:
            dst = basedir / name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
            log.info(f"    {dst.name} → {src}")

    # ------------------------------------------------------------------
    # Step 3: Run xirunpc.py with --corr_type rppi
    #
    # Bin type strategy:
    #   pimax=40: use a custom .npz to get our preferred rp edges while
    #             keeping the standard pi grid [-40,40] in 1 Mpc/h steps.
    #   pimax=20: use a custom .npz with pi grid [-20,20] in 1 Mpc/h steps.
    # In both cases the .npz is written to a temporary directory and cleaned
    # up automatically when the script exits.
    # ------------------------------------------------------------------
    log.info("  Step 3: Running xirunpc.py (corr_type=rppi)")

    edges_file = make_edges_file(pimax, tmpdir)
    log.info(f"    Edges file: {edges_file}  "
             f"(rp: {RP_EDGES[0]:.3f}–{RP_EDGES[-1]:.3f} Mpc/h, "
             f"pi: ±{pimax:.0f} Mpc/h)")

    env = os.environ.copy()
    env["PYTHONPATH"] = f":{LSS_PY}:{env.get('PYTHONPATH', '')}"

    weight_type = "default" if is_mock else "pip_angular_bitwise"
    njack       = "0"       if is_mock else str(args.njack)

    cmd = [
        "python", str(lss_script),
        "--tracer",      "BGS_BRIGHT",
        "--basedir",     str(basedir),
        "--region",      *args.regions,
        "--corr_type",   "rppi",
        "--njack",       njack,
        "--weight_type", weight_type,
        "--bin_type",    str(edges_file),
        "--zlim",        f"{zmin:.3f}", f"{zmax:.3f}",
        "--nran",        "1",
        "--outdir",      str(outdir),
    ]

    log.info("  Command: " + " ".join(cmd))
    t0 = time.time()

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        log.error(f"  xirunpc.py failed (return code {result.returncode}) "
                  f"for bin {bin_tag}")
        return False

    log.info(f"  Completed in {time.time()-t0:.1f}s")
    log.info(f"  rppi output: {outdir}/rppi/")
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    args     = parse_args()
    log      = setup_logging("measure_wp_cigale_bins")
    is_mock  = (args.mode == "mock")

    lss_script = Path(args.lss_script) if args.lss_script else LSS_SCRIPT
    if not lss_script.exists():
        log.error(f"xirunpc.py not found: {lss_script}")
        sys.exit(1)

    paths = make_paths(obs_seed=args.obs_seed, sig_lgv_ref=args.sig_lgv_ref,
                       sig_lgv_slope=args.sig_lgv_slope,
                       f_colour=args.f_colour, alpha=args.alpha)

    log.info(f"mode={args.mode}")
    log.info(f"r_p edges: {RP_EDGES[0]:.3f} – {RP_EDGES[-1]:.3f} Mpc/h "
             f"({len(RP_EDGES)-1} bins)")

    # Build list of bins to run
    single = (args.zmin is not None or args.zmax is not None
              or args.mstar_lo is not None or args.mstar_hi is not None)
    if single:
        if not all(v is not None for v in
                   [args.zmin, args.zmax, args.mstar_lo, args.mstar_hi]):
            log.error("Single-bin mode requires all of --zmin, --zmax, "
                      "--mstar_lo, --mstar_hi")
            sys.exit(1)
        matches = [b for b in KP_BINS
                   if (abs(b[0] - args.zmin)     < 1e-6
                       and abs(b[1] - args.zmax)     < 1e-6
                       and abs(b[2] - args.mstar_lo) < 1e-6
                       and abs(b[3] - args.mstar_hi) < 1e-6)]
        if not matches:
            log.error(f"No KP bin found matching z={args.zmin}–{args.zmax}, "
                      f"log M*={args.mstar_lo}–{args.mstar_hi}. "
                      f"Check values against KP_BINS table.")
            sys.exit(1)
        bins_to_run = matches
    else:
        bins_to_run = KP_BINS

    log.info(f"Running {len(bins_to_run)} bin(s)")

    # Temporary directory for edge .npz files — cleaned up on exit
    with tempfile.TemporaryDirectory(prefix="wp_edges_") as tmpdir:
        failed = []
        for zmin, zmax, mstar_lo, mstar_hi, pimax in bins_to_run:
            ok = run_bin(zmin, zmax, mstar_lo, mstar_hi, pimax,
                         args, paths, lss_script, is_mock, log, tmpdir)
            if not ok:
                failed.append((zmin, zmax, mstar_lo, mstar_hi))

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"Done. {len(bins_to_run) - len(failed)}/{len(bins_to_run)} "
             f"bins completed successfully.")
    if failed:
        log.warning("Failed bins:")
        for b in failed:
            log.warning(f"  z={b[0]:.3f}–{b[1]:.3f}  "
                        f"log M*={b[2]:.1f}–{b[3]:.1f}")


if __name__ == "__main__":
    main()
