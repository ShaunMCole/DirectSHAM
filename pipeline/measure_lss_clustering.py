"""
measure_lss_clustering.py — Measure two-point correlation function multipoles
for the BGS LSS large-scale structure sample, defined by a distance-modulus-
only absolute magnitude cut (no K-correction), following the DESI LSS pipeline
convention (mkCat_subsamp.py, is_float branch).

Sample definition:
    abr  = rmag - DM(z)        [Mr - 5 log h, no K-correction]
    DM   = 5 * log10(dc(z) * (1+z)) + 25
    dc   uses cosmo_uchuu (Om0=0.3089, H0=100 i.e. Mpc/h)
    cut  : abr < LSS_ABSMAG_CUT  AND  LSS_ZMIN < z < LSS_ZMAX

Measures clustering for BOTH:
  - Lightcone mock  (unit weights, no PIP)
  - BGS DR2 data    (PIP + angular bitwise weights, njack=60)

Output sits under the lightcone clustering directory:
    <lightcone_dir>/clustering/lss/

Must be run on a compute node:
    salloc --nodes=1 --qos=interactive --constraint=cpu \\
           --account=desi --mem=512GB --time=04:00:00

Usage
-----
# Both data and mock (default):
python measure_lss_clustering.py

# Mock only:
python measure_lss_clustering.py --mock_only

# Data only:
python measure_lss_clustering.py --data_only
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import Table

# ---------------------------------------------------------------------------
# Pipeline modules
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DEFAULT_SIG_LGV_REF, DEFAULT_SIG_LGV_SLOPE,
    DEFAULT_F_COLOUR, DEFAULT_ALPHA, DEFAULT_OBS_SEED,
    DATA_DATE_DIR, cosmo_uchuu,
)
from pipeline_utils import make_paths, make_lightcone_paths, setup_logging

LSS_SCRIPT = Path(os.environ.get("SCRATCH", "")) / "LSS/scripts/xirunpc.py"
LSS_PY     = Path(os.environ.get("SCRATCH", "")) / "LSS/py"

# ---------------------------------------------------------------------------
# LSS sample definition
# ---------------------------------------------------------------------------
LSS_ABSMAG_CUT = -20.7   # Mr - 5 log h, no K-correction
LSS_ZMIN       =  0.1
LSS_ZMAX       =  0.3

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sig_lgv_ref",   type=float, default=DEFAULT_SIG_LGV_REF)
    p.add_argument("--sig_lgv_slope", type=float, default=DEFAULT_SIG_LGV_SLOPE)
    p.add_argument("--f_colour",      type=float, default=DEFAULT_F_COLOUR)
    p.add_argument("--alpha",         type=float, default=DEFAULT_ALPHA)
    p.add_argument("--obs_seed",      type=int,   default=DEFAULT_OBS_SEED)
    p.add_argument("--regions", nargs="+", default=["NGC", "SGC"],
                   choices=["NGC", "SGC"])
    p.add_argument("--lss_script", type=str, default=None)
    p.add_argument("--mock_only", action="store_true",
                   help="Measure mock clustering only (skip data).")
    p.add_argument("--data_only", action="store_true",
                   help="Measure data clustering only (skip mock).")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Distance modulus using cosmo_uchuu (Mpc/h → Mr - 5 log h)
# ---------------------------------------------------------------------------

def distmod_uchuu(z):
    """Distance modulus in Mr - 5 log h units, using cosmo_uchuu (Om0=0.3089, H0=100)."""
    z = np.asarray(z, dtype=float)
    dc = cosmo_uchuu.comoving_distance(z).value   # Mpc/h  (H0=100)
    return 5.0 * np.log10(dc * (1.0 + z)) + 25.0

# ---------------------------------------------------------------------------
# Sample selection cut
# ---------------------------------------------------------------------------

def apply_lss_cuts(infile, outfile, log, is_mock=False, is_random=False):
    """
    Read infile, apply LSS DM-only absolute magnitude and redshift cuts,
    write the trimmed catalogue to outfile ready for xirunpc.py.

    Parameters
    ----------
    is_mock   : bool  — if True, rename Zobs→Z (remove real-space Z first)
    is_random : bool  — if True, no Zobs column; use Z directly
    """
    log.info(f"Reading {infile}")
    data = Table.read(str(infile))
    log.info(f"  {len(data):,} rows")

    # Confirm rmag column present
    if 'rmag' not in data.colnames:
        log.error(f"  'rmag' column not found in {infile}. "
                  f"Available columns: {data.colnames}")
        sys.exit(1)

    # Redshift column
    if is_mock and not is_random:
        z_col = 'Zobs' if 'Zobs' in data.colnames else 'Z'
        if z_col == 'Z':
            log.warning("  'Zobs' not found in mock — using 'Z' (real space!)")
    else:
        z_col = 'Z'

    z    = np.asarray(data[z_col], dtype=float)
    rmag = np.asarray(data['rmag'], dtype=float)

    # DM-only absolute magnitude
    dm  = distmod_uchuu(z)
    abr = rmag - dm

    mask = (z >= LSS_ZMIN) & (z <= LSS_ZMAX) & (abr < LSS_ABSMAG_CUT)
    log.info(f"  After LSS cuts (abr<{LSS_ABSMAG_CUT}, "
             f"{LSS_ZMIN}<z<{LSS_ZMAX}): {mask.sum():,} rows")

    data_cut = data[mask]

    # Unit weights if not present
    if 'WEIGHT' not in data_cut.colnames:
        data_cut['WEIGHT'] = np.ones(len(data_cut), dtype=np.float32)

    # For mocks: rename Zobs → Z so xirunpc.py uses redshift-space positions
    if is_mock and not is_random and 'Zobs' in data_cut.colnames:
        if 'Z' in data_cut.colnames:
            data_cut.remove_column('Z')
        data_cut.rename_column('Zobs', 'Z')

    data_cut.write(str(outfile), format='fits', overwrite=True)
    log.info(f"  Written: {outfile}")

# ---------------------------------------------------------------------------
# Run xirunpc.py
# ---------------------------------------------------------------------------

def make_edges_file(outdir, log):
    """
    Write a .npz file containing custom smu bin edges extending to 150 Mpc/h.
    Uses the same log spacing as xirunpc.py's built-in 'log' preset but
    extended to 150 Mpc/h, and mu edges matching its default.
    Returns the path to the .npz file.
    """
    s_edges  = np.geomspace(0.01, 150., 100)  # 99 bins, ~5-6 Mpc/h at BAO scale
    mu_edges = np.linspace(-1., 1., 201)
    edges_path = Path(outdir) / "smu_edges_150.npz"
    np.savez(str(edges_path), s=s_edges, mu=mu_edges)
    log.info(f"Written bin edges: {edges_path}  "
             f"(s: {s_edges[0]:.3f}–{s_edges[-1]:.1f} Mpc/h, {len(s_edges)-1} bins)")
    return edges_path


def run_xirunpc(basedir, outdir, regions, lss_script, env, log,
                weight_type='default', njack=0, zmin=LSS_ZMIN, zmax=LSS_ZMAX):
    """Call xirunpc.py as a subprocess."""
    edges_path = make_edges_file(outdir, log)
    cmd = [
        "python", str(lss_script),
        "--tracer",      "BGS_BRIGHT",
        "--basedir",     str(basedir),
        "--region",      *regions,
        "--corr_type",   "smu",
        "--njack",       str(njack),
        "--weight_type", weight_type,
        "--bin_type",    str(edges_path),   # custom edges extending to 150 Mpc/h
        "--zlim",        f"{zmin:.2f}", f"{zmax:.2f}",
        "--nran",        "1",
        "--outdir",      str(outdir),
    ]
    log.info("Command: " + " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        log.error(f"xirunpc.py failed (return code {result.returncode})")
        sys.exit(result.returncode)
    log.info(f"xirunpc.py completed in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    log  = setup_logging("measure_lss_clustering")

    log.info(
        f"LSS sample: abr < {LSS_ABSMAG_CUT}, "
        f"{LSS_ZMIN} < z < {LSS_ZMAX}  (DM-only, no K-correction)"
    )

    lss_script = Path(args.lss_script) if args.lss_script else LSS_SCRIPT
    if not lss_script.exists():
        log.error(f"xirunpc.py not found at {lss_script}")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = f":{LSS_PY}:{env.get('PYTHONPATH', '')}"

    # Resolve paths
    paths    = make_paths(obs_seed=args.obs_seed,
                          sig_lgv_ref=args.sig_lgv_ref,
                          sig_lgv_slope=args.sig_lgv_slope,
                          f_colour=args.f_colour, alpha=args.alpha)
    lc_paths = make_lightcone_paths(sig_lgv_ref=args.sig_lgv_ref,
                                    sig_lgv_slope=args.sig_lgv_slope,
                                    f_colour=args.f_colour, alpha=args.alpha)

    # Output root: lightcone clustering dir / lss/
    lss_clus_dir = lc_paths["clustering_dir"] / "lss"

    # =========================================================================
    # MOCK
    # =========================================================================
    if not args.data_only:
        log.info("=== MOCK clustering ===")
        mock_outdir = lss_clus_dir / "mock"
        mock_basedir = mock_outdir / "catalogues"
        mock_basedir.mkdir(parents=True, exist_ok=True)
        mock_outdir.mkdir(parents=True, exist_ok=True)

        for cap in args.regions:
            mock_path = lc_paths[f"mock_lightcone_{cap}"]
            if not mock_path.exists():
                log.error(f"Lightcone mock not found: {mock_path}")
                sys.exit(1)
            apply_lss_cuts(
                mock_path,
                mock_basedir / f"BGS_BRIGHT_{cap}_clustering.dat.fits",
                log, is_mock=True,
            )

            ran_path = paths[f"formock_ran_{cap}"]
            if not ran_path.exists():
                log.error(f"Formock random not found: {ran_path}")
                sys.exit(1)
            apply_lss_cuts(
                ran_path,
                mock_basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran.fits",
                log, is_random=True,
            )

        run_xirunpc(mock_basedir, mock_outdir, args.regions, lss_script, env,
                    log, weight_type='default', njack=0)

    # =========================================================================
    # DATA
    # =========================================================================
    if not args.mock_only:
        log.info("=== DATA clustering ===")
        data_outdir  = lss_clus_dir / "data"
        data_basedir = data_outdir / "catalogues"
        data_basedir.mkdir(parents=True, exist_ok=True)
        data_outdir.mkdir(parents=True, exist_ok=True)

        for cap in args.regions:
            dat_path = DATA_DATE_DIR / f"BGS_BRIGHT_6thOct_{cap}.fits"
            if not dat_path.exists():
                log.error(f"Data catalogue not found: {dat_path}")
                sys.exit(1)
            apply_lss_cuts(
                dat_path,
                data_basedir / f"BGS_BRIGHT_{cap}_clustering.dat.fits",
                log, is_mock=False,
            )

            ran_path = DATA_DATE_DIR / f"rancat_fordata_{cap}.fits"
            if not ran_path.exists():
                log.error(f"Fordata random not found: {ran_path}")
                sys.exit(1)
            apply_lss_cuts(
                ran_path,
                data_basedir / f"BGS_BRIGHT_{cap}_0_clustering.ran.fits",
                log, is_random=True,
            )

        # pip_angular_bitwise requires the parent full catalogue in basedir
        # so that xirunpc.py can compute angular weights from fibered targets.
        # Symlink the DESI parent catalogue (read-only CFS path) into basedir.
        DESI_LSS_BASE = Path(
            "/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2.1/"
        )
        # Symlink parent full data catalogue
        full_src = DESI_LSS_BASE / "BGS_BRIGHT_full_HPmapcut.dat.fits"
        full_lnk = data_basedir / "BGS_BRIGHT_full_HPmapcut.dat.fits"
        if not full_lnk.exists():
            if not full_src.exists():
                log.error(f"Parent full catalogue not found: {full_src}")
                sys.exit(1)
            full_lnk.symlink_to(full_src)
            log.info(f"Symlinked parent catalogue: {full_lnk} -> {full_src}")

        # Symlink parent full randoms catalogue
        ran_src = DESI_LSS_BASE / "BGS_BRIGHT_0_full_HPmapcut.ran.fits"
        ran_lnk = data_basedir / "BGS_BRIGHT_0_full_HPmapcut.ran.fits"
        if not ran_lnk.exists():
            if not ran_src.exists():
                log.error(f"Parent full randoms not found: {ran_src}")
                sys.exit(1)
            ran_lnk.symlink_to(ran_src)
            log.info(f"Symlinked parent randoms: {ran_lnk} -> {ran_src}")

        run_xirunpc(data_basedir, data_outdir, args.regions, lss_script, env,
                    log, weight_type='pip_angular_bitwise', njack=60)

    log.info("measure_lss_clustering.py complete.")
    log.info(f"Results in: {lss_clus_dir}")


if __name__ == "__main__":
    main()
