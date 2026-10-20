"""
make_mock.py — Adapt the random catalogue to the mock footprint, run the
direct SHAM algorithm, and write the mock galaxy catalogue.

This script covers stages 2 and 3 of MockVpeakZpeak.ipynb.  It reads the
Uchuu checkpoint file produced by make_checkpoint.py (the expensive step)
and the fordata random catalogue produced by make_random.py, and is light
enough to run on a login node.

North must be run before South because the South random catalogue is
downsampled to match the North NMULT value.

Usage examples
--------------
# North first (always):
python make_mock.py --reg N --obs_seed 0

# Then South:
python make_mock.py --reg S --obs_seed 0

# Skip rewriting the formock randoms (e.g. rerunning SHAM only):
python make_mock.py --reg N --obs_seed 0 --skip_randoms
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io.fits import getheader
from astropy.table import Table
from scipy.interpolate import interp1d

# Pipeline modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DEFAULT_OBS_SEED,
    DEFAULT_SIG_LGV_REF,
    DEFAULT_SIG_LGV_SLOPE,
    DEFAULT_F_COLOUR,
    DEFAULT_ALPHA,
    MIN_VPEAK,
    N_NEIGH_VPEAK,
    N_NEIGH_LUM,
    N_ZBINS,
    DELTA_Z,
    LBOX,
    FRAC_BOX,
    TILES_FILE,
    KCORR_REF_GMR,
    cosmo,
)
from pipeline_utils import make_paths, write_fits_meta, check_fits_meta, setup_logging


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--reg", required=True, choices=["N", "S"],
        help="Photometric region to process: N (North) or S (South). "
             "North must be run before South.",
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
        help=f"Gaussian scatter in log10(Mpeak_Scale) for assembly-bias colour assignment (default: {DEFAULT_F_COLOUR})",
    )
    p.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"Correlation between Vpeak scatter and FracZpeak (default: {DEFAULT_ALPHA})",
    )
    p.add_argument(
        "--min_vpeak", type=float, default=MIN_VPEAK,
        help=f"Min Vpeak used when making the checkpoint (default: {MIN_VPEAK})",
    )
    p.add_argument(
        "--n_neigh_vpeak", type=int, default=N_NEIGH_VPEAK,
        help=f"N_neigh_vpeak used when making the checkpoint (default: {N_NEIGH_VPEAK})",
    )
    p.add_argument(
        "--n_neigh_lum", type=int, default=N_NEIGH_LUM,
        help=f"Neighbours in luminosity for colour matching (default: {N_NEIGH_LUM}). "
             "Set to 0 to disable colour matching.",
    )
    p.add_argument(
        "--n_zbins", type=int, default=N_ZBINS,
        help=f"Number of redshift slices for SHAM (default: {N_ZBINS})",
    )
    p.add_argument(
        "--skip_randoms", action="store_true",
        help="Do not generate a new formock random catalogue even if one does "
             "not exist.  Normally you should not set this flag — make_mock.py "
             "will automatically reuse (symlink) an existing compatible random "
             "catalogue rather than regenerating it.",
    )
    return p.parse_args()


# =============================================================================
# Step 1: load checkpoint
# =============================================================================

def load_checkpoint(paths, args, log):
    """Read the Uchuu checkpoint file and verify its metadata."""
    path = paths["subhalo_checkpoint"]
    if not path.exists():
        log.error(f"Checkpoint file not found: {path}")
        log.error("Run make_checkpoint.py first.")
        sys.exit(1)

    log.info(f"Reading checkpoint: {path}")
    t0 = time.time()
    subhalos = Table.read(str(path))
    log.info(f"Read {len(subhalos):,} subhalos in {time.time()-t0:.1f}s")

    # Verify key parameters match what was requested on the command line.
    check_fits_meta(
        subhalos.meta,
        {
            "REG":        args.reg,
            "OBS_SEED":   args.obs_seed,
            "SIGLGVREF":  args.sig_lgv_ref,
            "SIGLGVSLP":  args.sig_lgv_slope,
            "MIN_VPEAK":  args.min_vpeak,
            "NNVPEAK":    args.n_neigh_vpeak,
        },
        strict=True,
    )
    log.info("Checkpoint metadata verified.")
    return subhalos


# =============================================================================
# Step 2: load fordata randoms
# =============================================================================

def load_fordata_randoms(paths, args, log):
    """Read the fordata random catalogue for this region."""
    path = paths[f"fordata_ran_{args.reg}"]
    if not path.exists():
        log.error(f"Fordata random catalogue not found: {path}")
        log.error("Run make_random.py first.")
        sys.exit(1)

    log.info(f"Reading fordata randoms: {path}")
    t0 = time.time()
    randoms = Table.read(str(path))
    log.info(f"Read {len(randoms):,} randoms in {time.time()-t0:.1f}s")

    # Verify the region tag stored in the file matches the requested region.
    check_fits_meta(randoms.meta, {"REG": args.reg}, strict=True)

    # Add integer ID column used during SHAM index matching.
    randoms['ID'] = np.arange(len(randoms), dtype=np.int64)

    nmult  = randoms.meta['NMULT']
    deltaz = randoms.meta['DELTAZ']
    log.info(f"NMULT={nmult}  DELTAZ={deltaz}")
    return randoms, nmult, deltaz


# =============================================================================
# Step 3: adapt randoms to mock footprint and handle NMULT matching
# =============================================================================

def make_formock_randoms(randoms, paths, args, log):
    """
    Reassign RA/DEC of the random catalogue to the mock sky footprint,
    correct NMULT for the change in sky coverage, and — for the South —
    downsample to match the North NMULT.

    Returns the updated randoms table and the formock fsky value.
    """
    import MockCode as mc
    import catalogue_analysis as ca

    nr = len(randoms)
    log.info(f"Generating {nr:,} random positions within the {args.reg} mock footprint")
    t0 = time.time()

    ra_new, dec_new, fsky = mc.generate_randoms_in_mask(
        usemask=True,
        fsky_only=False,
        nr=nr,
        tiles_file=str(TILES_FILE),
        reg=args.reg,
        RAmin=None, RAmax=None, DECmin=None, DECmax=None,
    )
    randoms['RA']  = ra_new
    randoms['DEC'] = dec_new
    log.info(f"Footprint fsky={fsky:.5f}  ({time.time()-t0:.1f}s)")

    # Correct NMULT for the ratio of sky areas: fordata vs formock footprint.
    Sel             = ca.selection(args.reg)
    fsky_fordata    = Sel['area']
    nmult_formock   = randoms.meta['NMULT'] * fsky_fordata / fsky
    randoms.meta['NMULT'] = nmult_formock
    randoms.meta['FSKY']  = fsky
    log.info(f"Corrected NMULT={nmult_formock:.4f}")

    # South: downsample to match the North NMULT value.
    if args.reg == 'S':
        path_formock_N = make_paths(reg='N')["formock_ran_N"]
        if not path_formock_N.exists():
            log.error(
                f"North formock randoms not found: {path_formock_N}\n"
                "North must be run before South."
            )
            sys.exit(1)

        header     = getheader(str(path_formock_N), ext=1)
        nmult_N    = header['NMULT']
        log.info(f"North NMULT={nmult_N:.4f}  South NMULT={nmult_formock:.4f}")

        if nmult_formock > nmult_N:
            n_target = int(round(len(randoms) * nmult_N / nmult_formock))
            rng      = np.random.default_rng(seed=12345)
            randoms  = randoms[rng.choice(len(randoms), size=n_target, replace=False)]
            randoms.meta['NMULT'] = nmult_N
            log.info(
                f"South downsampled to {len(randoms):,} rows to match "
                f"North NMULT={nmult_N:.4f}"
            )
            nmult_formock = nmult_N
        else:
            log.info("South NMULT <= North NMULT — no downsampling needed.")

    return randoms, fsky, nmult_formock


# =============================================================================
# Step 4: run SHAM
# =============================================================================

def run_sham(subhalos, randoms, fsky, nmult_formock, deltaz, args, log):
    """
    Run abundance_match_subhalos_to_randoms and return the matched ID arrays.
    """
    import MockCode as mc
    import catalogue_analysis as ca

    Sel    = ca.selection(args.reg)
    rsphere = LBOX * FRAC_BOX
    z_bins  = np.linspace(Sel['zmin'], Sel['zmax'], args.n_zbins)

    from config import DATA_DIR

    diagnostics_path = str(
        DATA_DIR / f"diagnostics_{args.reg}_seed{args.obs_seed:03d}"
                   f"_sig_lgv{args.sig_lgv_ref:.2f}_{args.sig_lgv_slope:.2f}.fits"
    )

    log.info(
        f"Running SHAM: {args.n_zbins} redshift slices  "
        f"N_neigh_lum={args.n_neigh_lum}  fsky={fsky:.5f}  NMULT={nmult_formock:.4f}"
    )
    log.info(f"Diagnostics will be written to: {diagnostics_path}")
    t0 = time.time()

    subhalo_id, galaxy_id, z_insufficient = mc.abundance_match_subhalos_to_randoms(
        subhalos=subhalos,
        ran=randoms,
        lbox=LBOX,
        rsphere=rsphere,
        fsky=fsky,
        nmult=nmult_formock,
        z_bins=z_bins,
        scatter_sigma=0.0,   # scatter already applied in make_checkpoint.py
        DeltaZ=deltaz,
        reg=args.reg,
        N_neigh=args.n_neigh_lum,
        Rejection=False,
        diagnostics_path=diagnostics_path,
    )
    log.info(f"SHAM matched {len(subhalo_id):,} pairs in {time.time()-t0:.1f}s")
    if z_insufficient is not None:
        log.info(f"Subhalo density insufficient up to z={z_insufficient:.3f}")
    return subhalo_id, galaxy_id, z_insufficient


# =============================================================================
# Step 5: transfer galaxy properties to matched subhalos
# =============================================================================

def build_mock_catalogue(subhalos, randoms, subhalo_id, galaxy_id, args, log):
    """
    Unscramble the matched ID arrays, copy galaxy photometric properties to
    the matched subhalos, recompute redshift-dependent observables at the
    subhalo redshift, and return the matched-only subset.
    """
    import catalogue_analysis as ca
    from kcorrections import DESI_KCorrection

    log.info("Transferring galaxy properties to matched subhalos")
    t0 = time.time()

    # Build fast index lookups using argsort + searchsorted.
    sub_ids        = subhalos['ID']
    ran_ids        = randoms['ID']
    sub_sorter     = np.argsort(sub_ids)
    ran_sorter     = np.argsort(ran_ids)
    sub_ids_sorted = sub_ids[sub_sorter]
    ran_ids_sorted = ran_ids[ran_sorter]

    sub_rows = sub_sorter[np.searchsorted(sub_ids_sorted, subhalo_id)]
    gal_rows = ran_sorter[np.searchsorted(ran_ids_sorted, galaxy_id)]
    log.info(f"Index unscrambling done in {time.time()-t0:.1f}s")

    # Initialise output arrays with NaN / sentinel values.
    nsub     = len(subhalos)
    Mr       = np.full(nsub, np.nan, dtype=np.float32)
    Mrobs    = np.full(nsub, np.nan, dtype=np.float32)
    Mg       = np.full(nsub, np.nan, dtype=np.float32)
    rmag     = np.full(nsub, np.nan, dtype=np.float32)
    gmag     = np.full(nsub, np.nan, dtype=np.float32)
    Zr       = np.full(nsub, np.nan, dtype=np.float32)
    RestGMR  = np.full(nsub, np.nan, dtype=np.float32)
    TargetID = np.full(nsub, -1,     dtype=randoms['TARGETID'].dtype)

    # Copy intrinsic properties from matched random catalogue rows.
    Mr[sub_rows]       = randoms['ABSMAG_RP1'][gal_rows]
    Mrobs[sub_rows]    = randoms['ABSMAG_R'][gal_rows]
    Mg[sub_rows]       = randoms['ABSMAG_GP1'][gal_rows]
    Zr[sub_rows]       = randoms['Z'][gal_rows]
    RestGMR[sub_rows]  = randoms['REST_GMR_0P1'][gal_rows]
    TargetID[sub_rows] = randoms['TARGETID'][gal_rows]

    # Recompute apparent magnitudes at the subhalo redshift.
    Sel      = ca.selection(args.reg)
    kcorr_r  = Desi_KCorrection_for(args.reg, 'R')
    kcorr_g  = Desi_KCorrection_for(args.reg, 'G')

    z_gal      = randoms['Z'][gal_rows]
    DMOD       = 25.0 + 5.0 * np.log10(cosmo.luminosity_distance(z_gal).value)

    rmag[sub_rows] = (
        randoms['ABSMAG_RP1'][gal_rows]
        + DMOD
        - Sel['Qr'] * (z_gal - 0.1)
        + kcorr_r.k(z_gal, randoms['REST_GMR_0P1'][gal_rows])
    )
    gmag[sub_rows] = (
        randoms['ABSMAG_GP1'][gal_rows]
        + DMOD
        - Sel['Qg'] * (z_gal - 0.1)
        + kcorr_g.k(z_gal, randoms['REST_GMR_0P1'][gal_rows])
    )

    # Attach arrays to the subhalo table.
    subhalos['ABSMAG_RP1'] = Mr
    subhalos['ABSMAG_R']   = Mrobs
    subhalos['ABSMAG_GP1'] = Mg
    subhalos['rmag']       = rmag
    subhalos['gmag']       = gmag
    subhalos['Zr']         = Zr
    subhalos['REST_GMR_0P1'] = RestGMR
    subhalos['TARGETID']   = TargetID

    del DMOD, RestGMR, Mr, Mrobs, Mg, rmag, gmag, Zr, TargetID, z_gal
    gc.collect()

    # Keep only matched rows (those with a finite rmag).
    import MockCode as mc
    matched = np.isfinite(subhalos['rmag'])
    subhalos['keep'] = matched
    mock = mc.filter_table_staged(
        subhalos, subhalos['keep'], drop_mask=True, mask_name='keep'
    )
    log.info(f"Mock catalogue: {len(mock):,} matched galaxies")

    # Rename cosmological redshift column and drop internal rank column.
    mock.rename_column('Zu', 'Z')
    if 'i1' in mock.colnames:
        mock.remove_column('i1')

    return mock


def Desi_KCorrection_for(reg, band):
    """Convenience wrapper to get a DESI k-correction for a region and band."""
    from kcorrections import DESI_KCorrection
    return DESI_KCorrection(band=band, file='jmext', photsys=reg)


# =============================================================================
# Symlink helper
# =============================================================================

def _make_symlink_if_useful(path_ran_out, args, log):
    """
    Create a convenience symlink that makes it explicit which formock random
    file a given (sig_lgv, obs_seed) mock used, without copying the data.

    The symlink is named like the mock catalogue but with _randoms suffix,
    e.g. rancat6thOct_formock_N_seed000_sig_lgv0.10.fits → rancat6thOct_formock_N.fits

    Silently skips if the symlink already points to the right target.
    """
    from config import DATA_DATE, RANCAT_FORMOCK_TEMPLATE

    # Build a descriptive symlink name that encodes sig_lgv and obs_seed.
    stem    = RANCAT_FORMOCK_TEMPLATE.format(date=DATA_DATE, reg=args.reg)
    stem    = stem.replace(".fits", f"_seed{args.obs_seed:03d}_sig_lgv{args.sig_lgv_ref:.2f}_{args.sig_lgv_slope:.2f}.fits")
    symlink = path_ran_out.parent / stem

    if symlink.exists() or symlink.is_symlink():
        if symlink.is_symlink() and symlink.resolve() == path_ran_out.resolve():
            return   # already points to the right place
        symlink.unlink()

    try:
        symlink.symlink_to(path_ran_out.name)
        log.info(f"Symlink: {symlink.name} → {path_ran_out.name}")
    except OSError as e:
        log.warning(f"Could not create symlink {symlink}: {e}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    log  = setup_logging("make_mock")
    log.info(
        f"reg={args.reg}  obs_seed={args.obs_seed}  "
        f"sig_lgv_ref={args.sig_lgv_ref}  sig_lgv_slope={args.sig_lgv_slope}  "
        f"f_colour={args.f_colour}  alpha={args.alpha}  "
        f"n_neigh_lum={args.n_neigh_lum}  n_zbins={args.n_zbins}  "
        f"skip_randoms={args.skip_randoms}"
    )

    # Collect all paths needed by this script.
    paths = make_paths(
        reg=args.reg,
        obs_seed=args.obs_seed,
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        f_colour=args.f_colour,
        alpha=args.alpha,
        min_vpeak=args.min_vpeak,
        n_neigh_vpeak=args.n_neigh_vpeak,
    )

    # -------------------------------------------------------------------------
    # 1. Load checkpoint
    # -------------------------------------------------------------------------
    subhalos = load_checkpoint(paths, args, log)

    # -------------------------------------------------------------------------
    # 2. Load fordata randoms
    # -------------------------------------------------------------------------
    randoms, nmult, deltaz = load_fordata_randoms(paths, args, log)

    # -------------------------------------------------------------------------
    # 3. Adapt randoms to mock footprint; handle NMULT matching N/S
    # -------------------------------------------------------------------------
    randoms, fsky, nmult_formock = make_formock_randoms(randoms, paths, args, log)

    # -------------------------------------------------------------------------
    # 4. Write or reuse the formock random catalogue
    #
    # The formock randoms depend only on the BGS data catalogue, DELTA_Z,
    # NMULT and the sky footprint — not on sig_lgv, obs_seed, n_neigh_lum
    # or any other halo parameter.  So one formock random file suffices for
    # all mock realisations made from the same data catalogue.
    #
    # Strategy:
    #   a) If a compatible file already exists on disk → symlink to it and
    #      skip the (slow) generate_randoms_in_mask step entirely.
    #   b) If no file exists and --skip_randoms was not set → write it.
    #   c) If no file exists and --skip_randoms was set → abort with an error.
    # -------------------------------------------------------------------------
    path_ran_out = paths[f"formock_ran_{args.reg}"]
    path_ran_out.parent.mkdir(parents=True, exist_ok=True)

    randoms_written = False

    if path_ran_out.exists() or path_ran_out.is_symlink():
        # Check the existing file is compatible with the current parameters.
        log.info(f"Formock random catalogue already exists: {path_ran_out}")
        from astropy.io.fits import getheader
        existing_meta = getheader(str(path_ran_out), ext=1)
        try:
            # Check REG and DELTAZ exactly, but allow NMULT to vary by up
            # to 1% since fsky is estimated by random sampling and varies
            # slightly between runs.
            check_fits_meta(
                existing_meta,
                {"REG": args.reg, "DELTAZ": deltaz},
                strict=True,
            )
            nmult_stored = existing_meta.get("NMULT", None)
            if nmult_stored is not None:
                nmult_reldiff = abs(nmult_stored - nmult_formock) / nmult_formock
                if nmult_reldiff > 0.002:
                    raise ValueError(
                        f"Metadata mismatch for 'NMULT': "
                        f"file has {nmult_stored:.6f}, "
                        f"expected {nmult_formock:.6f} "
                        f"(relative difference {nmult_reldiff:.4f} > 0.2%)"
                    )
            log.info(
                "Existing formock randoms are compatible — reusing without "
                "regenerating.  Loading for SHAM."
            )
            # Reload randoms from the existing file so they have the correct
            # RA/DEC (mock footprint) and all required columns.
            randoms = Table.read(str(path_ran_out))
            randoms['ID'] = np.arange(len(randoms), dtype=np.int64)
            fsky         = existing_meta['FSKY']
            nmult_formock = existing_meta['NMULT']
            randoms_written = True   # nothing to write
        except ValueError as e:
            log.warning(
                f"Existing formock randoms are NOT compatible ({e}). "
                "Regenerating."
            )
            # Fall through to regenerate; make_formock_randoms was already
            # called above so randoms already has the correct footprint RA/DEC.

    if not randoms_written:
        if args.skip_randoms:
            log.error(
                "--skip_randoms was set but no compatible formock random "
                f"catalogue exists at {path_ran_out}.  Either run without "
                "--skip_randoms or provide the file."
            )
            sys.exit(1)

        # Remove internal columns not needed in the output file.
        ran_to_write = randoms.copy()
        for col in ('v', 'ID'):
            if col in ran_to_write.colnames:
                ran_to_write.remove_column(col)

        write_fits_meta(ran_to_write.meta, {
            "STAGE":  "make_mock",
            "REG":    args.reg,
            "DELTAZ": deltaz,
            "NMULT":  nmult_formock,
            "FSKY":   fsky,
        })
        ran_to_write.write(str(path_ran_out), overwrite=True)
        log.info(f"Written formock randoms ({len(ran_to_write):,} rows): {path_ran_out}")
        del ran_to_write

        # Keep the ID column on the working randoms table for SHAM.
        if 'ID' not in randoms.colnames:
            randoms['ID'] = np.arange(len(randoms), dtype=np.int64)

    # Now check whether we can point a sig_lgv-specific symlink at this file.
    # This makes it easy to see which randoms each mock used, without copying.
    _make_symlink_if_useful(path_ran_out, args, log)

    # -------------------------------------------------------------------------
    # 5. Run SHAM
    # -------------------------------------------------------------------------
    subhalo_id, galaxy_id, z_insufficient = run_sham(
        subhalos, randoms, fsky, nmult_formock, deltaz, args, log
    )

    # -------------------------------------------------------------------------
    # 6. Build mock catalogue
    # -------------------------------------------------------------------------
    mock = build_mock_catalogue(
        subhalos, randoms, subhalo_id, galaxy_id, args, log
    )
    del subhalos, randoms, subhalo_id, galaxy_id
    gc.collect()

    # -------------------------------------------------------------------------
    # 7. Write mock catalogue
    # -------------------------------------------------------------------------
    path_mock = paths[f"mock_{args.reg}"]
    path_mock.parent.mkdir(parents=True, exist_ok=True)

    write_fits_meta(mock.meta, {
        "STAGE":      "make_mock",
        "REG":        args.reg,
        "OBS_SEED":   args.obs_seed,
        "SIGLGVREF":  args.sig_lgv_ref,
        "SIGLGVSLP":  args.sig_lgv_slope,
        "FCOLOUR":    args.f_colour,
        "ALPHA":      args.alpha,
        "MIN_VPEAK":  args.min_vpeak,
        "NNVPEAK":    args.n_neigh_vpeak,
        "NNLUM":      args.n_neigh_lum,
        "NZBINS":     args.n_zbins,
        "DELTAZ":     deltaz,
        "NMULT":      nmult_formock,
        "FSKY":       fsky,
        "Z_INSUF":    float(z_insufficient) if z_insufficient is not None else -1.0,
    })

    t0 = time.time()
    mock.write(str(path_mock), overwrite=True)
    log.info(f"Written mock catalogue in {time.time()-t0:.1f}s")
    log.info(f"→ {path_mock}")
    log.info("make_mock.py complete.")


if __name__ == "__main__":
    main()
