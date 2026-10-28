"""
add_stellar_masses.py
=====================
Add CIGALE and FastSpecFit (FSF) stellar mass / SFR columns to a mock
catalogue by matching on TARGETID.  When a TARGETID is absent from a VAC,
or is present but carries invalid values, a secondary "donor" galaxy is
found that is close in redshift, absolute magnitude and rest-frame colour
(REST_GMR_0P1).  Provenance flags record how each row was filled.

Flag values (cigflag / fsfflag)
    1  – direct TARGETID match with valid values
    2  – first-pass donor  (KD-tree within dz, dmag, dcol; random selection)
    3  – second-pass donor (within dmag, dcol only; nearest in redshift)
   -1  – no donor found; columns are NaN

The CIGALE and FSF passes are fully independent: vacmatching functions
accept a col argument (LOGM or LOGMSTAR) that controls which column is
inspected to identify bad rows and set unresolved values to NaN, and
fill_cols controls which columns are updated in each pass.

Usage
-----
# Default: both regions, using parameters from config.py
python add_stellar_masses.py

# Single region:
python add_stellar_masses.py --reg NGC

# Override model parameters:
python add_stellar_masses.py --reg NGC SGC --sig_lgv_ref 0.18 --sig_lgv_slope 0.30
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, join

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))

import config
import vacmatching as vacm
from pipeline_utils import make_paths, setup_logging

# ---------------------------------------------------------------------------
# VAC catalogue paths
# ---------------------------------------------------------------------------
FSF_CAT = Path(
    "/global/cfs/cdirs/desicollab/science/gqp/y3gqp/v1.0/parent/"
    "BGS_ANY-y3gqp-parent-fsf-v1.0.fits"
)
CIG_CAT = Path(
    "/global/cfs/cdirs/desi/science/gqp/y3gqp/v2.0/st_mass_MS_cigale/"
    "BGS_BRIGHT-loa-LSSv2.1-cigale-v1.0.fits"
)
# v1.0 VAC (superseded):
# /global/cfs/cdirs/desicollab/science/gqp/y3gqp/v1.0/st_mass/parent/
# BGS-y3gqp-parent-v.1.0-cigale-MS-v1.0.fits

# Columns to read from each VAC (TARGETID first so it is the join key)
CIG_COLS      = ["TARGETID", "LOGM", "LOGM_ERR", "LOGSFR", "LOGSFR_ERR", "CHI2"]
FSF_COLS      = ["TARGETID", "LOGMSTAR", "LOGMSTAR_IVAR", "SFR", "SFR_IVAR"]

# Columns to propagate to the mock (everything except TARGETID)
CIG_FILL_COLS = CIG_COLS[1:]
FSF_FILL_COLS = FSF_COLS[1:]

# Quality thresholds applied when loading the VACs.
# Cuts recommended by M. Siudek (Oct 2026): LOGM>6 and CHI2<2.
# The previous cuts (LOGM>4, CHI2<17) were too loose; the tighter CHI2<2
# removes galaxies with poor WISE photometry (negative W1/W2 fluxes for
# faint dwarfs) that CIGALE cannot fit. This is a photometry incompleteness,
# not a galaxy property issue.
CIG_LOGM_MIN     = 6.0    # LOGM <= this treated as invalid
CIG_CHI2_MAX     = 2.0    # CHI2 >= this treated as invalid
FSF_BAD_TARGETID = 999999

# FSF h-correction: FSF LOGMSTAR and SFR are computed assuming H0=67.4
# (h=0.674) distances, while our pipeline uses h=1 (H0=100) units.
# Converting to h=1 adds +0.34 dex to LOGMSTAR and log(SFR), i.e. SFR
# multiplied by 10^0.34. Note: for sSFR the h factors cancel.
# Correction confirmed by M. Siudek (Oct 2026).
# Stellar mass scales as d^2, so the correction is 2*log10(h_fsf/h_ours)
# where h_fsf = 0.674 (H0=67.4) and h_ours = 1.0 (H0=100).
# 2 * log10(1.0 / 0.674) = +0.34 dex
H_FSF  = 0.674   # h assumed by FSF (H0=67.4 km/s/Mpc)
H_OURS = 1.0     # h used in our pipeline (H0=100 km/s/Mpc)
FSF_LOGM_H_CORR = 2.0 * np.log10(H_OURS / H_FSF)   # +0.34 dex

# Default donor-search tolerances (can be overridden on the CLI)
DEFAULT_DZ   = 0.02
DEFAULT_DMAG = 0.1
DEFAULT_DCOL = 0.1


# ===========================================================================
# VAC loading
# ===========================================================================

def load_cigale(path):
    logging.info("Reading CIGALE VAC: %s", path)
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        cig = Table({c: data[c] for c in CIG_COLS})
    logging.info("  Raw rows: %d", len(cig))
    mask = (cig["LOGM"] > CIG_LOGM_MIN) & (cig["CHI2"] < CIG_CHI2_MAX)
    cig = cig[mask]
    logging.info("  After quality cut (LOGM>%.1f, CHI2<%.1f): %d rows",
                 CIG_LOGM_MIN, CIG_CHI2_MAX, len(cig))
    return cig


def load_fsf(path):
    logging.info("Reading FSF VAC: %s", path)
    with fits.open(path, memmap=True) as hdul:
        data = hdul["SPECPHOT"].data
        fsf = Table({c: data[c] for c in FSF_COLS})
    mask = fsf["TARGETID"] != FSF_BAD_TARGETID
    fsf = fsf[mask]
    logging.info("  After bad-TARGETID cut: %d rows", len(fsf))
    return fsf


# ===========================================================================
# Per-region processing
# ===========================================================================

def process_region(reg, paths, cig, fsf, dz, dmag, dcol,
                   cat_in=None, cat_out=None):
    """
    Load a catalogue, match to both VACs, apply donor fallback,
    and write an augmented output file.

    If cat_in/cat_out are provided they take precedence over the
    mock_{reg} path from make_paths().  Use this to augment the
    master BGS data file rather than a mock.
    """
    if cat_in is not None:
        mock_in  = Path(cat_in)
        mock_out = Path(cat_out) if cat_out else \
                   mock_in.with_name(mock_in.stem + "_augmented.fits")
        rename_z = True    # data file has Z; rename to Zobs for vacmatching
        label = str(mock_in.name)
    else:
        mock_in  = Path(paths[f"mock_{reg}"])
        mock_out = mock_in.with_name(mock_in.stem + "_augmented.fits")
        rename_z = False   # mock file already has Zobs
        label = f"mock_{reg}"

    logging.info("=" * 60)
    logging.info("Input : %s", mock_in)
    logging.info("Output: %s", mock_out)

    if not mock_in.exists():
        raise FileNotFoundError(f"Input file not found: {mock_in}")

    # ------------------------------------------------------------------
    # 1. Load catalogue
    # ------------------------------------------------------------------
    logging.info("Reading %s …", label)
    mock = Table.read(str(mock_in))

    # Rename Z -> Zobs so vacmatching.py can use its standard column name.
    # We rename back before writing so the output column name matches the
    # input.
    if rename_z:
        mock.rename_column("Z", "Zobs")
    n_rows = len(mock)
    logging.info("  %d rows", n_rows)

    # ------------------------------------------------------------------
    # 2. Left-join CIGALE
    # ------------------------------------------------------------------
    logging.info("Joining CIGALE on TARGETID …")
    mock = join(mock, cig, keys="TARGETID", join_type="left")
    assert len(mock) == n_rows, \
        f"Row count changed after CIGALE join: {n_rows} -> {len(mock)}"

    # ------------------------------------------------------------------
    # 3. Left-join FSF
    # ------------------------------------------------------------------
    logging.info("Joining FSF on TARGETID …")
    mock = join(mock, fsf, keys="TARGETID", join_type="left")
    assert len(mock) == n_rows, \
        f"Row count changed after FSF join: {n_rows} -> {len(mock)}"

    # ------------------------------------------------------------------
    # 3b. Apply FSF h-correction
    # FSF values are in H0=67.4 units; convert to h=1 (H0=100) units.
    # LOGMSTAR += 0.34 dex; SFR *= 10^0.34 (linear); ivar scales as SFR^-2.
    # ------------------------------------------------------------------
    logging.info("Applying FSF h-correction (+%.4f dex to LOGMSTAR and log SFR) …",
                 FSF_LOGM_H_CORR)
    # Only correct rows that have a valid FSF match (unmasked LOGMSTAR)
    fsf_matched = ~np.asarray(np.ma.getmaskarray(mock["LOGMSTAR"]))
    mock["LOGMSTAR"][fsf_matched] += FSF_LOGM_H_CORR
    sfr_factor = 10**FSF_LOGM_H_CORR
    mock["SFR"][fsf_matched]      *= sfr_factor
    # IVAR scales as 1/SFR^2, so SFR_IVAR /= sfr_factor^2
    mock["SFR_IVAR"][fsf_matched] /= sfr_factor**2

    # ------------------------------------------------------------------
    # 4. CIGALE donor matching
    # ------------------------------------------------------------------
    logging.info("CIGALE donor matching …")
    bad_cig = vacm.find_bad_rows(mock, col="LOGM")
    mock, donor1_cig = vacm.assign_donors_first_pass(
        mock, bad_cig, CIG_FILL_COLS, dz=dz, dmag=dmag, dcol=dcol
    )
    mock = vacm.assign_donors_second_pass(
        mock, bad_cig, CIG_FILL_COLS, dmag=dmag, dcol=dcol, col='LOGM'
    )
    mock = vacm.add_provenance_flags(
        mock, bad_cig, donor1_cig, flag="cigflag", col='LOGM'
    )
    mock["cigflag"] = np.array(mock["cigflag"], dtype=np.int16)

    # ------------------------------------------------------------------
    # 5. FSF donor matching
    # ------------------------------------------------------------------
    logging.info("FSF donor matching …")
    bad_fsf = vacm.find_bad_rows(mock, col="LOGMSTAR")
    mock, donor1_fsf = vacm.assign_donors_first_pass(
        mock, bad_fsf, FSF_FILL_COLS, dz=dz, dmag=dmag, dcol=dcol
    )
    mock = vacm.assign_donors_second_pass(
        mock, bad_fsf, FSF_FILL_COLS, dmag=dmag, dcol=dcol, col='LOGMSTAR'
    )
    mock = vacm.add_provenance_flags(
        mock, bad_fsf, donor1_fsf, flag="fsfflag", col='LOGMSTAR'
    )
    mock["fsfflag"] = np.array(mock["fsfflag"], dtype=np.int16)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    for label, flag_col in [("CIGALE", "cigflag"), ("FSF", "fsfflag")]:
        f = mock[flag_col]
        logging.info(
            "%s: direct=%d  donor1=%d  donor2=%d  unresolved=%d",
            label,
            (f == 1).sum(), (f == 2).sum(),
            (f == 3).sum(), (f == -1).sum(),
        )

    # ------------------------------------------------------------------
    # 7. Write output
    # ------------------------------------------------------------------
    # Rename Zobs back to Z if we renamed it on load (data catalogue mode)
    if rename_z:
        mock.rename_column("Zobs", "Z")
    logging.info("Writing %s …", mock_out)
    mock = Table(mock)
    mock.write(str(mock_out), overwrite=True)
    logging.info("Done.")
    return mock_out


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Add CIGALE and FSF stellar masses/SFRs to mock catalogues."
    )
    p.add_argument(
        "--reg", nargs="+", default=["NGC", "SGC"],
        choices=["NGC", "SGC"],
        help="Region(s) to process (default: NGC SGC)",
    )
    p.add_argument("--obs_seed",      type=int,   default=0)
    p.add_argument("--sig_lgv_ref",   type=float, default=config.DEFAULT_SIG_LGV_REF)
    p.add_argument("--sig_lgv_slope", type=float, default=config.DEFAULT_SIG_LGV_SLOPE)
    p.add_argument("--f_colour",      type=float, default=config.DEFAULT_F_COLOUR)
    p.add_argument("--alpha",         type=float, default=config.DEFAULT_ALPHA)
    p.add_argument("--fsf_cat",  type=Path, default=FSF_CAT,
                   help="Path to FSF VAC FITS file")
    p.add_argument("--cig_cat",  type=Path, default=CIG_CAT,
                   help="Path to CIGALE VAC FITS file")
    p.add_argument("--dz",   type=float, default=DEFAULT_DZ,
                   help="Redshift tolerance for first-pass donor KD-tree search")
    p.add_argument("--dmag", type=float, default=DEFAULT_DMAG,
                   help="Absolute-magnitude tolerance for donor search")
    p.add_argument("--dcol", type=float, default=DEFAULT_DCOL,
                   help="Rest-frame colour tolerance for donor search")
    p.add_argument("--cat", type=Path, default=None,
                   help="Augment this catalogue instead of the mock (e.g. master BGS data file). "
                        "Uses Z column for redshift. Ignores --reg.")
    p.add_argument("--cat_out", type=Path, default=None,
                   help="Output path for --cat mode (default: <stem>_augmented.fits)")
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logging("add_stellar_masses")
    root = logging.getLogger()
    root.setLevel(log.level)
    for h in log.handlers:
        root.addHandler(h)
    logging.info("Regions: %s", args.reg)

    paths = make_paths(
        obs_seed=args.obs_seed,
        sig_lgv_ref=args.sig_lgv_ref,
        sig_lgv_slope=args.sig_lgv_slope,
        f_colour=args.f_colour,
        alpha=args.alpha,
    )

    # Load both VACs once; share across regions
    cig = load_cigale(args.cig_cat)
    fsf = load_fsf(args.fsf_cat)

    if args.cat is not None:
        # Augment a specific catalogue (e.g. master BGS data file)
        process_region(None, paths, cig, fsf,
                       dz=args.dz, dmag=args.dmag, dcol=args.dcol,
                       cat_in=args.cat, cat_out=args.cat_out)
    else:
        for reg in args.reg:
            process_region(reg, paths, cig, fsf,
                           dz=args.dz, dmag=args.dmag, dcol=args.dcol)

    logging.info("All regions complete.")


if __name__ == "__main__":
    main()
