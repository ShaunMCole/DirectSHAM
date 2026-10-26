import numpy as np
from astropy.table import Column
from scipy.spatial import cKDTree


# ------------------------------------------------------------------
# Identify objects with unusable stellar-mass information.
#
# After matching a VAC onto the mock catalogue, some objects either:
#   (1) have no matching VAC entry, resulting in a masked column value;
#   (2) have a VAC match but a stellar mass of 0, which is treated as
#       an invalid/sentinel value.
#
# We create boolean masks for both cases and combine them into a
# single "bad" mask. This mask is subsequently used to identify
# objects requiring donor-based replacement of their derived galaxy
# properties.
# ------------------------------------------------------------------
def find_bad_rows(mock, col=None):
    # Boolean mask identifying objects with no VAC match.
    # The left join created a mask associated with all columns of the
    # table to flag objects without a match; here we extract that mask.
    nomatch = np.asarray(np.ma.getmaskarray(mock[col]))

    if col in ('LOGM', 'LOGMSTAR'):
        # Convert the masked column into a regular array, replacing
        # masked values with NaN so that numerical comparisons behave
        # predictably.
        logm = mock[col].filled(np.nan)

        # Identify objects whose matched VAC entry has a zero stellar
        # mass, treated as an invalid measurement.
        zero_logm = (logm == 0)

        # Combined mask: no VAC match or invalid stellar mass.
        bad = nomatch | zero_logm
    else:
        bad = nomatch

    # Report the number of objects in each category.
    print(f"no matching TARGETID:   {nomatch.sum():,}")
    if col in ('LOGM', 'LOGMSTAR'):
        print(f"zero_logm: {zero_logm.sum():,}")
    print(f"Total bad in some way:       {bad.sum():,}")

    return bad


# ------------------------------------------------------------------
# Populate missing galaxy properties using donor galaxies selected
# from a matched sample.
#
# Objects identified as having no valid VAC-derived properties are
# assigned values from a randomly selected donor galaxy with similar
# redshift, luminosity and rest-frame colour.
#
# The match is performed in the three-dimensional space:
#
#   Zobs
#   ABSMAG_RP1
#   REST_GMR_0P1   (tabulated rest-frame g-r, free of K-correction
#                   band dependence)
#
# using a KD-tree for efficient neighbour searching. Distances are
# scaled by user-defined tolerances so that a search radius of 1
# corresponds to:
#
#   |ΔZobs|           < dz
#   |ΔABSMAG_RP1|    < dmag
#   |ΔREST_GMR_0P1|  < dcol
#
# For each problematic object, one donor is selected at random from
# all candidate galaxies satisfying these criteria, and the stellar
# mass and star-formation properties are copied across, scaled by the
# luminosity ratio between donor and recipient. Objects for which no
# acceptable donor is found are recorded in the no_donor mask.
# ------------------------------------------------------------------
def assign_donors_first_pass(mock, bad, fill_cols, dz=0.02, dmag=0.1, dcol=0.1):
    # Rest-frame colour used as one of the matching variables.
    # REST_GMR_0P1 is the tabulated rest-frame g-r colour, free of
    # band-dependent K-correction effects.
    colour = mock['REST_GMR_0P1']

    # Define the donor sample (objects with valid properties) and
    # the recipient sample (objects requiring replacement values).
    good = ~bad

    # Construct the KD-tree search coordinates for the donor sample.
    # Dividing by the tolerances makes a distance of one correspond
    # to the adopted matching criteria.
    good_coords = np.column_stack([
        mock['Zobs'][good] / dz,
        mock['ABSMAG_RP1'][good] / dmag,
        colour[good] / dcol
    ])

    # Build KD-tree for efficient neighbour searches.
    tree = cKDTree(good_coords)

    # Construct equivalent coordinates for the objects requiring donors.
    bad_coords = np.column_stack([
        mock['Zobs'][bad] / dz,
        mock['ABSMAG_RP1'][bad] / dmag,
        colour[bad] / dcol
    ])

    # Store the catalogue indices of good and bad objects.
    good_idx = np.where(good)[0]
    bad_idx  = np.where(bad)[0]

    # Mask identifying objects for which no suitable donor was found.
    no_donor = np.zeros(len(mock), dtype=bool)

    # Mask that identifies rows for which a donor was found on this
    # first pass.
    donor1 = np.zeros(len(mock), dtype=bool)

    # Random-number generator used for donor selection.
    # Fixed seed ensures reproducibility.
    rng = np.random.default_rng(seed=42)

    # Counter for objects with no acceptable donor.
    n_nomatch = 0

    # Loop over all problematic objects.
    for i_bad, pos in zip(bad_idx, bad_coords):

        # Search for all donor galaxies lying within the adopted
        # tolerances in redshift, magnitude and colour.
        candidates = tree.query_ball_point(pos, r=1.0)

        # Record objects with no donor candidates.
        if len(candidates) == 0:
            no_donor[i_bad] = True
            n_nomatch += 1
            continue

        # Randomly select one donor from the candidate list.
        donor = good_idx[rng.choice(candidates)]

        # Record that a donor was found on this first pass.
        donor1[i_bad] = True

        # Luminosity scaling factor in dex.
        delta_mag = mock['ABSMAG_RP1'][donor] - mock['ABSMAG_RP1'][i_bad]

        # The fill_cols guards ensure each pass only updates its own
        # columns: CIGALE columns on the CIGALE pass, FSF columns on
        # the FSF pass.

        # Scale stellar mass and SFR by the luminosity ratio.
        if 'LOGM'    in fill_cols: mock['LOGM'][i_bad]    = mock['LOGM'][donor]    + 0.4 * delta_mag
        if 'LOGSFR'  in fill_cols: mock['LOGSFR'][i_bad]  = mock['LOGSFR'][donor]  + 0.4 * delta_mag
        if 'LOGMSTAR' in fill_cols: mock['LOGMSTAR'][i_bad] = mock['LOGMSTAR'][donor] + 0.4 * delta_mag
        if 'SFR'     in fill_cols: mock['SFR'][i_bad]     = mock['SFR'][donor] * 10**(0.4 * delta_mag)

        # Copy uncertainties unchanged.
        if 'LOGM_ERR'      in fill_cols: mock['LOGM_ERR'][i_bad]      = mock['LOGM_ERR'][donor]
        if 'LOGSFR_ERR'    in fill_cols: mock['LOGSFR_ERR'][i_bad]    = mock['LOGSFR_ERR'][donor]
        if 'LOGMSTAR_IVAR' in fill_cols: mock['LOGMSTAR_IVAR'][i_bad] = mock['LOGMSTAR_IVAR'][donor]
        if 'SFR_IVAR'      in fill_cols: mock['SFR_IVAR'][i_bad]      = mock['SFR_IVAR'][donor] / 10**(0.8 * delta_mag)

    # Report the number of objects still lacking a donor.
    print(f"No donor found for {n_nomatch:,} objects")

    return mock, donor1


# ------------------------------------------------------------------
# Attempt to populate any objects that remain unmatched after the
# initial KD-tree donor assignment.
#
# The remaining objects typically lie in sparsely populated regions
# of redshift–luminosity–colour space where no donor can be found
# within the adopted joint tolerances.
#
# For these objects we relax the redshift requirement completely,
# while retaining the original absolute-magnitude and colour
# tolerances. For each remaining object we:
#
#   1. Find all valid donor galaxies with similar luminosity
#      (ABSMAG_RP1) and rest-frame colour (REST_GMR_0P1).
#   2. Compute the redshift difference to each candidate donor.
#   3. Select the donor with the smallest redshift difference.
#   4. Copy the stellar-mass and star-formation quantities from
#      the donor to the target object, scaled by luminosity ratio.
#
# The col argument specifies which column to use to identify objects
# that still need a donor (LOGM for the CIGALE pass, LOGMSTAR for
# the FSF pass), ensuring the two passes are fully independent.
# fill_cols controls which columns are updated, again keeping the
# CIGALE and FSF passes separate.
# ------------------------------------------------------------------
def assign_donors_second_pass(mock, bad, fill_cols, dmag=0.1, dcol=0.1, col='LOGM'):
    # Identify objects whose derived properties remain unusable after
    # the first donor-assignment pass, using the appropriate column
    # for this pass (LOGM for CIGALE, LOGMSTAR for FSF).
    remaining = (
        np.ma.getmaskarray(mock[col]) |
        (mock[col].filled(np.nan) == 0)
    )
    good = ~bad

    # Catalogue indices of the remaining objects.
    remaining_idx = np.where(remaining)[0]

    # Rest-frame colour used in the matching.
    colour = mock['REST_GMR_0P1']

    # Loop over all remaining objects.
    for i_bad in remaining_idx:

        z0   = mock['Zobs'][i_bad]
        mag0 = mock['ABSMAG_RP1'][i_bad]
        col0 = colour[i_bad]

        # Find all valid donor galaxies satisfying the magnitude and
        # colour tolerances (redshift constraint relaxed).
        candidates = (
            good &
            (np.abs(mock['ABSMAG_RP1'] - mag0) < dmag) &
            (np.abs(colour - col0) < dcol)
        )

        if not np.any(candidates):
            continue

        cand_idx     = np.where(candidates)[0]
        dz_candidates = np.abs(mock['Zobs'][cand_idx] - z0)
        donor        = cand_idx[np.argmin(dz_candidates)]

        delta_mag = mock['ABSMAG_RP1'][donor] - mock['ABSMAG_RP1'][i_bad]

        # As in the first pass, fill_cols guards keep the CIGALE and
        # FSF passes updating only their own columns.
        if 'LOGM'    in fill_cols: mock['LOGM'][i_bad]    = mock['LOGM'][donor]    + 0.4 * delta_mag
        if 'LOGSFR'  in fill_cols: mock['LOGSFR'][i_bad]  = mock['LOGSFR'][donor]  + 0.4 * delta_mag
        if 'LOGMSTAR' in fill_cols: mock['LOGMSTAR'][i_bad] = mock['LOGMSTAR'][donor] + 0.4 * delta_mag
        if 'SFR'     in fill_cols: mock['SFR'][i_bad]     = mock['SFR'][donor] * 10**(0.4 * delta_mag)

        if 'LOGM_ERR'      in fill_cols: mock['LOGM_ERR'][i_bad]      = mock['LOGM_ERR'][donor]
        if 'LOGSFR_ERR'    in fill_cols: mock['LOGSFR_ERR'][i_bad]    = mock['LOGSFR_ERR'][donor]
        if 'LOGMSTAR_IVAR' in fill_cols: mock['LOGMSTAR_IVAR'][i_bad] = mock['LOGMSTAR_IVAR'][donor]
        if 'SFR_IVAR'      in fill_cols: mock['SFR_IVAR'][i_bad]      = mock['SFR_IVAR'][donor] / 10**(0.8 * delta_mag)

    # Report outcome of this second pass.
    print('After second pass')
    print(np.sum(np.ma.getmaskarray(mock[col])),
          ' objects that did not have an initial match remain and')
    print(np.sum(mock[col].filled(np.nan) == 0),
          ' objects that had a match but had logm=0 remain')

    # Set to NaN the values for objects for which no donor could be
    # found.  These probably have incorrect redshifts and should be
    # masked with mask=np.isfinite(mock[col]).
    remaining = np.ma.getmaskarray(mock[col])
    for c in fill_cols:
        mock[c][remaining] = np.nan

    return mock


# ------------------------------------------------------------------
# CIG/FSF provenance flag
#
# -1 : no donor could be found (stellar mass etc. set to NaN)
#  1 : direct TARGETID match with valid values
#  2 : donor found in first-pass KD-tree search
#  3 : donor found in second-pass nearest-redshift search within
#      colour and magnitude box
#
# The col argument specifies which column to inspect for validity
# (LOGM for the CIGALE pass, LOGMSTAR for the FSF pass), ensuring
# the two flags are set independently.
# ------------------------------------------------------------------
def add_provenance_flags(mock, bad, donor1, flag='cigflag', col='LOGM'):

    flags = np.ones(len(mock), dtype=np.int16)

    # Objects filled by the first-pass KD-tree donor search.
    first_pass_filled = donor1

    # Objects that failed the first pass but were recovered in the
    # second pass.
    second_pass_filled = bad & ~donor1 & np.isfinite(mock[col])

    # Objects that remain unrecovered after both passes.
    unresolved = ~np.isfinite(mock[col])

    flags[first_pass_filled]  = 2
    flags[second_pass_filled] = 3
    flags[unresolved]         = -1

    mock[flag] = Column(flags, name=flag, dtype=np.int16)

    print('flag=', flag)
    print('No match found :flag=-1:',                          np.sum(mock[flag] == -1))
    print('Matched TARGETID :flag=1:',                         np.sum(mock[flag] == 1))
    print('Matched col-mag-Z space :flag=2:',                  np.sum(mock[flag] == 2))
    print('Matched in col-mag space and nearest redshift:flag=3:', np.sum(mock[flag] == 3))
    return mock
