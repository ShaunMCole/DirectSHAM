import numpy as np
from astropy.table import Column
from scipy.spatial import cKDTree



# ------------------------------------------------------------------
# Identify objects with unusable stellar-mass information.
#
# After matching the CIG catalogue onto the mock catalogue, some
# objects either:
#   (1) have no matching CIG entry, resulting in a masked LOGM value;
#   (2) have a CIG match but a stellar mass of LOGM=0, which is
#       treated as an invalid/sentinel value.
#
# We create boolean masks for both cases and combine them into a
# single "bad" mask. This mask is subsequently used to identify
# objects requiring donor-based replacement of their derived galaxy
# properties.
# ------------------------------------------------------------------
def find_bad_rows(mock,col=None):
    # Boolean mask identifying objects with no VAC match
    # The left join created a mask asscoiated with all columns of the table to flag objects without a match.
    # Here we are just extracting that mask 
    nomatch = np.asarray(np.ma.getmaskarray(mock[col]))

    if (col=='LOGM'):
        # Convert the masked LOGM column into a regular array, replacing
        # masked values with NaN so that numerical comparisons behave
        # predictably.
        logm = mock[col].filled(np.nan)
    
        # Identify objects whose matched CIG entry has LOGM=0.
        # These are treated as invalid measurements.
        zero_logm = (logm == 0)
    
        # Combined mask of all objects requiring replacement:
        # either no CIG match or an invalid LOGM value.
        bad = nomatch | zero_logm
    elif (col=='LOGMSTAR'):
        # Convert the masked LOGMSTAR column into a regular array, replacing
        # masked values with NaN so that numerical comparisons behave
        # predictably.
        logm = mock[col].filled(np.nan)
    
        # Identify objects whose matched FSF entry has LOGM=0.
        # These are treated as invalid measurements.
        zero_logm = (logm == 0)
    
        # Combined mask of all objects requiring replacement:
        # either no FSF match or an invalid LOGM value.
        bad = nomatch | zero_logm
    else:
        bad= nomatch
    
    # Report the number of objects in each category.
    print(f"no matching TARGETID:   {nomatch.sum():,}")
    if ( (col=='LOGM')|(col=='LOGMSTAR') ): print(f"zero_logm: {zero_logm.sum():,}")
    print(f"Total bad in some way:       {bad.sum():,}")

    return bad

# ------------------------------------------------------------------
# Populate missing galaxy properties using donor galaxies selected
# from a matched sample.
#
# Objects identified as having no valid CIG-derived properties
# (either no TARGETID match or invalid LOGM=0 values) are assigned
# values from a randomly selected donor galaxy with similar redshift,
# luminosity and colour.
#
# The match is performed in the three-dimensional space:
#
#   Zobs
#   ABSMAG_RP1
#   ABSMAG_GP1 - ABSMAG_RP1
#
# using a KD-tree for efficient neighbour searching. Distances are
# scaled by user-defined tolerances so that a search radius of 1
# corresponds to:
#
#   |ΔZobs|        < dz
#   |ΔABSMAG_RP1| < dmag
#   |Δ(g-r)|      < dcol
#
# For each problematic object, one donor is selected at random from
# all candidate galaxies satisfying these criteria, and the stellar
# mass and star-formation properties are copied across. Objects for
# which no acceptable donor is found are recorded in the no_donor
# mask for subsequent inspection.
# ------------------------------------------------------------------
def assign_donors_first_pass(mock,bad,fill_cols,dz=0.02,dmag=0.1,dcol=0.1):
    # Calculate rest-frame colour used as one of the matching variables.
    colour = mock['ABSMAG_GP1'] - mock['ABSMAG_RP1']
    
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
    bad_idx = np.where(bad)[0]
    
    # Mask identifying objects for which no suitable donor was found. Initialized to all zeros (=False)
    no_donor = np.zeros(len(mock), dtype=bool)

    #Mask (initialized to False) that we use to identify the rows for which a donor was found on this first pass
    donor1 = np.zeros(len(mock), dtype=bool)
    
    # Random-number generator used for donor selection.
    # Fixed seed ensures reproducibility.
    rng = np.random.default_rng(seed=42)
    
    # Counter for objects with no acceptable donor.
    n_nomatch = 0
    
    # Loop over all problematic objects.
    for i_bad, pos in zip(bad_idx, bad_coords):
    
        # Search for all donor galaxies lying within the
        # adopted tolerances in redshift, magnitude and colour.
        candidates = tree.query_ball_point(pos, r=1.0)
    
        # Record objects with no donor candidates.
        if len(candidates) == 0:
            no_donor[i_bad] = True
            n_nomatch += 1
            continue
    
        # Randomly select one donor from the candidate list.
        donor = good_idx[rng.choice(candidates)]

        donor1[i_bad] = True #update the mask to indicate we found a donor
        
        # Luminosity scaling factor in dex
        delta_mag = (mock['ABSMAG_RP1'][donor]- mock['ABSMAG_RP1'][i_bad])
    
        # Scale stellar mass and SFR by the luminosity ratio
        if 'LOGM' in fill_cols: mock['LOGM'][i_bad] = (mock['LOGM'][donor]+ 0.4 * delta_mag)
        if 'LOGSFR' in fill_cols: mock['LOGSFR'][i_bad] = (mock['LOGSFR'][donor]+ 0.4 * delta_mag)
    
        # Copy uncertainties unchanged
        if 'LOGM_ERR' in fill_cols: mock['LOGM_ERR'][i_bad] = mock['LOGM_ERR'][donor]
        if 'LOGSFR_ERR' in fill_cols: mock['LOGSFR_ERR'][i_bad] = mock['LOGSFR_ERR'][donor]

        if 'LOGMSTAR' in fill_cols: mock['LOGMSTAR'][i_bad] = (mock['LOGMSTAR'][donor]+ 0.4 * delta_mag)
        if 'SFR' in fill_cols: mock['SFR'][i_bad] = (mock['SFR'][donor]*10**( 0.4 * delta_mag))    
        if 'LOGMSTAR_IVAR' in fill_cols: mock['LOGMSTAR_IVAR'][i_bad] = mock['LOGMSTAR_IVAR'][donor]
        if 'SFR_IVAR' in fill_cols: mock['SFR_IVAR'][i_bad] = (mock['SFR_IVAR'][donor]/10**( 0.8 * delta_mag))
    
    # Report the number of objects still lacking a donor.
    print(f"No donor found for {n_nomatch:,} objects")

    # Report success of this first pass at finding donors
    #
    #print('Of the objects for which we have not found a donor')
    #print(np.sum(np.ma.getmaskarray(mock['LOGM'])),' had no TARGETID match ')
    #print(np.sum(mock['LOGM'] == 0),' had TARGETID match but logm=0')
    #print('mask of objects in no_donor mask:',no_donor.sum())
    
    return mock,donor1


# ------------------------------------------------------------------
# Attempt to populate any objects that remain unmatched after the
# initial KD-tree donor assignment.
#
# The remaining objects typically lie in sparsely populated regions
# of redshift space where no donor can be found within the adopted
# joint tolerances in redshift, luminosity and colour.
#
# For these objects we relax the redshift requirement completely,
# while retaining the original absolute-magnitude and colour
# tolerances. For each remaining object we:
#
#   1. Find all valid donor galaxies with similar luminosity
#      (ABSMAG_RP1) and colour (ABSMAG_GP1 - ABSMAG_RP1).
#   2. Compute the redshift difference between the target object
#      and each candidate donor.
#   3. Select the donor with the smallest redshift difference.
#   4. Copy the stellar-mass and star-formation quantities from
#      the donor to the target object.
#
# This second-pass procedure is intended to recover a small number
# of objects that could not be handled by the original KD-tree
# matching while still ensuring close agreement in the physically
# most important observables (luminosity and colour).
# ------------------------------------------------------------------
def assign_donors_second_pass(mock,bad,fill_cols,dmag=0.1,dcol=0.1):
    # Identify objects whose derived properties remain unusable after
    # the first donor-assignment pass.
    remaining = (
        np.ma.getmaskarray(mock['LOGM']) |
        (mock['LOGM'].filled(np.nan) == 0)
    )
    good =~bad
    
    # Catalogue indices of the remaining objects.
    remaining_idx = np.where(remaining)[0]
    
    # Recompute galaxy colour used in the matching.
    colour = mock['ABSMAG_GP1'] - mock['ABSMAG_RP1']
    
    # Loop over all remaining objects.
    for i_bad in remaining_idx:
    
        # Properties of the target object 
        z0 = mock['Zobs'][i_bad]
        mag0 = mock['ABSMAG_RP1'][i_bad]
        col0 = colour[i_bad]
    
    
        # Find all valid donor galaxies satisfying the original
        # magnitude and colour tolerances.
        candidates =  good & (np.abs(mock['ABSMAG_RP1'] - mag0)< dmag) & (np.abs(colour - col0) < dcol) 
    
        # If no suitable donors exist, leave the object untouched.
        if not np.any(candidates):
            continue
    
        # Convert the candidate mask to catalogue indices.
        cand_idx = np.where(candidates)[0]
    
        # Compute redshift differences between the target object and
        # all candidate donors.
        dz_candidates = np.abs(mock['Zobs'][cand_idx] - z0)
    
        # Select the candidate with the smallest redshift difference.
        donor = cand_idx[np.argmin(dz_candidates)]
    
        # Luminosity scaling factor in dex
        delta_mag = (mock['ABSMAG_RP1'][donor]- mock['ABSMAG_RP1'][i_bad])
    
        # Scale stellar mass and SFR by the luminosity ratio
        if 'LOGM' in fill_cols: mock['LOGM'][i_bad] = (mock['LOGM'][donor]+ 0.4 * delta_mag)
        if 'LOGSFR' in fill_cols: mock['LOGSFR'][i_bad] = (mock['LOGSFR'][donor]+ 0.4 * delta_mag)
    
        # Copy uncertainties unchanged
        if 'LOGM_ERR' in fill_cols: mock['LOGM_ERR'][i_bad] = mock['LOGM_ERR'][donor]
        if 'LOGSFR_ERR' in fill_cols: mock['LOGSFR_ERR'][i_bad] = mock['LOGSFR_ERR'][donor]

        if 'LOGMSTAR' in fill_cols: mock['LOGMSTAR'][i_bad] = (mock['LOGMSTAR'][donor]+ 0.4 * delta_mag)
        if 'SFR' in fill_cols: mock['SFR'][i_bad] = (mock['SFR'][donor]*10**( 0.4 * delta_mag))    
        if 'LOGMSTAR_IVAR' in fill_cols: mock['LOGMSTAR_IVAR'][i_bad] = mock['LOGMSTAR_IVAR'][donor]
        if 'SFR_IVAR' in fill_cols: mock['SFR_IVAR'][i_bad] = (mock['SFR_IVAR'][donor]/10**( 0.8 * delta_mag))
    
    #Report success of this second pass at finding donors
    print('After second pass')
    print(np.sum(np.ma.getmaskarray(mock['LOGM'])),' objects that did not have an initial match remain and')
    print(np.sum(mock['LOGM'].filled(np.nan) == 0),' objects that had a match but had logm=0 remain')

    # Set to nan the values for the objects for which no donors could be found.  
    # They probably have incorrect redshifts and should be masked out with
    # mask=np.isfinite(mock['LOGM'])

    remaining = np.ma.getmaskarray(mock['LOGM'])

    for col in ['LOGM', 'LOGM_ERR', 'LOGSFR', 'LOGSFR_ERR']:
        mock[col][remaining] = np.nan
    
    return mock


# ------------------------------------------------------------------
# CIG/FSF provenance flag
#
# -1 : no donor could be found (LOGM etc. set to NaN)
#  1 : direct CIG match
#  2 : donor found in first-pass KD-tree search
#  3 : donor found in second-pass nearest-redshift search within colour and magnitude box
# ------------------------------------------------------------------
def add_provenance_flags(mock,bad,donor1,flag='cigflag', col='LOGM'):


    flags= np.ones(len(mock), dtype=np.int16)

    # Objects requiring donor assignment
    first_pass_filled = donor1
    
    # Objects that failed the first pass but were recovered in the
    # second pass
    second_pass_filled = bad & ~donor1 & np.isfinite(mock[col])
    
    # Objects that remain unrecovered
    unresolved = ~np.isfinite(mock[col])
    
    flags[first_pass_filled] = 2
    flags[second_pass_filled] = 3
    flags[unresolved] = -1
    mock[flag]=Column(flags,name=flag,dtype=np.int16)
    
    print('flag=',flag)
    print('No match found :flag=-1:', np.sum(mock[flag] == -1))
    print('Matched TARGETID :flag=1:', np.sum(mock[flag] == 1))
    print('Matched col-mag-Z space :flag=2:', np.sum(mock[flag] == 2))
    print('Matched in col-mag space and nearest redshift:flag=3:', np.sum(mock[flag] == 3))
    return mock