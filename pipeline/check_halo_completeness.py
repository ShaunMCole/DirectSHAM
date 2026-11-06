"""
check_halo_completeness.py
--------------------------
Check whether each Uchuu halo file is self-contained with respect to
parent-satellite relationships, i.e. for every satellite (pid != -1)
in the file, its parent central (id == pid) is also in the same file.

Run on a compute node after loading the DESI environment:
    python check_halo_completeness.py
"""

import glob
import h5py
import numpy as np

HALODIR = "/global/cfs/cdirs/desi/mocks/cai/Uchuu-SHAM/Uchuu-halo-catalogs/halodir_045"

# Find all halo files
halofiles = sorted(glob.glob(f"{HALODIR}/halolist_*.h5"))
if not halofiles:
    # Try alternative naming conventions
    halofiles = sorted(glob.glob(f"{HALODIR}/*.h5"))
if not halofiles:
    halofiles = sorted(glob.glob(f"{HALODIR}/*.hdf5"))

if not halofiles:
    print(f"No halo files found in {HALODIR}")
    print("Contents of directory:")
    import os
    for f in sorted(os.listdir(HALODIR)):
        print(f"  {f}")
    raise SystemExit(1)

print(f"Found {len(halofiles)} halo files")
print(f"First file: {halofiles[0]}")

# Print keys of first file so we know column names
with h5py.File(halofiles[0], 'r') as f:
    print(f"Keys in file: {list(f.keys())}")

# Check just the first file to start
print(f"\nChecking {halofiles[0]} ...")
with h5py.File(halofiles[0], 'r') as f:
    ids  = f['id'][:]
    pids = f['pid'][:]

satellites  = pids != -1
parent_ids  = pids[satellites]
id_set      = set(ids)
found       = np.array([p in id_set for p in parent_ids])

print(f"  Total haloes in file:              {len(ids):,}")
print(f"  Centrals (pid == -1):              {(~satellites).sum():,}")
print(f"  Satellites (pid != -1):            {satellites.sum():,}")
print(f"  Parents found in same file:        {found.sum():,}")
print(f"  Fraction with parent in same file: {found.mean():.6f}")
print(f"  Orphaned satellites:               {(~found).sum():,}")
