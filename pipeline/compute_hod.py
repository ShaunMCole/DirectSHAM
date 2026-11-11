"""
compute_hod.py — Compute and plot the HOD for each Mr threshold sample.

For each sample:
  Numerator:   histogram of Mhost for mock galaxies passing Mr and Zobs cuts,
               split into centrals (pid == -1) and satellites.
  Denominator: histogram of Mhost for all subhalos with pid == -1 and
               Zobs <= zmax from the checkpoint files.

The checkpoint files are already cut to the N and S footprints, so no
additional masking is needed. Zobs is used throughout for consistency.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitsio

# =============================================================================
# Configuration
# =============================================================================

MODEL_DIR = Path(
    "/global/cfs/cdirs/desi/users/smcole/DirectSHAM/6thOct/N_neigh_lum50"
    "/seed000_sig_lgv0.18_0.30_fc0.00_al0.00"
)

MOCK_CATS = [
    MODEL_DIR / "mock_NGC.fits",
    MODEL_DIR / "mock_SGC.fits",
]

CHECKPOINT_CATS = [
    MODEL_DIR / "subhalos_N_70.0_N_neigh_vpeak1000.fits",
    MODEL_DIR / "subhalos_S_70.0_N_neigh_vpeak1000.fits",
]

OUT_STEM = Path(__file__).resolve().parent / "hod"

# Mr threshold samples
SAMPLES = [
    dict(mr_cut=-22.0, zmin=0.02, zmax=0.50),
    dict(mr_cut=-21.5, zmin=0.02, zmax=0.42),
    dict(mr_cut=-21.0, zmin=0.02, zmax=0.34),
    dict(mr_cut=-20.5, zmin=0.02, zmax=0.28),
    dict(mr_cut=-20.0, zmin=0.02, zmax=0.23),
    dict(mr_cut=-19.5, zmin=0.02, zmax=0.19),
    dict(mr_cut=-19.0, zmin=0.02, zmax=0.15),
    dict(mr_cut=-18.5, zmin=0.02, zmax=0.12),
    dict(mr_cut=-18.0, zmin=0.02, zmax=0.10),
    dict(mr_cut=-17.5, zmin=0.02, zmax=0.08),
]

# Halo mass bins (log10 Mhost in Msun/h)
MMIN, MMAX, NM = 11.0, 15.0, 30
MBINS  = np.linspace(MMIN, MMAX, NM + 1)
MCENTS = 0.5 * (MBINS[:-1] + MBINS[1:])

# Colour sequence: bright (warm) to faint (cool)
COLORS = plt.cm.RdYlBu_r(np.linspace(0.05, 0.95, len(SAMPLES)))

# =============================================================================
# Load catalogues
# =============================================================================

def load_combined(paths, columns):
    parts = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"WARNING: not found: {p}")
            continue
        parts.append(fitsio.read(str(p), columns=columns))
    return np.concatenate(parts)


print("Loading mock catalogues...")
mock = load_combined(MOCK_CATS,
                     columns=['Mhost', 'pid', 'Zobs', 'ABSMAG_RP1'])

print("Loading checkpoint catalogues...")
chk  = load_combined(CHECKPOINT_CATS,
                     columns=['Mhost', 'pid', 'Zobs'])

# log10 halo mass
log_mvir_mock = np.log10(mock['Mhost'])
log_mvir_chk  = np.log10(chk['Mhost'])

# Central flag
is_cen_mock = mock['pid'] == -1
is_cen_chk  = chk['pid']  == -1

# =============================================================================
# Compute HODs
# =============================================================================

def compute_hod(mr_cut, zmin, zmax):
    """Return (Mcents, N_cen, N_sat, N_all) mean occupations per halo."""

    # --- Numerator: mock galaxies ---
    gal_mask = (mock['ABSMAG_RP1'] <= mr_cut) & (mock['Zobs'] >= zmin) & (mock['Zobs'] <= zmax)
    cen_mask = gal_mask & is_cen_mock
    sat_mask = gal_mask & ~is_cen_mock

    n_cen = np.histogram(log_mvir_mock[cen_mask], bins=MBINS)[0].astype(float)
    n_sat = np.histogram(log_mvir_mock[sat_mask], bins=MBINS)[0].astype(float)

    # --- Denominator: all host haloes (centrals) in same volume ---
    halo_mask = is_cen_chk & (chk['Zobs'] >= zmin) & (chk['Zobs'] <= zmax)
    n_halo    = np.histogram(log_mvir_chk[halo_mask], bins=MBINS)[0].astype(float)

    # Avoid division by zero
    good = n_halo > 0
    hod_cen = np.where(good, n_cen / n_halo, np.nan)
    hod_sat = np.where(good, n_sat / n_halo, np.nan)
    hod_all = np.where(good, (n_cen + n_sat) / n_halo, np.nan)

    return MCENTS, hod_cen, hod_sat, hod_all

# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots(figsize=(3.35, 3.8), constrained_layout=True)

for sample, color in zip(SAMPLES, COLORS):
    mr  = sample['mr_cut']
    z0  = sample['zmin']
    z1  = sample['zmax']
    mc, hod_cen, hod_sat, hod_all = compute_hod(mr, z0, z1)

    good = np.isfinite(hod_all) & (hod_all > 0)

    ax.plot(mc[good], hod_all[good], '-',  color=color, lw=1.5,
            label=rf'$M_r < {mr}$')
    ax.plot(mc[good], hod_cen[good], '--', color=color, lw=1.0)
    ax.plot(mc[good], hod_sat[good], ':',  color=color, lw=1.0)

ax.set_yscale('log')
ax.set_xlim(MMIN, MMAX)
ax.set_ylim(10**-2.5, 10**1.5)
ax.set_xlabel(r'$\log_{10}(M_{\rm host} \; [h^{-1} M_\odot])$')
ax.set_ylabel(r'$\langle N \rangle$')
ax.grid(alpha=0.2)

# Line style legend — top left
style_handles = [
    plt.Line2D([], [], color='k', ls='-',  lw=1.5, label='Total'),
    plt.Line2D([], [], color='k', ls='--', lw=1.0, label='Centrals'),
    plt.Line2D([], [], color='k', ls=':',  lw=1.0, label='Satellites'),
]
leg1 = ax.legend(handles=style_handles, fontsize=6, frameon=False,
                 loc='upper left')
ax.add_artist(leg1)

# Magnitude legend — bottom right, single column
mag_handles = [
    plt.Line2D([], [], color=c, ls='-', lw=1.5, label=rf'$M_r < {s["mr_cut"]}$')
    for s, c in zip(SAMPLES, COLORS)
]
ax.legend(handles=mag_handles, fontsize=6, frameon=False,
          loc='lower right', ncol=1)

for ext in ('pdf', 'jpg'):
    out = f"{OUT_STEM}.{ext}"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
plt.close(fig)
