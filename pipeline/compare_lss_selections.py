"""
compare_lss_selections.py
--------------------------
Compare two BGS absolute-magnitude sample definitions over 0.1 < z < 0.3:

  A) "LSS no-kcorr":  abr_nodkcorr = rmag - distmod(Zobs) < -20.7
                       (matches mkCat_subsamp.py is_float branch)

  B) "Pipeline":       ABSMAG_RP1 < -20.7
                       (Moore+2026 K-correction at 0.1(g-r)=0.77)

Produces a two-panel figure:
  Top:    Comoving number density n(z) [h^3 Mpc^-3] for both selections
          and the overlap (in both A and B).
  Middle: Fraction of A that is also in B, and vice versa, as a function of z.
  Bottom: Histograms of rest-frame 0.1(g-r) colour in three redshift bins,
          for both selections.

Can be run on the BGS data catalogue or on a mock.

Usage
-----
# On the mock (NGC):
python compare_lss_selections.py \
    --cat $DATA_DIR/6thOct/N_neigh_lum50/seed000_sig_lgv0.18_0.30_fc0.00_al0.00/mock_NGC.fits \
    --zcol Zobs --label "Mock NGC"

# On the data (NGC):
python compare_lss_selections.py \
    --cat $DATA_DIR/6thOct/BGS_BRIGHT_6thOct_NGC.fits \
    --zcol Z --label "BGS DR2 NGC"

# On both together:
python compare_lss_selections.py \
    --cat $DATA_DIR/6thOct/BGS_BRIGHT_6thOct_NGC.fits \
    --cat2 $DATA_DIR/6thOct/N_neigh_lum50/seed000_sig_lgv0.18_0.30_fc0.00_al0.00/mock_NGC.fits \
    --zcol Z --zcol2 Zobs --label "BGS DR2 NGC" --label2 "Mock NGC"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import cosmo   # FlatLambdaCDM(H0=100, Om0=0.313, Tcmb0=2.725)

# ── Constants ─────────────────────────────────────────────────────────────────
MAG_CUT  = -20.7
ZMIN     = 0.10
ZMAX     = 0.30
ZBINS_NZ = np.linspace(ZMIN, ZMAX, 41)   # 40 bins for n(z)
ZBINS_COL = [0.10, 0.16, 0.22, 0.30]     # 3 colour-histogram bins
COLOUR_BINS = np.linspace(0.0, 1.1, 45)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cat',    required=True,
                   help='Input catalogue (BGS data or mock).')
    p.add_argument('--cat2',   default=None,
                   help='Optional second catalogue for overlay.')
    p.add_argument('--zcol',   default='Zobs',
                   help='Redshift column to use (default: Zobs).')
    p.add_argument('--zcol2',  default=None,
                   help='Redshift column for second catalogue (default: same as --zcol).')
    p.add_argument('--label',  default='Catalogue 1')
    p.add_argument('--label2', default='Catalogue 2')
    p.add_argument('--weight_col', default=None,
                   help='Completeness weight column (e.g. WEIGHT_COMP). '
                        'If absent, unit weights are used.')
    p.add_argument('--absmag_cut', type=float, default=None,
                   help='Alternative ABSMAG_RP1 cut to use instead of MAG_CUT '
                        '(e.g. -20.45 to match n(z) of the no-kcorr selection).')
    p.add_argument('--outdir', default='.',
                   help='Output directory for figure.')
    p.add_argument('--suffix', default='',
                   help='Extra suffix for output filenames.')
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def distmod(z):
    """Luminosity distance modulus using pipeline cosmology."""
    return cosmo.distmod(z).value


def load_cat(path, zcol, weight_col):
    """Load catalogue and return relevant columns as numpy arrays."""
    print(f'Reading {path}...')
    cols = [zcol, 'rmag', 'ABSMAG_RP1', 'REST_GMR_0P1']
    if weight_col is not None:
        cols.append(weight_col)
    # Read only needed columns
    t = Table.read(path, format='fits')
    z       = np.array(t[zcol],          dtype=float)
    rmag    = np.array(t['rmag'],         dtype=float)
    absmag  = np.array(t['ABSMAG_RP1'],  dtype=float)
    colour  = np.array(t['REST_GMR_0P1'], dtype=float)
    if weight_col is not None and weight_col in t.colnames:
        wt = np.array(t[weight_col], dtype=float)
    else:
        wt = np.ones(len(z))
    return z, rmag, absmag, colour, wt


def abr_nodkcorr(z, rmag):
    """Absolute magnitude without K-correction (distance modulus only)."""
    return rmag - distmod(z)


def nz_density(z, wt, zbins, area_sqdeg):
    """
    Weighted comoving number density n(z) in h^3 Mpc^-3.
    area_sqdeg: survey area in square degrees (used for volume calculation).
    """
    area_sr   = area_sqdeg * (np.pi / 180.0)**2
    n         = np.zeros(len(zbins) - 1)
    for i in range(len(zbins) - 1):
        zlo, zhi = zbins[i], zbins[i+1]
        sel      = (z >= zlo) & (z < zhi)
        Vshell   = (cosmo.comoving_volume(zhi).value -
                    cosmo.comoving_volume(zlo).value) * (area_sr / (4*np.pi))
        n[i]     = np.sum(wt[sel]) / Vshell if Vshell > 0 else 0.0
    return n


def find_matched_absmag_cut(z, rmag, absmag, wt, zsel, target_count):
    """
    Find the ABSMAG_RP1 threshold that gives the same weighted total count
    as the no-kcorr selection over the full redshift range.
    Uses a simple percentile of the ABSMAG_RP1 distribution.
    """
    # Among galaxies passing the no-kcorr cut, what fraction of
    # the full ABSMAG_RP1 distribution do we need to keep?
    # Equivalently: find the ABSMAG_RP1 value at the (target_count/N_all) percentile.
    abr_nk   = abr_nodkcorr(z, rmag)
    in_zrange = zsel
    all_absmag = absmag[in_zrange]
    all_wt     = wt[in_zrange]
    # Sort by ABSMAG_RP1 and find cumulative weighted count
    idx    = np.argsort(all_absmag)
    cumwt  = np.cumsum(all_wt[idx])
    # Find the cut where cumulative count equals target_count
    i_cut  = np.searchsorted(cumwt, target_count)
    i_cut  = min(i_cut, len(idx) - 1)
    cut    = all_absmag[idx[i_cut]]
    return cut


def plot_one_cat(z, rmag, absmag, colour, wt, zbins_nz, zbins_col,
                 area_sqdeg, axes, color_A, color_B, color_AB, label,
                 absmag_cut=None):
    """
    Compute and plot n(z) and colour histograms for one catalogue.
    axes: (ax_nz, ax_frac, ax_col1, ax_col2, ax_col3)
    absmag_cut: if given, use this instead of MAG_CUT for the ABSMAG_RP1 selection.
    """
    ax_nz, ax_frac, ax_col1, ax_col2, ax_col3 = axes
    col_axes = [ax_col1, ax_col2, ax_col3]

    mag_cut_B = absmag_cut if absmag_cut is not None else MAG_CUT

    # Base redshift cut
    zsel = (z >= ZMIN) & (z < ZMAX)

    # Selection A: no-kcorr
    abr_nk = abr_nodkcorr(z, rmag)
    selA   = zsel & (abr_nk  < MAG_CUT)

    # Selection B: ABSMAG_RP1 (pipeline K-corrected, possibly at a different cut)
    selB   = zsel & (absmag  < mag_cut_B)

    # Overlap
    selAB  = selA & selB

    zc   = 0.5 * (zbins_nz[:-1] + zbins_nz[1:])

    nA   = nz_density(z[selA],  wt[selA],  zbins_nz, area_sqdeg)
    nB   = nz_density(z[selB],  wt[selB],  zbins_nz, area_sqdeg)
    nAB  = nz_density(z[selAB], wt[selAB], zbins_nz, area_sqdeg)

    ls   = '-'   # solid for first cat; caller may vary

    ax_nz.step(zc, nA,  where='mid', color=color_A,  lw=1.5, ls=ls,
               label=f'{label} — no-kcorr < {MAG_CUT}')
    ax_nz.step(zc, nB,  where='mid', color=color_B,  lw=1.5, ls=ls,
               label=f'{label} — ABSMAG_RP1 < {mag_cut_B:.2f}')
    ax_nz.step(zc, nAB, where='mid', color=color_AB, lw=1.0, ls='--',
               label=f'{label} — both')

    # Fraction of A in B and vice versa
    nA_all  = nz_density(z[selA], wt[selA],  zbins_nz, area_sqdeg)
    nB_all  = nz_density(z[selB], wt[selB],  zbins_nz, area_sqdeg)
    nAB_all = nz_density(z[selAB], wt[selAB], zbins_nz, area_sqdeg)

    fAinB = np.where(nA_all > 0, nAB_all / nA_all, np.nan)
    fBinA = np.where(nB_all > 0, nAB_all / nB_all, np.nan)

    ax_frac.step(zc, fAinB, where='mid', color=color_A, lw=1.5, ls=ls,
                 label=f'{label}: frac(no-kcorr in ABSMAG_RP1<{mag_cut_B:.2f})')
    ax_frac.step(zc, fBinA, where='mid', color=color_B, lw=1.5, ls='--',
                 label=f'{label}: frac(ABSMAG_RP1<{mag_cut_B:.2f} in no-kcorr)')

    # Colour histograms in three z bins
    for ax, zlo, zhi in zip(col_axes, zbins_col[:-1], zbins_col[1:]):
        czsel = (z >= zlo) & (z < zhi)
        for sel, col, lbl in [
            (selA & czsel, color_A, f'{label} no-kcorr<{MAG_CUT}'),
            (selB & czsel, color_B, f'{label} ABSMAG_RP1<{mag_cut_B:.2f}'),
        ]:
            c = colour[sel]
            w = wt[sel]
            h, edges = np.histogram(c, bins=COLOUR_BINS, weights=w, density=True)
            cc = 0.5 * (edges[:-1] + edges[1:])
            ax.step(cc, h, where='mid', color=col, lw=1.2, ls=ls, label=lbl)

    return selA, selB, selAB


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    zcol2 = args.zcol2 if args.zcol2 else args.zcol

    # Sky area: approximate from catalogue RA/DEC range.
    # For a proper n(z) this should come from the random catalogue FSKY.
    # Here we use a fixed approximation — the plot is relative so absolute
    # normalisation matters less than the ratio between A and B.
    # Override AREA_SQDEG below if you know the exact value.
    AREA_SQDEG_NGC = 3827.5   # DR2 North
    AREA_SQDEG_SGC = 8527.6   # DR2 South
    # Default: use NGC area; pass --suffix SGC to remind yourself to change it
    area = AREA_SQDEG_NGC

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 13))
    gs  = fig.add_gridspec(
        4, 3,
        height_ratios=[2, 1.2, 1.5, 1.5],
        hspace=0.45, wspace=0.35,
    )
    ax_nz   = fig.add_subplot(gs[0, :])
    ax_frac = fig.add_subplot(gs[1, :])
    ax_col1 = fig.add_subplot(gs[2, 0])
    ax_col2 = fig.add_subplot(gs[2, 1])
    ax_col3 = fig.add_subplot(gs[2, 2])
    axes = (ax_nz, ax_frac, ax_col1, ax_col2, ax_col3)

    # Colour scheme: no-kcorr = blue tones, ABSMAG_RP1 = red tones,
    # overlap = grey; cat2 uses dashed lines.
    color_A  = '#2166ac'   # blue — no-kcorr selection
    color_B  = '#d6604d'   # red  — ABSMAG_RP1 selection
    color_AB = '#777777'   # grey — overlap

    # ── Catalogue 1 ───────────────────────────────────────────────────────────
    z, rmag, absmag, colour, wt = load_cat(args.cat, args.zcol, args.weight_col)

    zsel = (z >= ZMIN) & (z < ZMAX)

    # Determine ABSMAG_RP1 cut to use
    if args.absmag_cut is not None:
        absmag_cut = args.absmag_cut
        print(f'Using user-supplied ABSMAG_RP1 cut: {absmag_cut:.3f}')
    else:
        # Auto-match: find ABSMAG_RP1 cut that gives the same weighted count
        # as the no-kcorr selection
        abr_nk = abr_nodkcorr(z, rmag)
        target = np.sum(wt[zsel & (abr_nk < MAG_CUT)])
        absmag_cut = find_matched_absmag_cut(z, rmag, absmag, wt, zsel, target)
        print(f'Auto-matched ABSMAG_RP1 cut: {absmag_cut:.3f}  '
              f'(target weighted N = {target:.0f})')

    selA1, selB1, selAB1 = plot_one_cat(
        z, rmag, absmag, colour, wt,
        ZBINS_NZ, ZBINS_COL, area, axes,
        color_A, color_B, color_AB, args.label,
        absmag_cut=absmag_cut,
    )

    # ── Catalogue 2 (optional overlay) ────────────────────────────────────────
    if args.cat2:
        z2, rmag2, absmag2, colour2, wt2 = load_cat(
            args.cat2, zcol2, args.weight_col
        )
        color_A2  = '#74add1'   # lighter blue
        color_B2  = '#f4a582'   # lighter red
        color_AB2 = '#bbbbbb'
        selA2, selB2, selAB2 = plot_one_cat(
            z2, rmag2, absmag2, colour2, wt2,
            ZBINS_NZ, ZBINS_COL, area, axes,
            color_A2, color_B2, color_AB2, args.label2,
        )
        # Make cat2 lines dashed — unfortunately plot_one_cat uses ls='-'
        # so we patch afterwards
        for ax in [ax_nz, ax_frac, ax_col1, ax_col2, ax_col3]:
            lines = ax.get_lines()
            n_first = sum(1 for l in lines
                          if l.get_color() in (color_A, color_B, color_AB))
            for l in lines[n_first:]:
                l.set_linestyle('--')

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax_nz.set_xlabel('Redshift z', fontsize=11)
    ax_nz.set_ylabel(r'$n(z)\ [h^3\,\mathrm{Mpc}^{-3}]$', fontsize=11)
    ax_nz.set_title(
        rf'BGS LSS sample comparison: $M_r^\mathrm{{nodkcorr}}$ vs '
        rf'$^{{0.1}}M_r$, cut at {MAG_CUT}, '
        rf'${ZMIN} < z < {ZMAX}$',
        fontsize=11,
    )
    ax_nz.set_xlim(ZMIN, ZMAX)
    ax_nz.set_yscale('log')
    ax_nz.legend(fontsize=8, ncol=2)
    ax_nz.grid(True, alpha=0.3)

    ax_frac.set_xlabel('Redshift z', fontsize=11)
    ax_frac.set_ylabel('Overlap fraction', fontsize=11)
    ax_frac.set_title('Fraction of each selection also passing the other cut', fontsize=10)
    ax_frac.set_xlim(ZMIN, ZMAX)
    ax_frac.set_ylim(0.5, 1.05)
    ax_frac.axhline(1.0, color='k', lw=0.7, ls=':')
    ax_frac.legend(fontsize=8)
    ax_frac.grid(True, alpha=0.3)

    for ax, zlo, zhi in zip(
        [ax_col1, ax_col2, ax_col3],
        ZBINS_COL[:-1], ZBINS_COL[1:]
    ):
        ax.set_xlabel(r'${}^{0.1}(g-r)$', fontsize=10)
        ax.set_ylabel('Prob. density', fontsize=10)
        ax.set_title(rf'${zlo:.2f} < z < {zhi:.2f}$', fontsize=10)
        ax.set_xlim(0.0, 1.1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Statistics summary ────────────────────────────────────────────────────
    nA1 = np.sum(selA1); nB1 = np.sum(selB1); nAB1 = np.sum(selAB1)
    print(f'\n{args.label}  (z in [{ZMIN},{ZMAX}]):')
    print(f'  no-kcorr < {MAG_CUT}:              {nA1:,}')
    print(f'  ABSMAG_RP1 < {absmag_cut:.3f}:         {nB1:,}')
    print(f'  Both:                              {nAB1:,}')
    print(f'  frac(no-kcorr in ABSMAG_RP1): {nAB1/nA1:.3f}')
    print(f'  frac(ABSMAG_RP1 in no-kcorr): {nAB1/nB1:.3f}')

    # Colour statistics per z bin
    print(f'\n  Median 0.1(g-r) by redshift bin:')
    for zlo, zhi in zip(ZBINS_COL[:-1], ZBINS_COL[1:]):
        czsel = (z >= zlo) & (z < zhi)
        mA = np.median(colour[selA1 & czsel])
        mB = np.median(colour[selB1 & czsel])
        print(f'    z=[{zlo:.2f},{zhi:.2f}]  no-kcorr: {mA:.3f}  ABSMAG_RP1: {mB:.3f}')

    # ── Save ──────────────────────────────────────────────────────────────────
    outdir  = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    suffix  = ('_' + args.suffix) if args.suffix else ''
    outfile = outdir / f'compare_lss_selections{suffix}.pdf'
    fig.savefig(outfile, bbox_inches='tight', dpi=150)
    print(f'\nFigure saved to {outfile}')

    # Also save as png for easy email attachment
    png = outfile.with_suffix('.png')
    fig.savefig(png, bbox_inches='tight', dpi=150)
    print(f'PNG saved to {png}')


if __name__ == '__main__':
    main()
