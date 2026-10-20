"""
inspect_mocks.py — Diagnostic plots comparing a mock galaxy catalogue against
the BGS data and the corresponding random catalogues.

Plot groups are selected with flags; nothing is produced unless at least one
flag is given.  All figures are saved as PDFs to a dated output directory and
are not displayed interactively unless --show is also passed.

Plot groups
-----------
  --sky         RA/DEC distributions, sky maps, cone/wedge plots
  --redshift    N(z), n(z) by magnitude cut, mock Z vs matched Zr
  --photometric Absolute magnitude, colour, and apparent magnitude distributions
  --halo        Vpeak vs ABSMAG_R, Vpeak vs distance, Mpeak_Scale vs colour
  --jackknife   dN/dz with jackknife errors (slower; run separately)

Usage examples
--------------
# Quick photometric check after a new mock:
python inspect_mocks.py --reg N --obs_seed 0 --photometric

# Full diagnostic suite:
python inspect_mocks.py --reg N --obs_seed 0 \\
    --sky --redshift --photometric --halo

# Paper-quality dN/dz with jackknife errors:
python inspect_mocks.py --reg N --obs_seed 0 --jackknife

# Display plots interactively as well as saving:
python inspect_mocks.py --reg N --obs_seed 0 --redshift --show
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive by default; overridden by --show
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# Pipeline modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, DEFAULT_OBS_SEED, DEFAULT_SIG_LGV_REF, DEFAULT_SIG_LGV_SLOPE, DEFAULT_F_COLOUR, DEFAULT_ALPHA
from pipeline_utils import make_paths, setup_logging


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--reg', required=True, choices=['N', 'S'],
                   help='Photometric region to inspect.')
    p.add_argument('--obs_seed', type=int, default=DEFAULT_OBS_SEED,
                   help=f'Observer position seed (default: {DEFAULT_OBS_SEED})')
    p.add_argument('--sig_lgv_ref', type=float, default=DEFAULT_SIG_LGV_REF,
                   help=f'Vpeak scatter at pivot Vpeak (default: {DEFAULT_SIG_LGV_REF})')
    p.add_argument('--sig_lgv_slope', type=float, default=DEFAULT_SIG_LGV_SLOPE,
                   help=f'Gradient of scatter with log10(Vpeak) (default: {DEFAULT_SIG_LGV_SLOPE})')
    p.add_argument('--f_colour', type=float, default=DEFAULT_F_COLOUR,
                   help=f'f_colour used when making the checkpoint (default: {DEFAULT_F_COLOUR})')
    p.add_argument('--alpha', type=float, default=DEFAULT_ALPHA,
                   help=f'alpha used when making the mock (default: {DEFAULT_ALPHA})')
    p.add_argument('--absmag_lim', type=float, default=-8.0,
                   help='Faint absolute magnitude limit applied to all catalogues '
                        'before plotting (default: -8.0, i.e. keep everything)')
    # Plot group flags
    p.add_argument('--sky',         action='store_true', help='Sky/angular plots')
    p.add_argument('--redshift',    action='store_true', help='Redshift distribution plots')
    p.add_argument('--photometric', action='store_true', help='Magnitude and colour plots')
    p.add_argument('--halo',        action='store_true', help='Halo property plots')
    p.add_argument('--jackknife',   action='store_true',
                   help='dN/dz with jackknife errors (slower)')
    p.add_argument('--show', action='store_true',
                   help='Display plots interactively as well as saving to disk')
    p.add_argument('--outdir', type=str, default=None,
                   help='Output directory for figures. Defaults to '
                        'DATA_DIR/diagnostics/seed{obs_seed}_sig_lgv{ref}_{slope}_{date}/')
    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def savefig(fig, outdir, name, show, log):
    path = outdir / f"{name}.pdf"
    fig.savefig(str(path), dpi=150, bbox_inches='tight')
    log.info(f"Saved: {path.name}")
    if show:
        plt.show()
    plt.close(fig)


def load_catalogue(path, label, log):
    from astropy.table import Table
    if not path.exists():
        log.error(f"{label} not found: {path}")
        sys.exit(1)
    log.info(f"Reading {label}: {path.name}")
    return Table.read(str(path))


def apply_absmag_cut(tables_dict, cut, log):
    """Apply a faint absolute magnitude cut to all catalogues that have ABSMAG_RP1."""
    out = {}
    for name, t in tables_dict.items():
        if 'ABSMAG_RP1' in t.colnames:
            mask = t['ABSMAG_RP1'] < cut
            out[name] = t[mask]
            log.info(f"  {name}: {len(t):,} → {len(out[name]):,} after ABSMAG_RP1 < {cut}")
        else:
            out[name] = t
    return out


# =============================================================================
# Plot group: sky / angular
# =============================================================================

def plot_sky(dat, mock, ran, ran_inp, reg, outdir, show, log):
    import catalogue_analysis as ca
    log.info("=== Sky / angular plots ===")

    # --- RA distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 360, 51)
    ax.hist(dat['RA'],     bins=bins, density=True, histtype='step', lw=1.5,
            weights=dat['WEIGHT'], label=f'Data (N={len(dat):,})')
    ax.hist(mock['RA'],    bins=bins, density=True, histtype='step', lw=1.5,
            ls='dashed',  label=f'Mock (N={len(mock):,})')
    ax.hist(ran['RA'],     bins=bins, density=True, histtype='step', lw=1.5,
            ls='dotted',  label=f'Formock randoms')
    ax.hist(ran_inp['RA'], bins=bins, density=True, histtype='step', lw=1.5,
            label=f'Fordata randoms')
    ax.set_xlabel('RA [deg]');  ax.set_ylabel('Normalised density')
    ax.set_xlim(0, 360);        ax.legend();  ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, outdir, 'ra_distribution', show, log)

    # --- DEC distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-20, 90, 53)
    ax.hist(dat['DEC'],     bins=bins, density=True, histtype='step', lw=1.5,
            weights=dat['WEIGHT'], label='Data')
    ax.hist(mock['DEC'],    bins=bins, density=True, histtype='step', lw=1.5,
            ls='dashed',  label='Mock')
    ax.hist(ran['DEC'],     bins=bins, density=True, histtype='step', lw=1.5,
            ls='dotted',  label='Formock randoms')
    ax.hist(ran_inp['DEC'], bins=bins, density=True, histtype='step', lw=1.5,
            label='Fordata randoms')
    ax.set_xlabel('DEC [deg]'); ax.set_ylabel('Normalised density')
    ax.set_xlim(-20, 85);       ax.legend();  ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig(fig, outdir, 'dec_distribution', show, log)

    # --- Sky maps ---
    for cat, name in [(dat, 'data'), (mock, 'mock'), (ran, 'formock_randoms')]:
        try:
            fig = plt.figure(figsize=(10, 5))
            ca.sky_plot(cat, reg)
            savefig(fig, outdir, f'skymap_{name}', show, log)
        except Exception as e:
            log.warning(f"sky_plot failed for {name}: {e}")

    # --- Cone / wedge plot ---
    log.info("Making cone plot")
    fig, axes = plt.subplots(2, 1, figsize=(4.5, 6.0), sharex=True, sharey=True)
    offset = 1.5 * np.pi / 2.0
    ra_min, ra_max, z_max = 180.0, 275.0, 0.50

    Sel = ca.selection(reg)
    if reg == 'S':
        dec_lo, dec_hi = -5.0, 0.0
    else:
        dec_lo, dec_hi = 30.0, 35.0

    def _cone_mask(cat, z_col='Z'):
        return (
            (cat[z_col] > Sel['zmin']) & (cat[z_col] < z_max) &
            (cat['rmag'] < Sel['faint']) &
            (cat['DEC'] > dec_lo) & (cat['DEC'] < dec_hi) &
            (cat['RA'] > ra_min) & (cat['RA'] < ra_max)
        )

    ra_arc = np.linspace(ra_min, ra_max, 200)
    fx = np.concatenate(([0], z_max * np.sin(np.radians(ra_arc) + offset), [0]))
    fy = np.concatenate(([0], z_max * np.cos(np.radians(ra_arc) + offset), [0]))

    for ax, cat, col, label, z_col in [
        (axes[0], dat,  'black', 'BGS Data', 'Z'),
        (axes[1], mock, 'blue',  'Mock',      'Zobs'),
    ]:
        m = _cone_mask(cat, z_col)
        x = cat[z_col][m] * np.sin(np.radians(cat['RA'][m]) + offset)
        y = cat[z_col][m] * np.cos(np.radians(cat['RA'][m]) + offset)
        ax.scatter(x, y, color=col, s=0.1, marker='.', linewidths=0)
        ax.plot(fx, fy, color='0.6', lw=1)
        ax.text(0.02, 0.92, label, transform=ax.transAxes, va='top')
        ax.set_aspect('equal');  ax.axis('off')

    fig.subplots_adjust(hspace=0.0)
    savefig(fig, outdir, 'cone_plot', show, log)


# =============================================================================
# Plot group: redshift
# =============================================================================

def plot_redshift(dat, mock, ran, ran_inp, reg, outdir, show, log):
    import catalogue_analysis as ca
    log.info("=== Redshift distribution plots ===")

    Sel = ca.selection(reg)
    fsky_formock = ran.meta.get('FSKY', Sel['area'])
    fsky_fordata = Sel['area']
    nmult_formock = ran.meta.get('NMULT', 1.0)
    nmult_fordata = ran_inp.meta.get('NMULT', 1.0)

    zbins = np.linspace(0.0, 0.6, 61)
    zcen  = 0.5 * (zbins[:-1] + zbins[1:])
    dz    = zbins[1] - zbins[0]

    # --- Basic N(z) ratio ---
    fig, ax = plt.subplots(figsize=(8, 5))
    z_bins_fine = np.linspace(0.01, 0.5, 800)
    z_cen_fine  = 0.5 * (z_bins_fine[:-1] + z_bins_fine[1:])

    h_dat,  _ = np.histogram(dat['Z'],       bins=z_bins_fine, weights=dat['WEIGHT'])
    h_ran,  _ = np.histogram(ran['Z'],       bins=z_bins_fine)
    h_mock, _ = np.histogram(mock['Z'],      bins=z_bins_fine)
    h_ri,   _ = np.histogram(ran_inp['Z'],   bins=z_bins_fine)

    h_dat  = h_dat.astype(float)  / fsky_fordata
    h_ran  = h_ran.astype(float)  / (fsky_formock * nmult_formock)
    h_mock = h_mock.astype(float) / fsky_formock
    h_ri   = h_ri.astype(float)   / (fsky_fordata * nmult_fordata)

    def _safe_ratio(a, b):
        return np.where(b > 0, a / b, np.nan)

    ax.plot(z_cen_fine, _safe_ratio(h_dat,  h_ri),  lw=1, label='Data / fordata ran')
    ax.plot(z_cen_fine, _safe_ratio(h_ran,  h_ri),  lw=1, label='Formock ran / fordata ran')
    ax.plot(z_cen_fine, _safe_ratio(h_mock, h_ri),  lw=1, label='Mock / fordata ran')
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('z');  ax.set_ylabel('N / N(fordata randoms)')
    ax.legend();  ax.grid(alpha=0.2)
    plt.tight_layout()
    savefig(fig, outdir, 'nz_ratio', show, log)

    # --- n(z) by magnitude cut ---
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    dc  = ca.cosmo.comoving_distance(zbins).value
    vol = (4.0 * np.pi * fsky_formock / 3.0) * (dc[1:]**3 - dc[:-1]**3)
    vol_dat = (4.0 * np.pi * fsky_fordata / 3.0) * (dc[1:]**3 - dc[:-1]**3)
    magcuts = [-20.0, -20.5, -21.0, -21.5, -22.0]
    colors  = ['C0', 'C1', 'C2', 'C3', 'C4']
    eps     = 1e-25

    for Mcut, c in zip(magcuts, colors):
        h, _ = np.histogram(dat['Z'][dat['ABSMAG_RP1'] < Mcut], bins=zbins,
                            weights=dat['WEIGHT'][dat['ABSMAG_RP1'] < Mcut])
        ax.scatter(zcen, np.log10(h / vol_dat + eps), color=c, s=10, marker='o',
                   label=rf'$M_r < {Mcut:.1f}$')
        h, _ = np.histogram(mock['Z'][mock['ABSMAG_RP1'] < Mcut], bins=zbins)
        ax.plot(zcen, np.log10(h * nmult_formock / len(ran) / vol + eps),
                color=c, lw=2, ls='--')
        h, _ = np.histogram(ran['Z'][ran['ABSMAG_RP1'] < Mcut], bins=zbins)
        ax.plot(zcen, np.log10(h / (vol * nmult_formock) + eps), color=c, lw=2)

    handles_extra = [
        mlines.Line2D([], [], marker='o', color='none', markerfacecolor='k',
                      markersize=5, label='BGS data'),
        mlines.Line2D([], [], color='k', lw=2, label='Randoms'),
        mlines.Line2D([], [], color='k', lw=2, ls='--', label='Mock'),
    ]
    leg1 = ax.legend(frameon=False, loc='upper right')
    ax.legend(handles=handles_extra, frameon=False, bbox_to_anchor=(1.0, 0.55),
              loc='center right')
    ax.add_artist(leg1)
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel(r'$\log_{10}\,\bar{n}(z)\ [h^3\,\mathrm{Mpc}^{-3}]$')
    ax.set_ylim([-4.5, -2]);  ax.set_xlim([0.05, 0.55])
    savefig(fig, outdir, 'nz_by_magcut', show, log)

    # --- Mock Z vs matched Zr ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mock['Z'], mock['Zr'], s=1, alpha=0.3, rasterized=True)
    ax.set_xlabel('Z  (subhalo cosmological redshift)')
    ax.set_ylabel('Zr  (matched random galaxy redshift)')
    ax.set_title('SHAM redshift matching diagnostic')
    plt.tight_layout()
    savefig(fig, outdir, 'z_vs_zr', show, log)


# =============================================================================
# Plot group: jackknife dN/dz
# =============================================================================

def plot_jackknife_dndz(dat, mock, ran, reg, outdir, show, log):
    import catalogue_analysis as ca
    log.info("=== Jackknife dN/dz ===")

    njack_map = {'N': (3, 3, 0), 'S': (4, 5, 0)}
    ndiv_ra, ndiv_dec, offset_jk = njack_map[reg]

    log.info("Setting up jackknife regions (this may take a minute)")
    dat['ijack'] = -999
    ran['ijack'] = -999
    njack, limits = ca.solve_jackknife_nonsq(ran, ndiv_ra=ndiv_ra,
                                             ndiv_dec=ndiv_dec, offset=275)
    ca.set_jackknife(ran, np.ones(len(ran), dtype=bool), limits, offset_jk, njack)
    ca.set_jackknife(dat, np.ones(len(dat), dtype=bool), limits, offset_jk, njack)

    zbins = np.linspace(0.0, 0.6, 61)
    zcen  = 0.5 * (zbins[:-1] + zbins[1:])
    dz    = zbins[1] - zbins[0]
    weights = np.asarray(dat['WEIGHT'])

    h_dat, _ = np.histogram(dat['Z'], bins=zbins, weights=weights)
    h_dat = h_dat / (h_dat.sum() * dz)

    jk_hists = []
    for ijack in range(njack):
        mask = (dat['ijack'] - offset_jk != ijack)
        h, _ = np.histogram(dat['Z'][mask], bins=zbins, weights=weights[mask])
        jk_hists.append(h / (h.sum() * dz))
    jk_hists = np.asarray(jk_hists)
    jk_err   = np.std(jk_hists, axis=0) * np.sqrt(njack - 1)

    h_ran, _ = np.histogram(ran['Z'], bins=zbins)
    h_ran = h_ran / (h_ran.sum() * dz)

    h_mock, _ = np.histogram(mock['Zobs'], bins=zbins)
    h_mock = h_mock / (h_mock.sum() * dz)

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.fill_between(zcen, h_dat - jk_err, h_dat + jk_err,
                    color='lightgrey', alpha=0.8, label='BGS jackknife')
    ax.plot(zcen, h_dat,  color='black', lw=2, label='BGS data')
    ax.plot(zcen, h_ran,  color='red',   lw=2, label='Randoms')
    ax.plot(zcen, h_mock, color='blue',  lw=2, ls='--', label='Mock')
    ax.set_xlim(0, 0.6);  ax.set_ylim(bottom=0)
    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel(r'$(1/N)\,dN/dz$')
    ax.legend(frameon=False)
    savefig(fig, outdir, 'dndz_jackknife', show, log)


# =============================================================================
# Plot group: photometric
# =============================================================================

def plot_photometric(dat, mock, ran, reg, outdir, show, log):
    log.info("=== Photometric plots ===")

    rng  = np.random.default_rng(42)
    nran = min(len(ran), 200_000)
    idx  = rng.choice(len(ran), size=nran, replace=False)
    z_rand   = ran['Z'][idx]
    m_rand   = ran['ABSMAG_RP1'][idx]
    col_rand = ran['ABSMAG_GP1'][idx] - ran['ABSMAG_RP1'][idx]

    z_bins_plot = [(0.0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 0.60)]
    mag_bins    = np.linspace(-25, -14, 60)
    col_bins    = np.linspace(0.0,  1.5, 40)
    colors      = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # --- Absolute magnitude distributions by redshift slice ---
    fig, ax = plt.subplots(figsize=(5, 4))
    for (zmin, zmax), c in zip(z_bins_plot, colors):
        s_dat  = (dat['Z']  >= zmin) & (dat['Z']  < zmax)
        s_rand = (z_rand    >= zmin) & (z_rand    < zmax)
        s_mock = (mock['Z'] >= zmin) & (mock['Z'] < zmax)
        ax.hist(dat['ABSMAG_RP1'][s_dat], bins=mag_bins, density=True,
                histtype='stepfilled', color='lightgrey', alpha=0.4)
        ax.hist(m_rand[s_rand], bins=mag_bins, density=True,
                histtype='step', color='red', lw=2, alpha=0.8)
        ax.hist(mock['ABSMAG_RP1'][s_mock], bins=mag_bins, density=True,
                histtype='step', color='blue', ls='--', lw=2,
                label=rf'${zmin:.2f}\leq z<{zmax:.2f}$')
    ax.set_xlabel(r'$M_r - 5\log h$');  ax.set_ylabel('Probability density')
    ax.set_xlim(-24, -14)
    handles = [mpatches.Patch(facecolor='lightgrey', label='BGS data'),
               mlines.Line2D([], [], color='red', lw=2, label='Randoms'),
               mlines.Line2D([], [], color='blue', lw=2, ls='--', label='Mock')]
    ax.legend(handles=handles, frameon=False, loc='upper right')
    plt.tight_layout()
    savefig(fig, outdir, 'absmag_by_zslice', show, log)

    # --- Colour distributions by redshift slice ---
    fig, axes = plt.subplots(len(z_bins_plot), 1, figsize=(4.5, 9), sharex=True)
    fig.subplots_adjust(hspace=0.0)
    ylims = [2.5, 3.5, 4.5, 8.5]
    for i, ((zmin, zmax), ylim) in enumerate(zip(z_bins_plot, ylims)):
        ax = axes[i]
        s_dat  = (dat['Z']  >= zmin) & (dat['Z']  < zmax)
        s_rand = (z_rand    >= zmin) & (z_rand    < zmax)
        s_mock = (mock['Z'] >= zmin) & (mock['Z'] < zmax)
        ax.hist(dat['ABSMAG_GP1'][s_dat] - dat['ABSMAG_RP1'][s_dat],
                bins=col_bins, density=True, histtype='stepfilled',
                color='lightgrey', alpha=0.8,
                label='BGS data' if i == 0 else None)
        ax.hist(col_rand[s_rand], bins=col_bins, density=True,
                histtype='step', color='red', lw=2,
                label='Randoms' if i == 0 else None)
        ax.hist(mock['ABSMAG_GP1'][s_mock] - mock['ABSMAG_RP1'][s_mock],
                bins=col_bins, density=True, histtype='step',
                color='blue', ls='--', lw=2,
                label='Mock' if i == 0 else None)
        ax.text(0.03, 0.88, rf'${zmin:.1f}\leq z<{zmax:.1f}$',
                transform=ax.transAxes, ha='left', va='top')
        ax.set_ylim(0, ylim)
    axes[len(z_bins_plot)//2].set_ylabel('Probability density')
    axes[-1].set_xlabel(r'${}^{0.1}(g-r)$')
    axes[0].set_xlim(0, 1.1)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, frameon=False, loc='lower left')
    savefig(fig, outdir, 'colour_by_zslice', show, log)

    # --- Absolute magnitude-redshift scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax in axes:
        ax.scatter(z_rand, m_rand, s=1, color='lightgray', marker=',',
                   rasterized=True, zorder=1)
    axes[0].scatter(dat['Z'],  dat['ABSMAG_RP1'],  s=2, color='blue',
                    marker='.', rasterized=True, zorder=2)
    axes[1].scatter(mock['Z'], mock['ABSMAG_RP1'], s=2, color='green',
                    marker='.', rasterized=True, zorder=2)
    axes[0].set_title('Data');   axes[1].set_title('Mock')
    for ax in axes:
        ax.set_xlabel('z');  ax.set_xlim([0, 0.6]);  ax.set_ylim([-25, -10])
    axes[0].set_ylabel(r'$M_r - 5\log h$')
    plt.tight_layout()
    savefig(fig, outdir, 'absmag_redshift_scatter', show, log)

    # --- Apparent magnitude-redshift scatter ---
    r_rand = ran['rmag'][idx]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax in axes:
        ax.scatter(z_rand, r_rand, s=1, color='lightgray', marker=',',
                   rasterized=True, zorder=1)
    axes[0].scatter(dat['Z'],  dat['rmag'],  s=2, color='blue',
                    marker='.', rasterized=True, zorder=2)
    axes[1].scatter(mock['Z'], mock['rmag'], s=2, color='green',
                    marker='.', rasterized=True, zorder=2)
    axes[0].set_title('Data');   axes[1].set_title('Mock')
    for ax in axes:
        ax.set_xlabel('z');  ax.set_xlim([0, 0.6]);  ax.set_ylim([12, 19.8])
    axes[0].set_ylabel(r'$r$')
    plt.tight_layout()
    savefig(fig, outdir, 'appmag_redshift_scatter', show, log)

    # --- Colour-redshift scatter ---
    col_dat  = dat['ABSMAG_GP1']  - dat['ABSMAG_RP1']
    col_mock = mock['ABSMAG_GP1'] - mock['ABSMAG_RP1']
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax in axes:
        ax.scatter(z_rand, col_rand, s=1, color='lightgray', marker=',',
                   rasterized=True, zorder=1)
    axes[0].scatter(dat['Z'],  col_dat,  s=2, color='blue',
                    marker='.', rasterized=True, zorder=2)
    axes[1].scatter(mock['Z'], col_mock, s=2, color='green',
                    marker='.', rasterized=True, zorder=2)
    axes[0].set_title('Data');  axes[1].set_title('Mock')
    for ax in axes:
        ax.set_xlabel('z');  ax.set_xlim([0, 0.6]);  ax.set_ylim([0.0, 1.5])
    axes[0].set_ylabel(r'restframe $G-R$')
    plt.tight_layout()
    savefig(fig, outdir, 'colour_redshift_scatter', show, log)


# =============================================================================
# Plot group: halo properties
# =============================================================================

def plot_halo(mock, outdir, show, log):
    log.info("=== Halo property plots ===")

    # --- Vpeak vs ABSMAG_R hexbin ---
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.hexbin(np.log10(mock['Vpeak']), mock['ABSMAG_R'],
                   C=mock['Z'], gridsize=200, reduce_C_function=np.mean,
                   cmap='viridis')
    ax.invert_yaxis()
    ax.set_xlabel(r'$\log_{10} V_\mathrm{peak}$')
    ax.set_ylabel(r"$M'_r - 5\log h$")
    ax.set_xlim([1.9, 3.4]);  ax.set_ylim([-15, -24])
    plt.colorbar(sc, ax=ax, label=r'$Z_u$')
    plt.tight_layout()
    savefig(fig, outdir, 'vpeak_vs_absmag', show, log)

    # --- Vpeak vs distance ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(mock['dist'], mock['Vpeak'], s=0.2, color='green',
               marker='.', rasterized=True, zorder=1)
    # Overplot the hardcoded distance-dependent cut from MockCode.pre_sort_subhalos
    dist_arr = np.linspace(0, 1600, 200)
    vpeak_cut = ((6.5e-7 * dist_arr - 9.02e-4) * dist_arr + 0.486) * dist_arr + 10.0
    ax.plot(dist_arr, vpeak_cut, color='red', lw=2, label='Vpeak cut')
    ax.set_xlabel('Comoving distance [Mpc/h]')
    ax.set_ylabel(r'$V_\mathrm{peak}$ [km/s]')
    ax.legend()
    plt.tight_layout()
    savefig(fig, outdir, 'vpeak_vs_distance', show, log)

    # --- Mpeak_Scale vs colour ---
    fig, ax = plt.subplots(figsize=(6, 5))
    col_mock = mock['ABSMAG_GP1'] - mock['ABSMAG_RP1']
    sc = ax.hexbin(mock['Mpeak_Scale'], col_mock,
                   C=mock['Z'], gridsize=200, reduce_C_function=np.mean,
                   cmap='viridis')
    # Running median
    bins   = np.linspace(mock['Mpeak_Scale'].min(), mock['Mpeak_Scale'].max(), 41)
    idx    = np.digitize(mock['Mpeak_Scale'], bins)
    bcen   = 0.5 * (bins[:-1] + bins[1:])
    median = np.array([np.median(col_mock[idx == i]) if np.any(idx == i) else np.nan
                       for i in range(1, len(bins))])
    ax.plot(bcen, median, color='red', lw=2, label='Running median')
    ax.set_xlabel(r'$M_\mathrm{peak,scale}$')
    ax.set_ylabel(r'restframe $G-R$')
    ax.legend()
    plt.colorbar(sc, ax=ax, label=r'$Z_u$')
    plt.tight_layout()
    savefig(fig, outdir, 'mpeakscale_vs_colour', show, log)

    # --- Vpeak vs ABSMAG_RP1 scatter ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mock['Vpeak'], mock['ABSMAG_RP1'], s=1, alpha=0.2, rasterized=True)
    ax.axvline(200, ls='--', color='k')
    ax.invert_yaxis()
    ax.set_xlabel(r'$V_\mathrm{peak}$ [km/s]')
    ax.set_ylabel(r'$M_r - 5\log h$')
    plt.tight_layout()
    savefig(fig, outdir, 'vpeak_vs_absmag_rp1', show, log)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Check at least one plot group was requested.
    any_plots = args.sky or args.redshift or args.photometric or args.halo or args.jackknife
    if not any_plots:
        print("No plot groups selected.  Use --sky, --redshift, --photometric, "
              "--halo, or --jackknife (or any combination).")
        sys.exit(0)

    if args.show:
        matplotlib.use('TkAgg')
        plt.ion()

    log = setup_logging("inspect_mocks")
    log.info(f"reg={args.reg}  obs_seed={args.obs_seed}  "
             f"sig_lgv_ref={args.sig_lgv_ref}  sig_lgv_slope={args.sig_lgv_slope}  "
             f"f_colour={args.f_colour}  alpha={args.alpha}")

    # Output directory.
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        tag    = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        outdir = (DATA_DIR / "diagnostics" /
                  f"{args.reg}_seed{args.obs_seed:03d}_sig_lgv{args.sig_lgv_ref:.2f}_{args.sig_lgv_slope:.2f}"
                  f"_fc{args.f_colour:.2f}_al{args.alpha:.2f}_{tag}")
    outdir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {outdir}")

    # Build paths.
    paths = make_paths(reg=args.reg, obs_seed=args.obs_seed,
                       sig_lgv_ref=args.sig_lgv_ref, sig_lgv_slope=args.sig_lgv_slope,
                       f_colour=args.f_colour, alpha=args.alpha)

    # Load catalogues.
    import catalogue_analysis as ca
    log.info("=== Loading catalogues ===")
    mock    = load_catalogue(paths[f"mock_{args.reg}"],       "Mock",            log)
    ran     = load_catalogue(paths[f"formock_ran_{args.reg}"],"Formock randoms", log)
    ran_inp = load_catalogue(paths[f"fordata_ran_{args.reg}"],"Fordata randoms", log)

    # Data catalogue — filter to the requested region.
    from config import BGS_CAT
    dat_full = load_catalogue(BGS_CAT, "BGS catalogue", log)
    dat      = dat_full[dat_full['reg'] == args.reg].copy()
    del dat_full
    log.info(f"Data: {len(dat):,} galaxies in region {args.reg}")

    # Apply faint absolute magnitude cut.
    cats = apply_absmag_cut(
        {'dat': dat, 'mock': mock, 'ran': ran, 'ran_inp': ran_inp},
        args.absmag_lim, log,
    )
    dat, mock, ran, ran_inp = cats['dat'], cats['mock'], cats['ran'], cats['ran_inp']

    # Run requested plot groups.
    if args.sky:
        plot_sky(dat, mock, ran, ran_inp, args.reg, outdir, args.show, log)

    if args.redshift:
        plot_redshift(dat, mock, ran, ran_inp, args.reg, outdir, args.show, log)

    if args.jackknife:
        plot_jackknife_dndz(dat, mock, ran, args.reg, outdir, args.show, log)

    if args.photometric:
        plot_photometric(dat, mock, ran, args.reg, outdir, args.show, log)

    if args.halo:
        plot_halo(mock, outdir, args.show, log)

    log.info(f"inspect_mocks.py complete.  Figures in: {outdir}")


if __name__ == '__main__':
    main()
