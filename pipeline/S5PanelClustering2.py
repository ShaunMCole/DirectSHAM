"""
S5PanelClustering2.py — Five-panel monochrome clustering figure for the MNRAS
paper, showing the Vpeak-dependent scatter model
(sig_lgv_ref=0.18, sig_lgv_slope=0.30, f_colour=0.00) against BGS DR2 data.

Single-column vertical layout with shared x-axis:
    [ Mr<-21.5 ]
    [ Mr<-20.5 ]
    [ Mr<-19.5 ]
    [ Mr<-18.5 ]
    [ Mr<-17.5 ]

Run from any directory:
    python S5PanelClustering2.py
    python S5PanelClustering2.py --lightcone
    python S5PanelClustering2.py --seed2 123          # compare seed000 vs seed123
    python S5PanelClustering2.py --seed2 123 --seed2_only  # seed123 vs data only
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycorr import TwoPointCorrelationFunction


def parse_args():
    p = argparse.ArgumentParser(
        description='Five-panel clustering figure for the MNRAS paper.'
    )
    p.add_argument('--lightcone', action='store_true',
                   help='Plot lightcone mock clustering instead of cut-sky mock.')
    p.add_argument('--seed2', type=int, default=None,
                   help='If set, overplot a second mock with this obs_seed '
                        '(e.g. --seed2 123 to compare seed000 vs seed123).')
    p.add_argument('--seed2_only', action='store_true',
                   help='If set with --seed2, show only seed2 mock (not seed000). '
                        'Useful to plot seed123 vs data without seed000 clutter.')
    return p.parse_args()

# =============================================================================
# Hardwired paths
# =============================================================================

DATA_DATE_DIR = Path(
    "/global/cfs/cdirs/desi/users/smcole/DirectSHAM/6thOct"
)

DATA_CLUSTERING_DIR = DATA_DATE_DIR / "clustering" / "data"
DATA_NJACK = 60

# =============================================================================
# The 5 samples
# =============================================================================

SAMPLES = [
    dict(mr_cut=-21.5, zmin=0.02, zmax=0.42),
    dict(mr_cut=-20.5, zmin=0.02, zmax=0.28),
    dict(mr_cut=-19.5, zmin=0.02, zmax=0.19),
    dict(mr_cut=-18.5, zmin=0.02, zmax=0.12),
    dict(mr_cut=-17.5, zmin=0.02, zmax=0.08),
]

YMAX = {
    -21.5: 150, -20.5: 119.99, -19.5: 99.99, -18.5: 99.99, -17.5: 99.99,
}

# Colours
COL_XI0_DATA  = '#1f1f1f'
COL_XI2_DATA  = '#888888'
COL_XI0_MOCK1 = '#d62728'   # seed000 (or sole mock)
COL_XI2_MOCK1 = '#ff9896'
COL_XI0_MOCK2 = '#1f77b4'   # seed2
COL_XI2_MOCK2 = '#aec7e8'

# =============================================================================
# Path helpers
# =============================================================================

def model_dir(obs_seed):
    return DATA_DATE_DIR / (
        f"N_neigh_lum50/seed{obs_seed:03d}_sig_lgv0.18_0.30_fc0.00_al0.00"
    )


def data_file(mr_cut, zmin, zmax):
    sample_tag = f"Mr{mr_cut}_z{zmin:.2f}-{zmax:.2f}"
    smu_dir    = DATA_CLUSTERING_DIR / sample_tag / "smu"
    pattern    = (
        f"allcounts_BGS_BRIGHT_GCcomb_{zmin}_{zmax}_pip_angular_bitwise"
        f"_log_njack{DATA_NJACK}_nran1_split20.npy"
    )
    return smu_dir / pattern


def mock_file(mr_cut, zmin, zmax, obs_seed=0, lightcone=False):
    lc_tag     = "_lightcone" if lightcone else ""
    sample_tag = f"Mr{mr_cut}_z{zmin:.2f}-{zmax:.2f}{lc_tag}"
    smu_dir    = model_dir(obs_seed) / "clustering" / sample_tag / "smu"
    pattern    = (
        f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.2f}_{zmax:.2f}_default"
        f"_log_njack0_nran1_split20.npy"
    )
    return smu_dir / pattern


# =============================================================================
# Loading and extraction
# =============================================================================

def load_result(path):
    path = Path(path)
    if not path.exists():
        print(f"WARNING: File not found: {path}")
        return None
    return TwoPointCorrelationFunction.load(str(path))


def extract_multipoles(result, ells=(0, 2)):
    s = result.sep
    if s.ndim == 2:
        s = np.nanmean(s, axis=1)
    try:
        xis, cov = result.get_corr(ells=tuple(ells), return_cov=True)
        xis  = np.atleast_2d(xis)
        ns   = xis.shape[1]
        errs = np.sqrt(np.diag(cov))
        err_dict = {ell: errs[i*ns:(i+1)*ns] for i, ell in enumerate(ells)}
        has_cov  = True
    except Exception:
        xis      = np.atleast_2d(result(ells=tuple(ells)))
        ns       = xis.shape[1]
        err_dict = {ell: np.zeros(ns) for ell in ells}
        has_cov  = False
    xi_dict = {ell: xis[i] for i, ell in enumerate(ells)}
    good    = np.isfinite(s) & (s > 0)
    for xi in xi_dict.values():
        good &= np.isfinite(xi)
    return s, xi_dict, err_dict, good, has_cov


def compute_chi2(s_dat, xi_dat, err_dat, s_mock, xi_mock,
                 ells, smin=5.0, smax=50.0):
    fit   = (s_dat >= smin) & (s_dat <= smax)
    s_fit = s_dat[fit]
    ndof  = fit.sum() * len(ells)
    residuals, weights = [], []
    for ell in ells:
        xi_d = xi_dat[ell][fit]
        xi_m = np.interp(s_fit, s_mock, xi_mock[ell])
        residuals.append(xi_d - xi_m)
        err  = err_dat[ell][fit]
        err  = np.where((err > 0) & np.isfinite(err), err, 1.0)
        weights.append(err)
    delta = np.concatenate(residuals)
    w     = np.concatenate(weights)
    return float(np.dot(delta / w, delta / w)), ndof


def plot_mock(ax, mr, z0, z1, obs_seed, lightcone, col0, col2,
              label0, label2, s_d, xi_dat_g, err_dat_g, ells):
    """Load and plot one mock; return legend handles and chi2/dof string."""
    result = load_result(mock_file(mr, z0, z1, obs_seed=obs_seed,
                                   lightcone=lightcone))
    handles = []
    chi2_str = '?'
    if result is not None and s_d is not None:
        s_m, xi_m, _, good_m, _ = extract_multipoles(result, ells)
        xi_m_g = {ell: xi_m[ell][good_m] for ell in ells}
        chi2, ndof = compute_chi2(
            s_d, xi_dat_g, err_dat_g, s_m[good_m], xi_m_g, ells)
        chi2_str = f"{chi2/ndof:.1f}"
        cols = {0: col0, 2: col2}
        ls   = {0: '-',  2: '--'}
        labs = {0: label0, 2: label2}
        for ell in ells:
            h, = ax.plot(s_m[good_m], s_m[good_m]**2 * xi_m_g[ell],
                         lw=0.7, color=cols[ell], ls=ls[ell], label=labs[ell])
            handles.append(h)
    return handles, chi2_str


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    lc_tag   = "_lightcone" if args.lightcone else ""
    seed2_tag = f"_seed{args.seed2:03d}" if args.seed2 is not None else ""
    only_tag  = "_only" if args.seed2_only else ""
    out_stem  = (
        Path(__file__).resolve().parent
        / f"S5PanelClustering2{lc_tag}{seed2_tag}{only_tag}"
    )

    ells = [0, 2]
    n    = len(SAMPLES)

    fig, axes = plt.subplots(
        n, 1,
        figsize=(3.35, 2.0 * n),
        sharex=True,
        gridspec_kw=dict(hspace=0),
    )

    for i, (sample, ax) in enumerate(zip(SAMPLES, axes)):
        mr = sample['mr_cut']
        z0 = sample['zmin']
        z1 = sample['zmax']

        ymax = YMAX.get(mr, 120)
        ymin = -ymax * 0.5

        legend_handles = []
        s_d = xi_dat_g = err_dat_g = None

        # --- Data ---
        data_result = load_result(data_file(mr, z0, z1))
        if data_result is not None:
            s_dat, xi_dat, err_dat, good_dat, has_cov = extract_multipoles(
                data_result, ells)
            s_d       = s_dat[good_dat]
            xi_dat_g  = {ell: xi_dat[ell][good_dat]  for ell in ells}
            err_dat_g = {ell: err_dat[ell][good_dat] for ell in ells}

            cols_data = {0: COL_XI0_DATA, 2: COL_XI2_DATA}
            markers   = {0: 'o', 2: 's'}
            labels    = {0: r'Data $\xi_0$', 2: r'Data $\xi_2$'}
            for ell in ells:
                y    = s_d**2 * xi_dat_g[ell]
                yerr = s_d**2 * err_dat_g[ell]
                if has_cov and np.any(err_dat_g[ell] > 0):
                    h = ax.errorbar(s_d, y, yerr=yerr,
                                    fmt=markers[ell], color=cols_data[ell],
                                    ms=1.5, capsize=1.5, lw=0.7, zorder=10,
                                    label=labels[ell])
                else:
                    h, = ax.plot(s_d, y, markers[ell],
                                 color=cols_data[ell], ms=1.5, zorder=10,
                                 label=labels[ell])
                legend_handles.append(h)

        # --- Mock 1 (seed000, unless seed2_only) ---
        if not args.seed2_only:
            seed1_label = 'seed000' if args.seed2 is not None else 'Mock'
            h1, chi2_1 = plot_mock(
                ax, mr, z0, z1, obs_seed=0,
                lightcone=args.lightcone,
                col0=COL_XI0_MOCK1, col2=COL_XI2_MOCK1,
                label0=fr'{seed1_label} $\xi_0$',
                label2=fr'{seed1_label} $\xi_2$',
                s_d=s_d, xi_dat_g=xi_dat_g, err_dat_g=err_dat_g, ells=ells,
            )
            legend_handles.extend(h1)

        # --- Mock 2 (seed2, if requested) ---
        if args.seed2 is not None:
            seed2_label = f'seed{args.seed2:03d}'
            h2, chi2_2 = plot_mock(
                ax, mr, z0, z1, obs_seed=args.seed2,
                lightcone=args.lightcone,
                col0=COL_XI0_MOCK2, col2=COL_XI2_MOCK2,
                label0=fr'{seed2_label} $\xi_0$',
                label2=fr'{seed2_label} $\xi_2$',
                s_d=s_d, xi_dat_g=xi_dat_g, err_dat_g=err_dat_g, ells=ells,
            )
            legend_handles.extend(h2)

        # Sample label
        sample_label = (
            f"$M_r < {mr}$,  "
            f"${z0:.2f} \\leq z \\leq {z1:.2f}$"
        )
        ax.text(
            0.03, 0.04, sample_label,
            transform=ax.transAxes,
            fontsize=8, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'),
        )

        # Legend — first panel only
        if i == 0:
            ax.legend(handles=legend_handles, fontsize=7,
                      loc='upper left', framealpha=0.8)

        ax.axhline(0, color='k', lw=0.7, ls='--')
        ax.set_xscale('log')
        ax.set_xlim(0.1, 80)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(r'$s^2\xi_\ell(s)$ [Mpc$^2h^{-2}$]', fontsize=9)
        ax.tick_params(labelsize=8, which='both', top=True, right=False)
        if i < n - 1:
            ax.tick_params(labelbottom=False, which='both')

    axes[-1].set_xlabel(r'$s$ [Mpc/$h$]', fontsize=9)
    axes[0].tick_params(which='both', top=False)

    plt.savefig(str(out_stem) + '.pdf', bbox_inches='tight')
    print(f"Saved: {out_stem}.pdf")
    plt.savefig(str(out_stem) + '.jpg', dpi=150, bbox_inches='tight')
    print(f"Saved: {out_stem}.jpg")
    plt.close(fig)


if __name__ == "__main__":
    main()
