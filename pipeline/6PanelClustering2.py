"""
6PanelClustering2.py — Six-panel monochrome clustering figure for the MNRAS
paper, showing the Vpeak-dependent scatter model
(sig_lgv_ref=0.18, sig_lgv_slope=0.30, f_colour=0.00) against BGS DR2 data.

Panels (2 columns x 3 rows):
    [ Mr<-22.0 ]  [ Mr<-21.5 ]
    [ Mr<-20.5 ]  [ Mr<-19.5 ]
    [ Mr<-18.5 ]  [ Mr<-17.5 ]

Hardwired paths — no command-line arguments needed.
Run from any directory:
    python 6PanelClustering2.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycorr import TwoPointCorrelationFunction

# =============================================================================
# Hardwired paths
# =============================================================================

DATA_DATE_DIR = Path(
    "/global/cfs/cdirs/desi/users/smcole/DirectSHAM/6thOct"
)
MODEL_DIR = DATA_DATE_DIR / (
    "N_neigh_lum50/seed000_sig_lgv0.18_0.30_fc0.00_al0.00"
)
OUT_STEM = Path(__file__).resolve().parent / "6PanelClustering2"

# Data clustering — PIP angular bitwise, njack=60
DATA_CLUSTERING_DIR = DATA_DATE_DIR / "clustering" / "data"
DATA_NJACK = 60

# =============================================================================
# The 6 samples
# =============================================================================

SAMPLES = [
    dict(mr_cut=-22.0, zmin=0.02, zmax=0.50),
    dict(mr_cut=-21.5, zmin=0.02, zmax=0.42),
    dict(mr_cut=-20.5, zmin=0.02, zmax=0.28),
    dict(mr_cut=-19.5, zmin=0.02, zmax=0.19),
    dict(mr_cut=-18.5, zmin=0.02, zmax=0.12),
    dict(mr_cut=-17.5, zmin=0.02, zmax=0.08),
]
LAYOUT = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]

YMAX = {
    -22.0: 200, -21.5: 150, -20.5: 120,
    -19.5: 100, -18.5: 100, -17.5: 100,
}

# Colours for monopole and quadrupole
COL_XI0_DATA = '#1f1f1f'
COL_XI2_DATA = '#888888'
COL_XI0_MOCK = '#d62728'
COL_XI2_MOCK = '#ff9896'

# =============================================================================
# Path helpers
# =============================================================================

def data_file(mr_cut, zmin, zmax):
    sample_tag = f"Mr{mr_cut}_z{zmin:.2f}-{zmax:.2f}"
    smu_dir    = DATA_CLUSTERING_DIR / sample_tag / "smu"
    pattern    = (
        f"allcounts_BGS_BRIGHT_GCcomb_{zmin}_{zmax}_pip_angular_bitwise"
        f"_log_njack{DATA_NJACK}_nran1_split20.npy"
    )
    return smu_dir / pattern


def mock_file(mr_cut, zmin, zmax):
    sample_tag = f"Mr{mr_cut}_z{zmin:.2f}-{zmax:.2f}"
    smu_dir    = MODEL_DIR / "clustering" / sample_tag / "smu"
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


# =============================================================================
# Main
# =============================================================================

def main():
    ells = [0, 2]

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    for sample, (row, col) in zip(SAMPLES, LAYOUT):
        ax = axes[row, col]
        mr = sample['mr_cut']
        z0 = sample['zmin']
        z1 = sample['zmax']

        ymax = YMAX.get(mr, 120)
        ymin = -ymax * 0.5

        ax.set_title(f"$M_r < {mr}$  |  ${z0:.2f} \\leq z \\leq {z1:.2f}$",
                     fontsize=10)

        # --- Data ---
        data_result = load_result(data_file(mr, z0, z1))
        mock_result = load_result(mock_file(mr, z0, z1))

        legend_handles = []

        if data_result is not None:
            s_dat, xi_dat, err_dat, good_dat, has_cov = extract_multipoles(
                data_result, ells)
            s_d = s_dat[good_dat]
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
                                    ms=2.5, capsize=2, lw=0.8, zorder=10,
                                    label=labels[ell])
                else:
                    h, = ax.plot(s_d, y, markers[ell],
                                 color=cols_data[ell], ms=2.5, zorder=10,
                                 label=labels[ell])
                legend_handles.append(h)

        # --- Mock ---
        if mock_result is not None and data_result is not None:
            s_m, xi_m, _, good_m, _ = extract_multipoles(mock_result, ells)
            xi_m_g = {ell: xi_m[ell][good_m] for ell in ells}

            chi2, ndof = compute_chi2(
                s_d, xi_dat_g, err_dat_g, s_m[good_m], xi_m_g, ells
            )
            chi2_str = f"{chi2/ndof:.1f}"

            cols_mock = {0: COL_XI0_MOCK, 2: COL_XI2_MOCK}
            ls_mock   = {0: '-', 2: '--'}
            labels_m  = {0: r'Mock $\xi_0$', 2: r'Mock $\xi_2$'}
            for ell in ells:
                h, = ax.plot(s_m[good_m], s_m[good_m]**2 * xi_m_g[ell],
                             lw=1.8, color=cols_mock[ell],
                             ls=ls_mock[ell], label=labels_m[ell])
                legend_handles.append(h)

            # chi2/df in panel title
            ax.set_title(
                f"$M_r < {mr}$  |  ${z0:.2f} \\leq z \\leq {z1:.2f}$",
                fontsize=9
            )

        ax.legend(handles=legend_handles, fontsize=7,
                  loc='upper left', framealpha=0.8)
        ax.axhline(0, color='k', lw=0.7, ls='--')
        ax.set_xscale('log')
        ax.set_xlim(0.1, 80)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'$s$ [Mpc/$h$]', fontsize=9)
        ax.set_ylabel(r'$s^2\xi_\ell(s)$ [Mpc$^2h^{-2}$]', fontsize=9)
        ax.tick_params(labelsize=8)

    plt.tight_layout()

    for ext in ('pdf', 'jpg'):
        outpath = Path(str(OUT_STEM) + f'.{ext}')
        dpi = 150 if ext == 'jpg' else None
        fig.savefig(str(outpath), dpi=dpi, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.close(fig)


if __name__ == "__main__":
    main()
