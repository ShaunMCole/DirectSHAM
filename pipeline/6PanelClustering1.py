"""
6PanelClustering1.py — Six-panel monochrome clustering figure for the MNRAS
paper, showing five constant-scatter models against BGS DR2 data.

Models: sig_lgv = 0.04, 0.06, 0.08, 0.15, 0.20

Panels (2 columns x 3 rows):
    [ Mr<-22.0 ]  [ Mr<-21.5 ]
    [ Mr<-20.5 ]  [ Mr<-19.5 ]
    [ Mr<-18.5 ]  [ Mr<-17.5 ]

Bright bins (Mr<-22 to Mr<-20.5) read from OldDirectSHAM.
Faint bins (Mr<-19.5 to Mr<-17.5) read from new DirectSHAM structure
once those runs are complete (currently missing — panels will be blank).

Hardwired paths — no command-line arguments needed.
Run from any directory:
    python 6PanelClustering1.py
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

OLD_MOCK_BASE = Path(
    "/global/cfs/cdirs/desi/users/smcole/OldDirectSHAM/clustering/mocks"
)
NEW_BASE = Path(
    "/global/cfs/cdirs/desi/users/smcole/DirectSHAM/6thOct"
)
DATA_CLUSTERING_DIR = NEW_BASE / "clustering" / "data"
OUT_STEM = Path(__file__).resolve().parent / "6PanelClustering1"

DATA_NJACK = 60

# =============================================================================
# Models — (sig_lgv_ref, sig_lgv_slope, label, colour)
# For constant-scatter models slope=0 and the old directory name is
# seed000_sig_lgv{ref}  (no slope suffix)
# =============================================================================

MODELS = [
    dict(ref=0.04, slope=0.0, label=r'$\sigma_{\lg V}=0.04$', color='#1f77b4'),
    dict(ref=0.06, slope=0.0, label=r'$\sigma_{\lg V}=0.06$', color='#2ca02c'),
    dict(ref=0.08, slope=0.0, label=r'$\sigma_{\lg V}=0.08$', color='#17becf'),
    dict(ref=0.15, slope=0.0, label=r'$\sigma_{\lg V}=0.15$', color='#9467bd'),
    dict(ref=0.20, slope=0.0, label=r'$\sigma_{\lg V}=0.20$', color='#e377c2'),
]

# =============================================================================
# The 6 samples — bright bins from old structure, faint from new
# =============================================================================

BRIGHT_SAMPLES = [
    dict(mr_cut=-22.0, zmin=0.02, zmax=0.50, old=True),
    dict(mr_cut=-21.5, zmin=0.02, zmax=0.42, old=True),
    dict(mr_cut=-20.5, zmin=0.02, zmax=0.28, old=True),
]
FAINT_SAMPLES = [
    dict(mr_cut=-19.5, zmin=0.02, zmax=0.19, old=False),
    dict(mr_cut=-18.5, zmin=0.02, zmax=0.12, old=False),
    dict(mr_cut=-17.5, zmin=0.02, zmax=0.08, old=False),
]
SAMPLES = BRIGHT_SAMPLES + FAINT_SAMPLES
LAYOUT  = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]

YMAX = {
    -22.0: 200, -21.5: 150, -20.5: 120,
    -19.5: 100, -18.5: 100, -17.5: 100,
}

# =============================================================================
# Path helpers
# =============================================================================

def data_file(mr_cut, zmin, zmax):
    """Data clustering — new structure, njack=60."""
    sample_tag = f"Mr{mr_cut}_z{zmin:.2f}-{zmax:.2f}"
    smu_dir    = DATA_CLUSTERING_DIR / sample_tag / "smu"
    # Data files use str(zmin)/str(zmax) formatting (e.g. 0.5 not 0.50)
    pattern    = (
        f"allcounts_BGS_BRIGHT_GCcomb_{zmin}_{zmax}_pip_angular_bitwise"
        f"_log_njack{DATA_NJACK}_nran1_split20.npy"
    )
    return smu_dir / pattern


def mock_file_old(mr_cut, zmin, zmax, ref):
    """Mock clustering — old structure, constant scatter."""
    # Old directory: seed000_sig_lgv0.04  (no slope)
    # Old sample dir: Mr-22.0_z0.02-0.5  (str formatting)
    zmin_str = str(zmin)
    zmax_str = str(zmax)
    model_dir  = OLD_MOCK_BASE / f"seed000_sig_lgv{ref:.2f}"
    sample_dir = model_dir / f"Mr{mr_cut}_z{zmin_str}-{zmax_str}"
    smu_dir    = sample_dir / "smu"
    pattern    = (
        f"allcounts_BGS_BRIGHT_GCcomb_{zmin_str}_{zmax_str}_default"
        f"_log_njack0_nran1_split20.npy"
    )
    return smu_dir / pattern


def mock_file_new(mr_cut, zmin, zmax, ref, slope=0.0):
    """Mock clustering — new structure, for faint bins once available."""
    model_dir  = (
        NEW_BASE / "N_neigh_lum50"
        / f"seed000_sig_lgv{ref:.2f}_{slope:.2f}_fc0.00_al0.00"
    )
    sample_dir = model_dir / "clustering" / f"Mr{mr_cut}_z{zmin:.2f}-{zmax:.2f}"
    smu_dir    = sample_dir / "smu"
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
        ax  = axes[row, col]
        mr  = sample['mr_cut']
        z0  = sample['zmin']
        z1  = sample['zmax']
        old = sample['old']

        ymax = YMAX.get(mr, 120)
        ymin = -ymax * 0.5

        ax.set_title(
            f"$M_r < {mr}$  |  ${z0:.2f} \\leq z \\leq {z1:.2f}$",
            fontsize=10
        )

        # --- Data ---
        data_result = load_result(data_file(mr, z0, z1))
        s_d = xi_dat_g = err_dat_g = None
        data_handles = []

        if data_result is not None:
            s_dat, xi_dat, err_dat, good_dat, has_cov = extract_multipoles(
                data_result, ells)
            s_d       = s_dat[good_dat]
            xi_dat_g  = {ell: xi_dat[ell][good_dat]  for ell in ells}
            err_dat_g = {ell: err_dat[ell][good_dat] for ell in ells}

            cols_data = {0: '#1f1f1f', 2: '#888888'}
            markers   = {0: 'o',      2: 's'}
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
                data_handles.append(h)

        # --- Mocks ---
        mock_handles = []
        for model in MODELS:
            ref   = model['ref']
            col   = model['color']
            label = model['label']

            if old:
                mf = mock_file_old(mr, z0, z1, ref)
            else:
                mf = mock_file_new(mr, z0, z1, ref, slope=0.0)

            mock_result = load_result(mf)
            if mock_result is None:
                continue

            s_m, xi_m, _, good_m, _ = extract_multipoles(mock_result, ells)
            xi_m_g = {ell: xi_m[ell][good_m] for ell in ells}

            if s_d is not None:
                chi2, ndof = compute_chi2(
                    s_d, xi_dat_g, err_dat_g, s_m[good_m], xi_m_g, ells
                )
                chi2_str = f"{chi2/ndof:.1f}"
                lbl = label + f"  $\\chi^2$/df={chi2_str}"
            else:
                lbl = label

            # Monopole — solid; quadrupole — dashed, same colour
            h, = ax.plot(s_m[good_m], s_m[good_m]**2 * xi_m_g[0],
                         lw=1.5, color=col, ls='-', label=lbl)
            ax.plot(s_m[good_m], s_m[good_m]**2 * xi_m_g[2],
                    lw=1.5, color=col, ls='--')
            mock_handles.append(h)

        # Legends — data in lower left, models in upper left
        if row == 0 and col == 0:
            data_legend = ax.legend(
                handles=data_handles, fontsize=7,
                loc='lower left', framealpha=0.8
            )
            ax.add_artist(data_legend)

        ax.legend(handles=mock_handles, fontsize=7,
                  loc='upper left', framealpha=0.8)

        ax.axhline(0, color='k', lw=0.7, ls='--')
        ax.set_xscale('log')
        ax.set_xlim(0.1, 80)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'$s$ [Mpc/$h$]', fontsize=9)
        ax.set_ylabel(r'$s^2\xi_\ell(s)$ [Mpc$^2h^{-2}$]', fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        r"BGS DR2 data vs Uchuu mocks — constant $\sigma_{\lg V}$ models",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    for ext in ('pdf', 'jpg'):
        outpath = Path(str(OUT_STEM) + f'.{ext}')
        dpi = 150 if ext == 'jpg' else None
        fig.savefig(str(outpath), dpi=dpi, bbox_inches='tight')
        print(f"Saved: {outpath}")

    plt.close(fig)


if __name__ == "__main__":
    main()
