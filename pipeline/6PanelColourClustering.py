"""
6PanelClustering2.py — Six-panel red/blue colour-split clustering figure
for the MNRAS paper, showing the Vpeak-dependent scatter model
(sig_lgv_ref=0.18, sig_lgv_slope=0.30, f_colour=0.00) against BGS DR2 data.

Panels (2 columns x 3 rows):
    [ Mr<-22.0 ]  [ Mr<-21.5 ]
    [ Mr<-20.5 ]  [ Mr<-19.5 ]
    [ Mr<-18.5 ]  [ Mr<-17.5 ]

Hardwired paths — no command-line arguments needed.
Run from any directory:
    python 6PanelColourClustering2.py
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
OUT_STEM = Path(__file__).resolve().parent / "6PanelColourClustering2"

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

# y-axis limits per magnitude bin
YMAX = {
    -22.0: 180, -21.5: 120, -20.5: 100,
    -19.5: 100, -18.5: 100, -17.5: 100,
}

# =============================================================================
# Colour palette — consistent with compare scripts
# =============================================================================

DATA_COLOURS = {
    'red':  {'xi0': '#8B0000', 'xi2': '#FF6B6B'},
    'blue': {'xi0': '#00008B', 'xi2': '#6B9FFF'},
}
DATA_MARKERS = {'xi0': 'o', 'xi2': 's'}
MOCK_LS      = {'xi0': '-',  'xi2': '--'}

# =============================================================================
# Path helpers
# =============================================================================

def data_file(mr_cut, zmin, zmax, colour):
    sample_tag = f"Mr{mr_cut}_z{zmin:.2f}-{zmax:.2f}_{colour}"
    smu_dir    = DATA_CLUSTERING_DIR / sample_tag / "smu"
    pattern    = (
        f"allcounts_BGS_BRIGHT_GCcomb_{zmin}_{zmax}_pip_angular_bitwise"
        f"_log_njack{DATA_NJACK}_nran1_split20.npy"
    )
    return smu_dir / pattern


def mock_file(mr_cut, zmin, zmax, colour):
    sample_tag = f"Mr{mr_cut}_z{zmin:.2f}-{zmax:.2f}_{colour}"
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
    result = TwoPointCorrelationFunction.load(str(path))
    return result


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
    def _interp(s_from, xi_from, s_to):
        return np.interp(s_to, s_from, xi_from)
    fit   = (s_dat >= smin) & (s_dat <= smax)
    s_fit = s_dat[fit]
    ndof  = fit.sum() * len(ells)
    residuals, weights = [], []
    for ell in ells:
        xi_d = xi_dat[ell][fit]
        xi_m = _interp(s_mock, xi_mock[ell], s_fit)
        residuals.append(xi_d - xi_m)
        err  = err_dat[ell][fit]
        err  = np.where((err > 0) & np.isfinite(err), err, 1.0)
        weights.append(err)
    delta = np.concatenate(residuals)
    w     = np.concatenate(weights)
    chi2  = float(np.dot(delta / w, delta / w))
    return chi2, ndof


# =============================================================================
# Main
# =============================================================================

def main():
    ells = [0, 2]
    colours = ['red', 'blue']

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    for sample, (row, col) in zip(SAMPLES, LAYOUT):
        ax  = axes[row, col]
        mr  = sample['mr_cut']
        z0  = sample['zmin']
        z1  = sample['zmax']

        ymax = YMAX.get(mr, 120)
        ymin = -ymax * 0.85

        ax.set_title(f"$M_r < {mr}$  |  ${z0:.2f} \\leq z \\leq {z1:.2f}$",
                     fontsize=10)

        data_handles = []

        for colour in colours:
            # --- Data ---
            df          = data_file(mr, z0, z1, colour)
            data_result = load_result(df)
            if data_result is None:
                continue

            s_dat, xi_dat, err_dat, good_dat, has_cov = extract_multipoles(
                data_result, ells
            )
            s_d = s_dat[good_dat]

            for ell in ells:
                c    = DATA_COLOURS[colour][f'xi{ell}']
                fmt  = DATA_MARKERS[f'xi{ell}']
                y    = s_d**2 * xi_dat[ell][good_dat]
                yerr = s_d**2 * err_dat[ell][good_dat]
                if has_cov and np.any(err_dat[ell][good_dat] > 0):
                    h = ax.errorbar(s_d, y, yerr=yerr, fmt=fmt, color=c,
                                    ms=4, capsize=2, lw=0.8, zorder=10,
                                    label=f'Data {colour} $\\xi_{ell}$')
                else:
                    h, = ax.plot(s_d, y, fmt, color=c, ms=4, zorder=10,
                                 label=f'Data {colour} $\\xi_{ell}$')
                if ell == 0:
                    data_handles.append(h)

            # --- Mock ---
            mf          = mock_file(mr, z0, z1, colour)
            mock_result = load_result(mf)
            if mock_result is None:
                continue

            s_m, xi_m, _, good_m, _ = extract_multipoles(mock_result, ells)

            # chi2 using jackknife errors — apply good_dat mask to xi/err first
            xi_dat_good  = {ell: xi_dat[ell][good_dat]  for ell in ells}
            err_dat_good = {ell: err_dat[ell][good_dat] for ell in ells}
            xi_m_good    = {ell: xi_m[ell][good_m]      for ell in ells}
            chi2, ndof = compute_chi2(
                s_dat[good_dat], xi_dat_good, err_dat_good,
                s_m[good_m], xi_m_good, ells
            )
            chi2_str = f"{chi2/ndof:.1f}"

            s_ = s_m[good_m]
            mock_label = f"{colour} $\\chi^2$/df={chi2_str}"
            first = True
            for ell in ells:
                line_col = DATA_COLOURS[colour]['xi0']
                lbl = mock_label if first else "_nolegend_"
                first = False
                ax.plot(s_, s_**2 * xi_m[ell][good_m],
                        lw=1.8, color=line_col,
                        ls=MOCK_LS[f'xi{ell}'], label=lbl)

        # Legends
        if row == 0 and col == 0:
            data_legend = ax.legend(
                handles=data_handles, fontsize=7,
                loc='lower left', framealpha=0.8
            )
            ax.add_artist(data_legend)

        mock_lines = [l for l in ax.get_lines()
                      if l.get_linestyle() == '-'
                      and not l.get_label().startswith('_')
                      and not l.get_label().startswith('Data')]
        ax.legend(handles=mock_lines, fontsize=7,
                  loc='upper left', framealpha=0.8)

        ax.axhline(0, color='k', lw=0.7, ls='--')
        ax.set_xscale('log')
        ax.set_xlim(0.1, 80)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'$s$ [Mpc/$h$]', fontsize=9)
        ax.set_ylabel(r'$s^2\xi_\ell(s)$ [Mpc$^2h^{-2}$]', fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        r"BGS DR2 data vs Uchuu mock — red/blue split"
        "\n"
        r"$\sigma_{\lg V}(V_\mathrm{peak}) = 0.18 + 0.30\,(\lg V_\mathrm{peak} - 2.6)$",
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
