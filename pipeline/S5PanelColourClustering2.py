"""
S5PanelColourClustering2.py — Five-panel red/blue colour-split clustering
figure for the MNRAS paper, showing the Vpeak-dependent scatter model
(sig_lgv_ref=0.18, sig_lgv_slope=0.30, f_colour=0.00) against BGS DR2 data.

Single-column vertical layout with shared x-axis:
    [ Mr<-21.5 ]
    [ Mr<-20.5 ]
    [ Mr<-19.5 ]
    [ Mr<-18.5 ]
    [ Mr<-17.5 ]

Hardwired paths — no command-line arguments needed.
Run from any directory:
    python S5PanelColourClustering2.py
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
OUT_STEM = Path(__file__).resolve().parent / "S5PanelColourClustering2"

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

# =============================================================================
# Colour palette
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


# =============================================================================
# Main
# =============================================================================

def main():
    ells    = [0, 2]
    colours = ['red', 'blue']
    n       = len(SAMPLES)

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

        data_handles = []
        mock_handles = []

        for colour in colours:
            # --- Data ---
            data_result = load_result(data_file(mr, z0, z1, colour))
            if data_result is not None:
                s_dat, xi_dat, err_dat, good_dat, has_cov = extract_multipoles(
                    data_result, ells)
                s_d      = s_dat[good_dat]
                xi_dat_g = {ell: xi_dat[ell][good_dat] for ell in ells}
                err_dat_g= {ell: err_dat[ell][good_dat] for ell in ells}

                for ell in ells:
                    c    = DATA_COLOURS[colour][f'xi{ell}']
                    fmt  = DATA_MARKERS[f'xi{ell}']
                    y    = s_d**2 * xi_dat_g[ell]
                    yerr = s_d**2 * err_dat_g[ell]
                    lbl = f'{colour.capitalize()} $\\xi_{ell}$'
                    if has_cov and np.any(err_dat_g[ell] > 0):
                        h = ax.errorbar(s_d, y, yerr=yerr,
                                        fmt=fmt, color=c,
                                        ms=1.5, capsize=1.5, lw=0.7,
                                        zorder=10, label=lbl)
                    else:
                        h, = ax.plot(s_d, y, fmt, color=c, ms=1.5,
                                     zorder=10, label=lbl)
                    data_handles.append(h)

            # --- Mock ---
            mock_result = load_result(mock_file(mr, z0, z1, colour))
            if mock_result is not None:
                s_m, xi_m, _, good_m, _ = extract_multipoles(mock_result, ells)
                xi_m_g = {ell: xi_m[ell][good_m] for ell in ells}
                s_ = s_m[good_m]

                line_col = DATA_COLOURS[colour]['xi0']
                for ell in ells:
                    h, = ax.plot(s_, s_**2 * xi_m_g[ell],
                                 lw=0.7, color=line_col,
                                 ls=MOCK_LS[f'xi{ell}'],
                                 label=f'Mock {colour}' if ell == 0 else '_nolegend_')
                    if ell == 0:
                        mock_handles.append(h)

        # Sample info — bottom left
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

        # Data legend top panel, mock legend second panel
        if i == 0:
            ax.legend(handles=data_handles, fontsize=7,
                      loc='upper left', framealpha=0.8)
        if i == 1:
            ax.legend(handles=mock_handles, fontsize=7,
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

    plt.savefig(str(OUT_STEM) + '.pdf', bbox_inches='tight')
    print(f"Saved: {OUT_STEM}.pdf")
    plt.savefig(str(OUT_STEM) + '.jpg', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUT_STEM}.jpg")
    plt.close(fig)


if __name__ == "__main__":
    main()
