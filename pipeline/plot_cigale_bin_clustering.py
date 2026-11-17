"""
plot_cigale_bin_clustering.py — Compare data and mock xi_0(s) for CIGALE
stellar mass bin x redshift bin samples (Shun Saito KP HOD paper cells).

One panel per redshift bin (6 panels in a 3x2 grid), each showing s^2 xi_0
for every mass bin present in that redshift range.  Data as points with
jackknife error bars, mock as solid lines, same colour per mass bin.

Run from ~/DirectSHAM/:
    python pipeline/plot_cigale_bin_clustering.py
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycorr import TwoPointCorrelationFunction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (DATA_DATE_DIR, DEFAULT_SIG_LGV_REF, DEFAULT_SIG_LGV_SLOPE,
                    DEFAULT_F_COLOUR, DEFAULT_ALPHA,
                    NLUM_DIR_TEMPLATE, MODEL_DIR_TEMPLATE)

# =============================================================================
# Sample definitions — all 19 CIGALE cells
# =============================================================================

# Colours per mass bin (matching Shun's Fig. 2 approximately)
MASS_COLORS = {
    (8.5,  9.0):  '#7f7f7f',   # grey
    (9.0,  9.5):  '#8B4513',   # brown
    (9.5,  10.0): '#9467bd',   # purple
    (10.0, 10.5): '#1f77b4',   # blue
    (10.5, 11.0): '#2ca02c',   # green
    (11.0, 12.0): '#d62728',   # red
}

MASS_LABEL = {
    (8.5,  9.0):  r'$M_\star \in [8.5, 9.0)$',
    (9.0,  9.5):  r'$M_\star \in [9.0, 9.5)$',
    (9.5,  10.0): r'$M_\star \in [9.5, 10.0)$',
    (10.0, 10.5): r'$M_\star \in [10.0, 10.5)$',
    (10.5, 11.0): r'$M_\star \in [10.5, 11.0)$',
    (11.0, 12.0): r'$M_\star \in [11.0, 12.0)$',
}

# Redshift bins with their mass bins
ZBINS = [
    dict(zmin=0.027, zmax=0.043, mass_bins=[
        (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]),
    dict(zmin=0.043, zmax=0.068, mass_bins=[
        (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]),
    dict(zmin=0.068, zmax=0.107, mass_bins=[
        (9.5, 10.0), (10.0, 10.5), (10.5, 11.0), (11.0, 12.0)]),
    dict(zmin=0.107, zmax=0.162, mass_bins=[
        (10.0, 10.5), (10.5, 11.0), (11.0, 12.0)]),
    dict(zmin=0.162, zmax=0.240, mass_bins=[
        (10.5, 11.0), (11.0, 12.0)]),
    dict(zmin=0.240, zmax=0.344, mass_bins=[
        (11.0, 12.0)]),
]

DATA_NJACK  = 60
SIG_REF     = DEFAULT_SIG_LGV_REF
SIG_SLOPE   = DEFAULT_SIG_LGV_SLOPE
OUT_STEM    = Path(__file__).resolve().parent / "cigale_bin_clustering"

# =============================================================================
# Path helpers
# =============================================================================

def _zfmt(z):
    s = f"{z:.3f}"
    # strip trailing zero only if file exists that way
    return s

def _find_npy(smu_dir, stem):
    """Try exact filename; if missing, try with trailing zeros stripped."""
    import re
    p = smu_dir / stem
    if p.exists():
        return p
    def strip(m):
        s = m.group(0)
        return s.rstrip('0').rstrip('.') if '.' in s else s
    stem2 = re.sub(r'\d+\.\d+', strip, stem)
    p2 = smu_dir / stem2
    return p2 if p2.exists() else p

def data_npy(zmin, zmax, mmin, mmax):
    tag = f"Mcig{mmin}-{mmax}_z{zmin:.3f}-{zmax:.3f}"
    smu = DATA_DATE_DIR / "clustering" / "data" / tag / "smu"
    stem = (f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.3f}_{zmax:.3f}"
            f"_pip_angular_bitwise_log_njack{DATA_NJACK}_nran1_split20.npy")
    return _find_npy(smu, stem)

def mock_npy(zmin, zmax, mmin, mmax, model_dir):
    tag = f"Mcig{mmin}-{mmax}_z{zmin:.3f}-{zmax:.3f}"
    smu = Path(model_dir) / "clustering" / tag / "smu"
    stem = (f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.3f}_{zmax:.3f}"
            f"_default_log_njack0_nran1_split20.npy")
    return _find_npy(smu, stem)

# =============================================================================
# Load / extract monopole
# =============================================================================

def load_result(path):
    path = Path(path)
    if not path.exists():
        print(f"  WARNING: not found: {path}")
        return None
    return TwoPointCorrelationFunction.load(str(path))

def extract_monopole_with_err(result):
    s = result.sep
    if s.ndim == 2:
        s = np.nanmean(s, axis=1)
    xi0_err = None
    try:
        xis, cov = result.get_corr(ells=(0,), return_cov=True)
        xi0 = np.atleast_1d(xis[0]).ravel()
        xi0_err = np.sqrt(np.diag(cov))[:len(s)]
    except Exception:
        xi0 = np.atleast_1d(result(ells=(0,))).ravel()
    ns = len(s)
    if len(xi0) > ns:
        xi0 = xi0[:ns]
    good = np.isfinite(s) & (s > 0) & np.isfinite(xi0)
    err = xi0_err[good] if xi0_err is not None else None
    return s[good], xi0[good], err

# =============================================================================
# Main
# =============================================================================

def main():
    nlum_dir  = DATA_DATE_DIR / NLUM_DIR_TEMPLATE.format(n_neigh_lum=50)
    model_dir = nlum_dir / MODEL_DIR_TEMPLATE.format(
        obs_seed=0, sig_lgv_ref=SIG_REF, sig_lgv_slope=SIG_SLOPE,
        f_colour=DEFAULT_F_COLOUR, alpha=DEFAULT_ALPHA)
    print(f"Model dir: {model_dir}")

    fig, axes = plt.subplots(3, 2, figsize=(10, 10),
                             sharex=True,
                             gridspec_kw=dict(hspace=0.3, wspace=0.3))
    axes_flat = axes.flatten()

    for ax, zbin in zip(axes_flat, ZBINS):
        zmin, zmax = zbin['zmin'], zbin['zmax']
        handles = []

        for (mmin, mmax) in zbin['mass_bins']:
            color = MASS_COLORS[(mmin, mmax)]
            label = MASS_LABEL[(mmin, mmax)]

            # --- Data ---
            dr = load_result(data_npy(zmin, zmax, mmin, mmax))
            if dr is not None:
                s_d, xi_d, xi_err = extract_monopole_with_err(dr)
                y    = s_d**2 * xi_d
                yerr = s_d**2 * xi_err if xi_err is not None else None
                if yerr is not None:
                    ax.errorbar(s_d, y, yerr=yerr, fmt='o', color=color,
                                ms=2.5, capsize=2, lw=0.8, zorder=10)
                else:
                    ax.plot(s_d, y, 'o', color=color, ms=2.5, zorder=10)

            # --- Mock ---
            mr = load_result(mock_npy(zmin, zmax, mmin, mmax, model_dir))
            if mr is not None:
                s_m, xi_m, _ = extract_monopole_with_err(mr)
                h, = ax.plot(s_m, s_m**2 * xi_m, '-', color=color,
                             lw=1.2, zorder=5, label=label)
                handles.append(h)

        ax.axhline(0, color='k', lw=0.5, ls='--')
        ax.set_xscale('log')
        ax.set_xlim(0.3, 60)
        ax.set_ylabel(r'$s^2\xi_0(s)\ [\mathrm{Mpc}^2h^{-2}]$', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(rf"${zmin:.3f} \leq z < {zmax:.3f}$", fontsize=9)
        if handles:
            ax.legend(handles=handles, fontsize=6.5, frameon=False,
                      loc='upper left')

    # Hide unused panel (we have 6 bins, 6 panels — all used)
    for ax in axes_flat[len(ZBINS):]:
        ax.set_visible(False)

    axes_flat[-2].set_xlabel(r'$s\ [\mathrm{Mpc}/h]$', fontsize=9)
    axes_flat[-1].set_xlabel(r'$s\ [\mathrm{Mpc}/h]$', fontsize=9)

    fig.suptitle(
        rf"CIGALE stellar mass bins: data (points) vs mock (lines)"
        "\n"
        rf"$\sigma_{{v,0}}={SIG_REF}$, slope$={SIG_SLOPE}$",
        fontsize=10)

    for ext in ('.pdf', '.jpg'):
        out = f"{OUT_STEM}{ext}"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
