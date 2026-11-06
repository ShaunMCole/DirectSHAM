"""
plot_lss_clustering.py — Single-panel clustering figure for the LSS large-
scale structure sample (DM-only Mr < -20.7, 0.1 < z < 0.3).

Shows ξ₀ and ξ₂ for:
  - BGS DR2 data     (black / grey, with jackknife error bars)
  - Lightcone mock   (red / pink, solid lines)
  - Cut-sky mock     (red / pink, dashed lines, optional via --cutsky)

X-axis extends to 150 Mpc/h to show large-scale signal.

Usage
-----
python plot_lss_clustering.py
python plot_lss_clustering.py --cutsky          # add cut-sky mock
python plot_lss_clustering.py --mock_only       # skip data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycorr import TwoPointCorrelationFunction

# ---------------------------------------------------------------------------
# Pipeline paths (mirrors config.py conventions)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DATE_DIR, DEFAULT_SIG_LGV_REF, DEFAULT_SIG_LGV_SLOPE, DEFAULT_F_COLOUR, DEFAULT_ALPHA
from pipeline_utils import make_paths, make_lightcone_paths

# ---------------------------------------------------------------------------
# LSS sample parameters (must match measure_lss_clustering.py)
# ---------------------------------------------------------------------------
LSS_ABSMAG_CUT = -20.7
LSS_ZMIN       =  0.1
LSS_ZMAX       =  0.3
DATA_NJACK     = 60

# Colours
COL_XI0_DATA  = '#1f1f1f'
COL_XI2_DATA  = '#888888'
COL_XI0_LC    = '#d62728'   # mock monopole (lightcone solid, cut-sky dashed)
COL_XI2_LC    = '#ff9896'   # mock quadrupole (lightcone solid, cut-sky dashed)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sig_lgv_ref",   type=float, default=DEFAULT_SIG_LGV_REF)
    p.add_argument("--sig_lgv_slope", type=float, default=DEFAULT_SIG_LGV_SLOPE)
    p.add_argument("--f_colour",      type=float, default=DEFAULT_F_COLOUR)
    p.add_argument("--alpha",         type=float, default=DEFAULT_ALPHA)
    p.add_argument("--cutsky",   action="store_true",
                   help="Also plot the cut-sky (snap45) mock.")
    p.add_argument("--mock_only", action="store_true",
                   help="Skip data; plot mock(s) only.")
    return p.parse_args()

# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------

def _smu_glob(smu_dir, gc='GCcomb'):
    """Return the first .npy file in smu_dir matching GCcomb."""
    smu_dir = Path(smu_dir)
    if not smu_dir.exists():
        return None
    hits = sorted(smu_dir.glob(f"*{gc}*.npy"))
    return hits[0] if hits else None


def data_file(lss_clus_dir):
    smu_dir = Path(lss_clus_dir) / "data" / "smu"
    return _smu_glob(smu_dir)


def mock_lc_file(lss_clus_dir):
    smu_dir = Path(lss_clus_dir) / "mock" / "smu"
    return _smu_glob(smu_dir)


def mock_cs_file(paths):
    """Cut-sky mock clustering result — lives under model_dir/clustering/lss/."""
    smu_dir = paths["model_dir"] / "clustering" / "lss" / "mock" / "smu"
    return _smu_glob(smu_dir)

# ---------------------------------------------------------------------------
# Loading and multipole extraction
# ---------------------------------------------------------------------------

def load_result(path):
    if path is None or not Path(path).exists():
        print(f"WARNING: file not found: {path}")
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
    good = np.isfinite(s) & (s > 0)
    for xi in xi_dict.values():
        good &= np.isfinite(xi)
    return s, xi_dict, err_dict, good, has_cov

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    ells = [0, 2]

    paths    = make_paths(obs_seed=0,
                          sig_lgv_ref=args.sig_lgv_ref,
                          sig_lgv_slope=args.sig_lgv_slope,
                          f_colour=args.f_colour, alpha=args.alpha)
    lc_paths = make_lightcone_paths(sig_lgv_ref=args.sig_lgv_ref,
                                    sig_lgv_slope=args.sig_lgv_slope,
                                    f_colour=args.f_colour, alpha=args.alpha)
    lss_clus_dir = lc_paths["clustering_dir"] / "lss"

    # --- Load results ---
    data_result = None if args.mock_only else load_result(data_file(lss_clus_dir))
    lc_result   = load_result(mock_lc_file(lss_clus_dir))
    cs_result   = load_result(mock_cs_file(paths)) if args.cutsky else None

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(3.35, 3.0))

    legend_handles = []

    # Data
    if data_result is not None:
        s, xi, err, good, has_cov = extract_multipoles(data_result, ells)
        s_d  = s[good]
        xi_d = {ell: xi[ell][good]  for ell in ells}
        er_d = {ell: err[ell][good] for ell in ells}
        cols    = {0: COL_XI0_DATA, 2: COL_XI2_DATA}
        markers = {0: 'o', 2: 's'}
        labels  = {0: r'Data $\xi_0$', 2: r'Data $\xi_2$'}
        for ell in ells:
            y    = s_d**2 * xi_d[ell]
            yerr = s_d**2 * er_d[ell]
            if has_cov and np.any(er_d[ell] > 0):
                h = ax.errorbar(s_d, y, yerr=yerr,
                                fmt=markers[ell], color=cols[ell],
                                ms=1.5, capsize=1.5, lw=0.7, zorder=10,
                                label=labels[ell])
            else:
                h, = ax.plot(s_d, y, markers[ell], color=cols[ell],
                             ms=1.5, zorder=10, label=labels[ell])
            legend_handles.append(h)

    # Lightcone mock
    if lc_result is not None:
        s, xi, _, good, _ = extract_multipoles(lc_result, ells)
        s_m  = s[good]
        xi_m = {ell: xi[ell][good] for ell in ells}
        cols_lc   = {0: COL_XI0_LC, 2: COL_XI2_LC}
        ls_lc     = {0: '-',        2: '--'}
        # Label as "Mock" if no cut-sky comparison; "Lightcone" if --cutsky
        lc_prefix = 'Lightcone' if args.cutsky else 'Mock'
        labels_lc = {0: rf'{lc_prefix} $\xi_0$', 2: rf'{lc_prefix} $\xi_2$'}
        for ell in ells:
            h, = ax.plot(s_m, s_m**2 * xi_m[ell],
                         lw=0.9, color=cols_lc[ell], ls=ls_lc[ell],
                         label=labels_lc[ell])
            legend_handles.append(h)

    # Cut-sky mock (optional)
    if cs_result is not None:
        s, xi, _, good, _ = extract_multipoles(cs_result, ells)
        s_m  = s[good]
        xi_m = {ell: xi[ell][good] for ell in ells}
        # Same colours as lightcone; dotted lines to distinguish from solid/dashed
        cols_cs   = {0: COL_XI0_LC, 2: COL_XI2_LC}
        ls_cs     = {0: ':',        2: (0, (3, 1, 1, 1))}   # dotted / dash-dot
        labels_cs = {0: r'Cut-sky $\xi_0$', 2: r'Cut-sky $\xi_2$'}
        for ell in ells:
            h, = ax.plot(s_m, s_m**2 * xi_m[ell],
                         lw=0.9, color=cols_cs[ell], ls=ls_cs[ell],
                         label=labels_cs[ell])
            legend_handles.append(h)

    # Annotations
    ax.text(0.03, 0.04,
            rf"$M_r^{{\rm LSS}} - 5\log h < {LSS_ABSMAG_CUT}$,  "
            rf"${LSS_ZMIN} \leq z \leq {LSS_ZMAX}$",
            transform=ax.transAxes, fontsize=7, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

    ax.legend(handles=legend_handles, fontsize=7, loc='upper left', framealpha=0.8)
    ax.axhline(0, color='k', lw=0.7, ls='--')
    ax.set_xscale('log')
    ax.set_xlim(0.5, 150)
    ax.set_xlabel(r'$s$ [Mpc/$h$]', fontsize=9)
    ax.set_ylabel(r'$s^2\xi_\ell(s)$ [Mpc$^2 h^{-2}$]', fontsize=9)
    ax.tick_params(labelsize=8, which='both', top=False, right=False)

    out_stem = Path(__file__).resolve().parent / "plot_lss_clustering"
    suffix   = "_cutsky" if args.cutsky else ""
    for ext in ("pdf", "jpg"):
        fout = f"{out_stem}{suffix}.{ext}"
        kw = dict(bbox_inches='tight')
        if ext == 'jpg':
            kw['dpi'] = 150
        plt.savefig(fout, **kw)
        print(f"Saved: {fout}")
    plt.close(fig)


if __name__ == "__main__":
    main()
