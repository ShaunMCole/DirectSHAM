"""
compare_clustering_vpeak_scatter.py — Five-panel comparison of BGS data and
Uchuu mocks using the Vpeak-dependent scatter parametrisation
    sig_lgv = sig_lgv_ref + sig_lgv_slope * (log10(Vpeak) - 2.6)

One or more (ref, slope) pairs can be overlaid for comparison.

Layout:
    [ Mr<-22.0 ]  [ Mr<-21.5 ]
    [ Mr<-21.0 ]  [ Mr<-20.5 ]
    [ Mr<-20.0 ]

Usage examples
--------------
# Single model (defaults from config):
python compare_clustering_vpeak_scatter.py

# Two models for comparison:
python compare_clustering_vpeak_scatter.py \
    --models 0.14 0.24  0.14 0.30

# Save figure:
python compare_clustering_vpeak_scatter.py \
    --models 0.14 0.24  0.14 0.30 \
    --savefig 5panel_vpeak_comparison.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import setup_logging, make_paths
from config import DATA_DATE_DIR, DEFAULT_SIG_LGV_REF, DEFAULT_SIG_LGV_SLOPE

# =============================================================================
# The 5 standard samples
# =============================================================================

SAMPLES = [
    dict(mr_cut=-19.5, zmin=0.02, zmax=0.19),
    dict(mr_cut=-19.0, zmin=0.02, zmax=0.15),
    dict(mr_cut=-18.5, zmin=0.02, zmax=0.12),
    dict(mr_cut=-18.0, zmin=0.02, zmax=0.10),
    dict(mr_cut=-17.5, zmin=0.02, zmax=0.08),
]

LAYOUT = [(0,0), (0,1), (1,0), (1,1), (2,0)]


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--models", nargs="+", type=float, default=None,
        help=(
            "Flat list of (ref, slope) pairs to plot, e.g. "
            "--models 0.14 0.24  0.14 0.30. "
            f"Default: ({DEFAULT_SIG_LGV_REF}, {DEFAULT_SIG_LGV_SLOPE})"
        ),
    )
    p.add_argument(
        "--obs_seed", type=int, default=0,
        help="Observer position seed (default: 0)",
    )
    p.add_argument(
        "--use_jackknife", action="store_true",
        help="Use jackknife diagonal errors for chi2.",
    )
    p.add_argument(
        "--smin", type=float, default=5.0,
        help="Minimum separation [Mpc/h] for chi2 (default: 5.0)",
    )
    p.add_argument(
        "--smax", type=float, default=50.0,
        help="Maximum separation [Mpc/h] for chi2 (default: 50.0)",
    )
    p.add_argument(
        "--xlim", nargs=2, type=float, default=[0.1, 80],
        help="x-axis limits (default: 0.1 80)",
    )
    p.add_argument(
        "--savefig", type=str, default=None,
        help="Path to save the figure (default: display interactively).",
    )
    p.add_argument(
        "--data_njack", type=int, default=60,
        help="njack value used when measuring data clustering (default: 60)",
    )
    args = p.parse_args()

    # Parse --models into list of (ref, slope) tuples
    if args.models is None:
        args.model_pairs = [(DEFAULT_SIG_LGV_REF, DEFAULT_SIG_LGV_SLOPE)]
    else:
        if len(args.models) % 2 != 0:
            p.error("--models requires an even number of values (ref slope pairs)")
        args.model_pairs = [
            (args.models[i], args.models[i+1])
            for i in range(0, len(args.models), 2)
        ]
    return args


# =============================================================================
# Path helpers
# =============================================================================

def data_file(mr_cut, zmin, zmax, njack):
    sample_tag = f"Mr{mr_cut}_z{zmin:.2f}-{zmax:.2f}"
    smu_dir    = DATA_DATE_DIR / "clustering" / "data" / sample_tag / "smu"
    pattern    = f"allcounts_BGS_BRIGHT_GCcomb_{zmin}_{zmax}_pip_angular_bitwise_log_njack{njack}_nran1_split20.npy"
    return smu_dir / pattern


def mock_file(mr_cut, zmin, zmax, obs_seed, sig_lgv_ref, sig_lgv_slope):
    sample_tag = f"Mr{mr_cut}_z{zmin:.2f}-{zmax:.2f}"
    paths   = make_paths(obs_seed=obs_seed, sig_lgv_ref=sig_lgv_ref,
                         sig_lgv_slope=sig_lgv_slope)
    smu_dir = paths["model_dir"] / "clustering" / sample_tag / "smu"
    pattern = f"allcounts_BGS_BRIGHT_GCcomb_{zmin}_{zmax}_default_log_njack0_nran1_split20.npy"
    return smu_dir / pattern


# =============================================================================
# Loading and extraction
# =============================================================================

def load_result(path, log):
    from pycorr import TwoPointCorrelationFunction
    path = Path(path)
    if not path.exists():
        log.warning(f"File not found: {path}")
        return None
    result = TwoPointCorrelationFunction.load(str(path))
    log.info(f"Loaded: {path.name}")
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
                 ells, smin, smax, use_jackknife=False):
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
        if use_jackknife:
            err = err_dat[ell][fit]
            err = np.where((err > 0) & np.isfinite(err), err, 1.0)
            weights.append(err)
        else:
            weights.append(np.ones(fit.sum()))
    delta = np.concatenate(residuals)
    w     = np.concatenate(weights)
    chi2  = float(np.dot(delta / w, delta / w))
    return chi2, ndof


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    log  = setup_logging("compare_clustering_vpeak_scatter")
    ells = [0, 2]

    data_styles = {
        0: ('o', 'black', r'Data $\xi_0$'),
        2: ('s', 'grey',  r'Data $\xi_2$'),
    }
    mock_ls = {0: '-', 2: '--'}

    n_models = len(args.model_pairs)
    colours  = [cm.tab10(i / 10.0) for i in range(n_models)]

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes[2, 1].set_visible(False)

    for sample, (row, col) in zip(SAMPLES, LAYOUT):
        ax  = axes[row, col]
        mr  = sample['mr_cut']
        z0  = sample['zmin']
        z1  = sample['zmax']
        is_top_left = (row == 0 and col == 0)

        ymax = {-22.0: 180, -21.5: 120, -21.0: 90, -20.5: 75, -20.0: 70}.get(mr, 120)
        ymin = -ymax * 0.85

        ax.set_title(f"$M_r < {mr}$  |  ${z0} \\leq z \\leq {z1}$", fontsize=10)

        # --- Data ---
        df = data_file(mr, z0, z1, args.data_njack)
        data_result = load_result(df, log)
        if data_result is None:
            ax.text(0.5, 0.5, 'Data not found', transform=ax.transAxes,
                    ha='center', va='center', color='red')
            ax.set_xscale('log'); ax.set_xlim(args.xlim); ax.set_ylim(ymin, ymax)
            continue

        s_dat, xi_dat, err_dat, good_dat, has_cov = extract_multipoles(data_result, ells)
        s_d = s_dat[good_dat]

        data_handles = []
        for ell in ells:
            fmt, c, lbl = data_styles[ell]
            y    = s_d**2 * xi_dat[ell][good_dat]
            yerr = s_d**2 * err_dat[ell][good_dat]
            if has_cov and np.any(err_dat[ell][good_dat] > 0):
                h = ax.errorbar(s_d, y, yerr=yerr, fmt=fmt, color=c,
                                ms=4, capsize=2, lw=0.8, zorder=10, label=lbl)
            else:
                h, = ax.plot(s_d, y, fmt, color=c, ms=4, zorder=10, label=lbl)
            data_handles.append(h)

        # --- Mocks ---
        solid_mock_handles = []
        for (ref, slope), colour in zip(args.model_pairs, colours):
            mf = mock_file(mr, z0, z1, args.obs_seed, ref, slope)
            mock_result = load_result(mf, log)
            if mock_result is None:
                continue

            s_m, xi_m, err_m, good_m, _ = extract_multipoles(mock_result, ells)
            chi2, ndof = compute_chi2(
                s_dat, xi_dat, err_dat, s_m, xi_m,
                ells=ells, smin=args.smin, smax=args.smax,
                use_jackknife=args.use_jackknife and has_cov,
            )
            log.info(
                f"Mr{mr} z{z0}-{z1}  ref={ref}  slope={slope}  "
                f"chi2={chi2:.1f}  ndof={ndof}  chi2/df={chi2/ndof:.2f}"
            )

            s_ = s_m[good_m]
            mock_label = f"ref={ref}  slope={slope}  $\\chi^2$/df={chi2/ndof:.1f}"
            first = True
            for ell in ells:
                lbl = mock_label if first else "_nolegend_"
                first = False
                h, = ax.plot(s_, s_**2 * xi_m[ell][good_m],
                             lw=1.5, color=colour, ls=mock_ls[ell], label=lbl)
            solid_mock_handles.append(h)  # solid line for legend

        # --- Legends ---
        if is_top_left:
            data_legend = ax.legend(handles=data_handles, fontsize=7,
                                    loc='lower left', framealpha=0.8)
            ax.add_artist(data_legend)
        # Collect solid mock lines by label for legend
        solid_lines = [l for l in ax.get_lines()
                       if l.get_linestyle() == '-' and not l.get_label().startswith('_')]
        ax.legend(handles=solid_lines, fontsize=7, loc='upper left', framealpha=0.8)

        ax.axhline(0, color='k', lw=0.7, ls='--')
        ax.set_xscale('log')
        ax.set_xlim(args.xlim)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'$s$ [Mpc/$h$]', fontsize=9)
        ax.set_ylabel(r'$s^2\xi_\ell(s)$ [Mpc$^2h^{-2}$]', fontsize=9)
        ax.tick_params(labelsize=8)

    model_str = "  ".join(f"({r},{sl})" for r, sl in args.model_pairs)
    fig.suptitle(
        f"BGS DR2 data vs Uchuu mocks  (obs_seed={args.obs_seed})  "
        f"[ref, slope] = {model_str}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    if args.savefig:
        fig.savefig(args.savefig, dpi=150, bbox_inches='tight')
        log.info(f"Figure saved: {args.savefig}")
    else:
        plt.show()

    log.info("compare_clustering_vpeak_scatter.py complete.")


if __name__ == "__main__":
    main()
