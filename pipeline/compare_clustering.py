"""
compare_clustering.py — Compare the two-point correlation function multipoles
of one or more mock catalogues against the BGS data measurement.

Produces a single-panel plot showing ξ₀ and ξ₂ multiplied by s², with the
data shown as black circles (ξ₀) and grey squares (ξ₂) with jackknife error
bars if available, and each mock as a solid line (ξ₀) and dashed line (ξ₂)
in the same colour.  Computes a χ² goodness-of-fit score for each mock over
a user-specified range of separations.

Usage examples
--------------
# Compare one mock against data:
python compare_clustering.py \\
    --data_file  /path/to/xi_data.npy \\
    --mock_files /path/to/xi_mock.npy \\
    --smin 5 --smax 50

# Use jackknife covariance for chi2:
python compare_clustering.py \\
    --data_file /path/to/xi_data.npy \\
    --mock_files /path/to/xi_mock*.npy \\
    --use_jackknife

# Show individual jackknife realisations:
python compare_clustering.py ... --show_jackknife

# Save the plot to disk:
python compare_clustering.py ... --savefig comparison.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Pipeline modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_utils import setup_logging


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data_file", required=True,
        help="Path to the data clustering .npy file (pycorr format).",
    )
    p.add_argument(
        "--mock_files", nargs="+", required=True,
        help="One or more mock clustering .npy files to compare against data.",
    )
    p.add_argument(
        "--smin", type=float, default=5.0,
        help="Minimum separation [Mpc/h] for chi2 computation (default: 5.0)",
    )
    p.add_argument(
        "--smax", type=float, default=50.0,
        help="Maximum separation [Mpc/h] for chi2 computation (default: 50.0)",
    )
    p.add_argument(
        "--use_jackknife", action="store_true",
        help=(
            "Use jackknife diagonal errors from the data file for chi2. "
            "If not set, assumes unit errors so chi2 measures shape only."
        ),
    )
    p.add_argument(
        "--show_jackknife", action="store_true",
        help=(
            "Overplot the individual jackknife realisations as thin lines "
            "(black for xi0, grey for xi2). Useful for assessing error bar reliability."
        ),
    )
    p.add_argument(
        "--ells", nargs="+", type=int, default=[0, 2],
        help="Multipoles to include in chi2 (default: 0 2)",
    )
    p.add_argument(
        "--ylim", nargs=2, type=float, default=[-100, 120],
        help="y-axis limits for the plot (default: -100 120)",
    )
    p.add_argument(
        "--xlim", nargs=2, type=float, default=[0.1, 80],
        help="x-axis limits for the plot (default: 0.1 80)",
    )
    p.add_argument(
        "--savefig", type=str, default=None,
        help="Save the figure to this path instead of displaying it.",
    )
    p.add_argument(
        "--title", type=str, default=None,
        help="Plot title. Auto-generated from sample parameters if not set.",
    )
    p.add_argument(
        "--mr_cut", type=float, default=None,
        help="Absolute magnitude cut (for plot title, e.g. -21.5)",
    )
    p.add_argument(
        "--zmin", type=float, default=None,
        help="Minimum redshift (for plot title)",
    )
    p.add_argument(
        "--zmax", type=float, default=None,
        help="Maximum redshift (for plot title)",
    )
    return p.parse_args()


# =============================================================================
# Loading
# =============================================================================

def load_result(path, log):
    """Load a pycorr TwoPointCorrelationFunction result from a .npy file."""
    from pycorr import TwoPointCorrelationFunction
    path = Path(path)
    if not path.exists():
        log.error(f"File not found: {path}")
        sys.exit(1)
    result = TwoPointCorrelationFunction.load(str(path))
    log.info(f"Loaded: {path.name}")
    return result


def extract_multipoles(result, ells=(0, 2), get_realisations=False):
    """
    Extract separation bins, multipoles, errors and optionally jackknife
    realisations from a pycorr result.

    Returns
    -------
    s : 1D array
        Bin centres [Mpc/h].
    xi_dict : dict
        {ell: xi_ell array} for each requested multipole.
    err_dict : dict
        {ell: err_ell array} — jackknife errors if available, else zeros.
    jack_dict : dict or None
        {ell: array of shape (nreal, ns)} if get_realisations=True and
        jackknife data is available, else {ell: None}.
    good : bool array
        True where s is finite and positive and all multipoles are finite.
    has_cov : bool
        Whether jackknife covariance was successfully extracted.
    """
    s = result.sep
    if s.ndim == 2:
        s = np.nanmean(s, axis=1)

    try:
        xis, cov = result.get_corr(ells=tuple(ells), return_cov=True)
        xis  = np.atleast_2d(xis)
        ns   = xis.shape[1]
        errs = np.sqrt(np.diag(cov))
        err_dict = {ell: errs[i*ns:(i+1)*ns] for i, ell in enumerate(ells)}

        # Extract individual jackknife realisations if requested.
        jack_dict = {ell: None for ell in ells}
        if get_realisations:
            try:
                nreal = result.nrealizations
                for i, ell in enumerate(ells):
                    reals = []
                    for jr in range(nreal):
                        r    = result.realization(jr)
                        xi_r = np.atleast_2d(r(ells=tuple(ells)))
                        reals.append(xi_r[i])
                    jack_dict[ell] = np.array(reals)  # (nreal, ns)
            except Exception as e:
                pass  # realisations unavailable; jack_dict stays None

        has_cov = True

    except Exception:
        xis      = np.atleast_2d(result(ells=tuple(ells)))
        ns       = xis.shape[1]
        err_dict = {ell: np.zeros(ns) for ell in ells}
        jack_dict = {ell: None for ell in ells}
        has_cov  = False

    xi_dict = {ell: xis[i] for i, ell in enumerate(ells)}
    good    = np.isfinite(s) & (s > 0)
    for xi in xi_dict.values():
        good &= np.isfinite(xi)

    return s, xi_dict, err_dict, jack_dict, good, has_cov


# =============================================================================
# chi2 computation
# =============================================================================

def compute_chi2(s_dat, xi_dat, err_dat, s_mock, xi_mock, ells, smin, smax,
                 use_jackknife=False, log=None):
    """
    Compute chi2 between data and mock multipoles over the s range [smin, smax].

    If use_jackknife is True uses diagonal jackknife errors; otherwise unit errors.
    """
    def _interp(s_from, xi_from, s_to):
        return np.interp(s_to, s_from, xi_from)

    fit   = (s_dat >= smin) & (s_dat <= smax)
    s_fit = s_dat[fit]
    ndof  = fit.sum() * len(ells)

    residuals = []
    weights   = []
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

    if log:
        log.info(
            f"  chi2={chi2:.2f}  ndof={ndof}  chi2/ndof={chi2/ndof:.3f}  "
            f"s=[{smin},{smax}] Mpc/h  ells={ells}  "
            f"{'jackknife errors' if use_jackknife else 'unit errors'}"
        )
    return chi2, ndof


# =============================================================================
# Plotting
# =============================================================================

def make_plot(s_dat, xi_dat, err_dat, jack_dat, good_dat, has_cov,
              mock_results, ells, xlim, ylim, title, savefig,
              show_jackknife, log):
    """
    Single-panel figure showing xi0 and xi2 multiplied by s².

    Data:  xi0 as filled black circles, xi2 as filled grey squares,
           with jackknife error bars if available.
           Optionally overplot individual jackknife realisations as thin lines.
    Mocks: xi0 as solid line, xi2 as dashed line, one colour per mock.
    """
    data_styles = {
        0: ('o', 'black', r'Data $\xi_0$'),
        2: ('s', 'grey',  r'Data $\xi_2$'),
        4: ('^', 'dimgrey', r'Data $\xi_4$'),
    }
    jack_colours = {0: 'black', 2: 'grey', 4: 'dimgrey'}
    mock_ls      = {0: '-', 2: '--', 4: ':'}

    fig, ax = plt.subplots(figsize=(7, 5))

    s_d = s_dat[good_dat]

    # --- Jackknife realisations (drawn first, behind everything else) ---
    if show_jackknife and has_cov:
        for ell in ells:
            jk = jack_dat.get(ell)
            if jk is not None:
                col = jack_colours.get(ell, 'black')
                for jr in range(jk.shape[0]):
                    ax.plot(s_d, s_d**2 * jk[jr][good_dat],
                            color=col, lw=0.3, alpha=0.35, zorder=1)

    # --- Data points with error bars ---
    for ell in ells:
        fmt, col, lbl = data_styles.get(ell, ('o', 'black', f'Data xi_{ell}'))
        xi_d  = xi_dat[ell][good_dat]
        err_d = err_dat[ell][good_dat]
        y     = s_d**2 * xi_d
        yerr  = s_d**2 * err_d

        if has_cov and np.any(err_d > 0):
            ax.errorbar(s_d, y, yerr=yerr,
                        fmt=fmt, color=col, ms=5, capsize=2, lw=0.8,
                        zorder=10, label=lbl)
        else:
            ax.plot(s_d, y,
                    fmt, color=col, ms=5, zorder=10, label=lbl)

    # --- Mocks ---
    colours = cm.tab10(np.linspace(0, 0.9, max(len(mock_results), 1)))
    for (label, s_m, xi_m, good_m, chi2, ndof), col in zip(mock_results, colours):
        s_ = s_m[good_m]
        first_ell = True
        for ell in ells:
            mock_label = f"Mock $\\xi_{ell}$: {label}" if first_ell else f"Mock $\\xi_{ell}$"
            first_ell  = False
            ax.plot(s_, s_**2 * xi_m[ell][good_m],
                    lw=1.8, color=col, ls=mock_ls.get(ell, '-'),
                    label=mock_label)

    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'$s$ [Mpc/$h$]')
    ax.set_ylabel(r'$s^2 \xi_\ell(s)$  [Mpc$^2 h^{-2}$]')
    ax.legend(fontsize=7, loc='upper left')
    if title:
        ax.set_title(title, fontsize=11)

    plt.tight_layout()

    if savefig:
        fig.savefig(savefig, dpi=150, bbox_inches='tight')
        log.info(f"Figure saved: {savefig}")
    else:
        plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    log  = setup_logging("compare_clustering")

    log.info(
        f"data={args.data_file}  nmocks={len(args.mock_files)}  "
        f"smin={args.smin}  smax={args.smax}  "
        f"ells={args.ells}  use_jackknife={args.use_jackknife}  "
        f"show_jackknife={args.show_jackknife}"
    )

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    log.info("=== Loading data clustering ===")
    data_result = load_result(args.data_file, log)
    s_dat, xi_dat, err_dat, jack_dat, good_dat, has_cov = extract_multipoles(
        data_result, args.ells, get_realisations=args.show_jackknife
    )
    if has_cov:
        log.info("Jackknife covariance found — error bars will be shown.")
        if args.show_jackknife:
            nreal = next(
                (v.shape[0] for v in jack_dat.values() if v is not None), 0
            )
            log.info(f"Jackknife realisations extracted: {nreal}")
    else:
        log.info("No jackknife covariance — plotting without error bars.")

    # -------------------------------------------------------------------------
    # Load mocks and compute chi2
    # -------------------------------------------------------------------------
    log.info("=== Loading mock clustering files ===")
    mock_results = []

    log.info(f"{'File':<55}  {'chi2':>8}  {'ndof':>5}  {'chi2/ndof':>9}")
    log.info("-" * 82)

    for fpath in args.mock_files:
        result = load_result(fpath, log)
        s_m, xi_m, err_m, _, good_m, _ = extract_multipoles(result, args.ells)

        chi2, ndof = compute_chi2(
            s_dat, xi_dat, err_dat,
            s_m, xi_m,
            ells=args.ells,
            smin=args.smin,
            smax=args.smax,
            use_jackknife=args.use_jackknife and has_cov,
            log=None,
        )

        label = Path(fpath).stem
        import re
        m = re.search(r'seed(\d+)_sig_lgv([\d.]+)', str(fpath))
        if m:
            label = f"Obs:{m.group(1)} sig_lgv={m.group(2)}"

        log.info(f"{label:<55}  {chi2:8.2f}  {ndof:5d}  {chi2/ndof:9.3f}")
        mock_results.append((label, s_m, xi_m, good_m, chi2, ndof))

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    log.info("")
    log.info("=== chi2 summary ===")
    log.info(
        f"s range: [{args.smin}, {args.smax}] Mpc/h  "
        f"ells: {args.ells}  "
        f"{'jackknife diagonal errors' if (args.use_jackknife and has_cov) else 'unit errors'}"
    )
    log.info(f"{'Mock':<55}  {'chi2':>8}  {'ndof':>5}  {'chi2/ndof':>9}")
    log.info("-" * 82)
    for label, _, _, _, chi2, ndof in mock_results:
        log.info(f"{label:<55}  {chi2:8.2f}  {ndof:5d}  {chi2/ndof:9.3f}")

    # -------------------------------------------------------------------------
    # Plot title
    # -------------------------------------------------------------------------
    if args.title:
        title = args.title
    elif args.mr_cut is not None and args.zmin is not None and args.zmax is not None:
        title = (
            f"$M_r < {args.mr_cut}$  |  "
            f"${args.zmin} \\leq z \\leq {args.zmax}$"
        )
    else:
        title = (
            f"BGS data vs mocks  |  "
            f"s=[{args.smin},{args.smax}] Mpc/h  ells={args.ells}"
        )

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    log.info("=== Making plot ===")
    make_plot(
        s_dat, xi_dat, err_dat, jack_dat, good_dat, has_cov,
        mock_results,
        ells=args.ells,
        xlim=args.xlim,
        ylim=args.ylim,
        title=title,
        savefig=args.savefig,
        show_jackknife=args.show_jackknife,
        log=log,
    )

    log.info("compare_clustering.py complete.")


if __name__ == "__main__":
    main()
