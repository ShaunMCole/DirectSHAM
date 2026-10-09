"""
compare_clustering.py — Compare the two-point correlation function multipoles
of one or more mock catalogues against the BGS data measurement.

Produces a two-panel plot (ξ₀ and ξ₂ multiplied by s²) with the data shown
as black points and each mock as a coloured line.  Computes a χ² goodness-
of-fit score for each mock over a user-specified range of separations.

The data clustering file is the .npy output of measure_clustering.py (loaded
via TwoPointCorrelationFunction.load()).  Mock files are the .npy outputs of
measure_mock_clustering.py.  Both are pycorr native format.

Usage examples
--------------
# Compare two scatter values against data, s in [5, 50] Mpc/h:
python compare_clustering.py \\
    --data_file  /path/to/xi_data.npy \\
    --mock_files /path/to/xi_mock_NGC+SGC_seed000_sig_lgv0.04.npy \\
                 /path/to/xi_mock_NGC+SGC_seed000_sig_lgv0.10.npy \\
    --smin 5 --smax 50

# Use jackknife covariance from the data file:
python compare_clustering.py \\
    --data_file /path/to/xi_data.npy \\
    --mock_files /path/to/xi_mock*.npy \\
    --use_jackknife

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
        help="Minimum separation [Mpc/h] for χ² computation (default: 5.0)",
    )
    p.add_argument(
        "--smax", type=float, default=50.0,
        help="Maximum separation [Mpc/h] for χ² computation (default: 50.0)",
    )
    p.add_argument(
        "--use_jackknife", action="store_true",
        help=(
            "Use jackknife covariance from the data file for χ². "
            "If not set, assumes unit (diagonal) errors so χ² measures "
            "shape agreement only."
        ),
    )
    p.add_argument(
        "--ells", nargs="+", type=int, default=[0, 2],
        help="Multipoles to include in χ² (default: 0 2)",
    )
    p.add_argument(
        "--ylim", nargs=2, type=float, default=[-100, 140],
        help="y-axis limits for the plot (default: -100 140)",
    )
    p.add_argument(
        "--xlim", nargs=2, type=float, default=[0.1, 100],
        help="x-axis limits for the plot (default: 0.1 100)",
    )
    p.add_argument(
        "--savefig", type=str, default=None,
        help="Save the figure to this path instead of displaying it.",
    )
    p.add_argument(
        "--title", type=str, default=None,
        help="Plot title. Auto-generated from sample parameters if not set.",
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


def extract_multipoles(result, ells=(0, 2)):
    """
    Extract separation bins and multipoles from a pycorr result.

    Returns
    -------
    s : array
        Bin centres [Mpc/h].
    xi_dict : dict
        {ell: xi_ell array} for each requested multipole.
    good : bool array
        True where s is finite and positive.
    """
    s, xis = result(ells=tuple(ells), return_sep=True)
    if not isinstance(xis, (list, tuple)):
        xis = [xis]
    xi_dict = {ell: xi for ell, xi in zip(ells, xis)}
    good    = np.isfinite(s) & (s > 0)
    for xi in xi_dict.values():
        good &= np.isfinite(xi)
    return s, xi_dict, good


# =============================================================================
# χ² computation
# =============================================================================

def compute_chi2(s_dat, xi_dat, s_mock, xi_mock, ells, smin, smax,
                 cov=None, log=None):
    """
    Compute χ² between data and mock multipoles over the s range [smin, smax].

    If cov is None, assumes unit diagonal errors (χ² measures shape only).
    If cov is provided it should be the full covariance matrix over the
    selected s bins and multipoles, stacked in ell order.

    Parameters
    ----------
    s_dat, s_mock : array
        Separation bin centres for data and mock respectively.
    xi_dat, xi_mock : dict {ell: array}
        Multipoles for data and mock.
    ells : list of int
    smin, smax : float
    cov : 2D array or None

    Returns
    -------
    chi2 : float
    ndof : int
        Number of data points in the fit range × number of multipoles.
    """
    def _interp(s_from, xi_from, s_to):
        return np.interp(s_to, s_from, xi_from)

    # Select data s bins within the fit range.
    fit  = (s_dat >= smin) & (s_dat <= smax)
    s_fit = s_dat[fit]
    ndof  = fit.sum() * len(ells)

    # Stack residuals over multipoles.
    residuals = []
    for ell in ells:
        xi_d = xi_dat[ell][fit]
        xi_m = _interp(s_mock, xi_mock[ell], s_fit)
        residuals.append(xi_d - xi_m)
    delta = np.concatenate(residuals)

    if cov is None:
        chi2 = float(np.dot(delta, delta))
    else:
        chi2 = float(delta @ np.linalg.solve(cov, delta))

    if log:
        log.info(
            f"  χ²={chi2:.2f}  ndof={ndof}  χ²/ndof={chi2/ndof:.3f}  "
            f"s=[{smin},{smax}] Mpc/h  ells={ells}"
        )
    return chi2, ndof


# =============================================================================
# Plotting
# =============================================================================

def make_plot(s_dat, xi_dat, good_dat,
              mock_results, ells, xlim, ylim, title, savefig, log):
    """
    Two-panel figure: ξ₀ (left) and ξ₂ (right), both multiplied by s².
    Data shown as black symbols, mocks as coloured lines.
    """
    ell_styles = {0: '-',  2: '--', 4: ':'}
    ell_labels = {0: r'$\xi_0$', 2: r'$\xi_2$', 4: r'$\xi_4$'}

    ncols  = len(ells)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), sharey=False)
    if ncols == 1:
        axes = [axes]

    # Colour cycle for mocks.
    colours = cm.tab10(np.linspace(0, 0.9, max(len(mock_results), 1)))

    for ax, ell in zip(axes, ells):
        # Data.
        s_d  = s_dat[good_dat]
        xi_d = xi_dat[ell][good_dat]
        ax.plot(
            s_d, s_d**2 * xi_d,
            'ko', ms=5, zorder=10,
            label=r'Data $' + ell_labels[ell][1:-1] + r'$',
        )

        # Mocks.
        for (label, s_m, xi_m, good_m, chi2, ndof), col in zip(mock_results, colours):
            s_   = s_m[good_m]
            xi_  = xi_m[ell][good_m]
            chi2_label = f"{label}  χ²/ndof={chi2/ndof:.2f}" if ndof > 0 else label
            ax.plot(
                s_, s_**2 * xi_,
                lw=1.8, color=col,
                label=chi2_label,
            )

        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.set_xscale('log')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r'$s$ [Mpc/$h$]')
        ax.set_ylabel(r'$s^2 \xi_\ell(s)$  [Mpc$^2 h^{-2}$]')
        ax.set_title(f'{"Monopole" if ell==0 else "Quadrupole" if ell==2 else f"ℓ={ell}"}')
        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=12)

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
        f"ells={args.ells}  use_jackknife={args.use_jackknife}"
    )

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------
    log.info("=== Loading data clustering ===")
    data_result            = load_result(args.data_file, log)
    s_dat, xi_dat, good_dat = extract_multipoles(data_result, args.ells)

    # Jackknife covariance (optional).
    cov = None
    if args.use_jackknife:
        try:
            # pycorr stores jackknife samples; get_covs() returns the
            # covariance matrix for the requested multipoles over all s bins.
            cov_full = data_result.get_covs(ells=tuple(args.ells))
            # Restrict to the fit s range.
            fit     = (s_dat >= args.smin) & (s_dat <= args.smax)
            nb      = fit.sum()
            nell    = len(args.ells)
            # Rows/cols are ordered ell0_s0...ell0_sN, ell2_s0...ell2_sN, ...
            idx = np.concatenate([
                np.where(fit)[0] + i * len(s_dat) for i in range(nell)
            ])
            cov = cov_full[np.ix_(idx, idx)]
            log.info(f"Using jackknife covariance: {cov.shape}")
        except Exception as e:
            log.warning(f"Could not extract jackknife covariance: {e}")
            log.warning("Falling back to unit errors.")

    # -------------------------------------------------------------------------
    # Load mocks and compute χ²
    # -------------------------------------------------------------------------
    log.info("=== Loading mock clustering files ===")
    mock_results = []

    log.info(
        f"{'File':<55}  {'χ²':>8}  {'ndof':>5}  {'χ²/ndof':>8}"
    )
    log.info("-" * 82)

    for fpath in args.mock_files:
        result            = load_result(fpath, log)
        s_m, xi_m, good_m = extract_multipoles(result, args.ells)

        chi2, ndof = compute_chi2(
            s_dat, xi_dat, s_m, xi_m,
            ells=args.ells,
            smin=args.smin,
            smax=args.smax,
            cov=cov,
            log=None,
        )

        label = Path(fpath).stem
        log.info(f"{label:<55}  {chi2:8.2f}  {ndof:5d}  {chi2/ndof:8.3f}")
        mock_results.append((label, s_m, xi_m, good_m, chi2, ndof))

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    log.info("")
    log.info("=== χ² summary ===")
    log.info(
        f"s range: [{args.smin}, {args.smax}] Mpc/h  "
        f"ells: {args.ells}  "
        f"{'jackknife cov' if cov is not None else 'unit errors'}"
    )
    log.info(f"{'Mock':<55}  {'χ²':>8}  {'ndof':>5}  {'χ²/ndof':>8}")
    log.info("-" * 82)
    for label, _, _, _, chi2, ndof in mock_results:
        log.info(f"{label:<55}  {chi2:8.2f}  {ndof:5d}  {chi2/ndof:8.3f}")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    log.info("=== Making plot ===")
    title = args.title or (
        f"BGS data vs mocks  |  "
        f"s=[{args.smin},{args.smax}] Mpc/h  "
        f"ells={args.ells}"
    )
    make_plot(
        s_dat, xi_dat, good_dat,
        mock_results,
        ells=args.ells,
        xlim=args.xlim,
        ylim=args.ylim,
        title=title,
        savefig=args.savefig,
        log=log,
    )

    log.info("compare_clustering.py complete.")


if __name__ == "__main__":
    main()
