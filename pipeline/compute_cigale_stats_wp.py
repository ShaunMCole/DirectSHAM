"""
compute_cigale_stats_wp.py — Compute linear bias from w(r_p) for CIGALE
stellar mass bin x redshift bin samples (Shun Saito KP HOD paper cells).

This is a companion to compute_cigale_stats.py, which fits bias from xi_0(s)
using the Kaiser monopole model.  Here the bias is estimated directly from
the projected correlation function:

    w(r_p) = b^2 * w_lin(r_p)

where w_lin(r_p) is the linear projected correlation function with the same
finite pi_max as the measurement:

    w_lin(r_p, pi_max) = 2 * integral_0^{pi_max} xi_lin(sqrt(r_p^2 + pi^2)) d pi
                       = 2 * integral_{r_p}^{sqrt(r_p^2+pi_max^2)} xi_lin(r) * r / sqrt(r^2 - r_p^2) dr

The second form (a change of variable r^2 = r_p^2 + pi^2) is used in the
implementation.  Using the same pi_max in the model as in the measurement is
important: the Abel transform with pi_max=infinity would overestimate w_lin,
biasing b low, with the error being largest at small pi_max (the z=0.027-0.043
bin with pi_max=20 Mpc/h would be most affected).

Crucially, redshift-space distortions cancel exactly in projection (to linear
order), so no Kaiser factor is needed and the estimator is simply
b^2 = w(r_p) / w_lin(r_p, pi_max).

The fit is performed over r_p in [RP_MIN, RP_MAX] Mpc/h where the linear
approximation is reliable and the signal-to-noise is good.  For data, the
jackknife covariance is used to weight the fit; for mocks (no jackknife),
an unweighted fit is used.

Median host halo masses and satellite fractions are identical to those in
compute_cigale_stats.py (read from the same mock catalogues).

Output: a LaTeX table printed to stdout (same format as compute_cigale_stats.py).

Usage
-----
python compute_cigale_stats_wp.py

# Skip bias fitting (useful while w(rp) runs are still in progress):
python compute_cigale_stats_wp.py --no_bias
"""

import argparse
from pathlib import Path
import numpy as np
import fitsio
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from pycorr import TwoPointCorrelationFunction

# =============================================================================
# Cosmology — same scipy fallback as compute_cigale_stats.py
# =============================================================================
try:
    from nbodykit.cosmology import Planck15
    from nbodykit.cosmology.correlation import pk_to_xi
    _HAVE_NBODYKIT = True
    print("Using nbodykit for cosmology and pk->xi.")
except ImportError:
    _HAVE_NBODYKIT = False
    print("nbodykit not found — using scipy fallback.")

if not _HAVE_NBODYKIT:
    def _E2(z, Om0, Ol0):
        return Om0 * (1 + z)**3 + Ol0

    def _growth_factor_unnorm(z, Om0, Ol0):
        zp  = np.linspace(z, 1000.0, 50000)
        ep  = np.sqrt(_E2(zp, Om0, Ol0))
        val = np.trapezoid((1 + zp) / ep**3, zp)
        return np.sqrt(_E2(z, Om0, Ol0)) * val

    class _FlatLCDM:
        def __init__(self, h, Omega0_b, Omega0_cdm, sigma8):
            self.Om0 = Omega0_b + Omega0_cdm
            self.Ol0 = 1.0 - self.Om0
            self._D0 = _growth_factor_unnorm(0.0, self.Om0, self.Ol0)

        def scale_independent_growth_factor(self, z):
            return _growth_factor_unnorm(z, self.Om0, self.Ol0) / self._D0

        def Omega_m(self, z):
            return self.Om0 * (1 + z)**3 / _E2(z, self.Om0, self.Ol0)

    def pk_to_xi(k, pk, ell=0):
        assert ell == 0
        lnk_dense = np.linspace(np.log(k[0]), np.log(k[-1]), 8192)
        k_dense   = np.exp(lnk_dense)
        pk_spl    = InterpolatedUnivariateSpline(np.log(k), pk, ext=1)
        pk_dense  = pk_spl(lnk_dense)
        r_grid    = np.logspace(np.log10(0.1), np.log10(200), 300)
        xi_grid   = np.empty_like(r_grid)
        for j, r in enumerate(r_grid):
            kr        = k_dense * r
            sinc_kr   = np.where(kr < 1e-6, 1.0, np.sin(kr) / kr)
            integrand = k_dense**3 * pk_dense * sinc_kr / (2.0 * np.pi**2)
            xi_grid[j] = np.trapezoid(integrand, lnk_dense)
        xi_spl = InterpolatedUnivariateSpline(np.log(r_grid), xi_grid, ext=1)
        return lambda s: xi_spl(np.log(np.asarray(s, dtype=float)))

# =============================================================================
# Configuration
# =============================================================================

NEW_BASE = Path(
    "/global/cfs/cdirs/desi/users/smcole/DirectSHAM/16thNov"
)
MODEL_DIR = (
    NEW_BASE / "N_neigh_lum50"
    / "seed000_sig_lgv0.24_0.42_fc0.00_al0.02"
)
DATA_CLUSTERING_BASE = NEW_BASE / "clustering" / "data"
MOCK_CLUSTERING_BASE = MODEL_DIR / "clustering"

PK_LIN_FILE = Path(
    "/global/cfs/cdirs/desi/users/smcole/DirectSHAM/camb_uchuu_matterpower_z0.dat"
)

MOCK_CATS = [
    MODEL_DIR / "mock_NGC.fits",
    MODEL_DIR / "mock_SGC.fits",
]

# Uchuu cosmology
H, OMEGA_B, OMEGA_CDM, SIGMA8 = 0.6774, 0.0486, 0.2603, 0.8159
GROWTH_INDEX = 0.6

# r_p fitting range [Mpc/h] — linear regime, good S/N
RP_MIN, RP_MAX = 2.0, 20.0

# pimax per redshift bin (KP Table I)
PIMAX = {
    (0.027, 0.043): 20.0,
    (0.043, 0.068): 40.0,
    (0.068, 0.107): 40.0,
    (0.107, 0.162): 40.0,
    (0.162, 0.240): 40.0,
    (0.240, 0.344): 40.0,
}

SAMPLES = [
    # z=0.027-0.043
    dict(zmin=0.027, zmax=0.043, mmin=8.5,  mmax=9.0),
    dict(zmin=0.027, zmax=0.043, mmin=9.0,  mmax=9.5),
    dict(zmin=0.027, zmax=0.043, mmin=9.5,  mmax=10.0),
    dict(zmin=0.027, zmax=0.043, mmin=10.0, mmax=10.5),
    dict(zmin=0.027, zmax=0.043, mmin=10.5, mmax=11.0),
    # z=0.043-0.068
    dict(zmin=0.043, zmax=0.068, mmin=9.0,  mmax=9.5),
    dict(zmin=0.043, zmax=0.068, mmin=9.5,  mmax=10.0),
    dict(zmin=0.043, zmax=0.068, mmin=10.0, mmax=10.5),
    dict(zmin=0.043, zmax=0.068, mmin=10.5, mmax=11.0),
    # z=0.068-0.107
    dict(zmin=0.068, zmax=0.107, mmin=9.5,  mmax=10.0),
    dict(zmin=0.068, zmax=0.107, mmin=10.0, mmax=10.5),
    dict(zmin=0.068, zmax=0.107, mmin=10.5, mmax=11.0),
    dict(zmin=0.068, zmax=0.107, mmin=11.0, mmax=12.0),
    # z=0.107-0.162
    dict(zmin=0.107, zmax=0.162, mmin=10.0, mmax=10.5),
    dict(zmin=0.107, zmax=0.162, mmin=10.5, mmax=11.0),
    dict(zmin=0.107, zmax=0.162, mmin=11.0, mmax=12.0),
    # z=0.162-0.240
    dict(zmin=0.162, zmax=0.240, mmin=10.5, mmax=11.0),
    dict(zmin=0.162, zmax=0.240, mmin=11.0, mmax=12.0),
    # z=0.240-0.344
    dict(zmin=0.240, zmax=0.344, mmin=11.0, mmax=12.0),
]

# =============================================================================
# Cosmology setup and linear w(r_p)
# =============================================================================

if _HAVE_NBODYKIT:
    cosmo = Planck15.clone(h=H, Omega0_b=OMEGA_B, Omega0_cdm=OMEGA_CDM)
    cosmo = cosmo.match(sigma8=SIGMA8)
else:
    cosmo = _FlatLCDM(h=H, Omega0_b=OMEGA_B, Omega0_cdm=OMEGA_CDM, sigma8=SIGMA8)

k, pk  = np.loadtxt(PK_LIN_FILE, comments='#', unpack=True)
xi_lin = pk_to_xi(k, pk, ell=0)

# Build a dense spline of xi_lin for fast Abel integration
_r_dense  = np.logspace(np.log10(0.05), np.log10(300.0), 2000)
_xi_dense = xi_lin(_r_dense)
_xi_spl   = InterpolatedUnivariateSpline(np.log(_r_dense), _xi_dense, ext=1)

def _xi_lin_fast(r):
    return _xi_spl(np.log(np.asarray(r, dtype=float)))


def _wp_lin_single(rp, pimax):
    """
    Compute w_lin(r_p, pi_max) with the same finite pi_max as the measurement:
        w_lin = 2 * integral_0^{pimax} xi_lin(sqrt(rp^2+pi^2)) d pi
              = 2 * integral_{rp}^{sqrt(rp^2+pimax^2)} xi_lin(r) * r / sqrt(r^2-rp^2) dr
    The second form (substitution r^2 = rp^2 + pi^2) is used here.
    Using infinite upper limit would overestimate w_lin and bias b low.
    """
    r_max = np.sqrt(rp**2 + pimax**2)
    # Change variable: r = rp * cosh(t), dr = rp * sinh(t) dt
    # but simple Gaussian quadrature is fine for smooth integrands.
    def integrand(r):
        return _xi_lin_fast(r) * r / np.sqrt(r**2 - rp**2)
    result, _ = quad(integrand, rp * (1 + 1e-6), r_max,
                     limit=200, epsrel=1e-5)
    return 2.0 * result


# Cache w_lin on the r_p grid for each pimax to avoid recomputation
_wp_lin_cache = {}

def wp_lin(rp_arr, pimax, z_eff):
    """
    Return w_lin(r_p) * D(z_eff)^2 on the array rp_arr.
    Results are cached per (pimax, z_eff) tuple.
    """
    growth = cosmo.scale_independent_growth_factor(z_eff)
    key    = (pimax, round(z_eff, 4))
    if key not in _wp_lin_cache:
        print(f"    Computing w_lin(r_p) for pimax={pimax}, z_eff={z_eff:.3f}...")
        _wp_lin_cache[key] = np.array(
            [_wp_lin_single(rp, pimax) for rp in rp_arr])
    return _wp_lin_cache[key] * growth**2


def fit_bias_wp(rp, wp, z_eff, pimax, wp_err=None):
    """
    Fit b^2 = w(r_p) / w_lin(r_p) over RP_MIN < r_p < RP_MAX.
    Returns (b, b_err).
    """
    keep   = (rp >= RP_MIN) & (rp <= RP_MAX) & (wp > 0) & np.isfinite(wp)
    rp_fit = rp[keep]
    wp_fit = wp[keep]
    sigma  = wp_err[keep] if wp_err is not None else None

    wl     = wp_lin(rp_fit, pimax, z_eff)
    model  = lambda rp, b: b**2 * wl
    popt, pcov = curve_fit(model, rp_fit, wp_fit, p0=[1.5],
                           sigma=sigma, absolute_sigma=(sigma is not None))
    return float(popt[0]), float(np.sqrt(pcov[0, 0]))


# =============================================================================
# Clustering file helpers
# =============================================================================

def _find_npy(directory, stem):
    """
    Return path to stem in directory.  If the exact name is missing, retry
    with trailing zeros stripped from floats — xirunpc.py sometimes writes
    e.g. '0.24' instead of '0.240'.
    """
    import re
    p = directory / stem
    if p.exists():
        return p
    def strip_zeros(m):
        s = m.group(0)
        return s.rstrip('0').rstrip('.') if '.' in s else s
    stem2 = re.sub(r'\d+\.\d+', strip_zeros, stem)
    p2 = directory / stem2
    return p2 if p2.exists() else p   # return original path if neither exists


def sample_tag(zmin, zmax, mmin, mmax):
    return f"Mcig{mmin}-{mmax}_z{zmin:.3f}-{zmax:.3f}"

def data_clustering_file(zmin, zmax, mmin, mmax):
    tag      = sample_tag(zmin, zmax, mmin, mmax)
    rppi_dir = DATA_CLUSTERING_BASE / tag / "rppi"
    stem     = (
        f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.3f}_{zmax:.3f}"
        f"_pip_angular_bitwise_custom_njack60_nran1_split20.npy"
    )
    return _find_npy(rppi_dir, stem)

def mock_clustering_file(zmin, zmax, mmin, mmax):
    tag      = sample_tag(zmin, zmax, mmin, mmax)
    rppi_dir = MOCK_CLUSTERING_BASE / tag / "rppi"
    stem     = (
        f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.3f}_{zmax:.3f}"
        f"_default_custom_njack0_nran1_split20.npy"
    )
    return _find_npy(rppi_dir, stem)

def load_result(path):
    path = Path(path)
    if not path.exists():
        print(f"  WARNING: not found: {path}")
        return None
    return TwoPointCorrelationFunction.load(str(path))

def extract_wp(result, pimax, return_err=False):
    """Extract w(r_p) and optionally jackknife errors from a pycorr result."""
    rp = result.sep
    if rp.ndim == 2:
        rp = np.nanmean(rp, axis=1)
    wp_err = None
    try:
        wp_arr, cov = result.get_corr(pimax=pimax, return_cov=True)
        wp     = np.atleast_1d(wp_arr).ravel()
        wp_err = np.sqrt(np.diag(cov))[:len(rp)]
    except Exception:
        wp = np.atleast_1d(result(pimax=pimax)).ravel()
    ns = len(rp)
    if len(wp) > ns:
        wp = wp[:ns]
    good = np.isfinite(rp) & (rp > 0) & np.isfinite(wp)
    if return_err:
        err = wp_err[good] if wp_err is not None else None
        return rp[good], wp[good], err
    return rp[good], wp[good]

# =============================================================================
# Mock catalogue cache (identical to compute_cigale_stats.py)
# =============================================================================

_MOCK_CACHE  = None
_CENTRAL_PID = None

def _load_mock_cache():
    global _MOCK_CACHE, _CENTRAL_PID
    if _MOCK_CACHE is not None:
        return
    cols  = ['Zobs', 'LOGM', 'Mhost', 'pid']
    parts = []
    for cat in MOCK_CATS:
        if not Path(cat).exists():
            print(f"  WARNING: mock catalogue not found: {cat}")
            continue
        parts.append(fitsio.read(str(cat), columns=cols))
    _MOCK_CACHE = np.concatenate(parts)
    pids, counts = np.unique(_MOCK_CACHE['pid'], return_counts=True)
    _CENTRAL_PID = pids[np.argmax(counts)]
    print(f"  Central sentinel pid = {_CENTRAL_PID} "
          f"({counts.max():,} centrals out of {len(_MOCK_CACHE):,} total)")

def get_median_redshift(zmin, zmax, mmin, mmax):
    _load_mock_cache()
    mask = ((_MOCK_CACHE['Zobs'] >= zmin) & (_MOCK_CACHE['Zobs'] <= zmax) &
            (_MOCK_CACHE['LOGM']  >  mmin) & (_MOCK_CACHE['LOGM']  <= mmax))
    if mask.sum() == 0:
        return 0.5 * (zmin + zmax)
    return float(np.median(_MOCK_CACHE['Zobs'][mask]))

def get_halo_properties(zmin, zmax, mmin, mmax):
    _load_mock_cache()
    mask = ((_MOCK_CACHE['Zobs'] >= zmin) & (_MOCK_CACHE['Zobs'] <= zmax) &
            (_MOCK_CACHE['LOGM']  >  mmin) & (_MOCK_CACHE['LOGM']  <= mmax))
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan
    mhost  = _MOCK_CACHE['Mhost'][mask]
    is_sat = _MOCK_CACHE['pid'][mask] != _CENTRAL_PID
    is_cen = ~is_sat
    n_sat_nan = np.isnan(mhost[is_sat]).sum()
    if n_sat_nan > 0:
        print(f"  Mhost NaN in satellites: {n_sat_nan}/{is_sat.sum()} "
              f"({100*n_sat_nan/max(is_sat.sum(),1):.2f}%) — using nanmedian")
    mcen = float(np.nanmedian(mhost[is_cen])) if is_cen.sum() > 0 else np.nan
    msat = float(np.nanmedian(mhost[is_sat])) if is_sat.sum() > 0 else np.nan
    fsat = float(is_sat.sum() / mask.sum())
    return mcen, msat, fsat

# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--no_bias', action='store_true',
                   help='Skip bias fitting (use when w(rp) runs are not yet complete)')
    return p.parse_args()


def main():
    args = parse_args()

    print("\nLoading mock catalogues...")
    _load_mock_cache()

    rows = []
    for s in SAMPLES:
        zmin, zmax = s['zmin'], s['zmax']
        mmin, mmax = s['mmin'], s['mmax']
        pimax      = PIMAX[(zmin, zmax)]
        tag        = sample_tag(zmin, zmax, mmin, mmax)
        print(f"\n--- {tag}  pimax={pimax:.0f} ---")

        zeff             = get_median_redshift(zmin, zmax, mmin, mmax)
        mcen, msat, fsat = get_halo_properties(zmin, zmax, mmin, mmax)

        b_data = b_data_err = np.nan
        b_mock = b_mock_err = np.nan

        if not args.no_bias:
            # Data bias
            dr = load_result(data_clustering_file(zmin, zmax, mmin, mmax))
            if dr is not None:
                rp_d, wp_d, wp_d_err = extract_wp(dr, pimax, return_err=True)
                b_data, b_data_err   = fit_bias_wp(rp_d, wp_d, zeff, pimax,
                                                    wp_err=wp_d_err)
                print(f"  Data  bias: {b_data:.3f} +/- {b_data_err:.3f}")
            else:
                print("  Data  bias: not available")

            # Mock bias
            mr = load_result(mock_clustering_file(zmin, zmax, mmin, mmax))
            if mr is not None:
                rp_m, wp_m = extract_wp(mr, pimax, return_err=False)
                b_mock, b_mock_err = fit_bias_wp(rp_m, wp_m, zeff, pimax)
                print(f"  Mock  bias: {b_mock:.3f} +/- {b_mock_err:.3f}")
            else:
                print("  Mock  bias: not available")

        log_mcen = np.log10(mcen) if np.isfinite(mcen) else np.nan
        log_msat = np.log10(msat) if np.isfinite(msat) else np.nan
        print(f"  z_eff={zeff:.3f}  log Mcen={log_mcen:.2f}  "
              f"log Msat={log_msat:.2f}  fsat={fsat:.3f}")

        rows.append(dict(zmin=zmin, zmax=zmax, mmin=mmin, mmax=mmax,
                         zeff=zeff, log_mcen=log_mcen, log_msat=log_msat,
                         fsat=fsat, b_data=b_data, b_data_err=b_data_err,
                         b_mock=b_mock, b_mock_err=b_mock_err))

    # -------------------------------------------------------------------------
    # LaTeX table (same format as compute_cigale_stats.py)
    # -------------------------------------------------------------------------
    print("\n\n% LaTeX table of CIGALE stellar mass bin samples (bias from w(rp))")
    print(r"\setlength{\tabcolsep}{5pt}")
    print(r"\begin{tabular}{cccccc}")
    print(r"\hline")
    print(r"$z_{\rm min} - z_{\rm max}$ & $\log M_\star$ range & "
          r"$\log \tilde{M}_{\rm host}^{\rm cen}$ & "
          r"$\log \tilde{M}_{\rm host}^{\rm sat}$ & "
          r"$f_{\rm sat}$ & $b$ \\")
    print(r"\hline")
    prev_z = None
    for r in rows:
        z_str = (f"${r['zmin']:.3f}$--${r['zmax']:.3f}$"
                 if (r['zmin'], r['zmax']) != prev_z else r"-")
        prev_z = (r['zmin'], r['zmax'])
        mcen_s = f"${r['log_mcen']:.2f}$" if np.isfinite(r['log_mcen']) else "-"
        msat_s = f"${r['log_msat']:.2f}$" if np.isfinite(r['log_msat']) else "-"
        if args.no_bias:
            b_s = r"$1.00$"
        else:
            b_s = f"${r['b_mock']:.3f}$" if np.isfinite(r['b_mock']) else "-"
        print(f"{z_str} & ${r['mmin']:.1f}$--${r['mmax']:.1f}$ & "
              f"{mcen_s} & {msat_s} & ${r['fsat']:.3f}$ & {b_s} \\\\")
    print(r"\hline")
    print(r"\end{tabular}")


if __name__ == '__main__':
    main()
