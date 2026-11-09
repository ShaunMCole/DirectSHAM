"""
compute_cigale_stats.py — Compute linear bias, median host halo masses and
satellite fractions for CIGALE stellar mass bin x redshift bin samples,
as requested by Shun Saito for the DESI KP HOD paper.

Samples are defined by windows in LOGM (CIGALE stellar mass) and redshift,
following Table 1 of the HOD paper draft.

The bias is fitted from xi_0(s) using the Kaiser monopole model:
    xi_0(s) = b^2 * (1 + 2/3 beta + 1/5 beta^2) * D(z)^2 * xi_lin(s, z=0)
where beta = f/b and f ~ Omega_m(z)^0.6.

Median host halo masses and satellite fractions are computed from the mock
catalogues (which carry Mhost and pid from the checkpoint files).

Output: a LaTeX table printed to stdout.

Usage
-----
# With clustering results available (full output including bias):
python compute_cigale_stats.py

# Skip bias fitting (useful while clustering runs are still in progress):
python compute_cigale_stats.py --no_bias
"""

import argparse
from pathlib import Path
import numpy as np
import fitsio
from scipy.optimize import curve_fit
from pycorr import TwoPointCorrelationFunction

# =============================================================================
# Cosmology and pk->xi (scipy fallback, no nbodykit needed)
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
    from scipy.interpolate import InterpolatedUnivariateSpline

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
            kr       = k_dense * r
            sinc_kr  = np.where(kr < 1e-6, 1.0, np.sin(kr) / kr)
            integrand = k_dense**3 * pk_dense * sinc_kr / (2.0 * np.pi**2)
            xi_grid[j] = np.trapezoid(integrand, lnk_dense)
        xi_spl = InterpolatedUnivariateSpline(np.log(r_grid), xi_grid, ext=1)
        return lambda s: xi_spl(np.log(np.asarray(s, dtype=float)))

# =============================================================================
# Configuration
# =============================================================================

NEW_BASE = Path(
    "/global/cfs/cdirs/desi/users/smcole/DirectSHAM/6thOct"
)
MODEL_DIR = (
    NEW_BASE / "N_neigh_lum50"
    / "seed000_sig_lgv0.18_0.30_fc0.00_al0.00"
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
S_MIN, S_MAX = 10.0, 40.0   # fitting range [Mpc/h]

# CIGALE samples from Shun's table (zmin, zmax, mstar_min, mstar_max)
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
# Cosmology setup
# =============================================================================

if _HAVE_NBODYKIT:
    cosmo = Planck15.clone(h=H, Omega0_b=OMEGA_B, Omega0_cdm=OMEGA_CDM)
    cosmo = cosmo.match(sigma8=SIGMA8)
else:
    cosmo = _FlatLCDM(h=H, Omega0_b=OMEGA_B, Omega0_cdm=OMEGA_CDM, sigma8=SIGMA8)

k, pk  = np.loadtxt(PK_LIN_FILE, comments='#', unpack=True)
xi_lin = pk_to_xi(k, pk, ell=0)

def xi0_model(s, b, f):
    beta = f / b
    return b**2 * (1.0 + 2.0/3.0*beta + 1.0/5.0*beta**2) * xi_lin(s)

def fit_bias(s, xi0, z_eff, xi0_err=None):
    growth    = cosmo.scale_independent_growth_factor(z_eff)
    omega_m_z = cosmo.Omega_m(z_eff)
    f         = float(omega_m_z) ** GROWTH_INDEX
    keep      = (s >= S_MIN) & (s <= S_MAX) & np.isfinite(xi0)
    s_fit     = s[keep]
    xi_fit    = xi0[keep] / growth**2
    sigma     = xi0_err[keep] / growth**2 if xi0_err is not None else None
    model     = lambda s, b: xi0_model(s, b, f)
    popt, pcov = curve_fit(model, s_fit, xi_fit, p0=[1.5],
                           sigma=sigma, absolute_sigma=True)
    return float(popt[0]), float(np.sqrt(pcov[0, 0]))

# =============================================================================
# Clustering file helpers
# =============================================================================

def sample_tag(zmin, zmax, mmin, mmax):
    return f"Mcig{mmin}-{mmax}_z{zmin:.3f}-{zmax:.3f}"

def data_clustering_file(zmin, zmax, mmin, mmax):
    tag     = sample_tag(zmin, zmax, mmin, mmax)
    smu_dir = DATA_CLUSTERING_BASE / tag / "smu"
    pattern = (
        f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.3f}_{zmax:.3f}"
        f"_pip_angular_bitwise_log_njack60_nran1_split20.npy"
    )
    return smu_dir / pattern

def mock_clustering_file(zmin, zmax, mmin, mmax):
    tag     = sample_tag(zmin, zmax, mmin, mmax)
    smu_dir = MOCK_CLUSTERING_BASE / tag / "smu"
    pattern = (
        f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.3f}_{zmax:.3f}"
        f"_default_log_njack0_nran1_split20.npy"
    )
    return smu_dir / pattern

def load_result(path):
    path = Path(path)
    if not path.exists():
        print(f"  WARNING: not found: {path}")
        return None
    return TwoPointCorrelationFunction.load(str(path))

def extract_monopole(result, return_err=False):
    s = result.sep
    if s.ndim == 2:
        s = np.nanmean(s, axis=1)
    xi0_err = None
    try:
        xis, cov = result.get_corr(ells=(0,), return_cov=True)
        xi0      = np.atleast_1d(xis[0]).ravel()
        xi0_err  = np.sqrt(np.diag(cov))[:len(s)]
    except Exception:
        xi0 = np.atleast_1d(result(ells=(0,))).ravel()
    ns = len(s)
    if len(xi0) > ns:
        xi0 = xi0[:ns]
    good = np.isfinite(s) & (s > 0) & np.isfinite(xi0)
    if return_err:
        err = xi0_err[good] if xi0_err is not None else None
        return s[good], xi0[good], err
    return s[good], xi0[good]

# =============================================================================
# Mock catalogue cache for halo properties
# =============================================================================

_MOCK_CACHE    = None
_CENTRAL_PID   = None

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
    """Return (median Mhost cen, median Mhost sat, sat fraction)."""
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
                   help='Skip bias fitting (use when clustering runs are not yet complete)')
    return p.parse_args()


def main():
    args = parse_args()

    print("\nLoading mock catalogues...")
    _load_mock_cache()

    rows = []
    for s in SAMPLES:
        zmin, zmax = s['zmin'], s['zmax']
        mmin, mmax = s['mmin'], s['mmax']
        tag = sample_tag(zmin, zmax, mmin, mmax)
        print(f"\n--- {tag} ---")

        # Median redshift and halo properties from mock
        zeff         = get_median_redshift(zmin, zmax, mmin, mmax)
        mcen, msat, fsat = get_halo_properties(zmin, zmax, mmin, mmax)

        b_data = b_data_err = np.nan
        b_mock = b_mock_err = np.nan

        if not args.no_bias:
            # Data bias
            dr = load_result(data_clustering_file(zmin, zmax, mmin, mmax))
            if dr is not None:
                s_d, xi_d, xi_d_err = extract_monopole(dr, return_err=True)
                b_data, b_data_err  = fit_bias(s_d, xi_d, zeff, xi0_err=xi_d_err)
                print(f"  Data  bias: {b_data:.3f} +/- {b_data_err:.3f}")
            else:
                print(f"  Data  bias: not available")

            # Mock bias
            mr = load_result(mock_clustering_file(zmin, zmax, mmin, mmax))
            if mr is not None:
                s_m, xi_m = extract_monopole(mr, return_err=False)
                b_mock, b_mock_err = fit_bias(s_m, xi_m, zeff)
                print(f"  Mock  bias: {b_mock:.3f} +/- {b_mock_err:.3f}")
            else:
                print(f"  Mock  bias: not available")

        log_mcen = np.log10(mcen) if np.isfinite(mcen) else np.nan
        log_msat = np.log10(msat) if np.isfinite(msat) else np.nan
        print(f"  z_eff={zeff:.3f}  log Mcen={log_mcen:.2f}  "
              f"log Msat={log_msat:.2f}  fsat={fsat:.3f}")

        rows.append(dict(zmin=zmin, zmax=zmax, mmin=mmin, mmax=mmax,
                         zeff=zeff, log_mcen=log_mcen, log_msat=log_msat,
                         fsat=fsat, b_data=b_data, b_data_err=b_data_err,
                         b_mock=b_mock, b_mock_err=b_mock_err))

    # -------------------------------------------------------------------------
    # LaTeX table
    # -------------------------------------------------------------------------
    print("\n\n% LaTeX table of CIGALE stellar mass bin samples")
    if args.no_bias:
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
                     if (r['zmin'], r['zmax']) != prev_z
                     else r"-")
            prev_z = (r['zmin'], r['zmax'])
            mcen_s = f"${r['log_mcen']:.2f}$" if np.isfinite(r['log_mcen']) else "-"
            msat_s = f"${r['log_msat']:.2f}$" if np.isfinite(r['log_msat']) else "-"
            print(f"{z_str} & ${r['mmin']:.1f}$--${r['mmax']:.1f}$ & "
                  f"{mcen_s} & {msat_s} & ${r['fsat']:.3f}$ & $1.00$ \\\\")
        print(r"\hline")
        print(r"\end{tabular}")
    else:
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
                     if (r['zmin'], r['zmax']) != prev_z
                     else r"-")
            prev_z = (r['zmin'], r['zmax'])
            b_m_s  = f"${r['b_mock']:.3f}$" if np.isfinite(r['b_mock']) else "-"
            mcen_s = f"${r['log_mcen']:.2f}$" if np.isfinite(r['log_mcen']) else "-"
            msat_s = f"${r['log_msat']:.2f}$" if np.isfinite(r['log_msat']) else "-"
            print(f"{z_str} & ${r['mmin']:.1f}$--${r['mmax']:.1f}$ & "
                  f"{mcen_s} & {msat_s} & ${r['fsat']:.3f}$ & {b_m_s} \\\\")
        print(r"\hline")
        print(r"\end{tabular}")


if __name__ == '__main__':
    main()
