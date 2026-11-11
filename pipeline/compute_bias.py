"""
compute_bias.py — Measure the linear bias from xi_0(s) for each BGS
magnitude-threshold sample and plot bias vs absolute magnitude.

Uses the Kaiser monopole model:
    xi_0(s) = b^2 * (1 + 2/3 beta + 1/5 beta^2) * xi_lin(s),
    beta = f / b,  f ~ Omega_m(z)^0.6

The mock xi_0 is rescaled to z=0 by dividing by D(z)^2 before fitting,
so it can be compared directly with xi_lin computed from the z=0 CAMB P(k).

Data xi_0 estimates are read from pycorr .npy files using the same path
helpers as S5PanelClustering1.py.  Both data and mock monopoles are fit.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pycorr import TwoPointCorrelationFunction
import fitsio

# nbodykit for cosmology and pk->xi transform; fall back to scipy if unavailable
try:
    from nbodykit.cosmology import Planck15
    from nbodykit.cosmology.correlation import pk_to_xi
    _HAVE_NBODYKIT = True
    print("Using nbodykit for cosmology and pk->xi.")
except ImportError:
    _HAVE_NBODYKIT = False
    print("nbodykit not found — using scipy fallback for cosmology and pk->xi.")

# ---------------------------------------------------------------------------
# Scipy fallback: growth factor, Omega_m(z), and pk->xi
# ---------------------------------------------------------------------------
if not _HAVE_NBODYKIT:
    from scipy.interpolate import InterpolatedUnivariateSpline

    def _E2(z, Om0, Ol0):
        """E^2(z) = (H(z)/H0)^2 for flat LCDM."""
        return Om0 * (1 + z)**3 + Ol0

    def _growth_factor_unnorm(z, Om0, Ol0):
        """Unnormalised linear growth factor D(z) via trapz on dense z-grid."""
        zp  = np.linspace(z, 1000.0, 50000)
        ep  = np.sqrt(_E2(zp, Om0, Ol0))
        val = np.trapezoid((1 + zp) / ep**3, zp)
        return np.sqrt(_E2(z, Om0, Ol0)) * val

    class _FlatLCDM:
        def __init__(self, h, Omega0_b, Omega0_cdm, sigma8):
            self.h      = h
            self.Om0    = Omega0_b + Omega0_cdm
            self.Ol0    = 1.0 - self.Om0
            self.sigma8 = sigma8
            self._D0    = _growth_factor_unnorm(0.0, self.Om0, self.Ol0)

        def scale_independent_growth_factor(self, z):
            return _growth_factor_unnorm(z, self.Om0, self.Ol0) / self._D0

        def Omega_m(self, z):
            E2 = _E2(z, self.Om0, self.Ol0)
            return self.Om0 * (1 + z)**3 / E2

    def pk_to_xi(k, pk, ell=0):
        """
        Compute xi_0(r) from P(k) via a dense log-space k-grid and trapz.
        Returns a callable xi(s) using spline interpolation.
        """
        assert ell == 0, "Only ell=0 supported in scipy fallback."
        # Interpolate P(k) onto a much denser log-k grid to resolve oscillations
        lnk_dense = np.linspace(np.log(k[0]), np.log(k[-1]), 8192)
        k_dense   = np.exp(lnk_dense)
        pk_spl    = InterpolatedUnivariateSpline(np.log(k), pk, ext=1)
        pk_dense  = pk_spl(lnk_dense)

        r_grid  = np.logspace(np.log10(0.1), np.log10(200), 300)
        xi_grid = np.empty_like(r_grid)
        for j, r in enumerate(r_grid):
            # integrand in ln(k): k^3 P(k) sin(kr)/(kr) / (2 pi^2)
            kr        = k_dense * r
            sinc_kr   = np.where(kr < 1e-6, 1.0, np.sin(kr) / kr)
            integrand = k_dense**3 * pk_dense * sinc_kr / (2.0 * np.pi**2)
            xi_grid[j] = np.trapezoid(integrand, lnk_dense)

        xi_spl = InterpolatedUnivariateSpline(np.log(r_grid), xi_grid, ext=1)
        return lambda s: xi_spl(np.log(np.asarray(s, dtype=float)))

# =============================================================================
# Configuration
# =============================================================================

PK_LIN_FILE = Path(
    "/global/cfs/cdirs/desi/users/smcole/DirectSHAM/camb_uchuu_matterpower_z0.dat"
)

OLD_MOCK_BASE = Path(
    "/global/cfs/cdirs/desi/users/smcole/OldDirectSHAM/clustering/mocks"
)
NEW_BASE = Path(
    "/global/cfs/cdirs/desi/users/smcole/DirectSHAM/6thOct"
)
DATA_CLUSTERING_DIR = NEW_BASE / "clustering" / "data"

OUT_STEM = Path(__file__).resolve().parent / "bias_vs_magnitude"

# The mock model we want bias for (default tuned catalogue)
SIG_LGV_REF   = 0.18
SIG_LGV_SLOPE = 0.30

DATA_NJACK = 60

# BGS data catalogues for computing median redshift
DATA_CATS = [
    NEW_BASE / "BGS_BRIGHT_6thOct_NGC.fits",
    NEW_BASE / "BGS_BRIGHT_6thOct_SGC.fits",
]

# Mock catalogues for halo mass and satellite fraction
MOCK_MODEL_DIR = (
    NEW_BASE / "N_neigh_lum50"
    / f"seed000_sig_lgv{0.18:.2f}_{0.30:.2f}_fc0.00_al0.00"
)
MOCK_CATS = [
    MOCK_MODEL_DIR / "mock_NGC.fits",
    MOCK_MODEL_DIR / "mock_SGC.fits",
]

# Fitting range in s [Mpc/h] — linear theory regime
S_MIN, S_MAX = 10.0, 40.0

# Uchuu cosmology
H         = 0.6774
OMEGA_B   = 0.0486
OMEGA_CDM = 0.2603
SIGMA8    = 0.8159
GROWTH_INDEX = 0.6   # f ~ Omega_m(z)^GROWTH_INDEX

# Samples — all 10 Mr thresholds; old=True reads from OldDirectSHAM,
# old=False from new DirectSHAM, old=None means no mock available
SAMPLES = [
    dict(mr_cut=-22.0, zmin=0.02, zmax=0.50, old=False),
    dict(mr_cut=-21.5, zmin=0.02, zmax=0.42, old=False),
    dict(mr_cut=-21.0, zmin=0.02, zmax=0.34, old=False),
    dict(mr_cut=-20.5, zmin=0.02, zmax=0.28, old=False),
    dict(mr_cut=-20.0, zmin=0.02, zmax=0.23, old=False),
    dict(mr_cut=-19.5, zmin=0.02, zmax=0.19, old=False),
    dict(mr_cut=-19.0, zmin=0.02, zmax=0.15, old=False),
    dict(mr_cut=-18.5, zmin=0.02, zmax=0.12, old=False),
    dict(mr_cut=-18.0, zmin=0.02, zmax=0.10, old=False),
    dict(mr_cut=-17.5, zmin=0.02, zmax=0.08, old=False),
]

# =============================================================================
# Cosmology and linear xi
# =============================================================================

if _HAVE_NBODYKIT:
    cosmo = Planck15.clone(h=H, Omega0_b=OMEGA_B, Omega0_cdm=OMEGA_CDM)
    cosmo = cosmo.match(sigma8=SIGMA8)
else:
    cosmo = _FlatLCDM(h=H, Omega0_b=OMEGA_B, Omega0_cdm=OMEGA_CDM, sigma8=SIGMA8)

k, pk   = np.loadtxt(PK_LIN_FILE, comments='#', unpack=True)
xi_lin  = pk_to_xi(k, pk, ell=0)   # callable xi_lin(s)

def xi0_model(s, b, f):
    """Kaiser monopole model."""
    beta = f / b
    return b**2 * (1.0 + 2.0/3.0 * beta + 1.0/5.0 * beta**2) * xi_lin(s)

def fit_bias(s, xi0, z_eff, xi0_err=None):
    """Fit bias from xi_0(s), rescaled to z=0, over [S_MIN, S_MAX].
    If xi0_err is provided (diagonal jackknife errors), the fit is weighted."""
    growth    = cosmo.scale_independent_growth_factor(z_eff)
    omega_m_z = cosmo.Omega_m(z_eff)
    f         = float(omega_m_z) ** GROWTH_INDEX

    keep = (s >= S_MIN) & (s <= S_MAX) & np.isfinite(xi0)
    s_fit  = s[keep]
    xi_fit = xi0[keep] / growth**2   # rescale to z=0

    sigma = xi0_err[keep] / growth**2 if xi0_err is not None else None

    model = lambda s, b: xi0_model(s, b, f)
    popt, pcov = curve_fit(model, s_fit, xi_fit, p0=[1.5],
                           sigma=sigma, absolute_sigma=True)
    b_fit = float(popt[0])
    b_err = float(np.sqrt(pcov[0, 0]))
    return b_fit, b_err

# Cache to avoid re-reading catalogues for every sample
_CAT_CACHE = None

def get_median_redshift(mr_cut, zmin, zmax, subset=''):
    """Return the median redshift of BGS galaxies passing Mr and z cuts."""
    global _CAT_CACHE
    if _CAT_CACHE is None:
        cols = ['Z', 'ABSMAG_RP1', 'REST_GMR_0P1']
        parts = []
        for cat in DATA_CATS:
            if not Path(cat).exists():
                print(f"  WARNING: catalogue not found: {cat}")
                continue
            d = fitsio.read(str(cat), columns=cols)
            parts.append(d)
        _CAT_CACHE = np.concatenate(parts)

    z   = _CAT_CACHE['Z']
    mr  = _CAT_CACHE['ABSMAG_RP1']
    gmr = _CAT_CACHE['REST_GMR_0P1']

    mask = (z >= zmin) & (z <= zmax) & (mr <= mr_cut)
    if subset == 'red':
        mask &= (gmr >= 0.6)
    elif subset == 'blue':
        mask &= (gmr < 0.6)

    if mask.sum() == 0:
        print(f"  WARNING: no galaxies found for median z computation, using midpoint")
        return 0.5 * (zmin + zmax)
    return float(np.median(z[mask]))

# Cache for mock catalogue halo properties
_MOCK_CACHE = None
_CENTRAL_PID = None   # sentinel value for centrals in pid column

def _load_mock_cache():
    global _MOCK_CACHE, _CENTRAL_PID
    if _MOCK_CACHE is not None:
        return
    cols = ['Zobs', 'ABSMAG_RP1', 'REST_GMR_0P1', 'Mhost', 'ID', 'pid']
    parts = []
    for cat in MOCK_CATS:
        if not Path(cat).exists():
            print(f"  WARNING: mock catalogue not found: {cat}")
            continue
        d = fitsio.read(str(cat), columns=cols)
        parts.append(d)
    _MOCK_CACHE = np.concatenate(parts)
    # Determine sentinel value for centrals: most common pid value
    pids, counts = np.unique(_MOCK_CACHE['pid'], return_counts=True)
    _CENTRAL_PID = pids[np.argmax(counts)]
    print(f"  Central sentinel pid = {_CENTRAL_PID} "
          f"(occurs {counts.max()} times out of {len(_MOCK_CACHE)})")

def get_halo_properties(mr_cut, zmin, zmax, subset=''):
    """Return (median Mhost centrals, median Mhost satellites, satellite fraction)
    for mock galaxies in sample."""
    _load_mock_cache()
    z   = _MOCK_CACHE['Zobs']
    mr  = _MOCK_CACHE['ABSMAG_RP1']
    gmr = _MOCK_CACHE['REST_GMR_0P1']
    pid = _MOCK_CACHE['pid']

    mask = (z >= zmin) & (z <= zmax) & (mr <= mr_cut)
    if subset == 'red':
        mask &= (gmr >= 0.6)
    elif subset == 'blue':
        mask &= (gmr < 0.6)

    if mask.sum() == 0:
        return np.nan, np.nan, np.nan

    mhost  = _MOCK_CACHE['Mhost'][mask]
    is_sat = pid[mask] != _CENTRAL_PID
    is_cen = ~is_sat

    median_mhost_cen = float(np.nanmedian(mhost[is_cen])) if is_cen.sum() > 0 else np.nan
    median_mhost_sat = float(np.nanmedian(mhost[is_sat])) if is_sat.sum() > 0 else np.nan
    sat_frac         = float(is_sat.sum() / mask.sum())
    n_sat = is_sat.sum()
    n_sat_nan = np.isnan(mhost[is_sat]).sum()
    n_cen_nan = np.isnan(mhost[is_cen]).sum()
    print(f"  Mhost NaN fraction: cen={n_cen_nan}/{is_cen.sum()} "
          f"({100*n_cen_nan/max(is_cen.sum(),1):.2f}%), "
          f"sat={n_sat_nan}/{n_sat} "
          f"({100*n_sat_nan/max(n_sat,1):.2f}%)")
    return median_mhost_cen, median_mhost_sat, sat_frac

# =============================================================================
# Path helpers (from S5PanelClustering1.py)
# =============================================================================

def data_file(mr_cut, zmin, zmax, subset=''):
    suffix     = f"_{subset}" if subset else ""
    sample_tag = f"Mr{mr_cut}_z{zmin:.2f}-{zmax:.2f}{suffix}"
    smu_dir    = DATA_CLUSTERING_DIR / sample_tag / "smu"
    pattern    = (
        f"allcounts_BGS_BRIGHT_GCcomb_{zmin}_{zmax}_pip_angular_bitwise"
        f"_log_njack{DATA_NJACK}_nran1_split20.npy"
    )
    return smu_dir / pattern

def mock_file_old(mr_cut, zmin, zmax, ref):
    model_dir  = OLD_MOCK_BASE / f"seed000_sig_lgv{ref:.2f}"
    sample_dir = model_dir / f"Mr{mr_cut}_z{zmin}-{zmax}"
    smu_dir    = sample_dir / "smu"
    pattern    = (
        f"allcounts_BGS_BRIGHT_GCcomb_{zmin}_{zmax}_default"
        f"_log_njack0_nran1_split20.npy"
    )
    return smu_dir / pattern

def mock_file_new(mr_cut, zmin, zmax, ref, slope=0.0, subset=''):
    suffix     = f"_{subset}" if subset else ""
    model_dir  = (
        NEW_BASE / "N_neigh_lum50"
        / f"seed000_sig_lgv{ref:.2f}_{slope:.2f}_fc0.00_al0.00"
    )
    sample_dir = model_dir / "clustering" / f"Mr{mr_cut}_z{zmin:.2f}-{zmax:.2f}{suffix}"
    smu_dir    = sample_dir / "smu"
    # Try both 0.10 and 0.1 style formatting for zmin/zmax
    for zmin_s, zmax_s in [
        (f"{zmin:.2f}", f"{zmax:.2f}"),
        (f"{zmin:.2f}".rstrip('0').rstrip('.'), f"{zmax:.2f}".rstrip('0').rstrip('.')),
    ]:
        p = smu_dir / (
            f"allcounts_BGS_BRIGHT_GCcomb_{zmin_s}_{zmax_s}_default"
            f"_log_njack0_nran1_split20.npy"
        )
        if p.exists():
            return p
    # Return the default form so the warning shows the attempted path
    return smu_dir / (
        f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.2f}_{zmax:.2f}_default"
        f"_log_njack0_nran1_split20.npy"
    )

def load_result(path):
    path = Path(path)
    if not path.exists():
        print(f"  WARNING: not found: {path}")
        return None
    return TwoPointCorrelationFunction.load(str(path))

def extract_monopole(result, return_err=False):
    """Return (s, xi0[, xi0_err]) arrays with finite values only.
    xi0_err is the diagonal jackknife error; None if no covariance available."""
    s = result.sep
    if s.ndim == 2:
        s = np.nanmean(s, axis=1)
    xi0_err = None
    try:
        xis, cov = result.get_corr(ells=(0,), return_cov=True)
        xi0 = np.atleast_1d(xis[0]).ravel()
        ns  = len(s)
        xi0_err = np.sqrt(np.diag(cov))[:ns]
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
# Main
# =============================================================================

def compute_bias_for_subset(subset=''):
    """Compute bias arrays for data and mock for a given subset ('', 'red', 'blue')."""
    mr_vals = []
    b_data = [];  b_data_err = []
    b_mock = [];  b_mock_err = []
    zeff_vals = []

    label = subset if subset else 'all'
    for sample in SAMPLES:
        mr   = sample['mr_cut']
        z0   = sample['zmin']
        z1   = sample['zmax']
        old  = sample['old']
        zeff = get_median_redshift(mr, z0, z1, subset=subset)
        zeff_vals.append(zeff)
        print(f"\n[{label}] Mr < {mr},  z=[{z0},{z1}],  z_eff(median)={zeff:.3f}")

        # --- Data ---
        dr = load_result(data_file(mr, z0, z1, subset=subset))
        if dr is not None:
            s_d, xi_d, xi_d_err = extract_monopole(dr, return_err=True)
            b, be = fit_bias(s_d, xi_d, zeff, xi0_err=xi_d_err)
            print(f"  Data  bias: {b:.3f} +/- {be:.3f}")
            b_data.append(b);    b_data_err.append(be)
        else:
            b_data.append(np.nan); b_data_err.append(np.nan)

        # --- Mock ---
        if old is None:
            b_mock.append(np.nan); b_mock_err.append(np.nan)
            mr_vals.append(mr)
            continue
        mf = mock_file_old(mr, z0, z1, SIG_LGV_REF) if old \
             else mock_file_new(mr, z0, z1, SIG_LGV_REF, SIG_LGV_SLOPE, subset=subset)
        mr_vals.append(mr)
        mr_r = load_result(mf)
        if mr_r is not None:
            s_m, xi_m = extract_monopole(mr_r, return_err=False)
            b, be = fit_bias(s_m, xi_m, zeff, xi0_err=None)
            print(f"  Mock  bias: {b:.3f} +/- {be:.3f}")
            b_mock.append(b);    b_mock_err.append(be)
        else:
            b_mock.append(np.nan); b_mock_err.append(np.nan)

    return (np.array(mr_vals),
            np.array(b_data),  np.array(b_data_err),
            np.array(b_mock),  np.array(b_mock_err),
            np.array(zeff_vals))


def main():
    SUBSETS = [
        dict(subset='',     data_color='black', mock_color='black', label='All'),
        dict(subset='red',  data_color='red',   mock_color='red',   label='Red'),
        dict(subset='blue', data_color='blue',  mock_color='blue',  label='Blue'),
    ]

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)

    results = {}
    for ss in SUBSETS:
        mr_arr, b_data_arr, b_data_err_arr, b_mock_arr, _, zeff_arr = \
            compute_bias_for_subset(subset=ss['subset'])
        results[ss['subset']] = dict(mr=mr_arr, b_data=b_data_arr,
                                     b_data_err=b_data_err_arr,
                                     b_mock=b_mock_arr, zeff=zeff_arr)
    for ss in SUBSETS:

        r = results[ss['subset']]
        ax.plot(r['mr'], r['b_mock'],
                's--', color=ss['mock_color'], ms=5, lw=1.2,
                label=rf"{ss['label']} mock")
        ax.errorbar(r['mr'], r['b_data'], yerr=r['b_data_err'],
                    fmt='o', color=ss['data_color'], ms=3.5, capsize=3, lw=1.2,
                    label=f"{ss['label']} data")

    ax.set_xlabel(r'$M_r$ threshold')
    ax.set_ylabel(r'Linear bias $b$')
    ax.set_ylim(0.8, 2.4)
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    for ext in ('pdf', 'jpg'):
        out = f"{OUT_STEM}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # LaTeX table: redshift ranges, median redshifts, halo mass, sat fraction
    # -------------------------------------------------------------------------
    r_all = results['']
    print("\n% LaTeX table of samples, redshift ranges, median redshifts, "
          "median host halo mass (centrals and satellites separately) and satellite fraction")
    print(r"\begin{tabular}{ccccccc}")
    print(r"\hline")
    print(r"$M_r$ & $z_{\rm min}$ & $z_{\rm max}$ & $\bar{z}$ & "
          r"$\log_{10}\tilde{M}_{\rm host}^{\rm cen}$ & "
          r"$\log_{10}\tilde{M}_{\rm host}^{\rm sat}$ & $f_{\rm sat}$ \\")
    print(r"\hline")
    for sample, zeff in zip(SAMPLES, r_all['zeff']):
        mr = sample['mr_cut']
        z0 = sample['zmin']
        z1 = sample['zmax']
        mcen, msat, fsat = get_halo_properties(mr, z0, z1, subset='')
        log_mcen = np.log10(mcen) if np.isfinite(mcen) else float('nan')
        log_msat = np.log10(msat) if np.isfinite(msat) else float('nan')
        print(rf"${mr:.1f}$ & ${z0:.2f}$ & ${z1:.2f}$ & ${zeff:.3f}$ & "
              rf"${log_mcen:.2f}$ & ${log_msat:.2f}$ & ${fsat:.3f}$ \\")
    print(r"\hline")
    print(r"\end{tabular}")


if __name__ == '__main__':
    main()
