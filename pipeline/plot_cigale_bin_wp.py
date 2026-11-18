"""
plot_cigale_bin_wp.py — Compare data and mock w(r_p) for CIGALE stellar mass
bin x redshift bin samples (Shun Saito KP HOD paper cells).

One panel per redshift bin (6 panels in a 3x2 grid), each showing w(r_p)
for every mass bin present in that redshift range.  Data as points with
jackknife error bars, mock as solid lines, same colour per mass bin.

π_max per redshift bin follows KP Table I:
  z=0.027–0.043 : 20 Mpc/h
  all others    : 40 Mpc/h

Run from ~/DirectSHAM/:
    python pipeline/plot_cigale_bin_wp.py
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

from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline

# =============================================================================
# Linear theory w(r_p) — same implementation as compute_cigale_stats_wp.py
# =============================================================================

PK_LIN_FILE = Path(
    "/global/cfs/cdirs/desi/users/smcole/DirectSHAM/camb_uchuu_matterpower_z0.dat"
)
H, OMEGA_B, OMEGA_CDM, SIGMA8 = 0.6774, 0.0486, 0.2603, 0.8159
GROWTH_INDEX = 0.6

def _E2(z, Om0, Ol0):
    return Om0 * (1 + z)**3 + Ol0

def _growth_factor_unnorm(z, Om0, Ol0):
    zp  = np.linspace(z, 1000.0, 50000)
    ep  = np.sqrt(_E2(zp, Om0, Ol0))
    val = np.trapezoid((1 + zp) / ep**3, zp)
    return np.sqrt(_E2(z, Om0, Ol0)) * val

class _FlatLCDM:
    def __init__(self, h, Omega0_b, Omega0_cdm):
        self.Om0 = Omega0_b + Omega0_cdm
        self.Ol0 = 1.0 - self.Om0
        self._D0 = _growth_factor_unnorm(0.0, self.Om0, self.Ol0)
    def scale_independent_growth_factor(self, z):
        return _growth_factor_unnorm(z, self.Om0, self.Ol0) / self._D0
    def Omega_m(self, z):
        return self.Om0 * (1 + z)**3 / _E2(z, self.Om0, self.Ol0)

_cosmo = _FlatLCDM(H, OMEGA_B, OMEGA_CDM)

def _build_xi_lin():
    k, pk = np.loadtxt(PK_LIN_FILE, comments='#', unpack=True)
    lnk   = np.linspace(np.log(k[0]), np.log(k[-1]), 8192)
    pk_s  = InterpolatedUnivariateSpline(np.log(k), pk, ext=1)(lnk)
    k_d   = np.exp(lnk)
    r_g   = np.logspace(np.log10(0.05), np.log10(300.), 2000)
    xi_g  = np.empty_like(r_g)
    for j, r in enumerate(r_g):
        kr  = k_d * r
        sc  = np.where(kr < 1e-6, 1.0, np.sin(kr) / kr)
        xi_g[j] = np.trapezoid(k_d**3 * pk_s * sc / (2*np.pi**2), lnk)
    spl = InterpolatedUnivariateSpline(np.log(r_g), xi_g, ext=1)
    return lambda r: spl(np.log(np.asarray(r, dtype=float)))

_xi_lin = _build_xi_lin()

def wp_lin(rp_arr, pimax, b, z_eff):
    """Compute b^2 * w_lin(rp, pimax) for an array of rp values."""
    D    = _cosmo.scale_independent_growth_factor(z_eff)
    Om_z = _cosmo.Omega_m(z_eff)
    f    = float(Om_z) ** GROWTH_INDEX
    # Kaiser factor for projected clustering (RSD cancel in projection to linear order)
    # but growth factor still applies: xi_lin(z) = D^2 * xi_lin(z=0)
    result = []
    for rp in rp_arr:
        r_max = np.sqrt(rp**2 + pimax**2)
        def integrand(r):
            return _xi_lin(r) * r / np.sqrt(r**2 - rp**2)
        val, _ = quad(integrand, rp * (1 + 1e-8), r_max, limit=200)
        result.append(2.0 * D**2 * b**2 * val)
    return np.array(result)

# =============================================================================
# Sample definitions — all 19 CIGALE cells
# =============================================================================

MASS_COLORS = {
    (8.5,  9.0):  '#7f7f7f',
    (9.0,  9.5):  '#8B4513',
    (9.5,  10.0): '#9467bd',
    (10.0, 10.5): '#1f77b4',
    (10.5, 11.0): '#2ca02c',
    (11.0, 12.0): '#d62728',
}

MASS_LABEL = {
    (8.5,  9.0):  r'$M_\star \in [8.5, 9.0)$',
    (9.0,  9.5):  r'$M_\star \in [9.0, 9.5)$',
    (9.5,  10.0): r'$M_\star \in [9.5, 10.0)$',
    (10.0, 10.5): r'$M_\star \in [10.0, 10.5)$',
    (10.5, 11.0): r'$M_\star \in [10.5, 11.0)$',
    (11.0, 12.0): r'$M_\star \in [11.0, 12.0)$',
}

ZBINS = [
    dict(zmin=0.027, zmax=0.043, pimax=20.0, mass_bins=[
        (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]),
    dict(zmin=0.043, zmax=0.068, pimax=40.0, mass_bins=[
        (9.0, 9.5), (9.5, 10.0), (10.0, 10.5), (10.5, 11.0)]),
    dict(zmin=0.068, zmax=0.107, pimax=40.0, mass_bins=[
        (9.5, 10.0), (10.0, 10.5), (10.5, 11.0), (11.0, 12.0)]),
    dict(zmin=0.107, zmax=0.162, pimax=40.0, mass_bins=[
        (10.0, 10.5), (10.5, 11.0), (11.0, 12.0)]),
    dict(zmin=0.162, zmax=0.240, pimax=40.0, mass_bins=[
        (10.5, 11.0), (11.0, 12.0)]),
    dict(zmin=0.240, zmax=0.344, pimax=40.0, mass_bins=[
        (11.0, 12.0)]),
]

DATA_NJACK = 60
SIG_REF    = DEFAULT_SIG_LGV_REF
SIG_SLOPE  = DEFAULT_SIG_LGV_SLOPE
OUT_STEM   = Path(__file__).resolve().parent / "cigale_bin_wp"
RP_MIN, RP_MAX = 2.0, 20.0   # fitting range for bias [Mpc/h]

def fit_bias_wp(rp, wp, pimax, z_eff):
    """Fit b from w(rp) = b^2 * w_lin(rp, pimax) over [RP_MIN, RP_MAX]."""
    from scipy.optimize import curve_fit
    keep = (rp >= RP_MIN) & (rp <= RP_MAX) & np.isfinite(wp) & (wp > 0)
    if keep.sum() < 3:
        return np.nan
    rp_fit = rp[keep]
    wp_fit = wp[keep]
    D    = _cosmo.scale_independent_growth_factor(z_eff)
    Om_z = _cosmo.Omega_m(z_eff)
    wlin = np.array([2 * D**2 * quad(
        lambda r: _xi_lin(r) * r / np.sqrt(r**2 - rp_i**2),
        rp_i * (1 + 1e-8), np.sqrt(rp_i**2 + pimax**2), limit=200)[0]
        for rp_i in rp_fit])
    good = wlin > 0
    if good.sum() < 3:
        return np.nan
    b2 = float(np.mean(wp_fit[good] / wlin[good]))
    return np.sqrt(max(b2, 0))

# =============================================================================
# Path helpers
# =============================================================================

def _find_npy(rppi_dir, stem):
    """Return path to .npy file; try exact name then with floats stripped."""
    import re
    p = rppi_dir / stem
    if p.exists():
        return p
    def strip(m):
        s = m.group(0)
        return s.rstrip('0').rstrip('.') if '.' in s else s
    stem2 = re.sub(r'\d+\.\d+', strip, stem)
    p2 = rppi_dir / stem2
    return p2 if p2.exists() else p

def data_npy(zmin, zmax, mmin, mmax):
    tag  = f"Mcig{mmin}-{mmax}_z{zmin:.3f}-{zmax:.3f}"
    rppi = DATA_DATE_DIR / "clustering" / "data" / tag / "rppi"
    stem = (f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.3f}_{zmax:.3f}"
            f"_pip_angular_bitwise_custom_njack{DATA_NJACK}_nran1_split20.npy")
    return _find_npy(rppi, stem)

def mock_npy(zmin, zmax, mmin, mmax, model_dir):
    tag  = f"Mcig{mmin}-{mmax}_z{zmin:.3f}-{zmax:.3f}"
    rppi = Path(model_dir) / "clustering" / tag / "rppi"
    stem = (f"allcounts_BGS_BRIGHT_GCcomb_{zmin:.3f}_{zmax:.3f}"
            f"_default_custom_njack0_nran1_split20.npy")
    return _find_npy(rppi, stem)

# =============================================================================
# Load / extract w(r_p)
# =============================================================================

def load_result(path):
    path = Path(path)
    if not path.exists():
        print(f"  WARNING: not found: {path}")
        return None
    return TwoPointCorrelationFunction.load(str(path))

def extract_wp_with_err(result, pimax):
    """
    Project ξ(r_p, π) → w(r_p) = 2 ∫₀^pimax ξ(r_p, π) dπ.
    Returns (rp_centres, wp, wp_err) where wp_err is from jackknife
    covariance if available, else None.
    """
    rp = result.sep  # shape (Nrp,) or (Nrp, Npi)
    if rp.ndim == 2:
        rp = np.nanmean(rp, axis=1)

    wp_err = None
    try:
        wp_arr, cov = result.get_corr(pimax=pimax, return_cov=True)
        wp = np.atleast_1d(wp_arr).ravel()
        wp_err = np.sqrt(np.diag(cov))[:len(rp)]
    except Exception:
        wp = np.atleast_1d(result(pimax=pimax)).ravel()

    ns = len(rp)
    if len(wp) > ns:
        wp = wp[:ns]

    good = np.isfinite(rp) & (rp > 0) & np.isfinite(wp)
    err  = wp_err[good] if wp_err is not None else None
    return rp[good], wp[good], err

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
        pimax      = zbin['pimax']
        handles    = []

        for (mmin, mmax) in zbin['mass_bins']:
            color = MASS_COLORS[(mmin, mmax)]
            label = MASS_LABEL[(mmin, mmax)]

            # --- Data ---
            dr = load_result(data_npy(zmin, zmax, mmin, mmax))
            if dr is not None:
                rp_d, wp_d, wp_err = extract_wp_with_err(dr, pimax)
                pos = (wp_d > 0) & (rp_d >= 2.0)
                if wp_err is not None:
                    ax.errorbar(rp_d[pos], wp_d[pos],
                                yerr=wp_err[pos], fmt='o', color=color,
                                ms=2.5, capsize=2, lw=0.8, zorder=10)
                else:
                    ax.plot(rp_d[pos], wp_d[pos], 'o', color=color,
                            ms=2.5, zorder=10)

            # --- Mock ---
            mr = load_result(mock_npy(zmin, zmax, mmin, mmax, model_dir))
            if mr is not None:
                rp_m, wp_m, _ = extract_wp_with_err(mr, pimax)
                pos_m = (wp_m > 0) & (rp_m >= 2.0)
                h, = ax.plot(rp_m[pos_m], wp_m[pos_m], '-', color=color,
                             lw=1.2, zorder=5, label=label)
                handles.append(h)

                # --- Linear theory overlay ---
                z_eff = 0.5 * (zmin + zmax)
                b_fit = fit_bias_wp(rp_m, wp_m, pimax, z_eff)
                if np.isfinite(b_fit):
                    rp_th = np.geomspace(0.1, 55, 60)
                    wp_th = wp_lin(rp_th, pimax, b_fit, z_eff)
                    pos_th = wp_th > 0
                    ax.plot(rp_th[pos_th], wp_th[pos_th], '--',
                            color=color, lw=0.8, alpha=0.6, zorder=4)
                    # Mark fitting range
                    fit_range = (rp_th >= RP_MIN) & (rp_th <= RP_MAX) & pos_th
                    ax.plot(rp_th[fit_range], wp_th[fit_range], '--',
                            color=color, lw=1.5, alpha=0.9, zorder=4)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(2.0, 55)
        ax.set_ylim(bottom=1.0)
        ax.set_ylabel(r'$w(r_p)\ [\mathrm{Mpc}/h]$', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(
            rf"${zmin:.3f} \leq z < {zmax:.3f}$"
            rf"$\quad \pi_{{\rm max}}={pimax:.0f}\ h^{{-1}}\mathrm{{Mpc}}$",
            fontsize=9)
        if handles:
            ax.legend(handles=handles, fontsize=6.5, frameon=False,
                      loc='upper right')

    for ax in axes_flat[len(ZBINS):]:
        ax.set_visible(False)

    axes_flat[-2].set_xlabel(r'$r_p\ [\mathrm{Mpc}/h]$', fontsize=9)
    axes_flat[-1].set_xlabel(r'$r_p\ [\mathrm{Mpc}/h]$', fontsize=9)

    fig.suptitle(
        r"CIGALE stellar mass bins: data (points), mock (solid), $b^2 w_{\rm lin}$ (dashed)"
        "\n"
        rf"$w(r_p)$, $\sigma_{{v,0}}={SIG_REF}$, slope$={SIG_SLOPE}$; "
        rf"dashed thick = fitting range $r_p \in [{RP_MIN:.0f},{RP_MAX:.0f}]$ Mpc/$h$",
        fontsize=9)

    for ext in ('.pdf', '.jpg'):
        out = f"{OUT_STEM}{ext}"
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
