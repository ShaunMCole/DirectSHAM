"""
plot_cumulative_nM.py
====================
Two-panel plot for comparing mock galaxy number densities with the Uchuu
halo mass function, to validate the median halo masses in the KP table.

Left panel:  Cumulative n(>M*) for mock galaxies in the redshift slice,
             split into all / centrals / satellites vs CIGALE stellar mass.

Right panel: Cumulative Uchuu HMF n(>Mhost) for centrals in the same slice
             (from checkpoint file, real-space redshift Zu, Vpeak>70 km/s).
             Horizontal lines mark n_cen for each KP stellar mass bin.
             Vertical dotted lines mark our median Mhost_cen from the table.
             Agreement between where the horizontal line meets the HMF curve
             and the vertical dotted line confirms self-consistency.

Usage
-----
python pipeline/plot_cumulative_nM.py --zmin 0.068 --zmax 0.107
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))
import config
from pipeline_utils import make_paths, setup_logging

cosmo = FlatLambdaCDM(H0=100, Om0=0.313)

# KP bin definitions
KP_BINS = [(9.5, 10.0), (10.0, 10.5), (10.5, 11.0), (11.0, 12.0)]

BIN_COLORS = plt.cm.tab10(np.linspace(0, 0.5, len(KP_BINS)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zmin", type=float, default=0.068)
    p.add_argument("--zmax", type=float, default=0.107)
    p.add_argument("--obs_seed",      type=int,   default=0)
    p.add_argument("--sig_lgv_ref",   type=float, default=config.DEFAULT_SIG_LGV_REF)
    p.add_argument("--sig_lgv_slope", type=float, default=config.DEFAULT_SIG_LGV_SLOPE)
    p.add_argument("--f_colour",      type=float, default=config.DEFAULT_F_COLOUR)
    p.add_argument("--alpha",         type=float, default=config.DEFAULT_ALPHA)
    p.add_argument("--cat", type=Path, default=None,
                   help="Augmented BGS data catalogue for data SMF overplot "
                        "(e.g. data/16thNov/BGS_BRIGHT_16thNov_augmented.fits)")
    p.add_argument("--outfile", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logging("plot_cumulative_nM")

    paths = make_paths(obs_seed=args.obs_seed,
                       sig_lgv_ref=args.sig_lgv_ref,
                       sig_lgv_slope=args.sig_lgv_slope,
                       f_colour=args.f_colour,
                       alpha=args.alpha)

    # -------------------------------------------------------------------------
    # Load mock galaxies
    # -------------------------------------------------------------------------
    tables = []
    fsky = 0.0
    for reg in ("NGC", "SGC"):
        mock_path = Path(paths[f"mock_{reg}"])
        if not mock_path.exists():
            log.warning("Not found: %s", mock_path)
            continue
        log.info("Reading %s ...", mock_path)
        t = Table.read(str(mock_path), format="fits")
        cols = [c for c in ["Zobs", "LOGM", "pid", "Mhost"] if c in t.colnames]
        tables.append(t[cols])
        ran_path = Path(paths[f"formock_ran_{reg}"])
        if ran_path.exists():
            from astropy.io.fits import getheader
            fsky += getheader(str(ran_path), ext=1).get("FSKY", 0.0)

    if not tables:
        raise RuntimeError("No mock catalogues found.")
    data = vstack(tables)
    log.info("Combined: %d rows, FSKY=%.6f", len(data), fsky)

    zobs  = np.asarray(data["Zobs"],  dtype=float)
    logm  = np.asarray(data["LOGM"],  dtype=float)
    pid   = np.asarray(data["pid"],   dtype=int)
    mhost = np.asarray(data["Mhost"], dtype=float)
    zmask = (zobs >= args.zmin) & (zobs <= args.zmax)
    zobs  = zobs[zmask]; logm = logm[zmask]
    pid   = pid[zmask];  mhost = mhost[zmask]
    is_cen = pid == -1
    is_sat = ~is_cen

    # Compute median Mhost for centrals in each KP bin (same method as compute_cigale_stats_wp.py)
    MHOST_CEN = {}
    for mlo, mhi in KP_BINS:
        mask_bin = (logm >= mlo) & (logm < mhi) & is_cen
        if mask_bin.sum() > 0:
            MHOST_CEN[(mlo, mhi)] = np.log10(float(np.nanmedian(mhost[mask_bin])))
        else:
            MHOST_CEN[(mlo, mhi)] = np.nan
        log.info("Bin [%.1f,%.1f): median Mhost_cen = %.2f (n=%d)",
                 mlo, mhi, MHOST_CEN[(mlo, mhi)], mask_bin.sum())
    log.info("After z cut: %d galaxies (%d cen, %d sat)",
             len(zobs), is_cen.sum(), is_sat.sum())

    # Comoving volume
    dc = cosmo.comoving_distance([args.zmin, args.zmax]).value
    vol = (4.0 * np.pi * fsky / 3.0) * (dc[1]**3 - dc[0]**3)
    log.info("Shell volume: %.3e (Mpc/h)^3", vol)

    # Cumulative stellar mass function
    mstar_grid = np.linspace(8.0, 12.5, 300)
    n_all = np.array([(logm >= m).sum() / vol for m in mstar_grid])
    n_cen = np.array([(logm[is_cen] >= m).sum() / vol for m in mstar_grid])
    n_sat = np.array([(logm[is_sat] >= m).sum() / vol for m in mstar_grid])

    # Number density of centrals in each KP bin
    n_cen_bins = {}
    for mlo, mhi in KP_BINS:
        mask = (logm >= mlo) & (logm < mhi) & is_cen
        n_cen_bins[(mlo, mhi)] = mask.sum() / vol

    # -------------------------------------------------------------------------
    # Optionally load data catalogue for SMF comparison
    # -------------------------------------------------------------------------
    logm_data = None
    if args.cat is not None and args.cat.exists():
        log.info("Reading data catalogue %s ...", args.cat)
        from astropy.io.fits import getheader
        tdat = Table.read(str(args.cat), format="fits")
        zdat = np.asarray(tdat["Z"],    dtype=float)
        lmdat = np.asarray(tdat["LOGM"], dtype=float)
        wdat  = np.asarray(tdat["WEIGHT"], dtype=float) \
                if "WEIGHT" in tdat.colnames else None
        dzmask = (zdat >= args.zmin) & (zdat <= args.zmax)
        logm_data = lmdat[dzmask]
        wdat_cut  = wdat[dzmask] if wdat is not None else None
        # Data volume: use FSKY from data file header
        hdr = getheader(str(args.cat), ext=1)
        fsky_dat = hdr.get("AREA_N", 0.0) + hdr.get("AREA_S", 0.0)
        dc_dat = cosmo.comoving_distance([args.zmin, args.zmax]).value
        vol_dat = (4.0 * np.pi * fsky_dat / 3.0) * (dc_dat[1]**3 - dc_dat[0]**3)
        log.info("Data: %d galaxies in z-slice, FSKY=%.6f, vol=%.3e",
                 dzmask.sum(), fsky_dat, vol_dat)
        n_data_smf = np.array([
            (np.sum(wdat_cut[logm_data >= m]) / vol_dat
             if wdat_cut is not None
             else (logm_data >= m).sum() / vol_dat)
            for m in mstar_grid])
    else:
        vol_dat = vol
        n_data_smf = None

    # -------------------------------------------------------------------------
    # Load Uchuu HMF from checkpoint (centrals, real-space Zu)
    # -------------------------------------------------------------------------
    chk_tables = []
    for reg in ("N", "S"):
        chk_path = Path(paths["model_dir"]) / \
                   f"subhalos_{reg}_70.0_N_neigh_vpeak1000.fits"
        if not chk_path.exists():
            log.warning("Checkpoint not found: %s", chk_path)
            continue
        log.info("Reading checkpoint %s ...", chk_path)
        t = Table.read(str(chk_path), format="fits")
        cols = [c for c in ["Mhost", "pid", "Zu"] if c in t.colnames]
        chk_tables.append(t[cols])

    if not chk_tables:
        raise RuntimeError("No checkpoint files found.")
    chk = vstack(chk_tables)
    chk_pid = np.asarray(chk["pid"],   dtype=int)
    chk_mh  = np.asarray(chk["Mhost"], dtype=float)
    chk_zu  = np.asarray(chk["Zu"],    dtype=float)
    chk_cen = (chk_pid == -1) & (chk_zu >= args.zmin) & (chk_zu <= args.zmax)
    mhost_arr = chk_mh[chk_cen]
    log.info("Checkpoint centrals in z-slice: %d", chk_cen.sum())

    # Cumulative HMF — all checkpoint centrals (includes unoccupied haloes)
    log_mhost_grid = np.linspace(10.5, 15.0, 300)
    n_hmf = np.array([(mhost_arr >= 10**m).sum() / vol
                      for m in log_mhost_grid])

    # Occupied HMF — Mhost of mock centrals only (from mock catalogue)
    mhost_occ = mhost[is_cen]   # Mhost of central galaxies in z-slice
    n_hmf_occ = np.array([(mhost_occ >= 10**m).sum() / vol
                           for m in log_mhost_grid])

    # -------------------------------------------------------------------------
    # Print summary table
    # -------------------------------------------------------------------------
    print(f"\n{'Bin':<15} {'log n_all':>10} {'log n_cen':>10} {'log n_sat':>10} "
          f"{'f_sat':>7} {'Mhost(table)':>13} {'Mhost(HMF)':>11} {'Mhost(occ)':>11}")
    print("-" * 95)
    for mlo, mhi in KP_BINS:
        mask_bin = (logm >= mlo) & (logm < mhi)
        n_b   = mask_bin.sum() / vol
        n_bc  = (mask_bin & is_cen).sum() / vol
        n_bs  = (mask_bin & is_sat).sum() / vol
        fsat  = (mask_bin & is_sat).sum() / max(mask_bin.sum(), 1)
        mhost_table = MHOST_CEN.get((mlo, mhi), np.nan)
        nc = n_bc
        # Where full HMF crosses n_cen
        if nc > 0 and (n_hmf > nc).any() and (n_hmf < nc).any():
            mhost_hmf = log_mhost_grid[np.argmin(np.abs(n_hmf - nc))]
        else:
            mhost_hmf = np.nan
        # Where occupied HMF crosses n_cen
        if nc > 0 and (n_hmf_occ > nc).any() and (n_hmf_occ < nc).any():
            mhost_occ_val = log_mhost_grid[np.argmin(np.abs(n_hmf_occ - nc))]
        else:
            mhost_occ_val = np.nan
        print(f"{mlo:.1f}-{mhi:.1f}{'':5} "
              f"{np.log10(n_b) if n_b>0 else np.nan:10.3f} "
              f"{np.log10(n_bc) if n_bc>0 else np.nan:10.3f} "
              f"{np.log10(n_bs) if n_bs>0 else np.nan:10.3f} "
              f"{fsat:7.3f} {mhost_table:13.2f} {mhost_hmf:11.2f} {mhost_occ_val:11.2f}")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)

    # --- Left panel: cumulative stellar mass function ---
    good = n_all > 0
    ax1.plot(mstar_grid[good], np.log10(n_all[good]), 'k-',  lw=1.5, label='Mock (all)')
    good = n_cen > 0
    ax1.plot(mstar_grid[good], np.log10(n_cen[good]), 'b--', lw=1.2, label='Mock (centrals)')
    good = n_sat > 0
    ax1.plot(mstar_grid[good], np.log10(n_sat[good]), 'r:',  lw=1.2, label='Mock (satellites)')
    if n_data_smf is not None:
        good = n_data_smf > 0
        ax1.plot(mstar_grid[good], np.log10(n_data_smf[good]),
                 'k--', lw=1.5, alpha=0.5, label='Data (all)')

    for (mlo, mhi), col in zip(KP_BINS, BIN_COLORS):
        ax1.axvline(mlo, color='grey', lw=0.6, ls='--', alpha=0.5)
    ax1.axvline(12.0, color='grey', lw=0.6, ls='--', alpha=0.5)

    ax1.set_xlabel(r'$\log_{10}(M_\star/M_\odot)$ [CIGALE]')
    ax1.set_ylabel(r'$\log_{10}\,n(>M_\star)\ [h^3\,\mathrm{Mpc}^{-3}]$')
    ax1.set_title(rf'Galaxy SMF: ${args.zmin:.3f} \leq z \leq {args.zmax:.3f}$')
    ax1.set_xlim(8.0, 12.5)
    ax1.set_ylim(-7, -1)
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(alpha=0.2)

    # --- Right panel: Uchuu HMF with n_cen markers ---
    good_hmf = n_hmf > 0
    ax2.plot(log_mhost_grid[good_hmf], np.log10(n_hmf[good_hmf]),
             'm-', lw=1.5, label='Uchuu HMF (all centrals)')
    good_occ = n_hmf_occ > 0
    ax2.plot(log_mhost_grid[good_occ], np.log10(n_hmf_occ[good_occ]),
             'm--', lw=1.2, label='Uchuu HMF (occupied only)')

    for (mlo, mhi), col in zip(KP_BINS, BIN_COLORS):
        nc = n_cen_bins[(mlo, mhi)]
        mhost_table = MHOST_CEN.get((mlo, mhi), np.nan)
        if nc > 0:
            # Horizontal line at n_cen for this bin
            ax2.axhline(np.log10(nc), color=col, lw=0.8, ls='--', alpha=0.8,
                        label=rf'$M_\star\in[{mlo},{mhi})$: $\log n_{{cen}}={np.log10(nc):.2f}$')
        # Vertical dotted line at our median Mhost from table
        if np.isfinite(mhost_table):
            ax2.axvline(mhost_table, color=col, lw=0.8, ls=':', alpha=0.8)

    ax2.set_xlabel(r'$\log_{10}(M_{\rm host}/h^{-1}M_\odot)$')
    ax2.set_ylabel(r'$\log_{10}\,n(>M_{\rm host})\ [h^3\,\mathrm{Mpc}^{-3}]$')
    ax2.set_title(rf'Uchuu HMF: ${args.zmin:.3f} \leq z \leq {args.zmax:.3f}$')
    ax2.set_xlim(10.5, 15.0)
    ax2.set_ylim(-7, -1)
    ax2.legend(fontsize=6, frameon=False, loc='upper right')
    ax2.grid(alpha=0.2)
    ax2.text(0.03, 0.05,
             'Dashed horizontal = $n_{\\rm cen}$ per bin\n'
             'Dotted vertical = median $M_{\\rm host}$ (table)',
             transform=ax2.transAxes, fontsize=6, va='bottom')

    if args.outfile is None:
        outfile = PIPELINE_DIR / \
                  f"cumulative_nM_z{args.zmin:.3f}-{args.zmax:.3f}.png"
    else:
        outfile = args.outfile
    fig.savefig(str(outfile), dpi=150, bbox_inches='tight')
    log.info("Saved %s", outfile)
    plt.close(fig)


if __name__ == "__main__":
    main()
