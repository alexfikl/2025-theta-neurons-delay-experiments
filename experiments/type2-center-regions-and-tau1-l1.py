from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import thetalib
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# ============================================================
# Type-2 center utilities
# ============================================================


def distinct_real_roots_in_open_interval(
    coeffs,
    a=-1.0,
    b=1.0,
    tol_real=thetalib.TOL_REAL,
    tol_merge=thetalib.TOL_MERGE,
):
    coeffs = list(np.asarray(coeffs, dtype=float))
    while coeffs and abs(coeffs[0]) < 1e-14:
        coeffs = coeffs[1:]
    if not coeffs:
        return np.array([], dtype=float)

    roots = np.roots(np.asarray(coeffs, dtype=float))
    roots = [
        r.real
        for r in roots
        if abs(r.imag) < tol_real and a + tol_merge < r.real < b - tol_merge
    ]
    if not roots:
        return np.array([], dtype=float)

    roots = np.array(sorted(roots), dtype=float)
    merged = [roots[0]]
    for r in roots[1:]:
        if abs(r - merged[-1]) <= tol_merge:
            merged[-1] = 0.5 * (merged[-1] + r)
        else:
            merged.append(r)
    return np.array(merged, dtype=float)


# ============================================================
# Region classification for the center only
# 0 = no type-2 center
# 1 = kappa < 0, unstable for small tau and remains unstable
# 2 = kappa < 0, unstable for tau<tau1 and regains stability at tau1
# 3 = kappa > 0, stable for tau<tau1, no switching
# 4 = kappa > 0, stable for tau<tau1, switching may occur
# ============================================================


def classify_center_region(kappa, eta):
    data = thetalib.d0_d1_from_center(kappa, eta)
    if data is None:
        return 0

    _, d0, d1 = data

    if kappa < -thetalib.KAPPA_EPS:
        return 2 if (3.0 * d0 > -5.0 * d1) else 1
    if kappa > thetalib.KAPPA_EPS:
        return 4 if (13.0 * d1 < 5.0 * d0) else 3
    return 0


def compute_center_region_grid(kmin, kmax, emin, emax, nk=480, ne=320):
    kappas = np.linspace(kmin, kmax, nk)
    etas = np.linspace(emin, emax, ne)
    REGION = np.zeros((ne, nk), dtype=int)
    for j, eta in enumerate(etas):
        for i, kappa in enumerate(kappas):
            REGION[j, i] = classify_center_region(kappa, eta)
    return kappas, etas, REGION


def region_label_position(kappas, etas, REGION, code):
    yy, xx = np.where(code == REGION)
    if len(xx) == 0:
        return None
    return float(np.median(kappas[xx])), float(np.median(etas[yy]))


# ============================================================
# Curves
# ============================================================


def saddle_node_curve(num=6000):
    z = np.linspace(-0.99999, 0.99999, num)
    kappa = 4.0 * (1.0 - z) / ((1.0 + z) ** 3 * (2.0 - z))
    eta = -((1.0 - z) ** 2 * (z**2 - 3.0 * z + 4.0)) / ((1.0 + z) ** 3 * (2.0 - z))
    return kappa, eta


def gamma_sw_plus(z):
    kappa = 20.0 * (1.0 - z) / (13.0 * (1.0 + z) ** 3 * (2.0 - z))
    eta = -((1.0 - z) ** 2 * (13.0 * z**2 - 23.0 * z + 4.0)) / (
        13.0 * (1.0 + z) ** 3 * (2.0 - z)
    )
    return kappa, eta


def gamma_sw_minus(z):
    kappa = -12.0 * (1.0 - z) / (5.0 * (1.0 + z) ** 3 * (2.0 - z))
    eta = ((1.0 - z) ** 2 * (28.0 - z - 5.0 * z**2)) / (
        5.0 * (1.0 + z) ** 3 * (2.0 - z)
    )
    return kappa, eta


# ============================================================
# Tensors and l1 at tau1 for the Dirac kernel
# ============================================================


def compute_tau1_and_l1_grids(kmin=-5.0, kmax=5.0, emin=-1.0, emax=1.0, nk=500, ne=320):
    kappas = np.linspace(kmin, kmax, nk)
    etas = np.linspace(emin, emax, ne)
    TAU1 = np.full((ne, nk), np.nan, dtype=float)
    L1 = np.full((ne, nk), np.nan, dtype=float)
    for j, eta in enumerate(etas):
        for i, kappa in enumerate(kappas):
            tau1, l1 = thetalib.tau1_and_l1_type2_dirac(kappa, eta)
            TAU1[j, i] = tau1
            L1[j, i] = l1
    return kappas, etas, TAU1, L1


# ============================================================
# Plot center-region figures
# ============================================================


def plot_full_center_regions(prefix="type2_center_regions_with_SN_and_sw_updated"):
    kmin, kmax = -5.0, 5.0
    emin, emax = -1.0, 1.0
    kappas, etas, REGION = compute_center_region_grid(kmin, kmax, emin, emax)
    K, E = np.meshgrid(kappas, etas)
    Z = np.ma.masked_where(REGION == 0, REGION)

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.contourf(K, E, Z, levels=[0.5, 1.5, 2.5, 3.5, 4.5], alpha=0.45)

    z = np.linspace(-0.9999, 0.9999, 12000)
    k_sn, e_sn = saddle_node_curve()
    k_p, e_p = gamma_sw_plus(z)
    k_m, e_m = gamma_sw_minus(z)

    def mask_curve(k, e):
        return (
            np.isfinite(k)
            & np.isfinite(e)
            & (k >= kmin)
            & (k <= kmax)
            & (e >= emin)
            & (e <= emax)
        )

    ax.plot(
        k_sn[mask_curve(k_sn, e_sn)],
        e_sn[mask_curve(k_sn, e_sn)],
        "--",
        linewidth=2.1,
        label=r"$\Gamma_{SN}^{(2)}$",
    )
    ax.plot(
        k_p[mask_curve(k_p, e_p)],
        e_p[mask_curve(k_p, e_p)],
        "-.",
        linewidth=2.1,
        label=r"$\Gamma_{\mathrm{sw}}^{(2,+)}$",
    )
    ax.plot(
        k_m[mask_curve(k_m, e_m)],
        e_m[mask_curve(k_m, e_m)],
        ":",
        linewidth=2.4,
        label=r"$\Gamma_{\mathrm{sw}}^{(2,-)}$",
    )

    ax.axhline(0.0, linewidth=1.0, alpha=0.55)
    ax.axvline(0.0, linewidth=1.0, alpha=0.55)

    ax.set_xlim(kmin, kmax)
    ax.set_ylim(emin, emax)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\eta$")
    ax.legend(loc="upper right", framealpha=0.92)

    labels = {
        1: r"unstable for small $\tau$" "\n" r"and remains unstable",
        2: r"unstable for $\tau<\tau_1^{(2)}$"
        "\n"
        r"regains stability at $\tau_1^{(2)}$",
        3: r"stable for $\tau<\tau_1^{(2)}$" "\n" r"no switching",
        4: r"stable for $\tau<\tau_1^{(2)}$" "\n" r"switching may occur",
    }
    for code, txt in labels.items():
        pos = region_label_position(kappas, etas, REGION, code)
        if pos is not None:
            ax.text(
                pos[0],
                pos[1],
                txt,
                ha="center",
                va="center",
                fontsize=10.5,
                bbox={
                    "boxstyle": "round,pad=0.24",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            )

    plt.tight_layout()
    # fig.savefig(f"{prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_negative_zoom(prefix="type2_center_regions_negative_zoom_updated"):
    kmin, kmax = -0.9, 0.25
    emin, emax = 0.0, 1.0
    kappas, etas, REGION = compute_center_region_grid(
        kmin, kmax, emin, emax, nk=420, ne=280
    )
    K, E = np.meshgrid(kappas, etas)
    Z = np.ma.masked_where((REGION == 0) | (REGION >= 3), REGION)

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.contourf(K, E, Z, levels=[0.5, 1.5, 2.5], alpha=0.45)

    z = np.linspace(-0.9999, 0.9999, 12000)
    k_m, e_m = gamma_sw_minus(z)
    mask_m = (
        np.isfinite(k_m)
        & np.isfinite(e_m)
        & (k_m >= kmin)
        & (k_m <= kmax)
        & (e_m >= emin)
        & (e_m <= emax)
    )
    ax.plot(
        k_m[mask_m],
        e_m[mask_m],
        ":",
        linewidth=2.4,
        label=r"$\Gamma_{\mathrm{sw}}^{(2,-)}$",
    )

    ax.axhline(0.0, linewidth=1.0, alpha=0.55)
    ax.axvline(0.0, linewidth=1.0, alpha=0.55)
    ax.set_xlim(kmin, kmax)
    ax.set_ylim(emin, emax)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\eta$")
    ax.set_title(r"Type 2 center, $\kappa<0$: delayed stabilization at $\tau_1$")
    ax.legend(loc="upper right", framealpha=0.92)

    labels = {
        1: r"remains unstable",
        2: r"regains stability" "\n" r"at $\tau_1$",
    }
    for code, txt in labels.items():
        pos = region_label_position(kappas, etas, REGION, code)
        if pos is not None:
            ax.text(
                pos[0],
                pos[1],
                txt,
                ha="center",
                va="center",
                fontsize=11,
                bbox={
                    "boxstyle": "round,pad=0.24",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            )

    plt.tight_layout()
    # fig.savefig(f"{prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plot tau1 and l1 with SN and Gamma_sw^{(2,-)}
# ============================================================


def plot_tau1_and_l1(
    tau_prefix="type2_dirac_first_critical_tau1_with_SN_and_swminus",
    l1_prefix="type2_dirac_l1_at_first_critical_tau1_with_SN_swminus_and_l10",
):
    kappas, etas, TAU1, L1 = compute_tau1_and_l1_grids()
    K, E = np.meshgrid(kappas, etas)

    z = np.linspace(-0.9999, 0.9999, 12000)
    k_sn, e_sn = saddle_node_curve()
    k_m, e_m = gamma_sw_minus(z)

    def mask_curve(k, e):
        return (
            np.isfinite(k)
            & np.isfinite(e)
            & (k >= -5.0)
            & (k <= 5.0)
            & (e >= -1.0)
            & (e <= 1.0)
        )

    mask_sn = mask_curve(k_sn, e_sn)
    mask_m = mask_curve(k_m, e_m)

    # tau1
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    Tau_masked = np.ma.masked_invalid(TAU1)
    tau_levels = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)
    tau_contours = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    cmap_tau = plt.get_cmap("plasma").copy()
    cmap_tau.set_bad("#ececec")
    norm_tau = BoundaryNorm(tau_levels, cmap_tau.N)

    cf = ax.contourf(
        K, E, Tau_masked, levels=tau_levels, cmap=cmap_tau, norm=norm_tau, extend="max"
    )
    cs = ax.contour(
        K, E, Tau_masked, levels=tau_contours, colors="white", linewidths=0.8
    )
    ax.clabel(cs, inline=True, fmt=r"$\tau=%g$", fontsize=8)

    ax.plot(k_sn[mask_sn], e_sn[mask_sn], "--", linewidth=2.1)
    ax.plot(k_m[mask_m], e_m[mask_m], ":", linewidth=2.4)

    ax.axhline(0.0, linewidth=1.0, alpha=0.55)
    ax.axvline(0.0, linewidth=1.0, alpha=0.55)
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\eta$")
    ax.legend(
        handles=[
            Line2D(
                [0], [0], linestyle="--", linewidth=2.1, label=r"$\Gamma_{SN}^{(2)}$"
            ),
            Line2D(
                [0],
                [0],
                linestyle=":",
                linewidth=2.4,
                label=r"$\Gamma_{\mathrm{sw}}^{(2,-)}$",
            ),
        ],
        loc="upper right",
        framealpha=0.92,
    )

    cbar = plt.colorbar(cf, ax=ax, ticks=tau_levels, extend="max")
    cbar.set_label(r"$\tau_1^{(2)}$")

    plt.tight_layout()
    # fig.savefig(f"{tau_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{tau_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)

    # l1
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    L1_masked = np.ma.masked_invalid(L1)
    l1_levels = np.array(
        [-4.5, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.5], dtype=float
    )
    l1_contours = [-4.0, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

    cmap_l1 = plt.get_cmap("viridis").copy()
    cmap_l1.set_bad("#ececec")
    norm_l1 = BoundaryNorm(l1_levels, cmap_l1.N)

    cf = ax.contourf(
        K, E, L1_masked, levels=l1_levels, cmap=cmap_l1, norm=norm_l1, extend="both"
    )
    cs = ax.contour(K, E, L1_masked, levels=l1_contours, colors="white", linewidths=0.8)
    ax.clabel(cs, inline=True, fmt=r"$\ell_1=%g$", fontsize=8)

    zero_contour = ax.contour(
        K, E, L1_masked, levels=[0.0], colors="black", linewidths=2.4
    )
    ax.clabel(zero_contour, inline=True, fmt=r"$\ell_1=0$", fontsize=9)

    ax.plot(k_sn[mask_sn], e_sn[mask_sn], "--", linewidth=2.1)
    ax.plot(k_m[mask_m], e_m[mask_m], ":", linewidth=2.4)

    ax.axhline(0.0, linewidth=1.0, alpha=0.55)
    ax.axvline(0.0, linewidth=1.0, alpha=0.55)
    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\eta$")
    ax.legend(
        handles=[
            Line2D(
                [0], [0], linestyle="--", linewidth=2.1, label=r"$\Gamma_{SN}^{(2)}$"
            ),
            Line2D(
                [0],
                [0],
                linestyle=":",
                linewidth=2.4,
                label=r"$\Gamma_{\mathrm{sw}}^{(2,-)}$",
            ),
            Line2D(
                [0],
                [0],
                linestyle="-",
                linewidth=2.4,
                color="black",
                label=r"$\ell_1=0$",
            ),
        ],
        loc="upper right",
        framealpha=0.92,
    )

    cbar = plt.colorbar(cf, ax=ax, ticks=l1_levels, extend="both")
    cbar.set_label(r"$\ell_1$")

    plt.tight_layout()
    # fig.savefig(f"{l1_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{l1_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    plot_full_center_regions()
    plot_negative_zoom()
    plot_tau1_and_l1()


if __name__ == "__main__":
    main()
