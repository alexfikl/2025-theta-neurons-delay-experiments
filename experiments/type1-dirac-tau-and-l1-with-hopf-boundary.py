from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import thetalib
from matplotlib.colors import BoundaryNorm

# ============================================================
# Domain and numerical tolerances
# ============================================================

KAPPA_MIN, KAPPA_MAX = 0.0, 5.0
ETA_MIN, ETA_MAX = -1.0, 0.0

NK = 700
NE = 400


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
# Exact Hopf boundary curve for the type 1 Dirac case
# ============================================================


def kappa_hopf_boundary(eta):
    """
    Exact boundary of the type 1 Hopf region (Dirac kernel):
        kappa_H(eta) = ((sqrt(9-8 eta)-1)^3)/(32 (sqrt(9-8 eta)-3)), eta<0
    """
    s = np.sqrt(9.0 - 8.0 * eta)
    return ((s - 1.0) ** 3) / (32.0 * (s - 3.0))


def hopf_boundary_visible(num=4000):
    """
    Return the visible part of the exact Hopf boundary in the plotting window.
    """
    eta_vals = np.linspace(ETA_MIN, -1e-6, num)
    kappa_vals = kappa_hopf_boundary(eta_vals)
    mask = (
        np.isfinite(kappa_vals) & (kappa_vals >= KAPPA_MIN) & (kappa_vals <= KAPPA_MAX)
    )
    return kappa_vals[mask], eta_vals[mask]


# ============================================================
# First critical delay tau_c and first Lyapunov coefficient l1
# ============================================================


def compute_tau_and_l1():
    kappas = np.linspace(KAPPA_MIN, KAPPA_MAX, NK)
    etas = np.linspace(ETA_MIN, ETA_MAX, NE)

    TAU = np.empty((NE, NK), dtype=float)
    L1 = np.empty((NE, NK), dtype=float)

    for j, eta in enumerate(etas):
        for i, kappa in enumerate(kappas):
            tau_c, l1 = thetalib.first_tau_and_l1_type1_dirac(kappa, eta)
            TAU[j, i] = tau_c
            L1[j, i] = l1

    return kappas, etas, TAU, L1


# ============================================================
# Helpers for manual contour-label placement
# ============================================================


def nearest_point_on_contour(cs, level_index, target_xy):
    """
    Find the point on the given contour level nearest to target_xy.
    """
    target = np.asarray(target_xy, dtype=float)
    best_point = None
    best_dist2 = np.inf

    segs = cs.allsegs[level_index]
    for seg in segs:
        if len(seg) == 0:
            continue
        diffs = seg - target
        dist2 = np.sum(diffs * diffs, axis=1)
        idx = np.argmin(dist2)
        if dist2[idx] < best_dist2:
            best_dist2 = dist2[idx]
            best_point = tuple(seg[idx])

    return best_point


# ============================================================
# Plot 1: first critical delay tau_c
# ============================================================


def plot_first_critical_tau_with_boundary(
    kappas, etas, TAU, save_prefix="type1_dirac_first_critical_tau_with_boundary"
):
    Tau_masked = np.ma.masked_invalid(TAU)

    levels = np.array([0.0, 0.5, 1, 2, 3, 4, 6, 8, 10], dtype=float)
    contours = [0.5, 1, 2, 4, 6, 8, 10]

    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("#ececec")
    norm = BoundaryNorm(levels, cmap.N)

    K, E = np.meshgrid(kappas, etas)

    _, ax = plt.subplots(figsize=(7.8, 5.3))

    cf = ax.contourf(
        K,
        E,
        Tau_masked,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="max",
    )

    cs = ax.contour(
        K,
        E,
        Tau_masked,
        levels=contours,
        colors="white",
        linewidths=0.9,
    )
    ax.clabel(cs, inline=True, fmt=r"$\tau=%g$", fontsize=9)

    # Exact Hopf boundary
    kb, eb = hopf_boundary_visible()
    ax.plot(kb, eb, linestyle="--", linewidth=2.2, color="black")

    ax.set_xlim(KAPPA_MIN, KAPPA_MAX)
    ax.set_ylim(ETA_MIN, ETA_MAX)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\eta$")

    cbar = plt.colorbar(cf, ax=ax, ticks=levels, extend="max")
    cbar.set_label(r"$\tau_0^{(1)}$")

    plt.tight_layout()
    # plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    plt.show()


# ============================================================
# Plot 2: first Lyapunov coefficient l1
# ============================================================


def plot_l1_with_boundary(
    kappas, etas, L1, save_prefix="type1_dirac_l1_first_tau_with_boundary"
):
    L1_masked = np.ma.masked_invalid(L1)

    levels = np.array([-4.5, -4.0, -3.0, -2.0, -1.5, -1.0, -0.7, -0.5, -0.3, -0.2, 0.0])
    contours = [-4.0, -3.0, -2.0, -1.5, -1.0, -0.7, -0.5, -0.3]

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#ececec")
    norm = BoundaryNorm(levels, cmap.N)

    K, E = np.meshgrid(kappas, etas)

    _, ax = plt.subplots(figsize=(7.8, 5.3))

    cf = ax.contourf(
        K,
        E,
        L1_masked,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="min",
    )

    cs = ax.contour(
        K,
        E,
        L1_masked,
        levels=contours,
        colors="white",
        linewidths=0.9,
    )

    # Manual interior contour-label positions
    target_positions = {
        -4.0: (4.55, -0.10),
        -3.0: (4.50, -0.20),
        -2.0: (4.15, -0.34),
        -1.5: (3.85, -0.52),
        -1.0: (2.55, -0.74),
        -0.7: (1.60, -0.92),
        -0.5: (1.15, -0.98),
        -0.3: (3.20, -0.86),
    }
    manual_positions = []
    for idx, lev in enumerate(contours):
        pt = nearest_point_on_contour(cs, idx, target_positions[lev])
        if pt is not None:
            manual_positions.append(pt)

    ax.clabel(cs, inline=True, fmt=r"$\ell_1=%g$", fontsize=8, manual=manual_positions)

    # Exact Hopf boundary
    kb, eb = hopf_boundary_visible()
    ax.plot(kb, eb, linestyle="--", linewidth=2.2, color="black")

    ax.set_xlim(KAPPA_MIN, KAPPA_MAX)
    ax.set_ylim(ETA_MIN, ETA_MAX)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\eta$")

    cbar = plt.colorbar(cf, ax=ax, ticks=levels)
    cbar.set_label(r"$\ell_1$")

    ax.text(
        3.55,
        -0.87,
        r"All visible values satisfy $\ell_1<0$",
        fontsize=11,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.82,
        },
    )

    plt.tight_layout()
    # plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    kappas, etas, TAU, L1 = compute_tau_and_l1()
    plot_first_critical_tau_with_boundary(kappas, etas, TAU)
    plot_l1_with_boundary(kappas, etas, L1)
