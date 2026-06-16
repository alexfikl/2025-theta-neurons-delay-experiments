from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

# ============================================================
# Global plotting style
# ============================================================

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
# Numerical tolerances
# ============================================================

TOL_COEFF = 1e-12
TOL_REAL = 1e-10
TOL_MERGE = 1e-7
TOL_ETA = 5e-4
TOL_STAB = 1e-9


# ============================================================
# Polynomial / equilibrium utilities
# ============================================================


def poly_roots(coeffs, tol=TOL_COEFF):
    """Return roots after trimming leading near-zero coefficients."""
    coeffs = np.asarray(coeffs, dtype=float)
    nz = np.flatnonzero(np.abs(coeffs) > tol)
    if len(nz) == 0:
        return np.array([], dtype=complex)
    coeffs = coeffs[nz[0] :]
    if len(coeffs) <= 1:
        return np.array([], dtype=complex)
    return np.roots(coeffs)


def distinct_real_roots_in_open_interval(
    coeffs, a=-1.0, b=1.0, tol_real=TOL_REAL, tol_merge=TOL_MERGE
):
    """
    Return distinct real roots of the polynomial in the open interval (a,b).
    Multiple roots are merged numerically.
    """
    roots = poly_roots(coeffs)
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


def Pprime(kappa, eta, c):  # noqa: N802
    """
    Derivative of the type 1 cubic
    P(c)=kappa c^3 - kappa c^2 + (eta-kappa-1)c + (eta+kappa+1).
    """
    return 3.0 * kappa * c**2 - 2.0 * kappa * c + (eta - kappa - 1.0)


def count_type1_equilibria(kappa, eta):
    """
    For tau = 0:
      N1 = total number of distinct type 1 equilibria
      S1 = number of asymptotically stable type 1 equilibria
    """
    coeffs = [kappa, -kappa, eta - kappa - 1.0, eta + kappa + 1.0]
    c_roots = distinct_real_roots_in_open_interval(coeffs, a=-1.0, b=1.0)

    # Each c in (-1,1) gives two equilibria: ± arccos(c)
    N1 = 2 * len(c_roots)

    # Special boundary equilibrium z*=1 occurs iff eta=0
    if abs(eta) < TOL_ETA:
        N1 += 1

    # Only the lower branch can be stable, and it is stable iff P'(c)<0
    S1 = sum(1 for c in c_roots if Pprime(kappa, eta, c) < -TOL_STAB)
    return N1, S1


def saddle_node_curve(
    num=5000, kappa_min=-5.0, kappa_max=5.0, eta_min=-1.0, eta_max=1.0
):
    """
    Type 1 saddle-node curve:
        kappa = -1 / ((1-c)(1+c)^2)
        eta   = c(c-1)/(1+c)^2
    """
    c = np.linspace(-0.999999, 0.999999, num)
    kappa = -1.0 / ((1.0 - c) * (1.0 + c) ** 2)
    eta = c * (c - 1.0) / (1.0 + c) ** 2

    mask = (
        (kappa >= kappa_min)
        & (kappa <= kappa_max)
        & (eta >= eta_min)
        & (eta <= eta_max)
    )
    return kappa[mask], eta[mask]


# ============================================================
# Dirac-kernel first critical delay for type 1 Hopf
# ============================================================


def first_critical_tau_type1_dirac(kappa, eta):
    """
    Return the smallest critical delay tau_c for a type 1 equilibrium
    that is asymptotically stable at tau=0 and undergoes Hopf in the
    Dirac-kernel case. If no such Hopf point exists, return np.nan.
    """
    coeffs = [kappa, -kappa, eta - kappa - 1.0, eta + kappa + 1.0]
    c_roots = distinct_real_roots_in_open_interval(coeffs)

    tau_candidates = []

    for c in c_roots:
        # Keep only stable type 1 equilibria at tau = 0
        if Pprime(kappa, eta, c) >= -TOL_STAB:
            continue

        # Lower branch phi* = -arccos(c)
        alpha = -2.0 * np.sqrt((1.0 - c) / (1.0 + c))
        beta = -2.0 * kappa * (1.0 - c**2) ** 1.5

        # Dirac Hopf condition
        if not (beta < 0.0 and beta**2 > alpha**2):
            continue

        omega = np.sqrt(beta**2 - alpha**2)
        theta = np.arccos(-alpha / beta)
        tau_c = theta / omega
        tau_candidates.append(tau_c)

    if tau_candidates:
        return min(tau_candidates)
    return np.nan


# ============================================================
# Figure 1: Type 1 equilibria (tau = 0)
# ============================================================


def plot_type1_equilibria_tau0(
    kappa_min=-5.0,
    kappa_max=5.0,
    eta_min=-1.0,
    eta_max=1.0,
    NK=1000,
    NE=500,
    save_prefix="type1_equilibria_tau0_no_curve_labels",
):
    kappas = np.linspace(kappa_min, kappa_max, NK)
    etas = np.linspace(eta_min, eta_max, NE)

    N1 = np.zeros((NE, NK), dtype=int)
    S1 = np.zeros((NE, NK), dtype=int)

    for j, eta in enumerate(etas):
        for i, kappa in enumerate(kappas):
            N1[j, i], S1[j, i] = count_type1_equilibria(kappa, eta)

    # Encode by pair (N1,S1)
    pair_to_code = {
        (0, 0): 0,
        (1, 0): 1,
        (2, 1): 2,
        (3, 0): 3,
        (4, 1): 4,
        (5, 1): 5,
        (6, 2): 6,
    }

    colors = {
        0: "#f2f2f2",  # light gray
        1: "#f7c948",  # mustard
        2: "#8ccf7e",  # green
        3: "#c8a2ff",  # lavender
        4: "#4f81c7",  # deep blue
        5: "#f2a65a",  # warm orange
        6: "#d95f5f",  # soft red
    }

    Z = np.zeros((NE, NK), dtype=int)
    for j in range(NE):
        for i in range(NK):
            Z[j, i] = pair_to_code.get((N1[j, i], S1[j, i]), -1)

    cmap = ListedColormap([colors[i] for i in range(7)])
    norm = BoundaryNorm(np.arange(-0.5, 7.5, 1.0), cmap.N)

    _, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.imshow(
        Z,
        origin="lower",
        extent=(kappa_min, kappa_max, eta_min, eta_max),
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    # Gamma_SN^(1): dashed curve
    ks, es = saddle_node_curve(
        kappa_min=kappa_min, kappa_max=kappa_max, eta_min=eta_min, eta_max=eta_max
    )
    ax.plot(ks, es, linestyle="--", linewidth=2.1, color="black")

    # Sigma: eta = 0 full straight line
    ax.axhline(0.0, linestyle="-", linewidth=2.1, color="#5b2ca0")

    # Cusp point
    ax.plot([-27 / 32], [-1 / 8], marker="o", markersize=5.5, color="black", zorder=5)
    ax.text(-0.62, -0.17, "cusp", fontsize=11)

    # Region labels
    bbox_kw = {
        "boxstyle": "round,pad=0.22",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.82,
    }

    ax.text(
        -2.95,
        0.56,
        r"$N_1=4,\ S_1=1$",
        ha="center",
        va="center",
        fontsize=12,
        bbox=bbox_kw,
    )
    ax.text(
        1.95,
        0.56,
        r"$N_1=0,\ S_1=0$",
        ha="center",
        va="center",
        fontsize=12,
        bbox=bbox_kw,
    )
    ax.text(
        1.80,
        -0.56,
        r"$N_1=2,\ S_1=1$",
        ha="center",
        va="center",
        fontsize=12,
        bbox=bbox_kw,
    )
    ax.text(
        -2.45,
        -0.065,
        r"$N_1=6,\ S_1=2$",
        ha="center",
        va="center",
        fontsize=11.5,
        bbox=bbox_kw,
    )

    ax.set_xlim(kappa_min, kappa_max)
    ax.set_ylim(eta_min, eta_max)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\eta$")
    ax.set_title(r"Type 1 equilibria ($\tau=0$)")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    plt.show()


# ============================================================
# Figure 2: First critical delay tau_c, restricted region
# ============================================================


def plot_first_critical_tau_dirac_restricted(
    kappa_min=0.0,
    kappa_max=5.0,
    eta_min=-1.0,
    eta_max=0.0,
    NK=700,
    NE=400,
    save_prefix="type1_dirac_first_critical_tau_restricted",
):
    kappas = np.linspace(kappa_min, kappa_max, NK)
    etas = np.linspace(eta_min, eta_max, NE)

    Tau = np.empty((NE, NK), dtype=float)
    for j, eta in enumerate(etas):
        for i, kappa in enumerate(kappas):
            Tau[j, i] = first_critical_tau_type1_dirac(kappa, eta)

    Tau_masked = np.ma.masked_invalid(Tau)

    # Levels chosen so tau < 0.5 is also colored
    levels = np.array([0.0, 0.5, 1, 2, 3, 4, 6, 8, 10], dtype=float)
    contours = [0.5, 1, 2, 4, 6, 8, 10]

    cmap = plt.get_cmap("plasma").copy()
    cmap.set_bad("#ececec")
    norm = BoundaryNorm(levels, cmap.N)

    K, E = np.meshgrid(kappas, etas)

    _, ax = plt.subplots(figsize=(7.8, 5.3))

    # Filled bands
    cf = ax.contourf(
        K,
        E,
        Tau_masked,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="max",
    )

    # Contour lines
    cs = ax.contour(
        K,
        E,
        Tau_masked,
        levels=contours,
        colors="white",
        linewidths=0.9,
    )
    ax.clabel(cs, inline=True, fmt=r"$\tau=%g$", fontsize=9)

    ax.set_xlim(kappa_min, kappa_max)
    ax.set_ylim(eta_min, eta_max)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$\eta$")
    ax.set_title(r"First critical delay $\tau_c$ for type 1 Hopf (Dirac kernel)")

    cbar = plt.colorbar(cf, ax=ax, ticks=levels, extend="max")
    cbar.set_label(r"$\tau_c$")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_prefix}.pdf", bbox_inches="tight")
    plt.show()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    plot_type1_equilibria_tau0()
    plot_first_critical_tau_dirac_restricted()
