from __future__ import annotations

from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import thetalib
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


KMIN, KMAX = -5.0, 5.0
EMIN, EMAX = -1.0, 1.0
taus = [0.1, 0.5, 1.0, 2.0]
NK = 320
NE = 260


def type1_features(kappa, eta):
    coeffs = [kappa, -kappa, eta - kappa - 1.0, eta + kappa + 1.0]
    c_roots = thetalib.distinct_real_roots_in_open_interval(coeffs)
    stable_neg_count = 0
    stable_pos_exists = 0
    tau0 = np.nan
    l1 = np.nan
    for c in c_roots:
        if thetalib.Pprime(kappa, eta, c) >= -1e-9:
            continue
        if kappa < 0:
            stable_neg_count += 1
        elif kappa > 0:
            stable_pos_exists = 1
            tau0, l1 = thetalib.first_tau_and_l1_type1_dirac(kappa, eta)
            break
    return stable_neg_count, stable_pos_exists, tau0, l1


def type2_crossings_up_to_tau_max(kappa, eta, tau_max):
    data = thetalib.d0_d1_from_center(kappa, eta)
    if data is None:
        return None
    zeta, d0, d1 = data
    stable_small_tau = kappa > 0
    criticals = []
    w_plus = np.sqrt(max(d0 + d1, 0.0))
    w_minus = np.sqrt(max(d0 - d1, 0.0))
    max_w = max(w_plus, w_minus)
    nmax = int(np.ceil(tau_max * max_w / np.pi)) + 10 if max_w > 0 else 1
    for n in range(1, nmax + 1):
        den = d0 - ((-1) ** n) * d1
        if den <= 1e-12:
            continue
        tn = n * np.pi / np.sqrt(den)
        if tn <= tau_max + 1e-12:
            criticals.append(tn)
    criticals = np.array(sorted(set(np.round(criticals, 12))), dtype=float)
    tau1, l1 = thetalib.tau1_and_l1_type2_dirac(kappa, eta)
    return {
        "zeta": zeta,
        "d0": d0,
        "d1": d1,
        "stable_small_tau": stable_small_tau,
        "criticals": criticals,
        "tau1": tau1,
        "l1": l1,
    }


def stable_type2_center_at_tau(info, tau):
    if info is None:
        return 0
    stable = info["stable_small_tau"]
    n_cross = np.searchsorted(info["criticals"], tau + 1e-12, side="right")
    if n_cross % 2 == 1:
        stable = not stable
    return int(stable)


def stable_type2_cycle_predicted(info, kappa, tau):
    if info is None:
        return 0
    tau1 = info["tau1"]
    l1 = info["l1"]
    if not np.isfinite(tau1) or not np.isfinite(l1) or l1 >= 0:
        return 0
    if kappa > 0 and tau > tau1:
        return 1
    if kappa < 0 and tau < tau1:
        return 1
    return 0


def stable_type1_counts_at_tau(features, kappa, tau):
    stable_neg_count, stable_pos_exists, tau0, l1 = features
    s1 = stable_neg_count
    c1 = 0
    if kappa > 0 and stable_pos_exists:
        if np.isfinite(tau0) and tau > tau0:
            if np.isfinite(l1) and l1 < 0:
                c1 = 1
        else:
            s1 += 1
    return s1, c1


def classify_point(kappa, eta, tau, t1_features, t2_info):
    s1, c1 = stable_type1_counts_at_tau(t1_features, kappa, tau)
    s2 = stable_type2_center_at_tau(t2_info, tau)
    c2 = stable_type2_cycle_predicted(t2_info, kappa, tau)
    return (s1, s2, c1 + c2)


def tuple_string(tup):
    return rf"$({tup[0]},{tup[1]},{tup[2]})$"


def type1_SN_curve(num=8000):  # noqa: N802
    c = np.linspace(-0.999999, 0.999999, num)
    kappa = -1.0 / ((1.0 - c) * (1.0 + c) ** 2)
    eta = c * (c - 1.0) / (1.0 + c) ** 2
    return kappa, eta


def type2_SN_curve(num=8000):  # noqa: N802
    z = np.linspace(-0.999999, 0.999999, num)
    kappa = 4.0 * (1.0 - z) / ((1.0 + z) ** 3 * (2.0 - z))
    eta = -((1.0 - z) ** 2 * (z**2 - 3.0 * z + 4.0)) / ((1.0 + z) ** 3 * (2.0 - z))
    return kappa, eta


def type1_Hopf_curves_fixed_tau(tau, nmax=12, m=4000):  # noqa: N802
    curves = []
    eps = 1e-4
    for n in range(nmax + 1):
        a = np.pi / 2 + n * np.pi + eps
        b = np.pi + n * np.pi - eps
        if a >= b:
            continue
        nu = np.linspace(a, b, m)
        cot = 1.0 / np.tan(nu)
        denom = 128.0 * tau**4 * nu**2 * cot**3 * np.sin(nu)
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = -((4.0 * tau**2 + nu**2 * cot**2) ** 3) / denom
            eta = (
                (nu**2 * cot**2)
                / (4.0 * tau**2)
                * ((4.0 * tau**2 + nu**2 * cot**2) / (8.0 * tau**2 * np.cos(nu)) - 1.0)
            )
            u = (nu / (2.0 * tau)) * cot
            c = (1.0 - u**2) / (1.0 + u**2)
        mask = (
            np.isfinite(kappa)
            & np.isfinite(eta)
            & np.isfinite(c)
            & (kappa >= KMIN)
            & (kappa <= KMAX)
            & (eta >= EMIN)
            & (eta <= EMAX)
            & (c > -1.0)
            & (c < 1.0)
        )
        if np.any(mask):
            ppr = np.array([
                thetalib.Pprime(kk, ee, cc)
                for kk, ee, cc in zip(kappa[mask], eta[mask], c[mask], strict=True)
            ])
            submask = ppr < -1e-8
            if np.any(submask):
                curves.append((kappa[mask][submask], eta[mask][submask]))
    return curves


def type2_Hopf_curves_fixed_tau(tau, nmax=18, m=7000):  # noqa: N802
    curves = []
    z = np.linspace(-0.999999, 0.999999, m)
    I = 1.5 - 2.0 * z + 0.5 * z**2  # noqa: E741
    for n in range(1, nmax + 1):
        base = 4.0 * ((1.0 - z) / (1.0 + z)) ** 2 - (n**2) * np.pi**2 / tau**2
        denom = (1.0 - z**2) * (2.0 - z)
        kappa = ((-1) ** n) * base / denom
        eta = ((1.0 - z) / (1.0 + z)) ** 2 - kappa * I
        Fp = kappa * (2.0 - z) - 4.0 * (1.0 - z) / (1.0 + z) ** 3
        mask = (
            np.isfinite(kappa)
            & np.isfinite(eta)
            & (kappa >= KMIN)
            & (kappa <= KMAX)
            & (eta >= EMIN)
            & (eta <= EMAX)
            & (Fp < -1e-8)
        )
        if np.any(mask):
            curves.append((kappa[mask], eta[mask]))
    return curves


explicit_colors = {
    (1, 0, 0): "#F2C14E",
    (2, 0, 0): "#B5651D",
    (0, 1, 0): "#E15759",
    (0, 0, 1): "#4E79A7",
    (0, 0, 2): "#1F4E79",
    (1, 1, 0): "#F28E2B",
    (1, 0, 1): "#9C7AC9",
    (0, 1, 1): "#7E5AA6",
    (1, 1, 1): "#8C8C4A",
    (0, 1, 2): "#5C6BC0",
}


def color_from_tuple(tup):
    return explicit_colors.get(tup, "#BBBBBB")


def largest_component_centroid(code_grid, code_value):
    mask = code_grid == code_value
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_coords = None
    best_size = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            coords = []
            while q:
                cy, cx = q.popleft()
                coords.append((cy, cx))
                if cy > 0 and mask[cy - 1, cx] and not visited[cy - 1, cx]:
                    visited[cy - 1, cx] = True
                    q.append((cy - 1, cx))
                if cy < h - 1 and mask[cy + 1, cx] and not visited[cy + 1, cx]:
                    visited[cy + 1, cx] = True
                    q.append((cy + 1, cx))
                if cx > 0 and mask[cy, cx - 1] and not visited[cy, cx - 1]:
                    visited[cy, cx - 1] = True
                    q.append((cy, cx - 1))
                if cx < w - 1 and mask[cy, cx + 1] and not visited[cy, cx + 1]:
                    visited[cy, cx + 1] = True
                    q.append((cy, cx + 1))
            if len(coords) > best_size:
                best_size = len(coords)
                best_coords = coords
    if best_coords is None:
        return None
    ys = np.array([c[0] for c in best_coords], dtype=float)
    xs = np.array([c[1] for c in best_coords], dtype=float)
    return round(np.median(ys)), round(np.median(xs))


def main():
    kappas = np.linspace(KMIN, KMAX, NK)
    etas = np.linspace(EMIN, EMAX, NE)

    T1_FEATURES = [[None for _ in range(NK)] for _ in range(NE)]
    T2_INFO = [[None for _ in range(NK)] for _ in range(NE)]

    for j, eta in enumerate(etas):
        for i, kappa in enumerate(kappas):
            if abs(kappa) < 1e-12:
                continue
            T1_FEATURES[j][i] = type1_features(kappa, eta)
            T2_INFO[j][i] = type2_crossings_up_to_tau_max(kappa, eta, max(taus))

    all_tuples = set()
    for tau in taus:
        for j, eta in enumerate(etas):
            for i, kappa in enumerate(kappas):
                if abs(kappa) < 1e-12:
                    continue
                tup = classify_point(kappa, eta, tau, T1_FEATURES[j][i], T2_INFO[j][i])
                if tup != (0, 0, 0):
                    all_tuples.add(tup)

    tuple_list = sorted(all_tuples)
    tuple_to_code = {tup: idx + 1 for idx, tup in enumerate(tuple_list)}
    code_to_tuple = {idx + 1: tup for idx, tup in enumerate(tuple_list)}

    cmap = ListedColormap([color_from_tuple(tup) for tup in tuple_list])
    norm = BoundaryNorm(np.arange(0.5, len(tuple_list) + 1.5, 1.0), cmap.N)

    k1_sn, e1_sn = type1_SN_curve()
    k2_sn, e2_sn = type2_SN_curve()
    mask1 = (
        np.isfinite(k1_sn)
        & np.isfinite(e1_sn)
        & (k1_sn >= KMIN)
        & (k1_sn <= KMAX)
        & (e1_sn >= EMIN)
        & (e1_sn <= EMAX)
    )
    mask2 = (
        np.isfinite(k2_sn)
        & np.isfinite(e2_sn)
        & (k2_sn >= KMIN)
        & (k2_sn <= KMAX)
        & (e2_sn >= EMIN)
        & (e2_sn <= EMAX)
    )

    hopf1_by_tau = {tau: type1_Hopf_curves_fixed_tau(tau) for tau in taus}
    hopf2_by_tau = {tau: type2_Hopf_curves_fixed_tau(tau) for tau in taus}

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.3), sharex=True, sharey=True)
    axes = axes.ravel()
    K, E = np.meshgrid(kappas, etas)

    for ax, tau in zip(axes, taus, strict=True):
        CODE = np.zeros((NE, NK), dtype=int)
        for j, eta in enumerate(etas):
            for i, kappa in enumerate(kappas):
                if abs(kappa) < 1e-12:
                    continue
                tup = classify_point(kappa, eta, tau, T1_FEATURES[j][i], T2_INFO[j][i])
                if tup != (0, 0, 0):
                    CODE[j, i] = tuple_to_code[tup]

        Z = np.ma.masked_where(CODE == 0, CODE)
        ax.set_facecolor("#f2f2f2")
        ax.contourf(
            K,
            E,
            Z,
            levels=np.arange(0.5, len(tuple_list) + 1.5, 1.0),
            cmap=cmap,
            norm=norm,
            alpha=0.62,
        )
        ax.plot(
            k1_sn[mask1], e1_sn[mask1], linestyle="--", color="black", linewidth=1.9
        )
        ax.plot(k2_sn[mask2], e2_sn[mask2], linestyle=":", color="black", linewidth=2.1)
        for kc, ec in hopf1_by_tau[tau]:
            ax.plot(kc, ec, color="tab:red", linewidth=1.8)
        for kc, ec in hopf2_by_tau[tau]:
            ax.plot(kc, ec, color="tab:blue", linestyle="-.", linewidth=1.8)
        for code, tup in code_to_tuple.items():
            pos = largest_component_centroid(CODE, code)
            if pos is None:
                continue
            y_idx, x_idx = pos
            ax.text(
                kappas[x_idx],
                etas[y_idx],
                tuple_string(tup),
                ha="center",
                va="center",
                fontsize=8.8,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
            )
        ax.axhline(0.0, linewidth=0.9, alpha=0.35, color="black")
        ax.axvline(0.0, linewidth=0.9, alpha=0.35, color="black")
        ax.set_xlim(KMIN, KMAX)
        ax.set_ylim(EMIN, EMAX)
        ax.set_title(rf"$\tau={tau}$")

    chosen_kappas = [4.0, -0.1, -3.0, -1.3, 3.0]
    chosen_etas = [0.92, 0.9, -0.5, -0.025, -0.4]
    labels = ["1", "2", "3.1", "3.2", "4"]
    for ax in axes:
        for kappa, eta, label in zip(chosen_kappas, chosen_etas, labels, strict=True):
            ax.text(
                kappa,
                eta,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                bbox={
                    "boxstyle": "circle,pad=0.3",
                    "facecolor": "black",
                    "edgecolor": "none",
                },
                zorder=5,
            )

    axes[0].set_ylabel(r"$\eta$")
    axes[2].set_ylabel(r"$\eta$")
    axes[2].set_xlabel(r"$\kappa$")
    axes[3].set_xlabel(r"$\kappa$")

    curve_handles = [
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="black",
            linewidth=1.9,
            label=r"$\Gamma_{SN}^{(1)}$",
        ),
        Line2D(
            [0],
            [0],
            linestyle=":",
            color="black",
            linewidth=2.1,
            label=r"$\Gamma_{SN}^{(2)}$",
        ),
        Line2D(
            [0],
            [0],
            linestyle="-",
            color="tab:red",
            linewidth=1.8,
            label=r"$\Gamma_{H}^{(1)}$",
        ),
        Line2D(
            [0],
            [0],
            linestyle="-.",
            color="tab:blue",
            linewidth=1.8,
            label=r"$\Gamma_{H}^{(2)}$",
        ),
    ]
    fig.legend(
        handles=curve_handles,
        loc="lower center",
        ncol=4,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.018),
        handlelength=3.0,
        columnspacing=1.8,
    )

    fig.subplots_adjust(
        left=0.07, right=0.99, top=0.95, bottom=0.10, wspace=0.08, hspace=0.14
    )
    fig.savefig(
        "attractor_classification_4panel_shared_legend_updated_colors_fullscript.pdf",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
