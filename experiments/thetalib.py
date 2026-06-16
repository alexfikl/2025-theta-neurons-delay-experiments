from __future__ import annotations

import numpy as np

TOL_COEFF = 1e-12
TOL_REAL = 1e-10
TOL_MERGE = 1e-7
TOL_STAB = 1e-9
KAPPA_EPS = 1e-6

# {{{ type1


def poly_roots(coeffs, tol=TOL_COEFF):
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
    return 3.0 * kappa * c**2 - 2.0 * kappa * c + (eta - kappa - 1.0)


def first_tau_and_l1_type1_dirac(kappa, eta):
    """
    Return (tau_c, l1) for the first Hopf bifurcation of a type 1 equilibrium
    that is asymptotically stable at tau=0, in the Dirac-kernel case.
    If no such Hopf point exists, return (nan, nan).
    """
    coeffs = [kappa, -kappa, eta - kappa - 1.0, eta + kappa + 1.0]
    c_roots = distinct_real_roots_in_open_interval(coeffs)

    candidates = []

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
        E = np.exp(-1j * theta)

        # Taylor coefficients on rho = 1
        s = -np.sqrt(1.0 - c**2)  # lower branch phi* in (-pi,0)
        A20 = 2.0 * c / (1.0 + c)
        A11 = -2.0 * kappa * (1.0 - c) ** 2 * (1.0 + c)
        A02 = 2.0 * kappa * (2.0 * c + 1.0) * (1.0 - c**2)

        A30 = -2.0 * s / (1.0 + c)
        A21 = -2.0 * kappa * c * (1.0 - c) * s
        A12 = -2.0 * kappa * (2.0 * c + 1.0) * (1.0 - c) * s
        A03 = 2.0 * kappa * (1.0 + c) * (4.0 * c - 1.0) * s

        def Delta(lam, alpha=alpha, beta=beta, tau_c=tau_c):  # noqa: N802
            return lam - alpha - beta * np.exp(-tau_c * lam)

        # Quadratic center-manifold coefficients
        p = (A20 + 2.0 * A11 * E + A02 * E**2) / Delta(2j * omega)
        m = (A20 + A11 * (E + np.conj(E)) + A02) / Delta(0.0)

        # Cubic resonant coefficient
        R = (
            0.5 * (A30 + A21 * (2.0 * E + np.conj(E)) + A12 * (2.0 + E**2) + A03 * E)
            + A20 * (m + 0.5 * p)
            + A11 * (m * (1.0 + E) + 0.5 * p * (E**2 + np.conj(E)))
            + A02 * E * (m + 0.5 * p)
        )

        ccoef = R / (1.0 + beta * tau_c * E)
        l1 = np.real(ccoef) / (2.0 * omega)

        candidates.append((tau_c, l1))

    if not candidates:
        return np.nan, np.nan

    # First Hopf point
    candidates.sort(key=lambda item: item[0])
    return candidates[0]


# }}}


# {{{ type2


def type2_equilibrium_roots(kappa, eta):
    coeffs = [
        0.5 * kappa,
        -kappa,
        eta - 2.0 * kappa - 1.0,
        2.0 * eta + kappa + 2.0,
        eta + 1.5 * kappa - 1.0,
    ]
    return distinct_real_roots_in_open_interval(coeffs)


def Fprime_type2(kappa, z):  # noqa: N802
    return kappa * (2.0 - z) - 4.0 * (1.0 - z) / (1.0 + z) ** 3


def type2_center_root(kappa, eta):
    roots = type2_equilibrium_roots(kappa, eta)
    if len(roots) == 0:
        return None

    centers = [z for z in roots if Fprime_type2(kappa, z) < 0.0]
    if not centers:
        return None

    return min(centers)


def d0_d1_from_center(kappa, eta):
    zeta = type2_center_root(kappa, eta)
    if zeta is None:
        return None
    d0 = 4.0 * ((1.0 - zeta) / (1.0 + zeta)) ** 2
    d1 = kappa * (1.0 - zeta**2) * (2.0 - zeta)
    return zeta, d0, d1


def B_apply(B, X, Y):  # noqa: N802
    return np.einsum("aij,i,j->a", B, X, Y)


def C_apply(C, X, Y, Z):  # noqa: N802
    return np.einsum("aijk,i,j,k->a", C, X, Y, Z)


def BC_tensors(zeta, kappa):  # noqa: N802
    z = zeta
    k = kappa

    B = np.zeros((2, 4, 4), dtype=complex)
    C = np.zeros((2, 4, 4, 4), dtype=complex)

    B[0, 0, 1] = B[0, 1, 0] = 4.0 * z / (1.0 + z) ** 2
    B[0, 1, 2] = B[0, 2, 1] = -k * (z - 2.0) * (z + 1.0)

    B[1, 0, 0] = -4.0 * z / (1.0 + z) ** 2
    B[1, 1, 1] = 4.0 * z / (1.0 + z) ** 2
    B[1, 0, 2] = B[1, 2, 0] = k * (z - 2.0) * (z + 1.0)
    B[1, 2, 2] = 0.5 * k * (z + 1.0) ** 2
    B[1, 3, 3] = -0.5 * k * (z + 1.0) ** 2

    entries = [
        (0, 0, 1, 2, -k * (z - 2.0)),
        (0, 0, 2, 1, -k * (z - 2.0)),
        (0, 1, 0, 2, -k * (z - 2.0)),
        (0, 1, 2, 0, -k * (z - 2.0)),
        (0, 1, 2, 2, -k * (z + 1.0)),
        (0, 1, 3, 3, k * (z + 1.0)),
        (0, 2, 0, 1, -k * (z - 2.0)),
        (0, 2, 1, 0, -k * (z - 2.0)),
        (0, 2, 1, 2, -k * (z + 1.0)),
        (0, 2, 2, 1, -k * (z + 1.0)),
        (0, 3, 1, 3, k * (z + 1.0)),
        (0, 3, 3, 1, k * (z + 1.0)),
        (1, 0, 0, 2, k * (z - 2.0)),
        (1, 0, 2, 0, k * (z - 2.0)),
        (1, 0, 2, 2, k * (z + 1.0)),
        (1, 0, 3, 3, -k * (z + 1.0)),
        (1, 1, 1, 2, -k * (z - 2.0)),
        (1, 1, 2, 1, -k * (z - 2.0)),
        (1, 2, 0, 0, k * (z - 2.0)),
        (1, 2, 0, 2, k * (z + 1.0)),
        (1, 2, 1, 1, -k * (z - 2.0)),
        (1, 2, 2, 0, k * (z + 1.0)),
        (1, 3, 0, 3, -k * (z + 1.0)),
        (1, 3, 3, 0, -k * (z + 1.0)),
    ]
    for comp, i, j, l, val in entries:  # noqa: E741
        C[comp, i, j, l] = val

    return B, C


def tau1_is_relevant(kappa, d0, d1):
    if kappa > KAPPA_EPS:
        return True
    if kappa < -KAPPA_EPS:
        return 3.0 * d0 > -5.0 * d1
    return False


def tau1_and_l1_type2_dirac(kappa, eta):
    data = d0_d1_from_center(kappa, eta)
    if data is None:
        return np.nan, np.nan

    zeta, d0, d1 = data
    if not tau1_is_relevant(kappa, d0, d1):
        return np.nan, np.nan

    a = 2.0 * (1.0 - zeta) / (1.0 + zeta)
    b = 0.5 * kappa * (1.0 + zeta) ** 2 * (2.0 - zeta)

    disc = a * a + a * b
    if disc <= 0.0:
        return np.nan, np.nan

    omega = np.sqrt(disc)
    tau1 = np.pi / omega
    E = -1.0 + 0.0j

    A0 = np.array([[0.0, -a], [a, 0.0]], dtype=complex)
    A1 = np.array([[0.0, 0.0], [-b, 0.0]], dtype=complex)

    q = np.array([1.0, -1j * omega / a], dtype=complex)
    p = np.array([1.0, 1j * a / omega], dtype=complex)

    denom = np.vdot(p, (np.eye(2, dtype=complex) + tau1 * A1 * E) @ q)
    if abs(denom) < 1e-14:
        return tau1, np.nan
    gamma = 1.0 / denom

    Q = np.concatenate([q, E * q])
    Qbar = np.conjugate(Q)

    B, C = BC_tensors(zeta, kappa)

    h20 = B_apply(B, Q, Q)
    h11 = B_apply(B, Q, Qbar)
    h02 = B_apply(B, Qbar, Qbar)

    g20 = gamma * np.vdot(p, h20)
    g11 = gamma * np.vdot(p, h11)
    g02 = gamma * np.vdot(p, h02)

    try:
        E1 = np.linalg.solve(2j * omega * np.eye(2) - A0 - A1 * (E**2), h20)
        E2 = np.linalg.solve(-(A0 + A1), h11)
    except np.linalg.LinAlgError:
        return tau1, np.nan

    W20_0 = (
        1j * g20 / omega * q
        + 1j * np.conjugate(g02) / (3.0 * omega) * np.conjugate(q)
        + E1
    )
    W20_t = (
        1j * g20 / omega * E * q
        + 1j * np.conjugate(g02) / (3.0 * omega) * np.conjugate(E * q)
        + (E**2) * E1
    )

    W11_0 = (
        -1j * g11 / omega * q + 1j * np.conjugate(g11) / omega * np.conjugate(q) + E2
    )
    W11_t = (
        -1j * g11 / omega * E * q
        + 1j * np.conjugate(g11) / omega * np.conjugate(E * q)
        + E2
    )

    H20 = np.concatenate([W20_0, W20_t])
    H11 = np.concatenate([W11_0, W11_t])

    h21 = C_apply(C, Q, Q, Qbar) + B_apply(B, Qbar, H20) + 2.0 * B_apply(B, Q, H11)
    g21 = gamma * np.vdot(p, h21)

    c1 = (
        1j / (2.0 * omega) * (g20 * g11 - 2.0 * abs(g11) ** 2 - abs(g02) ** 2 / 3.0)
        + 0.5 * g21
    )
    l1 = np.real(c1) / omega
    if not np.isfinite(tau1) or not np.isfinite(l1):
        return np.nan, np.nan
    return tau1, l1


# }}}
