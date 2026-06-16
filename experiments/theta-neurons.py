# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import enum
import logging
import multiprocessing
import pathlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import numpy.linalg as la
import rich.logging
from scipy.spatial import Voronoi

import orbitkit.symbolic.primitives as sym
from orbitkit.codegen.jitcdde import JiTCDDECompiledCode, JiTCDDETarget
from orbitkit.models.theta import FixedPoints, ThetaModel, find_fixed_points
from orbitkit.utils import slugify
from orbitkit.visualization import figure, rgbf_lerp_p, set_plotting_defaults

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import matplotlib.pyplot as mp

    from orbitkit.typing import Array1D, Array2D

log = logging.getLogger("theta")
log.propagate = False
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())


def set_recommended_matplotlib() -> None:
    dirname = pathlib.Path(__file__).parent
    set_plotting_defaults(dirname / "default.mplstyle", use_tex=True, dark=False)


set_recommended_matplotlib()


# {{{ settings


@dataclass(frozen=True)
class ParameterSet:
    descr: str
    """A short description of this parameter set."""

    kappa: float
    eta: float

    tfinal: float
    taus: tuple[float, ...]
    """A list of delays to use for the simulations."""

    npoints_per_sector: tuple[int, ...]
    """Number of random points to generate in each sector."""
    sectors: tuple[tuple[float, float, float, float], ...]
    """Each disk sector is given as a 4-tuple of ``(rmin, rmax, tmin, tmax)``,
    where the angles are given in degrees (not radians).
    """


@dataclass(frozen=True)
class Settings:
    filename: pathlib.Path
    """Configuration file path."""
    params: dict[str, ParameterSet]
    """A dictionary of parameter sets to simulate."""


def parse_settings(
    filename: pathlib.Path,
    *,
    select: Sequence[str] | None = None,
    tau: float | None = None,
    npoints: int | None = None,
) -> Settings:
    import tomllib

    if not filename.exists():
        log.error("Filename does not exist: '%s'", filename)

    with open(filename, "rb") as fp:
        data = tomllib.load(fp)
    selected = set(select) if select else set(data.get("main", {}).get("select", []))

    params = {}
    for name, param in data.get("params", {}).items():
        if selected and name not in selected:
            continue

        # taus
        taus = param.get("taus", []) if tau is None else [tau]
        if not isinstance(taus, list):
            raise ValueError(f"{filename.name}: params.{name}: taus is not a list")

        if not taus:
            raise ValueError(f"{filename.name}: params.{name}: taus is an empty list")

        if any(tau < 0.0 for tau in taus):
            raise ValueError(f"{filename.name}: params.{name}: taus must be positive")

        # kappa
        s_kappa = param.get("kappa")
        if s_kappa is None:
            raise ValueError(f"{filename.name}: params.{name}: kappa is not provided")

        if not isinstance(s_kappa, float):
            raise ValueError(f"{filename.name}: params.{name}: kappa is not a float")

        # eta
        s_eta = param.get("eta")
        if s_eta is None:
            raise ValueError(f"{filename.name}: params.{name}: eta is not provided")

        if not isinstance(s_eta, float):
            raise ValueError(f"{filename.name}: params.{name}: eta is not a float")

        # tfinal
        s_tfinal = param.get("tfinal", 100.0)
        if not isinstance(s_tfinal, float):
            raise ValueError(f"{filename.name}: params.{name}: tfinal is not a float")

        if s_tfinal <= 0.0:
            raise ValueError(f"{filename.name}: params.{name}: tfinal must be positive")

        # npoints
        s_npoints = param.get("npoints", 2048) if npoints is None else npoints
        if not isinstance(s_npoints, int):
            raise ValueError(f"{filename.name}: params.{name}: npoints is not an int")

        # sectors
        eps = 1.0e-4
        sectors = param.get("sectors")
        if sectors is None:
            sectors = [(0.0, 1.0 - eps, 0.0, 2.0 * np.pi)]
            npoints_per_sector = [s_npoints]
        else:
            if any(len(sector) != 4 for sector in sectors):
                raise ValueError(
                    f"{filename.name}: params.{name}: sectors must be 4-tuples"
                )

            # FIXME: should also insist that angles are in [0, 2pi]?
            sectors = [
                (
                    max(0.0, min(1.0 - eps, float(rmin))),
                    max(0.0, min(1.0 - eps, float(rmax))),
                    np.radians(thetamin),
                    np.radians(thetamax),
                )
                for rmin, rmax, thetamin, thetamax in sectors
            ]
            npoints_per_sector = [s_npoints] * len(sectors)

        # NOTE: force some points on the circle so the plots look nicer
        # sectors.append((1.0 - eps, 1.0 - eps, 0.0, 360.0))
        # npoints_per_sector.append(min(s_npoints, 256))

        if not sectors:
            raise ValueError(
                f"{filename.name}: params.{name}: sectors must be non-empty"
            )

        params[name] = ParameterSet(
            descr=param.get("descr", name),
            kappa=s_kappa,
            eta=s_eta,
            taus=tuple(taus),
            tfinal=s_tfinal,
            npoints_per_sector=tuple(npoints_per_sector),
            sectors=tuple(sectors),
        )

    return Settings(filename=filename, params=params)


# }}}


# {{{ solve_ivp


def generate_code(
    model: ThetaModel,
    *,
    module_location: pathlib.Path,
    max_delay: float = 0.0,
) -> JiTCDDECompiledCode:
    log.info("Model: %s", type(model))
    log.info("Equations:\n%s", model)

    # codegen
    target = JiTCDDETarget()
    code = target.generate_model_code(model, model.n)
    integrator = target.compile(code, module_location=module_location)

    return integrator


def solve_ivp(
    code: JiTCDDECompiledCode,
    model: ThetaModel,
    z0: Array1D[np.complexfloating[Any]],
    *,
    tspan: tuple[float, float],
    dt: float = 0.01,
    atol: float = 1.0e-9,
    rtol: float = 1.0e-6,
) -> tuple[Array1D[np.floating[Any]], Array1D[np.complexfloating[Any]]]:
    if code.module_location is not None and not code.module_location.exists():
        raise FileNotFoundError(code.module_location)

    from orbitkit.utils import TicTocTimer

    log.setLevel(logging.INFO)
    timer = TicTocTimer()
    timer.tic()

    eta = model.eta
    assert isinstance(eta, (int, float, np.floating))
    kappa = model.kappa
    assert isinstance(kappa, (int, float, np.floating))
    tau = model.h.avg
    assert isinstance(tau, (int, float, np.floating))

    # {{{ compile and set parameters

    code.reset()
    y0 = np.hstack([z0.real, z0.imag])

    code.set_initial_conditions(y0, tspan[0])
    code.set_parameters(eta=eta, kappa=kappa, tau=tau)

    # NOTE: using adjust_diff seems to give results a lot closer some literature.
    # Maybe that's what MATLAB uses as well? Or similar at least..
    # code.step_on_discontinuities()
    code.adjust_diff()

    # }}}

    # {{{ evolve

    ts = np.arange(tspan[0], tspan[1], dt)
    ys = np.empty(ts.size, dtype=z0.dtype)

    from jitcdde import UnsuccessfulIntegration

    try:
        for n in range(ts.size):
            result, _, _ = code.integrate(ts[n])
            ys[n] = result[0] + 1j * result[1]
    except UnsuccessfulIntegration as exc:
        log.error("Failed to integrate: %s.", exc, exc_info=exc)
        log.error("Model: %s at z0 = %s", model, z0)

        # skip this solution
        ys.fill(np.nan)

    if (linf_norm := np.linalg.norm(ys, ord=np.inf)) > 1.1:
        log.error("Solutions escaped the unit disk: %g.", linf_norm)
        log.error("Model: %s at z0 = %s", model, z0)
        ys.fill(np.nan)

    # }}}

    timer.toc()
    log.info(
        "Solved at (kappa, eta) = (%+.3f, %+.3f) for z0 = %+.5f%+.5fj: %s",
        model.kappa,
        model.eta,
        z0.real,
        z0.imag,
        timer,
    )

    return ts, ys


# }}}


# {{{ classify solutions


@enum.unique
class SolutionType(enum.IntEnum):
    OnCircle = 0
    """A solution with an equilibrium point on the unit circle."""
    InDisk = 1
    """An solution with an equilibrium point on the real line in the unit disk."""
    Cycle = 2
    """A solution with a limit cycle."""
    Unknown = 3
    """An unknown type of solution."""


@dataclass(frozen=True)
class SolutionGroup:
    equilibrium_point_index: int
    """An index into the array of equilibrium points, if any."""
    indices: Array1D[np.integer[Any]]
    """A list of indices into the solution array for this type of solution."""


@dataclass(frozen=True)
class SolutionGroups:
    ts: Array1D[np.floating[Any]]
    zs: Array2D[np.complexfloating[Any]]
    groups: dict[SolutionType, dict[int, SolutionGroup]]
    """A mapping from a solution type to a dict of solution groups. The key in
    the inner dict is an index ID for the solution itself.
    """

    def items(self) -> Iterator[tuple[SolutionType, int, SolutionGroup]]:
        for stype, grps in self.groups.items():
            for sindex, grp in grps.items():
                yield stype, sindex, grp


def match_fixed_point(
    fp: FixedPoints,
    zh: Array1D[np.complexfloating[Any]],
    *,
    eps: float = 1.0e-3,
) -> tuple[SolutionType, int]:
    for i, zfp in enumerate(fp.on_circle):
        if la.norm(zh - zfp) < eps * la.norm(zfp):
            return SolutionType.OnCircle, i

    for i, zfp in enumerate(fp.in_disk):
        if la.norm(zh - zfp) < eps * la.norm(zfp):
            return SolutionType.InDisk, i + fp.on_circle.size

    # TODO: is this actually useful? maybe in debug mode?
    maybe_error = la.norm(zh - zh[-1]) / la.norm(zh[-1])
    maybe_equilibrium_point = maybe_error < 10.0 * eps

    if maybe_equilibrium_point:
        log.warning(
            "Solution might be an equilibrium point, but the time span "
            "is not sufficiently large to determine it to the given "
            "tolerance 'eps = %.8e': error %.8e last diffs %s",
            eps,
            maybe_error,
            np.abs(np.diff(zh))[-5:],
        )

    return SolutionType.Unknown, -1


def match_limit_cycle(
    lcs: list[tuple[np.floating[Any], np.floating[Any]]],
    zh: Array1D[np.complexfloating[Any]],
    *,
    eps: float = 1.0e-3,
) -> tuple[SolutionType, int]:
    # FIXME: this is very crude and uses knowledge of our system:
    # - we know that the system only has at most two cycles
    # - we know that the cycles all go around the circle and differ in section

    zh_real = np.linalg.norm(zh.real, ord=np.inf)
    zh_imag = np.linalg.norm(zh.imag, ord=np.inf)

    for i, (lc_real, lc_imag) in enumerate(lcs):
        if (
            np.abs(zh_real - lc_real) < eps * lc_real
            and np.abs(zh_imag - lc_imag) < eps * lc_imag
        ):
            return SolutionType.Cycle, i

    if len(lcs) > 1:
        log.error("We do not have more than two limit cycles. What is this??")
        log.info("Point (%g, %g) LCs %s", zh_real, zh_imag, lcs)

        return SolutionType.Cycle, min(
            range(len(lcs)),
            key=lambda i: (
                np.abs(zh_real - lcs[i][0]) < eps * lcs[i][0]
                and np.abs(zh_imag - lcs[i][1]) < eps * lcs[i][1]
            ),
        )

    lcs.append((zh_real, zh_imag))
    return SolutionType.Cycle, len(lcs) - 1


def make_solution_groups(
    ts: Array1D[np.floating[Any]],
    zs: Array2D[np.complexfloating[Any]],
    fp: FixedPoints,
    *,
    min_points_per_group: int = 4,
    nhistory: int | None = None,
    eps: float = 1.0e-2,
) -> SolutionGroups:
    npoints, ntimesteps = zs.shape
    if nhistory is None:
        nhistory = int(0.1 * ntimesteps)

    lcs = []
    tmp_solution_groups: dict[SolutionType, dict[int, list[int]]] = {}
    for n in range(npoints):
        # find a matching equilibrium point, if any
        zh = zs[n, -nhistory:]
        stype, solution_type_index = match_fixed_point(fp, zh, eps=eps)

        # find a matching limit cycle, if any
        if stype == SolutionType.Unknown:
            stype, solution_type_index = match_limit_cycle(lcs, zh, eps=eps)

        sgrp = tmp_solution_groups.setdefault(stype, {})
        sgrp.setdefault(solution_type_index, []).append(n)

    # NOTE: sort indices by length of the trajectory in phase space. This is
    # mostly done for visualization purposes and to prune out small groups.
    dz = np.sum(np.abs(np.diff(zs[:, 1:], axis=1)), axis=1)

    istart = 0
    result: dict[SolutionType, dict[int, SolutionGroup]] = {}
    zs_new = np.empty_like(zs)
    for stype, grps in tmp_solution_groups.items():
        igrp = 0
        for sindex, indices in grps.items():
            if len(indices) < min_points_per_group:
                log.info(
                    "Skipping group '%s' with only %d points. Add more points or "
                    "increase the time horizon if this is an error.",
                    stype.name,
                    len(indices),
                )
                continue

            # make sure that the index indexes into the various arrays
            if stype == SolutionType.OnCircle:
                equilibrium_point_index = sindex
            elif stype == SolutionType.InDisk:
                equilibrium_point_index = sindex - fp.on_circle.size
            else:
                equilibrium_point_index = sindex

            # sort indices and select reorder the solutions
            nindices = len(indices)
            indices_new = sorted(indices, key=lambda n: dz[n])
            zs_new[istart : istart + nindices] = zs[indices_new]

            # save the indices in direct order
            result.setdefault(stype, {})[igrp] = SolutionGroup(
                equilibrium_point_index=equilibrium_point_index,
                indices=np.arange(istart, istart + nindices),
            )
            istart += nindices
            igrp += 1

    return SolutionGroups(ts=ts, zs=zs_new[:istart].copy(), groups=result)


# }}}


# {{{ plotting

# NOTE: colors from the default 'matlab' style
DEFAULT_COLORS = {
    # yellow
    (SolutionType.OnCircle, 0): "#E8C830",
    (SolutionType.OnCircle, 1): "#D4A820",
    # teal
    (SolutionType.InDisk, 0): "#20AAAA",
    (SolutionType.InDisk, 1): "#38C4C4",
    # dark blue
    (SolutionType.Cycle, 0): "#1A5FA8",
    (SolutionType.Cycle, 1): "#3A9FD8",
}

ProjectCallable: TypeAlias = (
    "Callable[[Array2D[np.floating[Any]]], Array1D[np.floating[Any]]]"
)


def index_to_color_map(
    sgs: SolutionGroups,
    *,
    min_points: int | None = None,
) -> dict[int, str]:
    if min_points is None:
        min_points = int(0.05 * sgs.zs.shape[0])

    index_to_color: dict[int, str] = {}
    for stype, sindex, grp in sgs.items():
        color = DEFAULT_COLORS[stype, sindex]
        if len(grp.indices) < min_points:
            color = DEFAULT_COLORS[stype, 1 - sindex]

        for i in grp.indices:
            index_to_color[i] = color

    return index_to_color


def add_circle(ax: mp.Axes) -> None:
    from matplotlib.patches import Circle

    circle = Circle((0.0, 0.0), 1.0, color="k", lw=2, fill=False)
    ax.add_patch(circle)


def add_fixed_points(ax: mp.Axes, sgs: SolutionGroups, fp: FixedPoints) -> None:
    zfp = fp.points

    ax.plot(zfp.real, zfp.imag, "X", ms=20, color="k")
    for stype, sindex, grp in sgs.items():
        if stype == SolutionType.OnCircle:
            ofp = fp.on_circle[grp.equilibrium_point_index]
        elif stype == SolutionType.InDisk:
            ofp = fp.in_disk[grp.equilibrium_point_index]
        else:
            continue

        color = DEFAULT_COLORS[stype, sindex]
        ax.plot(ofp.real, ofp.imag, "X", ms=10, color=color)


def clip_vertex_to_disk(
    x: Array2D[np.floating[Any]], radius: float = 1.0
) -> Array2D[np.floating[Any]]:
    r = la.norm(x, ord=2)
    if r <= radius:
        return x

    return radius / r * x


def clip_vertex_to_square(
    x: Array2D[np.floating[Any]], radius: float = 1.5 * np.sqrt(2.0)
) -> Array2D[np.floating[Any]]:
    s = radius / np.sqrt(2.0)
    if -s <= x[0] <= s and -s <= x[1] <= s:
        return x

    return np.array(np.minimum(np.maximum(x, -s), s))


def project_regions(
    vr: Voronoi,
    ridges: dict[int, list[tuple[int, int, int]]],
    p1: int,
    rindex: int,
    *,
    project: ProjectCallable | None = None,
    radius: float = 2.0,
) -> Array2D[np.floating[Any]]:
    if project is None:
        project = clip_vertex_to_disk

    vertices = [project(vr.vertices[v]) for v in vr.regions[rindex] if v != -1]
    center = np.mean(vr.points, axis=0)
    if len(vertices) == len(vr.regions[rindex]):
        return np.array(vertices)

    for p2, v1, v2 in ridges[p1]:
        if v2 < 0:
            v1, v2 = v2, v1  # noqa: PLW2901

        if v1 >= 0:
            continue

        t = vr.points[p2] - vr.points[p1]
        t /= la.norm(t)
        n = np.array([-t[1], t[0]])

        midpoint = (vr.points[p1] + vr.points[p2]) / 2.0
        direction = np.sign((midpoint - center) @ n) * n
        farpoint = vr.vertices[v2] + direction * radius
        vertices.append(project(farpoint))

    # sort clockwise for fill
    vs = np.array(vertices)
    cs = np.mean(vs, axis=0)
    angles = np.arctan2(vs[:, 1] - cs[1], vs[:, 0] - cs[0])

    return vs[np.argsort(angles)]


def lloyd_relaxation(
    points: Array2D[np.floating[Any]],
    *,
    iterations: int = 10,
    radius: float = 1.0,
) -> Array2D[np.floating[Any]]:
    """Redistribute points toward a uniform layout using Lloyd's algorithm.

    Each iteration replaces every seed with the centroid of its Voronoi cell,
    clipped to the disk of the given radius. This does not change the coloring
    (labels are re-assigned by nearest-neighbour lookup after relaxation); it
    only makes the cells more equal in size, smoothing the rendered boundaries.
    """
    pts = points.copy()

    for _ in range(iterations):
        vr = Voronoi(pts, qhull_options="Qbb Qc Qx")

        ridges: dict[int, list[tuple[int, int, int]]] = {}
        for (p1, p2), (v1, v2) in zip(vr.ridge_points, vr.ridge_vertices, strict=True):
            ridges.setdefault(p1, []).append((p2, v1, v2))
            ridges.setdefault(p2, []).append((p1, v1, v2))

        new_pts = np.empty_like(pts)
        for i, rindex in enumerate(vr.point_region):
            poly = project_regions(
                vr, ridges, i, rindex, project=clip_vertex_to_disk, radius=radius
            )
            new_pts[i] = np.mean(poly, axis=0)

        # clamp: centroids of boundary cells can drift just outside the disk
        r = np.linalg.norm(new_pts, axis=1, keepdims=True)
        outside = (r > radius).ravel()
        new_pts[outside] = new_pts[outside] / r[outside] * radius

        pts = new_pts

    return pts


def plot_regions(
    filename: pathlib.Path,
    sgs: SolutionGroups,
    fp: FixedPoints,
    *,
    lloyd_iterations: int = 32,
    overwrite: bool = False,
) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import ListedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.interpolate import NearestNDInterpolator

    zs = sgs.zs
    index_to_color = index_to_color_map(sgs)

    with figure(filename, overwrite=overwrite) as fig:
        ax = fig.gca()

        # plot regions of attraction
        points = np.vstack([zs[:, 0].real, zs[:, 0].imag]).T

        if lloyd_iterations > 0:
            # relax seeds toward a uniform layout, then re-color by nearest
            # neighbour lookup so no re-simulation is needed
            sim_colors = np.array([index_to_color[i] for i in range(len(points))])
            relaxed = lloyd_relaxation(points, iterations=lloyd_iterations)
            nn = NearestNDInterpolator(points, np.arange(len(points)))
            facecolor_array = sim_colors[nn(relaxed).astype(np.intp)]
            points = relaxed
        else:
            facecolor_array = np.array([index_to_color[i] for i in range(len(points))])

        vr = Voronoi(points, qhull_options="Qbb Qc Qx")

        ridges: dict[int, list[tuple[int, int, int]]] = {}
        for (p1, p2), (v1, v2) in zip(vr.ridge_points, vr.ridge_vertices, strict=True):
            ridges.setdefault(p1, []).append((p2, v1, v2))
            ridges.setdefault(p2, []).append((p1, v1, v2))

        polygons = []
        facecolors = []
        for i, rindex in enumerate(vr.point_region):
            polygon = project_regions(
                vr, ridges, i, rindex, project=clip_vertex_to_disk
            )

            facecolors.append(facecolor_array[i])
            polygons.append(polygon)

        cells = PolyCollection(
            polygons,
            edgecolor="face",
            facecolors=facecolors,
            rasterized=True,
        )
        ax.add_collection(cells)

        # plot fixed points
        add_circle(ax)
        add_fixed_points(ax, sgs, fp)

        # customize axes
        ax.set_xlabel(r"$\Re\{z\}$")
        ax.set_ylabel(r"$\Im\{z\}$")
        ax.set_aspect("equal")
        ax.grid(visible=False, which="both")

        # create a new mappable to show all the colors in the colorbar
        colors = list(DEFAULT_COLORS.values())
        sm = ScalarMappable(cmap=ListedColormap(colors))
        sm.set_clim(vmin=0, vmax=len(colors))

        # create a new axis that matches the height of the other ones
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)

        # set up the colorbar ticks
        cbar = fig.colorbar(sm, cax=cax, ticks=list(range(1, len(colors), 2)))
        cbar.set_ticklabels(["Type 1", "Type 2", "Cycle"], va="center", rotation=-90)
        cbar.ax.hlines(
            list(range(2, len(colors), 2)),
            0,
            1,
            colors="black",
            linewidths=8,
            transform=cbar.ax.transData,
        )


def plot_regions_imshow(
    filename: pathlib.Path,
    sgs: SolutionGroups,
    fp: FixedPoints,
    *,
    resolution: int = 512,
    overwrite: bool = False,
) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.interpolate import NearestNDInterpolator

    zfp = fp.points
    zs = sgs.zs

    # build a mapping from point index to an integer label (one per color slot)
    color_keys = list(DEFAULT_COLORS.keys())
    index_to_label: dict[int, int] = {}
    for stype, sindex, grp in sgs.items():
        label = color_keys.index((stype, sindex))
        for i in grp.indices:
            index_to_label[i] = label

    # (N, 2) array of simulation starting points and their integer labels
    sim_points = np.vstack([zs[:, 0].real, zs[:, 0].imag]).T
    labels = np.array(
        [index_to_label[i] for i in range(len(sim_points))], dtype=np.int32
    )

    colors = list(DEFAULT_COLORS.values())

    # build a regular pixel grid over the unit square
    lin = np.linspace(-1.0, 1.0, resolution)
    gx, gy = np.meshgrid(lin, lin)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])

    # nearest-neighbour interpolation: assign each pixel the label of the
    # closest simulation starting point
    interp = NearestNDInterpolator(sim_points, labels)
    grid_labels = interp(grid_points).reshape(resolution, resolution).astype(float)

    # mask pixels that fall outside the unit disk
    grid_labels[gx**2 + gy**2 > 1.0] = np.nan

    with figure(filename, overwrite=overwrite) as fig:
        ax = fig.gca()

        ax.imshow(
            grid_labels,
            origin="lower",
            extent=(-1.0, 1.0, -1.0, 1.0),
            cmap=ListedColormap(colors),
            vmin=-0.5,
            vmax=len(colors) - 0.5,
            interpolation="nearest",
            rasterized=True,
        )

        # unit-disk boundary
        circle = Circle((0.0, 0.0), 1.0, color="k", lw=2, fill=False)
        ax.add_patch(circle)

        ax.plot(zfp.real, zfp.imag, "X", ms=20, color="k")
        for stype, sindex, grp in sgs.items():
            if stype == SolutionType.OnCircle:
                ofp = fp.on_circle[grp.equilibrium_point_index]
            elif stype == SolutionType.InDisk:
                ofp = fp.in_disk[grp.equilibrium_point_index]
            else:
                continue

            color = DEFAULT_COLORS[stype, sindex]
            ax.plot(ofp.real, ofp.imag, "X", ms=10, color=color)

        ax.set_xlabel(r"$\Re\{z\}$")
        ax.set_ylabel(r"$\Im\{z\}$")
        ax.set_aspect("equal")
        ax.grid(visible=False, which="both")

        # colorbar — same style as plot_regions
        sm = ScalarMappable(cmap=ListedColormap(colors))
        sm.set_clim(vmin=0, vmax=len(colors))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.3)

        cbar = fig.colorbar(sm, cax=cax, ticks=list(range(1, len(colors), 2)))
        cbar.set_ticklabels(["Type 1", "Type 2", "Cycle"], va="center", rotation=-90)
        cbar.ax.hlines(
            list(range(2, len(colors), 2)),
            0,
            1,
            colors="black",
            linewidths=8,
            transform=cbar.ax.transData,
        )


def plot_trajectory_cart(
    filename: pathlib.Path,
    sgs: SolutionGroups,
    *,
    real: bool = True,
    nhistory: int | None = None,
    max_lc_trajectories: int = 16,
    max_fp_trajectories: int = 256,
    rng: np.random.Generator | None = None,
    overwrite: bool = False,
) -> None:
    if rng is None:
        rng = np.random.default_rng()

    ts = sgs.ts
    zs = sgs.zs
    if nhistory is None:
        nhistory = int(0.05 * ts.size)

    with figure(filename, figsize=(10, 5), overwrite=overwrite) as fig:
        ax = fig.gca()

        for stype, sindex, grp in sgs.items():
            if stype == SolutionType.Unknown:
                color = "k"
            else:
                color = DEFAULT_COLORS[stype, sindex]

            if stype == SolutionType.Cycle:
                nchosen = min(max_lc_trajectories, len(grp.indices))
            else:
                nchosen = min(max_fp_trajectories, len(grp.indices))

            for i, n in enumerate(rng.choice(grp.indices, nchosen, replace=False)):
                zs_n = zs[n, -nhistory:]
                ax.plot(
                    ts[-nhistory:],
                    zs_n.real if real else zs_n.imag,
                    color=color,
                    alpha=1.0 if i == 0 else 0.45,
                    lw=2 if i == 0 else 1,
                )

        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$\Re\{z\}$" if real else r"$\Im\{z\}$")
        ax.set_ylim((-1.05, 1.05))


def plot_trajectory_polar(
    filename: pathlib.Path,
    sgs: SolutionGroups,
    *,
    radius: bool = True,
    nhistory: int | None = None,
    max_lc_trajectories: int = 16,
    max_fp_trajectories: int = 256,
    rng: np.random.Generator | None = None,
    overwrite: bool = False,
) -> None:
    if rng is None:
        rng = np.random.default_rng()

    ts = sgs.ts
    zs = sgs.zs
    if nhistory is None:
        nhistory = int(0.05 * ts.size)

    with figure(filename, figsize=(10, 5), overwrite=overwrite) as fig:
        ax = fig.gca()

        for stype, sindex, grp in sgs.items():
            if stype == SolutionType.Unknown:
                color = "k"
            else:
                color = DEFAULT_COLORS[stype, sindex]

            if stype == SolutionType.Cycle:
                nchosen = min(max_lc_trajectories, len(grp.indices))
            else:
                nchosen = min(max_fp_trajectories, len(grp.indices))

            for i, n in enumerate(rng.choice(grp.indices, nchosen, replace=False)):
                zs_n = zs[n, -nhistory:]
                ax.plot(
                    ts[-nhistory:],
                    np.abs(zs_n) if radius else np.angle(zs_n),
                    color=color,
                    alpha=1.0 if i == 0 else 0.45,
                    lw=2 if i == 0 else 1,
                )

        ax.set_xlabel("$t$")
        ax.set_ylabel(r"$\rho$" if radius else r"$\phi$")
        ax.set_ylim((-0.05, 1.05) if radius else (-np.pi - 0.05, np.pi + 0.05))


def plot_phase(
    filename: pathlib.Path,
    sgs: SolutionGroups,
    fp: FixedPoints,
    *,
    nhistory: int | None = None,
    max_lc_trajectories: int | None = 1,
    max_fp_trajectories: int = 128,
    rng: np.random.Generator | None = None,
    overwrite: bool = False,
) -> None:
    from matplotlib.colors import to_rgb

    if rng is None:
        rng = np.random.default_rng()

    ts = sgs.ts
    zs = sgs.zs
    index_to_color = index_to_color_map(sgs)

    if nhistory is None:
        nhistory = int(1.0 * zs.shape[1])

    from orbitkit.visualization import plot_phase

    with figure(filename, overwrite=overwrite) as fig:
        ax = fig.gca()

        for stype, _, grp in sgs.items():
            if stype == SolutionType.Cycle:
                nchosen = (
                    min(max_lc_trajectories, len(grp.indices))
                    if max_lc_trajectories is not None
                    else len(grp.indices)
                )
                window = -nhistory
            else:
                nchosen = (
                    min(max_fp_trajectories, len(grp.indices))
                    if max_fp_trajectories is not None
                    else len(grp.indices)
                )
                window = 0

            for n in rng.choice(grp.indices, nchosen, replace=False):
                scolor = index_to_color[n]
                zn = zs[n, window:]
                r = np.abs(zn[-1]) if stype == SolutionType.Cycle else 1.0
                r = 1.0

                # change color based on radius
                color = rgbf_lerp_p((1.0, 1.0, 1.0), to_rgb(scolor), r)

                plot_phase(ax, zn.real, zn.imag, alpha=ts[window:] ** 0.25, color=color)

        # plot fixed points
        add_circle(ax)
        add_fixed_points(ax, sgs, fp)

        ax.set_xlabel(r"$\Re\{z\}$")
        ax.set_ylabel(r"$\Im\{z\}$")
        ax.set_aspect("equal")


def plot_trajectory_slices(
    filename: pathlib.Path,
    sgs: SolutionGroups,
    fp: FixedPoints,
    *,
    nhistory: int = 512,
    zfac: float = 4.0,
    height: float = 2.25,
    rng: np.random.Generator | None = None,
    overwrite: bool = False,
) -> None:
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import art3d, axes3d

    if rng is None:
        rng = np.random.default_rng()

    zfp = fp.points
    ts = sgs.ts
    zs = sgs.zs
    gray = (0, 0, 0, 0.65)

    # NOTE: this needs a bit of padding because otherwise it clips the labels
    with figure(filename, projection="3d", pad_inches=0.15, overwrite=overwrite) as fig:
        ax = fig.gca()
        assert isinstance(ax, axes3d.Axes3D)

        x, y = np.meshgrid(np.linspace(-1.2, 1.2), np.linspace(-1.2, 1.2))
        for sindex in range(3):
            zoffset = zfac * sindex
            ax.plot_surface(
                x,
                y,
                np.full_like(x, zoffset),
                color="k",
                alpha=0.15,
            )
        del x
        del y

        # plot trajectories
        for stype, sindex, grp in sgs.items():
            scolor = DEFAULT_COLORS[stype, sindex]
            zoffset = zfac * int(stype)

            # plot the unit circle
            circle = Circle((0.0, 0.0), 1.0, color=gray, lw=2, fill=False)
            art3d.pathpatch_2d_to_3d(circle, z=zoffset, zdir="z")
            ax.add_patch(circle)

            # scale time axis to fit between the slices
            if stype == SolutionType.Cycle:  # noqa: SIM108
                # scale time logarithmically in [0, 1]
                to = np.log(ts + 3.0e-16)
            else:
                to = ts**0.5
            to = height * (to - to[0]) / (to[-1] - to[0])

            # plot trajectories
            if stype == SolutionType.Cycle:
                last_n = grp.indices[-nhistory] if nhistory < grp.indices.size else -1
                indices = grp.indices[-nhistory:]
            else:
                last_n = min(nhistory, grp.indices.size)
                indices = rng.choice(grp.indices, last_n, replace=False)

            for n in indices:
                if np.linalg.norm(zs[n], ord=np.inf) > 1.0 + 1.0e-3:
                    continue

                last_n = max(last_n, n)

                # get a slightly changed color
                cfac = rng.uniform(0.65, 1.0)
                color = rgbf_lerp_p((1.0, 1.0, 1.0), to_rgb(scolor), cfac)

                ax.plot(
                    zs[n].real,
                    zs[n].imag,
                    zoffset + to,
                    alpha=0.75,
                    color=color,
                    linestyle="-",
                    linewidth=1,
                    rasterized=True,
                )

                # NOTE: this functions as a bit of shadow?
                # ax.plot(zs[n].real, zs[n].imag, zoffset, "k", alpha=0.01)

            # plot the last (i.e. longest) trajectory in gray to highlight it
            if last_n >= 0:
                ax.plot(
                    zs[last_n].real,
                    zs[last_n].imag,
                    zoffset + to,
                    alpha=0.45,
                    color=gray,
                    linestyle="--",
                    linewidth=1,
                    rasterized=True,
                )

            # plot fixed points
            tfp = np.full(zfp.shape, zoffset)
            ax.plot(zfp.real, zfp.imag, tfp, "o", ms=6, color=gray)

            if stype == SolutionType.InDisk:
                ofp = fp.in_disk[grp.equilibrium_point_index]
            elif stype == SolutionType.OnCircle:
                ofp = fp.on_circle[grp.equilibrium_point_index]
            else:
                continue

            ax.plot(ofp.real, ofp.imag, zoffset, "o", ms=3, color=scolor, zorder=10)

        ax.set_xlabel(r"$\Re\{z\}$")
        ax.set_ylabel(r"$\Im\{z\}$")

        # remove all tick labels
        ax.set_xticklabels([])
        ax.set_xticklabels([], minor=True)
        ax.set_yticklabels([])
        ax.set_yticklabels([], minor=True)
        ax.set_zticks([], [])
        ax.set_zticklabels([], minor=True)

        # fix all the plots to the same range so they don't jump around
        ax.set_zlim([0, 3 * zfac])
        ax.view_init(elev=15, azim=-60.0, roll=0)
        ax.set_box_aspect((1, 1, 1.1))

    # }}}


# {{{ main


def run_simulation(
    outfile: pathlib.Path,
    code: JiTCDDECompiledCode,
    model: ThetaModel,
    param: ParameterSet,
    *,
    rng: np.random.Generator | None = None,
    with_regions: bool = False,
    with_imshow: bool = False,
    with_complex: bool = False,
    with_polar: bool = False,
    with_phase: bool = False,
    with_trajectories: bool = False,
    max_workers: int | None = None,
    overwrite: bool = False,
) -> int:
    if rng is None:
        rng = np.random.default_rng(seed=42)

    if max_workers is None:
        max_workers = multiprocessing.cpu_count() // 2

    filename = outfile.with_suffix(".npz")
    if not overwrite and filename.exists():
        log.error("File already exists (use --overwrite to force): '%s'.", filename)
        return 1

    sectors = param.sectors
    npoints_per_sector = param.npoints_per_sector

    # {{{ set up random points

    from orbitkit.utils import generate_random_points_in_disk, tictoc

    with tictoc("generate points"):
        points = np.empty(sum(npoints_per_sector), dtype=np.complex128)

        i = 0
        for sector, npoints in zip(sectors, npoints_per_sector, strict=True):
            rmin, rmax, thetamin, thetamax = sector

            points[i : i + npoints] = generate_random_points_in_disk(
                npoints,
                rspan=(rmin, rmax),
                thetaspan=(thetamin, thetamax),
                dtype=points.dtype,
                rng=rng,
            )
            i += npoints

        log.info("Generated %d starting points.", len(points))

    # }}}

    # {{{ compute solutions

    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    log.info("Parameters: %s", param)
    log.info("Time span: [0.0, %g]", param.tfinal)

    zfp = find_fixed_points(model)
    log.info("Roots: Type1: %s", [f"{complex(fp):g}" for fp in zfp.on_circle])
    log.info("       Type2: %s", [f"{complex(fp):g}" for fp in zfp.in_disk])

    log.info("Solving %d Theta neuron equations.", len(points))

    worker = partial(
        solve_ivp,
        code,
        model,
        tspan=(0.0, param.tfinal),
        dt=0.01,
    )

    with tictoc("solve equations"):
        if max_workers <= 1:
            results = [worker(p) for p in iter(points)]
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # NOTE: all runs use the same time steps
                results = list(executor.map(worker, iter(points)))

        ts, _ = results[0]
        zs = np.vstack([z for _, z in results if np.all(np.isfinite(z))])

        log.info(
            "Found %d / %d infinite (or NaN) solutions.",
            points.size - zs.shape[0],
            points.size,
        )

        del results

    # }}}

    # {{{ plot

    log.info("Determining solution groups...")
    sgs = make_solution_groups(ts, zs, zfp)

    del ts
    del zs

    for stype, sindex, grp in sgs.items():
        log.info("%r: group %d: %d", stype, sindex, len(grp.indices))
    log.info("Kept %d out of %d points", sgs.zs.shape[0], points.shape[0])

    np.savez(
        filename,
        points=points,
        ts=sgs.ts,
        zs=sgs.zs,
        groups=sgs.groups,  # ty: ignore[invalid-argument-type]
        param=param,  # ty: ignore[invalid-argument-type]
        tau=model.h.tau,  # ty: ignore[unresolved-attribute]
    )

    try:
        if with_regions:
            filename = outfile.with_stem(f"{outfile.stem}_regions")
            plot_regions(filename, sgs, zfp, overwrite=overwrite)

        if with_imshow:
            filename = outfile.with_stem(f"{outfile.stem}_regions_imshow")
            plot_regions_imshow(filename, sgs, zfp, overwrite=overwrite)

        if with_complex:
            filename = outfile.with_stem(f"{outfile.stem}_real")
            plot_trajectory_cart(filename, sgs, real=True, rng=rng, overwrite=overwrite)

            filename = outfile.with_stem(f"{outfile.stem}_imag")
            plot_trajectory_cart(
                filename, sgs, real=False, rng=rng, overwrite=overwrite
            )

        if with_polar:
            filename = outfile.with_stem(f"{outfile.stem}_radius")
            plot_trajectory_polar(
                filename, sgs, radius=True, rng=rng, overwrite=overwrite
            )

            filename = outfile.with_stem(f"{outfile.stem}_angle")
            plot_trajectory_polar(
                filename, sgs, radius=False, rng=rng, overwrite=overwrite
            )

        if with_phase:
            filename = outfile.with_stem(f"{outfile.stem}_phase")
            plot_phase(filename, sgs, zfp, rng=rng, overwrite=overwrite)

        if with_trajectories:
            filename = outfile.with_stem(f"{outfile.stem}_trajectories")
            plot_trajectory_slices(filename, sgs, zfp, rng=rng, overwrite=overwrite)
    except FileExistsError as exc:
        log.error("%s (use --overwrite to force).", str(exc).capitalize())
        return 1

    # }}}

    return 0


def main(
    settings: Settings,
    *,
    outfile: pathlib.Path | None = None,
    npoints: int | None = None,
    with_regions: bool = False,
    with_imshow: bool = False,
    with_complex: bool = False,
    with_polar: bool = False,
    with_phase: bool = False,
    with_trajectories: bool = False,
    max_workers: int | None = None,
    overwrite: bool = False,
) -> int:
    import gc

    if outfile is None:
        outfile = settings.filename.with_suffix("")

    import tempfile

    rng = np.random.default_rng(seed=42)
    model = ThetaModel(
        kappa=sym.Variable("kappa"),
        eta=sym.Variable("eta"),
        h=sym.DiracDelayKernel(sym.Variable("tau")),
    )

    # NOTE: we try to use (eta, kappa, tau) as symbolic parameters, so we only
    # compile the module once. However, the tau=0.0 case does not work for jitcdde
    # in this setup because it tries to interpolate the past on like [0, 0] and
    # fails. To work around that, we create a hardcoded tau=0.0 module and it all
    # works nicely.
    # FIXME: Not clear that this actually saves that much, since compilation should
    # be rather quick regardless. Would be nice to benchmark a bit..
    code_dde = generate_code(
        model,
        module_location=pathlib.Path(tempfile.gettempdir()) / "jitcdde_theta_tau.so",
    )
    code_ode = generate_code(
        replace(model, h=sym.DiracDelayKernel(0.0)),
        module_location=pathlib.Path(tempfile.gettempdir()) / "jitcode_theta_tau0.so",
    )

    ret = 0
    for name, param in settings.params.items():
        for tau in param.taus:
            log.info("=" * 75)
            log.info("[%s] tau = %g", name, tau)
            log.info("=" * 75)
            runfile = outfile.with_stem(slugify(f"{outfile.stem}-{name}-{tau:.3f}"))

            model = ThetaModel(
                kappa=param.kappa,
                eta=param.eta,
                h=sym.DiracDelayKernel(tau),
            )

            ret += run_simulation(
                runfile,
                code_dde if tau > 0 else code_ode,
                model,
                param,
                rng=rng,
                with_regions=with_regions,
                with_imshow=with_imshow,
                with_complex=with_complex,
                with_polar=with_polar,
                with_phase=with_phase,
                with_trajectories=with_trajectories,
                max_workers=max_workers,
                overwrite=overwrite,
            )

            # NOTE: memory seems to be exploding (mainly when plotting, yey matplotlib)
            gc.collect()

    if code_dde.module_location is not None and code_dde.module_location.exists():
        code_dde.module_location.unlink()

    if code_ode.module_location is not None and code_ode.module_location.exists():
        code_ode.module_location.unlink()

    return ret


# }}}


# {{{ visualize


def visualize(
    outfile: pathlib.Path,
    *,
    with_regions: bool = False,
    with_imshow: bool = False,
    with_complex: bool = False,
    with_polar: bool = False,
    with_phase: bool = False,
    with_trajectories: bool = False,
    overwrite: bool = False,
) -> int:
    if not outfile.exists():
        log.error("File does not exist: %s.", outfile)
        return 1

    data = np.load(outfile, allow_pickle=True)

    sgs = SolutionGroups(
        ts=data["ts"],
        zs=data["zs"],
        groups=data["groups"][()],
    )

    param = data["param"][()]
    if "tau" not in data:
        tau = float(".".join(outfile.stem.split("_")[-2:]))
    else:
        tau = data["tau"]

    model = ThetaModel(
        kappa=param.kappa,
        eta=param.eta,
        h=sym.DiracDelayKernel(tau),
    )
    zfp = find_fixed_points(model)

    rng = np.random.default_rng(seed=42)
    outfile = outfile.with_suffix("")

    try:
        if with_regions:
            filename = outfile.with_stem(f"{outfile.stem}_regions")
            plot_regions(filename, sgs, zfp, overwrite=overwrite)

        if with_imshow:
            filename = outfile.with_stem(f"{outfile.stem}_regions_imshow")
            plot_regions_imshow(filename, sgs, zfp, overwrite=overwrite)

        if with_complex:
            filename = outfile.with_stem(f"{outfile.stem}_real")
            plot_trajectory_cart(filename, sgs, real=True, rng=rng, overwrite=overwrite)

            filename = outfile.with_stem(f"{outfile.stem}_imag")
            plot_trajectory_cart(
                filename, sgs, real=False, rng=rng, overwrite=overwrite
            )

        if with_polar:
            filename = outfile.with_stem(f"{outfile.stem}_radius")
            plot_trajectory_polar(
                filename, sgs, radius=True, rng=rng, overwrite=overwrite
            )

            filename = outfile.with_stem(f"{outfile.stem}_angle")
            plot_trajectory_polar(
                filename, sgs, radius=False, rng=rng, overwrite=overwrite
            )

        if with_phase:
            filename = outfile.with_stem(f"{outfile.stem}_phase")
            plot_phase(filename, sgs, zfp, rng=rng, overwrite=overwrite)

        if with_trajectories:
            filename = outfile.with_stem(f"{outfile.stem}_trajectories")
            plot_trajectory_slices(filename, sgs, zfp, rng=rng, overwrite=overwrite)
    except FileExistsError as exc:
        log.error("%s (use --overwrite to force).", str(exc).capitalize())

    return 0


# }}}


if __name__ == "__main__":
    import argparse

    # NOTE: don't want to oversaturate the system by default
    max_workers = multiprocessing.cpu_count() // 2

    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*", type=pathlib.Path)
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Basename for output files",
    )
    parser.add_argument(
        "-s",
        "--select",
        action="append",
        help="Select runs from the configuration file",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=None,
        help="Number of points per sector (overwrites parameter file)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Delay used in the equation (overwrites parameter file)",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--regions", action="store_true")
    parser.add_argument("--imshow", action="store_true")
    parser.add_argument("--complex", action="store_true")
    parser.add_argument("--polar", action="store_true")
    parser.add_argument("--phase", action="store_true")
    parser.add_argument("--trajectories", action="store_true")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max_workers,
        help="Maximum number of parallel runs (<= 1 will run sequentially)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    if args.visualize:
        ret = 0
        for filename in args.filenames:
            ret += visualize(
                filename,
                with_regions=args.regions,
                with_imshow=args.imshow,
                with_complex=args.complex,
                with_polar=args.polar,
                with_phase=args.phase,
                with_trajectories=args.trajectories,
                overwrite=args.overwrite,
            )
        raise SystemExit(ret)
    else:
        try:
            settings = parse_settings(
                args.filenames[0],
                select=args.select,
                tau=args.tau,
                npoints=args.npoints,
            )
        except ValueError as exc:
            log.error("Parsing settings failed: %s.", exc, exc_info=exc)
            raise SystemExit(1) from None

        raise SystemExit(
            main(
                settings,
                outfile=args.outfile,
                npoints=args.npoints,
                with_regions=args.regions,
                with_imshow=args.imshow,
                with_complex=args.complex,
                with_polar=args.polar,
                with_phase=args.phase,
                with_trajectories=args.trajectories,
                max_workers=args.jobs,
                overwrite=args.overwrite,
            )
        )
