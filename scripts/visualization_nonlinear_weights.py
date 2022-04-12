# Visual of 'weighting' function to help with debugging
import matplotlib.pyplot as plt

plt.close("all")
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from shapely.geometry import *

from math import pi

import numpy as np
from numpy import linalg as LA

from vartools.directional_space import UnitDirection, DirectionBase
from vartools.directional_space.visualization import circular_space_setup


def ring_coding(ob):
    # The codes will be all "LINETO" commands, except for "MOVETO"s at the
    # beginning of each subpath
    n = len(ob.coords)
    codes = np.ones(n, dtype=Path.code_type) * Path.LINETO
    codes[0] = Path.MOVETO
    return codes


def pathify(polygon):
    # Convert coordinates to path vertices. Objects produced by Shapely's
    # analytic methods have the proper coordinate order, no need to sort.
    vertices = np.concatenate(
        [np.asarray(polygon.exterior)] + [np.asarray(r) for r in polygon.interiors]
    )
    codes = np.concatenate(
        [ring_coding(polygon.exterior)] + [ring_coding(r) for r in polygon.interiors]
    )
    return Path(vertices, codes)


def visualize_convergence_weight_conv_dir_outside(save_figure=False):
    from dynamic_obstacle_avoidance.avoidance.rotation import (
        _get_nonlinear_inverted_weight,
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    n_grid = 80
    dim = 3

    inv_conv_radius = pi / 2.0
    base0 = DirectionBase(matrix=np.eye(dim))

    dir_inv_nonlinear = UnitDirection(base0)

    nx = ny = n_grid
    x_vals, y_vals = np.meshgrid(np.linspace(-pi, pi, nx), np.linspace(-pi, pi, ny))

    angles = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    values = np.zeros(angles.shape[1])

    # dir_inv_conv_rot = UnitDirection(base0).from_angle(np.array([pi*0.3, pi*0.2]))
    dir_inv_conv_rot = UnitDirection(base0).from_angle(np.array([pi * 0.6, pi * 0.4]))
    dir_inv_conv_rot.as_angle()

    for ii in range(angles.shape[1]):
        dir_inv_nonlinear.from_angle(angles[:, ii])

        if dir_inv_nonlinear.norm() > pi:
            pass
        values[ii] = _get_nonlinear_inverted_weight(
            dir_inv_conv_rot.norm(),
            dir_inv_nonlinear.norm(),
            inv_convergence_radius=inv_conv_radius,
            weight=1,
        )

    cs = ax.contourf(
        angles[0, :].reshape(n_grid, n_grid),
        angles[1, :].reshape(n_grid, n_grid),
        values.reshape(n_grid, n_grid),
        np.linspace(0, 1, 11),
        cmap=cm.YlGnBu,
        linewidth=0.2,
        edgecolors="k",
    )
    cbar = fig.colorbar(cs)

    mpl.rc("font", family="Times New Roman")
    # ax.set_xlabel(r'Angle x [deg]')
    # ax.set_ylabel(r'Angle y [deg]')
    ax.set_title(r"Inverted Convergence Direction")
    dir_angle = dir_inv_conv_rot.as_angle()
    ax.plot(dir_angle[0], dir_angle[1], "ko")

    circular_space_setup(ax)

    if save_figure:
        figure_name = "nonlinear_weight_conv_dir_outside"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

    plt.ion()
    plt.show()


def visualize_convergence_weight_conv_dir_inside(save_figure=False):
    from dynamic_obstacle_avoidance.avoidance.rotation import (
        _get_nonlinear_inverted_weight,
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    n_grid = 80
    dim = 3

    inv_conv_radius = pi / 2.0
    base0 = DirectionBase(matrix=np.eye(dim))

    dir_inv_nonlinear = UnitDirection(base0)

    nx = ny = n_grid
    x_vals, y_vals = np.meshgrid(np.linspace(-pi, pi, nx), np.linspace(-pi, pi, ny))

    angles = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    values = np.zeros(angles.shape[1])

    dir_inv_conv_rot = UnitDirection(base0).from_angle(np.array([pi * 0.3, pi * 0.2]))

    dir_inv_conv_rot.as_angle()

    if not dir_inv_conv_rot.norm() < inv_conv_radius:
        breakpoint()

    for ii in range(angles.shape[1]):
        dir_inv_nonlinear.from_angle(angles[:, ii])

        if dir_inv_nonlinear.norm() > pi:
            pass
        values[ii] = _get_nonlinear_inverted_weight(
            dir_inv_conv_rot.norm(),
            dir_inv_nonlinear.norm(),
            inv_convergence_radius=inv_conv_radius,
            weight=1,
        )

    cs = ax.contourf(
        angles[0, :].reshape(n_grid, n_grid),
        angles[1, :].reshape(n_grid, n_grid),
        values.reshape(n_grid, n_grid),
        np.linspace(0, 1, 11),
        cmap=cm.YlGnBu,
        linewidth=0.2,
        edgecolors="k",
    )
    cbar = fig.colorbar(cs)

    mpl.rc("font", family="Times New Roman")
    # ax.set_xlabel(r'Angle x [deg]')
    # ax.set_ylabel(r'Angle y [deg]')
    ax.set_title(r"Inverted Convergence Direction")
    dir_angle = dir_inv_conv_rot.as_angle()
    ax.plot(dir_angle[0], dir_angle[1], "ko")
    ax.text(dir_angle[0] + 0.1, dir_angle[1] - 0.3, r"$f_{conv}$")

    ax.spines.left.set_position("center")
    ax.spines.right.set_color("none")
    ax.spines.bottom.set_position("center")
    ax.spines.top.set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    plt.axis("equal")
    print("Done")

    circ_var = np.linspace(0, 2 * pi, 100)
    rad = pi
    plt.plot(
        np.cos(circ_var) * inv_conv_radius, np.sin(circ_var) * inv_conv_radius, "k--"
    )

    # polygon = Point(0, 0).buffer(10.0).difference(MultiPoint([(-5, 0), (5, 0)]).buffer(3.0))
    polygon = Point(0, 0).buffer(10.0).difference(Point(0, 0).buffer(pi))
    path = pathify(polygon)

    patch = PathPatch(path, facecolor="white", linewidth=0)

    dx = dy = 0.1
    ax.set_xlim([x_vals[0, 0] - dx, x_vals[-1, -1] + dx])
    ax.set_ylim([y_vals[0, 0] - dy, y_vals[-1, -1] + dy])

    ax.add_patch(patch)

    if save_figure:
        figure_name = "nonlinear_weight_conv_dir_inside"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

    plt.ion()
    plt.show()


def visualize_convergence_weight_inv_nonlinear_inside(save_figure=False):
    from dynamic_obstacle_avoidance.avoidance.rotation import (
        _get_nonlinear_inverted_weight,
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    n_grid = 80
    dim = 3

    inv_conv_radius = pi / 2.0
    base0 = DirectionBase(matrix=np.eye(dim))

    dir_inv_conv_rot = UnitDirection(base0)

    nx = ny = n_grid
    x_vals, y_vals = np.meshgrid(np.linspace(-pi, pi, nx), np.linspace(-pi, pi, ny))

    angles = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    values = np.zeros(angles.shape[1])

    dir_inv_nonlinear = UnitDirection(base0).from_angle(np.array([pi * 0.3, pi * 0.2]))

    dir_inv_nonlinear.as_angle()

    if not dir_inv_nonlinear.norm() < inv_conv_radius:
        breakpoint()

    for ii in range(angles.shape[1]):
        dir_inv_conv_rot.from_angle(angles[:, ii])

        if dir_inv_conv_rot.norm() > pi:
            pass
        values[ii] = _get_nonlinear_inverted_weight(
            dir_inv_conv_rot.norm(),
            dir_inv_nonlinear.norm(),
            inv_convergence_radius=inv_conv_radius,
            weight=1,
        )

    cs = ax.contourf(
        angles[0, :].reshape(n_grid, n_grid),
        angles[1, :].reshape(n_grid, n_grid),
        values.reshape(n_grid, n_grid),
        np.linspace(0, 1, 11),
        cmap=cm.YlGnBu,
        linewidth=0.2,
        edgecolors="k",
    )
    cbar = fig.colorbar(cs)

    mpl.rc("font", family="Times New Roman")
    # ax.set_xlabel(r'Angle x [deg]')
    # ax.set_ylabel(r'Angle y [deg]')
    ax.set_title(r"Inverted Convergence Direction")
    # ax.set_title(f"Nonlinear direction outside = {dir_inv_nonlinear.as_angle()}")
    dir_angle = dir_inv_nonlinear.as_angle()
    print(f"dir_angle={dir_angle}")
    ax.plot(dir_angle[0], dir_angle[1], "ko")
    # ax.plot(dir_angle[0], dir_angle[1], r"$f_{nonl}$")
    # ax.plot(dir_angle[0], dir_angle[1], r"f_(nonl)")
    ax.text(dir_angle[0] + 0.1, dir_angle[1] - 0.3, r"$f_{nonl}$")

    ax.spines.left.set_position("center")
    ax.spines.right.set_color("none")
    ax.spines.bottom.set_position("center")
    ax.spines.top.set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    plt.axis("equal")
    print("Done")

    circ_var = np.linspace(0, 2 * pi, 100)
    rad = pi
    plt.plot(
        np.cos(circ_var) * inv_conv_radius, np.sin(circ_var) * inv_conv_radius, "k--"
    )

    # polygon = Point(0, 0).buffer(10.0).difference(MultiPoint([(-5, 0), (5, 0)]).buffer(3.0))
    polygon = Point(0, 0).buffer(10.0).difference(Point(0, 0).buffer(pi))
    path = pathify(polygon)

    patch = PathPatch(path, facecolor="white", linewidth=0)

    dx = dy = 0.1
    ax.set_xlim([x_vals[0, 0] - dx, x_vals[-1, -1] + dx])
    ax.set_ylim([y_vals[0, 0] - dy, y_vals[-1, -1] + dy])

    ax.add_patch(patch)

    if save_figure:
        figure_name = "nonlinear_weight_inside"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

    plt.ion()
    plt.show()


def visualize_convergence_weight_inv_nonlinear_outside(save_figure=False):
    from dynamic_obstacle_avoidance.avoidance.rotation import (
        _get_nonlinear_inverted_weight,
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    n_grid = 80
    dim = 3

    inv_conv_radius = pi / 2.0
    base0 = DirectionBase(matrix=np.eye(dim))

    dir_inv_conv_rot = UnitDirection(base0)

    nx = ny = n_grid
    x_vals, y_vals = np.meshgrid(np.linspace(-pi, pi, nx), np.linspace(-pi, pi, ny))

    angles = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    values = np.zeros(angles.shape[1])

    dir_inv_nonlinear_outside = UnitDirection(base0).from_angle(
        np.array([pi * 0.6, pi * 0.4])
    )

    dir_inv_nonlinear_outside.as_angle()

    # Dummy check for silly developers
    if not dir_inv_nonlinear_outside.norm() > inv_conv_radius:
        breakpoint()

    # for ii, angle in enumerate(angles):
    for ii in range(angles.shape[1]):
        dir_inv_conv_rot.from_angle(angles[:, ii])

        if dir_inv_conv_rot.norm() > pi:
            pass
        values[ii] = _get_nonlinear_inverted_weight(
            dir_inv_conv_rot.norm(),
            dir_inv_nonlinear_outside.norm(),
            inv_convergence_radius=inv_conv_radius,
            weight=1,
        )
        # values[ii] = dir_inv_nonlinear_outside.norm()
        # values[ii] = dir_inv_nonlinear_outside.norm()
        # values[ii] = dir_inv_nonlinear_outside.get_distance_to(dir_inv_conv_rot)

    cs = ax.contourf(
        angles[0, :].reshape(n_grid, n_grid),
        angles[1, :].reshape(n_grid, n_grid),
        values.reshape(n_grid, n_grid),
        # np.arange(0.0, pi, np.pi/10.0),
        np.linspace(0, 1, 11),
        # np.linspace(0, 10, 11),
        # cmap='hot_r',
        cmap=cm.YlGnBu,
        linewidth=0.2,
        edgecolors="k",
    )
    cbar = fig.colorbar(
        cs,
        # cax=cbar_ax, ticks=np.linspace(0, np.pi, 5)
    )

    mpl.rc("font", family="Times New Roman")
    # ax.set_xlabel(r'Angle x [deg]')
    # ax.set_ylabel(r'Angle y [deg]')
    ax.set_title(r"Inverted Convergence Direction")
    # ax.set_title(f"Nonlinear direction outside = {dir_inv_nonlinear_outside.as_angle()}")
    dir_angle = dir_inv_nonlinear_outside.as_angle()
    print(f"dir_angle={dir_angle}")
    ax.plot(dir_angle[0], dir_angle[1], "ko")
    # ax.plot(dir_angle[0], dir_angle[1], r"$f_{nonl}$")
    # ax.plot(dir_angle[0], dir_angle[1], r"f_(nonl)")
    ax.text(dir_angle[0] + 0.1, dir_angle[1] - 0.3, r"$f_{nonl}$")

    ax.spines.left.set_position("center")
    ax.spines.right.set_color("none")
    ax.spines.bottom.set_position("center")
    ax.spines.top.set_color("none")
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    plt.axis("equal")
    print("Done")

    circ_var = np.linspace(0, 2 * pi, 100)
    rad = pi
    plt.plot(
        np.cos(circ_var) * inv_conv_radius, np.sin(circ_var) * inv_conv_radius, "k--"
    )

    # polygon = Point(0, 0).buffer(10.0).difference(MultiPoint([(-5, 0), (5, 0)]).buffer(3.0))
    polygon = Point(0, 0).buffer(10.0).difference(Point(0, 0).buffer(pi))
    path = pathify(polygon)

    patch = PathPatch(path, facecolor="white", linewidth=0)

    dx = dy = 0.1
    ax.set_xlim([x_vals[0, 0] - dx, x_vals[-1, -1] + dx])
    ax.set_ylim([y_vals[0, 0] - dy, y_vals[-1, -1] + dy])

    ax.add_patch(patch)

    if save_figure:
        figure_name = "nonlinear_weight_outside"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

    plt.ion()
    plt.show()


if __name__ == "__main__":
    # visualize_convergence_weight_inv_nonlinear_outside(save_figure=True)
    # visualize_convergence_weight_inv_nonlinear_inside(save_figure=True)
    visualize_convergence_weight_conv_dir_inside(save_figure=True)
    # visualize_convergence_weight_conv_dir_outside(save_figure=True)
