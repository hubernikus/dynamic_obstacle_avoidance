""" Visual of 'weighting' function to help with debugging."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib as mpl

from math import pi

import numpy as np
from numpy import linalg as LA

from vartools.directional_space import UnitDirection, DirectionBase
from vartools.directional_space.visualization import circular_space_setup

plt.close("all")
plt.ion()


def visualize_combined_nonlinear_weight_nonl_outside(save_figure=False):
    # from dynamic_obstacle_avoidance.avoidance.rotation import _get_nonlinear_inverted_weight
    from dynamic_obstacle_avoidance.avoidance.rotation import (
        _get_projected_nonlinear_velocity,
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    n_grid = 100
    dim = 3

    inv_conv_radius = pi / 2.0
    base0 = DirectionBase(matrix=np.eye(dim))

    inv_conv_rotated = UnitDirection(base0)

    x_vals, y_vals = np.meshgrid(
        np.linspace(-pi, pi, n_grid), np.linspace(-pi, pi, n_grid)
    )

    angles = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    values = np.zeros(angles.shape[1])

    inv_nonlinear = UnitDirection(base0).from_angle(np.array([pi * 0.6, pi * 0.4]))
    inv_nonlinear.as_angle()

    for ii in range(angles.shape[1]):
        inv_conv_rotated.from_angle(angles[:, ii])

        if inv_conv_rotated.norm() > pi:
            # Don't do the edge-cases
            continue

        # In-outputs in normal space
        dir_nonl_rotated = _get_projected_nonlinear_velocity(
            dir_conv_rotated=inv_conv_rotated.invert_normal(),
            dir_nonlinear=inv_nonlinear.invert_normal(),
            weight=1,
            convergence_radius=pi / 2,
        )
        inv_nonl_rotated = dir_nonl_rotated.invert_normal()

        tot_norm = (inv_nonlinear - inv_conv_rotated).norm()
        if tot_norm:  # nonzero
            values[ii] = (inv_nonlinear - inv_nonl_rotated).norm() / tot_norm

    cs = ax.contourf(
        angles[0, :].reshape(n_grid, n_grid),
        angles[1, :].reshape(n_grid, n_grid),
        values.reshape(n_grid, n_grid),
        # np.arange(0.0, pi, np.pi/10.0),
        # np.linspace(0, 1, 11),
        np.linspace(0, 1.0, 11),
        cmap=cm.YlGnBu,
        linewidth=0.2,
        edgecolors="k",
    )
    cbar = fig.colorbar(
        cs,
        # cax=cbar_ax, ticks=np.linspace(0, np.pi, 5)
    )

    mpl.rc("font", family="Times New Roman")
    ax.set_title(r"Inverted Convergence Direction")
    dir_angle = inv_nonlinear.as_angle()
    ax.plot(dir_angle[0], dir_angle[1], "ko")
    ax.text(dir_angle[0] + 0.1, dir_angle[1] - 0.3, r"$f_{nonl}$")

    # Slight cropping of the space radius to 'hide' the visual-matplotlib interpolation
    circular_space_setup(ax, space_radius=pi - 0.1)

    if save_figure:
        figure_name = "nonlinear_combined_weight_displacement_outside"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def visualize_combined_nonlinear_weight_nonl_inside(save_figure=False):
    # from dynamic_obstacle_avoidance.avoidance.rotation import _get_nonlinear_inverted_weight
    from dynamic_obstacle_avoidance.avoidance.rotation import (
        _get_projected_nonlinear_velocity,
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    n_grid = 20
    dim = 3

    inv_conv_radius = pi / 2.0
    base0 = DirectionBase(matrix=np.eye(dim))

    inv_conv_rotated = UnitDirection(base0)

    x_vals, y_vals = np.meshgrid(
        np.linspace(-pi, pi, n_grid), np.linspace(-pi, pi, n_grid)
    )

    angles = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    values = np.zeros(angles.shape[1])

    inv_nonlinear = UnitDirection(base0).from_angle(np.array([pi * 0.3, pi * 0.2]))
    inv_nonlinear.as_angle()

    for ii in range(angles.shape[1]):
        inv_conv_rotated.from_angle(angles[:, ii])

        if inv_conv_rotated.norm() > pi:
            # Don't do the edge-cases
            continue

        # In-outputs in normal space
        dir_nonl_rotated = _get_projected_nonlinear_velocity(
            dir_conv_rotated=inv_conv_rotated.invert_normal(),
            dir_nonlinear=inv_nonlinear.invert_normal(),
            weight=1,
            convergence_radius=pi / 2,
        )
        inv_nonl_rotated = dir_nonl_rotated.invert_normal()

        tot_norm = (inv_nonlinear - inv_conv_rotated).norm()
        if tot_norm:  # nonzero
            values[ii] = (inv_nonlinear - inv_nonl_rotated).norm() / tot_norm

    cs = ax.contourf(
        angles[0, :].reshape(n_grid, n_grid),
        angles[1, :].reshape(n_grid, n_grid),
        values.reshape(n_grid, n_grid),
        # np.arange(0.0, pi, np.pi/10.0),
        # np.linspace(0, 1, 11),
        np.linspace(0, 1.0, 11),
        cmap=cm.YlGnBu,
        linewidth=0.2,
        edgecolors="k",
    )
    cbar = fig.colorbar(
        cs,
        # cax=cbar_ax, ticks=np.linspace(0, np.pi, 5)
    )

    mpl.rc("font", family="Times New Roman")
    ax.set_title(r"Inverted Convergence Direction")
    dir_angle = inv_nonlinear.as_angle()
    print(f"dir_angle={dir_angle}")
    ax.plot(dir_angle[0], dir_angle[1], "ko")
    ax.text(dir_angle[0] + 0.1, dir_angle[1] - 0.3, r"$f_{nonl}$")

    # Slight cropping of the space radius to 'hide' the visual-matplotlib interpolation
    circular_space_setup(ax, space_radius=pi - 0.1)

    if save_figure:
        figure_name = "nonlinear_combined_weight_displacement_inside"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def visualize_combined_nonlinear_weight_rot_inside(save_figure=False):
    # from dynamic_obstacle_avoidance.avoidance.rotation import _get_nonlinear_inverted_weight
    from dynamic_obstacle_avoidance.avoidance.rotation import (
        _get_projected_nonlinear_velocity,
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    n_grid = 20
    dim = 3

    inv_conv_radius = pi / 2.0
    base0 = DirectionBase(matrix=np.eye(dim))

    inv_nonlinear = UnitDirection(base0)

    x_vals, y_vals = np.meshgrid(
        np.linspace(-pi, pi, n_grid), np.linspace(-pi, pi, n_grid)
    )

    angles = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    values = np.zeros(angles.shape[1])

    inv_conv_rotated = UnitDirection(base0).from_angle(np.array([pi * 0.3, pi * 0.2]))
    # inv_conv_rotated.as_angle()

    for ii in range(angles.shape[1]):
        inv_nonlinear.from_angle(angles[:, ii])

        if inv_nonlinear.norm() > pi:
            # Don't do the edge-cases
            continue

        # In-outputs in normal space
        dir_nonl_rotated = _get_projected_nonlinear_velocity(
            dir_conv_rotated=inv_conv_rotated.invert_normal(),
            dir_nonlinear=inv_nonlinear.invert_normal(),
            weight=1,
            convergence_radius=pi / 2,
        )
        inv_nonl_rotated = dir_nonl_rotated.invert_normal()

        tot_norm = (inv_nonlinear - inv_conv_rotated).norm()
        if tot_norm:  # nonzero
            values[ii] = (inv_nonlinear - inv_nonl_rotated).norm() / tot_norm

    cs = ax.contourf(
        angles[0, :].reshape(n_grid, n_grid),
        angles[1, :].reshape(n_grid, n_grid),
        values.reshape(n_grid, n_grid),
        # np.arange(0.0, pi, np.pi/10.0),
        # np.linspace(0, 1, 11),
        np.linspace(0, 1.0, 11),
        cmap=cm.YlGnBu,
        linewidth=0.2,
        edgecolors="k",
    )
    cbar = fig.colorbar(
        cs,
        # cax=cbar_ax, ticks=np.linspace(0, np.pi, 5)
    )

    mpl.rc("font", family="Times New Roman")
    ax.set_title(r"Inverted Convergence Direction")
    dir_angle = inv_conv_rotated.as_angle()
    ax.plot(dir_angle[0], dir_angle[1], "ko")
    ax.text(dir_angle[0] + 0.1, dir_angle[1] - 0.3, r"$f_{rot}$")

    # Slight cropping of the space radius to 'hide' the visual-matplotlib interpolation
    circular_space_setup(ax, space_radius=pi - 0.1)

    if save_figure:
        figure_name = "nonlinear_combined_weight_displacement_inside"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def visualize_combined_nonlinear_weight_rot_outside(save_figure=False):
    # from dynamic_obstacle_avoidance.avoidance.rotation import _get_nonlinear_inverted_weight
    from dynamic_obstacle_avoidance.avoidance.rotation import (
        _get_projected_nonlinear_velocity,
    )

    fig, ax = plt.subplots(figsize=(7.5, 6))
    n_grid = 20
    dim = 3

    inv_conv_radius = pi / 2.0
    base0 = DirectionBase(matrix=np.eye(dim))

    inv_nonlinear = UnitDirection(base0)

    x_vals, y_vals = np.meshgrid(
        np.linspace(-pi, pi, n_grid), np.linspace(-pi, pi, n_grid)
    )

    angles = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    values = np.zeros(angles.shape[1])

    inv_conv_rotated = UnitDirection(base0).from_angle(np.array([pi * 0.6, pi * 0.4]))
    # inv_conv_rotated.as_angle()

    for ii in range(angles.shape[1]):
        inv_nonlinear.from_angle(angles[:, ii])

        if inv_nonlinear.norm() > pi:
            # Don't do the edge-cases
            continue

        # In-outputs in normal space
        dir_nonl_rotated = _get_projected_nonlinear_velocity(
            dir_conv_rotated=inv_conv_rotated.invert_normal(),
            dir_nonlinear=inv_nonlinear.invert_normal(),
            weight=1,
            convergence_radius=pi / 2,
        )
        inv_nonl_rotated = dir_nonl_rotated.invert_normal()

        tot_norm = (inv_nonlinear - inv_conv_rotated).norm()
        if tot_norm:  # nonzero
            values[ii] = (inv_nonlinear - inv_nonl_rotated).norm() / tot_norm

    cs = ax.contourf(
        angles[0, :].reshape(n_grid, n_grid),
        angles[1, :].reshape(n_grid, n_grid),
        values.reshape(n_grid, n_grid),
        # np.arange(0.0, pi, np.pi/10.0),
        # np.linspace(0, 1, 11),
        np.linspace(0, 1.0, 11),
        cmap=cm.YlGnBu,
        linewidth=0.2,
        edgecolors="k",
    )
    cbar = fig.colorbar(
        cs,
        # cax=cbar_ax, ticks=np.linspace(0, np.pi, 5)
    )

    mpl.rc("font", family="Times New Roman")
    ax.set_title(r"Inverted Convergence Direction")
    dir_angle = inv_conv_rotated.as_angle()
    ax.plot(dir_angle[0], dir_angle[1], "ko")
    ax.text(dir_angle[0] + 0.1, dir_angle[1] - 0.3, r"$f_{rot}$")

    # Slight cropping of the space radius to 'hide' the visual-matplotlib interpolation
    circular_space_setup(ax, space_radius=pi - 0.1)

    if save_figure:
        figure_name = "nonlinear_combined_weight_displacement_outside"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    visualize_combined_nonlinear_weight_nonl_inside(save_figure=True)
    visualize_combined_nonlinear_weight_nonl_outside(save_figure=True)
    visualize_combined_nonlinear_weight_rot_inside(save_figure=True)
    # visualize_combined_nonlinear_weight_rot_outside(save_figure=True)
