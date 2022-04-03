"""
Gammafield evaluation visualization functions for different scenarios.
"""
# Author: Lukas Huber
# Date: 2021-05-18
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021
import warnings

import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.visualization import plot_obstacles


def gamma_field_visualization(
    obstacle,
    n_resolution=30,
    x_lim=None,
    y_lim=None,
    dim=2,
    fig=None,
    ax=None,
    grid_number=None,
):
    """Draw the gamma of one obstacle."""
    if grid_number is not None:
        warnings.warn("'grid_number' is depreciated. Use 'n_resolution' instead.")
        n_resolution = grid_number

    if fig is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)

    else:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

    x_vals = np.linspace(x_lim[0], x_lim[1], n_resolution)
    y_vals = np.linspace(y_lim[0], y_lim[1], n_resolution)

    gamma_values = np.zeros((n_resolution, n_resolution))
    positions = np.zeros((dim, n_resolution, n_resolution))

    for ix in range(n_resolution):
        for iy in range(n_resolution):
            positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]

            gamma_values[ix, iy] = obstacle.get_gamma(
                positions[:, ix, iy], in_global_frame=True
            )

    cs = ax.contourf(
        positions[0, :, :],
        positions[1, :, :],
        gamma_values,
        np.arange(1.0, 2.0, 0.1),
        extend="max",
        alpha=0.6,
        zorder=-3,
    )
    cbar = fig.colorbar(cs)


def gamma_field_multihull(
    obstacle_list,
    it_obs,
    n_resolution=30,
    x_lim=None,
    y_lim=None,
    dim=2,
    ax=None,
):
    """Draw a list of obstacles and evaluate the gamma-field of the obstacle at 'it_obs'."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    elif x_lim is None:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()

    else:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    plot_obstacles(
        ax=ax,
        obs=obstacle_list,
        x_range=x_lim,
        y_range=y_lim,
        noTicks=True,
        showLabel=False,
    )

    x_vals = np.linspace(x_lim[0], x_lim[1], n_resolution)
    y_vals = np.linspace(y_lim[0], y_lim[1], n_resolution)

    gamma_values = np.zeros((n_resolution, n_resolution))
    positions = np.zeros((dim, n_resolution, n_resolution))

    for ix in range(n_resolution):
        for iy in range(n_resolution):
            positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]

            obstacle_list.update_relative_reference_point(position=positions[:, ix, iy])

            gamma_values[ix, iy] = obstacle_list[it_obs].get_gamma(
                positions[:, ix, iy], in_global_frame=True
            )

    cs = ax.contourf(
        positions[0, :, :],
        positions[1, :, :],
        gamma_values,
        np.arange(1.0, 5.1, 0.2),
        extend="max",
        alpha=0.6,
        zorder=2,
    )
