#!/USSR/bin/python3
"""
Reference Point Search
"""
# Author: LukasHuber
# Created: 2020-02-28
# Email: Lukas.huber@epfl.ch
import numpy as np
from numpy import pi

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer


def main(save_figure=False, n_grid=50, x_lim=[-2, 4], y_lim=[-3, 3]):

    environment = ObstacleContainer()

    environment.append(
        Ellipse(
            center_position=np.array([0, 0]),
            orientation=40 * pi / 180,
            axes_length=[1.2, 1.6],
        )
    )

    environment.append(
        Ellipse(
            center_position=np.array([2, 0]),
            orientation=-40 * pi / 180,
            axes_length=[1.2, 1.6],
        )
    )

    fig, axs = plt.subplots(figsize=(11, 4), nrows=1, ncols=3)

    nx = ny = n_grid
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    n_levels = 20
    level_boundaries = np.linspace(0.00001, 3.0, n_levels + 1)
    alpha_value = 0.9
    # First obstacle
    gammas0 = np.zeros(positions.shape[1])
    for jj in range(positions.shape[1]):
        gammas0[jj] = environment[0].get_gamma(positions[:, jj], in_global_frame=True)

    surf = axs[0].contourf(
        x_vals[0, :],
        y_vals[:, 0],
        gammas0.reshape(n_grid, n_grid),
        level_boundaries,
        extend="max",
        alpha=alpha_value,
    )

    # Second obstacle
    gammas1 = np.zeros(positions.shape[1])
    for jj in range(positions.shape[1]):
        gammas1[jj] = environment[1].get_gamma(positions[:, jj], in_global_frame=True)

    surf = axs[1].contourf(
        x_vals[0, :],
        y_vals[:, 0],
        gammas1.reshape(n_grid, n_grid),
        level_boundaries,
        extend="max",
        alpha=alpha_value,
    )
    # Mixed obstacle
    gamma_margin = 1.2
    # gamma_margin = 1
    # gammas_mixed = 1/(1-gammas0) + 1/(1-gammas1)
    ind_inside = np.logical_and(gammas0 < gamma_margin, gammas1 < gamma_margin)
    gammas_mixed = np.ones(gammas1.shape) * (1e6)

    gammas_mixed[ind_inside] = gamma_margin / (
        gamma_margin - gammas0[ind_inside]
    ) + gamma_margin / (gamma_margin - gammas1[ind_inside])

    # gammas_mixed[ind_inside] = gammas_mixed[ind_inside]/15
    gammas_mixed[ind_inside] = gammas_mixed[ind_inside] / 5
    # gammas_mixed[ind_inside] = gammas_mixed[ind_inside] - 10
    # gammas_mixed = 1/(1-gammas0) + 1/(1-gammas1)

    surf = axs[2].contourf(
        x_vals[0, :],
        y_vals[:, 0],
        gammas_mixed.reshape(n_grid, n_grid),
        level_boundaries,
        # np.linspace(0, 100, 100),
        extend="max",
        alpha=alpha_value,
    )

    # cmap=cm.coolwarm, # linewidth=0,
    # antialiased=False, alpha=0.9)

    for ax in axs:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        ax.set_aspect("equal", adjustable="box")

        for obs in environment:
            obs.draw_obstacle()

            ax.plot(
                obs.boundary_points_global[0, :], obs.boundary_points_global[1, :], "k:"
            )

            # ax.tick_params(axis='both', which='major',
            # labelbottom=False, labelleft=False,
            # bottom=False, top=False, left=False, right=False)

    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.905, 0.2, 0.020, 0.6])
    fig.colorbar(surf, cax=cbar_ax)

    # ax.xlabel("")
    # ax.ylabel("")

    if save_figure:
        figure_name = "gamma_descent_values"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    main(save_figure=True)
