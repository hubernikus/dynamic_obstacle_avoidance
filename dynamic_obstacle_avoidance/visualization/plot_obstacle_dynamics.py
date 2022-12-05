from typing import Callable

import numpy as np
import numpy.typing as npt

Vector = npt.ArrayLike

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.containers import ObstacleContainer


def plot_obstacle_dynamics(
    obstacle_container: ObstacleContainer,
    dynamics: Callable[[Vector], Vector],
    x_lim: list[float],
    y_lim: list[float],
    n_grid: int = 20,
    ax=None,
    attractor_position=None,
    do_quiver=True,
    show_ticks=True,
):
    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], n_grid),
        np.linspace(y_lim[0], y_lim[1], n_grid),
    )
    positions = np.array([xx.flatten(), yy.flatten()])
    velocities = np.zeros_like(positions)

    if len(obstacle_container):
        for pp in range(positions.shape[1]):
            # print(f"{positions[:, pp]=} | {velocities[:, pp]=}")
            # print(f"gamma = {obstacle_container.get_minimum_gamma(positions[:, pp])}")
            if obstacle_container.get_minimum_gamma(positions[:, pp]) <= 1:
                continue
            velocities[:, pp] = dynamics(positions[:, pp])
    else:
        for pp in range(positions.shape[1]):
            velocities[:, pp] = dynamics(positions[:, pp])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if do_quiver:
        ax.quiver(
            positions[0, :],
            positions[1, :],
            velocities[0, :],
            velocities[1, :],
            color="blue",
            # color="red",
            scale=50,
            width=0.007,
            zorder=-1,
        )
    else:
        ax.streamplot(
            positions[0, :].reshape(n_grid, n_grid),
            positions[1, :].reshape(n_grid, n_grid),
            velocities[0, :].reshape(n_grid, n_grid),
            velocities[1, :].reshape(n_grid, n_grid),
            color="blue",
            # color="red",
            # scale=50,
            zorder=-1,
        )
    if attractor_position is not None:
        ax.scatter(
            attractor_position[0],
            attractor_position[1],
            marker="*",
            s=200,
            color="black",
            zorder=5,
        )
    ax.axis("equal")

    if not show_ticks:
        ax.tick_params(
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

    return (fig, ax)
