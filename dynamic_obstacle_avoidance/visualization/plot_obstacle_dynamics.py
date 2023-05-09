from typing import Callable, Optional

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
    vectorfield_color="blue",
    collision_check_functor: Optional[Callable[[Vector], float]] = None,
    quiver_scale: int = 50,
    quiver_axbPlpha: float = 1,
    kwargs_quiver: dict = {},
):
    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], n_grid),
        np.linspace(y_lim[0], y_lim[1], n_grid),
    )
    positions = np.array([xx.flatten(), yy.flatten()])
    velocities = np.zeros_like(positions)

    if collision_check_functor is not None:
        for pp in range(positions.shape[1]):
            if collision_check_functor(positions[:, pp]):
                continue

            velocities[:, pp] = dynamics(positions[:, pp])

    elif len(obstacle_container):
        for pp in range(positions.shape[1]):
            if not obstacle_container.is_collision_free(positions[:, pp]):
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
            color=vectorfield_color,
            # color="red",
            # scale=quiver_scale,
            # alpha=quiver_alpha,
            # width=0.007,
            zorder=-1,
            **kwargs_quiver,
        )
    else:
        ax.streamplot(
            positions[0, :].reshape(n_grid, n_grid),
            positions[1, :].reshape(n_grid, n_grid),
            velocities[0, :].reshape(n_grid, n_grid),
            velocities[1, :].reshape(n_grid, n_grid),
            color=vectorfield_color,
            # color="red",
            # scale=50,
            zorder=-2,
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
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    # fig.tight_layout()

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
