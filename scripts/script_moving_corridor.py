#!/USSR/bin/python3
""" Evaluating a multi-corridor environment """
# Author: Lukas Huber
# Created: 2022-03-21
# Email: lukas.huber@epfl.ch

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance import (
    obs_avoidance_interpolation_moving,
)

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)

from vartools.dynamical_systems import LinearSystem


def plot_multihulls(
    obstacle_environment,
    x_lim,
    y_lim,
    ax=None,
    show_margin=True,
    plot_boundary_multicolor=False,
):
    base_color_list = ["blue", "orange", "purple"]
    # Plot
    obstacle_color = np.array([176, 124, 124]) / 255.0

    if ax is None:
        fig, ax = plt.subplots()

    outer_boundary = np.array(
        [
            [x_lim[0], x_lim[1], x_lim[1], x_lim[0]],
            [y_lim[0], y_lim[0], y_lim[1], y_lim[1]],
        ]
    ).T

    boundary_polygon = plt.Polygon(outer_boundary, alpha=0.8, zorder=-4)

    boundary_polygon.set_color(obstacle_color)
    ax.add_patch(boundary_polygon)
    obs_polygon = []
    for ii, obs in enumerate(obstacle_environment):
        x_obs = np.array(obs.get_boundary_xy()).T
        obs_polygon.append(plt.Polygon(x_obs, alpha=1.0, zorder=-3))
        obs_polygon[-1].set_color(np.array([1.0, 1.0, 1.0]))
        ax.add_patch(obs_polygon[-1])

        if show_margin:
            x_obs_sf = np.array(obs.get_boundary_with_margin_xy()).T

            if plot_boundary_multicolor:
                ax.plot(
                    x_obs_sf[:, 0],
                    x_obs_sf[:, 1],
                    "--",
                    linewidth=5,
                    color=base_color_list[ii],
                )
            else:
                ax.plot(x_obs_sf[:, 0], x_obs_sf[:, 1], "k--")

    ax.axis("equal")

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    for ii, ds in enumerate(obstacle_environment.dynamical_systems):
        attractor = ds.attractor_position
        if plot_boundary_multicolor:
            ax.plot(
                attractor[0],
                attractor[1],
                "*",
                color=base_color_list[ii],
                markeredgewidth=2,
                markersize=12,
                zorder=5,
            )

            ax.plot(
                attractor[0],
                attractor[1],
                "o",
                color="white",
                markeredgewidth=2,
                markersize=15,
                zorder=4,
            )
        else:
            ax.plot(
                attractor[0],
                attractor[1],
                "k*",
                markeredgewidth=2,
                markersize=12,
                zorder=5,
            )


class SimpleMultiBoundary(ObstacleContainer):
    """Multiboundarycontainer with multiple dynamical systems."""

    def __init__(self, dynamical_systems=None, **kwargs):
        super().__init__(**kwargs)

        # List of dynamical systems
        if dynamical_systems is None:
            self.dynamical_systems = []
        else:
            self.dynamical_systems = dynamical_systems

    def get_importance_weight(self, position):
        """Evaluate weights with the following conditions:
        1) Sum = 1
        2) Gamma=1 -> weight=0"""
        gammas = np.zeros(self.n_obstacles)
        for ii, obs in enumerate(self._obstacle_list):
            gammas[ii] = obs.get_gamma(position, in_global_frame=True)

        it_inside = gammas > 1
        if not any(it_inside):
            # Zero velocity if collided with all walls
            return np.zeros(gammas.shape)

        weights = np.maximum(gammas - 1, 0)
        weights = weights / np.sum(weights)
        return weights

    def evaluate_multiboundary(self, position, initial_velocity=None):
        weights = self.get_importance_weight(position)

        modulated_dynamics = np.zeros(position.shape)
        for oo, obs in enumerate(self._obstacle_list):
            if weights[oo] <= 0:
                continue

            if initial_velocity is None:
                velocity = self.dynamical_systems[oo].evaluate(position)
            else:
                velocity = initial_velocity

            mod_vel_oo = obs_avoidance_interpolation_moving(position, velocity, [obs])
            modulated_dynamics = modulated_dynamics + mod_vel_oo * weights[oo]

        return modulated_dynamics


def main_moving_in_corridor(n_resolution=100, save_figure=False):
    x_lim = [-1, 12]
    y_lim = [-0.4, 10.0]

    # initial_dynamics = LinearSystem(attractor_position=np.array([9.0, 2.0]))

    obstacle_environment = SimpleMultiBoundary()
    obstacle_environment.append(
        Cuboid(
            center_position=[5, 8],
            orientation=0,
            axes_length=[9, 3],
            is_boundary=True,
            tail_effect=False,
        )
    )

    obstacle_environment.dynamical_systems.append(
        LinearSystem(attractor_position=np.array([2.0, 6.5]))
    )

    obstacle_environment.append(
        Cuboid(
            center_position=[2, 5],
            orientation=0,
            axes_length=[3, 9],
            is_boundary=True,
            tail_effect=False,
        )
    )

    obstacle_environment.dynamical_systems.append(
        LinearSystem(attractor_position=np.array([3.5, 2.0]))
    )

    obstacle_environment.append(
        Cuboid(
            center_position=[5, 2],
            orientation=0,
            axes_length=[9, 3],
            is_boundary=True,
            tail_effect=False,
        )
    )

    obstacle_environment.dynamical_systems.append(
        LinearSystem(attractor_position=np.array([9.0, 2.0]))
    )

    nx = n_resolution
    ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    modulated_velocity = np.zeros(positions.shape)

    for ii in range(positions.shape[1]):
        # initial_velocity = initial_dynamics.evaluate(positions[:, ii])
        modulated_velocity[:, ii] = obstacle_environment.evaluate_multiboundary(
            positions[:, ii],
            # initial_velocity,
        )

    figsize = (5.0, 4)

    fig, ax = plt.subplots(figsize=figsize)
    # ax.quiver(
    # positions[0, :],
    # positions[1, :],
    # modulated_velocity[0, :],
    # modulated_velocity[1, :],
    # )

    ax.streamplot(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        modulated_velocity[0, :].reshape(nx, ny),
        modulated_velocity[1, :].reshape(nx, ny),
        color="black",
    )

    plot_multihulls(
        obstacle_environment, x_lim, y_lim, ax=ax, plot_boundary_multicolor=True
    )

    if save_figure:
        figure_name = "multiboundary_with_vectorfield"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight", dpi=600)

    fig, ax = plt.subplots(figsize=figsize)
    obstacle_environment.dynamical_systems = [
        obstacle_environment.dynamical_systems[-1]
    ]
    plot_multihulls(obstacle_environment, x_lim, y_lim, ax=ax, show_margin=False)

    if save_figure:
        figure_name = "multiboundary"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight", dpi=600)


def plot_weights(n_resolution=100):
    x_lim = [-2, 13]
    y_lim = [-1, 10]

    obstacle_environment = SimpleMultiBoundary()
    obstacle_environment.append(
        Cuboid(
            center_position=[5, 2],
            orientation=0,
            axes_length=[9, 3],
            is_boundary=True,
        )
    )

    obstacle_environment.append(
        Cuboid(
            center_position=[2, 5],
            orientation=0,
            axes_length=[3, 9],
            is_boundary=True,
        )
    )

    obstacle_environment.append(
        Cuboid(
            center_position=[5, 8],
            orientation=0,
            axes_length=[9, 3],
            is_boundary=True,
        )
    )

    nx = n_resolution
    ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    weights = np.zeros((len(obstacle_environment), positions.shape[1]))

    for ii in range(positions.shape[1]):
        weights[:, ii] = obstacle_environment.get_importance_weight(positions[:, ii])

    levels = np.linspace(0 + 1e-2, 1 - 1e-2, 20)

    fig, axs = plt.subplots(1, len(obstacle_environment), figsize=(19, 4.0))
    for ii, ax in enumerate(axs):
        plot_multihulls(obstacle_environment, x_lim, y_lim, ax=ax)

        cols = axs[ii].contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            weights[ii, :].reshape(nx, ny),
            levels=levels,
            extend="max",
            zorder=-1,
            # cmap='hot',
            # cmap='Blues_r',
            cmap="Blues",
            alpha=0.8,
        )

        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)


if (__name__) == "__main__":
    plt.close("all")

    main_moving_in_corridor(save_figure=True)
    # plot_weights()
