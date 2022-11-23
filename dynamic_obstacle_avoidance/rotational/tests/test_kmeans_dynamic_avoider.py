#!/USSR/bin/python3.10
""" Test KMeans with Dynamic Avoider.

TODO:
-----
> Fix bug (!)
> Combine with (rotational) obstacle avoidance
> Avoid the overrotation (?)
"""
# Author: Lukas Huber
# Created: 2022-11-22
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2022

import math
from typing import Callable

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.handwritting_handler import MotionDataHandler
from vartools.dynamical_systems import DynamicalSystem

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from dynamic_obstacle_avoidance.rotational.rotation_container import RotationContainer
from dynamic_obstacle_avoidance.rotational.rotational_avoidance import (
    obstacle_avoidance_rotational,
)

# from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.rotational.kmeans_motion_learner import (
    KMeansMotionLearner,
)
from dynamic_obstacle_avoidance.rotational.datatypes import Vector

from dynamic_obstacle_avoidance.rotational.tests import helper_functions


def move_obstacles_2d(
    obstacle_container: ObstacleContainer, timestep: float = 0.1
) -> None:
    """Updates position and orientation of dynamic obstacles with internal velocity."""
    for obs in obstacle_container:
        if obs.linear_velocity is not None:
            obs.center_position = obs.center_position + obs.linear_velocity * timestep
        if obs.angular_velocity is not None:
            obs.orientation = obs.orientation + obs.angular_velocity * dt


def plot_obstacle_dynamics(
    obstacle_container: ObstacleContainer,
    dynamics: Callable[[Vector], Vector],
    x_lim: list[float],
    y_lim: list[float],
    n_grid: int = 20,
    ax=None,
    attractor_position=None,
):
    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], n_grid),
        np.linspace(y_lim[0], y_lim[1], n_grid),
    )
    positions = np.array([xx.flatten(), yy.flatten()])
    velocities = np.zeros_like(positions)

    for pp in range(positions.shape[1]):
        # print(f"{positions[:, pp]=} | {velocities[:, pp]=}")
        if obstacle_container.get_minimum_gamma(positions[:, pp]) <= 1:
            continue

        velocities[:, pp] = dynamics(positions[:, pp])

    ax.quiver(
        positions[0, :],
        positions[1, :],
        velocities[0, :],
        velocities[1, :],
        # color="red",
        scale=50,
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

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    return (fig, ax)


class AvoiderWithKMeansTrajectory:
    def __init__(
        self,
        kmeans_learner: KMeansMotionLearner,
        environment_container: ObstacleContainer,
    ) -> None:
        self.environment_container = environment_container
        self.kmeans_learner = kmeans_learner

        # This radius is used for the avoidance of the obstacles
        self.margin_radius = 1

    def reposition_obstacle_free(self):
        """Reposition clusters to be a minimum away from the obstacles."""

        # Update radius such that there is a minimum overlap between the obstacles
        # so far we increase the size equally | they could be rotated, too
        pass

    def evaluate(self, position):
        learned_motion = self.kmeans_learner.evaluate(position)
        initial_velocity = self.kmeans_learner.predict_averaged_lyapunov(position)

        avoidance_velocity = obstacle_avoidance_rotational(
            position=position,
            initial_velocity=learned_motion,
            convergence_velocity=initial_velocity,
            obstacle_list=self.environment_container,
            sticky_surface=False,
        )

        # TODO: what about the graph avoider (would it help?)
        return avoidance_velocity


def create_boundary_vectorfield():
    positions = np.array(
        [
            [-1, 0],
            [-2, 0],
            [-2, 2],
            [0, 2],
            [2, 2],
            [2, 0],
            [1, 0],
        ]
    )

    attractor_position = np.array([0.8, 0])

    datahandler = MotionDataHandler(position=positions)
    datahandler.velocity = datahandler.position[1:, :] - datahandler.position[:-1, :]
    datahandler.velocity = np.vstack(
        (datahandler.velocity, [attractor_position - datahandler.position[-1, :]])
    )
    datahandler.attractor = attractor_position
    datahandler.sequence_value = np.linspace(0, 1, positions.shape[0])
    return KMeansMotionLearner(datahandler, n_clusters=positions.shape[0])


def _test_kmeans_dynamic_avoider(visualize=False, save_figure=False):
    """Not really a test - but rather a visualization."""
    main_learner = create_boundary_vectorfield()

    obstacle_container = RotationContainer()
    obstacle_container.append(
        Cuboid(
            center_position=np.array([0, 1]),
            axes_length=np.array([4, 1]),
            # orientation=30 / 90.0 * math.pi,
        )
    )

    obstacle_container.append(
        Cuboid(
            center_position=np.array([0, -0.25]),
            axes_length=np.array([1, 1.5]),
            # orientation=30 / 90.0 * math.pi,
        )
    )

    obstacle_container.append(
        Cuboid(
            center_position=np.array([0, -2.0]),
            axes_length=np.array([8, 2]),
            # orientation=30 / 90.0 * math.pi,
        )
    )

    motion_handler = AvoiderWithKMeansTrajectory(
        kmeans_learner=main_learner,
        environment_container=obstacle_container,
    )

    # # Test quickly the gamma minimum function
    # position = np.array([-1, 1])
    # min_gamma = obstacle_container.get_minimum_gamma(position)
    # breakpoint()

    # # Teset positoin
    # position = np.array([1.9, -0.7])
    # velocity = main_learner.predict(position)
    # breakpoint()

    # # Test positions
    # position = np.array([-1.9, -0.7])
    # velocity = main_learner.predict(position)
    # breakpoint()

    if visualize:
        plt.close("all")
        x_lim, y_lim = [-4, 4], [-1.8, 3.5]
        figsize = (10, 7)

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            ax=ax,
            obstacle_container=motion_handler.environment_container,
            dynamics=motion_handler.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            attractor_position=main_learner.attractor_position,
        )
        plot_obstacles(obstacle_container, ax=ax, x_lim=x_lim, y_lim=y_lim)
        if save_figure:
            fig_name = "t_table_collision_free_dynamics"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight")

            fig, ax = plt.subplots(figsize=figsize)
        plot_obstacles(obstacle_container, ax=ax, x_lim=x_lim, y_lim=y_lim)

        if save_figure:
            fig_name = "t_table_obstacles_only"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight")

        fig, ax = plt.subplots(figsize=figsize)
        main_learner.plot_kmeans(ax=ax, x_lim=x_lim, y_lim=y_lim)

        if save_figure:
            fig_name = "t_table_resulting_clustering"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight")

        fig, ax = plt.subplots(figsize=figsize)
        helper_functions.plot_global_dynamics(main_learner, x_lim, y_lim, ax=ax)
        plot_obstacles(obstacle_container, ax=ax, x_lim=x_lim, y_lim=y_lim)

        if save_figure:
            fig_name = "t_table_global_dynamics"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight")


def _test_dynamic_avoidance():
    pass


if (__name__) == "__main__":
    figtype = ".png"
    # figtype = ".pdf"

    _test_kmeans_dynamic_avoider(visualize=True, save_figure=False)

    print("[Rotational Tests] Done tests")
