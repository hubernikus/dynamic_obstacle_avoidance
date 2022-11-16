"""
Examples of how to use an obstacle-boundary mix,
i.e., an obstacle which can be entered

This could be bag in task-space, or a complex-learned obstacle.
"""
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-06-21

import networkx as nx

import copy

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid

from dynamic_obstacle_avoidance.rotational.utils import gamma_normal_gradient_descent
from dynamic_obstacle_avoidance.rotational.multi_hull_and_obstacle import (
    MultiHullAndObstacle,
)


def test_boundary_obstacle_weight(visualize=False, savefig=False):
    outer_obstacle = Cuboid(
        center_position=np.array([0, 0]),
        axes_length=np.array([2, 2]),
        is_boundary=False,
    )

    subhull = []
    subhull.append(
        Ellipse(
            center_position=np.array([0.8, 0]),
            axes_length=np.array([1.5, 1.0]),
            is_boundary=True,
        )
    )
    subhull[-1].set_reference_point(np.array([1.1, 0]), in_global_frame=True)

    my_hullobstacle = MultiHullAndObstacle(
        outer_obstacle=outer_obstacle, inner_obstacles=subhull
    )

    if visualize:
        n_resolution = 100
        x_lim = [-1.5, 1.5]
        y_lim = [-1.5, 1.5]

        n_x = n_y = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_x),
            np.linspace(y_lim[0], y_lim[1], n_y),
        )

        fig, axs = plt.subplots(1, my_hullobstacle.n_elements, figsize=(15, 6))
        # levels = np.linspace(0.01, 1., 20 + 1)
        levels = np.linspace(1e-5, 1.0 + 1e-5, 40 + 1)
        cmap = "YlGn"

        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        weights = np.zeros((my_hullobstacle.n_elements, positions.shape[1]))

        for ii in range(positions.shape[1]):
            my_hullobstacle._evaluate_weights(position=positions[:, ii])
            weights[:, ii] = my_hullobstacle.weights

        for oo in range(my_hullobstacle.n_elements):
            cs0 = axs[oo].contourf(
                x_vals,
                y_vals,
                weights[oo, :].reshape(x_vals.shape),
                levels=levels,
                cmap=cmap,
            )
            axs[oo].set_title(f"Weight obstacle {oo}")

        for ax in axs:
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(x_lim)
            ax.set_ylim(x_lim)
            ax.grid()

        plt.colorbar(cs0, ax=axs)

        if savefig:
            figname = "simple_hull_obstacle_weights"
            plt.savefig("figures/" + figname + ".pdf", bbox_inches="tight")

        my_hullobstacle.plot_obstacle(x_lim=x_lim, y_lim=y_lim)

        if savefig:
            figname = "simple_hull_obstacle"
            plt.savefig("figures/" + figname + ".pdf", bbox_inches="tight")

    # Inside the ellipse
    position = np.array([0.5, -0.1])
    my_hullobstacle._evaluate_weights(position=position)
    weights = my_hullobstacle.weights
    assert np.isclose(sum(weights), 1), "Weights don't sum up to one."
    assert np.isclose(weights[my_hullobstacle._indices_inner[0]], 1)

    position = np.array([1.2, 0])
    my_hullobstacle._evaluate_weights(position=position)
    weights = my_hullobstacle.weights
    assert np.isclose(sum(weights), 1), "Weights don't sum up to one."

    position = np.array([2.0, 0.1])
    my_hullobstacle._evaluate_weights(position=position)
    weights = my_hullobstacle.weights
    assert np.isclose(sum(weights), 1), "Weights don't sum up to one."
    assert np.isclose(weights[my_hullobstacle._indices_outer], 1)


def _test_mixed_boundary_obstacle_reference(visualize=False):
    outer_obstacle = Cuboid(
        center_position=np.array([0, 0]),
        axes_length=np.array([2, 2]),
        is_boundary=False,
    )

    subhull = []
    subhull.append(
        Ellipse(
            center_position=np.array([0.8, 0]),
            axes_length=np.array([1.5, 1.0]),
            is_boundary=True,
        )
    )

    subhull.append(
        Ellipse(
            center_position=np.array([-0.2, 0.1]),
            axes_length=np.array([1.0, 1.2]),
            is_boundary=True,
        )
    )

    my_hullobstacle = MultiHullAndObstacle(
        outer_obstacle=outer_obstacle, inner_obstacles=subhull
    )

    # my_hullobstacle.inner_obstacles[-1].is_boundary = False
    entrance_position = gamma_normal_gradient_descent(
        [outer_obstacle, subhull[0]],
        powers=[-2, -2],  # -> both in free space, i.e. > 0
        factors=[1, -1],
    )

    connection_position = gamma_normal_gradient_descent(
        [subhull[0], subhull[1]],
        powers=[-2, -2],  # -> both in free space, i.e. > 0
        factors=[-1, -1],
    )

    if visualize:
        n_resolution = 100
        x_lim = [-1.5, 1.5]
        y_lim = [-1.5, 1.5]

        n_x = n_y = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_x),
            np.linspace(y_lim[0], y_lim[1], n_y),
        )

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        my_hullobstacle.plot_obstacle(x_lim=x_lim, y_lim=y_lim, ax=ax)

        ax.plot(entrance_position[0], entrance_position[1], "r+")
        ax.plot(connection_position[0], connection_position[1], "g+")

    assert (
        my_hullobstacle.outer_obstacle.get_gamma(
            entrance_position, in_global_frame=True
        )
        > 1
    )
    assert (
        my_hullobstacle.inner_obstacles[0].get_gamma(
            entrance_position, in_global_frame=True
        )
        > 1
    )

    assert (
        my_hullobstacle.inner_obstacles[0].get_gamma(
            connection_position, in_global_frame=True
        )
        > 1
    )
    assert (
        my_hullobstacle.inner_obstacles[1].get_gamma(
            connection_position, in_global_frame=True
        )
        > 1
    )

    # Automatically generate hirarchy and descent -> check that it is the same
    my_hullobstacle.evaluate_hirarchy_and_reference_points()

    assert len(my_hullobstacle._graph.nodes) == 3
    assert len(my_hullobstacle._graph.edges) == 2
    automated_connection = my_hullobstacle._graph[my_hullobstacle.inner_obstacles[0]][
        my_hullobstacle.inner_obstacles[1]
    ]["intersection"]
    assert np.allclose(automated_connection, connection_position)

    # assert len(my_hullobstacle._entrance_positions) == 1
    # assert my_hullobstacle._entrance_obstacles[0] == my_hullobstacle.inner_obstacles[0]
    # assert np.allclose(my_hullobstacle._entrance_positions[0], entrance_position)
    automated_entrance = my_hullobstacle._graph[my_hullobstacle.inner_obstacles[0]][0][
        "intersection"
    ]
    assert np.allclose(automated_entrance, entrance_position)


def test_obstacle_without_interior(visualize=False):
    outer_obstacle = Cuboid(
        center_position=np.array([0, 0]),
        axes_length=np.array([2, 2]),
        is_boundary=False,
    )

    subhull = []

    my_hullobstacle = MultiHullAndObstacle(
        outer_obstacle=outer_obstacle, inner_obstacles=subhull
    )

    my_hullobstacle.evaluate_hirarchy_and_reference_points()
    my_hullobstacle.set_attractor(np.array([1.3, 0.2]))

    if visualize:
        n_resolution = 30

        x_lim = [-3.0, 3.0]
        y_lim = [-3.0, 3.0]

        n_x = n_y = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_x),
            np.linspace(y_lim[0], y_lim[1], n_y),
        )

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        my_hullobstacle.plot_obstacle(
            x_lim=x_lim, y_lim=y_lim, ax=ax, plot_attractors=True
        )

        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        velocities = np.zeros(positions.shape)

        for ii in range(positions.shape[1]):
            velocities[:, ii] = my_hullobstacle.evaluate(position=positions[:, ii])

        ax.quiver(
            positions[0, :],
            positions[1, :],
            velocities[0, :],
            velocities[1, :],
            color="k",
            zorder=4,
        )

        # Going around int the correct direction
        velocity = my_hullobstacle.evaluate(position=np.array([-1.01, 0.3]))
        assert np.allclose(
            velocity, [0, 1], atol=1e-1
        ), "Vector not pointing in the correct direction."


def test_shortes_path(visualize=False, savefig=False):
    # outer_obstacle = Cuboid(
    #     center_position=np.array([0, 0]),
    #     axes_length=np.array([2, 2]),
    #     is_boundary=False,
    # )

    # subhull = []
    # subhull.append(
    #     Ellipse(
    #         center_position=np.array([0.8, 0]),
    #         axes_length=np.array([1.5, 1.0]),
    #         is_boundary=True,
    #     )
    # )

    # subhull.append(
    #     Ellipse(
    #         center_position=np.array([-0.2, 0.1]),
    #         axes_length=np.array([1.0, 1.2]),
    #         is_boundary=True,
    #     )
    # )

    # my_hullobstacle = MultiHullAndObstacle(
    #     outer_obstacle=outer_obstacle, inner_obstacles=subhull
    # )

    # my_hullobstacle.evaluate_hirarchy_and_reference_points()
    # my_hullobstacle.set_attractor(np.array([0, 0]))

    outer_obstacle = Cuboid(
        center_position=np.array([0, 0]),
        axes_length=np.array([2, 2]),
        is_boundary=False,
    )

    subhull = []
    subhull.append(
        Ellipse(
            center_position=np.array([0.5, -0.4]),
            axes_length=np.array([1.5, 0.5]),
            is_boundary=True,
        )
    )

    subhull.append(
        Ellipse(
            center_position=np.array([-0.2, 0.1]),
            axes_length=np.array([1.0, 1.2]),
            is_boundary=True,
        )
    )

    my_hullobstacle = MultiHullAndObstacle(
        outer_obstacle=outer_obstacle, inner_obstacles=subhull
    )

    my_hullobstacle.evaluate_hirarchy_and_reference_points()
    my_hullobstacle.set_attractor(np.array([-0.3, 0.4]))

    if visualize:
        n_resolution = 50

        # x_lim = [-1.5, 1.5]
        # y_lim = [-1.5, 1.5]

        x_lim = [-2.0, 2.0]
        y_lim = [-2.0, 2.0]

        n_x = n_y = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_x),
            np.linspace(y_lim[0], y_lim[1], n_y),
        )

        # position = np.array([0.6, -0.4])
        # velocity = my_hullobstacle.evaluate(position=position)

        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        velocities = np.zeros(positions.shape)

        # Weight container for all obstacles
        n_obs = 3
        weights = np.zeros((n_obs, positions.shape[1]))

        velocities_partial = np.zeros((positions.shape[0], positions.shape[1], n_obs))

        for ii in range(positions.shape[1]):
            velocities[:, ii] = my_hullobstacle.evaluate(position=positions[:, ii])
            weights[:, ii] = my_hullobstacle.weights

            obs_hashes = my_hullobstacle.inner_obstacles + [None]
            for ind, obs_hash in enumerate(obs_hashes):
                if my_hullobstacle.weights[ind]:
                    velocities_partial[
                        :, ii, ind
                    ] = my_hullobstacle._get_local_dynamics(
                        positions[:, ii],
                        obs_hash=obs_hash,
                        gamma=my_hullobstacle.gamma_list[ind],
                    )

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        my_hullobstacle.plot_obstacle(
            x_lim=x_lim, y_lim=y_lim, ax=ax, plot_attractors=True
        )

        if savefig:
            figname = "clustering_obstacles"
            plt.savefig("figures/" + figname + ".pdf", bbox_inches="tight")

        # ax.quiver(
        #     positions[0, :],
        #     positions[1, :],
        #     velocities[0, :],
        #     velocities[1, :],
        #     color="k",
        #     zorder=4,
        # )

        ax.streamplot(
            positions[0, :].reshape(n_x, n_y),
            positions[1, :].reshape(n_x, n_y),
            velocities[0, :].reshape(n_x, n_y),
            velocities[1, :].reshape(n_x, n_y),
            color="blue",
            zorder=4,
        )

        if savefig:
            figname = "moving_outside_to_inside"
            plt.savefig("figures/" + figname + ".pdf", bbox_inches="tight")

        # _, axs = plt.subplots(1, my_hullobstacle.n_elements, figsize=(15, 3.5))

        for oo in range(my_hullobstacle.n_elements):
            fig, ax = plt.subplots(1, 1, figsize=(8, 7))
            # axs[oo].quiver(
            ax.quiver(
                positions[0, :],
                positions[1, :],
                velocities_partial[0, :, oo],
                velocities_partial[1, :, oo],
                # color="k",
                color="blue",
                zorder=4,
                scale=45,
            )

            # ax.streamplot(
            #     positions[0, :].reshape(n_x, n_y),
            #     positions[1, :].reshape(n_x, n_y),
            #     velocities_partial[0, :, oo].reshape(n_x, n_y),
            #     velocities_partial[1, :, oo].reshape(n_x, n_y),
            #     color="blue",
            #     zorder=4,
            # )

            my_hullobstacle.plot_obstacle(
                x_lim=x_lim, y_lim=y_lim, ax=ax, plot_attractors=True
            )

            if savefig:
                figname = f"moving_outside_partial_{oo}"
                plt.savefig("figures/" + figname + ".pdf", bbox_inches="tight")

        # Weights
        # fig, axs = plt.subplots(1, my_hullobstacle.n_elements, figsize=(15, 3.5))
        # for oo in range(my_hullobstacle.n_elements):
        #     levels = np.linspace(0, 1, 21)
        #     # levels = np.linspace(1, 10, 10)
        #     cs0 = axs[oo].contourf(
        #         x_vals,
        #         y_vals,
        #         weights[oo, :].reshape(x_vals.shape),
        #         levels=levels,
        #         alpha=0.7,
        #         zorder=5,
        #         # cmap=cmap,
        #     )
        #     axs[oo].set_title(f"obstacle #{oo}")
        #     my_hullobstacle.plot_obstacle(
        #         x_lim=x_lim,
        #         y_lim=y_lim,
        #         ax=axs[oo],
        #     )
        # plt.colorbar(cs0, ax=axs)


def _test_multiholes_obstacle(visualize=False, savefig=False):
    # TODO: this does not work yet (!)
    outer_obstacle = Cuboid(
        center_position=np.array([0, 0]),
        axes_length=np.array([2, 3]),
        is_boundary=False,
    )

    subhull = []
    subhull.append(
        Ellipse(
            center_position=np.array([-0.5, 1.0]),
            axes_length=np.array([1.5, 0.5]),
            is_boundary=True,
        )
    )

    subhull.append(
        Ellipse(
            center_position=np.array([0.5, -0.3]),
            axes_length=np.array([1.4, 1.2]),
            is_boundary=True,
        )
    )

    my_hullobstacle = MultiHullAndObstacle(
        outer_obstacle=outer_obstacle, inner_obstacles=subhull
    )

    if visualize:
        my_hullobstacle.evaluate_hirarchy_and_reference_points()
        my_hullobstacle.set_attractor(np.array([-0.3, 0.4]))

        x_lim = [-2.0, 2.0]
        y_lim = [-2.0, 2.0]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        my_hullobstacle.plot_obstacle(
            x_lim=x_lim, y_lim=y_lim, ax=ax, plot_attractors=True
        )

        n_resolution = 50

        n_x = n_y = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_x),
            np.linspace(y_lim[0], y_lim[1], n_y),
        )

        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        velocities = np.zeros(positions.shape)

        # Weight container for all obstacles
        n_obs = 3
        weights = np.zeros((n_obs, positions.shape[1]))

        velocities_partial = np.zeros((positions.shape[0], positions.shape[1], n_obs))

        for ii in range(positions.shape[1]):
            velocities[:, ii] = my_hullobstacle.evaluate(position=positions[:, ii])
            weights[:, ii] = my_hullobstacle.weights

            obs_hashes = my_hullobstacle.inner_obstacles + [None]
            for ind, obs_hash in enumerate(obs_hashes):
                if my_hullobstacle.weights[ind]:
                    velocities_partial[
                        :, ii, ind
                    ] = my_hullobstacle._get_local_dynamics(
                        positions[:, ii],
                        obs_hash=obs_hash,
                        gamma=my_hullobstacle.gamma_list[ind],
                    )

        ax.streamplot(
            positions[0, :].reshape(n_x, n_y),
            positions[1, :].reshape(n_x, n_y),
            velocities[0, :].reshape(n_x, n_y),
            velocities[1, :].reshape(n_x, n_y),
            color="blue",
            zorder=4,
        )

        if savefig:
            figname = "obstacle_with_mulitholes"
            plt.savefig("figures/" + figname + ".pdf", bbox_inches="tight")


if (__name__) == "__main__":
    plt.close("all")
    # test_boundary_obstacle_weight(visualize=True, savefig=True)
    # _test_mixed_boundary_obstacle_reference(visualize=True)
    # test_shortes_path(visualize=True, savefig=False)

    # _test_multiholes_obstacle(visualize=True, savefig=True)

    # test_obstacle_without_interior(visualize=True)
