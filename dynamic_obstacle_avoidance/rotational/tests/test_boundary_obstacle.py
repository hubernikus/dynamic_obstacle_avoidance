"""
Examples of how to use an obstacle-boundary mix,
i.e., an obstacle which can be entered

This could be bag in task-space, or a complex-learned obstacle.
"""
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-06-21

import copy

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid

from dynamic_obstacle_avoidance.rotational.multi_hull_and_obstacle import (
    MultiHullAndObstacle,
)


def test_boundary_obstacle_weight(visualize=False):
    obstacle = []
    obstacle.append(
        Cuboid(
            center_position=np.array([0, 0]),
            axes_length=np.array([2, 2]),
            is_boundary=False,
        )
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
        obstacle_list=obstacle, boundary_list=subhull
    )

    if visualize:
        n_resolution = 100
        x_lim = [-1.5, 1.5]
        y_lim = [-1.5, 1.5]

        nx = ny = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx),
            np.linspace(y_lim[0], y_lim[1], ny),
        )

        fig, axs = plt.subplots(1, my_hullobstacle.n_elements, figsize=(15, 6))
        # levels = np.linspace(0.01, 1., 20 + 1)
        levels = np.linspace(1e-5, 1.0 + 1e-5, 40 + 1)
        cmap = "YlGn"

        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        weights = np.zeros((my_hullobstacle.n_elements, positions.shape[1]))

        print("Start loop")
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
            axs[oo].set_title(f"Gamma Obstacle {oo}")

            # delta_pos = [0.1, 0.1]
            # axs[oo].text(
            #     gmm_ellipse.get_center_position(it_gmm)[0] + delta_pos[0],
            #     gmm_ellipse.get_center_position(it_gmm)[1] + delta_pos[1],
            #     s=(f"{it_gmm}"),
            #     fontsize="large",
            #     fontweight="bold",
            # )

        # print(f"roots = {gmm_ellipse.gmm_index_graph.get_root_indices()}")

        for ax in axs:
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(x_lim)
            ax.set_ylim(x_lim)
            ax.grid()

        plt.colorbar(cs0, ax=axs)

        my_hullobstacle.plot_obstacle(x_lim=x_lim, y_lim=y_lim)

    # Inside the ellipse
    position = np.array([0.5, -0.1])
    my_hullobstacle._evaluate_weights(position=position)
    weights = my_hullobstacle.weights
    assert np.isclose(sum(weights), 1), "Weights don't sum up to one."
    assert np.isclose(weights[my_hullobstacle._indices_boundaries[0]], 1)

    position = np.array([1.2, 0])
    my_hullobstacle._evaluate_weights(position=position)
    weights = my_hullobstacle.weights
    assert np.isclose(sum(weights), 1), "Weights don't sum up to one."

    position = np.array([2.0, 0.1])
    my_hullobstacle._evaluate_weights(position=position)
    weights = my_hullobstacle.weights
    assert np.isclose(sum(weights), 1), "Weights don't sum up to one."
    assert np.isclose(weights[my_hullobstacle._indices_obstacles[0]], 1)


if (__name__) == "__main__":
    plt.close("all")
    test_boundary_obstacle_weight(visualize=True)
