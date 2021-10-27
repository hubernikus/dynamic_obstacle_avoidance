""" Test Polygon. """
# Author: Lukas Huber
#
import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import Cuboid, Polygon
from dynamic_obstacle_avoidance.containers import GradientContainer

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)

from vartools.dynamical_systems import plot_dynamical_system_streamplot
from vartools.dynamical_systems import plot_dynamical_system_quiver


def visualize_square(n_resolution=20, x_lim=[-5, 5], y_lim=[-5, 5]):
    plt.ion()
    plt.show()

    main_cuboid = Cuboid(
        axes_length=[5, 3],
        center_position=[1, 0.2],
        orientation=0,
    )

    edge_points = [
        [1, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
    ]

    edge_points = [
        [1, -4],
        [1, 1],
        [-2, 1],
        [-2, -1],
        [-1, -1],
        [-1, -3],
    ]

    main_cuboid = Polygon(
        edge_points=np.array(edge_points).T,
        # center_position=[1, 0.2],
        orientation=0,
    )

    obs_list = GradientContainer()
    obs_list.append(main_cuboid)

    nx = ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx),
        np.linspace(y_lim[0], y_lim[1], ny),
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    normals = np.zeros(positions.shape)
    for it in range(positions.shape[1]):
        if main_cuboid.get_gamma(positions[:, it], in_global_frame=True) > 1:
            normals[:, it] = main_cuboid.get_normal_direction(
                positions[:, it], in_global_frame=True
            )

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_obstacles(ax=ax, obs=obs_list, x_range=x_lim, y_range=y_lim)

    ax.quiver(
        positions[0, :],
        positions[1, :],
        -normals[0, :],
        -normals[1, :],
        color="blue",
    )


if (__name__) == "__main__":
    visualize_square()
