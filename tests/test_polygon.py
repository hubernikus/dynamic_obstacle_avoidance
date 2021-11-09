""" Test Polygon. """
# Author: Lukas Huber
# Created: 2021-11-09
# License: BSD (c) 2021
from math import pi

import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import Cuboid, Polygon
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)

from vartools.dynamical_systems import plot_dynamical_system_streamplot
from vartools.dynamical_systems import plot_dynamical_system_quiver


def test_simple_cube_creation(visualize=False):
    my_obstacle = Cuboid(
        center_position=np.array([1, 2]),
        orientation=40 * pi / 180,
        axes_length=np.array([1, 2]),
        )

    if visualize:
        fig, axs = plt.subplots(1, 2)

        x_lim = [-2, 6]
        y_lim = [-2, 4]

        my_obstacle.plot2D(axs[0])

        # plot_obstacles(axs[0], obs_list, x_lim, y_lim)

    my_obstacle.set_reference_point(np.array([2, 3]), in_global_frame=True)

    if visualize:
        x_lim = [-2, 6]
        y_lim = [-2, 4]

        my_obstacle.plot2D(axs[1])


if (__name__) == "__main__":
    test_simple_cube_creation(visualize=True)
