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

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from dynamic_obstacle_avoidance.rotational.multi_hull_and_obstacle import (
    MultiHullAndObstacle,
)


def animation_obstacles():
    x_lim = [-10, 10]
    y_lim = [-3, 3]

    n_obstacles = 1

    radius = 0.8
    margin_absolut = 0.5

    obstacle_environment = RotationContainer()

    obstacle_environment.append(
        Ellipse(
            axes_length=np.array([radius, radius]),
            center_position=np.zeros([-8, 1]),
            linear_velocity=np.zeros([1, 0.1]),
            margin_absolut=margin_absolut,
        )
    )


if (__name__) == "__main__":
    plt.close("all")

    main()
