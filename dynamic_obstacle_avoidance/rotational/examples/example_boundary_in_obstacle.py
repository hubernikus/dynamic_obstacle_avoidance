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

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from dynamic_obstacle_avoidance.rotational.multi_hull_and_obstacle import (
    MultiHullAndObstacle,
)


def main():
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
            axes_length=np.array([1.0, 0.5]),
            is_boundary=True,
        )
    )
    # gamma = subhull[-1].get_gamma(np.array([1.1, 0]), in_global_frame=True)
    # print(gamma)

    subhull[-1].set_reference_point(np.array([1.1, 0]), in_global_frame=True)

    # subhull.append(
    #     Ellipse(
    #         center_position=np.array([0.1, 0]),
    #         axes_length=np.array([0.9, 1.2]),
    #         is_boundary=False,
    #     )
    # )

    my_hullobstaclle = MultiHullAndObstacle(
        obstacle_list=obstacle, boundary_list=subhull
    )

    my_hullobstaclle.plot_obstacle(x_lim=[-2, 2], y_lim=[-2, 2])


if (__name__) == "__main__":
    plt.close("all")

    main()
