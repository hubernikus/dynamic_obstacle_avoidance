"""
Testing script to ensure that 
"""
import pytest

import numpy as np

from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import Polygon


def test_inverted_obstacle_ellipse():
    my_obstacle = Ellipse(
        center_position=np.array([0, 0]),
        axes_length=np.array([4, 4]),
        is_boundary=True,
    )

    point = np.array([0, 1])
    gamma = my_obstacle.get_gamma(point, in_global_frame=True)
    assert gamma > 1, "Gamma is not greater than 1 for ellipse-hull."


def test_inverted_obstacle_cuboid():
    my_obstacle = Cuboid(
        center_position=np.array([0, 0]),
        axes_length=np.array([4, 4]),
        is_boundary=True,
    )
    point = np.array([0.5, 0])

    # Once this is working, one can move on
    gamma = my_obstacle.get_gamma(point, in_global_frame=True)
    assert gamma > 1, "Gamma is not greater than 1 for couboid-hull."

    x_range = [-1, 11]
    y_range = [-1, 11]
    obs_xaxis = [x_range[0] + 1, x_range[1] - 1]
    obs_yaxis = [y_range[0] + 1, y_range[1] - 1]

    edge_points = np.array(
        [
            [obs_xaxis[0], obs_xaxis[0], obs_xaxis[1], obs_xaxis[1]],
            [obs_yaxis[1], obs_yaxis[0], obs_yaxis[0], obs_yaxis[1]],
        ]
    )

    my_obstacle = Polygon(
        edge_points=edge_points,
        is_boundary=True,
        tail_effect=False,
    )

    point = np.array([4.29041985, 9.64265615])

    gamma = my_obstacle.get_gamma(point, in_global_frame=True)

    assert gamma > 1, f"Gamma={gamma} at position={point}"


if (__name__) == "__main__":
    # test_inverted_obstacle_ellipse()
    test_inverted_obstacle_cuboid()

    print("Done testing")
