""" Test the drawing of different obstacles."""
# Author: Lukas Huber
# Created: 2021-11-09
# License: BSD (c) 2021

import matplotlib.pyplot as plt
import numpy as np

from dynamic_obstacle_avoidance.obstacles import Cuboid, Polygon, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles


def test_draw_polygon(visualize=False):
    """Triangle object."""
    edge_points = [
        [0, 0],
        [1, 0],
        [0, 1],
    ]

    my_obstacle = Polygon(edge_points=np.array(edge_points).T)

    if visualize:
        fig, ax = plt.subplots()
        my_obstacle.plot_obstacles(ax=ax)


def test_draw_ellipse(visualize=False):
    """Triangle object."""
    axes_length = np.array([1, 2])
    my_obstacle = Ellipse(
        center_position=np.array([0, 0]), axes_length=axes_length, margin_absolut=0
    )

    if visualize:
        fig, ax = plt.subplots()
        my_obstacle.plot2D(ax=ax)


def test_draw_polygon_with_margin():
    pass


def test_draw_polygon_ref_point():
    pass


def test_draw_polygon_ref_point():
    pass


if (__name__) == "__main__":
    # test_draw_polygon(visualize=True)
    test_draw_ellipse(visualize=True)

    # pass
