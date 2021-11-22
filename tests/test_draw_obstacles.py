""" Test the drawing of different obstacles."""
# Author: Lukas Huber
# Created: 2021-11-09
# License: BSD (c) 2021

import matplotlib.pyplot as plt
import numpy as np

from dynamic_obstacle_avoidance.obstacles import Cuboid, Polygon


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


def test_draw_polygon_with_margin():
    pass


def test_draw_polygon_ref_point():
    pass


def test_draw_polygon_ref_point():
    pass


if (__name__) == "__main__":
    test_draw_polygon(visualize=True)

    # pass
