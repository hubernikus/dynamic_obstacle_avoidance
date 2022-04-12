#!/USSR/bin/python3.9
""" Test multimensional obstacles. """
# Author: Lukas Huber
# Created: 2021-08-04
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

# import unittest
from math import pi

import numpy as np
from numpy import linalg as LA

import shapely

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes
from dynamic_obstacle_avoidance.obstacles import CuboidXd

# from dynamic_obstacle_avoidance import obstacles


def test_multidimensional_ellipse(visualize=False):

    axes_length = np.array([2, 1])
    center_position = np.array([0, 0])

    ellipse = shapely.affinity.scale(
        shapely.geometry.Point(center_position[0], center_position[1]).buffer(1),
        axes_length[0],
        axes_length[1],
    )
    # ellipse = shapely.affinity.rotate(ellipse, 50)

    obstacle = EllipseWithAxes(center_position=center_position, axes_length=axes_length)

    point0 = np.array([3, 1])
    normal0 = obstacle.get_normal_direction(point0, in_obstacle_frame=False)
    surf_point0 = obstacle.get_point_on_surface(point0, in_obstacle_frame=False)

    if visualize:
        import matplotlib.pyplot as plt  # TODO: remove for production

        plt.ion()

        fig, ax = plt.subplots()

        xx, yy = ellipse.exterior.xy
        ax.plot(xx, yy, color="black", alpha=0.3)

        polygon_path = plt.Polygon(np.vstack((xx, yy)).T, alpha=0.1, zorder=-4)
        polygon_path.set_color("black")
        ax.add_patch(polygon_path)

        ax.set_aspect("equal")

        ax.arrow(surf_point0[0], surf_point0[1], normal0[0], normal0[1])

        ax.plot(surf_point0[0], surf_point0[1], "ro")

    pass


def test_multidimensional_cuboid(visualize=False):
    axes_length = np.array([2.0, 1.0])
    center_position = np.array([0, 0])

    cuboid = shapely.geometry.box(
        center_position[0] - axes_length[0],
        center_position[1] - axes_length[1],
        center_position[0] + axes_length[0],
        center_position[1] + axes_length[1],
    )
    # cuboid = shapely.affinity.rotate(cuboid, 50)

    # obstacle = obstacles.ellipse_xd.EllipseWithAxes(
    obstacle = CuboidXd(center_position=center_position, axes_length=axes_length)

    point0 = np.array([3, 2])
    normal0 = obstacle.get_normal_direction(point0, in_obstacle_frame=False)
    surf_point0 = obstacle.get_point_on_surface(point0, in_obstacle_frame=False)

    if visualize:
        import matplotlib.pyplot as plt  # TODO: remove for production

        plt.ion()

        fig, ax = plt.subplots()

        xx, yy = cuboid.exterior.xy
        ax.plot(xx, yy, color="black", alpha=0.3)

        polygon_path = plt.Polygon(np.vstack((xx, yy)).T, alpha=0.1, zorder=-4)
        polygon_path.set_color("black")
        ax.add_patch(polygon_path)

        ax.set_aspect("equal")

        ax.plot(point0[0], point0[1], "ko")

        ax.arrow(surf_point0[0], surf_point0[1], normal0[0], normal0[1], color="red")

        ax.plot(surf_point0[0], surf_point0[1], "ro")

    pass


if (__name__) == "__main__":
    # test_multidimensional_ellipse(True)
    test_multidimensional_cuboid(True)
