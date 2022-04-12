#!/USSR/bin/python3
"""
Testing script to ensure that 
"""
from math import pi

import pytest

import matplotlib.pyplot as plt

import numpy as np
from numpy import linalg as LA

from dynamic_obstacle_avoidance.obstacles import Cuboid, Ellipse, Polygon
from dynamic_obstacle_avoidance.containers import ShapelyContainer

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    plot_obstacles,
)


def test_two_ellipses_nontouching(visualize=False):
    obs_list = ShapelyContainer()
    obs_list.append(
        Ellipse(center_position=np.array([0, 0]), axes_length=np.array([1, 2]))
    )
    obs_list.append(
        Ellipse(center_position=np.array([3, 0]), axes_length=np.array([1.3, 0.8]))
    )
    obs_list.update_reference_points()

    if visualize:
        fig, ax = plt.subplots()

        x_lim = [-2, 6]
        y_lim = [-3, 3]

        plot_obstacles(ax, obs_list, x_lim, y_lim)

        for obs in obs_list:
            ref_point = obs.global_reference_point
            plt.plot(ref_point[0], ref_point[1], "k+")

    # Test gamma for own obstacle
    assert (
        obs_list[0].get_gamma(obs_list[0].global_reference_point, in_global_frame=True)
        < 1
    ), "Reference point is outside"

    assert (
        obs_list[1].get_gamma(obs_list[1].global_reference_point, in_global_frame=True)
        < 1
    ), "Reference point is outside"


def test_two_ellipses_touching(visualize=False):
    obs_list = ShapelyContainer()

    obs_list.append(
        Ellipse(center_position=np.array([0, 0]), axes_length=np.array([1, 2]))
    )
    obs_list.append(
        Ellipse(center_position=np.array([2, 0]), axes_length=np.array([1.3, 0.8]))
    )

    obs_list.update_reference_points()

    if visualize:
        fig, ax = plt.subplots()
        x_lim = [-2, 6]
        y_lim = [-3, 3]
        plot_obstacles(ax, obs_list, x_lim, y_lim)

        for obs in obs_list:
            ref_point = obs.global_reference_point
            plt.plot(ref_point[0], ref_point[1], "k+")

    # Test gamma for own obstacle
    assert (
        obs_list[0].get_gamma(obs_list[1].global_reference_point, in_global_frame=True)
        < 1
    ), "Reference point is outside"

    assert (
        obs_list[0].get_gamma(obs_list[1].global_reference_point, in_global_frame=True)
        < 1
    ), "Reference point is outside"


def test_ellipse_wall_inside(visualize=False):
    obs_list = ShapelyContainer()

    obs_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([3, 4]),
            is_boundary=True,
        )
    )
    obs_list.append(
        Ellipse(center_position=np.array([1, 0]), axes_length=np.array([1.3, 0.8]))
    )
    obs_list.update_reference_points()

    if visualize:
        fig, ax = plt.subplots()
        x_lim = [-6, 6]
        y_lim = [-6, 6]
        plot_obstacles(ax, obs_list, x_lim, y_lim)

        for obs in obs_list:
            ref_point = obs.global_reference_point
            plt.plot(ref_point[0], ref_point[1], "k+")

    # 0 => boundary | 1 => obstacle
    assert (
        obs_list[0].get_gamma(obs_list[1].global_reference_point, in_global_frame=True)
        > 1
    ), "Reference point is outside"

    assert (
        obs_list[1].get_gamma(obs_list[1].global_reference_point, in_global_frame=True)
        < 1
    ), "Reference point is outside"


def test_ellipse_wall_intersection(visualize=False):
    obs_list = ShapelyContainer()

    obs_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([4, 5]),
            is_boundary=True,
        )
    )
    obs_list.append(
        Ellipse(
            center_position=np.array([4, 0]),
            axes_length=np.array([1.3, 0.8]),
            orientation=30 * pi / 180,
        )
    )
    obs_list.update_reference_points()

    if visualize:
        fig, ax = plt.subplots()
        x_lim = [-6, 6]
        y_lim = [-6, 6]
        plot_obstacles(ax, obs_list, x_lim, y_lim)

        for obs in obs_list:
            if obs.is_boundary:
                continue

            ref_point = obs.global_reference_point
            plt.plot(ref_point[0], ref_point[1], "k+")

    # 0 => boundary | 1 => obstacle
    assert (
        obs_list[0].get_gamma(obs_list[1].global_reference_point, in_global_frame=True)
        < 1
    ), "Reference point is outside"

    assert (
        obs_list[1].get_gamma(obs_list[1].global_reference_point, in_global_frame=True)
        < 1
    ), "Reference point is outside"


def test_polygons_close(visualize=False):
    obs_list = ShapelyContainer()

    obs_list.append(
        Cuboid(
            name="CuboidLeft",
            center_position=np.array([0, 0]),
            axes_length=np.array([3, 4]),
        ),
    )
    print("Done")

    obs_list.append(
        Cuboid(
            name="CuboidRight",
            center_position=np.array([4, 0]),
            axes_length=np.array([1.3, 2]),
            orientation=30 * pi / 180,
        )
    )
    print("Done2")
    obs_list.update_reference_points()

    if visualize:
        fig, ax = plt.subplots()
        x_lim = [-6, 6]
        y_lim = [-6, 6]
        plot_obstacles(ax, obs_list, x_lim, y_lim)

        for obs in obs_list:
            ref_point = obs.global_reference_point
            plt.plot(ref_point[0], ref_point[1], "k+")


def test_automated_reference_point(visualize=False):
    x_range = [-1, 11]
    y_range = [-1, 11]

    obs_list = ShapelyContainer()

    obs_xaxis = [x_range[0] + 1, x_range[1] - 1]
    obs_yaxis = [y_range[0] + 1, y_range[1] - 1]
    edge_points = np.array(
        [
            [obs_xaxis[0], obs_xaxis[0], obs_xaxis[1], obs_xaxis[1]],
            [obs_yaxis[1], obs_yaxis[0], obs_yaxis[0], obs_yaxis[1]],
        ]
    )
    obs_list.append(
        Polygon(
            edge_points=edge_points,
            is_boundary=True,
            tail_effect=False,
        )
    )

    attractor_position = np.array([7.5, 1.7])

    obs_list.append(
        Cuboid(
            axes_length=[12, 2.5],
            center_position=[11, 5],
            tail_effect=False,
        )
    )

    # Ellipse intersecting with wall
    obs_list.append(
        Ellipse(
            center_position=np.array([10.87610974, 3.0902647]),
            orientation=np.array([0.1133965]),
            axes_length=np.array([1.53458869, 0.90004227]),
        )
    )

    # Free standing ellipse
    obs_list.append(
        Ellipse(
            center_position=np.array([3.11145135, 2.80324546]),
            orientation=np.array([-0.19044557]),
            axes_length=np.array([0.74123339, 1.34090768]),
        )
    )

    obs_list.update_reference_points()

    if visualize:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)

        plot_obstacles(ax, obs_list, x_range, y_range, showLabel=False)

        for ii in [2, 3]:
            print(f"obs{ii} axes lengths {obs_list[ii].axes_length}")

    if True:
        # Stop here temporarily
        return 0

    # First ellipse is connected to the wall
    assert (
        obs_list[0].get_gamma(obs_list[1].reference_point, in_global_frame=True) < 1
    ), f"Reference point ought be inside the wall."

    ii = 1
    assert LA.norm(obs_list[ii].local_reference_point) < np.max(
        obs_list[ii].axes_length
    ), f"Reference point is expected close to the ellipse {ii}."

    i = 2
    assert LA.norm(obs_list[ii].local_reference_point) < np.max(
        obs_list[ii].axes_length
    ), f"Reference point is expected close to the ellipse {ii}."


def three_ellipses_intersections(visualize=True):
    obs_list = ShapelyContainer()
    obs_list.append(
        Ellipse(center_position=np.array([0, 0]), axes_length=np.array([1, 2]))
    )
    obs_list.append(
        Ellipse(center_position=np.array([3, 0]), axes_length=np.array([1.3, 0.8]))
    )
    obs_list.append(
        Ellipse(center_position=np.array([2, -0.3]), axes_length=np.array([1.3, 0.8]))
    )

    obs_list.update_reference_points()

    if visualize:
        fig, ax = plt.subplots()

        x_lim = [-2, 6]
        y_lim = [-3, 3]

        plot_obstacles(ax, obs_list, x_lim, y_lim)

        for obs in obs_list:
            ref_point = obs.global_reference_point
            plt.plot(ref_point[0], ref_point[1], "k+")


def ellipses_and_wall_intersection(visualize=True):
    obs_list = ShapelyContainer()
    obs_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([3, 2]),
            is_boundary=True,
        )
    )
    obs_list.append(
        Ellipse(center_position=np.array([3, 0]), axes_length=np.array([1.3, 0.8]))
    )

    obs_list.update_reference_points()

    assert (
        obs_list[0].get_gamma(obs_list[1].global_reference_point, in_global_frame=True)
        < 1
    ), f"Reference point ought to be inside the wall."

    assert (
        obs_list[1].get_gamma(obs_list[1].global_reference_point, in_global_frame=True)
        < 1
    ), f"Reference point to be inside the obstacle."

    if visualize:
        fig, ax = plt.subplots()

        x_lim = [-2, 6]
        y_lim = [-3, 3]

        plot_obstacles(ax, obs_list, x_lim, y_lim)

        for obs in obs_list:
            ref_point = obs.global_reference_point
            plt.plot(ref_point[0], ref_point[1], "k+")


def two_ellipses_and_wall_intersection(visualize=True):
    obs_list = ShapelyContainer()
    obs_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([3, 5]),
            is_boundary=True,
        )
    )
    obs_list.append(
        Ellipse(center_position=np.array([3, 0]), axes_length=np.array([1.3, 0.8]))
    )

    obs_list.append(
        Ellipse(center_position=np.array([2, 0]), axes_length=np.array([0.8, 1.8]))
    )

    obs_list.update_reference_points()

    if visualize:
        fig, ax = plt.subplots()

        x_lim = [-2, 6]
        y_lim = [-3, 3]

        plot_obstacles(ax, obs_list, x_lim, y_lim)

        for obs in obs_list:
            ref_point = obs.global_reference_point
            plt.plot(ref_point[0], ref_point[1], "k+")

    assert (
        obs_list[0].get_gamma(obs_list[1].global_reference_point, in_global_frame=True)
        < 1
    ), f"Reference point ought to be inside the wall."

    assert (
        obs_list[0].get_gamma(obs_list[2].global_reference_point, in_global_frame=True)
        < 1
    ), f"Reference point ought to be inside the wall."


def two_ellipses_and_wall_partial_intersection(visualize=True):
    obs_list = ShapelyContainer()
    obs_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([3, 5]),
            is_boundary=True,
        )
    )
    obs_list.append(
        Ellipse(center_position=np.array([3, 0]), axes_length=np.array([1.3, 0.8]))
    )

    obs_list.append(
        Ellipse(center_position=np.array([-1, 0]), axes_length=np.array([0.8, 1.8]))
    )

    obs_list.update_reference_points()

    if visualize:
        fig, ax = plt.subplots()

        x_lim = [-2, 6]
        y_lim = [-3, 3]

        plot_obstacles(ax, obs_list, x_lim, y_lim)

        for obs in obs_list:
            ref_point = obs.global_reference_point
            plt.plot(ref_point[0], ref_point[1], "k+")

    assert (
        obs_list[0].get_gamma(obs_list[1].global_reference_point, in_global_frame=True)
        < 1
    ), f"Reference point ought to be inside the wall."

    assert (
        obs_list[0].get_gamma(obs_list[2].global_reference_point, in_global_frame=True)
        > 1
    ), f"Reference point ought to be inside the wall."


if (__name__) == "__main__":
    # test_two_ellipses_nontouching(visualize=True)
    # test_two_ellipses_touching(visualize=True)

    # test_ellipse_wall_inside(visualize=True)
    # test_ellipse_wall_intersection(visualize=True)

    # test_polygons_close(visualize=True)
    # test_automated_reference_point(visualize=True)

    # three_ellipses_intersections(visualize=True)

    # ellipses_and_wall_intersection(visualize=True)
    # two_ellipses_and_wall_intersection(visualize=True)

    two_ellipses_and_wall_partial_intersection(visualize=True)

    pass
