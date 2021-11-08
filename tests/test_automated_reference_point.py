"""
Testing script to ensure that 
"""
import pytest

import matplotlib.pyplot as plt

import numpy as np
from numpy import linalg as LA

from dynamic_obstacle_avoidance.obstacles import Cuboid, Ellipse, Polygon
from dynamic_obstacle_avoidance.containers import GradientContainer, ShapelyContainer

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    plot_obstacles,
)


def test_two_ellipses_nontouching():
    obs_list = ShapelyContainer()

    obs_list.append(
        Ellipse(center_position=np.array([0, 0]), axes_length=np.array([1, 2]))
    )

    obs_list.append(
        Ellipse(center_position=np.array([2, 0]), axes_length=np.array([1.3, 0.8]))
    )

    obs_list.update_reference_points()

    fig = plt.figure(3, dpi=300)
    ax = plt.subplot(1, 2, 1)

    plot_obstacles(ax, obs_list, x_lim=[-2, 6], y_lim=[-1, 4])

    for obs in obs_list:
        ref_point = obs.global_reference_point
        plt.plot(ref_point[0], ref_point[1], "k+")


def test_automated_reference_point(visualize=False):
    x_range = [-1, 11]
    y_range = [-1, 11]

    obs_list = GradientContainer()

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
            print(f"obs{ii} axes lengths ", obs_list[ii].axes_length)

    if True:
        # Stop here temporarily
        return 0

    # First ellipse is connected to the wall
    assert (
        obs_list[0].get_gamma(obs_list[1].reference_point, in_global_frame=True) < 1
    ), print("Reference point ought be inside the wall.")

    ii = 1
    assert LA.norm(obs_list[ii].local_reference_point) < np.max(
        obs_list[ii].axes_length
    ), f"Reference point is expected close to the ellipse {ii}."

    i = 2
    assert LA.norm(obs_list[ii].local_reference_point) < np.max(
        obs_list[ii].axes_length
    ), print(f"Reference point is expected close to the ellipse {ii}.")


if (__name__) == "__main__":
    test_two_ellipses_nontouching()
    # test_automated_reference_point(visualize=True)
    pass
