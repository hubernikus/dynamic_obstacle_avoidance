"""
Test script for obstacle avoidance algorithm
Test normal formation
"""
import unittest

import numpy as np
from math import pi

from dynamic_obstacle_avoidance.containers import GradientContainer, ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import Ellipse, CircularObstacle
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)

from scipy.spatial.transform import Rotation as Rotation


def test_ellipse_reference_point_inside(visualize=False):
    """Visualize ellipse with reference point far away"""
    # Create Ellipse
    obs = Ellipse(
        axes_length=[1.2, 2.0],
        center_position=[0.0, 0.0],
        orientation=30.0 / 180 * pi,
    )

    # Reset reference point
    obs.set_reference_point(np.array([1, 0.3]), in_global_frame=True)

    assert obs.get_gamma(obs.reference_point) < 1

    obs_list = GradientContainer()
    obs_list.append(obs)

    if visualize:
        Simulation_vectorFields(
            x_range=[-3, 3],
            y_range=[-3, 3],
            point_grid=100,
            obs=obs_list,
            draw_vectorField=False,
            automatic_reference_point=False,
        )


def test_visualization_ellipse_with_ref_point_outside(visualize=False):
    """Visualize ellipse with reference point far away"""
    # Create Ellipse
    obs = Ellipse(
        axes_length=[1.2, 2.0],
        center_position=[0.0, 0.0],
        orientation=30.0 / 180 * pi,
    )

    # Set reference point outside
    obs.set_reference_point(np.array([2, 1]), in_global_frame=True)

    obs_list = GradientContainer()
    obs_list.append(obs)

    if visualize:
        Simulation_vectorFields(
            x_range=[-3, 3],
            y_range=[-3, 3],
            point_grid=0,
            obs=obs_list,
            draw_vectorField=False,
            automatic_reference_point=False,
        )


def test_visualization_circular_reference_point_outside(visualize=False):
    """Visualize circular-obstacle with reference point far away"""
    obs = CircularObstacle(
        radius=1.5,
        center_position=[0.0, 0.0],
        orientation=0.0 / 180 * pi,
    )

    obs.set_reference_point(np.array([1.2, 1.9]), in_global_frame=True)

    obs_list = GradientContainer()
    obs_list.append(obs)

    if visualize:
        Simulation_vectorFields(
            x_range=[-3, 3],
            y_range=[-3, 3],
            point_grid=0,
            obs=obs_list,
            draw_vectorField=False,
            automatic_reference_point=False,
        )


def test_normal_directions(x_lim=[-5, 5], y_lim=[-5, 5]):
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Ellipse(
            axes_length=[1.0, 2.0],
            center_position=np.array([2.0, 0.0]),
            orientation=(0 * pi / 180),
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    import matplotlib.pyplot as plt

    plt.ion()
    plt.show()

    fig, ax = plt.subplots()

    plot_obstacles(
        ax=ax, obstacle_container=obstacle_environment, x_lim=x_lim, y_lim=y_lim
    )

    nx = ny = 30
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    norm_dirs = np.zeros(positions.shape)
    ref_dirs = np.zeros(positions.shape)

    for it in range(positions.shape[1]):
        norm_dirs[:, it] = obstacle_environment[0].get_normal_direction(
            positions[:, it], in_global_frame=True
        )

        ref_dirs[:, it] = obstacle_environment[0].get_reference_direction(
            positions[:, it], in_global_frame=True
        )

        assert (
            np.dot(norm_dirs[:, it], ref_dirs[:, it]) < 0
        ), "Not pointing in the same direction."

    ref_dirs = ref_dirs * (-1)

    ax.quiver(
        positions[0, :],
        positions[1, :],
        ref_dirs[0, :],
        ref_dirs[1, :],
        color="red",
        scale=30,
        alpha=0.8,
    )

    ax.quiver(
        positions[0, :],
        positions[1, :],
        norm_dirs[0, :],
        norm_dirs[1, :],
        color="blue",
        scale=30,
        alpha=0.8,
    )


if (__name__) == "__main__":
    test_normal_directions()
