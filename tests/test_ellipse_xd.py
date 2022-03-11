"""
Test script for obstacle avoidance algorithm
Test normal formation
"""
import unittest

import numpy as np
from math import pi

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)

from scipy.spatial.transform import Rotation as Rotation


def test_gamma_function(n_resolution=10, visualize=False):
    x_lim = [-4.5, 4.5]
    y_lim = [-3.5, 3.5]

    nx = n_resolution
    ny = n_resolution

    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    obstacle = EllipseWithAxes(
        center_position=np.array([0, 0]), axes_length=np.array([1, 1])
    )

    gammas = np.zeros(positions.shape[1])

    for ii in range(positions.shape[1]):
        gammas[ii] = obstacle.get_gamma(
            position=positions[:, ii], in_obstacle_frame=True
        )

    if visualize:
        fig, ax = plt.subplots(figsize=(6, 5))

        levels = np.linspace(0, 5, 10)
        ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas.reshape(nx, ny),
            levels=levels,
        )

        obs_boundary = np.array(obstacle.get_boundary_with_margin_xy())
        ax.plot(obs_boundary[0, :], obs_boundary[1, :], "--", color="k")

        ax.axis("equal")


def test_surface_point(visualize=False):
    obstacle = EllipseWithAxes(
        center_position=np.array([0, 0]), axes_length=np.array([2, 2])
    )

    position = np.array([2, 2])
    surf_point = obstacle.get_point_on_surface(position, in_obstacle_frame=True)
    assert np.allclose(surf_point, position / LA.norm(position))

    position = np.array([-3, 2])
    surf_point = obstacle.get_point_on_surface(position, in_obstacle_frame=True)
    assert np.allclose(surf_point, position / LA.norm(position))


def test_normal_directions():
    pass


if (__name__) == "__main__":
    test_surface_point()

    test_gamma_function(visualize=True, n_resolution=100)
    # test_normal_directions()
