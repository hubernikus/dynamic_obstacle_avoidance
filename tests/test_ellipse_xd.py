"""
Test script for obstacle avoidance algorithm
Test normal formation
"""
# Author: Lukas Huber
# Created: 2021-03-11
# Email: lukas.huber@epfl.ch

import unittest

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)

from scipy.spatial.transform import Rotation as Rotation


def test_gamma_and_normal(n_resolution=10, visualize=False):
    x_lim = [-4.5, 4.5]
    y_lim = [-3.5, 3.5]

    nx = n_resolution
    ny = n_resolution

    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    normals = np.zeros(positions.shape)
    # reference_dirs = np.zeros(positions.shape)

    obstacle = EllipseWithAxes(
        center_position=np.array([0, 0]),
        orientation=30 * np.pi / 180,
        axes_length=np.array([2, 4]),
    )

    gammas = np.zeros(positions.shape[1])

    for ii in range(positions.shape[1]):
        gammas[ii] = obstacle.get_gamma(
            position=positions[:, ii], in_obstacle_frame=False
        )

        normals[:, ii] = obstacle.get_normal_direction(
            position=positions[:, ii], in_obstacle_frame=False
        )

        # reference_dirs[:, ii] = obstacle.get_reference_direction(
        #     position=positions[:, ii], in_obstacle_frame=True
        #     )

        surf_point = obstacle.get_point_on_surface(
            positions[:, ii], in_obstacle_frame=True
        )

    if visualize:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.axis("equal")

        levels = np.linspace(0, 5, 10)
        ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas.reshape(nx, ny),
            levels=levels,
        )

        ax.quiver(
            positions[0, :],
            positions[1, :],
            normals[0, :],
            normals[1, :],
            color="black",
        )

        obs_boundary = np.array(obstacle.get_boundary_with_margin_xy())
        ax.plot(obs_boundary[0, :], obs_boundary[1, :], "--", color="k")


def test_gamma_for_circular_ellipse(visualize=False):
    """Two opposing points should have the same gamma."""
    obstacle = EllipseWithAxes(
        center_position=np.array([0, 0]), axes_length=np.array([2, 2])
    )
    position = np.array([1, 1])
    gamma1 = obstacle.get_gamma(position, in_obstacle_frame=True)

    position = np.array([-1, -1])
    gamma2 = obstacle.get_gamma(position, in_obstacle_frame=True)

    assert np.isclose(gamma1, gamma2)


def test_surface_point_for_equal_axes(visualize=False):
    """Surface points should be projected onto sphere with radius=1."""
    obstacle = EllipseWithAxes(
        center_position=np.array([0, 0]), axes_length=np.array([2, 2])
    )

    position = np.array([1, 1])
    surf_point = obstacle.get_point_on_surface(position, in_obstacle_frame=True)
    assert np.allclose(surf_point, position / LA.norm(position))

    position = np.array([-3, 2])
    surf_point = obstacle.get_point_on_surface(position, in_obstacle_frame=True)
    assert np.allclose(surf_point, position / LA.norm(position))

    position = np.array([-1, -1])
    surf_point = obstacle.get_point_on_surface(position, in_obstacle_frame=True)
    assert np.allclose(surf_point, position / LA.norm(position))


def test_normal_and_reference_directions(visualize=False):
    x_lim = [-5, 5]
    y_lim = [-5, 5]

    n_resolution = 10
    nx = n_resolution
    ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    obstacle = EllipseWithAxes(
        center_position=np.array([0, 0]),
        axes_length=np.array([5, 8]),
    )
    
    if visualize:
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        normals = np.zeros(positions.shape)
        references = np.zeros(positions.shape)

        for it in range(positions.shape[1]):
            if not LA.norm(positions[:, it] - obstacle.center_position):
                # Not defined at the center
                continue

            references[:, it] = obstacle.get_normal_direction(
                positions[:, it], in_global_frame=True
            )

            normals[:, it] = obstacle.get_reference_direction(
                positions[:, it], in_global_frame=True
            )

        fig, ax = plt.subplots(figsize=(6, 5))

        ax.quiver(
            positions[0, :],
            positions[1, :],
            normals[0, :],
            normals[1, :],
            color="red",
        )

        ax.quiver(
            positions[0, :],
            positions[1, :],
            references[0, :],
            references[1, :],
            color="blue",
        )

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.axis("equal")

        obs_boundary = np.array(obstacle.get_boundary_with_margin_xy())
        ax.plot(obs_boundary[0, :], obs_boundary[1, :], "--", color="k")

    position = np.array([-10, 10])
    reference = obstacle.get_normal_direction(position, in_global_frame=True)
    normal = obstacle.get_reference_direction(position, in_global_frame=True)
    assert reference.dot(normal) < 0, "Print reference and normal are not opposing."

    position = np.array([10, 10])
    reference = obstacle.get_normal_direction(position, in_global_frame=True)
    normal = obstacle.get_reference_direction(position, in_global_frame=True)
    assert reference.dot(normal) < 0, "Print reference and normal are not opposing."


if (__name__) == "__main__":
    # test_surface_point_for_equal_axes()
    # test_gamma_for_circular_ellipse()

    # test_gamma_and_normal(visualize=True, n_resolution=20)
    # test_normal_and_reference_directions(visualize=False)

    # test_normal_directions()

    print("Tests done.")
