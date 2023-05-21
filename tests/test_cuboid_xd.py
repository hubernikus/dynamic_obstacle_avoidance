"""
Test script for obstacle avoidance algorithm
Test normal formation
"""
import math

import numpy as np
from numpy import linalg as LA

from scipy.spatial.transform import Rotation as Rotation

from vartools.states import Pose
from vartools.math import get_intersection_with_circle, IntersectionType

from dynamic_obstacle_avoidance.obstacles import CuboidXd
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)


def test_gamma_function(n_resolution=10, visualize=False):
    x_lim = [-4, 4]
    y_lim = [-3, 3]

    nx = n_resolution
    ny = n_resolution

    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    normals = np.zeros(positions.shape)

    obstacle = CuboidXd(
        center_position=np.array([0, 0]),
        orientation=20 * np.pi / 180,
        axes_length=np.array([1, 2]),
        margin_absolut=1.0,
    )

    gammas = np.zeros(positions.shape[1])

    for ii in range(positions.shape[1]):
        gammas[ii] = obstacle.get_gamma(
            position=positions[:, ii], in_obstacle_frame=False
        )

        normals[:, ii] = obstacle.get_normal_direction(
            position=positions[:, ii], in_obstacle_frame=False
        )

    if visualize:
        fig, ax = plt.subplots(figsize=(6, 5))

        levels = np.linspace(0, 4, 21)
        contour = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas.reshape(nx, ny),
            levels=levels,
        )

        cbar = fig.colorbar(contour)

        ax.quiver(
            positions[0, :],
            positions[1, :],
            normals[0, :],
            normals[1, :],
            color="black",
        )

        obs_boundary = np.array(obstacle.get_boundary_with_margin_xy())
        ax.plot(obs_boundary[0, :], obs_boundary[1, :], "--", color="k")


def test_cube_intersection():
    """Intersection test in 2D"""
    cube = CuboidXd(center_position=np.array([0, 0]), axes_length=np.array([5.0, 1]))

    # Position 1
    position = np.array([1.0, 0])
    direction = np.array([-1.0, -1])

    intersection = cube.get_intersection_with_surface(
        start_position=position,
        direction=direction,
        in_global_frame=True,
        intersection_type=IntersectionType.CLOSE,
    )
    assert intersection[1] == -0.5

    intersection = cube.get_intersection_with_surface(
        start_position=position,
        direction=direction,
        in_global_frame=True,
        intersection_type=IntersectionType.FAR,
    )
    assert intersection[1] == 0.5

    # Position 2
    position = np.array([0.0, 1.0])
    direction = np.array([-1, -1])
    intersection = cube.get_intersection_with_surface(
        start_position=position,
        direction=direction,
        in_global_frame=True,
        intersection_type=IntersectionType.CLOSE,
    )

    assert intersection[1] == 0.5

    intersection = cube.get_intersection_with_surface(
        start_position=position,
        direction=direction,
        in_global_frame=True,
        intersection_type=IntersectionType.FAR,
    )
    assert intersection[1] == -0.5


def test_cube_outside_position():
    position = np.array([0.55555556, -0.55555556])
    direction = np.array([-0.55555556, -0.94444444])

    cube = CuboidXd(center_position=np.array([0, 0]), axes_length=np.array([1.0, 4.0]))

    intersection = cube.get_intersection_with_surface(
        start_position=position,
        direction=direction,
        in_global_frame=True,
        intersection_type=IntersectionType.CLOSE,
    )

    assert intersection[0] == 0.5
    assert (
        intersection[1] < position[1] and intersection[1] > -0.5 * cube.axes_length[1]
    )

    position = np.array([2.2413793103448283, -5.0])
    direction = np.array([-2.2413793103448283, 6.5])
    intersection = cube.get_intersection_with_surface(
        start_position=position,
        direction=direction,
        in_global_frame=True,
        intersection_type=IntersectionType.CLOSE,
    )
    assert np.isclose(intersection[0], 0.5)

    position = np.array([-0.5172413793103443, -3.9655172413793105])
    direction = np.array([0.5172413793103443, 5.4655172413793105])
    intersection = cube.get_intersection_with_surface(
        start_position=position,
        direction=direction,
        in_global_frame=True,
        intersection_type=IntersectionType.CLOSE,
    )
    assert np.isclose(intersection[1], -2)


def test_normal_direction():
    # Evaluate surface point just on boundary
    cube = CuboidXd(center_position=np.array([0, 0]), axes_length=np.array([1.0, 4.0]))
    position = np.array([-0.5, 1.513157894736842])
    normal = cube.get_normal_direction(position, in_global_frame=True)
    reference = cube.get_reference_direction(position, in_global_frame=True)
    assert np.dot(normal, reference) < 0.0  # Redundant but nice to understand

    # Simple cubic-cube
    cube = CuboidXd(center_position=np.array([0, 0]), axes_length=np.array([1.0, 1.0]))
    position = np.array([2, 0])
    normal = cube.get_normal_direction(position, in_global_frame=True)
    reference = cube.get_reference_direction(position, in_global_frame=True)
    assert np.allclose(normal, [1.0, 0])
    assert np.allclose(reference, [-1.0, 0])
    assert np.dot(normal, reference) < 0.0  # Redundant but nice to understand


def test_surface_position():
    position = np.array([-1.0, 0])

    cube = CuboidXd(
        center_position=np.array([0.0, 0.0]),
        orientation=0.0,
        axes_length=np.array([0.4, 0.7]),
    )
    surf_point = cube.get_point_on_surface(position, in_obstacle_frame=True)
    assert np.allclose(surf_point, [0.2, 0])


def test_normal_direction():
    position = np.array([-4.239800261245738, -2.5311849573737155])
    cube = CuboidXd(
        center_position=np.array([-4.906512403591768, -1.8412917602246444]),
        axes_length=np.array([0.7, 0.6]),
        margin_absolut=0.5,
    )

    normal = cube.get_normal_direction(position, in_global_frame=True)
    assert not np.any(np.isnan(normal))

    position = np.array([-0.6898931971490712, -0.6667121423460305])
    gamma = cube.get_gamma(position, in_global_frame=False)
    assert math.isclose(gamma, 1, abs_tol=1e-3)


def test_simple_cuboid_with_margin():
    cube = CuboidXd(
        center_position=np.array([0.0, 0]),
        axes_length=np.array([2.0, 2.0]),
        margin_absolut=1.0,
    )

    position = np.array([1.9, -1.9])
    normal = cube.get_normal_direction(position, in_global_frame=True)
    assert np.all(np.isclose(normal, np.array([1, -1]) / np.sqrt(2)))


def test_3d_cube_far_away():
    cube = CuboidXd(
        center_position=np.zeros(3),
        axes_length=np.array([0.19] * 3),
        linear_velocity=np.zeros(3),
        margin_absolut=0.12,
        distance_scaling=10.0,
    )

    position = np.array([2.3, -1.8, 2.4])
    normal = cube.get_normal_direction(position)
    gamma = cube.get_gamma(position)

    assert gamma > 10
    assert normal[0] > 0 and normal[1] < 1 and normal[2] > 0


def test_gamma(visualize=False):
    obstacle = CuboidXd(
        pose=Pose(position=np.array([5.0, 2.0]), orientation=-10 / 180.0 * math.pi),
        axes_length=np.array([1.0, 2]),
    )

    if visualize:
        x_lim = [-0.1, 7]
        y_lim = [-0.1, 4]

        fig, ax = plt.subplots(figsize=(5, 4))
        n_resolution = 100

        pos_x_lim = x_lim
        pos_y_lim = y_lim

        nx = ny = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(pos_x_lim[0], pos_x_lim[1], nx),
            np.linspace(pos_y_lim[0], pos_y_lim[1], ny),
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gammas = np.zeros((positions.shape[1]))

        for ii in range(positions.shape[1]):
            gammas[ii] = obstacle.get_gamma(positions[:, ii], in_global_frame=True)

        cont = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas.reshape(nx, ny),
            levels=np.arange(0.0, 10, 1.0),
            zorder=-2,
            alpha=0.4,
        )

    position = np.array([4.95, 2.22])
    gamma = obstacle.get_gamma(position, in_global_frame=True)
    assert 0 < gamma < 1, "Gamma below 1 inside the obstacle"

    position = np.array([2.8, 0.8])
    gamma = obstacle.get_gamma(position, in_global_frame=True)
    assert 1 < gamma < 10, "Unexpected value outside the obstacle"


def test_normal_with_margin(visualize=False):
    obstacle = CuboidXd(
        pose=Pose(np.array([1.55, -1.45]), orientation=1.5707963267948966),
        axes_length=np.array([4.5, 0.4]),
        margin_absolut=0.5,
    )
    obstacle.set_reference_point(np.array([1.55, -3.5]), in_global_frame=True)
    position = np.array([0.8499904639878112, -0.03889134570119339])

    if visualize:
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacles(
            ax=ax,
            obstacle_container=[obstacle],
            x_lim=[-6.5, 6.5],
            y_lim=[-5.5, 5.5],
        )
        # Check all normal directions
        ax.plot(position[0], position[1], "ok")
        normal = obstacle.get_normal_direction(position, in_global_frame=True)
        ax.arrow(position[0], position[1], normal[0], normal[1], color=colors[ii])

    normal = obstacle.get_normal_direction(position, in_global_frame=True)
    assert np.allclose(normal, [-1, 0], atol=1e-3)


if (__name__) == "__main__":
    import matplotlib.pyplot as plt

    test_normal_with_margin(visualize=False)

    test_gamma(visualize=False)
    test_gamma_function(visualize=False, n_resolution=30)

    test_cube_intersection()
    test_cube_outside_position()
    test_normal_direction()
    test_surface_position()
    test_normal_direction()
    test_simple_cuboid_with_margin()
    test_3d_cube_far_away()
