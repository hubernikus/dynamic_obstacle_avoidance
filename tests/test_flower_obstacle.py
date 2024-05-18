""" """
# Author: Lukas Huber
# Email: lukas.huber@epfl.ch

from math import pi

import numpy as np

from vartools.angle_math import *



# from dynamic_obstacle_avoidance.utils import *
# from dynamic_obstacle_avoidance.avoidance.obs_common_section import *
# from dynamic_obstacle_avoidance.avoidance.obs_dynamic_center_3d import *

from dynamic_obstacle_avoidance.obstacles import StarshapedFlower


def test_gamma_value(visualize=False):
    center = np.array([2.2, 0.0])
    obstacle = StarshapedFlower(
        center_position=center,
        radius_magnitude=0.2,
        number_of_edges=5,
        radius_mean=0.75,
        orientation=33 / 180 * pi,
        distance_scaling=1,
        # tail_effect=False,
        # is_boundary=True,
    )

    if visualize:
        x_lim = [-1, 5]
        y_lim = [-3, 3]
        n_grid = 100

        nx = ny = n_grid
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx),
            np.linspace(y_lim[0], y_lim[1], ny),
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gammas = np.zeros(positions.shape[1])

        for pp in range(positions.shape[1]):
            gammas[pp] = obstacle.get_gamma(positions[:, pp], in_global_frame=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacles(
            obstacle_container=[obstacle],
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            alpha_obstacle=0.0,
        )

        cs = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas.reshape(nx, ny),
            levels=np.linspace(0, 10.0, 21),
            extend="both",
            zorder=-2,
            # cmap="Blues",
        )
        fig.colorbar(cs)

    # Test gamma a bit away from the center
    gamma_value = obstacle.get_gamma(center + 0.1, in_global_frame=True)
    assert 0 < gamma_value < 1

    # Test at the center
    gamma_value = obstacle.get_gamma(center, in_global_frame=True)
    assert np.isclose(0, gamma_value)


def test_radius_computation():
    center = np.array([2.2, 0.0])
    obstacle = StarshapedFlower(
        center_position=center,
        radius_magnitude=0.2,
        number_of_edges=5,
        radius_mean=0.75,
        orientation=33 / 180 * pi,
        distance_scaling=1,
    )

    # Test gamma a bit away from the center
    radius = obstacle.get_gamma(center, in_global_frame=True)
    assert not np.isnan(radius)


def test_surface_intersection(visualize=False):
    obstacle = StarshapedFlower(
        center_position=np.array([2.2, 0.0]),
        radius_magnitude=0.3,
        number_of_edges=4,
        radius_mean=0.75,
        orientation=33 / 180 * math.pi,
        distance_scaling=1.0,
        # tail_effect=False,
        # is_boundary=True,
    )

    if visualize:
        x_lim = [-1, 5]
        y_lim = [-3, 3]

        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacles(obstacle_container=[obstacle], ax=ax, x_lim=x_lim, y_lim=y_lim)

    position = np.array([1.5, 0.0])
    surf_point = obstacle.get_intersection_with_surface(
        start_position=obstacle.center_position,
        direction=(position - obstacle.center_position),
        in_global_frame=True,
    )
    assert surf_point[0] > 0, "All the obstacle is at x>0"

    dir_pos = position - obstacle.center_position
    dir_pos = dir_pos / np.linalg.norm(dir_pos)
    dir_surf = surf_point - obstacle.center_position
    dir_surf = dir_surf / np.linalg.norm(dir_surf)
    assert np.allclose(dir_pos, dir_surf), " Same direction expected."


def test_normal_direction(visualize=False):
    obstacle = StarshapedFlower(
        center_position=np.array([2.2, 0.0]),
        radius_magnitude=0.3,
        number_of_edges=4,
        radius_mean=0.75,
        orientation=33 / 180 * math.pi,
        distance_scaling=1.0,
        # tail_effect=False,
        # is_boundary=True,
    )

    if visualize:
        x_lim = [-1, 5]
        y_lim = [-3, 3]

        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacles(obstacle_container=[obstacle], ax=ax, x_lim=x_lim, y_lim=y_lim)

    position = np.array([2.70, 1.0])
    normal = obstacle.get_normal_direction(position, in_global_frame=True)
    reference = obstacle.get_reference_direction(position, in_global_frame=True)
    assert np.dot(reference, normal) < 0, " Normal opposing the reference."

    position = np.array([1.3, -2.0])
    normal = obstacle.get_normal_direction(position, in_global_frame=True)
    reference = obstacle.get_reference_direction(position, in_global_frame=True)
    assert np.dot(reference, normal) < 0, " Normal opposing the reference."


def test_gamma_with_scaling(visualize=False):
    np.array([0.0, 0.0])

    obstacle = StarshapedFlower(
        center_position=np.array([2.2, 0.0]),
        radius_magnitude=0.3,
        number_of_edges=4,
        radius_mean=0.75,
        orientation=33 / 180 * math.pi,
        # distance_scaling=0.5,
        distance_scaling=2.0,
        # tail_effect=False,
        # is_boundary=True,
    )

    if visualize:
        x_lim = [-0.1, 3.5]
        y_lim = [-1.5, 1.5]
        figsize = (10, 8)
        n_resolution = 100

        fig, ax = plt.subplots(figsize=figsize)

        # Create points
        nx = ny = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gammas = np.zeros(positions.shape[1])

        for pp in range(positions.shape[1]):
            gammas[pp] = obstacle.get_gamma(positions[:, pp], in_global_frame=True)

        cs = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas.reshape(nx, ny),
            levels=np.linspace(0, 10, 21),
            extend="max",
            zorder=-1,
        )
        fig.colorbar(cs)

        plot_obstacles(
            ax=ax,
            obstacle_container=[obstacle],
            alpha_obstacle=0.0,
            draw_reference=True,
            draw_center=False,
        )

    position = np.array([2.0, 0.18])
    gamma = obstacle.get_gamma(position, in_global_frame=True)
    assert 0 < gamma < 1, "Inside obstacle gamma-value < 1."

    position = np.array([1.7, 0.8])
    gamma = obstacle.get_gamma(position, in_global_frame=True)
    assert 0 < gamma < 1, "Inside obstacle gamma-value < 1."


if (__name__) == "__main__":
    # test_starshape_flower(visualize=True)
    # test_gamma_value(visualize=False)
    # test_radius_computation()

    # test_surface_intersection(visualize=False)
    # test_normal_direction(visualize=True)

    test_gamma_with_scaling(visualize=False)

    print("Tests done.")
