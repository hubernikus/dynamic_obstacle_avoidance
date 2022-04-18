""" Test Polygon. """
# Author: Lukas Huber
# Created: 2021-11-09
# License: BSD (c) 2021
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import Cuboid, Polygon
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)

from vartools.dynamical_systems import plot_dynamical_system_streamplot
from vartools.dynamical_systems import plot_dynamical_system_quiver


def is_vector_in_array(vector: np.array, array: np.array) -> bool:
    return np.any(
        np.isclose(LA.norm(array - np.tile(vector, (array.shape[1], 1)).T, axis=0), 0)
    )


def test_triangle_normals(visualize=False, x_lim=[-1, 2], y_lim=[-1, 2]):
    edge_points = [
        [0, 0],
        [1, 0],
        [0, 1],
    ]

    my_obstacle = Polygon(edge_points=np.array(edge_points).T)

    if visualize:
        fig, ax = plt.subplots()

        my_obstacle.plot2D(ax)
        ax.set_aspect("equal")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    pos = np.array([1, 1])

    edge_points = my_obstacle.shapely.get_local_edge_points()

    # Check if the three normals are correct
    tangents, normals = my_obstacle.get_tangents_and_normals_of_edge(edge_points)
    dists = my_obstacle.get_normal_distance_to_surface_pannels(
        pos, edge_points, normals
    )

    assert is_vector_in_array(vector=np.array([0, -1]), array=normals)
    assert is_vector_in_array(vector=np.array([-1, 0]), array=normals)
    assert is_vector_in_array(vector=np.array([1, 1]) / np.sqrt(2), array=normals)

    # Summed normal for position
    normal = my_obstacle.get_normal_direction(pos, in_global_frame=True)

    if visualize:
        pos_array = np.tile(pos, (dists.shape[0], 1)).T
        ax.quiver(
            pos_array[0, :], pos_array[1, :], normals[0, :], normals[1, :], color="blue"
        )

        ax.arrow(pos[0], pos[1], normal[0], normal[1], color="green")


def test_simple_cube_creation_and_ref_point_displacement(visualize=False):
    my_obstacle = Cuboid(
        center_position=np.array([1, 2]),
        orientation=40 * pi / 180,
        axes_length=np.array([1, 2]),
    )

    if visualize:
        fig, axs = plt.subplots(1, 2)
        x_lim = [-2, 6]
        y_lim = [-2, 4]

        my_obstacle.plot2D(axs[0])
        axs[0].set_aspect("equal")

    my_obstacle.set_reference_point(np.array([2, 3]), in_global_frame=True)

    if visualize:
        x_lim = [-2, 6]
        y_lim = [-2, 4]
        my_obstacle.plot2D(axs[1])
        axs[1].set_aspect("equal")


def test_normal_of_cube_sideways(visualize=False, x_lim=[-4, 4], y_lim=[-4, 4]):
    my_obstacle = Cuboid(
        center_position=np.array([0, 0]),
        orientation=0,
        axes_length=np.array([1, 1]),
    )

    if visualize:
        fig, ax = plt.subplots()

        my_obstacle.plot2D(ax)
        ax.set_aspect("equal")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    # Local positions
    pos = np.array([2, 0])
    edge_points = my_obstacle.shapely.get_local_edge_points()

    tangents, normals = my_obstacle.get_tangents_and_normals_of_edge(edge_points)
    dists = my_obstacle.get_normal_distance_to_surface_pannels(
        pos, edge_points, normals
    )

    for ii, dist in enumerate(dists):
        if np.allclose(normals[:, ii], [1, 0]):
            assert dist > 0
        else:
            assert np.isclose(dist, 0)

    normal = my_obstacle.get_normal_direction(pos, in_global_frame=True)
    assert np.allclose(normal, [1, 0])

    if visualize:
        pos_array = np.tile(pos, (dists.shape[0], 1)).T
        ax.quiver(
            pos_array[0, :], pos_array[1, :], normals[0, :], normals[1, :], color="blue"
        )

        ax.arrow(pos[0], pos[1], normal[0], normal[1], color="green")


def test_normal_of_cube_up(visualize=False, x_lim=[-4, 4], y_lim=[-4, 4]):
    my_obstacle = Cuboid(
        center_position=np.array([0, 0]),
        orientation=0,
        axes_length=np.array([1, 1]),
    )

    if visualize:
        fig, ax = plt.subplots()

        my_obstacle.plot2D(ax)
        ax.set_aspect("equal")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    # What about around the edges
    pos = np.array([2, 2])

    edge_points = my_obstacle.shapely.get_local_edge_points()

    tangents, normals = my_obstacle.get_tangents_and_normals_of_edge(edge_points)
    dists = my_obstacle.get_normal_distance_to_surface_pannels(
        pos, edge_points, normals
    )

    for ii, dist in enumerate(dists):
        if np.allclose(normals[:, ii], [1, 0]) or np.allclose(normals[:, ii], [0, 1]):
            assert dist > 0
        else:
            assert np.isclose(dist, 0)

    normal = my_obstacle.get_normal_direction(pos, in_global_frame=True)

    true_norm = np.array([1, 1])
    true_norm = true_norm / LA.norm(true_norm)
    assert np.allclose(normal, true_norm), f"Normal is equal to: {normal}"

    if visualize:
        pos_array = np.tile(pos, (dists.shape[0], 1)).T
        ax.quiver(
            pos_array[0, :], pos_array[1, :], normals[0, :], normals[1, :], color="blue"
        )

        ax.arrow(pos[0], pos[1], normal[0], normal[1], color="green")


def test_normal_of_cube_outside_reference_on_top(
    visualize=False, x_lim=[-4, 4], y_lim=[-4, 4]
):
    my_obstacle = Cuboid(
        center_position=np.array([0, 0]),
        orientation=0,
        axes_length=np.array([1, 1]),
    )

    my_obstacle.set_reference_point(np.array([0, 2]))

    if visualize:
        fig, ax = plt.subplots()

        my_obstacle.plot2D(ax)
        ax.set_aspect("equal")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    # What about around the edges
    pos = np.array([0, 3])

    edge_points = my_obstacle.shapely.get_local_edge_points()
    normal = my_obstacle.get_normal_direction(pos, in_global_frame=True)

    if visualize:
        ax.arrow(pos[0], pos[1], normal[0], normal[1], color="green")

    assert np.allclose(normal, np.array([0, 1])), f"Normal is equal to: {normal}"


def test_normal_vectors_cuboid(visualize=False, x_lim=[-4, 4], y_lim=[-4, 4]):
    my_obstacle = Cuboid(
        center_position=np.array([1, 2]),
        orientation=40 * pi / 180,
        axes_length=np.array([1, 2]),
    )

    position = np.array([2, 3])
    normal = my_obstacle.get_normal_direction(position, in_global_frame=True)
    ref = my_obstacle.get_outwards_reference_direction(position, in_global_frame=True)

    assert np.dot(normal, ref) > 0, f"Normal is not pointing outwards at {position}."

    if visualize:
        fig, ax = plt.subplots()

        my_obstacle.plot2D(ax)
        ax.set_aspect("equal")

        n_grid = 10

        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_grid),
            np.linspace(y_lim[0], y_lim[1], n_grid),
        )

        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        normals = np.zeros(positions.shape)
        refs = np.zeros(positions.shape)

        for ii in range(positions.shape[1]):
            if my_obstacle.get_gamma(positions[:, ii], in_global_frame=True) < 1:
                continue

            normals[:, ii] = my_obstacle.get_normal_direction(
                positions[:, ii], in_global_frame=True
            )
            refs[:, ii] = my_obstacle.get_outwards_reference_direction(
                positions[:, ii], in_global_frame=True
            )
            assert (
                np.dot(normals[:, ii], refs[:, ii]) > 0
            ), f"Normal is not pointing outwards at {positions[:, ii]}."

        ax.quiver(
            positions[0, :], positions[1, :], normals[0, :], normals[1, :], color="blue"
        )


def test_normal_vectors_cuboid_outside_ref(
    visualize=False, x_lim=[-4, 4], y_lim=[-2, 6]
):
    my_obstacle = Cuboid(
        center_position=np.array([1, 2]),
        orientation=40 * pi / 180,
        axes_length=np.array([1, 2]),
    )

    my_obstacle.set_reference_point(np.array([3, 2]), in_global_frame=False)

    position = np.array([3, -4])
    normal = my_obstacle.get_normal_direction(position, in_global_frame=True)
    ref = my_obstacle.get_outwards_reference_direction(position, in_global_frame=True)

    assert np.dot(normal, ref) > 0, f"Normal is not pointing outwards at {position}."

    if visualize:
        n_grid = 10
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_grid),
            np.linspace(y_lim[0], y_lim[1], n_grid),
        )

        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        normals = np.zeros(positions.shape)
        refs = np.zeros(positions.shape)

        for ii in range(positions.shape[1]):
            if my_obstacle.get_gamma(positions[:, ii], in_global_frame=True) < 1:
                continue

            normals[:, ii] = my_obstacle.get_normal_direction(
                positions[:, ii], in_global_frame=True
            )
            refs[:, ii] = my_obstacle.get_outwards_reference_direction(
                positions[:, ii], in_global_frame=True
            )

            assert (
                np.dot(normals[:, ii], refs[:, ii]) > 0
            ), f"Normal is not pointing outwards at {positions[:, ii]}."

        fig, ax = plt.subplots()

        my_obstacle.plot2D(ax)
        ax.set_aspect("equal")

        ax.quiver(
            positions[0, :], positions[1, :], normals[0, :], normals[1, :], color="blue"
        )


def test_gamma_of_cube():
    pass


if (__name__) == "__main__":
    # test_triangle_normals(visualize=False)
    # test_simple_cube_creation_and_ref_point_displacement(visualize=False)
    # test_normal_of_cube_sideways(visualize=False)
    # test_normal_of_cube_up(visualize=False)
    # test_normal_of_cube_outside_reference_on_top(visualize=True)

    # test_normal_vectors_cuboid(visualize=True)
    # test_normal_vectors_cuboid_outside_ref(visualize=True)

    print("Selected tests are done.")
    pass
