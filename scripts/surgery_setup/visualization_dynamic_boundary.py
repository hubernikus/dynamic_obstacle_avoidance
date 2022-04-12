"""
VISUAL TESTING

Test script for obstacle avoidance algorithm
Test normal formation
"""

# TODO: TEST on: moving general creation, moving, gamma values, obstacle container

import numpy as np
import matplotlib.pyplot as plt

plt.ion()
plt.close("all")

from dynamic_obstacle_avoidance.obstacle_avoidance.dynamic_boundaries_polygon import (
    DynamicBoundariesPolygon,
)


def test_obstacle_list_creation():
    """Create obstacle list"""
    obs = GradientContainer()  # create empty obstacle list

    pass


def visualization_gamma(z_val=None):
    """Ensure that with increasing distance gamma increases (or equal)."""

    obs = DynamicBoundariesPolygon(is_surgery_setup=True)

    d_dist = 0.01
    x_range = [obs.x_min - d_dist, obs.x_max + d_dist]
    y_range = [obs.y_min - d_dist, obs.y_max + d_dist]
    # z_range = 0

    # n_grid = 10
    n_grid = 50
    x_vals = np.linspace(x_range[0], x_range[1], n_grid)
    y_vals = np.linspace(y_range[0], y_range[1], n_grid)

    if z_val is None:
        z_val = np.random.rand() * (obs.z_max - obs.z_min) + obs.z_min

    positions = np.zeros((obs.dim, n_grid, n_grid))
    gammas = np.zeros((n_grid, n_grid))

    # position = np.array([-0.1, -0.1, 0])
    # surf_point = obs.line_search_surface_point(position)

    obs.set_inflation_parameter(np.random.rand(4))

    obs.draw_obstacle(num_points=40, z_val=z_val, in_global_frame=False)

    fig = plt.figure(figsize=(10, 8))
    plt.title("Gamma value with z={}".format(np.round(z_val, 4)))
    plt.plot(0, 0, "k+", linewidth=18, markeredgewidth=4, markersize=13)
    plt.plot(
        obs.boundary_points_local[0, :],
        obs.boundary_points_local[1, :],
        "k--",
        linewidth=2,
        marker="+",
    )

    print("Evaluation height (global) z={}".format(z_val))
    print("Inflation param", obs.inflation_percentage)
    for ix in range(n_grid):
        for iy in range(n_grid):
            positions[:, ix, iy] = [x_vals[ix], y_vals[iy], z_val]

            gammas[ix, iy] = obs.get_gamma(positions[:, ix, iy], in_global_frame=True)

    cs = plt.contourf(
        positions[0, :, :],
        positions[1, :, :],
        gammas,
        np.arange(0, 10.0, 1.0),
        extend="max",
        alpha=0.6,
        zorder=-3,
    )
    cbar = fig.colorbar(cs)


def visualization_angle_weight(z_val=None):
    """Visualize the angle weight inside"""
    obs = DynamicBoundariesPolygon(is_surgery_setup=True)

    d_dist = 0.01
    x_range = [obs.x_min - d_dist, obs.x_max + d_dist]
    y_range = [obs.y_min - d_dist, obs.y_max + d_dist]
    # z_range = 0

    # x_range = [-0.05, -0.025]
    # y_range = [-0.0125, 0.0125]

    n_grid = 20
    x_vals = np.linspace(x_range[0], x_range[1], n_grid)
    y_vals = np.linspace(y_range[0], y_range[1], n_grid)

    if z_val is None:
        # Z value in local frame
        z_val = np.random.rand() * (obs.z_max - obs.z_min) - obs.center_position[2]
    inflation_parameter = np.random.rand(4)

    positions = np.zeros((obs.dim, n_grid, n_grid))
    angle_weights = np.zeros((obs.n_planes, n_grid, n_grid))

    # TODO: debug
    # z_val=-0.021399769780811193
    # inflation_parameter = np.array([0.9348198, 0.01415316, 0.98964482, 0.38309011])

    obs.set_inflation_parameter(inflation_parameter)
    obs.draw_obstacle(num_points=40, z_val=z_val, in_global_frame=False)

    fig = plt.figure(figsize=(16, 3.5))

    print("Global evaluation height (z_val={})".format(z_val))
    print(
        "Inflation parameter (inflation_parameter={})".format(obs.inflation_percentage)
    )

    # point = np.array([0.0466594, 0.07935, z_val])
    # point = np.array([-0.0318749, 0.000174833, z_val])
    # point = np.array([0.0804396, 0.077189, z_val])
    # weights =  obs.get_plane_weights(point
    # import pdb; pdb.set_trace()     ##### DEBUG #####

    for ix in range(n_grid):
        for iy in range(n_grid):
            positions[:, ix, iy] = [x_vals[ix], y_vals[iy], z_val]
            # In local frame only
            angle_weights[:, ix, iy] = obs.get_plane_weights(positions[:, ix, iy])

    for it_plane in range(obs.n_planes):
        # for it_plane in [3]:
        plt.subplot(1, 4, it_plane + 1)
        plt.title("Plane {}".format(it_plane))

        plt.plot(0, 0, "k+", linewidth=18, markeredgewidth=4, markersize=13)
        plt.plot(
            obs.boundary_points_local[0, :],
            obs.boundary_points_local[1, :],
            "k--",
            linewidth=2,
            marker="+",
        )

        cs = plt.contourf(
            positions[0, :, :],
            positions[1, :, :],
            angle_weights[it_plane, :, :],
            np.arange(0, 1.0, 0.025),
            extend="max",
            alpha=0.6,
            zorder=-3,
        )

        if it_plane > 0:
            plt.yticks([])

        plt.axis("equal")
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cs, cax=cbar_ax)


def visualization_indivdual_normal_vector(z_val=None):
    """Visualize the normal vector inside"""
    obs = DynamicBoundariesPolygon(is_surgery_setup=True)

    d_dist = 0.01
    x_range = [obs.x_min - d_dist, obs.x_max + d_dist]
    y_range = [obs.y_min - d_dist, obs.y_max + d_dist]
    # z_range = 0

    n_grid = 10
    x_vals = np.linspace(x_range[0], x_range[1], n_grid)
    y_vals = np.linspace(y_range[0], y_range[1], n_grid)

    if z_val is None:
        # Z value in local frame
        z_val = np.random.rand() * (obs.z_max - obs.z_min) - obs.center_position[2]

    positions = np.zeros((obs.dim, n_grid, n_grid))
    normal_vectors = np.zeros((obs.dim, n_grid, n_grid))

    obs.set_inflation_parameter(np.random.rand(4))

    obs.draw_obstacle(num_points=40, z_val=z_val, in_global_frame=False)

    fig = plt.figure(figsize=(10, 8))

    print("Evaluation height (global) z={}".format(z_val))
    print("Inflation param", obs.inflation_percentage)

    it_plane = 0
    pos = np.array([-0.08, -0.05, z_val])
    normal = obs._get_normal_direction_numerical_to_plane(pos, plane_index=it_plane)

    for it_plane in range(obs.n_planes):
        plt.subplot(2, 2, it_plane + 1)
        plt.title("Plane {}".format(it_plane))

        plt.plot(0, 0, "k+", linewidth=18, markeredgewidth=4, markersize=13)
        plt.plot(
            obs.boundary_points_local[0, :],
            obs.boundary_points_local[1, :],
            "k--",
            linewidth=2,
            marker="+",
        )

        for ix in range(n_grid):
            for iy in range(n_grid):
                positions[:, ix, iy] = [x_vals[ix], y_vals[iy], z_val]
                # In local frame only
                normal_vectors[
                    :, ix, iy
                ] = obs._get_normal_direction_numerical_to_plane(
                    positions[:, ix, iy], plane_index=it_plane
                )

                mag_norm = np.linalg.norm(normal_vectors[:, ix, iy])
                if mag_norm:
                    normal_vectors[:, ix, iy] = normal_vectors[:, ix, iy] / mag_norm

        plt.quiver(positions[0], positions[1], normal_vectors[0], normal_vectors[1])
        plt.axis("equal")


def visualization_normal_vector(z_val=None):
    """Visualize the normal vector inside"""
    obs = DynamicBoundariesPolygon(is_surgery_setup=True)

    d_dist = 0.01
    x_range = [obs.x_min - d_dist, obs.x_max + d_dist]
    y_range = [obs.y_min - d_dist, obs.y_max + d_dist]
    # z_range = 0

    n_grid = 40
    x_vals = np.linspace(x_range[0], x_range[1], n_grid)
    y_vals = np.linspace(y_range[0], y_range[1], n_grid)

    if z_val is None:
        # Z value in local frame
        z_val = np.random.rand() * (obs.z_max - obs.z_min) - obs.center_position[2]

    positions = np.zeros((obs.dim, n_grid, n_grid))
    normal_vectors = np.zeros((obs.dim, n_grid, n_grid))

    obs.set_inflation_parameter(np.random.rand(4))

    z_val = 0.05
    obs.set_inflation_parameter(np.array([0.0, 0, 0, 0]))

    obs.draw_obstacle(num_points=40, z_val=z_val, in_global_frame=False)

    fig = plt.figure(figsize=(10, 8))

    print("Evaluation height (global) z={}".format(z_val))
    print("Inflation param", obs.inflation_percentage)

    plt.plot(0, 0, "k+", linewidth=18, markeredgewidth=4, markersize=13)
    plt.plot(
        obs.boundary_points_local[0, :],
        obs.boundary_points_local[1, :],
        "k--",
        linewidth=2,
        marker="+",
    )

    for ix in range(n_grid):
        for iy in range(n_grid):
            positions[:, ix, iy] = [x_vals[ix], y_vals[iy], z_val]
            gamma = obs.get_gamma(positions[:, ix, iy])
            if gamma < 1:
                continue

            normal_vectors[:, ix, iy] = obs.get_normal_direction(positions[:, ix, iy])

            mag_norm = np.linalg.norm(normal_vectors[:, ix, iy])

            if mag_norm:
                normal_vectors[:, ix, iy] = normal_vectors[:, ix, iy] / mag_norm

    plt.quiver(positions[0], positions[1], normal_vectors[0], normal_vectors[1])
    plt.axis("equal")


def visualization_vector_field_2d():
    pass


if (__name__) == "__main__":
    # Different possible visualization function. Uncomment to execute.
    # visualization_gamma()

    # The weights which are used to compute normal and angular/dynamic velocity
    visualization_angle_weight()

    # Individual normal for each surface point
    # visualization_indivdual_normal_vector()

    # Individual normal for each surface point
    # visualization_normal_vector()

    print("Selected tests complete.")
