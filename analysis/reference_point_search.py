#!/USSR/bin/python3

"""
Reference Point Search

@author LukasHuber
@date 2020-02-28
@conact Lukas.huber@epfl.ch
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
from numpy import pi

# from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import Ellipse


def main(save_figure=False, n_points=100):
    # def eval_eucledian_dsitance(phi_1, phi_2):
    angle_position = np.zeros((2, n_points, n_points))
    distance = np.zeros((n_points, n_points))
    dist_proj = np.zeros((n_points, n_points))

    case = [2]

    phi_2 = np.linspace(0, 2 * pi, n_points)
    phi_1 = np.linspace(-pi, pi, n_points)

    # phi_1 = np.linspace(-pi, pi, n_points)
    # phi_2 = np.linspace(0, 2*pi, n_points)
    # phi_1 = np.linspace(pi/2.0, 2*pi-pi/2.0, n_points)
    # phi_2 = np.linspace(-pi/2.0, pi/2.0, n_points)

    directions_1 = np.vstack((np.cos(phi_1), np.sin(phi_1)))
    directions_2 = np.vstack((np.cos(phi_2), np.sin(phi_2)))

    if 1 in case:
        center_1 = np.array([0.0, 0])
        radius_1 = 2

        center_2 = np.array([0.0, 0])
        radius_2 = 1

        x_lim = [-4, 4]
        y_lim = [-3, 3]

        center_dir = center_2 - center_1

        norm_center_dir = np.linalg.norm(center_dir, 2)
        if norm_center_dir:
            center_dir = center_dir / norm_center_dir

        for ii in range(n_points):
            for jj in range(n_points):
                angle_position[:, ii, jj] = [phi_1[ii], phi_2[jj]]
                surface_point_1 = radius_1 * directions_1[:, ii] + center_1
                surface_point_2 = radius_2 * directions_2[:, jj] + center_2

                dist_dir = surface_point_2 - surface_point_1

                distance[ii, jj] = np.linalg.norm(surface_point_2 - surface_point_1)

                dist_proj[ii, jj] = dist_dir.T.dot(center_dir)

    elif 2 in case:
        # obs1 = Ellipse(axes_length=[5.0, 0.8], orientation=30./180*pi, center_position=[0.0, 0])
        # obs1 = Ellipse(axes_length=[1.0, 1.0], orientation=50./180*pi, center_position=[0.0, 0])
        # obs2 = Ellipse(axes_length=[1.0, 1.0], center_position=[3, 0])

        obs1 = Ellipse(
            axes_length=[2.0, 2.0],
            orientation=50.0 / 180 * pi,
            center_position=[0.0, 0],
        )
        obs2 = Ellipse(axes_length=[2.0, 2.0], center_position=[3, 0])

        center_dir = obs2.center_position - obs1.center_position
        norm_center_dir = np.linalg.norm(center_dir, 2)
        if norm_center_dir:
            center_dir = center_dir / norm_center_dir

        obs_orientation = Ellipse(
            orientation=np.arctan2(center_dir[1], center_dir[0]),
            center_position=np.array([0, 0]),
        )

        dim = obs1.dim

        surface_points_1 = np.zeros((dim, n_points))
        surface_points_2 = np.zeros((dim, n_points))

        normal_vectors_1 = np.zeros((dim, n_points))

        dist_norm = np.zeros((n_points, n_points))

        for ii in range(n_points):
            surface_points_1[:, ii] = obs1.get_intersection_with_surface(
                direction=directions_1[:, ii],
                only_positive_direction=True,
                in_global_frame=True,
            )
            # surface_points_1[:, ii] = obs1.transform_relative2global(surface_points_1[:, ii])

            normal_vectors_1[:, ii] = obs1.get_normal_direction(
                position=surface_points_1[:, ii], in_global_frame=True
            )

        for jj in range(n_points):
            surface_points_2[:, jj] = obs2.get_intersection_with_surface(
                direction=directions_2[:, jj],
                only_positive_direction=True,
                in_global_frame=True,
            )
            # surface_points_2[:, jj] = obs2.transform_relative2global(surface_points_2[:, jj])

        for ii in range(n_points):
            for jj in range(n_points):
                angle_position[:, ii, jj] = [phi_1[ii], phi_2[jj]]
                # surface_points_1 = np.tile(surface_point_1, (n_points, ))
                dist_dir = surface_points_2[:, jj] - surface_points_1[:, ii]
                distance[ii, jj] = np.linalg.norm(dist_dir, axis=0)
                dist_proj[ii, jj] = dist_dir.T.dot(center_dir)

                dist_dir_local = obs_orientation.transform_global2relative_dir(dist_dir)

                # dist_norm1[ii, jj] = dist_dir_local[0] + np.sum(np.abs(dist_dir_local[1:]))
                dist_norm[ii, jj] = -dist_dir.T.dot(normal_vectors_1[:, ii])

        (ii_min, jj_min) = np.unravel_index(
            np.argmin(distance, axis=None), (n_points, n_points)
        )

        plt.figure()
        plt.plot(surface_points_1[0, :], surface_points_1[1, :], "k", label="Ellipse 1")
        plt.plot(surface_points_2[0, :], surface_points_2[1, :], "k", label="Ellipse 2")
        plt.plot(
            surface_points_1[0, ii_min],
            surface_points_1[1, ii_min],
            "*r",
            label="Minimum",
        )
        # plt.plot(surface_points_2[0, jj_min], surface_points_2[1, jj_min], '*r', label='Minimum')
        plt.plot(surface_points_2[0, jj_min], surface_points_2[1, jj_min], "*r")
        plt.legend()
        plt.grid()
        plt.axis("equal")
        # plt.plot(directions_1[0, :], directions_1[1, :], 'o')

        plt.ion()
        plt.show()

    # def gradient_descent_distance(self, obs1, obs2):
    # dist = sqrt(\sum_i point(phi_i)^2)
    # =>
    # d dist / d phi_j  = 1/(2*sqrt ()) *2* point(phi_j) * d point/ d phi_j

    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # plt.close('all')
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(
        angle_position[0, :, :],
        angle_position[1, :, :],
        distance,
        cmap=cm.coolwarm,
        # linewidth=0,
        antialiased=False,
    )
    plt.xlabel("$\phi_1$")
    plt.ylabel("$\phi_2$")
    plt.title("Eucledian Distance")
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(
        angle_position[0, :, :],
        angle_position[1, :, :],
        dist_proj,
        cmap=cm.coolwarm,
        # linewidth=0,
        antialiased=False,
    )
    plt.title("Projected Eucledian Distance")
    plt.xlabel("$\phi_1$")
    plt.ylabel("$\phi_2$")
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(
        angle_position[0, :, :],
        angle_position[1, :, :],
        dist_norm,
        cmap=cm.coolwarm,
        # linewidth=0,
        antialiased=False,
    )

    plt.title("Norm 1 in Connection Frame")
    plt.xlabel("$\phi_1$")
    plt.ylabel("$\phi_2$")
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    if save_figure:
        figure_name = "figure_comparsion"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    main()
