#!/USSR/bin/python3
"""
Reference Point Search
"""
# Author: Lukas Huber
# Date:  2020-02-28
# Email: lukas.huber@epfl.ch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
from numpy import pi

# from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import get_orthogonal_basis, get_angle_space_inverse
from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_angle_space_inverse

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import Obstacle, Ellipse, Cuboid


def main(save_figure=False, n_points=100, case=[1]):
    # def eval_eucledian_dsitance(phi_1, phi_2):
    angle_position = np.zeros((2, n_points, n_points))
    distance = np.zeros((n_points, n_points))
    dist_proj = np.zeros((n_points, n_points))

    case = [1]

    phi_1 = np.linspace(-pi, pi, n_points)
    phi_2 = np.linspace(-pi, pi, n_points)

    directions_1 = np.vstack((np.cos(phi_1), np.sin(phi_1)))
    directions_2 = np.vstack((np.cos(phi_2), np.sin(phi_2)))

    obs1 = Cuboid(
        axes_length=[0.7, 1.0],
        orientation=10.0 / 180 * pi,
        center_position=[0.0, 0.0],
        margin_absolut=0.0,
    )
    # obs1 = Ellipse(axes_length=[3.0, 0.5], orientation=00./180*pi, center_position=[0.0, 0.0])
    obs2 = Ellipse(
        axes_length=[0.7, 1.0],
        orientation=30.0 / 180 * pi,
        center_position=[2.0, 0.0],
        margin_absolut=0.2,
    )

    # obs1 = Ellipse(axes_length=[3.0, 2.0], orientation=00./180*pi, center_position=[0.0, 0.0])
    # obs2 = Ellipse(axes_length=[2.0, 1.5], orientation=30./180*pi, center_position=[3.0, 0.0], margin_absolut=0.2)

    # obs1 = Ellipse(axes_length=[1, 4.0], orientation=45./180*pi, center_position=[0.0, 0.0])
    # obs2 = Ellipse(axes_length=[6.0, 1.0], orientation=00./180*pi, center_position=[3.0, 0.0])

    setup = 2
    if setup == 0:
        obs1 = Ellipse(
            center_position=[0.0, 0.0],
            axes_length=[0.8, 1.2],
            # margin_absolut=1.0,
            margin_absolut=0.5,
            orientation=30 * pi / 180,
            linear_velocity=[0.0, 0],
        )

        obs2 = Ellipse(
            center_position=[3.0, 0.0],
            axes_length=[0.8, 1.2],
            # margin_absolut=1.0,
            margin_absolut=0.5,
            orientation=-30 * pi / 180,
            linear_velocity=[-0.1, 0],
        )

    elif setup == 1:
        obs1 = Ellipse(
            center_position=[0.0, 0.0],
            axes_length=[0.8, 1.2],
            # margin_absolut=1.0,
            margin_absolut=0.5,
            orientation=30 * pi / 180,
            linear_velocity=[0.0, 0],
        )

        dphi = 6.1
        it = 0

        obs2 = Ellipse(
            center_position=[3.5, 0.0],
            axes_length=[0.8, 2.2],
            # margin_absolut=1.0,
            margin_absolut=0.5,
            # orientation=-30*pi/180,
            orientation=pi / 180 * (233.3333 + it * dphi),
            linear_velocity=[0.0, 0],
            angular_velocity=0,
        )

    elif setup == 2:
        obs1 = Cuboid(
            center_position=[0.0, 0.0],
            axes_length=[0.5, 0.8],
            margin_absolut=0.5,
            orientation=30 * pi / 180,
            linear_velocity=[0.0, 0],
        )

        obs2 = Ellipse(
            center_position=[3.5, 0.0],
            axes_length=[1.2, 2.0],
            margin_absolut=0.5,
            orientation=pi / 180 * 50,
            linear_velocity=[0.0, 0],
            angular_velocity=0,
        )

    obs = ObstacleContainer()
    obs.append(obs1)
    obs.append(obs2)

    for obstacle in obs:
        obstacle.draw_obstacle(numPoints=n_points)

    if 1 in case:
        center_dir = obs2.center_position - obs1.center_position
        norm_center_dir = np.linalg.norm(center_dir, 2)
        if norm_center_dir:
            center_dir = center_dir / norm_center_dir

        dim = obs1.dim

        surface_points_1 = np.zeros((dim, n_points))
        surface_points_2 = np.zeros((dim, n_points))

        surface_derivative1 = np.zeros((dim - 1, dim, n_points))
        surface_derivative2 = np.zeros((dim - 1, dim, n_points))

        norm_derivative1 = np.zeros((dim - 1, dim, n_points))
        norm_derivative2 = np.zeros((dim - 1, dim, n_points))

        dist_norm = np.zeros((n_points, n_points))
        d_dphi = np.zeros(((dim - 1) * 2, n_points, n_points))

        norm_dist_1 = np.zeros((dim, n_points))
        norm_dist_2 = np.zeros((dim, n_points))

        NullMatrix_1 = get_orthogonal_basis(center_dir)
        NullMatrix_2 = get_orthogonal_basis(-center_dir)

        for ii in range(n_points):
            # surface_points_1[:, ii] = obs1.get_intersection_with_surface(direction=directions_1[:, ii], only_positive_direction=True, in_global_frame=True)
            # surface_derivative1[:, ii] = obs1.get_surface_derivative_angle(phi_1[ii], in_global_frame=True)
            surface_derivative1[:, :, ii] = obs1.get_surface_derivative_angle_num(
                np.array([phi_1[ii]]), NullMatrix=NullMatrix_1, in_global_frame=True
            )
            norm_derivative1[:, :, ii] = obs[0].get_normal_derivative_angle_num(
                np.array([phi_1[ii]]), NullMatrix=NullMatrix_1, in_global_frame=True
            )
            norm_dist_1[:, ii] = obs[0].get_normal_direction(
                surface_points_1[:, ii], in_global_frame=True
            )

            directions_1 = get_angle_space_inverse(
                np.array([phi_1[ii]]), NullMatrix=NullMatrix_1
            )
            surface_points_1[:, ii] = obs1.get_local_radius_point(
                direction=directions_1, in_global_frame=True
            )

            # directions_1 = np.array([np.cos(ang_1), np.sin(ang_1)])
            # directions_2 = np.array([np.cos(ang_2), np.sin(ang_2)])

        for jj in range(n_points):
            # surface_points_2[:, jj] = obs2.get_intersection_with_surface(direction=directions_2[:, jj], only_positive_direction=True, in_global_frame=True)

            # surface_derivative2[:, jj] =  obs2.get_surface_derivative_angle(phi_2[jj], in_global_frame=True)
            surface_derivative2[:, :, jj] = obs2.get_surface_derivative_angle_num(
                np.array([phi_2[jj]]), NullMatrix=NullMatrix_2, in_global_frame=True
            )
            norm_derivative2[:, :, jj] = obs[1].get_normal_derivative_angle_num(
                np.array([phi_2[jj]]), NullMatrix=NullMatrix_1, in_global_frame=True
            )

            norm_dist_2[:, jj] = obs[1].get_normal_direction(
                surface_points_2[:, jj], in_global_frame=True
            )
            directions_2 = get_angle_space_inverse(
                np.array([phi_2[jj]]), NullMatrix=NullMatrix_2
            )
            surface_points_2[:, jj] = obs2.get_local_radius_point(
                direction=directions_2, in_global_frame=True
            )

        proj_matrix = np.zeros((dim, dim))
        proj_matrix[:, 0] = center_dir
        proj_matrix[:, 1] = [-proj_matrix[1, 0], -proj_matrix[0, 0]]

        for ii in range(n_points):
            for jj in range(n_points):
                angle_position[:, ii, jj] = [phi_1[ii], phi_2[jj]]

                dist_dir = surface_points_2[:, jj] - surface_points_1[:, ii]
                distance[ii, jj] = np.linalg.norm(dist_dir, axis=0)

                # TODO: check sign
                d_dphi[0, ii, jj] = (
                    1.0 / distance[ii, jj] * surface_derivative1[:, :, ii].dot(dist_dir)
                )
                d_dphi[1, ii, jj] = (
                    -1.0
                    / distance[ii, jj]
                    * surface_derivative2[:, :, jj].dot(dist_dir)
                )

                norm_sum = norm_dist_1[:, ii] - norm_dist_2[:, jj]

                # d_dphi[0, ii, jj] = norm_derivative1[:, :, ii].dot(dist_dir) + surface_derivative1[:, :, ii].dot(norm_sum)
                # d_dphi[1, ii, jj] = norm_derivative2[:, :, jj].dot(dist_dir) + surface_derivative2[:, :, jj].dot(norm_sum)

                # d_dphi[1, ii, jj] = -1.0/distance[ii, jj]*surface_derivative2[:, jj].dot(dist_dir)

                # dist_proj[ii, jj] = dist_dir.T.dot(center_dir)
                # dist_proj = (norm_dist_1[:, ii]-norm_dist_2[:, jj]).T.dot(center_dir)

                proj_matr = np.zeros((dim, 2))
                proj_matr[:, 0] = norm_dist_1[:, ii]
                proj_matr[:, 1] = [-proj_matr[1, 0], proj_matr[0, 0]]

                dist_type = 0
                if dist_type == 0:
                    distance[ii, jj] = np.linalg.norm(dist_dir, axis=0)

                elif dist_type == 1:
                    dist_proj = np.linalg.pinv(proj_matr).dot(dist_dir)
                    dist_sqr = dist_proj**2
                    dist_sqr[0] = np.copysign(dist_sqr[0], dist_proj[0])

                    dist_sum = np.sum(dist_sqr)
                    dist_proj = (norm_dist_1[:, ii] - norm_dist_2[:, jj]).T.dot(
                        center_dir
                    )

                    proj_matr = np.zeros((dim, 2))
                    proj_matr[:, 0] = norm_dist_2[:, jj]
                    proj_matr[:, 1] = [-proj_matr[1, 0], proj_matr[0, 0]]

                    dist_proj = np.linalg.pinv(proj_matr).dot(-dist_dir)
                    dist_sqr = dist_proj**2
                    dist_sqr[0] = np.copysign(dist_sqr[0], dist_proj[0])

                    dist_sum += np.sum(dist_sqr)

                    # distance[ii, jj] = np.sum(dist_sqr) + distance[ii, jj]

                    distance[ii, jj] = dist_sum

                elif dist_type == 2:
                    distance[ii, jj] = 0.5 * (
                        norm_dist_1[:, ii] - norm_dist_2[:, jj]
                    ).T.dot(dist_dir)

                # distance[ii, jj] = dist_proj[ii, jj]*distance[ii, jj]
                # distance[ii, jj] = dist_proj[ii, jj]

                # distance[ii, jj] = np.copysign(distance[ii, jj], dist_dir.dot(center_dir))
                # val =  np.copysign(distance[ii, jj], dist_dir.dot(center_dir))

        (ii_min, jj_min) = np.unravel_index(
            np.argmin(distance, axis=None), (n_points, n_points)
        )

        show_normals = False
        if show_normals:
            plt.figure()
            plt.plot(surface_points_1[0, :], surface_points_1[1, :])
            # Outline
            plt.plot(surface_points_1[0, :], surface_points_1[1, :], alpha=0.7)
            plt.plot(surface_points_2[0, :], surface_points_2[1, :], alpha=0.7)
            # Normals
            # plt.quiver(surface_points_1[0, :], surface_points_1[1, :], norm_dist_1[0, :], norm_dist_1[1, :])
            # plt.quiver(surface_points_2[0, :], surface_points_2[1, :], norm_dist_2[0, :], norm_dist_2[1,:])
            # Tangents
            plt.quiver(
                surface_points_1[0, :],
                surface_points_1[1, :],
                surface_derivative1[0, :],
                surface_derivative1[1, :],
            )
            plt.quiver(
                surface_points_2[0, :],
                surface_points_2[1, :],
                surface_derivative2[0, :],
                surface_derivative2[1, :],
            )

            plt.axis("equal")
            plt.grid()
        # plt.plot()
        # plt.title("Surface Derivative")

        # plt.figure()
        # plt.quiver(phi_1.reshape(n_points, 1), np.zeros((n_points, 1)), surface_derivative1[0, :].reshape(n_points, 1), surface_derivative1[1, :].reshape(n_points, 1))
        # plt.xlabel('$\phi_1$')
        # plt.title('Surface Direction')

        if False:
            print("sur dir \n", np.round(surface_derivative1, 2))
            print("sur dir \n", np.round(surface_derivative2, 2))
            # print('sur dir \n', np.round(surface_points_1, 2))
            # print('sur dir \n', np.round(surface_points_2, 2))
            raise ValueError("Stop script")

        # plt.figure()
        # plt.quiver(phi_2.reshape(n_points, 1), np.zeros((n_points, 1)), surface_derivative2[0, :].reshape(n_points, 1), surface_derivative2[1, :].reshape(n_points, 1))

        # plt.xlabel('$\phi_2$')
        # plt.title('Surface Direction')

        plt.ion()
        plt.show()

        # phi_1 = phi_1*180/pi
        # phi_2 = phi_2*180/pi
        # angle_position = angle_position*180/pi

        show_plot_3d = True
        if show_plot_3d:
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            surf = ax.plot_surface(
                angle_position[0, :, :],
                angle_position[1, :, :],
                distance,
                cmap=cm.coolwarm,  # linewidth=0,
                antialiased=False,
                alpha=0.9,
            )

            freq_quiver = 3
            ind_ = np.logical_not(np.mod(np.arange(n_points), freq_quiver))
            # plt.quiver(angle_position[0, ind_, :][:, ind_], angle_position[1, ind_, :][:, ind_], np.zeros(np.sum(ind_)), d_dphi[0, ind_,  :][:, ind_], d_dphi[1, ind_,  :][:, ind_], np.zeros(np.sum(ind_)), color='k')
            plt.xlabel("$\phi_1$ [rad]")
            plt.ylabel("$\phi_2$ [rad]")
            # plt.title('Eucledian Distance')
            fig.colorbar(surf, shrink=0.5, aspect=5)

        show_stream_plot = True
        if show_stream_plot:
            fig = plt.figure(figsize=(4, 3))
            cs = plt.contourf(phi_1, phi_2, distance.T, zorder=-3, alpha=0.6)
            cbar = fig.colorbar(cs)
            plt.streamplot(
                phi_1, phi_2, d_dphi[0, :, :].T, d_dphi[1, :, :].T, zorder=-2, color="k"
            )
            # plt.quiver(angle_position[0, :, :], angle_position[1, :, :], d_dphi[0, :,  :], d_dphi[1, :,  :])
            # plt.streamplot(angle_position[0, :, :], angle_position[1, :, :], d_dphi[0, :,  :], d_dphi[1, :,  :])
            plt.plot(phi_1[ii_min], phi_2[jj_min], "*r", zorder=1)
            plt.grid()
            plt.axis("equal")
            plt.xlim(phi_1[0], phi_1[-1])
            plt.ylim(phi_2[0], phi_2[-1])
            plt.xlabel("$\phi_1$")
            plt.ylabel("$\phi_2$")
            # plt.title('Streamline of Gradient')

            if save_figure:
                figure_name = "2d_plot_angle_distance_descent"
                plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

        show_quiver = False
        if show_quiver:
            plt.figure(figsize=(6, 6))
            plt.quiver(
                angle_position[0, :, :],
                angle_position[1, :, :],
                d_dphi[0, :, :],
                d_dphi[1, :, :],
            )
            plt.plot(phi_1[ii_min], phi_2[jj_min], "*r", zorder=1)
            plt.grid()
            plt.axis("equal")
            plt.xlim(phi_1[0], phi_1[-1])
            plt.ylim(phi_2[0], phi_2[-1])
            plt.xlabel("$\phi_1$")
            plt.ylabel("$\phi_2$")
            plt.title("Streamline of Gradient")

    if 2 in case:
        dim_ang = 2 * (obs1.dim - 1)

        dist_dir = obs2.center_position - obs1.center_position

        ang_1_0 = np.arctan2(dist_dir[1], dist_dir[0])
        ang_2_0 = np.arctan2(-dist_dir[1], -dist_dir[0])

        ang_1 = 0
        ang_2 = 0

        angles_optimization = np.array([[ang_1], [ang_2]])
        print("angso", angles_optimization)
        distance_opt = np.zeros((0))

        step_size = 0.03
        convergence_err = 1e-3

        it_count = 0
        max_it = 1000

        NullMatrix_1 = get_orthogonal_basis(dist_dir)
        NullMatrix_2 = get_orthogonal_basis(-dist_dir)
        while True:
            dist_old = -1

            # surface_derivative1 = obs1.get_surface_derivative_angle(ang_1, in_global_frame=True)
            # surface_derivative2 = obs2.get_surface_derivative_angle(ang_2, in_global_frame=True)

            # dir_1 = np.array([np.cos(ang_1), np.sin(ang_1)])
            # dir_2 = np.array([np.cos(ang_2), np.sin(ang_2)])

            surface_derivative1 = obs1.get_surface_derivative_angle_num(
                np.array([ang_1]), NullMatrix=NullMatrix_1, in_global_frame=True
            )
            surface_derivative2 = obs2.get_surface_derivative_angle_num(
                np.array([ang_2]), NullMatrix=NullMatrix_2, in_global_frame=True
            )

            directions_1 = get_angle_space_inverse(
                np.array([ang_1]), NullMatrix=NullMatrix_1
            )
            directions_2 = get_angle_space_inverse(
                np.array([ang_2]), NullMatrix=NullMatrix_2
            )

            # directions_1 = np.array([np.cos(ang_1), np.sin(ang_1)])
            # directions_2 = np.array([np.cos(ang_2), np.sin(ang_2)])

            surface_point_1 = obs1.get_local_radius_point(
                direction=directions_1, in_global_frame=True
            )
            surface_point_2 = obs2.get_local_radius_point(
                direction=directions_2, in_global_frame=True
            )

            dist_dir = surface_point_2 - surface_point_1
            distance_opt = np.append(distance_opt, [np.linalg.norm(dist_dir)])

            d_dphi_opt = (
                0.5
                / distance_opt[-1]
                * np.array(
                    [
                        -dist_dir.dot(surface_derivative1.squeeze()),
                        dist_dir.dot(surface_derivative2.squeeze()),
                    ]
                )
            )
            d_dphi_opt = d_dphi_opt.squeeze()  # TODO: squeeze earlier

            new_angles = angles_optimization[:, -1] - step_size * d_dphi_opt

            angles_optimization = np.append(
                angles_optimization, new_angles.reshape(dim_ang, 1), axis=1
            )

            ang_1, ang_2 = new_angles[0], new_angles[1]
            it_count += 1

            # Gradient descent step
            if (
                it_count > max_it
                or np.linalg.norm(
                    angles_optimization[:, -1] - angles_optimization[:, -2]
                )
                < convergence_err
            ):
                break

            if np.abs(ang_1) > pi / 2.0 or np.abs(ang_2) > pi / 2.0:  # restart
                ang_1 = 0
                ang_2 = 0

        print("Converged after {} iterations.".format(it_count))

    if 1 in case and 2 in case:
        angles_optimization[0, :] = angles_optimization[0, :]
        angles_optimization[1, :] = angles_optimization[1, :]

        # plt.figure()
        angles_optimization[0, :] = angles_optimization[0, :]
        plt.plot(
            angles_optimization[0, :],
            angles_optimization[1, :],
            "g",
            zorder=2,
            linewidth=4,
        )
        plt.plot(
            angles_optimization[0, -1],
            angles_optimization[1, -1],
            "*g",
            label="Minimum Gradient",
            zorder=2,
        )
        plt.plot(angles_optimization[0, 0], angles_optimization[1, 0], "og", zorder=2)
        plt.plot(
            phi_1[ii_min], phi_2[jj_min], "*r", label="Minimum Numerical", zorder=2
        )
        plt.axis("equal")
        plt.legend(loc=2)

        plt.figure()
        plt.plot(
            np.arange(it_count),
            distance_opt,
            "g",
            label="Gradient Descend (min={})".format(np.round(distance_opt[-1], 3)),
        )
        plt.plot(
            np.sum(it_count) - 1,
            distance[ii_min, jj_min],
            ".r",
            label="Numerical Distance = {}".format(
                np.round(distance[ii_min, jj_min], 3)
            ),
        )
        # plt.grid()
        plt.xlabel("Iteration")
        plt.ylabel("Distance")
        plt.grid()
        plt.legend()

    if 1 in case:
        fig = plt.figure(figsize=(3.8, 3.4))
        # plt.figure()
        plt.plot(
            surface_points_1[0, :],
            surface_points_1[1, :],
            "k",
            # label='Ellipse 1',
        )
        plt.plot(
            surface_points_2[0, :],
            surface_points_2[1, :],
            "k",
            # label='Ellipse 2',
        )
        plt.plot(
            surface_points_1[0, ii_min],
            surface_points_1[1, ii_min],
            "*r",
            label="Minimum at $\phi_1$={} / $\phi_2$ = {}".format(
                np.round(phi_1[ii_min] * 180 / pi, 0),
                np.round(phi_2[jj_min] * 180 / pi, 0),
            ),
        )
        # plt.plot(surface_points_2[0, jj_min], surface_points_2[1, jj_min], '*r', label='Minimum')
        plt.plot(surface_points_2[0, jj_min], surface_points_2[1, jj_min], "*r")
        plt.legend(loc=4)
        plt.grid()
        plt.axis("equal")

        if save_figure:
            figure_name = "2d_two_object_gradient_descent"
            plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

        if 2 in case:
            plt.plot(
                surface_point_1[0],
                surface_point_1[1],
                "*g",
                label="Minimum Gradient @ $\phi_1$={} / $\phi_2$ = {}".format(
                    np.round(ang_1 * 180 / pi, 0), np.round(ang_2 * 180 / pi, 0)
                ),
            )

            plt.plot(surface_point_2[0], surface_point_2[1], "*g")
        # plt.plot(directions_1[0, :], directions_1[1, :], 'o')


if (__name__) == "__main__":
    main(save_figure=False)

    plt.ion()
    plt.show()
