#!/USSR/bin/python3
"""
Test script for obstacle avoidance algorithm
Test normal formation
"""
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import Cuboid, Polygon
from dynamic_obstacle_avoidance.containers import ObstacleContainer, GradientContainer
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem

import unittest


class TestPolygon(unittest.TestCase):
    def test_single_polygon(self, visualize=False):
        margin_absolut = 0.1

        obs = Cuboid(
            center_position=np.array([0.2, 2.4]),
            axes_length=[0.4, 2.4],
            margin_absolut=margin_absolut,
            orientation=-30 * pi / 180,
            tail_effect=False,
        )

        obstacle_environment = ObstacleContainer()
        obstacle_environment.append(obs)

        point0 = obs.center_position + np.array([1, 0])
        point1 = obs.center_position + np.array([2, 0])

        gamma0 = obs.get_gamma(point0, in_global_frame=True)
        gamma1 = obs.get_gamma(point1, in_global_frame=True)

        self.assertTrue(gamma0 < gamma1)

        radius0 = obs.get_local_radius(point0, in_global_frame=True)
        radius1 = obs.get_local_radius(point1, in_global_frame=True)

        self.assertTrue(np.isclose(radius0, radius1))

        if visualize:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            x_lim = [-3, 3]
            y_lim = [-0.5, 3.5]

            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

            n_resolution = 40
            dim = 2
            x_vals = np.linspace(x_lim[0], x_lim[1], n_resolution)
            y_vals = np.linspace(y_lim[0], y_lim[1], n_resolution)

            gamma_values = np.zeros((n_resolution, n_resolution))
            positions = np.zeros((dim, n_resolution, n_resolution))

            for ix in range(n_resolution):
                for iy in range(n_resolution):
                    positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]

                    gamma_values[ix, iy] = obs.get_gamma(
                        positions[:, ix, iy], in_global_frame=True
                    )

            cs = ax.contourf(
                positions[0, :, :],
                positions[1, :, :],
                gamma_values,
                np.arange(1.0, 5.0, 0.5),
                # cmap=plt.get_cmap('autumn'),
                cmap=plt.get_cmap("hot"),
                extend="max",
                alpha=0.6,
                zorder=-3,
            )

            cbar = fig.colorbar(cs)

    def test_cuboids_multigamma(
        self,
        visualize=False,
        save_figure=False,
        n_resolution=50,
    ):
        margin_absolut = 0.1
        obstacle_environment = ObstacleContainer()
        obstacle_environment.append(
            Cuboid(
                center_position=np.array([0.2, 2.4]),
                axes_length=[0.4, 2.4],
                margin_absolut=margin_absolut,
                orientation=-30 * pi / 180,
                tail_effect=False,
                repulsion_coeff=1.5,
            )
        )

        obstacle_environment.append(
            Cuboid(
                center_position=np.array([1.2, 0.25]),
                axes_length=[0.4, 1.45],
                margin_absolut=margin_absolut,
                orientation=0 * pi / 180,
                tail_effect=False,
                repulsion_coeff=1.5,
            )
        )

        attractor_position = np.array([2, 0.7])
        initial_dynamics = LinearSystem(
            attractor_position=attractor_position,
            maximum_velocity=1,
            distance_decrease=0.3,
        )

        dynamic_avoider = ModulationAvoider(
            initial_dynamics=initial_dynamics, obstacle_environment=obstacle_environment
        )

        # Function was changed
        # point0 = np.array([1.05, 1.82])
        # gamma0 = dynamic_avoider.get_gamma_product(point0)

        # point1 = np.array([1.64, 2.08])
        # gamma1 = dynamic_avoider.get_gamma_product(point1)

        # self.assertTrue(gamma0 < gamma1)

        if visualize:
            fig, ax = plt.subplots(figsize=(14, 9))
            x_lim = [-3, 3]
            y_lim = [-0.5, 3.5]

            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

            dim = 2
            x_vals = np.linspace(x_lim[0], x_lim[1], n_resolution)
            y_vals = np.linspace(y_lim[0], y_lim[1], n_resolution)

            gamma_values = np.zeros((n_resolution, n_resolution))
            positions = np.zeros((dim, n_resolution, n_resolution))
            velocities = np.zeros((dim, n_resolution, n_resolution))

            for ix in range(n_resolution):
                for iy in range(n_resolution):
                    positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]

                    gamma_values[ix, iy] = dynamic_avoider.get_gamma_product(
                        positions[:, ix, iy]
                    )

                    if gamma_values[ix, iy] <= 1:
                        continue

                    velocities[:, ix, iy] = dynamic_avoider.evaluate(
                        positions[:, ix, iy]
                    )

            cs = ax.contourf(
                positions[0, :, :],
                positions[1, :, :],
                gamma_values,
                np.arange(1.0, 5.0, 0.2),
                # cmap=plt.get_cmap('autumn'),
                cmap=plt.get_cmap("hot"),
                extend="max",
                alpha=0.6,
                zorder=-3,
            )

            cbar = fig.colorbar(cs)

            # breakpoint()
            ax.streamplot(
                positions[0, :, :].T,
                positions[1, :, :].T,
                velocities[0, :, :].T,
                velocities[1, :, :].T,
                color="k",
            )
            # ax.quiver(positions[0, :, :].T, positions[1, :, :].T,
            # velocities[0, :, :].T, velocities[1, :, :].T, color='k')

            plt.plot(
                dynamic_avoider.initial_dynamics.attractor_position[0],
                dynamic_avoider.initial_dynamics.attractor_position[1],
                "k*",
                markeredgewidth=2,
                markersize=10,
            )

            if save_figure:
                figName = "gamma_danger_field_for_multiobstacle_and_vector_field"
                plt.savefig("figures/" + figName + ".png", bbox_inches="tight")

            position = np.array([-0.39436336, 1.14369659])
            gamma = dynamic_avoider.get_gamma_product(position)

            plt.plot(position[0], position[1], "ko")

            # print('gamma', gamma)
            # breakpoint()

    def test_polygon_multigamma(
        self,
        visualize=False,
        save_figure=False,
        n_resolution=50,
    ):
        wall_width = 0.02
        margin_absolut = 0.05
        edge_points = [
            [0.5 - wall_width, 1 + margin_absolut],
            [0.5 + wall_width, 1 + margin_absolut],
            [0.5 + wall_width, 2.0 - wall_width],
            [1.5, 2.0 - wall_width],
            [1.5, 2.0 + wall_width],
            [0.5 - wall_width, 2.0 + wall_width],
        ]

        center_position = np.array([0.5, 2.0])
        attractor_position = np.array([-1, 0])

        obstacle_environment = ObstacleContainer()
        obstacle_environment.append(
            Polygon(
                edge_points=np.array(edge_points).T,
                center_position=center_position,
                margin_absolut=margin_absolut,
                absolute_edge_position=True,
                tail_effect=False,
                repulsion_coeff=1.4,
            )
        )

        obstacle_environment.append(
            Cuboid(
                axes_length=[0.9, wall_width * 2],
                center_position=np.array([1, (-1) * (margin_absolut + wall_width)]),
                margin_absolut=margin_absolut,
                tail_effect=False,
            )
        )

        attractor_position = np.array([2, 0.7])
        initial_dynamics = LinearSystem(
            attractor_position=attractor_position,
            maximum_velocity=1,
            distance_decrease=0.3,
        )

        dynamic_avoider = ModulationAvoider(
            initial_dynamics=initial_dynamics, obstacle_environment=obstacle_environment
        )

        if visualize:
            fig, ax = plt.subplots(figsize=(14, 9))
            x_lim = [-1.5, 2]
            y_lim = [-0.5, 2.5]

            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

            dim = 2
            x_vals = np.linspace(x_lim[0], x_lim[1], n_resolution)
            y_vals = np.linspace(y_lim[0], y_lim[1], n_resolution)

            gamma_values = np.zeros((n_resolution, n_resolution))
            positions = np.zeros((dim, n_resolution, n_resolution))
            # velocities = np.zeros((dim, n_resolution, n_resolution))

            for ix in range(n_resolution):
                for iy in range(n_resolution):
                    positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]

                    gamma_values[ix, iy] = dynamic_avoider.get_gamma_product(
                        positions[:, ix, iy]
                    )

                    if gamma_values[ix, iy] <= 1:
                        continue

                    # velocities[:, ix, iy] = dynamic_avoider.evaluate(positions[:, ix, iy])

            cs = ax.contourf(
                positions[0, :, :],
                positions[1, :, :],
                gamma_values,
                np.arange(1.0, 5.0, 0.2),
                # cmap=plt.get_cmap('autumn'),
                cmap=plt.get_cmap("hot"),
                extend="max",
                alpha=0.6,
                zorder=-3,
            )

            cbar = fig.colorbar(cs)

            # breakpoint()
            # ax.streamplot(positions[0, :, :].T, positions[1, :, :].T,
            # velocities[0, :, :].T, velocities[1, :, :].T, color='k')
            # ax.quiver(positions[0, :, :].T, positions[1, :, :].T,
            # velocities[0, :, :].T, velocities[1, :, :].T, color='k')

            plt.plot(
                dynamic_avoider.initial_dynamics.attractor_position[0],
                dynamic_avoider.initial_dynamics.attractor_position[1],
                "k*",
                markeredgewidth=2,
                markersize=10,
            )

            if save_figure:
                figName = "gamma_danger_field_for_multipolygon_and_vector_field"
                plt.savefig("figures/" + figName + ".png", bbox_inches="tight")

            position = np.array([-0.39436336, 1.14369659])
            gamma = dynamic_avoider.get_gamma_product(position)

            plt.plot(position[0], position[1], "ko")

            # print('gamma', gamma)
            # breakpoint()


if (__name__) == "__main__":
    run_all = False
    if run_all:
        unittest.main(argv=["first-arg-is-ignored"], exit=False)

    else:
        plt.close("all")
        plt.ion()
        plt.show()

        my_tester = TestPolygon()
        my_tester.test_single_polygon(visualize=True)
        my_tester.test_polygon_multigamma(
            visualize=True, save_figure=True, n_resolution=100
        )
        my_tester.test_polygon_multigamma(
            visualize=True, save_figure=False, n_resolution=100
        )
