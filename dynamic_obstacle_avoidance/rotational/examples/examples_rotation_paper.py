#!/USSR/bin/python3
""" Script to show lab environment on computer """
# Author: Lukas Huber
# License: BSD (c) 2021

import warnings
import copy
from functools import partial
import math

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

from vartools.dynamical_systems import LinearSystem, QuadraticAxisConvergence
from dynamic_obstacle_avoidance.rotational.dynamics import WavyLinearDynamics

# from vartools.dynamical_systems import SinusAttractorSystem
# from vartools.dynamical_systems import BifurcationSpiral

from vartools.dynamical_systems import plot_dynamical_system_streamplot

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import StarshapedFlower

from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

from dynamic_obstacle_avoidance.rotational.multiboundary_container import (
    MultiBoundaryContainer,
)

from dynamic_obstacle_avoidance.rotational.rotation_container import RotationContainer
from dynamic_obstacle_avoidance.rotational.rotational_avoidance import (
    obstacle_avoidance_rotational,
)
from dynamic_obstacle_avoidance.rotational.rotational_avoider import RotationalAvoider

from dynamic_obstacle_avoidance.visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)

from vartools.visualization import VectorfieldPlotter

# plt.close('all')
plt.ion()


def function_integrator(
    start_point,
    dynamics,
    stopping_functor,
    stepsize: float = 0.1,
    err_abs: float = 1e-1,
    it_max: int = 100,
):
    points = np.zeros((start_point.shape[0], it_max + 1))
    points[:, 0] = start_point

    for ii in range(it_max):
        velocity = dynamics(points[:, ii])
        points[:, ii + 1] = points[:, ii] + velocity * stepsize

        if stopping_functor is not None and stopping_functor(points[:, ii]):
            print(f"Stopped at it={ii}")
            break

        if err_abs is not None and LA.norm(velocity) < err_abs:
            print(f"Converged at it={ii}")
            break

    return points[:, : ii + 1]


def inverted_star_obstacle_avoidance(visualize=False, save_figure=False):
    obstacle_list = RotationContainer()

    obstacle_list.append(
        StarshapedFlower(
            center_position=np.array([0, 0]),
            radius_magnitude=2,
            number_of_edges=5,
            radius_mean=7,
            orientation=0.0 / 180 * pi,
            tail_effect=False,
            is_boundary=True,
        )
    )
    # initial_dynamics = WavyLinearDynamics(attractor_position=np.array([0, 0]))
    initial_dynamics = WavyLinearDynamics(attractor_position=np.array([6.8, -1]))
    convergence_dynamics = LinearSystem(
        attractor_position=initial_dynamics.attractor_position
    )

    obstacle_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
        convergence_system=convergence_dynamics,
    )

    if visualize:
        plt.close("all")

        x_lim = [-10, 10]
        y_lim = [-10, 10]

        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

        Simulation_vectorFields(
            x_lim,
            y_lim,
            n_resolution=100,
            # n_resolution=20,
            obstacle_list=obstacle_list,
            # obstacle_list=[],
            saveFigure=False,
            noTicks=True,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=initial_dynamics.evaluate,
            obs_avoidance_func=obstacle_avoider.avoid,
            automatic_reference_point=False,
            pos_attractor=initial_dynamics.attractor_position,
            fig_and_ax_handle=(fig, ax),
            # Quiver or Streamplot
            show_streamplot=True,
            # show_streamplot=False,
        )

        if save_figure:
            fig_name = "wavy_nonlinear_ds_in_star_obstacle"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)


def visualization_inverted_ellipsoid(visualize=False, save_figure=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([7, 4]),
            is_boundary=True,
            orientation=30 / 180.0 * math.pi,
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([2.2, 1]))
    convergence_dynamics = initial_dynamics

    # main_avoider = RotationalAvoider()
    # my_avoider = partial(main_avoider.avoid, convergence_radius=math.pi)

    obstacle_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
        convergence_system=convergence_dynamics,
    )
    obstacle_avoider.convergence_radius = math.pi

    if visualize:
        plt.close("all")

        x_lim = [-3.4, 3.4]
        y_lim = [-3.4, 3.4]

        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

        Simulation_vectorFields(
            x_lim=x_lim,
            y_lim=y_lim,
            n_resolution=100,
            # n_resolution=20,
            obstacle_list=obstacle_avoider.obstacle_environment,
            # obstacle_list=[],
            saveFigure=False,
            noTicks=True,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=obstacle_avoider.initial_dynamics.evaluate,
            obs_avoidance_func=obstacle_avoider.avoid,
            automatic_reference_point=False,
            pos_attractor=initial_dynamics.attractor_position,
            fig_and_ax_handle=(fig, ax),
            # Quiver or Streamplot
            show_streamplot=True,
            # show_streamplot=False,
        )

        if save_figure:
            fig_name = "linear_dynamics_in_repulsive_ellipse_wall"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)


def quiver_single_circle_linear_repulsive(visualize=False, save_figure=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2, 2]),
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([2.5, 0]))
    obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)

    main_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
        convergence_radius=math.pi,
    )

    if visualize:
        # Arbitrary constant velocity
        tmp_dynamics = LinearSystem(attractor_position=np.array([2.0, 0]))
        tmp_dynamics.distance_decrease = 0.1
        obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)
        # ConvergingDynamics=ConstantValue (initial_velocity)
        tmp_avoider = RotationalAvoider(
            initial_dynamics=initial_dynamics,
            obstacle_environment=obstacle_list,
            convergence_radius=math.pi,
        )
        x_lim = [-2, 3]
        y_lim = [-2.2, 2.2]
        n_grid = 13
        alpha_obstacle = 1.0

        plt.close("all")

        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacle_dynamics(
            obstacle_container=obstacle_list,
            dynamics=tmp_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            attractor_position=initial_dynamics.attractor_position,
            do_quiver=True,
            show_ticks=False,
        )

        plot_obstacles(
            obstacle_container=obstacle_list,
            ax=ax,
            alpha_obstacle=alpha_obstacle,
        )

        if save_figure:
            fig_name = "circular_repulsion_pi"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)

        tmp_avoider.convergence_radius = math.pi * 3.0 / 4.0
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacle_dynamics(
            obstacle_container=obstacle_list,
            dynamics=tmp_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            attractor_position=initial_dynamics.attractor_position,
            do_quiver=True,
            show_ticks=False,
        )

        plot_obstacles(
            obstacle_container=obstacle_list,
            ax=ax,
            alpha_obstacle=alpha_obstacle,
        )

        if save_figure:
            fig_name = "circular_repulsion_pi_3_4"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)

        tmp_avoider.convergence_radius = math.pi * 1.0 / 2.0
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacle_dynamics(
            obstacle_container=obstacle_list,
            dynamics=tmp_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            n_grid=n_grid,
            ax=ax,
            attractor_position=initial_dynamics.attractor_position,
            do_quiver=True,
            show_ticks=False,
        )

        plot_obstacles(
            obstacle_container=obstacle_list,
            ax=ax,
            alpha_obstacle=alpha_obstacle,
        )

        if save_figure:
            fig_name = "circular_repulsion_pi_1_2"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)


def integration_smoothness_around_ellipse(visualize=False, save_figure=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2, 2]),
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([3.5, 0]))
    obstacle_list.set_convergence_directions(converging_dynamics=initial_dynamics)

    main_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
        # convergence_radius=math.pi/2,
        tail_rotation=False,
    )

    if visualize:
        plt.close("all")
        x_lim = [-3, 3.8]
        y_lim = [-3.2, 3.3]

        it_max = 100

        n_grid = 10
        points_bottom = np.vstack(
            (np.linspace(x_lim[0], x_lim[1], n_grid), y_lim[0] * np.ones(n_grid))
        )

        points_left = np.vstack(
            (x_lim[0] * np.ones(n_grid), np.linspace(y_lim[1], y_lim[0], n_grid))
        )

        points_top = np.vstack(
            (np.linspace(x_lim[0], x_lim[1], n_grid), y_lim[1] * np.ones(n_grid))
        )

        all_points = np.hstack((points_bottom, points_left, points_top))
        is_colliding = (
            lambda x: main_avoider.obstacle_environment.get_minimum_gamma(x) < 1
        )

        # smoothness = ["00"]
        smoothness = ["00", "03", "10"]
        for ss_string in smoothness:
            # ss =
            ss = float(ss_string) / 10.0
            if not ss:
                ss = 1e-5
            main_avoider.smooth_continuation_power = ss

            fig, ax = plt.subplots(figsize=(3.5, 3))

            for pp in range(all_points.shape[1]):
                points = function_integrator(
                    start_point=all_points[:, pp],
                    dynamics=main_avoider.evaluate,
                    stopping_functor=is_colliding,
                    stepsize=0.03,
                    err_abs=1e-1,
                    it_max=200,
                )

                ax.plot(points[0, :], points[1, :], color="blue")

            point_convergence = np.array([x_lim[0], 0.0])
            points = function_integrator(
                start_point=point_convergence,
                dynamics=main_avoider.evaluate,
                stopping_functor=is_colliding,
                stepsize=0.03,
                err_abs=1e-1,
                it_max=200,
            )
            ax.plot(points[0, :], points[1, :], color="red", linewidth=2)

            plot_obstacles(
                obstacle_container=obstacle_list,
                ax=ax,
                alpha_obstacle=1.0,
                x_lim=x_lim,
                y_lim=y_lim,
            )

            ax.plot(
                initial_dynamics.attractor_position[0],
                initial_dynamics.attractor_position[1],
                "k*",
                linewidth=18.0,
                markersize=18,
                zorder=5,
            )

            ax.tick_params(
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )

            if save_figure:
                fig_name = "comparison_linear_integration_" + ss_string
                fig.savefig(
                    "figures/" + fig_name + figtype, bbox_inches="tight", dpi=300
                )


if (__name__) == "__main__":
    figtype = ".pdf"
    # figtype = ".png"

    plt.ion()

    # inverted_star_obstacle_avoidance(visualize=True, save_figure=True)
    # visualization_inverted_ellipsoid(visualize=True, save_figure=True)
    # quiver_single_circle_linear_repulsive(visualize=True, save_figure=False)
    integration_smoothness_around_ellipse(visualize=True, save_figure=True)

    print("--- done ---")
