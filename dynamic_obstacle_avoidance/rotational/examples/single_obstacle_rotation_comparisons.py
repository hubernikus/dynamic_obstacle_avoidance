#!/USSR/bin/python3
""" Script to show lab environment on computer """
# Author: Lukas Huber
# Github: hubernikus
# License: BSD (c) 2021

import warnings
import copy

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

from vartools.dynamical_systems import LinearSystem, QuadraticAxisConvergence
from vartools.dynamical_systems import BifurcationSpiral
from vartools.dynamical_systems import plot_dynamical_system_streamplot

from dynamic_obstacle_avoidance.obstacles import Ellipse, StarshapedFlower
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

from dynamic_obstacle_avoidance.rotational.multiboundary_container import (
    MultiBoundaryContainer,
)
from dynamic_obstacle_avoidance.rotational.rotation_container import RotationContainer
from dynamic_obstacle_avoidance.rotational.rotation import obstacle_avoidance_rotational
from dynamic_obstacle_avoidance.rotational.rotational_avoider import RotationalAvoider


from dynamic_obstacle_avoidance.visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)

from vartools.visualization import VectorfieldPlotter


def single_ellipse_linear_triple_plot_quiver(
    n_resolution=100, save_figure=False, show_streamplot=True
):
    figure_name = "comparison_linear_"

    initial_dynamics = LinearSystem(attractor_position=np.array([8, 0]))

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2.5, 5]),
            orientation=0.0 / 180 * pi,
            is_boundary=False,
            tail_effect=False,
        )
    )
    obstacle_list.set_convergence_directions(initial_dynamics)

    my_plotter = VectorfieldPlotter(
        y_lim=[-10, 10],
        x_lim=[-10, 10],
        # figsize=(10.0, 8.0),
        figsize=(4.0, 3.5),
        attractor_position=initial_dynamics.attractor_position,
    )

    my_plotter.plottype = "quiver"
    my_plotter.obstacle_alpha = 1

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )
    my_avoider.smooth_continuation_power = 0.3

    my_plotter.plot(
        # lambda x: obstacle_list[0].get_normal_direction(x, in_global_frame=True),
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_rotated")

    my_avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    my_plotter.create_new_figure()
    my_plotter.plot(
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_modulated")

    my_plotter.create_new_figure()
    my_plotter.plot(
        initial_dynamics.evaluate,
        obstacle_list=None,
        check_functor=None,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_initial")


def rotated_ellipse_linear_triple_plot_quiver(
    n_resolution=100, save_figure=False, show_streamplot=True
):
    figure_name = "comparison_rotated_"

    initial_dynamics = LinearSystem(attractor_position=np.array([8, 0]))

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2.5, 5]),
            orientation=30.0 / 180 * pi,
            is_boundary=False,
            tail_effect=False,
        )
    )
    obstacle_list.set_convergence_directions(initial_dynamics)

    my_plotter = VectorfieldPlotter(
        y_lim=[-10, 10],
        x_lim=[-10, 10],
        # figsize=(10.0, 8.0),
        figsize=(4.0, 3.5),
        attractor_position=initial_dynamics.attractor_position,
    )

    my_plotter.plottype = "quiver"
    my_plotter.obstacle_alpha = 1

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )
    my_avoider.smooth_continuation_power = 0.3

    my_plotter.plot(
        # lambda x: obstacle_list[0].get_normal_direction(x, in_global_frame=True),
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_rotated")

    my_avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    my_plotter.create_new_figure()
    my_plotter.plot(
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_modulated")

    my_plotter.create_new_figure()
    my_plotter.plot(
        initial_dynamics.evaluate,
        obstacle_list=None,
        check_functor=None,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_initial")


def single_ellipse_nonlinear_triple_plot(n_resolution=100, save_figure=False):
    figure_name = "comparison_nonlinear_vectorfield"

    initial_dynamics = QuadraticAxisConvergence(
        stretching_factor=3,
        maximum_velocity=1.0,
        dimension=2,
        attractor_position=np.array([8, 0]),
    )

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([3, 3]),
            orientation=0.0 / 180 * pi,
            is_boundary=False,
            tail_effect=False,
        )
    )
    obstacle_list.set_convergence_directions(initial_dynamics)

    my_plotter = VectorfieldPlotter(
        x_lim=[-10, 10],
        y_lim=[-10, 10],
        figsize=(4.5, 4.0),
        attractor_position=initial_dynamics.attractor_position,
    )

    my_plotter.obstacle_alpha = 1

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    my_plotter.plot(
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_rotated")

    my_avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    my_plotter.create_new_figure()
    my_plotter.plot(
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_modulated")

    my_plotter.create_new_figure()
    my_plotter.plot(
        initial_dynamics.evaluate,
        obstacle_list=None,
        check_functor=None,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_initial")


def single_ellipse_linear_triple_plot_streampline(
    n_resolution=100, save_figure=False, show_streamplot=True
):
    figure_name = "comparison_linear_streamline"

    initial_dynamics = LinearSystem(attractor_position=np.array([8, 0]))

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2.5, 5]),
            orientation=0.0 / 180 * pi,
            is_boundary=False,
            tail_effect=False,
        )
    )
    obstacle_list.set_convergence_directions(initial_dynamics)

    my_plotter = VectorfieldPlotter(
        y_lim=[-10, 10],
        x_lim=[-10, 10],
        # figsize=(10.0, 8.0),
        figsize=(4.0, 3.5),
        attractor_position=initial_dynamics.attractor_position,
    )

    my_plotter.obstacle_alpha = 1

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )
    my_avoider.smooth_continuation_power = 0.3
    my_plotter.plot(
        # lambda x: obstacle_list[0].get_normal_direction(x, in_global_frame=True),
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_03")

    # New one
    my_avoider.smooth_continuation_power = 0.0
    my_plotter.create_new_figure()
    my_plotter.plot(
        # lambda x: obstacle_list[0].get_normal_direction(x, in_global_frame=True),
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_00")

    # New one
    my_avoider.smooth_continuation_power = 1.0
    my_plotter.create_new_figure()
    my_plotter.plot(
        # lambda x: obstacle_list[0].get_normal_direction(x, in_global_frame=True),
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_10")


def single_ellipse_linear_triple_integration_lines(
    save_figure=False, it_max=300, dt_step=0.02
):
    figure_name = "comparison_linear_integration"

    # initial_dynamics = QuadraticAxisConvergence(
    # stretching_factor=3,
    # maximum_velocity=1.0,
    # dimension=2,
    # attractor_position=np.array([8, 0]),
    # )
    initial_dynamics = LinearSystem(attractor_position=np.array([8, 0]))

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([3, 3]),
            orientation=0.0 / 180 * pi,
            is_boundary=False,
            tail_effect=False,
        )
    )
    obstacle_list.set_convergence_directions(initial_dynamics)

    x_lim = [-10, 10]
    y_lim = [-10, 10]

    my_plotter = VectorfieldPlotter(
        x_lim=x_lim,
        y_lim=y_lim,
        figsize=(4.5, 4.0),
        attractor_position=initial_dynamics.attractor_position,
    )
    my_plotter.obstacle_alpha = 1
    my_plotter.dt_step = dt_step
    my_plotter.it_max = it_max

    initial_positions = np.vstack(
        (
            np.linspace([x_lim[0], y_lim[0]], [x_lim[1], y_lim[0]], 10),
            np.linspace([x_lim[0], y_lim[0]], [x_lim[0], y_lim[1]], 10),
            np.linspace([x_lim[0], y_lim[1]], [x_lim[1], y_lim[1]], 10),
        )
    ).T

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    # New continuation power value
    my_avoider.smooth_continuation_power = 0.3
    my_plotter.plot_streamlines(
        initial_positions,
        my_avoider.evaluate,
        collision_functor=obstacle_list.has_collided,
        convergence_functor=initial_dynamics.has_converged,
        obstacle_list=obstacle_list,
    )

    if save_figure:
        my_plotter.save(figure_name + "_03")

    # if True:
    # return

    # New continuation power value
    my_avoider.smooth_continuation_power = 0.0

    my_plotter.create_new_figure()
    my_plotter.plot_streamlines(
        initial_positions,
        my_avoider.evaluate,
        collision_functor=obstacle_list.has_collided,
        convergence_functor=initial_dynamics.has_converged,
        obstacle_list=obstacle_list,
    )
    if save_figure:
        my_plotter.save(figure_name + "_00")

    # New continuation power value
    my_avoider.smooth_continuation_power = 1.0

    my_plotter.create_new_figure()
    my_plotter.plot_streamlines(
        initial_positions,
        my_avoider.evaluate,
        collision_functor=obstacle_list.has_collided,
        convergence_functor=initial_dynamics.has_converged,
        obstacle_list=obstacle_list,
    )
    if save_figure:
        my_plotter.save(figure_name + "_10")


def single_ellipse_spiral_triple_plot(save_figure=False, n_resolution=40):
    # TODO: this does not work very well...
    figure_name = "spiral_single_ellipse"

    initial_dynamics = LinearSystem(
        attractor_position=np.array([0, -5]),
        A_matrix=np.array([[-1.0, -3.0], [3.0, -1.0]]),
    )

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([3, 3]),
            orientation=30.0 / 180 * pi,
            is_boundary=False,
            tail_effect=False,
        )
    )
    obstacle_list.set_convergence_directions(initial_dynamics)
    my_plotter = VectorfieldPlotter(
        x_lim=[-10, 10],
        y_lim=[-10, 10],
        figsize=(4.5, 4.0),
        attractor_position=initial_dynamics.attractor_position,
    )

    my_plotter.obstacle_alpha = 1

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    my_plotter.plot(
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_rotated")

    my_avoider = ModulationAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    my_plotter.create_new_figure()
    my_plotter.plot(
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_modulated")

    my_plotter.create_new_figure()
    my_plotter.plot(
        initial_dynamics.evaluate,
        obstacle_list=None,
        check_functor=None,
        n_resolution=n_resolution,
    )
    if save_figure:
        my_plotter.save(figure_name + "_initial")


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # single_ellipse_linear_triple_plot_quiver(save_figure=True, n_resolution=15)
    # single_ellipse_linear_triple_integration_lines(save_figure=False)
    # single_ellipse_linear_triple_plot_streampline(save_figure=False, n_resolution=30)
    # single_ellipse_nonlinear_triple_plot(save_figure=True, n_resolution=40)

    # rotated_ellipse_linear_triple_plot_quiver(save_figure=False, n_resolution=30)

    pass
