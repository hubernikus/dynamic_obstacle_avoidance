#!/USSR/bin/python3
""" Script to show lab environment on computer """
# Author: Lukas Huber
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

# plt.close('all')
plt.ion()


def starshape():
    obs_list = RotationContainer()

    obs_list.append(
        StarshapedFlower(
            center_position=np.array([0, 0]),
            radius_magnitude=2,
            radius_mean=4,
            orientation=0.0 / 180 * pi,
            tail_effect=False,
        )
    )
    return obs_list


def starshape_hull():
    obs_list = RotationContainer()

    obs_list.append(
        StarshapedFlower(
            center_position=np.array([0, 0]),
            radius_magnitude=2,
            radius_mean=7,
            orientation=0.0 / 180 * pi,
            tail_effect=False,
            is_boundary=True,
        )
    )
    return obs_list


def single_ellipse_hull():
    obs_list = RotationContainer()

    obs_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([6, 9]),
            orientation=0.0 / 180 * pi,
            is_boundary=True,
            tail_effect=False,
        )
    )
    return obs_list


def multiple_ellipse_hulls():
    obs_list = MultiBoundaryContainer()

    obs_list.append(
        Ellipse(
            center_position=np.array([6, 0]),
            axes_length=np.array([5, 2]),
            orientation=50.0 / 180 * pi,
            is_boundary=True,
        ),
        parent=-1,
    )
    obs_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([5, 2]),
            orientation=-50.0 / 180 * pi,
            is_boundary=True,
        ),
        parent=-1,
    )
    obs_list.append(
        Ellipse(
            center_position=np.array([-6, 0]),
            axes_length=np.array([5, 2]),
            orientation=50.0 / 180 * pi,
            is_boundary=True,
        ),
        parent=-1,
    )

    return obs_list


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
    # my_avoider.smooth_continuation_power = 0.1
    my_avoider.smooth_continuation_power = 0.3

    # my_plotter.positions = np.linspace([-3.32, 6.00], [-2.0, 6.00], 30).T
    my_plotter.plot(
        # lambda x: obstacle_list[0].get_normal_direction(x, in_global_frame=True),
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    # if True:
    # return

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


def single_ellipse_linear_triple_integration_lines(
    save_figure=False, it_max=100, dt_step=0.1
):
    figure_name = "comparison_linear_integration"

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
    my_plotter.it_max = it_max

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    my_plotter.plot_streamlines()

    my_plotter.plot(
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
    )

    if True:
        return

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


def single_ellipse_spiral_analysis(save_figure=False, n_resolution=40):
    x_lim = [-10, 10]
    y_lim = [-10, 10]

    InitialSystem = LinearSystem(
        attractor_position=np.array([0, -5]),
        A_matrix=np.array([[-1.0, -3.0], [3.0, -1.0]]),
    )

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # axs = [None, None, ax]

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
    obstacle_list.set_convergence_directions(InitialSystem)

    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=InitialSystem.evaluate,
        obs_avoidance_func=obstacle_avoidance_rotational,
        automatic_reference_point=False,
        pos_attractor=InitialSystem.attractor_position,
        fig_and_ax_handle=(fig, axs[2]),
        # Quiver or Streamplot
        show_streamplot=True,
        # show_streamplot=False,
    )

    plot_dynamical_system_streamplot(
        dynamicaL_system=obstacle_list._ConvergenceDynamics[0],
        x_lim=x_lim,
        y_lim=y_lim,
        axes_equal=True,
        fig_ax_handle=(fig, axs[1]),
        n_resolution=n_resolution,
    )

    plot_obstacles(
        ax=axs[1],
        obstacle_container=obstacle_list,
        x_range=x_lim,
        y_range=y_lim,
        noTicks=True,
        showLabel=False,
        alpha_obstacle=0.4,
    )

    obstacle_list = []
    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=InitialSystem.evaluate,
        obs_avoidance_func=obstacle_avoidance_rotational,
        automatic_reference_point=False,
        pos_attractor=InitialSystem.attractor_position,
        fig_and_ax_handle=(fig, axs[0]),
        # Quiver or Streamplot
        show_streamplot=True,
        # show_streamplot=False,
    )

    if save_figure:
        figure_name = "spiral_single_ellipse_analysis"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def single_ellipse_multiattractor_triple_plot(save_figure=False, n_resolution=40):
    x_lim = [-10, 10]
    y_lim = [-6, 14]

    InitialSystem = BifurcationSpiral(maximum_velocity=1)

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([1, 6]),
            axes_length=np.array([2, 5]),
            orientation=80.0 / 180.0 * pi,
        )
    )
    obstacle_list.set_convergence_directions(InitialSystem)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # axs = [None, None, ax]

    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=InitialSystem.evaluate,
        obs_avoidance_func=obstacle_avoidance_rotational,
        automatic_reference_point=False,
        pos_attractor=InitialSystem.attractor_position,
        fig_and_ax_handle=(fig, axs[2]),
        # Quiver or Streamplot
        show_streamplot=True,
        # show_streamplot=False,
    )

    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=InitialSystem.evaluate,
        # obs_avoidance_func=obstacle_avoidance_rotational,
        automatic_reference_point=False,
        pos_attractor=InitialSystem.attractor_position,
        fig_and_ax_handle=(fig, axs[1]),
        # Quiver or Streamplot
        show_streamplot=True,
        # show_streamplot=False,
    )

    obstacle_list = []
    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=InitialSystem.evaluate,
        obs_avoidance_func=obstacle_avoidance_rotational,
        automatic_reference_point=False,
        pos_attractor=InitialSystem.attractor_position,
        fig_and_ax_handle=(fig, axs[0]),
        # Quiver or Streamplot
        show_streamplot=True,
        # show_streamplot=False,
    )

    if save_figure:
        figure_name = "multiattractor_single_ellipse"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def single_ellipse_multiattractor_analysis(save_figure=False, n_resolution=40):
    x_lim = [-10, 10]
    y_lim = [-10, 10]

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # axs = [None, None, ax]
    InitialSystem = BifurcationSpiral(maximum_velocity=1)

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([1, 3]),
            axes_length=np.array([2, 5]),
            orientation=80.0 / 180.0 * pi,
        )
    )
    obstacle_list.set_convergence_directions(InitialSystem)

    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=InitialSystem.evaluate,
        obs_avoidance_func=obstacle_avoidance_rotational,
        automatic_reference_point=False,
        pos_attractor=InitialSystem.attractor_position,
        fig_and_ax_handle=(fig, axs[2]),
        # Quiver or Streamplot
        show_streamplot=True,
        # show_streamplot=False,
    )

    plot_dynamical_system_streamplot(
        dynamical_system=obstacle_list._ConvergenceDynamics[0],
        x_lim=x_lim,
        y_lim=y_lim,
        axes_equal=True,
        fig_ax_handle=(fig, axs[1]),
        n_resolution=n_resolution,
    )

    plot_obstacles(
        ax=axs[1],
        obstacle_container=obstacle_list,
        x_range=x_lim,
        y_range=y_lim,
        noTicks=True,
        showLabel=False,
        alpha_obstacle=0.4,
    )

    obstacle_list = []
    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=InitialSystem.evaluate,
        obs_avoidance_func=obstacle_avoidance_rotational,
        automatic_reference_point=False,
        pos_attractor=InitialSystem.attractor_position,
        fig_and_ax_handle=(fig, axs[0]),
        # Quiver or Streamplot
        show_streamplot=True,
        # show_streamplot=False,
    )

    if save_figure:
        figure_name = "spiral_multiattractor_analysis"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def single_ellipse_hull_linear_triple_plot(save_figure=False, n_resolution=40):
    """Moving inside an 'ellipse hull with linear dynamics."""
    x_lim = [-10, 10]
    y_lim = [-10, 10]

    InitialSystem = LinearSystem(attractor_position=np.array([0, -4]))

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # axs = [None, None, ax]

    obstacle_list = single_ellipse_hull()
    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=InitialSystem.evaluate,
        obs_avoidance_func=obstacle_avoidance_rotational,
        automatic_reference_point=False,
        pos_attractor=InitialSystem.attractor_position,
        fig_and_ax_handle=(fig, axs[2]),
        # Quiver or Streamplot
        show_streamplot=True,
        # show_streamplot=False,
    )

    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=InitialSystem.evaluate,
        pos_attractor=InitialSystem.attractor_position,
        automatic_reference_point=False,
        fig_and_ax_handle=(fig, axs[1]),
    )

    obstacle_list = []
    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=InitialSystem.evaluate,
        pos_attractor=InitialSystem.attractor_position,
        automatic_reference_point=False,
        fig_and_ax_handle=(fig, axs[0]),
    )

    if save_figure:
        figure_name = "comparison_linear_hull"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def starshape_hull_linear_triple_plot(save_figure=False, n_resolution=20):
    """Moving inside an 'ellipse hull with linear dynamics."""
    x_lim = [-10, 10]
    y_lim = [-10, 10]

    pos_attractor = np.array([0, -6])

    def initial_ds(position):
        return evaluate_linear_dynamical_system(position, center_position=pos_attractor)

    def obs_avoidance(*args, **kwargs):
        def get_convergence_direction(position):
            return evaluate_linear_dynamical_system(
                position, center_position=pos_attractor
            )

        return obstacle_avoidance_rotational(
            *args, **kwargs, get_convergence_direction=get_convergence_direction
        )

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # axs = [None, None, ax]

    obstacle_list = starshape_hull()
    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=initial_ds,
        obs_avoidance_func=obs_avoidance,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[2]),
        # Quiver or Streamplot
        show_streamplot=True,
        # show_streamplot=False,
    )
    # if True:
    # return

    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=initial_ds,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[1]),
    )

    obstacle_list = []
    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=initial_ds,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[0]),
    )

    if save_figure:
        figure_name = "comparison_starshape_hull"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def starshape_linear_triple_plot(save_figure=False, n_resolution=20):
    """Moving inside an 'ellipse hull with linear dynamics."""
    x_lim = [-10, 10]
    y_lim = [-10, 10]

    pos_attractor = np.array([4, -4])

    def initial_ds(position):
        return evaluate_linear_dynamical_system(position, center_position=pos_attractor)

    def obs_avoidance(*args, **kwargs):
        def get_convergence_direction(position):
            return evaluate_linear_dynamical_system(
                position, center_position=pos_attractor
            )

        return obstacle_avoidance_rotational(
            *args, **kwargs, get_convergence_direction=get_convergence_direction
        )

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # axs = [None, None, ax]

    obstacle_list = starshape()
    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=initial_ds,
        obs_avoidance_func=obs_avoidance,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[2]),
        # Quiver or Streamplot
        show_streamplot=True,
        # show_streamplot=False,
    )
    # if True:
    # return

    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=initial_ds,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[1]),
    )

    obstacle_list = []
    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=initial_ds,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[0]),
    )

    if save_figure:
        figure_name = "comparison_starshape_linear"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def multiple_hull_empty(save_figure=False):
    x_lim = [-10, 10]
    y_lim = [-10, 10]

    pos_attractor = np.array([9, 3])

    obstacle_list = multiple_ellipse_hulls()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    Simulation_vectorFields(
        x_lim,
        y_lim,
        0,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=False,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, ax),
        show_streamplot=False,
        # show_streamplot=True,
    )

    if save_figure:
        figure_name = "multiple_hull_empty"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def multiple_hull_linear(save_figure=False, n_resolution=4):
    """Multiple ellipse hull."""
    x_lim = [-10, 10]
    y_lim = [-10, 10]

    pos_attractor = np.array([9, 3])

    def initial_ds(position):
        return evaluate_linear_dynamical_system(position, center_position=pos_attractor)

    def obs_avoidance(*args, **kwargs):
        def get_convergence_direction(position):
            return evaluate_linear_dynamical_system(
                position, center_position=pos_attractor
            )

        return obstacle_avoidance_rotational(
            *args, **kwargs, get_convergence_direction=get_convergence_direction
        )

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    obstacle_list = multiple_ellipse_hulls()
    # obstacle_list.update_intersection_graph(attractor_position=pos_attractor)
    # obstacle_list.update_intersection_graph(attractor_position=pos_attractor)
    obstacle_list.update_intersection_graph(attractor_position=pos_attractor)

    Simulation_vectorFields(
        x_lim,
        y_lim,
        n_resolution,
        obstacle_list,
        saveFigure=False,
        noTicks=True,
        showLabel=False,
        draw_vectorField=True,
        dynamical_system=initial_ds,
        obs_avoidance_func=obs_avoidance,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, ax),
        show_streamplot=True,
        # show_streamplot=True,
    )

    obstacle_list.plot_convergence_attractor(ax=ax, attractor_position=pos_attractor)

    if save_figure:
        figure_name = "multiple_hull_linear_streamplot"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # single_ellipse_linear_triple_plot_quiver(save_figure=True, n_resolution=15)
    single_ellipse_linear_triple_integration_lines(save_figure=False)
    # single_ellipse_nonlinear_triple_plot(save_figure=True, n_resolution=40)

    # single_ellipse_linear_triple_plot(save_figure=True, n_resolution=15)

    # single_ellipse_spiral_triple_plot(save_figure=False, n_resolution=30)
    # single_ellipse_spiral_analysis(save_figure=True, n_resolution=30)

    # TODO: analyse this local-convergence better / what velocity is needed.
    # What do we need to know about the field?
    # single_ellipse_multiattractor_triple_plot(save_figure=True, n_resolution=100)
    # single_ellipse_multiattractor_analysis(save_figure=False, n_resolution=100)
    # single_ellipse_hull_linear_triple_plot(save_figure=True, n_resolution=100)
    # single_ellipse_hull_nonlinear_triple_plot(save_figure=True, n_resolution=100)
    # starshape_hull_linear_triple_plot(save_figure=True, n_resolution=100)

    # starshape_linear_triple_plot(save_figure=False, n_resolution=100)
    # starshape_hull_linear_triple_plot(save_figure=False, n_resolution=100)

    # multiple_hull_linear(save_figure=True, n_resolution=200)

    # multiple_hull_empty(save_figure=True)
    # single_ellipse_hull(save_figure=True)

    pass
