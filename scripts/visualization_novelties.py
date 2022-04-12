# !/USSR/bin/python3
"""
Script which creates a variety of examples of local modulation of a vector field with obstacle avoidance. 
"""
# Author:  Lukas Huber
# Email: lukas.huber@epfl.ch
# Created: 2018-02-15

import sys
import os
import warnings

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacles import (
    Ellipse,
    Polygon,
    StarshapedFlower,
    Cuboid,
)
from dynamic_obstacle_avoidance.containers import GradientContainer


def visualize_simple_ellipse(n_resolution=10, save_figure=True):
    obs = GradientContainer()  # create empty obstacle list
    x_lim = [-0.6, 4.1]
    y_lim = [-2.1, 2.1]

    dynamical_system = LinearSystem(attractor_position=[0.0, 0.0])

    figsize = (6, 5)

    obs.append(
        Ellipse(
            axes_length=[0.5, 0.5],
            center_position=[2.0, 0.0],
            p=[1, 1],
            orientation=0.0 / 180 * pi,
            margin_absolut=0.4,
            is_boundary=False,
            has_sticky_surface=False,
            reactivity=1,
            tail_effect=True,
        )
    )

    fig = plt.figure(figsize=figsize)
    ax = plt.subplots(2, 1, 1)
    plt_speed_line_and_qolo(
        points_init=np.array([3.5, 0.2]),
        attractorPos=dynamical_system.attractor_position,
        obs=obs,
        fig_and_ax_handle=(fig, ax),
    )

    fig_mod, ax_mod = Simulation_vectorFields(
        x_lim,
        y_lim,
        obs=obs,
        xAttractor=dynamical_system.attractor_position,
        saveFigure=save_figure,
        figName="circular_sticky_surface",
        noTicks=False,
        draw_vectorField=True,
        automatic_reference_point=True,
        point_grid=n_resolution,
        show_streamplot=False,
        figureSize=figsize,
        reference_point_number=False,
        showLabel=False,
        normalize_vectors=False,
        dynamicalSystem=dynamical_system.evaluate,
        fig_and_ax_handle=(fig, ax),
    )
    # return

    obs.has_sticky_surface = False

    fig_mod, ax_mod = Simulation_vectorFields(
        x_lim,
        y_lim,
        obs=obs,
        xAttractor=xAttractor,
        saveFigure=save_figure,
        figName="circular_nonsticky_surface",
        noTicks=False,
        draw_vectorField=True,
        automatic_reference_point=True,
        point_grid=n_resolution,
        show_streamplot=False,
        figureSize=figsize,
        reference_point_number=False,
        showLabel=False,
        normalize_vectors=False,
        dynamicalSystem=dynamical_system.evaluate,
    )

    plt_speed_line_and_qolo(points_init=np.array([4.6, 0.2]), attractorPos=xAttractor)


def visualize_intersecting_ellipse(
    n_resolution=40,
    save_figures=False,
):
    obs = GradientContainer()  # create empty obstacle list
    x_lim = [-0.6, 5.1]
    y_lim = [-2.1, 2.1]

    # x_lim = [1.6, 3.1]
    # y_lim = [-2.0, -1.0]

    xAttractor = [0, 0]
    figsize = (5, 3.0)
    point_init = np.array([4.5, -0.9])

    obs = GradientContainer()
    obs.append(
        Ellipse(
            axes_length=[0.7, 0.3],
            center_position=[2.0, 0.5],
            p=[1, 1],
            orientation=20.0 / 180 * pi,
            margin_absolut=0.5,
            is_boundary=False,
        )
    )

    obs.append(
        Ellipse(
            axes_length=[0.7, 0.3],
            center_position=[2.0, -0.5],
            p=[1, 1],
            orientation=-40.0 / 180 * pi,
            margin_absolut=0.5,
            is_boundary=False,
            reactivity=1,
        )
    )

    obs.update_reference_points()

    # del obs[0]

    # gridspec_kw={"width_ratios":[1,1, 0.05]})

    # fig = plt.figure(figsize=(8, 3.0))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9.75, 3))
    # fig, axes = plt.subplots(nrows=1, ncols=1)
    # ax = axes
    # ax = plt.subplot(1, 2, 1)

    for oo in range(len(obs)):
        obs[oo].has_sticky_surface = False

    # fig, ax = plt.subplots(figsize=figsize)
    # ax = plt.subplot(1, 2, 2)

    max_value_magnitude = 0.2

    ax = axes.flat[0]
    line = plt_speed_line_and_qolo(
        points_init=point_init,
        attractorPos=xAttractor,
        obs=obs,
        fig_and_ax_handle=(fig, ax),
        dt=0.02,
        normalize_magnitude=False,
        max_value=max_value_magnitude,
    )

    # Why does 'automatic reference point' create a mess?
    fig_mod, ax_mod = Simulation_vectorFields(
        x_lim,
        y_lim,
        obs=obs,
        xAttractor=xAttractor,
        saveFigure=False,
        figName="intersecting_ellipses_nonsticky_surfaces",
        noTicks=False,
        draw_vectorField=True,
        automatic_reference_point=False,
        point_grid=n_resolution,
        show_streamplot=False,
        normalize_vectors=False,
        dynamicalSystem=linearAttractor_const,
        figureSize=figsize,
        reference_point_number=False,
        showLabel=False,
        fig_and_ax_handle=(fig, ax),
    )
    ax.set_xticks([])
    ax.set_yticks([])

    for oo in range(len(obs)):
        obs[oo].has_sticky_surface = True

    ax = axes.flat[1]
    plt_speed_line_and_qolo(
        points_init=point_init,
        attractorPos=xAttractor,
        obs=obs,
        fig_and_ax_handle=(fig, ax),
        dt=0.02,
        normalize_magnitude=False,
        max_value=max_value_magnitude,
    )

    fig_mod, ax_mod = Simulation_vectorFields(
        x_lim,
        y_lim,
        obs=obs,
        xAttractor=xAttractor,
        saveFigure=False,
        figName="intersecting_ellipses_sticky_surfaces",
        noTicks=False,
        draw_vectorField=True,
        automatic_reference_point=False,
        point_grid=n_resolution,
        show_streamplot=False,
        normalize_vectors=False,
        dynamicalSystem=linearAttractor_const,
        figureSize=figsize,
        reference_point_number=False,
        showLabel=False,
        fig_and_ax_handle=(fig, ax),
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # ax.axes.yaxis.set_ticklabels([])
    # ip = InsetPosition(ax2, [1.05, 0, 0.05, 1])
    # ip = InsetPosition(ax2, [1.05, -1, 0.05, 2.0])
    # cax.set_axes_locator(ip)
    # fig.colorbar(line, cax=cax, ax=[ax1, ax2])
    # fig.colorbar(line, cax=cax)
    fig.colorbar(line, ax=axes.ravel().tolist(), shrink=0.9)

    if save_figures:
        figName = "comparison_speed_modualtion"
        plt.savefig("../figures/" + figName + ".png", bbox_inches="tight")


def visualize_repulsive_cube(
    robot_margin=0.35,
    n_resolution=20,
    save_figure=False,
    point_init=np.array([4.5, 0.2]),
):
    x_lim = [-2, 6]
    y_lim = [-2, 2]

    figsize = (10, 10.0)

    pos_attractor = np.array([-1, 0])
    dynamics = LinearSystem(attractor_position=pos_attractor)

    obs = GradientContainer()  # create empty obstacle list
    # obs.append(Polygon(
    # edge_points=[[-1.8, 5.4, 5.4, -1.8],
    # [-1.8,-1.8, 1.8, 1.8]],
    # center_position=[3.0, -1],
    # orientation=0./180*pi,
    # margin_absolut=robot_margin,
    # is_boundary=True,
    # tail_effect=False,
    # ))

    obs.append(
        Cuboid(
            axes_length=[0.4, 0.4],
            center_position=[2.0, 0.0],
            orientation=0.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=False,
            repulsion_coeff=1.0,
            tail_effect=False,
            name="center_cube",
        )
    )

    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    # ax = axes.flat[1]
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(2, 1, 1)
    line = plt_speed_line_and_qolo(
        points_init=point_init,
        attractorPos=pos_attractor,
        obs=obs,
        fig_and_ax_handle=(fig, ax),
        dt=0.02,
        line_color=[102.0 / 255, 204.0 / 255, 0.0 / 255],
    )

    Simulation_vectorFields(
        x_lim,
        y_lim,
        obs=obs,
        xAttractor=pos_attractor,
        saveFigure=False,
        figName=None,
        noTicks=True,
        draw_vectorField=True,
        automatic_reference_point=False,
        point_grid=n_resolution,
        show_streamplot=True,
        normalize_vectors=False,
        dynamicalSystem=dynamics.evaluate,
        figureSize=figsize,
        reference_point_number=False,
        showLabel=False,
        fig_and_ax_handle=(fig, ax),
    )

    obs["center_cube"].repulsion_coeff = 3.0

    # fig, ax = plt.subplots(figsize=figsize)
    # ax = axes.flat[1]
    ax = plt.subplot(2, 1, 2)
    line = plt_speed_line_and_qolo(
        points_init=point_init,
        attractorPos=pos_attractor,
        obs=obs,
        fig_and_ax_handle=(fig, ax),
        dt=0.02,
        line_color=[102.0 / 255, 204.0 / 255, 0.0 / 255],
    )

    Simulation_vectorFields(
        x_lim,
        y_lim,
        obs=obs,
        xAttractor=pos_attractor,
        saveFigure=False,
        figName=None,
        noTicks=True,
        draw_vectorField=True,
        automatic_reference_point=False,
        point_grid=n_resolution,
        show_streamplot=True,
        normalize_vectors=False,
        dynamicalSystem=dynamics.evaluate,
        figureSize=figsize,
        reference_point_number=False,
        showLabel=False,
        fig_and_ax_handle=(fig, ax),
    )

    if save_figure:
        figName = "nonrepulsive_cube_in_hallway"
        plt.savefig("../figures/" + figName + ".png", bbox_inches="tight")
    # import pdb; pdb.set_trace()


def visualize_circular_boundary(
    robot_margin=0.35,
    n_resolution=20,
    save_figure=False,
    point_init=np.array([3.2, -2.2]),
):
    x_lim = [-2, 6]
    y_lim = [-3.5, 3.5]

    figsize = (5.5, 4.5)
    pos_attractor = [0, 0]

    obs = GradientContainer()  # create empty obstacle list

    obs.append(
        Ellipse(
            axes_length=[2.9, 3.5],
            center_position=[2.0, 0.0],
            orientation=40.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=True,
            repulsion_coeff=1.0,
            tail_effect=False,
            # name="center_cube",
        )
    )

    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize)
    # ax = axes.flat[1]
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)

    line = plt_speed_line_and_qolo(
        points_init=point_init,
        attractorPos=pos_attractor,
        obs=obs,
        fig_and_ax_handle=(fig, ax),
        dt=0.02,
        line_color=[102.0 / 255, 204.0 / 255, 0.0 / 255],
    )

    Simulation_vectorFields(
        x_lim,
        y_lim,
        obs=obs,
        xAttractor=pos_attractor,
        saveFigure=save_figure,
        figName="linearSystem_boundaryEllipse",
        noTicks=True,
        draw_vectorField=True,
        automatic_reference_point=False,
        point_grid=n_resolution,
        show_streamplot=True,
        normalize_vectors=False,
        dynamicalSystem=linearAttractor_const,
        figureSize=figsize,
        reference_point_number=False,
        showLabel=False,
        fig_and_ax_handle=(fig, ax),
    )

    # if save_figure:
    # figName = "linearSystem_boundaryEllipse"
    # plt.savefig('../figures/' + figName + '.png', bbox_inches='tight')


def visualize_starshaped_boundary(
    robot_margin=0.35,
    n_resolution=20,
    save_figure=False,
    point_init=np.array([2.6, -3.0]),
):
    x_lim = [-2.1, 7.1]
    y_lim = [-4.1, 4.1]

    figsize = (5.5, 4.5)
    pos_attractor = [0, 0]

    obs = GradientContainer()  # create empty obstacle list

    obs.append(
        StarshapedFlower(
            # axes_length=[2.9, 3.5],
            radius_magnitude=1.0,
            radius_mean=3,
            number_of_edges=4,
            center_position=[2.5, 0.0],
            orientation=00.0 / 180 * pi,
            # margin_absolut=robot_margin,
            is_boundary=True,
            repulsion_coeff=1.0,
            tail_effect=False,
            # name="center_cube",
        )
    )

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1)

    line = plt_speed_line_and_qolo(
        points_init=point_init,
        attractorPos=pos_attractor,
        obs=obs,
        fig_and_ax_handle=(fig, ax),
        dt=0.02,
        line_color=[102.0 / 255, 204.0 / 255, 0.0 / 255],
    )

    Simulation_vectorFields(
        x_lim,
        y_lim,
        obs=obs,
        xAttractor=pos_attractor,
        saveFigure=save_figure,
        figName="linearSystem_starShaped",
        noTicks=True,
        draw_vectorField=True,
        automatic_reference_point=False,
        point_grid=n_resolution,
        show_streamplot=True,
        normalize_vectors=False,
        dynamicalSystem=linearAttractor_const,
        figureSize=figsize,
        reference_point_number=False,
        showLabel=False,
        fig_and_ax_handle=(fig, ax),
    )


def visualize_edge_obstacle(
    robot_margin=0.35,
    n_resolution=20,
    save_figure=False,
    point_init=np.array([0.5, -2.0]),
    dynamic_simulation=False,
    static_simulation=False,
):
    plt.close("all")

    x_lim = [-3.1, 2.1]
    y_lim = [-3.1, 2.1]

    figsize = (5.5, 4.5)
    pos_attractor = [-2.0, 1.0]

    obs = GradientContainer()  # create empty obstacle list

    obs.append(
        Polygon(
            # number_of_edges=4,
            edge_points=[[-2, 0, 0, -1, -1, -2], [-2, -2, -1, -1, 0, 0]],
            center_position=[-1.7, -1.7],
            orientation=0.0 / 180 * pi,
            # margin_absolut=robot_margin,
            # is_boundary=True,
            repulsion_coeff=1.0,
            tail_effect=False,
            name="center_cube",
        )
    )

    obs.append(
        Polygon(
            # number_of_edges=4,
            edge_points=[[-0.0, 0.5, 0.5, 0.0], [-0.0, 0.0, 0.5, 0.5]],
            # center_position=[2.5, 0.0],
            orientation=0.0 / 180 * pi,
            # margin_absolut=robot_margin,
            # is_boundary=True,
            repulsion_coeff=1.0,
            tail_effect=False,
            # name="center_cube",
        )
    )

    if static_simulation:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)

        line = plt_speed_line_and_qolo(
            points_init=point_init,
            attractorPos=pos_attractor,
            obs=obs,
            fig_and_ax_handle=(fig, ax),
            dt=0.02,
            line_color=[102.0 / 255, 204.0 / 255, 0.0 / 255],
        )

        Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            xAttractor=pos_attractor,
            saveFigure=save_figure,
            figName="edge_obstacles_several",
            noTicks=True,
            draw_vectorField=True,
            automatic_reference_point=False,
            point_grid=n_resolution,
            show_streamplot=True,
            normalize_vectors=False,
            dynamicalSystem=linearAttractor_const,
            figureSize=figsize,
            reference_point_number=False,
            showLabel=False,
            fig_and_ax_handle=(fig, ax),
        )

    elif dynamic_simulation:
        SimulationHandler = DynamicAnimationQOLO(
            obstacle_list=obs,
            attractor_position=pos_attractor,
            x_lim=x_lim,
            y_lim=y_lim,
            position_init=point_init,
        )
        SimulationHandler.run_dynamic_animation(scale_qolo=0.5)
    else:
        warnings.warn("No simulation type chosen.")


def visualize_edge_boundary(
    robot_margin=0.35,
    n_resolution=20,
    save_figure=False,
    point_init=np.array([2.5, -2.0]),
    dynamic_simulation=False,
    static_simulation=False,
):
    plt.close("all")

    x_lim = [-4.5, 4.5]
    y_lim = [-4.5, 4.5]

    figsize = (5.5, 4.5)
    dynamical_system = LinearSystem(attractor_position=[-2.0, 2.0])

    obs = GradientContainer()  # create empty obstacle list

    obs.append(
        Polygon(
            # number_of_edges=4,
            edge_points=[[-4, 4, 4, 0, 0, -4], [-3, -3, 1, 1, 3, 3]],
            center_position=[-1, 0],
            orientation=0.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=True,
            repulsion_coeff=1.0,
            tail_effect=False,
            name="center_cube",
        )
    )

    obs.append(
        Cuboid(
            # number_of_edges=4,
            axes_length=[0.8, 0.8],
            # edge_points=[[-0.0, 0.5, 0.5, 0.0],
            # [-0.0, 0.0, 0.5, 0.5]],
            center_position=[-1.0, 0.0],
            orientation=0.0 / 180 * pi,
            margin_absolut=robot_margin,
            # is_boundary=True,
            repulsion_coeff=1.0,
            tail_effect=False,
            name="center_cube",
        )
    )

    if static_simulation:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)

        line = plt_speed_line_and_qolo(
            points_init=point_init,
            attractorPos=dynamical_system.attractor_position,
            obs=obs,
            fig_and_ax_handle=(fig, ax),
            dt=0.02,
            line_color=[102.0 / 255, 204.0 / 255, 0.0 / 255],
        )

        Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            pos_attractor=dynamical_system.attractor_position,
            saveFigure=save_figure,
            figName="sharp_boundary_with_obstacle",
            noTicks=True,
            draw_vectorField=True,
            automatic_reference_point=False,
            point_grid=n_resolution,
            show_streamplot=True,
            normalize_vectors=False,
            dynamical_system=dynamical_system.evaluate,
            figureSize=figsize,
            reference_point_number=False,
            showLabel=False,
            fig_and_ax_handle=(fig, ax),
        )
    elif dynamic_simulation:

        SimulationHandler = DynamicAnimationQOLO(
            obstacle_list=obs,
            attractor_position=pos_attractor,
            x_lim=x_lim,
            y_lim=y_lim,
            position_init=point_init,
        )
        SimulationHandler.run_dynamic_animation(scale_qolo=0.8, des_speed=0.5)

    else:
        warnings.warn("No simulation type chosen")


def visualize_dynamic_boundary(
    robot_margin=0.35,
    n_resolution=20,
    save_figure=False,
    point_init=np.array([0, -3.5]),
    dynamic_simulation=False,
    static_simulation=False,
    pos_attractor=[-4.0, 0.0],
    x_lim=[-6.1, 6.1],
    y_lim=[-6.1, 6.1],
    # x_lim=[-2.0, 2.0],
    # y_lim=[-2.0, 2.0],
    figsize=(5.5, 4.5),
):

    obs_list = GradientContainer()

    def temp_func(arg):
        return

    obs_list.append(
        StarshapedFlower(
            radius_mean=4.2,
            radius_magnitude=0,
            is_boundary=True,
            is_deforming=True,
            property_functions={
                "radius_magnitude": (lambda arg: (1.0 - np.cos(arg * 1.0)) * 0.8)
            },
            # property_functions={'radius_magnitude': temp_func},
            time_now=0,
        )
    )

    plt.close("all")

    SimulationHandler = DynamicAnimationQOLO(
        obstacle_list=obs_list,
        attractor_position=pos_attractor,
        x_lim=x_lim,
        y_lim=y_lim,
        position_init=point_init,
    )

    SimulationHandler.run_dynamic_animation(
        scale_qolo=0.8,
        des_speed=0.8,
        dt_simulation=0.03,
        show_plot_vectorfield=True,
        # show_plot_vectorfield=False,
    )


def visualize_crowd_scene(n_resolution=10, save_figure=True):
    obs = GradientContainer()  # create empty obstacle list
    x_lim = [-0.6, 4.1]
    y_lim = [-2.1, 2.1]

    xAttractor = [0.0, 0.0]

    figsize = (6, 5)

    obs.append(
        CircularObstacle(
            center_position=[2.0, 0.0],
            tail_effect=False,
            radius=0.35,
            margin_absolut=0.4,
            is_boundary=False,
            has_sticky_surface=False,
            reactivity=1,
        )
    )

    obs[-1].sigma = 3  # Exponential weight for veloctiy reduction
    obs[-1].reactivity = 3  # Veloctiy reduction

    fig = plt.figure(figsize=figsize)
    fig, ax = plt.subplots()

    plt_speed_line_and_qolo(
        points_init=np.array([3.5, 0.2]),
        attractorPos=xAttractor,
        obs=obs,
        fig_and_ax_handle=(fig, ax),
    )

    fig_mod, ax_mod = Simulation_vectorFields(
        x_lim,
        y_lim,
        obs=obs,
        xAttractor=xAttractor,
        saveFigure=save_figure,
        figName="circular_sticky_surface",
        noTicks=False,
        draw_vectorField=True,
        automatic_reference_point=True,
        point_grid=n_resolution,
        show_streamplot=False,
        figureSize=figsize,
        reference_point_number=False,
        showLabel=False,
        normalize_vectors=False,
        dynamicalSystem=linearAttractor_const,
        fig_and_ax_handle=(fig, ax),
    )


if (__name__) == "__main__":
    plt.ion()
    # visualize_simple_ellipse(n_resolution=20)

    # visualize_intersecting_ellipse(save_figures=False, n_resolution=20)

    # visualize_repulsive_cube(n_resolution=100, save_figure=True)

    # visualize_circular_boundary(n_resolution=100, save_figure=True)

    # visualize_starshaped_boundary(n_resolution=100, save_figure=True)

    # visualize_edge_obstacle(n_resolution=10, save_figure=False, static_simulation=True)
    # visualize_edge_obstacle(n_resolution=10, save_figure=False, dynamic_simulation=True)

    visualize_edge_boundary(n_resolution=50, save_figure=False, static_simulation=True)
    # visualize_edge_boundary(n_resolution=10, save_figure=False, dynamic_simulation=True)

    # visualize_edge_boundary(n_resolution=100, save_figure=True)

    # visualize_dynamic_boundary(save_figure=False, dynamic_simulation=True)

    # visualize_crowd_scene(n_resolution=20)


# Run function
