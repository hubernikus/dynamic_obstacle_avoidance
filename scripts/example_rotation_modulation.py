#!/USSR/bin/python3
""" Script to show lab environment on computer """

# Author: Lukas Huber
# License: BSD @ 2021

import warnings
import copy

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

from vartools.dynamicalsys.closedform import ds_quadratic_axis_convergence
from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system

from dynamic_obstacle_avoidance.obstacles import BaseContainer
from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.avoidance import obstacle_avoidance_rotational
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving

from dynamic_obstacle_avoidance.visualization import Simulation_vectorFields, plot_obstacles

# plt.close('all')
plt.ion()

def single_ellipse():
    obs_list = BaseContainer()
    
    obs_list.append(
        Ellipse(
        center_position=np.array([0, 0]), 
        axes_length=np.array([2, 5]),
        orientation=0./180*pi,
        )
    )
    return obs_list

def multiple_ellipse_hulls():
    obs_list = BaseContainer()

    obs_list.append(
        Ellipse(
        center_position=np.array([-6, 0]), 
        axes_length=np.array([5, 2]),
        orientation=50./180*pi,
        is_boundary=True,
        )
    )
    obs_list.append(
        Ellipse(
        center_position=np.array([0, 0]), 
        axes_length=np.array([5, 2]),
        orientation=-50./180*pi,
        is_boundary=True,
        )
    )
    obs_list.append(
        Ellipse(
        center_position=np.array([6, 0]), 
        axes_length=np.array([5, 2]),
        orientation=50./180*pi,
        is_boundary=True,
        )
    )
    return obs_list


def single_ellipse_linear_triple_plot(n_resolution=100, save_figure=False):
    x_lim = [-10, 10]
    y_lim = [-10, 10]
    
    pos_attractor = np.array([8, 0])

    def initial_ds(position):
        return evaluate_linear_dynamical_system(position, center_position=pos_attractor)
    
    def obs_avoidance(*args, **kwargs):
        def get_convergence_direction(position):
            return evaluate_linear_dynamical_system(position, center_position=pos_attractor)
        return obstacle_avoidance_rotational(*args, **kwargs, get_convergence_direction=get_convergence_direction)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    obstacle_list = single_ellipse()
    Simulation_vectorFields(
        x_lim, y_lim, n_resolution, obstacle_list,
        saveFigure=False, 
        noTicks=True, showLabel=False,
        draw_vectorField=True,
        dynamical_system=initial_ds,
        obs_avoidance_func=obs_avoidance,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[2]),
        show_streamplot=True,
        )
    
    obstacle_list = []
    Simulation_vectorFields(
        x_lim, y_lim, n_resolution, obstacle_list,
        saveFigure=False, 
        noTicks=True, showLabel=False,
        draw_vectorField=True,
        dynamical_system=initial_ds,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[0]),
        )
    
    obstacle_list = single_ellipse()
    Simulation_vectorFields(
        x_lim, y_lim, n_resolution, obstacle_list,
        saveFigure=False, 
        noTicks=True, showLabel=False,
        draw_vectorField=True,
        dynamical_system=initial_ds,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[1]),
        )

    if save_figure:
        figure_name = "comparison_linear_vectorfield"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches='tight')


def single_ellipse_nonlinear_triple_plot(n_resolution=100, save_figure=False):
    x_lim = [-10, 10]
    y_lim = [-10, 10]
    
    pos_attractor = np.array([8, 0])

    def initial_ds(x):
        return ds_quadratic_axis_convergence(
            x,  center_position=pos_attractor, stretching_factor=3,
            max_vel=1.0
            )

    def obs_avoidance(*args, **kwargs):
        def get_convergence_direction(position):
            return evaluate_linear_dynamical_system(position, center_position=pos_attractor)
        return obstacle_avoidance_rotational(*args, **kwargs, get_convergence_direction=get_convergence_direction)

    fig, axs = plt.subplots(1, 3, figsize=(15, 7))

    obstacle_list = single_ellipse()
    Simulation_vectorFields(
        x_lim, y_lim, n_resolution, obstacle_list,
        saveFigure=False,
        noTicks=True, showLabel=False,
        draw_vectorField=True,
        dynamical_system=initial_ds,
        obs_avoidance_func=obs_avoidance,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[2]),
        show_streamplot=True,
        )
    
    obstacle_list = []
    Simulation_vectorFields(
        x_lim, y_lim, n_resolution, obstacle_list,
        saveFigure=False, 
        noTicks=True, showLabel=False,
        draw_vectorField=True,
        dynamical_system=initial_ds,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[0]),
        )
    
    obstacle_list = single_ellipse()
    Simulation_vectorFields(
        x_lim, y_lim, n_resolution, obstacle_list,
        saveFigure=False, 
        noTicks=True, showLabel=False,
        draw_vectorField=True,
        dynamical_system=initial_ds,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[1]),
        )

    if save_figure:
        figure_name = "comparison_nonlinear_vectorfield"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches='tight')

def multiple_hull_linear(save_figure=False, n_resolution=10):
    """ Multiple ellipse hull. """

    x_lim = [-10, 10]
    y_lim = [-10, 10]
    
    pos_attractor = np.array([8, 0])

    def initial_ds(x):
        return ds_quadratic_axis_convergence(
            x,  center_position=pos_attractor, stretching_factor=3,
            max_vel=1.0
            )

    def obs_avoidance(*args, **kwargs):
        def get_convergence_direction(position):
            return evaluate_linear_dynamical_system(position, center_position=pos_attractor)
        return obstacle_avoidance_rotational(*args, **kwargs,
                                             get_convergence_direction=get_convergence_direction)

    fig, ax = plt.subplots(1, 1, figsize=(15, 7))

    obstacle_list = multiple_ellipse_hulls()
    Simulation_vectorFields(
        x_lim, y_lim, n_resolution, obstacle_list,
        saveFigure=False, 
        noTicks=True, showLabel=False,
        draw_vectorField=True,
        dynamical_system=initial_ds,
        obs_avoidance_func=obs_avoidance,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, ax),
        show_streamplot=False,
    )
    

if (__name__)=="__main__":
    # single_ellipse_linear_triple_plot(save_figure=True)
    # single_ellipse_nonlinear_triple_plot(save_figure=False)

    multiple_hull_linear(save_figure=False)

    # single_ellipse_hull(save_figure=True)

