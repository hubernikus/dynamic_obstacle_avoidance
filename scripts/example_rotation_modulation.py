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

from dynamic_obstacle_avoidance.obstacles import BaseContainer
from dynamic_obstacle_avoidance.obstacles import Obstacle, Ellipse
from dynamic_obstacle_avoidance.avoidance import obstacle_avoidance_rotational
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving

from dynamic_obstacle_avoidance.visualization import Simulation_vectorFields, plot_obstacles

plt.close('all')
plt.ion()


def multiple_ellipse_hull():
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
    

def parallel_ds(position, direction):
    return direction

if (__name__)=="__main__":
    x_lim = [-10, 10]
    y_lim = [-10, 10]
    
    n_resolution = 50

    pos_attractor = np.array([8, 0])

    def initial_ds(x):
        return ds_quadratic_axis_convergence(
            x,  center_position=pos_attractor, stretching_factor=3)
        

    fig, axs = plt.subplots(1, 3, figsize=(15, 7))

    obstacle_list = single_ellipse()
    Simulation_vectorFields(
        x_lim, y_lim, n_resolution, obstacle_list,
        saveFigure=False, figName='rotational_avoidance',
        noTicks=True, 
        draw_vectorField=True,
        dynamical_system=initial_ds,
        obs_avoidance_func=obstacle_avoidance_rotational,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[2]),
        )

    
    obstacle_list = []
    Simulation_vectorFields(
        x_lim, y_lim, n_resolution, obstacle_list,
        saveFigure=False, figName='rotational_avoidance',
        noTicks=True, 
        draw_vectorField=True,
        dynamical_system=initial_ds,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[0]),
        )
    
    obstacle_list = single_ellipse()
    Simulation_vectorFields(
        x_lim, y_lim, n_resolution, obstacle_list,
        saveFigure=False, figName='rotational_avoidance',
        noTicks=True, 
        draw_vectorField=True,
        dynamical_system=initial_ds,
        automatic_reference_point=False,
        pos_attractor=pos_attractor,
        fig_and_ax_handle=(fig, axs[1]),
        )



    # for obs in obs_list:
    #     obs.draw_obstacle()
        
    # fig = plt.figure(figsize=(12, 8))
    # ax = plt.subplot(1, 1, 1)
    
    # plot_obstacles(ax, obs_list, x_lim, y_lim, showLabel=False)

    # nx = ny = n_resolution
    # x_grid, y_grid = np.meshgrid(np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny))
    # positions = np.vstack((x_grid.reshape(1,-1), y_grid.reshape(1,-1))).reshape(-1, nx, ny)
    
    # velocity_init = np.zeros(positions.shape)
    # # vel_constant = np.array([1, 0])
    # # velocity_init = np.tile(vel_constant, (ny, nx, 1))
    # # velocity_init = np.swapaxes(velocity_init, 0, 2)
    
    # velocity_mod = np.zeros(positions.shape)

    # for ix in range(n_resolution):
    #     for iy in range(n_resolution):
    #         velocity_init[:, ix, iy] = ds_quadratic_axis_convergence(positions[:, ix, iy])

    # plt.quiver()
