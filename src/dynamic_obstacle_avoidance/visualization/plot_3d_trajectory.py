"""Three (3) dimensional representation of obstacle avoidance. """
# Author: Lukas Huber
# Date: 2021-06-25
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import os
import warnings

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

plt.ion()  # Show plot without stopping code


def plot_obstacles(ObstacleContainer, ax=None):
    """ """ 
    if ax is None:
        # breakpoint()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    # else:
        # fig, ax = fig_and_ax_handle

    for obs in ObstacleContainer:
        data_points = obs.draw_obstacle()
        ax.plot_surface(data_points[0], data_points[1], data_points[2],
                        rstride=4, cstride=4, color=np.array([176, 124, 124])/255.)


def plot_obstacles_and_trajectory_3d(
    ObstacleContainer,
    # initial_dynamical_system=,
    start_positions=None,
    # func_obstacle_avoidance=,
    x_lim=None, y_lim=None, z_lim=None,
    fig_and_ax_handle=None,
    ):

    dimension = 3 # 3D-visualization
    
    if fig_and_ax_handle is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.gca(projection='3d')
    else:
        fig, ax = fig_and_ax_handle
        
    plot_obstacles(ObstacleContainer=ObstacleContainer, ax=ax)
    
    # ax.set_aspect('equal')
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if z_lim is not None:
        ax.set_zlim(z_lim)

