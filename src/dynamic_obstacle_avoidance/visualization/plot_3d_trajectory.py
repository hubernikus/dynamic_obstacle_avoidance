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

def plot_obstacles(obstacle_list, fig_and_ax_handle=None):
    """ """ 
    if fig_and_ax_handle is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    else:
        fig, ax = fig_and_ax_handle


def plot_obstacles_and_trajectory_3d(
    obstacle_container, initial_dynamical_system=,
    start_positions=None,
    obstacle_avoidance_function=,
    x_lim=None, y_lim=None,
    fig_and_ax_handle=None,
    ):
    
    if fig_and_ax_handle is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    else:
        fig, ax = fig_and_ax_handle
        
    plot_obstacles(fig_and_ax_handle=fig_and_ax_handle)
    
    dim = 3 # 3D-visualization
