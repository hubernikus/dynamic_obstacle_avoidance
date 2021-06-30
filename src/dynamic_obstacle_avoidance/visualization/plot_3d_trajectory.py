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

from vartools.dynamical_systems import LinearSystem

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
    InitialDynamics, func_obstacle_avoidance,
    start_positions=None, n_points=2,
    delta_time=0.01, n_max_it=10000,
    convergence_margin=1e-4,
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

    if start_positions is None:
        start_positions = np.random.rand(dimension, n_points)
        
        start_positions[0, :] = start_positions[0, :]*(x_lim[1]-x_lim[0]) + x_lim[0]
        start_positions[1, :] = start_positions[1, :]*(y_lim[1]-y_lim[0]) + y_lim[0]
        start_positions[2, :] = start_positions[2, :]*(z_lim[1]-z_lim[0]) + z_lim[0]
    else:
        n_points = start_positions.shape[1]

    trajectory_points = [np.zeros((dimension, n_max_it)) for ii in range(n_points)]
    for ii in range(n_points):
        trajectory_points[ii][:, 0] = start_positions[:, ii]

    active_trajectories = np.ones((n_points), dtype=bool)
    for it_step in range(n_max_it-1):
        for ii in np.arange(n_points)[active_trajectories]:
            initial_velocity = InitialDynamics.evaluate(trajectory_points[ii][:, it_step])
            modulated_velocity = func_obstacle_avoidance(trajectory_points[ii][:, it_step],
                                                         initial_velocity,
                                                         ObstacleContainer)
            trajectory_points[ii][:, it_step+1] = (trajectory_points[ii][:, it_step]
                                                   + modulated_velocity*delta_time)

            # Check convergence
            if np.linalg.norm(trajectory_points[ii][:, it_step+1]
                              - InitialDynamics.attractor_position) < convergence_margin:
                print(f"Trajectory {ii} has converged at step {it_step}.")
                active_trajectories[ii] = False

        if not any(active_trajectories):
            print("All trajectories have converged. Stopping the simulation.")
            break

    for ii in np.arange(n_points):
        ax.plot([trajectory_points[ii][0, 0]],
                [trajectory_points[ii][1, 0]],
                [trajectory_points[ii][2, 0]], 'k.' )
        
        ax.plot(trajectory_points[ii][0, :],
                trajectory_points[ii][1, :],
                trajectory_points[ii][2, :])
                            
    if hasattr(InitialDynamics, 'attractor_position'):
        ax.plot([InitialDynamics.attractor_position[0]],
                [InitialDynamics.attractor_position[1]],
                [InitialDynamics.attractor_position[2]], 'k*')
                
                

    # ax.set_aspect('equal')
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if z_lim is not None:
        ax.set_zlim(z_lim)
        
    plt.ion()
    plt.show()
