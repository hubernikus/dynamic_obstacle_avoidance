"""
Replicating the paper of
'Safety of Dynamical Systems With MultipleNon-Convex Unsafe Sets UsingControl Barrier Functions'
"""
from abc import ABC, abstractmethod
import time
import copy

import numpy as np
from numpy import linalg as LA

from cvxopt import solvers, matrix

import matplotlib.pyplot as plt
from matplotlib import cm

from vartools.dynamical_systems import DynamicalSystem, LinearSystem
from vartools.dynamical_systems import plot_dynamical_system_streamplot
from vartools.math import get_numerical_gradient, get_numerical_hessian
from vartools.math import get_numerical_hessian_fast
from vartools.math import get_scaled_orthogonal_projection

from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.obstacles import Obstacle, Sphere
from dynamic_obstacle_avoidance.containers import BaseContainer

from barrier_functions import BarrierFunction, CirclularBarrier, DoubleBlobBarrier
from barrier_functions import BarrierFromObstacleList

from _base_qp import ControllerQP
from navigation import SphereToStarTransformer
from double_blob_obstacle import DoubleBlob
from sphere_world_optimizer import SphereWorldOptimizer, ClosedLoopQP
from control_dynamics import StaticControlDynamics

    
def animation_double_worlds(
    start_position, it_max=100, delta_time=0.01, wait_time=0.1):
    x_lim = [-2, 6]
    y_lim = [-4, 6]
    dimension = 2

    # Set to 1000 as describe din paper.
    # Does this work or do we need barrier function (!?)
    sphere_world = SphereWorldOptimizer(
        attractor_position=np.array([0, 0]),
        lambda_constant=1000)
    
    sphere_world.append(
        DoubleBlob(
            a_value=1, b_value=1.1,
            center_position=[0, 3],
            is_boundary=False,
            ))

    f_x = LinearSystem(
        A_matrix=np.array([[-6, 0],
                           [0, -1]])
        )

    g_x = StaticControlDynamics(A_matrix=np.eye(dimension))
    # g_x = LinearSystem(A_matrix=np.eye(dimension))

    qp_controller = ClosedLoopQP(f_x=f_x, g_x=g_x)

    controller_sphere = NonconvexAvoidanceCBF(obstacle_container=sphere_world,
                                              qp_control_optimizer=qp_controller)
    
    # fig, ax = plt.subplots(figsize=(7.5, 6))
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 6))
    axs = ax
    
    plot_obstacles_boundary(ax, controller)
    
    n_obs_plus_boundary = len(sphere_world)

    trajectory = np.zeros((dimension, it_max+1))
    # trajectory[:, 0] = start_position

    traj_spher = np.zeros((dimension, it_max+1))
    traj_spher[:, 0] = controller.obstacle_container.transform_to_sphereworld(start_position)
               
    plt_outline = [None] * n_obs_plus_boundary 
    plt_center = [None] * (n_obs_plus_boundary-1)
    plt_positions = None
    # Main loop
    for it in range(it_max):
        # update_in_sphere_world
        # velocity = controller.evaluate(position=trajectory[:, it])
        # velocity = controller.evaluate(position=trajectory[:, it])
        # trajectory[:, it+1] = trajectory[:, it] + velocity*delta_time
        
        vel_sphere = controller.evaluate_in_sphere_world(position=traj_spher[:, it])
        traj_spher[:, it+1] = traj_spher[:, it] + vel_sphere*delta_time

        trajectory[:, it+1] = controller.obstacle_container.transform_from_sphereworld(
            traj_spher[:, it+1])
        
        plot_obstacles_boundary(ax, controller)

        ax.plot(traj_spher[0, :it+1], traj_spher[1, :it+1], 'r')
        ax.plot(traj_spher[0, 0], traj_spher[1, 0], 'r*')
        ax.plot(traj_spher[0, it+1], traj_spher[1, it+1], 'ro')
        
        # plt.show()
        # time.sleep(wait_time)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        print(f"Loop #{it}")

        plt.pause(wait_time)
        
        ax.clear()

        if not len(plt.get_fignums()):
            # No figure active
            print("Animation ended by closing of figures.")
            break

    if False:
        # Plot everything
        for ii in range(len(sphere_world)):
            obs = sphere_world.sphere_world_list[ii]
            obs.draw_obstacle()
            plt_outline[ii] = boundary_points = obs.boundary_points_global
            plt_center[ii] = ax.plot(boundary_points[0, :], boundary_points[1, :], 'k')
            
            if not obs.is_boundary:
                ax.plot(obs.center_position[0], obs.center_position[1], 'k+')


if (__name__) == "__main__":
    plt.ion()
    solvers.options['show_progress'] = False
    # plot_main_vector_field()
    
    # plot_barrier_function()
    # plot_integrate_trajectory()
    
    # plot_spherial_dynamic_container()
    animation_spherical_wold(start_position=np.array([4.0, 5]))

    plt.show()
