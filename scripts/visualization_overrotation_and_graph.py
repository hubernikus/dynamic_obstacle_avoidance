""" Visualization of convergence-summing / learned trajectory. """
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
i
from vartools.dynamical_systems import DynamicalSystem, LinearSystem
from vartools.directional_space import UnitDirection, DirectionBase
from vartools.dynamical_systems import SinusAttractorSystem
from vartools.dynamical_systems import ConstVelocityDecreasingAtAttractor

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.containers import MultiBoundaryContainer
from dynamic_obstacle_avoidance.visualization import Simulation_vectorFields, plot_obstacles

# TODO: move to trajectory learning
from dynamic_obstacle_avoidance.avoidance.multihull_convergence import get_desired_radius, multihull_attraction


def get_positions(x_lim, y_lim, n_resolution, flattened=False):
    """ Returns array of n_resolution*n_resolution of positions between x_lim & y_lim """
    dimension = 2
    nx, ny = n_resolution, n_resolution
    x_vals, y_vals = np.meshgrid(np.linspace(x_lim[0], x_lim[1], nx),
                                 np.linspace(y_lim[0], y_lim[1], ny))
    
    positions = np.vstack((x_vals.reshape(1,-1), y_vals.reshape(1,-1)))

    if not flattened:
        positions = positions.reshape(2, n_resolution, n_resolution)
    return positions


def visualize_overrotation_two_ellipse(save_figure=False, comparison_quiver=False,
                                       n_resolution=20,
                                       ):
    dim = 2
        
    x_lim = [-8, 1]
    y_lim = [-4, 4]

    # InitialDynamics = LinearSystem(attractor_position=np.array([6, -5]))
    limiter = ConstVelocityDecreasingAtAttractor(const_velocity=1.0, distance_decrease=1.0)
    initial_dynamics = SinusAttractorSystem(trimmer=limiter, attractor_position=np.zeros(dim))
    
    obstacle_list = MultiBoundaryContainer()
    obstacle_list.append(
        Ellipse(
        center_position=np.array([-1, -1]), 
        axes_length=np.array([2.5, 1.5]),
        orientation=40./180*pi,
        is_boundary=True,
        )
    )
    obstacle_list.append(
        Ellipse(
        center_position=np.array([-3, -1]), 
        axes_length=np.array([2.5, 1.5]),
        orientation=-40./180*pi,
        is_boundary=True,
        )
    )

    # obstacle_list.append(
        # Ellipse(
        # center_position=np.array([-5, -1]), 
        # axes_length=np.array([2.5, 1.5]),
        # orientation=40./180*pi,
        # is_boundary=True,
        # )
    # )
    
    convering_dynamics = LinearSystem(
        attractor_position=initial_dynamics.attractor_position, maximum_velocity=0.5)

    obstacle_list.set_convergence_directions(convering_dynamics)

    # fig, axs = plt.subplots(1, 2, figsize=(14, 8))
    # ax = axs[1]
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    # ax = axs[1]
        
    positions = get_positions(x_lim, y_lim, n_resolution, flattened=True)

    initial_velocities = np.zeros(positions.shape)
    rotated_velocities = np.zeros(positions.shape)

    for it in range(positions.shape[1]):
        initial_velocities[:, it] = initial_dynamics.evaluate(positions[:, it])
        rotated_velocities[:, it] = multihull_attraction(
            positions[:, it], initial_velocities[:, it], obstacle_list)

    plot_obstacles(ax=ax, obs=obstacle_list, x_range=x_lim, y_range=y_lim,
                   noTicks=False, showLabel=False, draw_wall_reference=True,
                   alpha_obstacle=0)
    ax.plot(initial_dynamics.attractor_position[0],
                initial_dynamics.attractor_position[1], 'k*',
                linewidth=18.0, markersize=18, zorder=5)

    ax.tick_params(axis='both', which='major',bottom=False,
                   top=False, left=False, right=False, labelbottom=False, labelleft=False)

    if comparison_quiver:
        ax.quiver(positions[0, :], positions[1, :],
                  initial_velocities[0, :], initial_velocities[1, :],
                  color='blue', zorder=3)
        ax.quiver(positions[0, :], positions[1, :],
                  rotated_velocities[0, :], rotated_velocities[1, :],
                  color='black', zorder=3)
        
        if save_figure:
            fig_name = "quiver_two_ellipse_wavy_attraction"
            plt.savefig("figures/" + fig_name + ".png", bbox_inches='tight')
    
    else:
        nn = n_resolution
        stream_color='blue'
        ax.streamplot(
            positions[0, :].reshape(nn, nn), positions[1, :].reshape(nn, nn),
            rotated_velocities[0, :].reshape(nn, nn), rotated_velocities[1, :].reshape(nn, nn),
            color=stream_color,
            # zorder=3,
            )

        if save_figure:
            fig_name = "streamplot_two_ellipse"
            plt.savefig("figures/" + fig_name + ".png", bbox_inches='tight')
            
    
def visualize_single_ellipse_overroation1(save_figure=False):
    pass


if (__name__) == '__main__':
    # visualize_overrotation_two_ellipse(save_figure=True, comparison_quiver=True)

    # TODO: test with resolution=100, since bugs start turning up...
    visualize_overrotation_two_ellipse(save_figure=True, n_resolution=100)
