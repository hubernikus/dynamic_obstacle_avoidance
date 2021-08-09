""" Visualization of convergence-summing / learned trajectory. """
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.dynamical_systems import LocallyRotated

from dynamic_obstacle_avoidance.containers import MultiBoundaryContainer
from dynamic_obstacle_avoidance.obstacles import Ellipse

# from dynamic_obstacle_avoidance.visualization import plot_obstacles
from vartools.dynamical_systems import plot_dynamical_system_streamplot
from vartools.dynamical_systems import plot_dynamical_system_quiver

def draw_initial_locally_rotated(save_figure=False):
    """ """
    ds_list = []
    obs_list = MultiBoundaryContainer()
    obs_list.append(
        Ellipse(
        center_position=np.array([-8, 0]), 
        axes_length=np.array([3, 1]),
        orientation=10./180*pi,
        ))
    ds_list.append(LocallyRotated(mean_rotation=[-5/4*np.pi]
                                  ).from_ellipse(obs_list[-1]))

    obs_list.append(
            Ellipse(
            center_position=np.array([-10, 3]), 
            axes_length=np.array([3, 1]),
            orientation=90./180*pi,
            ))
    ds_list.append(LocallyRotated(mean_rotation=[-np.pi/2]
                                  ).from_ellipse(obs_list[-1]))

    obs_list.append(
        Ellipse(
        center_position=np.array([-4, 6]), 
        axes_length=np.array([6, 1.5]),
        orientation=0./180*pi,
        ))
    
    ds_list.append(LocallyRotated(mean_rotation=[-np.pi/4]
                                  ).from_ellipse(obs_list[-1]))
    
    obs_list.append(
            Ellipse(
            center_position=np.array([2, 3]), 
            axes_length=np.array([3, 1]),
            orientation=-90./180*pi,
            ))
    ds_list.append(LocallyRotated(mean_rotation=[-np.pi/5]
                                  ).from_ellipse(obs_list[-1]))

    obs_list.append(
            Ellipse(
            center_position=np.array([1, 0]), 
            axes_length=-np.array([2, 1]),
            orientation=-10./180*pi,
            ))
    ds_list.append(LocallyRotated(mean_rotation=[0]
                                  ).from_ellipse(obs_list[-1]))

    x_lim = [-14, 5]
    y_lim = [-3, 7]

    for ii in range(len(ds_list)):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        plot_obstacles(ax=ax, obs=obs_list, x_range=x_lim, y_range=y_lim,
                       showLabel=False, noTicks=True, alpha_obstacle=0.0)
        
        plot_dynamical_system_quiver(DynamicalSystem=ds_list[ii],
                                     x_lim=x_lim, y_lim=y_lim, fig_ax_handle=(fig, ax),
                                     n_resolution=40,
                                     )
        
        # plot_dynamical_system_streamplot(DynamicalSystem=ds_list[ii],
                                     # x_lim=x_lim, y_lim=y_lim, fig_ax_handle=(fig, ax),
                                     # n_resolution=40,
                                     # )

        ax.plot(ds_list[ii].center_position[0], ds_list[ii].center_position[1], 
                'k*',linewidth=18.0, markersize=18, zorder=5)

        boundary_points = obs_list[ii].boundary_points_global_closed.T
        ax.plot(boundary_points[:, 0], boundary_points[:, 1], 'r')

        if save_figure:
            figure_name = f"locally_rotated_ds_of_obs_{ii}"
            plt.savefig("figures/" + figure_name + ".png", bbox_inches='tight')

if (__name__)=="__main__":
    draw_initial_locally_rotated(save_figure=True)
