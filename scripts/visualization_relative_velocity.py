# !/USSR/bin/python3
"""
Script which creates a variety of examples of local modulation of a vector field with obstacle avoidance. 
"""
import sys
import os

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    plot_obstacles,
    Simulation_vectorFields,
)  #
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import *
from dynamic_obstacle_avoidance.obstacle_avoidance.flower_shape import StarshapedFlower
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import (
    obs_avoidance_interpolation_moving,
)

__author__ = "LukasHuber"
__email__ = "lukas.huber@epfl.ch"
__date__ = "2020-11-01"


def test_get_relative_speed_around(n_grid_num=40):
    """Normal has to point alongside reference"""
    obs = Ellipse(
        axes_length=[2, 2.0],
        linear_velocity=np.array([-4.0, 4.0]),
        center_position=[0.0, 0.0],
        orientation=0.0 / 180 * np.pi,
        sigma=10,
        reactivity=3,
        tail_effect=False,
    )

    obs_list = GradientContainer()
    obs_list.append(obs)

    # Check 10 random points
    x_range = [-10, 10]
    y_range = [-10, 10]

    x_vals = np.linspace(x_range[0], x_range[1], n_grid_num)
    y_vals = np.linspace(y_range[0], y_range[1], n_grid_num)

    dim = 2
    pos = np.zeros((dim, n_grid_num, n_grid_num))
    rel_vel = np.zeros((dim, n_grid_num, n_grid_num))

    # for ix in range(n_grid_num):
    # for iy in range(n_grid_num):
    # pos[:, ix, iy] = [x_vals[ix], y_vals[iy]]
    # xd = np.array([1, 1])
    # rel_vel[:, ix, iy] = obs_avoidance_interpolation_moving(pos[:, ix, iy], xd, obs=obs_list)

    # plt.figure()
    fig, ax = plt.subplots()
    # plot_obstacles(ax, obs=obs_list, x_range=x_range, y_range=y_range)
    # ax.quiver(pos[0, :, :], pos[1, :, :],  rel_vel[0, :, :], rel_vel[1, :, :], zorder=0)

    Simulation_vectorFields(
        x_range,
        y_range,
        obs=obs_list,
        # xAttractor=self.attractor_position,
        saveFigure=False,
        figName="linearSystem_boundaryCuboid",
        noTicks=False,
        draw_vectorField=True,
        show_streamplot=True,
        point_grid=n_grid_num,
        normalize_vectors=False,
        reference_point_number=False,
        drawVelArrow=True,
        automatic_reference_point=True,
        xAttractor=np.array([8, 0]),
        fig_and_ax_handle=(fig, ax),
    )


if (__name__) == "__main__":
    plt.ion()
    test_get_relative_speed_around()
    # visualize_simple_ellipse(n_resolution=20)
