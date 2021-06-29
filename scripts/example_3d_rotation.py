#!/USSR/bin/python3
""" Script to show lab environment on computer """
# Author: Lukas Huber
# License: BSD (c) 2021

import warnings
import copy

import numpy as np
from numpy import pi

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.containers import MultiBoundaryContainer, RotationContainer
from dynamic_obstacle_avoidance.visualization.plot_3d_trajectory import plot_obstacles_and_trajectory_3d

def test_single_ellipse_3d():
    ObstacleEnvironment = RotationContainer()
    ObstacleEnvironment.append(
        Ellipse(
        center_position=np.array([0, 1, 0]), 
        axes_length=np.array([0.1, 0.3, 0.2]),
        orientation=[0, 0, 0], # Set orientation as euler
        tail_effect=False,
        )
    )
    ObstacleEnvironment.append(
        Ellipse(
        center_position=np.array([-0.5, 1, 1]), 
        axes_length=np.array([0.2, 0.2, 0.1]),
        orientation=[0, 0, 0], # Set orientation as euler
        tail_effect=False,
        )
    )
    xyz_lim=[-1.5, 1.5]
    # plot_obstacles_and_trajectory_3d(ObstacleEnvironment)
    plot_obstacles_and_trajectory_3d(
        ObstacleEnvironment,
        x_lim=xyz_lim, y_lim=xyz_lim, z_lim=xyz_lim)
                          # x_lim=[-2, 2], y_lim=[-2, 2], z_lim=[-2, 2])
    
if (__name__)=="__main__":
    test_single_ellipse_3d()
    

