#!/USSR/bin/python3
""" Script to show lab environment on computer """
# Author: Lukas Huber
# License: BSD (c) 2021

import warnings
import copy

import numpy as np
from numpy import pi

from dynamic_obstacle_avoidance.containers import MultiBoundaryContainer
from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.visualization.plot_3d_trajectory import plot_obstacles_and_trajectory_3d

def test_single_ellipse_3d(self):
    obs_list = BaseContainer()
    
    obs_list.append(
        Ellipse(
        center_position=np.array([0, 1, 0]), 
        axes_length=np.array([1, 1, 1]),
        orientation=[0, 0, 0], # Set orientation as euler
        tail_effect=False,
        )
    )
    plot_obstacles_and_trajectory_3d()
    pass
    
    

if (__name__)=="__main__":
    test_ellipse_3d()
    

