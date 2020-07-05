########################################################################
# Command to automatically reload libraries -- in ipython before exectureion
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import *
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import *

__author__ =  "LukasHuber"
__email__ = "lukas.huber@epfl.ch"
__date__ =  "2018-02-15"

import numpy as np
import matplotlib.pyplot as plt


########################################################################

obstacles = GradientContainer()

obstacles.append(Cuboid(
    axes_length=[8, 9.6],
    center_position=[3, 1],
    orientation=0./180*pi,
    margin_absolut=0.0,
    is_boundary=True,
))

center_dists = np.array([[0., 0.],
                         [4., 4.]])

radius = obstacles[0].get_local_radius_point(direction=center_dists[:, 1],
                                             in_global_frame=True) 

# print('radius', radius)
