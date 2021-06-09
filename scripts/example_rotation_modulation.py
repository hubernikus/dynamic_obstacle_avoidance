#!/USSR/bin/python3
""" Script to show lab environment on computer """
# author: Lukas Huber
# email: hubernikus@gmail.com

import warnings
import copy

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.avoidance.utils import *

from dynamic_obstacle_avoidance.obstacles import BaseContainer

from dynamic_obstacle_avoidance.obstacles import Obstacle, Ellipse

from dynamic_obstacle_avoidance.avoidance import obstacle_avoidance_rotational
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving


plt.close('all')
plt.ion()

# TO

if (__name__)=="__main__":
    obs_list = BaseContainer()

    obs_list.append(
        Ellipse(
        position=np.array([-2, 0]), 
        axes_length=np.array([3, 2]),
        orientation=30./180*pi,
        )
    )

    obs_list.append(
        Ellipse(
        position=np.array([2, 3]), 
        axes_length=np.array([3, 2]),
        orientation=-30./180*pi,
        )
    )

    
    
    plt.plot()
 
    pass
