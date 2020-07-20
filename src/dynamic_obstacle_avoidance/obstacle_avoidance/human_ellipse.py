#!/USSR/bin/python3
'''
Script to create human specific pedestrian obstacle descriptions.
'''

from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import *
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import Ellipse

__date__ = "2019-10-15"
__author__ =  "Lukas Huber"
__mail__ =  "lukas.huber@epfl.ch"

import time
import numpy as np
from math import sin, cos, pi, ceil
import warnings, sys

import numpy.linalg as LA
import matplotlib.pyplot as plt



class PedestrianTracker(Ellipse):
    def __init__(self,
                 axes_length=[1.0, 1.0],
                 *args, **kwargs):
        
        if sys.version_info>(3,0):
            super().__init__(axes_length=axes_length, *args, **kwargs)
        else:
            super(Ellipse, self).__init__(axes_length=axes_length, *args, **kwargs)

        self.is_dynamic = True
        


class HumanEllipse(Ellipse):
    # Ellipse with proxemics
    # Intimate-, Personal-, Social-, Public- Spaces
    
    # first axis in direction of vision
    # second axis aligned with shoulders
    def __init__(self, axes_length=[0.4, 1.1],
                 public_axis=[16.0, 8.0], public_center=[4.0, 0.0],
                 personal_axis=[8, 3.0], personal_center=[2.0, 0.0],
                 *args, **kwargs):

        axes_length = np.array(axes_length)
        super().__init__(axes_length=axes_length, *args, **kwargs)
        
        self.public_axis = np.array(public_axis)
        self.public_center = np.array(public_center) # in local frame

        self.personal_axis = np.array(personal_axis)
        self.personal_center = np.array(personal_center) # in local frame
        
    def repulsion_force(self, position):
        raise NotImplementedError()

    def repulsion_force(self, position):
        raise NotImplementedError()

    
