#!/USSR/bin/python3
'''
@date 2019-10-15
@author Lukas Huber 
@email lukas.huber@epfl.ch
'''

import time
import numpy as np
from math import sin, cos, pi, ceil
import warnings, sys

import numpy.linalg as LA


# import quaternion # numpy-quaternion 

# import dynamic_obstacle_avoidance

from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import *

from dynamic_obstacle_avoidance.obstacle_avoidance.state import *
from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import angle_modulo
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_dynamic_center_3d import *

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *

import matplotlib.pyplot as plt

# import quaternion

visualize_debug = False

class StarshapedFlower(Obstacle):
    def __init__(self,  radius_magnitude=1, radius_mean=2, number_of_edges=4,
                 *args, **kwargs):
        if sys.version_info>(3,0):
            super().__init__(*args, **kwargs)
        else:
            super(StarshapedFlower, self).__init__(*args, **kwargs)

        # Object Specific Paramters
        self.radius_magnitude=radius_magnitude
        self.radius_mean=radius_mean

        self.number_of_edges=number_of_edges

        self.is_convex = False # What for?
        

    def get_radius_of_angle(self, angle, in_global_frame=False):
        if in_global_frame:
            angle -= self.orientation
        return self.radius_mean+self.radius_magnitude*np.cos((angle)*self.number_of_edges)
    

    def get_radiusDerivative_of_angle(self, angle, in_global_frame=False):
        if in_global_frame:
            angle -= self.orientation
        return -self.radius_magnitude*self.number_of_edges*np.sin((angle)*self.number_of_edges)


    def draw_obstacle(self, include_margin=False, n_curve_points=100, numPoints=None):
        # warnings.warn("Remove numPoints from function argument.")

        angular_coordinates = np.linspace(0,2*pi, n_curve_points)
        radius_angle = self.get_radius_of_angle(angular_coordinates)

        if self.dim==2:
            direction = np.vstack(( np.cos(angular_coordinates), np.sin(angular_coordinates) ))

        self.x_obs = (radius_angle * direction)
        self.x_obs_sf = (radius_angle * self.sf * direction)

        if self.orientation: # nonzero
            for jj in range(self.x_obs.shape[1]):
                self.x_obs[:, jj] = self.rotMatrix.dot(self.x_obs[:, jj]) + np.array([self.center_position])
            for jj in range(self.x_obs_sf.shape[1]):
                self.x_obs_sf[:,jj] = self.rotMatrix.dot(self.x_obs_sf[:, jj]) + np.array([self.center_position])

        self.x_obs = self.x_obs.T
        self.x_obs_sf = self.x_obs_sf.T


    def get_gamma(self, position, in_global_frame=False, norm_order=2):
        if not type(position)==np.ndarray:
            position = np.array(position)

        # Rename
        if in_global_frame:
            position = self.transform_global2relative(position)

        mag_position = LA.norm(position)
        if mag_position==0:
            if self.is_boundary:
                return sys.float_info.max
            else:
                return 0

        direction = np.arctan2(position[1], position[0])

        Gamma = mag_position/self.get_radius_of_angle(direction)

        # TODO extend rule to include points with Gamma < 1 for both cases
        if self.is_boundary:
            Gamma = 1/Gamma

        return Gamma


    def get_normal_direction(self, position, in_global_frame=False, normalize=True):
        if in_global_frame:
            position = self.transform_global2relative(position)

        mag_position = LA.norm(position)
        if not mag_position:
            return np.ones(self.dim)/self.dim # just return one direction

        direction = np.arctan2(position[1], position[0])
        derivative_radius_of_angle = self.get_radiusDerivative_of_angle(direction)

        radius = self.get_radius_of_angle(direction)

        normal_vector = np.array(([
            derivative_radius_of_angle*np.sin(direction) + radius*np.cos(direction),
            - derivative_radius_of_angle*np.cos(direction) + radius*np.sin(direction)]))

        if normalize:
            mag_vector = LA.norm(normal_vector)
            if mag_vector: #nonzero
                normal_vector = normal_vector/mag_vector
                
        if False:
            # self.draw_reference_hull(normal_vector, position)
            pos_abs = self.transform_relative2global(position)
            norm_abs = self.transform_relative2global_dir(normal_vector)
            plt.quiver(pos_abs[0], pos_abs[1], norm_abs[0], norm_abs[1], color='g')
            ref_abs = self.get_reference_direction(position)
            ref_abs = self.transform_relative2global_dir(ref_abs)
            plt.quiver(pos_abs[0], pos_abs[1], ref_abs[0], ref_abs[1], color='k')

            plt.ion()
            plt.show()

        if in_global_frame:
            normal_vector = self.transform_relative2global_dir(normal_vector)
            
        return normal_vector

    
