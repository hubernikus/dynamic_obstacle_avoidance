""" Ellipse Obstacle for Obstacle Avoidance and Visualization Purposes. """
# Author Lukas Huber
# Email lukas.huber@epfl.ch
# License BSD

# import warnings

import numpy as np
from numpy import linalg as LA

from vartools import linalg

from dynamic_obstacle_avoidance import obstacles


class CuboidWithAxes(obstacles.Obstacle):
    def __init__(
        self,
        axes_length: np.ndarray,
        *args,
        **kwargs):
        
        super().__init__(*args, **kwargs)

        self.axes_length = axes_length

    @property
    def axes_length(self):
        return self._axes_length

    @axes_length.setter
    def axes_length(self, value: np.ndarray):
        if any(value <= 0):
            raise ValueError("Zero axes input not tolerated.")
        self._axes_length = value

    def get_normal_direction(self, position, in_obstacle_frame: bool = True):
        if not in_obstacle_frame:
            position = self.pose.transform_position_from_reference_to_local(position)

        ind_relevant = np.abs(position) > self.axes_length

        if not any(ind_relevant):
            return np.ones(position.shape) / position.shape[0]

        normal = np.zeros(position.shape)
        normal[ind_relevant] = (
            position[ind_relevant] -
            np.copysign(self.axes_length[ind_relevant], position[ind_relevant])
            )

        # No normalization chack needed, since at least one axes was relevatn
        normal = normal / LA.norm(normal)
        
        if not in_obstacle_frame:
            normal = self.pose.transform_direction_from_reference_to_local(normal)

        return normal

    def get_distance_to_surface(self, position, in_obstacle_frame: bool = True):
        if not in_obstacle_frame:
            position = self.pose.transform_position_from_reference_to_local(position)

        relative_position = np.abs(position) - self.axes_length
        relative_position = np.maximum(relative_position, 0)

        return LA.norm(relative_position)
            
    def get_gamma(self, position, in_obstacle_frame: bool = True):
        if not in_obstacle_frame:
            position = self.pose.transform_position_from_reference_to_local(position)
            
        distance = self.get_distance_to_surface(
            position=position, in_obstacle_frame=in_obstacle_frame)

        return distance + 1

    def get_point_on_surface(self, position, in_obstacle_frame: bool = True):
        if not in_obstacle_frame:
            cube_position = self.pose.transform_position_from_reference_to_local(position)

        cube_position = cube_position / self.axes_length

        ind_max = np.argmax(cube_position)
        
        return position * self.axes_length[ind_max] / position[ind_max]
