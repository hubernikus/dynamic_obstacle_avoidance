""" Ellipse Obstacle for Obstacle Avoidance and Visualization Purposes. """
# Author Lukas Huber
# Email lukas.huber@epfl.ch
# License BSD

# import warnings

import numpy as np
from numpy import linalg as LA

from vartools import linalg

from dynamic_obstacle_avoidance import obstacles


class EllipseWithAxes(obstacles.Obstacle):
    def __init__(
        self,
        axes_length: np.ndarray,
        curvature: float = 1,
        *args,
        **kwargs):
        
        super().__init__(*args, **kwargs)

        self.axes_length = axes_length
        self.curvature = curvature

    @property
    def axes_length(self):
        return self._axes_length

    @axes_length.setter
    def axes_length(self, value: np.ndarray):
        if any(value <= 0):
            raise ValueError("Zero axes input not tolerated.")
        self._axes_length = value
        
    def get_gamma(
        self,
        position: np.ndarray,
        in_obstacle_frame: bool = True):
        """ Gets a gamma which is not directly related to the axes length."""
        distance = self.get_point_on_surface(
            position=position, in_obstacle_frame=in_obstacle_frame)

        return distance + 1

    def get_normal_direction(
        self,
        position: np.ndarray,
        in_obstacle_frame: bool = True):
        if not in_obstacle_frame:
            position = self.pose.transform_position_from_reference_to_local(position)

        normal = (self.curvature/self.axes_length
                  *(position/self.axes_length)**(2*self.curvature-1)
                  )

        # Normalize
        normal_norm = LA.norm(normal)
        if normal_norm:
            normal = normal / normal_norm
        else:
            normal[0] = 1
        
        if not in_obstacle_frame:
            normal = self.pose.transform_direction_from_reference_to_local(normal)

        return normal

    def get_point_on_surface(
        self,
        position: np.ndarray,
        in_obstacle_frame: bool = True):
        """ Returns the point on the surface from the center with respect to position. """
        if not in_obstacle_frame:
            position = self.pose.transform_position_from_reference_to_local(position)

        # Position in the circle-world
        circle_position = position / self.axes_length
        
        pos_norm = LA.norm(circle_position)
        if not pos_norm:
            surface_point = np.zeros(position.shape)
            surface_point[0] = self.axes_length[0]

        else:
            surface_point = position / pos_norm

        if not in_obstacle_frame:
            surface_point = self.pose.transform_position_from_local_to_reference(surface_point)
            
        return surface_point


class HyperSphere(obstacles.Obstacle):
    # TODO: is this really worth it? Speed up towards ellipse is minimal...
    def __init__(
        self,
        radius: float,
        *args,
        **kwargs):
        
        super().__init__(*args, **kwargs)

        self.radius = radius
        
    def get_gamma(
        self,
        position: np.ndarray,
        in_obstacle_frame: bool = True):
        """ Gets a gamma which is not directly related to the axes length."""
        distance = self.get_point_on_surface(
            position=position, in_obstacle_frame=in_obstacle_frame)

        return distance + 1

    def get_normal_direction(
        self,
        position: np.ndarray,
        in_obstacle_frame: bool = True):
        
        if not in_obstacle_frame:
            position = self.pose.transform_position_from_reference_to_local(position)

        pos_norm = LA.norm(position)
        if not pos_norm:
            return np.ones(position.shape) / position.shape[0]
        
        normal = position / pos_norm
        
        if not in_obstacle_frame:
            normal = self.pose.transform_direction_from_reference_to_local(normal)

        return normal

    def get_point_on_surface(
        self,
        position: np.ndarray,
        in_obstacle_frame: bool = True):
        """ Returns the point on the surface from the center with respect to position. """
        if not in_obstacle_frame:
            position = self.pose.transform_position_from_reference_to_local(position)

        pos_norm = LA.norm(position)
        if not pos_norm:
            surface_point = np.zeros(position.shape)
            surface_point[0] = self.axes_length[0]

        else:
            surface_point = position / pos_norm

        if not in_obstacle_frame:
            surface_point = self.pose.transform_position_from_local_to_reference(surface_point)

        return surface_point


class EllipseWithCovariance(obstacles.Obstacle):
    # TODO: Create ellipse based on covariance matrix -> allows combining with learning (GMM)
    pass
