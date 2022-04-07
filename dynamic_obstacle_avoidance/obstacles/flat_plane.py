"""
Dummy robot models for cluttered obstacle environment + testing
"""
from math import pi

import numpy as np
from numpy import linalg as LA

from ._base import Obstacle


class FlatPlane(Obstacle):
    """
    Flat Plan which has refernce_direct=normal_direction

    Properties
    ----------
    position
    """

    def __init__(
        self,
        center_position,
        normal,
        reference_distance=1,
        width=1,
        height=1,
        obstacle_color=None,
    ):
        super().__init__(center_position=center_position)
        # self.position = position

        self.normal = normal / LA.norm(normal)
        self.orientation = None

        # If gamma should be scaled (depending on the environment)
        self.reference_distance = reference_distance

        # For displaying purposes
        self.width = width
        self.height = height

        # self.dim = self.position.shape[0]
        # self._rotation_matrix =

    @property
    def normal(self):
        return self._normal

    @normal.setter
    def normal(self, value):
        self._normal = np.array(value)

    def get_normal_direction(self, position, in_global_frame=False):
        return self.normal

    def get_reference_direction(self, position, in_global_frame=False):
        return self.normal

    def get_gamma(self, position, in_global_frame=False):
        if in_global_frame:
            position = position - self.position
        dist = (position).dot(self.normal)
        dist = dist / self.reference_distance
        return dist

    def draw_obstacle(self, n_grid=None):
        """Draw the obstacle for a 2D environment."""
        if self.dimension != 2:
            raise NotImplementedError("Drawing of obstacle not implemented for dim!=2")
        self.boundary_points_local = np.zeros((self.dimension, 4))

        tangent = np.array([-self.normal[1], self.normal[0]])
        self.boundary_points_local[:, 0] = (
            self.center_position - tangent * self.width / 2.0
        )

        self.boundary_points_local[:, 1] = (
            self.boundary_points_local[:, 0] - self.normal * self.height
        )

        self.boundary_points_local[:, 2] = (
            self.boundary_points_local[:, 1] + tangent * self.width
        )

        self.boundary_points_local[:, 3] = (
            self.boundary_points_local[:, 2] + self.normal * self.height
        )

        # if not self.margin_absolut: # zero margin
        self.boundary_points_margin_local = self.boundary_points_local
