from typing import Optional

import numpy as np
from numpy import linalg as LA

import shapely

from vartools import linalg
from vartools.math import get_intersection_with_circle, CircleIntersectionType

from dynamic_obstacle_avoidance import obstacles


class HyperSphere(obstacles.Obstacle):
    # TODO: is this really worth it? Speed up towards ellipse is minimal...
    def __init__(self, radius: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.radius = radius

    def get_gamma(self, position: np.ndarray, in_obstacle_frame: bool = True):
        """Gets a gamma which is not directly related to the axes length."""
        distance = self.get_point_on_surface(
            position=position, in_obstacle_frame=in_obstacle_frame
        )

        gamma = distance + 1

        if self.is_boundary:
            gamma = 1 / gamma

        return gamma

    def get_normal_direction(
        self, position: np.ndarray, in_obstacle_frame: bool = True
    ):
        if not in_obstacle_frame:
            position = self.pose.transform_position_to_relative(position)

        pos_norm = LA.norm(position)
        if not pos_norm:
            return np.ones(position.shape) / position.shape[0]

        normal = position / pos_norm

        if not in_obstacle_frame:
            normal = self.pose.transform_position_from_relative(normal)

        return normal

    def get_point_on_surface(
        self,
        position: np.ndarray,
        in_obstacle_frame: bool = True,
    ):
        """Returns the point on the surface from the center with respect to position."""

        if not in_obstacle_frame:
            position = self.pose.transform_position_to_relative(position)

        pos_norm = LA.norm(position)
        if not pos_norm:
            surface_point = np.zeros(position.shape)
            surface_point[0] = self.semiaxes[0]

        else:
            surface_point = position / pos_norm

        if not in_obstacle_frame:
            surface_point = self.pose.transform_position_from_relative(surface_point)

        return surface_point
