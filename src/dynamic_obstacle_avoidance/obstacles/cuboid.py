"""
Polygon Obstacle for Avoidance Calculations
"""
# Author: Lukas Huber
# Date: created: 2020-02-28
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import sys
import warnings
import copy
import time

from math import pi

import numpy as np
from numpy import linalg as LA

import shapely

# from shapely.geometry import Polygon

from vartools.angle_math import *

from ._base import GammaType
from .polygon import Polygon


class Cuboid(Polygon):
    def __init__(
        self,
        axes_length=[1, 1],
        margin_absolut=0,
        expansion_speed_axes=None,
        wall_thickness=None,
        relative_expansion_speed=None,
        *args,
        **kwargs
    ):
        """
        This class defines obstacles to modulate the DS around it
        At current stage the function focuses on Ellipsoids,
        but can be extended to more general obstacles
        """
        self.axes_length = np.array(axes_length)

        # Different expansion models [relative vs. absolute]
        self._expansion_speed_axes = None
        self._relative_expansion_speed = None

        if expansion_speed_axes is not None:
            self._expansion_speed_axes = np.array(expansion_speed_axes)
            is_deforming = True
        elif relative_expansion_speed is not None:
            self._relative_expansion_speed = np.array(relative_expansion_speed)
            is_deforming = True
        else:
            is_deforming = False

        self.dim = self.axes_length.shape[0]  # Dimension of space

        if not self.dim == 2:
            raise ValueError(
                "Cuboid not yet defined for dimensions= {}".format(self.dim)
            )

        edge_points = self.get_edge_points_from_axes()

        super().__init__(
            *args,
            is_deforming=is_deforming,
            edge_points=edge_points,
            absolute_edge_position=False,
            margin_absolut=margin_absolut,
            **kwargs
        )

        self.wall_thickness = wall_thickness

        # Create shapely object
        # TODO update shapely position (!?)
        edge = self.edge_points

        edge = np.vstack((edge.T, edge[:, 0]))
        self._shapely = shapely.geometry.Polygon(edge).buffer(self.margin_absolut)

    @property
    def axes_length(self):
        return self._axes_length

    @axes_length.setter
    def axes_length(self, value):
        self._axes_length = np.maximum(value, np.zeros(value.shape))

    @property
    def global_outer_edge_points(self):
        if self.wall_thickness is not None:
            outer_axes_length = self.axes_length + 2 * self.wall_thickness
            edge_points = self.get_edge_points_from_axes(axes_length=outer_axes_length)
            edge_points = self.transform_relative2global(edge_points)
            return edge_points
        else:
            return None

    @property
    def expansion_speed_axes(self):
        if self._expansion_speed_axes is not None:
            return self._expansion_speed_axes
        else:
            return self._relative_expansion_speed * self.axes_length

    @expansion_speed_axes.setter
    def expansion_speed_axes(self, value):
        self._expansion_speed_axes = value

    def get_reference_length(self):
        return np.linalg.norm(self.axes_length) / 2.0 + self.margin_absolut

    def get_relative_expansion(self, delta_time):
        if self._relative_expansion_speed is not None:
            exp_speed = self._relative_expansion_speed
        else:
            exp_speed = self._expansion_speed_axes / self.axes_length
        return 1 + exp_speed * delta_time

    def get_edge_points_from_axes(self, axes_length=None):
        if axes_length is None:
            axes_length = self.axes_length

        edge_points = np.zeros((self.dim, 4))
        edge_points[:, 2] = axes_length / 2.0 * np.array([1, 1])
        edge_points[:, 3] = axes_length / 2.0 * np.array([-1, 1])
        edge_points[:, 0] = axes_length / 2.0 * np.array([-1, -1])
        edge_points[:, 1] = axes_length / 2.0 * np.array([1, -1])

        return edge_points

    def update_deforming_obstacle(self, delta_time):
        rel_expansion = self.get_relative_expansion(delta_time)
        self.axes_length = self.axes_length * rel_expansion
        self.wall_thickness = self.wall_thickness * rel_expansion
        self.edge_points = self.get_edge_points_from_axes()

        self.draw_obstacle()
