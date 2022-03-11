""" Ellipse Obstacle for Obstacle Avoidance and Visualization Purposes. """
# Author Lukas Huber
# Email lukas.huber@epfl.ch
# License BSD

# import warnings

import numpy as np
from numpy import linalg as LA

import shapely

from vartools import linalg

from dynamic_obstacle_avoidance import obstacles


class EllipseWithAxes(obstacles.Obstacle):
    """The presented class does not have
    methods such as `extend_hull_around_reference'.
    """
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
        value = np.array(value)
        if any(value <= 0):
            raise ValueError("Zero axes input not tolerated.")
        self._axes_length = value

    @property
    def axes_with_margin(self)-> np.ndarray:
        if self.is_boundary:
            return self._axes_length - 2*self.margin_absolut
        else:
            return self._axes_length + 2*self.margin_absolut

    @property
    def semiaxes(self) -> np.ndarray:
        return self._axes_length / 2.0

    @property
    def semiaxes_with_magin(self) -> np.ndarray:
        if self.is_boundary:
            return self._axes_length / 2.0 - self.margin_absolut
        else:
            return self._axes_length / 2.0 + self.margin_absolut

    def get_characteristic_length(self):
        """Get a characeteric (or maximal) length of the obstacle.
        For an ellipse obstacle,the longest axes."""
        return np.prod(self.semiaxes + self.margin_absolut) ** (1 / self.dimension)

    def get_shapely(self, semiaxes=None):
        if semiaxes is None:
            semiaxes = self.semiaxes

        position = self.center_position
        orientation_in_degree = self.orientation_in_degree

        ellipse = shapely.geometry.Point(position[0], position[1]).buffer(1)
        ellipse = shapely.affinity.scale(ellipse, semiaxes[0], semiaxes[1])
        ellipse = shapely.affinity.rotate(ellipse, orientation_in_degree)

        return ellipse

    def get_boundary_xy(self, in_global_frame: bool = True):
        """Two dimensional xy-values of the boundary -
        shapely is used for the creation"""
        if not in_global_frame:
            raise NotImplementedError()

        ellipse = self.get_shapely()
        return ellipse.exterior.xy

    def get_boundary_with_margin_xy(self, in_global_frame: bool = True):
        if not in_global_frame:
            raise NotImplementedError()

        if self.is_boundary:
            ellipse = self.get_shapely(semiaxes=self.semiaxes - self.margin_absolut)
        else:
            ellipse = self.get_shapely(semiaxes=self.semiaxes + self.margin_absolut)

        return ellipse.exterior.xy

    def get_gamma(
        self,
        position: np.ndarray,
        in_obstacle_frame: bool = True,
        in_global_frame: bool = None,
    ):
        """Gets a gamma which is not directly related to the axes length."""

        if in_global_frame is not None:
            in_obstacle_frame = not (in_global_frame)

        if not in_obstacle_frame:
            position = self.pose.transform_position_from_reference_to_local(position)

        surface_point = self.get_point_on_surface(
            position=position, in_obstacle_frame=True
        )

        distance_surface = LA.norm(surface_point)
        distance_position = LA.norm(position)

        if distance_position > distance_surface:
            distance = LA.norm(position - surface_point)
        else:
            distance = distance_position / distance_surface - 1

        return distance + 1

    def get_normal_direction(
        self,
        position: np.ndarray,
        in_obstacle_frame: bool = True,
        in_global_frame: bool = None,
    ):
        if in_global_frame is not None:
            # Legacy value
            in_obstacle_frame = not (in_global_frame)

        if not in_obstacle_frame:
            position = self.pose.transform_position_from_reference_to_local(position)

        normal = (2*self.curvature/self.axes_with_margin
                  *(position/self.axes_with_margin)**(2*self.curvature-1)
                  )

        # Normalize
        normal_norm = LA.norm(normal)
        if normal_norm:
            normal = normal / normal_norm
        else:
            normal[0] = 1
        if not in_obstacle_frame:
            normal = self.pose.transform_direction_from_local_to_reference(normal)

        return normal

    def get_point_on_surface(
        self, position: np.ndarray, in_obstacle_frame: bool = True
    ):
        """Returns the point on the surface from the center with respect to position."""
        if not in_obstacle_frame:
            position = self.pose.transform_position_from_reference_to_local(position)

        # Position in the circle-world
        circle_position = position / self.semiaxes_with_magin

        pos_norm = LA.norm(circle_position)
        if not pos_norm:
            surface_point = np.zeros(position.shape)
            surface_point[0] = self.semiaxes_with_magin[0]

        else:
            surface_point = position / pos_norm

        # surface_point = surface_point * self.semiaxes

        if not in_obstacle_frame:
            surface_point = self.pose.transform_position_from_local_to_reference(
                surface_point
            )

        return surface_point


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
        
        return distance + 1

    def get_normal_direction(
        self, position: np.ndarray, in_obstacle_frame: bool = True
    ):

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
        self, position: np.ndarray, in_obstacle_frame: bool = True
    ):
        """Returns the point on the surface from the center with respect to position."""
        if not in_obstacle_frame:
            position = self.pose.transform_position_from_reference_to_local(position)

        pos_norm = LA.norm(position)
        if not pos_norm:
            surface_point = np.zeros(position.shape)
            surface_point[0] = self.semiaxes[0]

        else:
            surface_point = position / pos_norm

        if not in_obstacle_frame:
            surface_point = self.pose.transform_position_from_local_to_reference(
                surface_point
            )

        return surface_point


class EllipseWithCovariance(obstacles.Obstacle):
    # TODO: Create ellipse based on covariance matrix -> allows combining with learning (GMM)
    pass
