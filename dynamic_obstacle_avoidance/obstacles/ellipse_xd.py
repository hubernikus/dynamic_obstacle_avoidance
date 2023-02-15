""" Ellipse Obstacle for Obstacle Avoidance and Visualization Purposes. """
# Author Lukas Huber
# Email lukas.huber@epfl.ch
# License BSD

# import warnings
from typing import Optional

import numpy as np
from numpy import linalg as LA

import shapely

from vartools import linalg
from vartools.math import get_intersection_with_circle, CircleIntersectionType

from dynamic_obstacle_avoidance import obstacles


class EllipseWithAxes(obstacles.Obstacle):
    """The presented class does not have
    methods such as `extend_hull_around_reference'.
    """

    def __init__(self, axes_length: np.ndarray, curvature: float = 1, *args, **kwargs):
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
    def axes_with_margin(self) -> np.ndarray:
        if self.is_boundary:
            return self._axes_length - 2 * self.margin_absolut
        else:
            return self._axes_length + 2 * self.margin_absolut

    @property
    def semiaxes(self) -> np.ndarray:
        return self._axes_length / 2.0

    @property
    def semiaxes_with_magin(self) -> np.ndarray:
        if self.is_boundary:
            return self._axes_length / 2.0 - self.margin_absolut
        else:
            return self._axes_length / 2.0 + self.margin_absolut

    def set_reference_point(
        self,
        position: np.ndarray,
        in_obstacle_frame: bool = True,
        in_global_frame: bool = None,
    ) -> None:
        """Set the reference point"""
        # TODO: this can be portet to the `_base` class

        if in_global_frame is not None:
            # Legacy value
            in_obstacle_frame = not (in_global_frame)

        if (
            not self.is_boundary
            and self.get_gamma(position=position, in_obstacle_frame=in_obstacle_frame)
            > 1
        ) or (
            self.is_boundary
            and self.get_gamma(position=position, in_obstacle_frame=in_obstacle_frame)
            < 1
        ):
            raise NotImplementedError(
                "Automatic reference point extension is not implemented."
            )

        if not in_obstacle_frame:
            position = self.pose.transform_position_to_relative(position)

        self._reference_point = position

    def get_reference_point(
        self, in_obstacle_frame: bool = False, in_global_frame: bool = None
    ) -> np.ndarray:
        if in_global_frame is not None:
            # Legacy value
            in_obstacle_frame = not (in_global_frame)

        if not in_obstacle_frame:
            return self.pose.transform_position_from_relative(self._reference_point)
        else:
            return self._reference_point

    def get_characteristic_length(self) -> None:
        """Get a characeteric (or maximal) length of the obstacle.
        For an ellipse obstacle,the longest axes."""
        return np.prod(self.semiaxes + self.margin_absolut) ** (1 / self.dimension)

    def get_shapely(self, semiaxes: np.ndarray = None):
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
        in_global_frame: Optional[bool] = None,
        margin_absolut: Optional[float] = None,
    ):
        """Gets a gamma which is not directly related to the axes length."""
        if in_global_frame is not None:
            in_obstacle_frame = not (in_global_frame)

        if not in_obstacle_frame:
            position = self.pose.transform_position_to_relative(position)

        if margin_absolut is None:
            margin_absolut = self.margin_absolut

        surface_point = self.get_point_on_surface(
            position=position,
            in_obstacle_frame=True,
            margin_absolut=margin_absolut,
        )

        distance_surface = LA.norm(surface_point) * self.distance_scaling
        distance_position = LA.norm(position) * self.distance_scaling

        if distance_position > distance_surface:
            distance = LA.norm(position - surface_point)
        else:
            distance = distance_position / distance_surface - 1

        gamma = distance * self.distance_scaling + 1
        if self.is_boundary:
            gamma = 1 / gamma

        return gamma

    def get_local_radius(
        self,
        position: np.ndarray,
        in_relative_frame: bool = True,
        in_global_frame: Optional[bool] = None,
        margin_absolut: Optional[float] = None,
    ) -> float:
        if in_global_frame is not None:
            in_relative_frame = not (in_global_frame)

        if not in_relative_frame:
            in_relative_frame = True
            position = self.pose.transform_position_to_relative(position)

        surface_point = self.get_point_on_surface(
            position=position,
            in_obstacle_frame=in_relative_frame,
            margin_absolut=margin_absolut,
        )

        return LA.norm(surface_point)

    def get_normal_direction(
        self,
        position: np.ndarray,
        in_obstacle_frame: bool = True,
        in_global_frame: bool = None,
    ) -> np.ndarray:
        if in_global_frame is not None:
            # Legacy value
            in_obstacle_frame = not (in_global_frame)

        if not in_obstacle_frame:
            position = self.pose.transform_position_to_relative(position)

        normal = (
            2
            * self.curvature
            / self.axes_with_margin
            * (position / self.axes_with_margin) ** (2 * self.curvature - 1)
        )

        # Normalize
        normal_norm = LA.norm(normal)
        if normal_norm:
            normal = normal / normal_norm
        else:
            normal[0] = 1

        if not in_obstacle_frame:
            normal = self.pose.transform_direction_from_relative(normal)

        if self.is_boundary:
            normal = (-1) * normal

        return normal

    def get_surface_intersection_with_line(
        self, point0: np.ndarray, point1: np.ndarray, in_global_frame: bool = False
    ):
        if in_global_frame:
            point0 = self.pose.transfrom_positions_to_relative(point0)
            point1 = self.pose.transfrom_positions_to_relative(point1)

        point0 = point0 / self.semiaxes_with_magin
        point1 = point1 / self.semiaxes_with_magin

        surface_point = get_intersection_with_circle(
            point0, direction=(point1 - point0), radius=1, only_positive=True
        )

        if in_global_frame:
            surface_point = self.pose.transform_position_from_relative(surface_point)

        return surface_point

    def get_point_on_surface(
        self,
        position: np.ndarray,
        in_obstacle_frame: bool = True,
        margin_absolut: float = None,
        in_global_frame: float = None,
    ):
        """Returns the point on the surface from the center with respect to position."""
        if in_global_frame is not None:
            # Legacy value
            in_obstacle_frame = not (in_global_frame)

        if not in_obstacle_frame:
            position = self.pose.transform_position_to_relative(position)

        # Position in the circle-world
        if margin_absolut is None:
            circle_position = position / self.semiaxes_with_magin
        else:
            circle_position = position / (self.semiaxes + margin_absolut)

        pos_norm = LA.norm(circle_position)
        if not pos_norm:
            surface_point = np.zeros(position.shape)
            if margin_absolut is None:
                surface_point[0] = self.semiaxes_with_magin[0]
            else:
                surface_point[0] = self.semiaxes[0] + margin_absolut

        else:
            surface_point = position / pos_norm

        # surface_point = surface_point * self.semiaxes

        if not in_obstacle_frame:
            surface_point = self.pose.transform_position_from_relative(surface_point)

        return surface_point

    def get_intersection_with_surface(
        self,
        start_position: np.ndarray,
        direction: np.ndarray,
        in_global_frame: bool = False,
        intersection_type=CircleIntersectionType.CLOSE,
    ) -> Optional[np.ndarray]:
        if in_global_frame:
            # Currently only implemented for ellipse
            start_position = self.pose.transform_position_to_relative(start_position)
            direction = self.pose.transform_direction_to_relative(direction)

        # Stretch according to ellipse axes (radius)
        rel_pos = start_position / self.axes_with_margin
        rel_dir = direction / self.axes_with_margin

        # Intersection with unit circle
        surface_rel_pos = get_intersection_with_circle(
            start_position=rel_pos,
            direction=rel_dir,
            radius=0.5,
            intersection_type=intersection_type,
        )

        if surface_rel_pos is None:
            return None

        # Relative
        surface_pos = surface_rel_pos * self.axes_with_margin

        if in_global_frame:
            return self.pose.transform_position_from_relative(surface_pos)
        else:
            return surface_pos


class EllipseWithCovariance(obstacles.Obstacle):
    # TODO: Create ellipse based on covariance matrix -> allows combining with learning (GMM)
    pass
