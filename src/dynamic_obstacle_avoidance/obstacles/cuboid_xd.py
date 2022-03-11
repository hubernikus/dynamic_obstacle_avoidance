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


class CuboidWithAxes(obstacles.Obstacle):
    """Cuboid with axes length.
    methods such as `extend_hull_around_reference'."""
    def __init__(
        self,
        axes_length: np.ndarray,
        *args,
        **kwargs):
        super().__init__(*args, **kwargs)

        self.axes_length = axes_length

    @property
    def axes_length(self) -> np.ndarray:
        return self._axes_length

    @axes_length.setter
    def axes_length(self, value: np.ndarray):
        value = np.array(value)

        if any(value <= 0):
            raise ValueError("Zero axes input not tolerated.")
        self._axes_length = value

    @property
    def semiaxes(self) -> np.ndarray:
        return self._axes_length / 2.0

    def get_characteristic_length(self) -> np.ndarray:
        """Get a characeteric (or maximal) length of the obstacle.
        For an ellipse obstacle,the longest axes."""
        return np.prod(self.semiaxes + self.margin_absolut) ** (1 / self.dimension)

    def set_reference_point(self, position: np.ndarray, in_obstacle_frame=True) -> None:
        """Set the reference point"""
        # TODO: this can be portet to the `_base` class
        if self.get_gamma(position=position, in_obstacle_frame=in_obstacle_frame) >= 1:
            raise NotImplementedError(
                "Automatic reference point extension is not implemented."
            )

        if not in_obstacle_frame:
            position = self.pose.transform_position_from_reference_to_local(position)

        self._reference_point = position

    def get_reference_point(
        self, in_obstacle_frame: bool = False, in_global_frame: bool = None
    ) -> np.ndarray:
        if in_global_frame is not None:
            # Legacy value
            in_obstacle_frame = not (in_global_frame)

        if not in_obstacle_frame:
            return self.pose.transform_position_from_local_to_reference(
                self._reference_point
            )
        else:
            return self._reference_point

    def get_shapely(self, semiaxes=None):
        """Get shapely of a standard ellipse."""
        if semiaxes is None:
            semiaxes = self.semiaxes

        position = self.center_position
        orientation_in_degree = self.orientation_in_degree

        cuboid = shapely.geometry.box(
            position[0] - semiaxes[0],
            position[1] - semiaxes[1],
            position[0] + semiaxes[0],
            position[1] + semiaxes[1],
        )

        cuboid = shapely.affinity.rotate(cuboid, orientation_in_degree)

        return cuboid

    def get_boundary_xy(self, in_obstacle_frame: bool = False):
        """Two dimensional xy-values of the boundary -
        shapely is used for the creation"""
        if in_obstacle_frame:
            raise NotImplementedError()

        cuboid = self.get_shapely()
        return cuboid.exterior.xy

    def get_boundary_with_margin_xy(self, in_obstacle_frame: bool = False):
        if in_obstacle_frame:
            raise NotImplementedError()

        cuboid = self.get_shapely(semiaxes=self.semiaxes)
        if self.is_boundary:
            cuboid = cuboid.buffer((-1) * self.margin_absolut)
        else:
            cuboid = cuboid.buffer(self.margin_absolut)

        return cuboid.exterior.xy

    def get_normal_direction(
        self, position, in_obstacle_frame: bool = True, in_global_frame: bool = None
    ):

        if in_global_frame is not None:
            # Legacy value
            in_obstacle_frame = not (in_global_frame)

        if not in_obstacle_frame:
            position = self.pose.transform_position_from_reference_to_local(position)

        ind_relevant = np.abs(position) > self.semiaxes

        if not any(ind_relevant):
            # Take the inverse
            gamma = self.get_gamma(position, in_obstacle_frame=True)
            position = position / gamma ** 2
            ind_relevant = np.abs(position) > self.semiaxes

            breakpoint()
            # return np.ones(position.shape) / position.shape[0]

        # relevant_axes =
        # relevant_pos = position

        normal = np.zeros(position.shape)
        normal[ind_relevant] = position[ind_relevant] - np.copysign(
            self.semiaxes[ind_relevant], position[ind_relevant]
        )
        # No normalization chack needed, since at least one axes was relevatn
        normal = normal / LA.norm(normal)

        if not in_obstacle_frame:
            normal = self.pose.transform_direction_from_local_to_reference(normal)


        return normal

    def get_distance_to_surface(self, position, in_obstacle_frame: bool = True):
        if not in_obstacle_frame:
            position = self.pose.transform_position_from_reference_to_local(position)

        relative_position = np.abs(position) - self.semiaxes

        if any(relative_position > 0):
            relative_position = np.maximum(relative_position, 0)
            distance = LA.norm(relative_position)
            if distance > self.margin_absolut:
                return distance

        relative_position = relative_position / (self.semiaxes + self.margin_absolut)
        return np.max(relative_position)

    def get_gamma(
        self, position, in_obstacle_frame: bool = True, in_global_frame: bool = None
    ):

        if in_global_frame is not None:
            in_obstacle_frame = not (in_global_frame)

        distance = self.get_distance_to_surface(
            position=position, in_obstacle_frame=in_obstacle_frame
        )

        return distance + 1

    def get_point_on_surface(self, position, in_obstacle_frame: bool = True):
        if not in_obstacle_frame:
            cube_position = self.pose.transform_position_from_reference_to_local(
                position
            )

        cube_position = cube_position / self.semiaxes

        ind_max = np.argmax(cube_position)

        return position * self.semiaxes[ind_max] / position[ind_max]
