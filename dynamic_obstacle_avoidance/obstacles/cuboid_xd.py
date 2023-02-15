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
from vartools.math import get_intersection_with_circle, IntersectionType

from dynamic_obstacle_avoidance import obstacles


class CuboidXd(obstacles.Obstacle):
    """Cuboid with axes length.
    methods such as `extend_hull_around_reference'."""

    def __init__(self, axes_length: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.axes_length = np.array(axes_length)

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

    def get_characteristic_length(self) -> np.ndarray:
        """Get a characeteric (or maximal) length of the obstacle.
        For an ellipse obstacle,the longest axes."""
        return np.prod(self.semiaxes + self.margin_absolut) ** (1 / self.dimension)

    def set_reference_point(
        self,
        position: np.ndarray,
        in_global_frame: bool = False,
        in_obstacle_frame: Optional[bool] = None,
    ) -> None:
        """Set the reference point"""
        # TODO: this can be portet to the `_base` class
        if in_obstacle_frame is None:
            in_obstacle_frame = not in_global_frame

        if self.get_gamma(position=position, in_obstacle_frame=in_obstacle_frame) >= 1:
            raise NotImplementedError(
                "Automatic reference point extension is not implemented."
            )

        if in_global_frame:
            position = self.pose.transform_position_to_relative(position)

        self._reference_point = position

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
            position = self.pose.transform_position_to_relative(position)

        ind_relevant = np.abs(position) > self.semiaxes

        if not any(ind_relevant):
            # Mirror at the boundary (Take the inverse)
            minimum_factor = max(np.abs(position) / self.semiaxes)

            # gamma = self.get_gamma(
            # position, in_obstacle_frame=True, is_boundary=False, margin_absolut=0)
            position = position / minimum_factor**2
            ind_relevant = np.abs(position) > self.semiaxes

            # return np.ones(position.shape) / position.shape[0]

        normal = np.zeros(position.shape)
        normal[ind_relevant] = position[ind_relevant] - np.copysign(
            self.semiaxes[ind_relevant], position[ind_relevant]
        )

        normal_norm = LA.norm(normal)
        if not normal_norm:
            normal[0] = 1
            normal_norm = 1

        # No normalization chack needed, since at least one axes was relevatn
        normal = normal / LA.norm(normal)

        if not in_obstacle_frame:
            normal = self.pose.transform_direction_from_relative(normal)

        if self.is_boundary:
            normal = (-1) * normal

        return normal

    def get_distance_to_surface(
        self, position, in_obstacle_frame: bool = True, margin_absolut: float = None
    ):
        if not in_obstacle_frame:
            position = self.pose.transform_position_to_relative(position)

        if margin_absolut is None:
            margin_absolut = self.margin_absolut

        relative_position = np.abs(position) - self.semiaxes

        if any(relative_position > 0):
            # Corner case is treated separately
            relative_position = np.maximum(relative_position, 0)
            distance = LA.norm(relative_position)

            if distance > margin_absolut:
                return distance - margin_absolut

            distance = margin_absolut - distance

        else:
            distance = margin_absolut + (-1) * np.max(relative_position)

        # Case: within margin but outside boundary -> edges have to be rounded
        pos_norm = LA.norm(position)

        # Negative distance [0, -1] beacuse inside
        return (-1) * distance / (pos_norm + distance)

        # relative_position = relative_position / (self.semiaxes + self.margin_absolut)
        # return np.max(relative_position) - self.margin_absolut

    def get_gamma(
        self,
        position,
        in_obstacle_frame: bool = True,
        in_global_frame: Optional[bool] = None,
        margin_absolut=None,
        is_boundary=None,
    ):
        if in_global_frame is not None:
            in_obstacle_frame = not (in_global_frame)

        distance_surface = self.get_distance_to_surface(
            position=position,
            in_obstacle_frame=in_obstacle_frame,
            margin_absolut=margin_absolut,
        )

        # Allow to have various behavior
        distance_surface = distance_surface * self.distance_scaling

        is_boundary = is_boundary or self.is_boundary
        if distance_surface < 0:
            # or (distance_surface > 0  and not is_boundary)):
            self.boundary_power_factor = 1
            distance_center = LA.norm(position)
            gamma = distance_center / (distance_center - distance_surface)

            # gamma = distance_center / (distance_center - distance_surface)

            # print("gamma boundary", gamma)
            # gamma = (1 - gamma) ** self.boundary_power_factor
            gamma = gamma**self.boundary_power_factor

        else:
            gamma = distance_surface * self.distance_scaling + 1

        if is_boundary:
            return 1 / gamma

        return gamma

    def get_point_on_surface(
        self,
        position,
        in_obstacle_frame: bool = True,
        in_global_frame: bool = None,
        margin_absolut: bool = None,
    ):
        if in_global_frame is not None:
            # Legacy value
            in_obstacle_frame = not (in_global_frame)

        if not in_obstacle_frame:
            position = self.pose.transform_position_to_relative(position)

        if margin_absolut is None:
            semiaxes = self.semiaxes_with_magin
        else:
            semiaxes = self.semiaxes + margin_absolut

        cube_position = position / semiaxes

        ind_max = np.argmax(cube_position)

        return position * semiaxes[ind_max] / position[ind_max]

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

    def get_intersection_with_surface(
        self,
        start_position: np.ndarray,
        direction: np.ndarray,
        in_global_frame: bool = False,
        intersection_type=IntersectionType.CLOSE,
    ) -> Optional[np.ndarray]:
        if in_global_frame:
            start_position = self.pose.transform_position_to_relative(start_position)
            direction = self.pose.transform_direction_to_relative(direction)

        positive_plane_intersect = (
            0.5 * self.axes_with_margin - start_position
        ) / direction
        negative_plane_intersect = (
            -0.5 * self.axes_with_margin - start_position
        ) / direction

        if self.get_gamma(start_position, in_global_frame=False) > 1:
            if (
                np.dot(
                    direction / self.axes_with_margin,
                    start_position / self.axes_with_margin,
                )
                > 0
            ):
                direction = (-1) * direction
                positive_plane_intersect = (-1) * positive_plane_intersect
                negative_plane_intersect = (-1) * negative_plane_intersect

            all_intersections = np.vstack(
                (negative_plane_intersect, positive_plane_intersect)
            )
            # Is outside of the obstacle - we neglect the direction
            # all_intersects = np.abs(all_intersect)
            if intersection_type == IntersectionType.BOTH:
                min_dist = np.max(np.min(all_intersections, axis=0))
                pos_min = start_position + min_dist * direction

                if self.get_gamma(pos_min, in_global_frame=False) < 1:
                    return None

                max_dist = np.min(np.max(all_intersections, axis=0))

                pos_max = start_position + max_dist * direction
                if in_global_frame:
                    pos_min = self.pose.transform_position_from_relative(pos_min)
                    pos_max = self.pose.transform_position_from_relative(pos_max)
                return np.hstack((pos_min, pos_max)).T

            if intersection_type == IntersectionType.CLOSE:
                distance = np.max(np.min(all_intersections, axis=0))

            elif intersection_type == IntersectionType.FAR:
                distance = np.min(np.max(all_intersections, axis=0))

            position = start_position + distance * direction
            if self.get_gamma(position, in_global_frame=False) < 1:
                breakpoint()
                return None

            if in_global_frame:
                return self.pose.transform_position_from_relative(position)
            return position

        all_intersections = np.vstack(
            (negative_plane_intersect, positive_plane_intersect)
        )
        pos_ind = all_intersections >= 0
        positive_intersect = all_intersections[pos_ind]
        negative_intersect = all_intersections[np.logical_not(pos_ind)]

        # Else: we are inside the obstacle
        if intersection_type == IntersectionType.BOTH:
            min_dist = np.min(negative_intersect)
            max_dist = np.max(positive_intersect)

            pos_min = start_position + min_dist * direction
            pos_max = start_position + max_dist * direction
            if in_global_frame:
                pos_min = self.pose.transform_position_from_relative(pos_min)
                pos_max = self.pose.transform_position_from_relative(pos_max)
            return np.hstack((pos_min, pos_max)).T

        if intersection_type == IntersectionType.FAR:
            distance = np.max(negative_intersect)

        elif intersection_type == IntersectionType.CLOSE:
            distance = np.min(positive_intersect)

        position = start_position + distance * direction
        if in_global_frame:
            return self.pose.transform_position_from_relative(position)
        return position


def test_cube_intersection():
    """Intersection test in 2D"""
    cube = CuboidXd(center_position=np.array([0, 0]), axes_length=np.array([5.0, 1]))

    # Position 1
    position = np.array([1.0, 0])
    direction = np.array([-1.0, -1])

    intersection = cube.get_intersection_with_surface(
        start_position=position,
        direction=direction,
        in_global_frame=True,
        intersection_type=IntersectionType.CLOSE,
    )
    assert intersection[1] == -0.5

    intersection = cube.get_intersection_with_surface(
        start_position=position,
        direction=direction,
        in_global_frame=True,
        intersection_type=IntersectionType.FAR,
    )
    assert intersection[1] == 0.5

    # Position 2
    position = np.array([0.0, 1.0])
    direction = np.array([-1, -1])
    intersection = cube.get_intersection_with_surface(
        start_position=position,
        direction=direction,
        in_global_frame=True,
        intersection_type=IntersectionType.CLOSE,
    )

    assert intersection[1] == 0.5

    intersection = cube.get_intersection_with_surface(
        start_position=position,
        direction=direction,
        in_global_frame=True,
        intersection_type=IntersectionType.FAR,
    )
    assert intersection[1] == -0.5


def test_cube_outside_position():
    position = np.array([0.55555556, -0.55555556])
    direction = np.array([-0.55555556, -0.94444444])

    cube = CuboidXd(center_position=np.array([0, 0]), axes_length=np.array([1.0, 4.0]))

    intersection = cube.get_intersection_with_surface(
        start_position=position,
        direction=direction,
        in_global_frame=True,
        intersection_type=IntersectionType.CLOSE,
    )

    assert intersection[0] == 0.5
    assert (
        intersection[1] < position[1] and intersection[1] > -0.5 * cube.axes_length[1]
    )

    position = np.array([2.2413793103448283, -5.0])
    direction = np.array([-2.2413793103448283, 6.5])
    intersection = cube.get_intersection_with_surface(
        start_position=position,
        direction=direction,
        in_global_frame=True,
        intersection_type=IntersectionType.CLOSE,
    )
    assert np.isclose(intersection[0], 0.5)

    position = np.array([-0.5172413793103443, -3.9655172413793105])
    direction = np.array([0.5172413793103443, 5.4655172413793105])
    intersection = cube.get_intersection_with_surface(
        start_position=position,
        direction=direction,
        in_global_frame=True,
        intersection_type=IntersectionType.CLOSE,
    )
    assert np.isclose(intersection[1], -2)


if (__name__) == "__main__":
    # test_cube_intersection()
    test_cube_outside_position()
