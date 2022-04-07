"""
Two dimensional boundary obstacles with gaps. Agent can exit and enter them.
"""
# Author: Lukas Huber
# Date: 2021-05-12
# Email:lukas.huber@epfl.ch

import sys
import os
import warnings
import copy

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib import ticker

from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry.polygon import LinearRing

from vartools.angle_math import (
    angle_is_in_between,
    angle_difference_directional,
)

from dynamic_obstacle_avoidance.obstacles import Cuboid


class BoundaryCuboidWithGaps(Cuboid):
    """2D boundary obstacle which allows to include doors (e.g. rooms).
    Currently only implemented for one door [can be extend in the future]."""

    def __init__(
        self, *args, gap_points_absolute=None, gap_points_relative=None, **kwargs
    ):
        kwargs["is_boundary"] = True
        kwargs["tail_effect"] = False

        # kwargs['sticky'] = True

        if gap_points_absolute is not None:
            self._gap_points = (
                gap_points_absolute
                - np.tile(self.center_position, (gap_points_absolute.shape[1], 1)).T
            )
        elif gap_points_relative is not None:
            self._gap_points = np.array(gap_points_relative)

        else:
            raise Exception("No gap array assigned")

        if sys.version_info > (3, 0):  # TODO: remove in future
            super().__init__(*args, **kwargs)
        else:
            super(BoundaryCuboidWithGaps, **kwargs)

        self.guiding_reference_point = True
        self.has_gap_points = True

    @property
    def gap_center(self):
        """Local gap center."""
        return np.mean(self._gap_points, axis=1)

    @property
    def local_gap_center(self):
        return self.gap_center

    # @lru_cached_property(maxsize=1, arg_name_list=['gap_points'])
    # @lru_cached_property
    @property
    def gap_angles(self):
        """Gap angles from center to hole in 2D."""
        # TODO: DIY-cache lookup decorator
        # Check if cache already exists
        args_list = [self._gap_points]

        if hasattr(self, "_gap_angles_arg_cache"):
            result_from_cache = True
            for ii in range(len(args_list)):
                if type(args_list[ii]) is np.ndarray:
                    if not np.all(args_list[ii] == self._gap_angles_arg_cache[ii]):
                        result_from_cache = False
                        break
                else:
                    if not all(args_list[ii] == self._gap_angles_arg_cache[ii]):
                        result_from_cache = False
                        break

            if result_from_cache:
                return self._gap_angles

        self._gap_angles_arg_cache = args_list

        gap_angles = np.zeros(2)
        for ii in range(gap_angles.shape[0]):
            gap_angles[ii] = np.arctan2(
                self._gap_points[1, ii], self._gap_points[0, ii]
            )

        if angle_difference_directional(gap_angles[1], gap_angles[0]):
            # warnings.warn('Angles wrong why around.')
            gap_angles[1], gap_angles[0] = gap_angles[0], gap_angles[1]

        self._gap_angles = gap_angles
        # TODO: test(!)
        return self._gap_angles

    @property
    def boundary_line(self):
        # DYI-cache lookup
        args_list = [self.edge_points]

        if hasattr(self, "_boundary_line_arg_cache"):
            result_from_cache = True
            for ii in range(len(args_list)):
                if type(args_list[ii]) is np.ndarray:
                    if not np.all(args_list[ii] == self._boundary_line_arg_cache[ii]):
                        result_from_cache = False
                        break
                else:
                    if not all(args_list[ii] == self._boundary_line_arg_cache[ii]):
                        result_from_cache = False
                        break

            if result_from_cache:
                return self._boundary_line

        self._boundary_line_arg_cache = copy.deepcopy(args_list)

        self._boundary_line = LinearRing(
            [tuple(self.edge_points[:, ii]) for ii in range(self.edge_points.shape[1])]
        )
        return self._boundary_line

    def get_global_gap_points(self):
        return self.transform_relative2global(self._gap_points)

    def get_local_gap_points(self):
        return self._gap_points

    def get_global_gap_center(self):
        return self.transform_relative2global(self.gap_center)

    def get_deformation_velocity(self, position, in_global_frame=False):
        """Get deformatkion velocity."""
        if in_global_frame:
            position = self.transform_global2relative(position)

        # At center / reference point (no velocity.
        # Smootheness follows from Gamma -> infinity at the same time.
        if not np.linalg.norm(position):
            return np.zeros(self.dim)

        normal_direction = self.get_normal_direction(
            position, in_global_frame=False, normalize=True
        )
        deformation_velocity = (-1) * normal_direction * self.expansion_speed_axes

        if in_global_frame:
            deformation_velocity = self.transform_global2relative(deformation_velocity)

        return deformation_velocity

    def update_step(self, delta_time):
        """Update position & orientation."""
        self.update_deforming_obstacle(delta_time)

        if self.linear_velocity is not None:
            self.center_position = (
                self.center_position + self.linear_velocity * delta_time
            )

        if self.angular_velocity is not None:
            # breakpoint()
            self.orientation = self.orientation + self.angular_velocity * delta_time

    def update_deforming_obstacle(self, delta_time):
        """Update if obstacle is deforming."""

        self._gap_points = (
            self._gap_points
            * np.tile(
                self.get_relative_expansion(delta_time),
                (self._gap_points.shape[1], 1),
            ).T
        )

        if sys.version_info > (3, 0):
            super().update_deforming_obstacle(delta_time)
        else:
            super(BoundaryCuboidWithGaps, self).update_deforming_obstacle(delta_time)

    def get_gamma(
        self,
        position,
        in_global_frame=False,
        with_reference_point_expansion=True,
        gamma_distance=None,
    ):
        """Caclulate Gamma for 2D-Wall Case with selected Reference-Point."""
        if in_global_frame:
            position = self.transform_global2relative(position)

        ref_point = self.get_projected_reference(position, in_global_frame=False)

        ref_dir = position - ref_point
        ref_norm = np.linalg.norm(ref_dir)

        if not ref_norm:
            # Aligned at center. Gamma >> 1 (almost infinity)
            return 1e30

        max_dist = self.get_maximal_distance()

        # Create line which for sure crossed the border
        ref_line = LineString(
            [
                tuple(ref_point),
                tuple(ref_point + ref_dir * max_dist / ref_norm),
            ]
        )

        intersec = ref_line.intersection(self.boundary_line)
        point_projected_on_surface = np.array([intersec.x, intersec.y])

        dist_surface = np.linalg.norm(point_projected_on_surface - ref_point)

        # if self.is_boundary:
        if gamma_distance is None:
            gamma = dist_surface / ref_norm
        else:
            gamma = (ref_norm - dist_surface) / dist_surface + 1
            gamma = 1.0 / gamma

        # import pdb; pdb.set_trace()
        return gamma

    def get_reference_direction(self, position, in_global_frame=False, normalize=True):
        """Reference direction based on guiding reference point"""
        if in_global_frame:
            position = self.transform_global2relative(position)

        ref_point = self.get_projected_reference(position, in_global_frame=False)
        reference_direction = -(position - ref_point)

        if normalize:
            ref_norm = np.linalg.norm(reference_direction)
            if ref_norm:  # nonzero
                reference_direction = reference_direction / ref_norm

        if in_global_frame:
            reference_direction = self.transform_relative2global_dir(
                reference_direction
            )

        return reference_direction

    def get_projected_reference(self, position, in_global_frame=True):
        position_abs = copy.deepcopy(position)
        if in_global_frame:
            position = self.transform_global2relative(position)

        # Check if in between gap-center-triangle
        position_angle = np.arctan2(position[1], position[0])

        if angle_is_in_between(position_angle, self.gap_angles[0], self.gap_angles[1]):
            # Point is in gap-center-triangle
            return position_abs

        elif np.linalg.norm(position - self.gap_center) >= np.linalg.norm(
            self.gap_center
        ):
            # Point is further away from gap than center, -> place at center
            reference_point = np.zeros(self.dim)
        else:
            # Project on gap-center-triangle-border
            it_gap = 1 if position_angle > 0 else 0

            # Shapely for Center etc.
            pp = Point(self.gap_center[0], self.gap_center[1])
            cc = pp.buffer(np.linalg.norm(position - self.gap_center)).boundary
            ll = LineString([(0, 0), tuple(self._gap_points[:, it_gap])])
            intersec = cc.intersection(ll)

            try:
                reference_point = np.array([intersec.x, intersec.y])
            except AttributeError:
                # Several points / Line Object -> take closest one
                point = np.array([intersec[0].x, intersec[0].y])
                dist_closest = np.linalg.norm(point)
                ind_closest = 0

                for pp in range(1, len(intersec)):
                    point = np.array([intersec[pp].x, intersec[pp].y])
                    dist_new = np.linalg.norm(point)

                    if dist_new < dist_closest:
                        ind_closest = pp
                        dist_closest = dist_new
                reference_point = np.array(
                    [intersec[ind_closest].x, intersec[ind_closest].y]
                )

        if in_global_frame:
            reference_point = self.transform_relative2global(reference_point)

        return reference_point

    def get_gap_outside_point(self, dist_relative=3, in_global_frame=True):
        """The point which is outside the wall (in front of the gap)."""
        outside_gap_point = (
            self.gap_center
            + self.get_local_gap_to_exit_dir() * dist_relative * self.wall_thickness
        )

        if in_global_frame:
            outside_gap_point = self.transform_relative2global(outside_gap_point)
        return outside_gap_point

    def get_local_gap_to_exit_dir(self, in_global_frame=False):
        dir_perp = self._gap_points[:, 1] - self._gap_points[:, 0]
        dir_perp = np.array([-dir_perp[1], dir_perp[0]]) / np.linalg.norm(dir_perp)

        if in_global_frame:
            dir_perp = self.transform_local2relative_dir(dir_perp)

        return dir_perp

    def get_gap_patch(self, ax, x_lim, y_lim):
        gap_points = self.get_local_gap_points()

        x_lim = self.transform_global2relative(np.array(x_lim))
        y_lim = self.transform_global2relative(np.array(y_lim))
        width_max = np.max((np.abs(x_lim), np.abs(y_lim)))

        dir_perp = self.get_local_gap_to_exit_dir()

        # White patch
        edge_points = np.zeros((self.dim, 4))
        edge_points[:, 0] = gap_points[:, 1] + dir_perp * (2 * width_max)
        edge_points[:, 1] = gap_points[:, 1]
        edge_points[:, 2] = gap_points[:, 0]
        edge_points[:, 3] = gap_points[:, 0] + dir_perp * (2 * width_max)

        edge_points = self.transform_relative2global(edge_points)

        door_wall_path = plt.Polygon(edge_points.T, alpha=1.0, zorder=3)
        door_wall_path.set_color([1, 1, 1])
        ax.add_patch(door_wall_path)
