"""
Container encapsulates all obstacles.
Gradient container finds the dynamic reference point through gradient descent.
"""
# Author: "LukasHuber"
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import warnings, sys
import numpy as np
import copy
import time

from shapely.ops import nearest_points

from dynamic_obstacle_avoidance.utils import get_reference_weight

from dynamic_obstacle_avoidance.obstacles import CircularObstacle

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.avoidance.obs_dynamic_center_3d import *

# BUGS / IMPROVEMENTS:
#    - Bug-fix (with low priority),
#            the automatic-extension of the hull of the ellipse does not work
#    - Gradient descent: change function


class ShapelyContainer(ObstacleContainer):
    """Obstacle Container which can be used with gradient search. It additionally stores
    the closest boundary point between obstacles."""

    # Shapely only possible in 2D
    dim = 2

    def __init__(self, obs_list=None, distance_margin=1e-6):
        super().__init__(obs_list)

        self.distance_margin = distance_margin
        self._boundary_reference_points = np.zeros((self.dim, len(self), len(self)))
        self._distance_matrix = DistanceMatrix(n_obs=len(self))
        self.shapely_with_boundary = [None for ii in range(len(self))]

    def append(self, value):  # Compatibility with normal list.
        """Add new obstacle to the end of the container."""
        super().append(value)
        # TODO: alternative for computational speed!
        # Always reset dist matrix
        self._boundary_reference_points = np.zeros((self.dim, len(self), len(self)))
        self._distance_matrix = DistanceMatrix(n_obs=len(self))
        self.shapely_with_boundary = [None for ii in range(len(self))]

    def __delitem__(self, key):  # Compatibility with normal list.
        """Remove obstacle from container list."""
        super().__delitem__(key)

        self._boundary_reference_points = np.zeros((self.dim, len(self), len(self)))
        self._distance_matrix = DistanceMatrix(n_obs=len(self))
        del self.shapely_with_boundary[key]

    @property
    def index_wall(self):
        ind_wall = None

        for it, obs in zip(range(self.n_obstacles), self._obstacle_list):
            if obs.is_boundary:
                ind_wall = it
                break
        return ind_wall

    def are_intersecting(self, ii, jj):
        if ii == jj:
            return None

        if self[ii].get_gamma(self[jj].center_position, in_global_frame=True) < 1:
            return True

        if self[jj].get_gamma(self[ii].center_position, in_global_frame=True) < 1:
            return True

        return self.shapely_with_boundary[ii].intersects(self.shapely_with_boundary[jj])

    def evaluate_intersection_position(self, ii, jj):
        # TODO: better intersection evaluation
        # better position [negative distance? up to -1?]
        intersections = self.shapely_with_boundary[ii].intersection(
            self.shapely_with_boundary[jj]
        )

        intersection_center = (
            np.max(intersections, axis=0) - np.min(intersections, axis=0)
        ) * 0.5
        self._boundary_reference_points[ii, jj, :] = self._boundary_reference_points[
            jj, ii, :
        ] = intersection_center

        self._distance_matrix[ii, jj] = 0

    def evaluate_boundary_reference_points(self, ii, jj):
        p1, p2 = nearest_points(dilated1, dilated2)

        self._boundary_reference_points[ii, jj, :] = p1
        self._boundary_reference_points[jj, ii, :] = p2

        self._distance_matrix[ii, jj] = LA.norm(p2 - p1)

    def get_distance(self, ii, jj=None):
        """Distance between obstacles ii and jj"""
        return self._distance_matrix[ii, jj]

    def reset_obstacles_have_moved(self):
        """Resets obstacles in list such that they have NOT moved."""
        for obs in self._obstacle_list:
            obs.has_moved = False

    def reset_reference_points(self):
        """Set the reference points at the center of the obstacle to not
        interfer with any evaluation."""
        for obs in self._obstacle_list:
            obs.set_reference_point(np.zeros(obs.dim), in_global_frame=False)

    def get_if_obstacle_is_updated(self, ii):
        return self._obstacle_is_updated[ii]

    def get_distance_weight(self, distance_list, distance_margin):
        # TODO: better weighting which includes also inside points
        dist = np.array(distance_list)

        ind_zero = dist == 0
        if np.sum(ind_zero):
            return ind_zero / np.sum(ind_zero)

        weight = 1.0 / distance_list - 1.0 / distance_margin
        if np.sum(weight) > 1:
            weight = weight / np.sum(weight)
        return weight

    def update_reference_points(self):
        # TODO: check for all if have moved
        for ii in range(self.n_obstacles):
            if self[ii].has_moved or self.shapely_with_boundary[ii] is None:
                self[ii].create_shapely()
                breakpoint()

        for ii in range(self.n_obstacles):
            for jj in range(ii + 1, self.n_obstacles):
                if not self[ii].has_moved and not self[jj].has_moved:
                    continue

                if self.are_intersecting(ii, jj):
                    self.evaluate_intersection_position(ii, jj)
                else:
                    self.evaluate_boundary_reference_points(ii, jj)

        for ii in range(self.n_obstacles):
            distance_list = []
            boundary_ref_points = []

            for jj in range(self.n_obstacles):
                if ii == jj:
                    continue

                new_dist = self.get_distance(ii, jj)
                if new_dist < self.dist_margin:

                    distance_list.append(new_dist)
                    boundary_ref_points.append(
                        self._boundary_reference_points[:, ii, jj]
                    )

            weights = self.get_distance_weight(
                distance_list, distance_margin=self.distance_margin
            )
            if np.sum(weights) < 1:
                weights.append(1 - np.sum(weights))
                boundary_ref_points.append(self[ii].center_position)

            weighted_ref_point = np.sum(
                np.array(boundary_ref_points).T
                * np.tile(weights, (boundary_ref_points[0].shape[0], 1)),
                axis=1,
            )
            self[ii].set_reference_point(weighted_ref_point, in_global_frame=True)
