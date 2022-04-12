"""
Container encapsulates all obstacles.
Shapely Container finds the dynamic reference point through gradient descent.
"""
# Author: "LukasHuber"
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import warnings, sys
import copy
import time

import numpy as np
from numpy import linalg as LA

from shapely.ops import nearest_points

from dynamic_obstacle_avoidance.utils import get_reference_weight

from dynamic_obstacle_avoidance.obstacles import CircularObstacle

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance.obs_common_section import (
    DistanceMatrix,
    IntersectionMatrix,
    get_single_reference_point,
    get_intersection_cluster,
)


class ShapelyContainer(ObstacleContainer):
    """Obstacle Container which can be used with gradient search. It additionally stores
    the closest boundary point between obstacles."""

    # Shapely only possible in 2D
    dim = 2

    def __init__(self, obs_list=None, distance_margin=1e6):
        super().__init__(obs_list)

        self.distance_margin = distance_margin
        self._boundary_reference_points = None
        self._distance_matrix = None

    def append(self, value):  # Compatibility with normal list.
        """Add new obstacle to the end of the container."""
        super().append(value)
        # TODO: alternative for computational speed!
        # Always reset dist matrix
        # self._boundary_reference_points = np.zeros((self.dim, len(self), len(self)))
        # self._distance_matrix = DistanceMatrix(n_obs=len(self))
        self._boundary_reference_points = None
        self._distance_matrix = None

    def __delitem__(self, key):  # Compatibility with normal list.
        """Remove obstacle from container list."""
        # TODO: alternative for computational speed!
        super().__delitem__(key)

        # self._boundary_reference_points = np.zeros((self.dim, len(self), len(self)))
        # self._distance_matrix = DistanceMatrix(n_obs=len(self))
        self._boundary_reference_points = None
        self._distance_matrix = None

    @property
    def index_wall(self):
        return self.get_wall_index()

    def get_wall_index(self):
        """Look for the index of the wall and return it."""
        for it, obs in zip(range(self.n_obstacles), self._obstacle_list):
            if obs.is_boundary:
                return it

    def single_boundary_and_nonequal_check(self, ii, jj):
        if ii == jj:
            raise NotImplementedError("Not defined for two boundaries.")
        if self[ii].is_boundary:
            if self[jj].is_boundary:
                raise NotImplementedError("Not defined for two boundaries.")
            ii, jj = jj, ii
        return ii, jj

    def are_intersecting(self, ii, jj):
        ii, jj = self.single_boundary_and_nonequal_check(ii, jj)

        if (
            self[ii].get_gamma(self[jj].center_position, in_global_frame=True) < 1
            and not self[jj].is_boundary
        ):
            return True

        if self[jj].get_gamma(self[ii].center_position, in_global_frame=True) < 1:
            return True

        if self[jj].is_boundary:
            # For a boundary, check if (1) the obstacle is uniquely outside of the boundary
            # or (2) there is no intersection of the boundary-line & the obstacle
            return not self[ii].shapely.global_margin.intersects(
                self[jj].shapely.global_margin
            ) or self[ii].shapely.global_margin.intersects(
                self[jj].shapely.global_margin.boundary
            )
        else:
            return self[ii].shapely.global_margin.intersects(
                self[jj].shapely.global_margin
            )

    def evaluate_intersection_position(self, ii, jj):
        """Evaluate the insterection point."""
        ii, jj = self.single_boundary_and_nonequal_check(ii, jj)

        # TODO: better intersection evaluation
        # better position [negative distance? up to -1?]
        if self[jj].is_boundary:
            intersections = (
                self[ii].shapely.global_margin.symmetric_difference(
                    self[jj].shapely.global_margin
                )
            ).difference(self[jj].shapely.global_margin)

        else:
            intersections = self[ii].shapely.global_margin.intersection(
                self[jj].shapely.global_margin
            )
        intersection_center = np.array(intersections.centroid.coords.xy).squeeze()

        self._boundary_reference_points[:, ii, jj] = intersection_center
        # Store it for boundaries, too
        self._boundary_reference_points[:, jj, ii] = intersection_center

        self._distance_matrix[ii, jj] = 0

    def evaluate_boundary_reference_points(self, ii, jj):
        """Evaluate and set the distance and boundary
        reference for NON-intersecting obstacles."""
        ii, jj = self.single_boundary_and_nonequal_check(ii, jj)

        if self[jj].is_boundary:
            p1, p2 = nearest_points(
                self[ii].shapely.global_margin, self[jj].shapely.global_margin.boundary
            )
            self._boundary_reference_points[:, ii, jj] = p1

        else:
            p1, p2 = nearest_points(
                self[ii].shapely.global_margin, self[jj].shapely.global_margin
            )

            self._boundary_reference_points[:, ii, jj] = p1
            self._boundary_reference_points[:, jj, ii] = p2

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
        if not len(distance_list):
            return distance_list

        distance_list = np.array(distance_list)

        ind_zero = np.isclose(distance_list, 0)
        if np.sum(ind_zero):
            return ind_zero / np.sum(ind_zero)

        weight = 1.0 / distance_list - 1.0 / distance_margin
        if np.sum(weight) > 1:
            weight = weight / np.sum(weight)
        return weight

    def update_intersecting_obstacles(self):
        """Updates the reference points of the intersecintg obstacles.
        and return a (bool) list which indicates which obstacles are intersecting."""
        intersecting_obstacles = np.arange(len(self))

        intersection_matrix = IntersectionMatrix(n_obs=len(self))

        for ii in range(len(self)):
            for jj in range(ii + 1, len(self)):
                if self._distance_matrix[ii, jj]:
                    # Nonzero -> nonintersecting
                    continue
                intersection_matrix[ii, jj] = self._boundary_reference_points[:, ii, jj]

                intersecting_obstacles[ii] = True
                intersecting_obstacles[jj] = True

        # TODO: the functiions bellow should be member-methods
        intersecting_obs = get_intersection_cluster(intersection_matrix, self)
        # intersecting_obs = np.array(intersecting_obs).flatten()

        return intersecting_obstacles

    def update_reference_points(self, create_shapely=False):
        # todo: check for all if have moved
        if create_shapely:
            for ii in range(self.n_obstacles):
                if self[ii].has_moved or self[ii].shapely is None:
                    self[ii].create_shapely()
        if self._boundary_reference_points is None:
            self._boundary_reference_points = np.zeros((self.dim, len(self), len(self)))
        if self._distance_matrix is None:
            self._distance_matrix = DistanceMatrix(n_obs=len(self))

        for ii in range(self.n_obstacles):
            for jj in range(ii + 1, self.n_obstacles):
                if not self[ii].has_moved and not self[jj].has_moved:
                    continue

                if self.are_intersecting(ii, jj):
                    self.evaluate_intersection_position(ii, jj)
                else:
                    self.evaluate_boundary_reference_points(ii, jj)

            self[ii].has_moved = False

        intersecting_obstacles = self.update_intersecting_obstacles()
        for ii in np.arange(self.n_obstacles)[np.logical_not(intersecting_obstacles)]:
            distance_list = [
                self.get_distance(ii, jj) if ii != jj else 2 * self.distance_margin
                for ii in range(self.n_obstacles)
            ]
            # make array for easier comparison
            distance_list = np.array(distance_list)
            ind_nonzero = distance_list < self.distance_margin

            if not any(ind_nonzero):
                self[ii].set_reference_point(
                    np.zeros(self.dimension), in_global_frame=False
                )
                continue

            distance_list = distance_list[ind_nonzero]
            boundary_ref_points = self._boundary_reference_points[
                :, ii, ind_nonzero
            ].reshape(self.dimension, -1)

            weights = self.get_distance_weight(
                distance_list, distance_margin=self.distance_margin
            )

            weighted_ref_point = np.sum(
                np.array(boundary_ref_points)
                * np.tile(weights, (boundary_ref_points.shape[0], 1)),
                axis=1,
            )

            # Add normal center_position if the outside are not pulling a lot
            sum_weight = np.sum(weights)
            if sum_weight < 1:
                weighted_ref_point += self[ii].center_position * (1 - sum_weight)

            self[ii].set_reference_point(weighted_ref_point, in_global_frame=True)


class SphereContainer(ShapelyContainer):
    """Environment with circles only and no boundary."""

    def are_intersecting(self, ii, jj):
        ii, jj = self.single_boundary_and_nonequal_check(ii, jj)

        dist = LA.norm(self[ii].center_position - self[jj].center_position)

        return dist < (self[ii].radius_with_margin + self[jj].radius_with_margin)

    def evaluate_intersection_position(self, ii, jj):
        ii, jj = self.single_boundary_and_nonequal_check(ii, jj)

        intersection_center = 0.5 * (
            self[ii].center_position - self[jj].center_position
        )

        self._boundary_reference_points[:, ii, jj] = intersection_center

        # Store it for boundaries, too
        self._boundary_reference_points[:, jj, ii] = intersection_center

        self._distance_matrix[ii, jj] = 0

    def evaluate_boundary_reference_points(self, ii, jj):
        ii, jj = self.single_boundary_and_nonequal_check(ii, jj)

        center_vect = self[ii].center_position - self[jj].center_position
        center_vect = center_vect / LA.norm(center_vect)

        try:
            self._boundary_reference_points[:, ii, jj] = (
                self[ii].center_position - center_vect * self[ii].radius_with_margin
            )
            self._boundary_reference_points[:, jj, ii] = (
                self[jj].center_position - center_vect * self[jj].radius_with_margin
            )
        except:
            breakpoint()

        self._distance_matrix[ii, jj] = LA.norm(
            self._boundary_reference_points[:, ii, jj]
            - self._boundary_reference_points[:, jj, ii]
        )
