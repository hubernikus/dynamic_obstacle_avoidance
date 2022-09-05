"""
Obstacle based on a kmeans-clustering
"""
# Created: 2022-09-04
# Github: hubernikus
# License: BSD (c) 2022

import sys
import warnings
import math

import numpy as np
from numpy import linalg as LA

from vartools.math import get_intersection_between_line_and_plane
from vartools.directional_space import get_directional_weighted_sum

from dynamic_obstacle_avoidance.obstacles import Obstacle

from dynamic_obstacle_avoidance.rotational.datatypes import Vector, VectorArray


class KmeansObstacle(Obstacle):
    """Pseudo obstacle based on a learned kmeans clustering."""

    def __init__(
        self,
        kmeans,
        radius: float,
        index: int,
        is_boundary: bool = True,
        **kwargs,
    ):
        self.kmeans = kmeans
        self.radius = radius
        self._index = index

        super().__init__(is_boundary=is_boundary, **kwargs)

        # Only calculate the normal direction between obstacles once (!)
        # and the ones which are interesting
        self.ind_relevant = np.arange(self.n_clusters)
        self.ind_relevant = np.delete(self.ind_relevant, self._index)

        self._center_positions = 0.5 * (
            np.tile(
                self.kmeans.cluster_centers_[self._index, :],
                (self.ind_relevant.shape[0], 1),
            )
            + self.kmeans.cluster_centers_[self.ind_relevant, :]
        )

        # Since obstacle are convex -> intersection needs to be part of one or the other
        labels = self.kmeans.predict(self._center_positions)
        ind_close = np.logical_or(labels == self.ind_relevant, labels == self._index)
        self.ind_relevant = self.ind_relevant[ind_close]
        self._center_positions = self._center_positions[ind_close, :]

        normal_directions = np.zeros((self.n_clusters, self.dimension))
        normal_directions[self.ind_relevant, :] = self.kmeans.cluster_centers_[
            self.ind_relevant, :
        ] - np.tile(
            self.kmeans.cluster_centers_[self._index, :],
            (self.ind_relevant.shape[0], 1),
        )
        # Only look at points which are even possible to interesect
        # ind_good_distances = distances < 2 * self.radius
        # self.ind_relevant = self.ind_relevant[ind_good_distances]

        self._normal_directions = (
            normal_directions[self.ind_relevant, :]
            / np.tile(
                LA.norm(normal_directions[self.ind_relevant, :], axis=1),
                (self.dimension, 1),
            ).T
        )

        self.successor_index = []

    # => Now defined as factory function alongside MotionLearnerThrougKMeans
    # @classmethod
    # def from_learner(cls, kmeans_learner: MotionLearnerThrougKMeans, index: int):
    #     """Alternative constructor by using kmeans-leraner only."""
    #     instance = cls(
    #         Kmeans=kmeans_learner.kmeans,
    #         radius=kmeans_learner.region_radius_,
    #         index=index,
    #     )

    #     instance.successor_index = [
    #         ii for ii in kmeans_learner._graph.successors(index)
    #     ]

    #     return instance

    @property
    def normal_directions(self) -> VectorArray:
        """Returns the full array of normal directions."""
        # TODO: this seems a redundant step -> check if it can be avoided
        normal_directions = np.zeros((self.n_clusters, self.dimension))
        normal_directions[self.ind_relevant, :] = self._normal_directions
        return normal_directions

    @property
    def inbetween_points(self) -> VectorArray:
        inbetween_points = np.zeros((self.n_clusters, self.dimension))
        inbetween_points[self.ind_relevant, :] = self._center_positions
        return inbetween_points

    def get_inbetween_position(self, index) -> Vector:
        return self._center_positions[self.ind_relevant, :]

    @property
    def num_relevant(self) -> int:
        return self.ind_relevant.shape[0]

    @property
    def ind_relevant_and_self(self) -> np.ndarray:
        return np.hstack((self.ind_relevant, self._index))

    @property
    def center_position(self) -> Vector:
        """Returns global center point."""
        return self.kmeans.cluster_centers_[self._index, :]

    @center_position.setter
    def center_position(self, value) -> None:
        """Returns global center point."""
        warnings.warn("Position is not being set.")

    @property
    def reference_point(self) -> Vector:
        """Returns global reference point."""
        return self.kmeans.cluster_centers_[self._index, :]

    @reference_point.setter
    def reference_point(self, value) -> None:
        """Returns global reference point."""
        if LA.norm(value):  # Nonzero value
            raise NotImplementedError(
                "Reference point is not reset for KMeans-Obstacle."
            )

    def get_reference_direction(self, position: Vector, in_global_frame: bool = False):
        if in_global_frame:
            direction = self.kmeans.cluster_centers_[self._index, :] - position
        else:
            direction = position * (-1)

        if norm_dir := LA.norm(direction):
            direction = direction / norm_dir

        return direction

    @property
    def dimension(self) -> int:
        return self.kmeans.cluster_centers_.shape[1]

    @property
    def n_clusters(self) -> int:
        return self.kmeans.cluster_centers_.shape[0]

    def is_inside(self, position: Vector, in_global_frame: bool = False) -> bool:
        gamma = self.get_gamma(position, in_global_frame, ind_transparent=-1)
        # gamma = self._get_gamma_without_transition(position, in_global_frame)
        return gamma > 1

    def _get_gamma_without_transition(
        self, position: Vector, in_global_frame: bool = False
    ) -> float:
        if not in_global_frame:
            # position = self.pose.transform_position_from_relative(position)
            position = position + self.center_position

        distance_position = LA.norm(position - self.center_position)
        if not distance_position:
            if self.is_boundary:
                return sys.float_info.max
            else:
                return 0.0

        surface_position = self.get_point_on_surface(position, in_global_frame=True)
        distance_surface = LA.norm(surface_position - self.center_position)

        if self.is_boundary:
            return distance_surface / distance_position

        else:
            if distance_position > distance_surface:
                return distance_position - distance_surface
            else:
                return distance_surface / distance_position
            return distance_surface / distance_position

    def get_gamma(
        self,
        position: Vector,
        in_global_frame: bool = False,
        ind_transparent: int = None,
    ) -> float:
        """Returns the gamma value based on the input position.

        ind_transparent: Enables smooth flow through towards the transparent index"""
        # TODO: maybe check if it's really smooth with respect to the transparent border
        if in_global_frame:
            relative_position = position - self.center_position
        else:
            relative_position = position
            position = position + self.center_position

        if not (position_norm := LA.norm(relative_position)):
            if self.is_boundary:
                return sys.float_info.max
            else:
                0

        distances_surface = self._get_normal_distances(position, is_boundary=False)
        distances_surface[self._index] = position_norm - self.radius

        max_dist = np.max(distances_surface[self.ind_relevant_and_self])
        if max_dist < 0:
            # Position is inside -> project it to the outside
            position = (
                self.center_position
                + ((position_norm - max_dist) / position_norm) ** 2 * relative_position
            )

            weights = self._get_normal_distances(position, is_boundary=False)
            weights[self._index] = (
                LA.norm(position - self.center_position) - self.radius
            )

        elif max_dist == 0:
            if not len(self.successor_index) or position_norm == self.radius:
                # Point is exactly on the surface -> return relative position
                return 1
            else:
                # Point is exactly in gap
                return sys.float_info.max

        else:
            weights = distances_surface

        # Only consider positive ones for the weight
        weights = np.maximum(weights, 0)
        weights = weights / np.sum(weights)

        if self.is_boundary:
            local_radiuses = position_norm - distances_surface

            if ind_transparent is None:
                if len(self.successor_index) > 1:
                    raise NotImplementedError("Only implemented for single successor")

                elif len(self.successor_index) == 0:
                    return np.sum(weights * local_radiuses) / position_norm

                ind_transparent = self.successor_index[0]

            elif ind_transparent < 0:
                # No transparent
                return np.sum(weights * local_radiuses) / position_norm

            if weights[ind_transparent] > 0:
                weights = np.maximum(weights - weights[ind_transparent], 0)
                weights[ind_transparent] = 1 - np.sum(weights)

                if weights[ind_transparent] < 1:
                    local_radiuses[ind_transparent] = local_radiuses[
                        ind_transparent
                    ] / (1 - weights[ind_transparent])
                else:
                    return sys.float_info.max
            return np.sum(weights * local_radiuses) / position_norm

        else:
            for ii in self.successor_index:
                raise NotImplementedError()

            mean_dist = np.sum(weights * distances_surface)
            # Normal obstacle
            if mean_dist > 0:
                mean_dist / self.radius + 1

            else:
                # Proportional inside
                return position_norm / (position_norm - mean_dist)

    def get_normal_direction(self, position, in_global_frame: bool = False) -> Vector:
        """Returns smooth-weighted normal direction around the obstacles."""
        if in_global_frame:
            relative_position = position - self.center_position
        else:
            relative_position = position
            position = position + self.center_position

        relative_position = position - self.center_position
        if not (position_norm := LA.norm(relative_position)):
            # Some direction (normed).
            position[0] = 1
            return position
        distances_surface = self._get_normal_distances(position, is_boundary=False)
        distances_surface[self._index] = position_norm - self.radius

        if (max_dist := np.max(distances_surface[self.ind_relevant_and_self])) < 0:
            # Position is inside -> project it to the outside
            position = (
                self.center_position
                + ((position_norm - max_dist) / position_norm) ** 2 * relative_position
            )

            distances_surface = self._get_normal_distances(position, is_boundary=False)
            distances_surface[self._index] = (
                LA.norm(position - self.center_position) - self.radius
            )

            # Only consider positive ones for the weight
            weights = np.maximum(distances_surface, 0)
            weights = weights / np.sum(weights)

        elif max_dist == 0:
            arg_max = np.argmax(distances_surface[self.ind_relevant_and_self])
            weights = np.zeros(distances_surface.shape)
            weights[self.ind_relevant_and_self[arg_max]] = 1
            weights = weights / np.sum(weights)

            # # Point is exactly on the surface -> return relative position
            # if in_global_frame:
            #     return relative_position / position_norm
            # else:
            #     return self.pose.transform_direction_to_relative(
            #         relative_position / position_norm
            #     )

        else:
            # Only consider positive ones for the weight
            weights = np.maximum(distances_surface, 0)
            weights = weights / np.sum(weights)

        # The deviation at index is zero -> do the summing without it
        # instead of adding it to the normal_directions
        weights[self._index] = 0

        weighted_direction = get_directional_weighted_sum(
            null_direction=(relative_position / position_norm),
            directions=self.normal_directions.T,
            weights=weights,
            total_weight=np.sum(weights),
        )

        return weighted_direction
        # if in_global_frame:
        #     return weighted_direction
        # else:
        #     return self.pose.transform_direction_to_relative(weighted_direction)

    def _get_surface_weights(self, position: Vector) -> np.ndarray:
        """Get the surface weights in the global frame."""
        relative_position = position - self.center_position
        position_norm = LA.norm(relative_position)

        distances_surface = self._get_normal_distances(position, is_boundary=False)
        distances_surface[self._index] = position_norm - self.radius

        max_dist = np.max(distances_surface[self.ind_relevant_and_self])
        if max_dist < 0:
            # Position is inside -> project it to the outside
            position = (
                self.center_position
                + ((position_norm - max_dist) / position_norm) ** 2 * relative_position
            )

            distances_surface = self._get_normal_distances(position, is_boundary=False)
            distances_surface[self._index] = (
                LA.norm(position - self.center_position) - self.radius
            )

        elif max_dist == 0:
            return np.zeros_like(distances_surface)

        return distances_surface

    def get_point_on_surface(
        self, position: Vector, in_global_frame: bool = False
    ) -> Vector:

        if in_global_frame:
            direction = position - self.center_position
        else:
            direction = position
            position = position + self.center_position

        if dir_norm := LA.norm(direction):
            direction = direction / dir_norm
        else:
            # Random value
            direction[0] = 1

        # Default guess: point is on the circle-surface
        boundary_position = self.center_position + direction * self.radius

        # Find closest boundary
        dists = self._get_normal_distances(boundary_position, is_boundary=False)

        surface_distance = self.radius
        # Only change the default if there is an intersection
        for ii in np.arange(dists.shape[0])[dists > 0]:
            intersection = get_intersection_between_line_and_plane(
                self.center_position,
                direction,
                self.inbetween_points[ii, :],
                self.normal_directions[ii, :],
            )

            if LA.norm(self.center_position - intersection) < surface_distance:
                surface_distance = LA.norm(self.center_position - intersection)
                boundary_position = intersection

        # Somehow working with the normal leads to wrong estimation in corners..
        # if any(dists > 0):
        #     max_ind = np.argmax(dists)
        #     boundary_position = self.center_position + direction * (
        #         self.radius
        #         - dists[max_ind] / np.dot(self.normal_directions[max_ind, :], direction)
        #     )

        if not in_global_frame:
            # position = self.pose.transform_position_to_relative(position)
            boundary_position = boundary_position - self.center_position

        return boundary_position

    def _get_normal_distances(
        self, position: Vector, is_boundary: bool = None
    ) -> VectorArray:
        """Returns a tuple with all normal directions and tangent directions with respect to the
        surrounding."""
        if is_boundary or (is_boundary is None and self.is_boundary):
            raise NotImplementedError()

        center_dists = np.zeros(self.kmeans.cluster_centers_.shape[0])
        center_dists[self.ind_relevant] = np.sum(
            self._normal_directions
            * (
                np.tile(position, (self.ind_relevant.shape[0], 1))
                - self._center_positions
            ),
            axis=1,
        )
        return center_dists

    def evaluate_surface_points(self, n_points: int = 100) -> VectorArray:
        if self.dimension != 2:
            raise NotImplementedError(
                "This is only implemented and defined for 2-dimensions."
            )
        # self.surface_points = np.zeros((self.dimension, w))

        angle = np.linspace(0, 2 * math.pi, n_points)
        self.surface_points = (
            np.vstack((np.cos(angle), np.sin(angle)))
            + np.tile(self.kmeans.cluster_centers_[self._index, :], (n_points, 1)).T
        )

        for ii in range(n_points):
            self.surface_points[:, ii] = self.get_point_on_surface(
                self.surface_points[:, ii], in_global_frame=True
            )

        return self.surface_points
