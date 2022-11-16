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


class KMeansObstacle(Obstacle):
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

        # Since obstacle are convex -> intersection needs to be part of one or the other
        # Note that another obstacle can be in between the obstacles,
        dists = LA.norm(
            np.tile(
                self.kmeans.cluster_centers_[self._index, :],
                (self.ind_relevant.shape[0], 1),
            )
            - self.kmeans.cluster_centers_[self.ind_relevant, :],
            axis=1,
        )
        self.ind_relevant = self.ind_relevant[dists < 2 * self.radius]

        self._inbetween_points = 0.5 * (
            np.tile(
                self.kmeans.cluster_centers_[self._index, :],
                (self.ind_relevant.shape[0], 1),
            )
            + self.kmeans.cluster_centers_[self.ind_relevant, :]
        )

        # hence the interesections do NOT need to be part of each other
        # labels = self.kmeans.predict(self._inbetween_points)
        # ind_close = np.logical_or(labels == self.ind_relevant, labels == self._index)
        # self.ind_relevant = self.ind_relevant[ind_close]
        # self._inbetween_points = self._inbetween_points[ind_close, :]

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
    #         KMeans=kmeans_learner.kmeans,
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
        inbetween_points[self.ind_relevant, :] = self._inbetween_points
        return inbetween_points

    def get_inbetween_position(self, index) -> Vector:
        return self._inbetween_points[self.ind_relevant, :]

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
        # return self.kmeans.predict(position.reshape(1, -1))[0] == self._index
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

    def project_position_to_outside(
        self, position: Vector, in_global_frame: bool = False
    ) -> Vector:
        # position_surface = self.get_surface_point(relative_position)
        raise NotImplementedError()

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
                return 0

        surf_position = self.get_point_on_surface(position, in_global_frame=True)
        surf_norm = LA.norm(surf_position - self.center_position)

        if position_norm < surf_norm:
            # Project towards the outside
            proj_position = relative_position * (surf_norm / position_norm) ** 2
            proj_position_norm = LA.norm(proj_position)

            # Put to global frame
            proj_position = proj_position + self.center_position

        elif position_norm == surf_norm:
            # Surface points cannot be disregarded yet, since we need to check for
            # transition regions
            if ind_transparent is None and len(self.successor_index) != 1:
                # We are on the surface -> gamma = 1 / or infty if in gap
                return 1.0

            ind_transparent = self.successor_index[0]

            if np.dot(
                surf_position - self.inbetween_points[ind_transparent, :],
                self.normal_directions[ind_transparent, :],
            ):
                # NOT on the transparent surface
                return 1.0

            return sys.float_info.max
        else:

            proj_position = position
            proj_position_norm = position_norm

        distances_surface = self._get_normal_distances(proj_position, is_boundary=False)
        distances_surface[self._index] = proj_position_norm - self.radius

        # Only consider positive ones for the weight
        weights = np.maximum(distances_surface, 0)
        weights = weights / np.sum(weights)

        # argmax = np.argmax(distances_surface[self.ind_relevant_and_self])
        # max_dist = distances_surface[self.ind_relevant_and_self[argmax]]

        # if max_dist < 0:
        #     # Get the real distance to the surface
        #     if self.ind_relevant_and_self[argmax] != self._index:
        #         # Project distance onto normal, if it is not with respect to radius
        #         dot_prod = np.dot(
        #             relative_position / LA.norm(relative_position),
        #             self._normal_directions[argmax, :],
        #         )
        #         dist_to_surface = max_dist / dot_prod
        #     else:
        #         dist_to_surface = max_dist

        #     # Position is inside -> project it to the outside
        #     proj_position = (
        #         self.center_position
        #         + (position_norm / (position_norm + dist_to_surface)) ** 2
        #         * relative_position
        #     )

        #     weights = self._get_normal_distances(proj_position, is_boundary=False)
        #     weights[self._index] = (
        #         LA.norm(proj_position - self.center_position) - self.radius
        #     )

        #     if np.sum(np.maximum(weights, 0)) == 0:
        #         # Debugging stop
        #         breakpoint()

        # elif max_dist == 0:
        #     if not len(self.successor_index) or position_norm == self.radius:
        #         # Point is exactly on the surface -> return relative position
        #         return 1
        #     else:
        #         # Point is exactly in gap
        #         return sys.float_info.max

        # else:
        #     weights = distances_surface

        if self.is_boundary:
            # TODO: maybe here we actually have to consider the projected_normal distance (?!)
            local_radiuses = proj_position_norm - distances_surface

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

                if weights[ind_transparent] >= 1:
                    return sys.float_info.max

                local_radiuses[ind_transparent] = local_radiuses[ind_transparent] / (
                    1 - weights[ind_transparent]
                )

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
            if self.is_boundary:
                position = (-1) * position

            return position

        surf_position = self.get_point_on_surface(position, in_global_frame=True)
        surf_norm = LA.norm(surf_position - self.center_position)

        if position_norm < surf_norm:
            # Project towards the outside
            proj_position = relative_position * (surf_norm / position_norm) ** 2
            proj_position_norm = LA.norm(proj_position)

            # Put to global frame
            proj_position = proj_position + self.center_position

        elif position_norm == surf_norm:
            # Surface points cannot be disregarded --- we're exactly on it(!)
            ind_closest = np.argsort(
                LA.norm(
                    self.kmeans.cluster_centers_[self.ind_relevant, :]
                    - np.tile(self.center_position, (len(self.ind_relevant), 1)),
                    axis=0,
                )
            )
            # breakpoint()

            dist = LA.norm(
                self.kmeans.cluster_centers_[self.ind_relevant[ind_closest[0]], :]
                - self.center_position
            )

            if dist >= self.radius:
                normal = relative_position / LA.norm(relative_position)
                if self.is_boundary:
                    normal = (-1) * normal
                return normal

            normal_plane = (
                self.kmeans.cluster_centers_[self.ind_relevant[ind_closest[0]], :]
                - self.kmeans.cluster_centers_[self.ind_relevant[ind_closest[1]], :]
            )

            normal = normal_plane / LA.norm(normal_plane)
            if self.is_boundary:
                normal = (-1) * normal
            return normal
        else:

            proj_position = position
            proj_position_norm = position_norm

        distances_surface = self._get_normal_distances(proj_position, is_boundary=False)
        distances_surface[self._index] = proj_position_norm - self.radius

        # Only consider positive ones for the weight
        weights = np.maximum(distances_surface, 0)
        weights = weights / np.sum(weights)

        # distances_surface = self._get_normal_distances(position, is_boundary=False)
        # distances_surface[self._index] = position_norm - self.radius

        # argmax = np.argmax(distances_surface[self.ind_relevant_and_self])
        # max_dist = distances_surface[self.ind_relevant_and_self[argmax]]

        # if max_dist < 0:
        #     # Position is inside -> project it to the outside
        #     if self.ind_relevant_and_self[argmax] != self._index:
        #         # Project distance onto normal, if it is not with respect to radius
        #         max_dist = max_dist / (
        #             np.dot(
        #                 relative_position / LA.norm(relative_position),
        #                 self._normal_directions[argmax, :],
        #             )
        #         )

        #     # Position is inside -> project it to the outside
        #     proj_position = (
        #         self.center_position
        #         + ((position_norm - max_dist) / position_norm) ** 2 * relative_position
        #     )

        #     distances_surface = self._get_normal_distances(
        #         proj_position, is_boundary=False
        #     )
        #     distances_surface[self._index] = (
        #         LA.norm(proj_position - self.center_position) - self.radius
        #     )

        #     # Only consider positive ones for the weight
        #     weights = np.maximum(distances_surface, 0)
        #     weights = weights / np.sum(weights)

        # elif max_dist == 0:
        #     arg_max = np.argmax(distances_surface[self.ind_relevant_and_self])
        #     weights = np.zeros(distances_surface.shape)
        #     weights[self.ind_relevant_and_self[arg_max]] = 1
        #     weights = weights / np.sum(weights)

        #     # # Point is exactly on the surface -> return relative position
        #     # if in_global_frame:
        #     #     return relative_position / position_norm
        #     # else:
        #     #     return self.pose.transform_direction_to_relative(
        #     #         relative_position / position_norm
        #     #     )

        # else:
        #     # Only consider positive ones for the weight
        #     weights = np.maximum(distances_surface, 0)
        #     weights = weights / np.sum(weights)

        # The deviation at index is zero -> do the summing without it
        # instead of adding it to the normal_directions
        weights[self._index] = 0

        weighted_direction = get_directional_weighted_sum(
            null_direction=(relative_position / position_norm),
            directions=self.normal_directions.T,
            weights=weights,
            total_weight=np.sum(weights),
        )

        if self.is_boundary:
            weighted_direction = (-1) * weighted_direction

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
        # dists = self._get_normal_distances(boundary_position, is_boundary=False)
        dists = self._normal_directions @ direction

        surface_distance = self.radius
        # Only change the default if there is an intersection
        # for ii in self.ind_relevant[dists > 0]:
        normals = self.normal_directions
        inbetweeners = self.inbetween_points

        for ii in self.ind_relevant[abs(dists) > 1e-9]:
            intersection = get_intersection_between_line_and_plane(
                self.center_position,
                direction,
                inbetweeners[ii, :],
                normals[ii, :],
                positive_only=True,
            )
            if intersection is None:
                continue

            if LA.norm(self.center_position - intersection) < surface_distance:
                surface_distance = LA.norm(self.center_position - intersection)
                boundary_position = intersection

        if not in_global_frame:
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
                - self._inbetween_points
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


def test_distributed_kmeans():
    RANDOM_SEED = 1
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    centers = np.array(
        [
            [-3.24225835, 0.84629337],
            [-4.43803427, 0.88446869],
            [-0.86091289, 0.9189247],
            [-4.04617905, 0.69277709],
        ]
    )

    datahandler = MotionDataHandler(
        # position=np.array([[-1, 0], [1, 0], [1, 2], [-1, 2]])
        position=centers
    )

    datahandler.velocity = datahandler.position[1:, :] - datahandler.position[:-1, :]
    datahandler.velocity = np.vstack((datahandler.velocity, [[0, 0]]))
    datahandler.attractor = np.array([0.0, 1])
    datahandler.sequence_value = np.linspace(0, 1, 4)

    dimension = 2

    main_learner = MotionLearnerThrougKMeans(data=datahandler)

    from dynamic_obstacle_avoidance.rotational.tests.test_kmeans_to_localstability import (
        create_kmeans_obstacle_from_learner,
    )

    # Make sure surface point is close (much smaller than the surface)
    position = np.array([-2.5, 5])
    index = 0
    tmp_obstacle = create_kmeans_obstacle_from_learner(main_learner, index)
    surf_point = tmp_obstacle.get_point_on_surface(position, in_global_frame=True)
    assert LA.norm(surf_point - tmp_obstacle.center_position) < 2

    # index = 0
    # tmp_obstacle = create_kmeans_obstacle_from_learner(main_learner, index)
    # position = np.array([-3.04769137, -0.12654154])
    # tmp_obstacle.radius = 1.6677169816152546
    # gamma = tmp_obstacle.get_gamma(position, in_global_frame=True)

    plt.close("all")
    fig, ax = plt.subplots()
    for ii in range(main_learner.n_clusters):
        tmp_obstacle = create_kmeans_obstacle_from_learner(main_learner, ii)

        positions = tmp_obstacle.evaluate_surface_points(n_points=500)
        ax.plot(
            positions[0, :],
            positions[1, :],
            color="black",
            linewidth=3.5,
            zorder=20,
        )
        ax.plot(tmp_obstacle.center_position[0], tmp_obstacle.center_position[1], "k+")

    ax.plot(datahandler.attractor[0], datahandler.attractor[1], "k*")

    main_learner.plot_kmeans(ax=ax)

    pass


if (__name__) == "__main__":
    plt.ion()
    # plt.close("all")

    test_distributed_kmeans()
