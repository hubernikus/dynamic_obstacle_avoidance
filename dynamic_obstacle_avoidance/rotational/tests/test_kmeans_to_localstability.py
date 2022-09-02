"""
Class which allows learning of transition regions / funnels for locally stable regions which ensure
global attraction to final goal

TODO / method:
> K-means learning of transition; ensure that
>> directional space is used for the learning
>> the k-weight ensures transition from one to next [common boundary (!)]
>> between non-consecutive regions there is a 'transition' (use perpendicular arc addition)
>> between consecutive regions there is a smooth flow-through in one direction! which ensures transition
>> the 'welcoming' arc in the subsequent region is cropped such that it does NOT overly any additional obstacle

> GMM for obstacle avoidance
>> Place GMM's such that they are ]-pi/2, pi/2[ with the local 'straight?' dynamics
>> (Maybe) additionally ensure that the flow stays within the 
"""

import sys
import copy
import random
import warnings
import math

from dataclasses import dataclass

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import networkx as nx

from sklearn.cluster import KMeans

from vartools.dynamical_systems import LinearSystem, ConstantValue

from vartools.directional_space import get_directional_weighted_sum

from vartools.handwritting_handler import HandwrittingHandler
from vartools.math import get_intersection_between_line_and_plane

from dynamic_obstacle_avoidance.obstacles import Obstacle

from dynamic_obstacle_avoidance.rotational.datatypes import Vector, VectorArray

NodeType = int


@dataclass
class MotionDataHandler:
    """Stores (and imports) data for evaluation with the various learners.

    Attributes
    ----------
    positions: numpy-VectorArray of shape [n_datapoints x dimension]
    velocities: numpy-VectorArray of shape [n_datapoints x dimension]
    directions: numpy-VectorArray of shape [n_datapoints - 1 x dimension]
    time: numpy-Array of shape[n_datapoints]
    """

    position: VectorArray = None
    velocity: VectorArray = None
    sequence_value: VectorArray = None

    direction: VectorArray = None

    attractor: Vector = None

    @property
    def dimension(self) -> int:
        return self.position.shape[1]

    # def normalize(self):
    #     self.mean_positions = np.mean(self.positions)
    #     self.var_positions = np.variance(self.positions)
    #     self.positions = (seplf.positions - self.mean_positions) / self.var_positions

    @property
    def X(self) -> VectorArray:
        return np.hstack(
            (self.position, self.velocity, self.sequence_value.reshape(-1, 1))
        )


class MotionLearnerThrougKMeans:
    def __init__(self, data: HandwrittingHandler, n_clusters: int = 4):
        self.data = data
        self.n_clusters = n_clusters

        self._graph = nx.DiGraph()

        self.radius_factor = 0.7
        # self.region_radius_ = 1

        # self._graph = None

        # Finally
        self.evaluate_local_sets()

    def get_feature_labels(self) -> np.ndarray:
        return np.arange(self.kmeans.cluster_centers_.shape[0])

    def get_number_of_features(self) -> int:
        """Returns number of features."""
        return self.kmeans.cluster_centers_.shape[0]

    def get_parents(self, index: int) -> list[int]:
        return list(self._graph.predecessors(index))

    def get_children(self, index: int) -> list[int]:
        return list(self._graph.successors(index))

    def evaluate_local_sets(self) -> None:
        self.full_kmeans = KMeans(
            init="k-means++", n_clusters=self.n_clusters, n_init=5
        )

        self.full_kmeans.fit(self.data.X)

        # TODO: would be nice, if you would not have to redo the learning to just be overwritten
        # self.kmeans = KMeans(init="k-means++", n_clusters=self.n_clusters, n_init=4)
        # self.kmeans.fit(self.data.X[:, : self.data.dimension])

        self.kmeans = copy.deepcopy(self.full_kmeans)

        # Reduce k_means to position only (!)
        self.kmeans.n_features_in_ = self.data.dimension
        self.kmeans.cluster_centers_ = self.full_kmeans.cluster_centers_[
            :, : self.data.dimension
        ].copy(order="C")
        self.kmeans.cluster_centers_ = self.kmeans.cluster_centers_.copy(order="C")

        # Evaluate hierarchy and get the 'minimum' distance
        # Get hierarchy just from existing 'sequence label'

        self._evaluate_cluster_hierarchy()
        self.region_radius_ = self.radius_factor * np.max(self.distances_parent)

        self._evaluate_local_dynamics()

        # Create succession obstacles
        self.region_obstacles = []
        for ii in range(self.n_clusters):
            self.region_obstacles.append(
                KmeansObstacle(
                    radius=self.region_radius_,
                    kmeans=self.kmeans,
                    index=ii,
                )
            )

            # Assumption of only one predecessor (!)
            # TODO: several predecessors and successors (?!)
            self.region_obstacles[ii].successor_index = [
                jj for jj in self._graph.successors(ii)
            ]

        # TODO: learn local deviations

    def predict(self, position: Vector) -> Vector:
        # Get k-means-weights
        breakpoint()

        cluster_label = self.kmeans.predict(position.reshapse(1, -1))[0]
        weights = self._get_sequence_weights(position, cluster_label)

        ind_relevant = np.arange(self.num_cluster)[weights > 0]

        weights = weights[ind_relevant]
        dynamics = np.zeros((self.dimension, ind_relevant.shape[0]))

        for ii, index in ind_relevant:
            dynamics[ii] = self._dynamics.evaluate(ii, position)

    def _get_sequence_weights(
        self,
        position: Vector,
        cluster_label: int,
        parent_factor: float = 0.25,
        gamma_cutoff: float = 4.0,
    ) -> np.ndarray:
        """Returns the weights whith which each of the superior clusters is considered

        parent_factor in ]0, 1[: determines far into the new obstacle one can enter.
        gamma_cutoff: ensure local convergences through impenetrability of walls
        """

        parents = self.get_parents(cluster_label)
        if len(parents) > 1:
            raise NotImplementedError("How to treat a cluster with multiple parents?.")

        gamma = self.region_obstacles[cluster_label].get_gamma(
            position, ind_transparent=parents
        )

        center_dist = LA.norm(position - self.kmeans.cluster_centers_[cluster_label, :])
        mean_dist = 0.5 * (
            center_dist
            + LA.norm(position - -self.kmeans.cluster_centers_[parents[0], :])
        )

        tmp_weight = center_dist - mean_dist * parent_factor
        if tmp_weight > 0 and gamma > gamma_cutoff:
            tmp_weight /= (1 - parent_factor) * mean_dist
            # Ensure it stops at boundary
            tmp_weight *= 1 - 1 / (gamma - gamma_cutoff)

        weights = np.zeros((self.n_clusters))
        weights[parents[0]] = weights
        weights[cluster_label] = 1 - np.sum(weights[cluster_label])
        return weights

    def _evaluate_local_dynamics(self):
        """Assigns constant-value-dynamics to all but the first DS."""

        # self._dynamics = [None for _ in self.get_number_of_features()]
        self._dynamics = []

        for label in self.get_feature_labels():
            # if self._graph.nodes[label].pre < 0:
            # pred = next(self._graph.predecessors(label))[0]
            if self._graph.nodes[label]["level"] == 0:
                # Zero level => Parent is root
                self._dynamics.append(
                    LinearSystem(attractor_position=self.data.attractor)
                )
                continue

            ind = np.arange(self.kmeans.labels_.shape[0])[self.kmeans.labels_ == label]

            direction = np.mean(self.data.velocity[ind, :], axis=0)

            if norm_dir := LA.norm(direction):
                direction = direction / norm_dir
            else:
                # Use the K-Means dynamics as default
                direction = self._graph.nodes[label]["direction"]

            self._dynamics.append(ConstantValue(direction))

    def _check_that_main_direction_is_towards_parent(
        self, ind_node: NodeType, direction: Vector, it_max: int = 100
    ):
        """Checks that the main direction point towards the intersection between
        parent and node"""
        ind_parent = self._graph.nodes[ind_node]["parent"]

        mean_position = (
            self.kmeans.cluster_centers_[ind_node, :]
            + self.kmeans.cluster_centers_[ind_parent, :]
        )

        intersection_position = get_intersection_between_line_and_plane(
            self.kmeans.cluster_centers_[ind_node, :],
            direction,
            mean_position,
            self.kmeans.cluster_centers_[ind_node, :]
            - self.kmeans.cluster_centers_[ind_parent, :],
        )

        if self.kmeans.predict(intersection_position) in [ind_node, ind_parent]:
            # Distance does not need to be checked, since intersection position is in
            # the middle by construction
            return

        for ii in range(it_max):
            raise NotImplementedError("TODO: Automatically update the label.")

    def _evaluate_cluster_hierarchy(self):
        """Evaluates the sequence of each cluster along the trajectory.
        -> this can only be used for demonstration which always take the same path."""
        # TODO generalize for multiple (inconsistent sequences) learning
        average_sequence = np.zeros(self.get_number_of_features())
        self.distances_parent = np.zeros_like(average_sequence)

        for ii, label in enumerate(self.get_feature_labels()):
            average_sequence[ii] = np.mean(
                self.data.sequence_value[self.kmeans.labels_ == label]
            )
        sorted_list = np.argsort(average_sequence)[::-1]

        # Set attractor first
        parent_id = -1
        direction = (
            self.kmeans.cluster_centers_[sorted_list[0], :] - self.data.attractor
        )

        if dir_norm := LA.norm(direction):
            direction = direction / dir_norm

        else:
            # What should be done in this case ?! -> go to one level higher?
            raise NotImplementedError()

        # Distance to attractor has to be multiplied by two, to ensure that it's within
        self.distances_parent[0] = dir_norm * 2.0

        self._graph.add_node(sorted_list[0], level=0, direction=direction)

        for jj, ind_node in enumerate(sorted_list[1:], 1):
            ind_parent = sorted_list[jj - 1]

            direction = (
                self.kmeans.cluster_centers_[ind_node, :]
                - self.kmeans.cluster_centers_[ind_parent, :]
            )

            if dir_norm := LA.norm(direction):
                direction = direction / dir_norm
            else:
                raise ValueError("Two kmeans are aligned - check the cluster.")

            self.distances_parent[jj] = dir_norm

            self._graph.add_node(
                ind_node,
                level=self._graph.nodes[ind_parent]["level"] + 1,
                direction=direction,
            )

            self._graph.add_edge(ind_node, ind_parent)

    def plot_kmeans(
        self,
        mesh_distance: float = 0.01,
        limit_to_radius=True,
        ax=None,
        x_lim=None,
        y_lim=None,
    ):
        reduced_data = self.data.X[:, : self.data.dimension]

        if x_lim is None:
            # Plot the decision boundary. For that, we will assign a color to each
            x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        else:
            x_min, x_max = x_lim
        if y_lim is None:
            y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        else:
            y_min, y_max = y_lim

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, mesh_distance),
            np.arange(y_min, y_max, mesh_distance),
        )

        n_points = xx.shape[0] * xx.shape[1]
        # Obtain labels for each point in mesh. Use last trained model.
        Z = self.kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        if limit_to_radius:
            value_far = -1
            for label in self.get_feature_labels():

                xx_flat = xx.flatten()
                yy_flat = yy.flatten()

                ind_level = Z == label

                ind = np.arange(xx_flat.shape[0])[ind_level]

                pos = np.array([xx_flat[ind], yy_flat[ind]]).T

                dist = LA.norm(
                    pos
                    - np.tile(
                        self.kmeans.cluster_centers_[label, :], (np.sum(ind_level), 1)
                    ),
                    axis=1,
                )
                ind = ind[dist > self.region_radius_]

                Z[ind] = value_far

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        if ax is None:
            _, ax = plt.subplots()

        # ax.clf()
        ax.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect="auto",
            origin="lower",
        )

        ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
        # Plot the centroids as a white X
        centroids = self.kmeans.cluster_centers_
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=169,
            linewidths=3,
            color="w",
            zorder=10,
        )

        for ii in range(self.get_number_of_features()):
            d_txt = 0.15
            ax.text(
                self.kmeans.cluster_centers_[ii, 0] + d_txt,
                self.kmeans.cluster_centers_[ii, 1] + d_txt,
                self._graph.nodes[ii]["level"],
                fontsize=20,
                color="white",
            )

        # Plot attractor
        ax.scatter(
            self.data.attractor[0],
            self.data.attractor[1],
            marker="*",
            s=200,
            color="white",
            zorder=10,
        )


class KmeansObstacle(Obstacle):
    """Pseudo obstacle based on a learned kmeans clustering."""

    def __init__(
        self,
        radius: float,
        kmeans: KMeans,
        index: int,
        is_boundary: bool = True,
        main_learner=None,
        **kwargs,
    ):
        self.kmeans = kmeans
        self._index = index
        self.radius = radius

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

        if main_learner is None:
            self.successor_index = []
        else:
            # Assumption of only one predecessor (!)
            self.successor_index = [
                ii for ii in main_learner._graph.successors(self._index)
            ]

    @property
    def normal_directions(self) -> VectorArray:
        """Returns the full array of normal directions."""
        # TODO: this seems a redundant step -> check if it can be avoided
        normal_directions = np.zeros((self.n_clusters, self.dimension))
        normal_directions[self.ind_relevant, :] = self._normal_directions
        return normal_directions

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

    @property
    def dimension(self) -> int:
        return self.kmeans.cluster_centers_.shape[1]

    @property
    def n_clusters(self) -> int:
        return self.kmeans.cluster_centers_.shape[0]

    def _get_gamma_from_point(
        self, position: Vector, in_global_frame: bool = False
    ) -> float:
        if not in_global_frame:
            position = self.pose.transform_position_from_relative(position)

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
        if not in_global_frame:
            position = self.pose.transform_position_from_relative(position)

        relative_position = position - self.center_position
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
        if not in_global_frame:
            position = self.pose.transform_position_from_relative(position)

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

        if in_global_frame:
            return weighted_direction
        else:
            return self.pose.transform_direction_to_relative(weighted_direction)

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
        if not in_global_frame:
            position = self.pose.transform_position_from_relative(position)

        direction = position - self.center_position

        if dir_norm := LA.norm(direction):
            direction = direction / dir_norm
        else:
            # Random value
            direction[0] = 1

        # Default guess: point is on the circle-surface
        boundary_position = self.center_position + direction * self.radius

        # Find closest boundary
        dists = self._get_normal_distances(boundary_position, is_boundary=False)

        # Only change the default if there is an intersection
        if any(dists > 0):
            max_ind = np.argmax(dists)
            boundary_position = self.center_position + direction * (
                self.radius
                - dists[max_ind] / np.dot(self.normal_directions[max_ind, :], direction)
            )

        if not in_global_frame:
            position = self.pose.transform_position_to_relative(position)

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


def test_four_cluster_kmean(visualize=True):
    """Test the intersection and surface points"""
    data = np.array([[-1, 0], [1, 0], [1, 2], [-1, 2]])

    dimension = 2
    kmeans = KMeans(init="k-means++", n_clusters=4, n_init=2)
    kmeans.fit(data)

    kmeans.n_features_in_ = dimension
    kmeans.cluster_centers_ = np.array(data).copy(order="C").astype(np.double)

    region_obstacle = KmeansObstacle(radius=1.5, kmeans=kmeans, index=0)

    if visualize:
        plt.close("all")
        fig, ax = plt.subplots(figsize=(14, 9))

        for ii in range(kmeans.n_clusters):
            tmp_obstacle = KmeansObstacle(radius=1.5, kmeans=kmeans, index=ii)
            positions = tmp_obstacle.evaluate_surface_points()
            ax.plot(positions[0, :], positions[1, :], color="black", linewidth=3.5)

        ff = 1.2
        # Test normal
        positions = get_grid_points(
            region_obstacle.center_position[0],
            region_obstacle.radius * ff,
            region_obstacle.center_position[1],
            region_obstacle.radius * ff,
            n_points=10,
        )

        normals = np.zeros_like(positions)

        for ii in range(positions.shape[1]):
            if region_obstacle.get_gamma(positions[:, ii], in_global_frame=True) < 1:
                continue

            normals[:, ii] = region_obstacle.get_normal_direction(
                positions[:, ii], in_global_frame=True
            )

            if any(np.isnan(normals[:, ii])):
                breakpoint()

        ax.quiver(
            positions[0, :], positions[1, :], normals[0, :], normals[1, :], scale=15
        )

        ax.axis("equal")

    # Test - somewhere in the middle
    position = np.array([2, -1])
    surface_position = region_obstacle.get_point_on_surface(
        position, in_global_frame=True
    )
    assert np.isclose(surface_position[0], 0)

    normal_direction = region_obstacle.get_normal_direction(
        position, in_global_frame=True
    )
    # Is in between the two vectors
    assert np.cross([-1, 0], normal_direction) > 0
    assert np.cross([0, 1], normal_direction) < 0

    # Test
    position = np.array([0.25, 0])
    surface_position = region_obstacle.get_point_on_surface(
        position, in_global_frame=True
    )
    assert np.allclose(surface_position, [0, 0])

    normal_direction = region_obstacle.get_normal_direction(
        position, in_global_frame=True
    )
    assert np.allclose(normal_direction, [1, 0])

    # Test 3
    position = np.array([-1, -2])
    surface_position = region_obstacle.get_point_on_surface(
        position, in_global_frame=True
    )
    assert np.allclose(surface_position, [-1, -1.5])

    normal_direction = region_obstacle.get_normal_direction(
        position, in_global_frame=True
    )
    assert np.allclose(normal_direction, [0, -1])

    # Test gammas
    position = np.array([-0.4, -0.1])
    gamma = region_obstacle.get_gamma(position, in_global_frame=True)
    assert gamma > 1

    position = np.array([-3.0, -1.6])
    gamma = region_obstacle.get_gamma(position, in_global_frame=True)
    assert gamma < 1

    position = np.array([0.2, -0.1])
    gamma = region_obstacle.get_gamma(position, in_global_frame=True)
    assert gamma < 1


def _test_a_matrix_loader(save_figure=False):
    plt.ion()
    plt.close("all")

    RANDOM_SEED = 1
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    data = HandwrittingHandler(file_name="2D_Ashape.mat")
    main_learner = MotionLearnerThrougKMeans(data)

    fig, ax_kmeans = plt.subplots()
    main_learner.plot_kmeans(ax=ax_kmeans)
    if save_figure:
        fig_name = "kmeans_a_shape"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    fig, ax = plt.subplots()
    reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    ax.set_xlim(ax_kmeans.get_xlim())
    ax.set_ylim(ax_kmeans.get_ylim())
    if save_figure:
        fig_name = "raw_data_a_shape"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    for ii in range(main_learner.kmeans.n_clusters):
        ax = axs[ii % 2, ii // 2]

        main_learner.plot_kmeans(ax=ax)

        # Plot a specific obstacle
        region_obstacle = KmeansObstacle(
            radius=main_learner.region_radius_, kmeans=main_learner.kmeans, index=ii
        )

        ff = 1.2
        # Test normal
        positions = get_grid_points(
            main_learner.kmeans.cluster_centers_[ii, 0],
            main_learner.region_radius_ * ff,
            main_learner.kmeans.cluster_centers_[ii, 1],
            main_learner.region_radius_ * ff,
            n_points=10,
        )

        normals = np.zeros_like(positions)

        for ii in range(positions.shape[1]):
            if region_obstacle.get_gamma(positions[:, ii], in_global_frame=True) < 1:
                continue

            normals[:, ii] = region_obstacle.get_normal_direction(
                positions[:, ii], in_global_frame=True
            )

            if any(np.isnan(normals[:, ii])):
                breakpoint()

        ax.quiver(
            positions[0, :], positions[1, :], normals[0, :], normals[1, :], scale=15
        )
        ax.axis("equal")

    if save_figure:
        fig_name = "kmeans_obstacles_multiplot_normal"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


def get_grid_points(mean_x, delta_x, mean_y, delta_y, n_points):
    """Returns grid based on input x and y values."""
    x_min = mean_x - delta_x
    x_max = mean_x + delta_x

    y_min = mean_y - delta_y
    y_max = mean_y + delta_y

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_points),
        np.linspace(y_min, y_max, n_points),
    )

    return np.array([xx.flatten(), yy.flatten()])


def _test_modulation_values(save_figure=False):
    plt.ion()
    # plt.close("all")

    RANDOM_SEED = 1
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    data = HandwrittingHandler(file_name="2D_Ashape.mat")
    main_learner = MotionLearnerThrougKMeans(data)

    fig, ax_kmeans = plt.subplots()
    main_learner.plot_kmeans(ax=ax_kmeans)

    x_lim = ax_kmeans.get_xlim()
    y_lim = ax_kmeans.get_ylim()

    ii = 2
    fig, ax = plt.subplots()

    for ii in range(main_learner.kmeans.n_clusters):
        # Plot a specific obstacle
        region_obstacle = KmeansObstacle(
            radius=main_learner.region_radius_, kmeans=main_learner.kmeans, index=ii
        )

        positions = region_obstacle.evaluate_surface_points()
        ax.plot(positions[0, :], positions[1, :], color="black")
        ax.axis("equal")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        ff = 1.2

        # Test normal
        positions = get_grid_points(
            main_learner.kmeans.cluster_centers_[ii, 0],
            main_learner.region_radius_ * ff,
            main_learner.kmeans.cluster_centers_[ii, 1],
            main_learner.region_radius_ * ff,
            n_points=10,
        )

        velocities = np.zeros_like(positions)

        for jj in range(positions.shape[1]):
            if region_obstacle.get_gamma(positions[:, jj], in_global_frame=True) < 1:
                continue

            velocities[:, jj] = main_learner._dynamics[ii].evaluate(positions[:, jj])

        ax.quiver(
            positions[0, :],
            positions[1, :],
            velocities[0, :],
            velocities[1, :],
            scale=15,
        )

        plt.show()

    if save_figure:
        fig_name = "consecutive_linear_dynamics"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")
        # fig, axs = plt.subplots(2, 2, figsize=(14, 9))
        # for ii in range(main_learner.kmeans.n_clusters):
        # ax = axs[ii % 2, ii // 2]


def test_gamma_and_modulation(visualize=False, save_figure=False):
    """Test the intersection and surface points"""
    plt.ion()
    plt.close("all")

    # Generate very simple dataset
    RANDOM_SEED = 1
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    datahandler = MotionDataHandler(
        position=np.array([[-1, 0], [1, 0], [2, 1], [1, 2]])
    )
    datahandler.velocity = datahandler.position[1:, :] - datahandler.position[:-1, :]
    datahandler.velocity = np.vstack((datahandler.velocity, [[0, 0]]))
    datahandler.attractor = np.array([0.5, 2])
    datahandler.sequence_value = np.linspace(0, 1, 4)

    x_lim = [-3, 5]
    y_lim = [-2.0, 4.0]

    # Learn KMeans from DataSet
    main_learner = MotionLearnerThrougKMeans(datahandler)

    # Find most left obstacle
    n_clusters = main_learner.kmeans.cluster_centers_.shape[0]
    index = np.arange(n_clusters)[
        LA.norm(
            main_learner.kmeans.cluster_centers_ - np.tile([-1, 0], (n_clusters, 1)),
            axis=1,
        )
        == 0
    ][0]

    region_obstacle = KmeansObstacle(
        radius=main_learner.region_radius_,
        kmeans=main_learner.kmeans,
        index=index,
        main_learner=main_learner,
    )

    # Check gamma at the boundary
    position = region_obstacle.center_position.copy()
    position[0] = position[0] - region_obstacle.radius
    gamma = region_obstacle.get_gamma(position, in_global_frame=True)
    assert np.isclose(gamma, 1), "Gamma is expected to be close to 1."

    # Check gamma towards the successor
    position = 0.5 * (
        region_obstacle.center_position
        + main_learner.kmeans.cluster_centers_[region_obstacle.successor_index[0], :]
    )
    gamma = region_obstacle.get_gamma(position, in_global_frame=True)
    assert gamma > 1e9, "Gamma is expected to be very large."

    position[0] = position[0] - region_obstacle.radius * 0.1
    gamma = region_obstacle.get_gamma(position, in_global_frame=True)
    assert gamma > 1e9, "Gamma is expected to be very large."

    # Check inside the obstacle
    position = region_obstacle.center_position.copy()
    position[1] = position[1] + 0.5 * region_obstacle.radius
    gamma = region_obstacle.get_gamma(position, in_global_frame=True)
    assert gamma > 1 and gamma < 10, "Gamma is expected to be in lower positive range."

    if visualize:
        fig, ax = plt.subplots()
        main_learner.plot_kmeans(ax=ax, x_lim=x_lim, y_lim=y_lim)
        ax.axis("equal")

        if save_figure:
            fig_name = "artificial_four_regions_kmeans"
            fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

        fig, ax = _plot_gamma_of_learner(
            main_learner, x_lim, y_lim, hierarchy_passing_gamma=False
        )

        if save_figure:
            fig_name = "gamma_values_without_transition"
            fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

        fig, ax = _plot_gamma_of_learner(
            main_learner, x_lim, y_lim, hierarchy_passing_gamma=True
        )

        if save_figure:
            fig_name = "gamma_values_with_transition"
            fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


def _plot_gamma_of_learner(main_learner, x_lim, y_lim, hierarchy_passing_gamma=True):
    """A local helper function to plot the gamma fields."""
    fig, ax = plt.subplots()

    levels = np.linspace(1, 21, 51)  # For gamma visualization

    for ii in range(main_learner.kmeans.n_clusters):
        if hierarchy_passing_gamma:
            region_obstacle = KmeansObstacle(
                radius=main_learner.region_radius_,
                kmeans=main_learner.kmeans,
                index=ii,
                main_learner=main_learner,
            )
        else:
            region_obstacle = KmeansObstacle(
                radius=main_learner.region_radius_,
                kmeans=main_learner.kmeans,
                index=ii,
            )

        positions = region_obstacle.evaluate_surface_points()
        ax.plot(positions[0, :], positions[1, :], color="black", linewidth=3.5)

        ff = 1.2
        n_grid = 60
        positions = get_grid_points(
            main_learner.kmeans.cluster_centers_[ii, 0],
            main_learner.region_radius_ * ff,
            main_learner.kmeans.cluster_centers_[ii, 1],
            main_learner.region_radius_ * ff,
            n_points=n_grid,
        )

        gammas = np.zeros(positions.shape[1])
        for jj in range(positions.shape[1]):

            if (
                LA.norm(positions[:, jj] - region_obstacle.center_position)
                > region_obstacle.radius
            ):
                # For nicer visualization, only internally
                continue

            gammas[jj] = region_obstacle.get_gamma(
                positions[:, jj], in_global_frame=True
            )

        cntr = ax.contourf(
            positions[0, :].reshape(n_grid, n_grid),
            positions[1, :].reshape(n_grid, n_grid),
            gammas.reshape(n_grid, n_grid),
            levels=levels,
            # cmap="Blues_r",
            # cmap="magma",
            cmap="pink",
            # alpha=0.7,
        )

    cbar = fig.colorbar(cntr)

    ax.axis("equal")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    return fig, ax


if (__name__) == "__main__":
    # test_four_cluster_kmean(visualize=False)
    # test_gamma_and_modulation(visualize=False, save_figure=False)

    # _test_a_matrix_loader(save_figure=False)
    # _test_gamma_values(save_figure=True)

    print("Tests finished.")
