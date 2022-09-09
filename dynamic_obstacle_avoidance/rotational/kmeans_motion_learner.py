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

>> Allow for 'friendly-neighbour' => a cluster which shares the direction,
  and allows full flow through (!)

> GMM for obstacle avoidance
>> Place GMM's such that they are ]-pi/2, pi/2[ with the local 'straight?' dynamics
>> (Maybe) additionally ensure that the flow stays within the 
"""

import sys
import copy
import random
import warnings
import math
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import networkx as nx

from sklearn.cluster import KMeans

from vartools.dynamical_systems import LinearSystem, ConstantValue

from vartools.directional_space import get_directional_weighted_sum

from vartools.handwritting_handler import MotionDataHandler, HandwrittingHandler
from vartools.math import get_intersection_between_line_and_plane

from dynamic_obstacle_avoidance.obstacles import Obstacle

from dynamic_obstacle_avoidance.rotational.rotational_avoidance import (
    obstacle_avoidance_rotational,
)
from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationTree
from dynamic_obstacle_avoidance.rotational.kmeans_obstacle import KMeansObstacle
from dynamic_obstacle_avoidance.rotational.tests.test_nonlinear_deviation import (
    MultiOutputSVR,
    DeviationOfConstantFlow,
    PerpendicularDeviatoinOfLinearDS,
)


from dynamic_obstacle_avoidance.rotational.base_logger import logger

from dynamic_obstacle_avoidance.rotational.datatypes import Vector, VectorArray

NodeType = int


class KMeansMotionLearner:
    def __init__(
        self, data: HandwrittingHandler, n_clusters: int = 4, radius_factor: float = 0.7
    ):
        self.data = data
        self.n_clusters_fit = n_clusters

        self._graph = nx.DiGraph()
        self._directions = VectorRotationTree()

        self.radius_factor = radius_factor

        # Finally
        self.fit()

    def get_feature_labels(self) -> np.ndarray:
        return np.arange(self.kmeans.cluster_centers_.shape[0])

    def get_number_of_features(self) -> int:
        """Returns number of features."""
        return self.kmeans.cluster_centers_.shape[0]

    def get_predecessors(self, index: int) -> list[int]:
        return list(self._graph.predecessors(index))

    def get_successors(self, index: int) -> list[int]:
        return list(self._graph.successors(index))

    @property
    def n_clusters(self) -> int:
        return self.kmeans.n_clusters

    def fit(self) -> None:
        # Evaluate hierarchy and get the 'minimum' distance
        # Get hierarchy just from existing 'sequence label'
        self._fit_kmeans()

        # Iteratively improve the clustering
        it_fit_max = 1
        for ii in range(it_fit_max):
            self._fit_remove_sparse_clusters()
            self._fit_cluster_hierarchy()
            self.region_radius_ = self.radius_factor * np.max(self.distances_parent)

        self._fit_local_dynamics()

        # Create kmeans-obstacles
        self.region_obstacles = []
        for ii in range(self.n_clusters):
            self.region_obstacles.append(create_kmeans_obstacle_from_learner(self, ii))

            # Assumption of only one predecessor (!)
            # TODO: several predecessors and successors (?!)
            self.region_obstacles[ii].successor_index = self.get_successors(ii)

    def _fit_kmeans(self):
        """Fits kmeans on data."""
        self.full_kmeans = KMeans(
            init="k-means++", n_clusters=self.n_clusters_fit, n_init=5
        )

        self.full_kmeans.fit(self.data.X)
        self.kmeans = copy.deepcopy(self.full_kmeans)

        # Reduce k_means to position only (!)
        self.kmeans.n_features_in_ = self.data.dimension
        # self.kmeans.cluster_centers_ = self.full_kmeans.cluster_centers_[
        #     :, : self.data.dimension
        # ].copy(order="C")

        self.kmeans.cluster_centers_ = np.zeros(
            (self.full_kmeans.n_clusters, self.data.dimension)
        )
        # Update with assigned points
        # TODO: is this actually better ?!
        for ii in range(self.kmeans.n_clusters):
            self.kmeans.cluster_centers_[ii, :] = np.mean(
                self.data.position[self.kmeans.labels_ == ii, :], axis=0
            )

        self.kmeans.cluster_centers_ = self.kmeans.cluster_centers_.copy(order="C")

        # Update labels
        self.kmeans.labels_ = self.kmeans.predict(self.data.position)

    def _fit_remove_sparse_clusters(self, _cluster_removal_factor: float = 0.2) -> None:
        """Removes clusters which are very empty."""
        min_num_of_samples = (
            _cluster_removal_factor * self.data.num_samples / self.n_clusters
        )

        n_delta_label = 0
        for ii in range(self.kmeans.n_clusters):
            ind_label = ii == self.kmeans.labels_

            if np.sum(ind_label) < min_num_of_samples:
                # Remove emtpy clusters
                self.kmeans.n_clusters = self.kmeans.n_clusters - 1

                self.kmeans.cluster_centers_ = np.delete(
                    self.kmeans.cluster_centers_, obj=ii - n_delta_label, axis=0
                )

                n_delta_label += 1
                logger.info(f"Removed cluster with label {ii}.")
                # print(f"Removed cluster with label {ii}.")

            elif n_delta_label:
                # Update labels to have sequence of numbers
                self.kmeans.labels_[ind_label] = (
                    self.kmeans.labels_[ind_label] - n_delta_label
                )

    def _fit_cluster_hierarchy(self):
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

    def _fit_local_dynamics(self):
        """Assigns constant-value-dynamics to all but the first DS."""

        # self._dynamics = [None for _ in self.get_number_of_features()]
        self._dynamics = []

        for ii, label in enumerate(self.get_feature_labels()):
            # if self._graph.nodes[label].pre < 0:
            # pred = next(self._graph.predecessors(label))[0]
            if self._graph.nodes[label]["level"] == 0:
                # Zero level => Parent is root
                self._dynamics.append(
                    PerpendicularDeviatoinOfLinearDS(
                        attractor_position=self.data.attractor,
                        regressor=MultiOutputSVR(kernel="rbf", gamma=0.1),
                    )
                )
                # continue

            else:
                ind = np.arange(self.kmeans.labels_.shape[0])[
                    self.kmeans.labels_ == label
                ]

                direction = np.mean(self.data.velocity[ind, :], axis=0)

                if norm_dir := LA.norm(direction):
                    direction = direction / norm_dir
                else:
                    # Use the K-Means dynamics as default
                    direction = self._graph.nodes[label]["direction"]

                # TODO: how do other regressors perform (?)
                self._dynamics.append(
                    DeviationOfConstantFlow(
                        reference_velocity=direction,
                        regressor=MultiOutputSVR(kernel="rbf", gamma=0.1),
                    )
                )

            # TODO: maybe multiple fits -> choose the best (?)
            labels_local = np.array(
                [label] + self.get_predecessors(label) + self.get_successors(label)
            )
            indexes_local = np.max(
                np.tile(self.kmeans.labels_, (labels_local.shape[0], 1))
                == np.tile(labels_local, (self.kmeans.labels_.shape[0], 1)).T,
                axis=0,
            )
            # breakpoint()
            # print("label", label)
            self._dynamics[label].fit_from_velocities(
                self.data.position[indexes_local, :],
                self.data.velocity[indexes_local, :],
            )

    def _check_that_main_direction_is_towards_parent(
        self, ind_node: NodeType, direction: Vector, it_max: int = 100
    ) -> None:
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

    def evaluate(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, position: Vector) -> Vector:
        # Get k-means-weights
        position = np.array(position)

        cluster_label = self.kmeans.predict(position.reshape(1, -1))[0]
        weights = self._predict_sequence_weights(position, cluster_label)

        ind_relevant = np.arange(self.n_clusters)[weights > 0]

        velocities = np.zeros((self.data.dimension, self.n_clusters))

        for index in ind_relevant:
            # TODO: there is an issue if the 'linear attractor'
            velocities[:, index] = self._dynamics[index].evaluate(position)

        # Modulate only the one which we are currently in
        velocities[:, cluster_label] = obstacle_avoidance_rotational(
            position,
            velocities[:, cluster_label],
            [self.region_obstacles[cluster_label]],
            convergence_velocity=self._dynamics[
                cluster_label
            ].evaluate_convergence_velocity(position),
            sticky_surface=False,
        )

        if np.sum(weights) < 1:
            # TODO: allow for 'partial' weight, for e.g.,:
            # - in between two non-neighboring ellipses
            # - to transition from the outside to the inside (!)
            # => create a 'transition margin' to allow for this!
            # (make sure invariance of the region)
            raise NotImplementedError()

        # TODO: use partial vector_rotation (instead)
        weighted_direction = get_directional_weighted_sum(
            null_direction=velocities[:, cluster_label],
            directions=velocities,
            weights=weights,
        )

        return weighted_direction

    def _predict_directional_sum(self, position: Vector, weights, velocities) -> Vector:
        tmp_directions = VectorRotationTree()
        # TODO
        # -> sorted hierarchy
        # -> additionaly add modulated velocities / directions
        # -> return weighted evaluation

    def _predict_sequence_weights(
        self,
        position: Vector,
        cluster_label: int = None,
        parent_factor: float = 0.5,
        gamma_cutoff: float = 2.0,
    ) -> np.ndarray:
        """Returns the weights whith which each of the superior clusters is considered

        parent_factor in ]0, 1[: determines far into the new obstacle one can enter.
        gamma_cutoff: ensure local convergences through impenetrability of walls
        """
        # TODO:
        # -> evalution for several obstacles (i.e. in the outside region)
        # -> how to additonally incoorporate a margin for smooth transition / vectorfield

        if cluster_label is None:
            cluster_label = self.kmeans.predict(position.reshape(1, -1))[0]

        parents = self.get_predecessors(cluster_label)
        if len(parents) > 1:
            raise NotImplementedError("How to treat a cluster with multiple parents?.")

        elif not len(parents):
            # No parent -> whole weight on child
            weights = np.zeros((self.n_clusters))
            weights[cluster_label] = 1
            return weights

        parent = parents[0]

        center = 0.5 * (
            self.kmeans.cluster_centers_[cluster_label, :]
            + self.kmeans.cluster_centers_[parent, :]
        )
        norm_dist = (
            self.kmeans.cluster_centers_[cluster_label, :]
            - self.kmeans.cluster_centers_[parent, :]
        )
        norm_dist = norm_dist / LA.norm(norm_dist)

        pos_dist = norm_dist.dot(position - center)
        center_dist = norm_dist.dot(
            self.kmeans.cluster_centers_[cluster_label, :] - center
        )

        tmp_weight = max(1 - max(pos_dist, 0) / (center_dist * parent_factor), 0)

        if tmp_weight > 0:
            # Gamma of label limits the weight of the superiors
            # to ensure it stops at boundary
            gamma = self.region_obstacles[cluster_label].get_gamma(
                position, in_global_frame=True, ind_transparent=parent
            )

            tmp_weight *= 1 - 1 / (1 + max(gamma - gamma_cutoff, 0))

        weights = np.zeros((self.n_clusters))
        weights[parent] = tmp_weight
        weights[cluster_label] = 1 - np.sum(weights)
        return weights

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
            level = self._graph.nodes[ii]["level"]
            ax.text(
                self.kmeans.cluster_centers_[ii, 0] + d_txt,
                self.kmeans.cluster_centers_[ii, 1] + d_txt,
                f"{ii} @ {level}",
                fontsize=20,
                color="black",
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

    def plot_boundaries(self, ax, plot_attractor=False) -> None:
        for ii in range(self.kmeans.n_clusters):
            tmp_obstacle = create_kmeans_obstacle_from_learner(self, ii)

            positions = tmp_obstacle.evaluate_surface_points()
            ax.plot(
                positions[0, :],
                positions[1, :],
                color="black",
                linewidth=3.5,
                zorder=20,
            )

        if plot_attractor:
            ax.scatter(
                self.data.attractor[0],
                self.data.attractor[1],
                marker="*",
                s=200,
                color="black",
                zorder=10,
            )


def create_kmeans_obstacle_from_learner(
    learner: KMeansMotionLearner, index: int
) -> KMeansObstacle:
    """Simple KMeans-factory.

    Note that this is defined alongside the MotionLearnerThroughKMeans,
    to avoid circular imports."""
    instance = KMeansObstacle(
        kmeans=learner.kmeans,
        radius=learner.region_radius_,
        index=index,
    )

    instance.successor_index = learner.get_successors(index)

    return instance
