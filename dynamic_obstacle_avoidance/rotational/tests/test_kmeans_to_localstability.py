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

from dynamic_obstacle_avoidance.rotational.rotational_avoidance import (
    obstacle_avoidance_rotational,
)
from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationTree
from dynamic_obstacle_avoidance.rotational.kmeans_obstacle import KmeansObstacle
from dynamic_obstacle_avoidance.rotational.tests.test_nonlinear_deviation import (
    MultiOutputSVR,
    DeviationOfConstantFlow,
    PerpendicularDeviatoinOfLinearDS,
)

from dynamic_obstacle_avoidance.rotational.datatypes import Vector, VectorArray

NodeType = int

figure_type = ".png"
# figure_type = ".pdf"


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
    def __init__(
        self, data: HandwrittingHandler, n_clusters: int = 4, radius_factor: float = 0.7
    ):
        self.data = data
        self.n_clusters = n_clusters

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

    def fit(self) -> None:
        self.full_kmeans = KMeans(
            init="k-means++", n_clusters=self.n_clusters, n_init=5
        )

        self.full_kmeans.fit(self.data.X)
        self.kmeans = copy.deepcopy(self.full_kmeans)

        # Reduce k_means to position only (!)
        self.kmeans.n_features_in_ = self.data.dimension
        self.kmeans.cluster_centers_ = self.full_kmeans.cluster_centers_[
            :, : self.data.dimension
        ].copy(order="C")
        self.kmeans.cluster_centers_ = self.kmeans.cluster_centers_.copy(order="C")

        # Evaluate hierarchy and get the 'minimum' distance
        # Get hierarchy just from existing 'sequence label'

        self._fit_cluster_hierarchy()
        self.region_radius_ = self.radius_factor * np.max(self.distances_parent)

        self._fit_local_dynamics()

        # Create succession obstacles
        self.region_obstacles = []
        for ii in range(self.n_clusters):
            self.region_obstacles.append(create_kmeans_obstacle_from_learner(self, ii))

            # Assumption of only one predecessor (!)
            # TODO: several predecessors and successors (?!)
            self.region_obstacles[ii].successor_index = self.get_successors(ii)

        # TODO: learn local deviations

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
            convergence_velocity=velocities[:, cluster_label],
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

    def plot_boundaries(self, ax) -> None:
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


def create_kmeans_obstacle_from_learner(
    learner: MotionLearnerThrougKMeans, index: int
) -> KmeansObstacle:
    """Simple KMeans-factory.

    Note that this is defined alongside the MotionLearnerThroughKmeans,
    to avoid circular imports."""
    instance = KmeansObstacle(
        kmeans=learner.kmeans,
        radius=learner.region_radius_,
        index=index,
    )

    instance.successor_index = learner.get_successors(index)

    return instance


def test_surface_position_and_normal(visualize=True):
    """Test the intersection and surface points"""
    datahandler = MotionDataHandler(
        # position=np.array([[-1, 0], [1, 0], [1, 2], [-1, 2]])
        position=np.array([[-1, 0], [1, 0], [2, 1], [1, 2]])
    )

    datahandler.velocity = datahandler.position[1:, :] - datahandler.position[:-1, :]
    datahandler.velocity = np.vstack((datahandler.velocity, [[0, 0]]))
    datahandler.attractor = np.array([0.5, 2])
    datahandler.sequence_value = np.linspace(0, 1, 4)

    dimension = 2
    kmeans = KMeans(init="k-means++", n_clusters=4, n_init=2)
    kmeans.fit(datahandler.position)

    kmeans.n_features_in_ = dimension
    kmeans.cluster_centers_ = (
        np.array(datahandler.position).copy(order="C").astype(np.double)
    )

    if visualize:
        plt.close("all")
        x_lim, y_lim = [-3, 5], [-2.0, 4.0]

        radius = 1.5

        fig, ax = plt.subplots(figsize=(14, 9))
        main_learner = MotionLearnerThrougKMeans(datahandler)
        main_learner.kmeans = kmeans
        main_learner.region_radius_ = radius
        main_learner.plot_kmeans(x_lim=x_lim, y_lim=y_lim, ax=ax)

        ax.axis("equal")

        for ii in range(kmeans.n_clusters):
            # for ii in [1]:
            tmp_obstacle = KmeansObstacle(radius=radius, kmeans=kmeans, index=ii)
            positions = tmp_obstacle.evaluate_surface_points()
            ax.plot(positions[0, :], positions[1, :], color="black", linewidth=3.5)

        region_obstacle = KmeansObstacle(radius=radius, kmeans=kmeans, index=0)

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

    region_obstacle = KmeansObstacle(radius=radius, kmeans=kmeans, index=0)
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


def test_gamma_kmeans(visualize=False, save_figure=False):
    """Test the intersection and surface points"""
    # TODO: maybe additional check how well gamma is working
    main_learner = create_four_point_datahandler()

    index = main_learner.kmeans.predict([[-1, 0]])[0]
    region_obstacle = create_kmeans_obstacle_from_learner(main_learner, index)

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
        plt.ion()
        plt.close("all")

        x_lim = [-3, 5]
        y_lim = [-2.0, 4.0]

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
            region_obstacle = create_kmeans_obstacle_from_learner(main_learner, ii)

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


def create_four_point_datahandler() -> MotionLearnerThrougKMeans:
    """Helper function to create handler"""
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

    return MotionLearnerThrougKMeans(datahandler, radius_factor=0.55)


def _test_evaluate_partial_deviations(visualize=False, save_figure=False):
    # Generate very simple dataset
    main_learner = create_four_point_datahandler()
    x_lim = [-2.1, 3.1]
    y_lim = [-1.1, 3.1]

    # Get bottom left obstacle
    index = main_learner.kmeans.predict([[1, 0]])

    # ii = 1
    # position = np.array([1.5, 3.2])
    ii = 3
    # position = np.array([2.5, 1.8])
    position = np.array([2.4, 1.9])

    region_obstacle = create_kmeans_obstacle_from_learner(main_learner, ii)
    initial_velocity = main_learner._dynamics[ii].evaluate(position)

    # norm_dir = region_obstacle.get_normal_direction(position)
    # ref_dir = region_obstacle.get_reference_direction(position)
    # breakpoint()

    modulated_velocity = obstacle_avoidance_rotational(
        position,
        initial_velocity,
        [region_obstacle],
        convergence_velocity=initial_velocity,
    )

    if visualize:
        plt.ion()
        plt.close("all")

        n_grid = 20
        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_grid),
            np.linspace(y_lim[0], y_lim[1], n_grid),
        )
        positions = np.array([xx.flatten(), yy.flatten()])

        velocities = np.zeros_like(positions)

        k_obstacles = [
            create_kmeans_obstacle_from_learner(main_learner, ii)
            for ii in range(main_learner.n_clusters)
        ]

        for pp in range(positions.shape[1]):
            is_inside = False
            for obs in k_obstacles:
                if obs.is_inside(positions[:, pp], in_global_frame=True):
                    is_inside = True
                    break

            if not is_inside:
                continue

            velocities[:, pp] = main_learner.predict(positions[:, pp])

        fig, ax = plt.subplots()
        main_learner.plot_boundaries(ax=ax)
        ax.quiver(
            positions[0, :],
            positions[1, :],
            velocities[0, :],
            velocities[1, :],
            # color="red",
            scale=40,
        )
        ax.axis("equal")

        if save_figure:
            fig_name = "basic_rotated_velocity"
            fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

        fig_init, axs_init = plt.subplots(2, 2, figsize=(14, 9))
        fig_mod, axs_mod = plt.subplots(2, 2, figsize=(14, 9))

        for ii in range(main_learner.n_clusters):
            ax_ini = axs_init[ii % 2, ii // 2]
            ax_mod = axs_mod[ii % 2, ii // 2]

            main_learner.plot_boundaries(ax=ax_ini)
            main_learner.plot_boundaries(ax=ax_mod)

            ff = 1.05
            n_grid = 10
            positions = get_grid_points(
                main_learner.kmeans.cluster_centers_[ii, 0],
                main_learner.region_radius_ * ff,
                main_learner.kmeans.cluster_centers_[ii, 1],
                main_learner.region_radius_ * ff,
                n_points=n_grid,
            )
            initial_velocities = np.zeros_like(positions)
            modulated_velocities = np.zeros_like(positions)
            region_obstacle = create_kmeans_obstacle_from_learner(main_learner, ii)

            for jj in range(positions.shape[1]):
                if not region_obstacle.is_inside(
                    positions[:, jj], in_global_frame=True
                ):
                    continue

                initial_velocities[:, jj] = main_learner._dynamics[ii].evaluate(
                    positions[:, jj]
                )

                modulated_velocities[:, jj] = obstacle_avoidance_rotational(
                    positions[:, jj],
                    initial_velocities[:, jj],
                    [region_obstacle],
                    convergence_velocity=initial_velocities[:, jj],
                    sticky_surface=False,
                )

            ax_ini.quiver(
                positions[0, :],
                positions[1, :],
                initial_velocities[0, :],
                initial_velocities[1, :],
                # color="blue",
                scale=15,
            )
            ax_ini.axis("equal")

            ax_mod.quiver(
                positions[0, :],
                positions[1, :],
                modulated_velocities[0, :],
                modulated_velocities[1, :],
                # color="red",
                scale=15,
            )
            ax_mod.axis("equal")

        if save_figure:
            fig_name = "initial_local_velocities"
            fig_init.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

            fig_name = "modulated_local_velocities"
            fig_mod.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


def test_transition_weight(visualize=False, save_figure=False):
    main_learner = create_four_point_datahandler()
    x_lim, y_lim = [-3, 5], [-1.5, 4.5]

    index = main_learner.kmeans.predict([[1, 0]])[0]
    ind_parent = main_learner.get_predecessors(index)[0]

    if visualize:
        plt.close("all")
        plt.ion()

        x_lim_test = [-2.5, 2.5]
        y_lim_test = [-1.5, 1.5]

        n_grid = 40
        xx, yy = np.meshgrid(
            np.linspace(x_lim_test[0], x_lim_test[1], n_grid),
            np.linspace(y_lim_test[0], y_lim_test[1], n_grid),
        )
        positions = np.array([xx.flatten(), yy.flatten()])
        weights = np.zeros((main_learner.n_clusters, positions.shape[1]))

        for pp in range(positions.shape[1]):
            weights[:, pp] = main_learner._fit_sequence_weights(positions[:, pp], index)
        # breakpoint()

        fig, ax = plt.subplots()
        main_learner.plot_kmeans(x_lim=x_lim, y_lim=y_lim, ax=ax)

        fig, ax = plt.subplots()
        main_learner.plot_boundaries(ax=ax)

        levels = np.linspace(0, 1, 11)

        cntr = ax.contourf(
            positions[0, :].reshape(n_grid, n_grid),
            positions[1, :].reshape(n_grid, n_grid),
            weights[index].reshape(n_grid, n_grid),
            levels=levels,
            cmap="cool",
            # alpha=0.7,
        )
        fig.colorbar(cntr)
        ax.axis("equal")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        if save_figure:
            fig_name = f"transition_weights_{index}"
            fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

        # Plot normal gamma with parent
        fig, ax = plt.subplots()
        region_obstacle = create_kmeans_obstacle_from_learner(main_learner, index)
        gammas = np.zeros(positions.shape[1])
        for pp in range(positions.shape[1]):
            gammas[pp] = region_obstacle.get_gamma(
                positions[:, pp],
                in_global_frame=True,
                ind_transparent=ind_parent,
            )

        levels = np.linspace(1, 10, 10)
        main_learner.plot_boundaries(ax=ax)
        cntr = ax.contourf(
            positions[0, :].reshape(n_grid, n_grid),
            positions[1, :].reshape(n_grid, n_grid),
            gammas.reshape(n_grid, n_grid),
            levels=levels,
            cmap="cool",
            extend="max",
            # alpha=0.7,
        )
        fig.colorbar(cntr)
        ax.axis("equal")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        if save_figure:
            fig_name = f"gamma_values_cluster_{index}"
            fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    # Weight behind (inside parent-cluster)
    position = np.array([-0.91, 0.60])
    weights = main_learner._predict_sequence_weights(position, index)
    expected_weights = np.zeros_like(weights)
    expected_weights[ind_parent] = 1
    assert np.allclose(weights, expected_weights)

    # Weight at top border (inside index-cluster)
    position = np.array([0.137, -0.625])
    weights = main_learner._predict_sequence_weights(position, index)
    expected_weights = np.zeros_like(weights)
    expected_weights[index] = 1
    assert np.allclose(weights, expected_weights)

    # Weight at parent center
    position = main_learner.kmeans.cluster_centers_[ind_parent, :]
    weights = main_learner._predict_sequence_weights(position, index)
    expected_weights = np.zeros_like(weights)
    expected_weights[ind_parent] = 1
    assert np.allclose(weights, expected_weights)

    # Weight at cluster center point
    position = main_learner.kmeans.cluster_centers_[index, :]
    weights = main_learner._predict_sequence_weights(position, index)
    expected_weights = np.zeros_like(weights)
    expected_weights[index] = 1
    assert np.allclose(weights, expected_weights)

    # Weight at transition point
    position = np.array([0, 0])
    weights = main_learner._predict_sequence_weights(position, index)
    expected_weights = np.zeros_like(weights)
    expected_weights[ind_parent] = 1
    assert np.allclose(weights, expected_weights)


def test_normals(visualize=False, save_figure=False):
    # TODO: some real test
    main_learner = create_four_point_datahandler()
    x_lim, y_lim = [-3.0, 4.5], [-1.5, 3.5]

    if visualize:
        plt.ion()
        plt.close("all")

        fig, ax_kmeans = plt.subplots()
        main_learner.plot_kmeans(ax=ax_kmeans, x_lim=x_lim, y_lim=y_lim)
        ax_kmeans.axis("equal")
        ax_kmeans.set_xlim(x_lim)
        ax_kmeans.set_ylim(y_lim)

        if save_figure:
            fig_name = "kmeans_shape"
            fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

        fig, axs = plt.subplots(2, 2, figsize=(14, 9))
        for ii in range(main_learner.kmeans.n_clusters):
            ax = axs[ii % 2, ii // 2]

            main_learner.plot_kmeans(ax=ax, x_lim=x_lim, y_lim=y_lim)

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
                if (
                    region_obstacle.get_gamma(positions[:, ii], in_global_frame=True)
                    < 1
                ):
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


def _test_local_deviation(visualize=False, save_figure=False):
    plt.close("all")

    RANDOM_SEED = 1
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    x_lim = [-5.5, 0.5]
    y_lim = [-1.5, 3.0]

    data = HandwrittingHandler(file_name="2D_Ashape.mat")
    main_learner = MotionLearnerThrougKMeans(data)

    fig, ax_kmeans = plt.subplots()
    main_learner.plot_kmeans(ax=ax_kmeans, x_lim=x_lim, y_lim=y_lim)
    ax_kmeans.axis("equal")
    # ax_kmeans.set_xlim(x_lim)
    # ax_kmeans.set_ylim(y_lim)

    if save_figure:
        fig_name = "kmeans_a_shape"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    fig, ax = plt.subplots()
    reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        fig_name = "raw_data_a_shape"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    for index in range(main_learner.kmeans.n_clusters):
        # if index == 1:
        # continue
        print(f"Doing index {index}")

        index_neighbourhood = np.array(
            [index]
            + main_learner.get_successors(index)
            + main_learner.get_predecessors(index)
        )

        x_min = (
            np.min(main_learner.kmeans.cluster_centers_[index_neighbourhood, 0])
            - main_learner.region_radius_
        )
        x_max = (
            np.max(main_learner.kmeans.cluster_centers_[index_neighbourhood, 0])
            + main_learner.region_radius_
        )

        y_min = (
            np.min(main_learner.kmeans.cluster_centers_[index_neighbourhood, 1])
            - main_learner.region_radius_
        )
        y_max = (
            np.max(main_learner.kmeans.cluster_centers_[index_neighbourhood, 1])
            + main_learner.region_radius_
        )

        n_grid = 40
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, n_grid),
            np.linspace(y_min, y_max, n_grid),
        )
        positions = np.array([xx.flatten(), yy.flatten()])
        predictions = np.zeros(positions.shape[0])

        levels = np.linspace(-pi / 2, pi / 2, 50)

        predictions = main_learner._dynamics[index].predict(positions.T)
        for pp in range(positions.shape[1]):
            is_inside = False
            for ii in index_neighbourhood:
                if main_learner.region_obstacles[ii].is_inside(
                    positions[:, pp], in_global_frame=True
                ):
                    is_inside = True
                    break

            if not is_inside:
                predictions[pp] = 0

        fig, ax = plt.subplots(figsize=(10, 8))
        # fig, ax = fig.subplots(figsize=(14, 9))
        cntr = ax.contourf(
            positions[0, :].reshape(n_grid, n_grid),
            positions[1, :].reshape(n_grid, n_grid),
            predictions.reshape(n_grid, n_grid),
            levels=levels,
            # cmap="cool",
            cmap="seismic",
            # alpha=0.7,
            extend="both",
        )
        fig.colorbar(cntr)

        main_learner.plot_boundaries(ax=ax)

        reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
        ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

        # Plot attractor
        ax.scatter(
            main_learner.data.attractor[0],
            main_learner.data.attractor[1],
            marker="*",
            s=200,
            color="black",
            zorder=10,
        )

        ax.axis("equal")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # main_learner.kmeans.
        # breakpoint()

        if save_figure:
            fig_name = f"local_rotation_around_neighbourhood_{index}"
            fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    # test_surface_position_and_normal(visualize=True)
    # test_gamma_kmeans(visualize=True, save_figure=False)
    # test_transition_weight(visualize=True, save_figure=True)
    # test_normals(visualize=True)

    _test_local_deviation(save_figure=True)
    # _test_evaluate_partial_deviations(visualize=True, save_figure=False)
    # _test_gamma_values(save_figure=True)

    print("Tests finished.")
