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

import os
import sys
import copy
import random
import warnings

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import scipy

import networkx as nx

from sklearn.cluster import KMeans

from vartools.dynamical_systems import LinearSystem, ConstantValue

from vartools.directional_space import get_angle_space_of_array
from vartools.directional_space import get_directional_weighted_sum

from dynamic_obstacle_avoidance.obstacles import Obstacle

from dynamic_obstacle_avoidance.rotational.datatypes import Vector, VectorArray


NodeType = int


class HandwrittingHandler:
    def __init__(self, file_name, directory_name: str = None, dimension: int = 2):
        if directory_name is None:
            # self.directory_name = "default/directory"
            self.directory_name = os.path.join(
                "/home", "lukas", "Code", "motion_learning_direction_space", "dataset"
            )
        else:
            self.directory_name = directory_name
        self.file_name = file_name

        self.dimension = dimension

        self.load_data_from_mat()

    def load_data_from_mat(self, feat_in=None, attractor=None):
        """Load data from file mat-file & evaluate specific parameters"""

        self.dataset = scipy.io.loadmat(
            os.path.join(self.directory_name, self.file_name)
        )

        if feat_in is None:
            self.feat_in = [0, 1]

        ii = 0  # Only take the first fold.
        self.position = self.dataset["data"][0, ii][: self.dimension, :].T
        self.velocity = self.dataset["data"][0, ii][
            self.dimension : self.dimension * 2, :
        ].T

        self.sequence_value = np.linspace(0, 1, self.dataset["data"][0, ii].shape[1])

        for it_set in range(1, self.dataset["data"].shape[1]):
            self.position = np.vstack(
                (self.position, self.dataset["data"][0, it_set][:2, :].T)
            )
            self.velocity = np.vstack(
                (self.velocity, self.dataset["data"][0, it_set][2:4, :].T)
            )

            # TODO include velocity - rectify
            self.sequence_value = np.hstack(
                (
                    self.sequence_value,
                    np.linspace(0, 1, self.dataset["data"][0, it_set].shape[1]),
                )
            )

        direction = get_angle_space_of_array(
            directions=self.velocity.T,
            positions=self.position.T,
            func_vel_default=LinearSystem(dimension=self.dimension).evaluate,
        )

        self.X = np.hstack((self.position, self.velocity, direction.T))

        # self.X = self.normalize_velocity(self.X)

        self.num_samples = self.X.shape[0]
        self.dim_gmm = self.X.shape[1]

        weightDir = 4

        if attractor is None:
            self.attractor = np.zeros((self.dimension))

            for it_set in range(0, self.dataset["data"].shape[1]):
                self.attractor = (
                    self.attractor
                    + self.dataset["data"][0, it_set][:2, -1].T
                    / self.dataset["data"].shape[1]
                )
                # print("pos_attractor", self.dataset["data"][0, it_set][:2, -1].T)

            print(f"Obstained attractor position [x, y] = {self.attractor}.")

            self.null_ds = LinearSystem(attractor_position=self.attractor)

        elif attractor is False:
            # Does not have attractor
            self.attractor = False
            self.null_ds = attracting_circle
        else:
            self.position_attractor = np.array(attractor)

            self.null_ds = LinearSystem(attractor_position=self.attractor)

        # Normalize dataset
        normalize_dataset = False
        if normalize_dataset:
            self.meanX = np.mean(self.X, axis=0)

            self.meanX = np.zeros(4)
            # X = X - np.tile(meanX , (X.shape[0],1))
            self.varX = np.var(self.X, axis=0)

            # All distances should have same variance
            self.varX[: self.dim] = np.mean(self.varX[: self.dim])

            # All directions should have same variance
            self.varX[self.dim : 2 * self.dim - 1] = np.mean(
                self.varX[self.dim : 2 * self.dim - 1]
            )

            # Stronger weight on directions!
            self.varX[self.dim : 2 * self.dim - 1] = (
                self.varX[self.dim : 2 * self.dim - 1] * 1 / weightDir
            )

            self.X = self.X / np.tile(self.varX, (self.X.shape[0], 1))

        else:
            self.meanX = None
            self.varX = None


def find_intersection_between_line_and_plane(
    line_position: Vector,
    line_direction: Vector,
    plane_position: Vector,
    plane_normal: Vector,
) -> Vector:
    """Returns the intersection position of a plane and a point."""
    basis = get_orthogonal_basis(plane_normal)
    basis[:, 0] = (-1) * line_direction

    factors = LA.pinv(basis) @ (line_position - plane_position)

    return line_position + line_direction * factors[0]


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

    def get_n_features(self) -> int:
        """Returns number of features."""
        # TODO: depreciated
        return self.kmeans.cluster_centers_.shape[0]

    def get_number_of_features(self) -> int:
        """Returns number of features."""
        return self.kmeans.cluster_centers_.shape[0]

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

        # Radius of
        self._evaluate_cluster_hierarchy()
        self.region_radius_ = self.radius_factor * np.max(self.distances_parent)

        self._evaluate_mean_dynamics()

    def _evaluate_mean_dynamics(self):
        """Assigns constant-value-dynamics to all but the first DS."""

        self._dynamics = []

        for ii, label in enumerate(self.get_feature_labels()):
            # if self._graph.nodes[label].pre < 0:
            # pred = next(self._graph.predecessors(label))[0]
            if self._graph.nodes[label]["level"] == 0:
                # Zero level => Parent is root
                self._dynamics.append(
                    LinearSystem(attractor_position=self.data.attractor)
                )

            ind = np.arange(self.kmeans.labels_.shape[0])[self.kmeans.labels_ == label]
            direction = np.mean(self.data.velocity[ind, :], axis=0)

            if norm_dir := LA.norm(direction):
                direction = direction / LA.norm(direction)
            else:
                # Use the K-Means dynamics as default
                direction = self._graph.nodes[label]["direction"]

            self._dynamics.append(ConstantValue(direction))

    def _check_that_main_direction_is_towards_parent(
        self, ind_node: NodeType, direction: Vector, it_max: int = 100
    ):
        """Checks that the main direction point towards the intersection between parent and node"""
        ind_parent = self._graph.nodes[ind_node]["parent"]

        mean_position = (
            self.kmeans.cluster_centers_[ind_node, :]
            + self.kmeans.cluster_centers_[ind_parent, :]
        )

        intersection_position = find_intersection_between_line_and_plane(
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

    def plot_kmeans(self, mesh_distance: float = 0.01, limit_to_radius=True, ax=None):
        reduced_data = self.data.X[:, : self.data.dimension]

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
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
        **kwargs,
    ):
        self._kmeans = kmeans
        self._index = index
        self.radius = radius

        super().__init__(is_boundary=is_boundary, **kwargs)

        # Only calculate the normal direction between obstacles once (!)
        # and the ones which are interesting
        self.ind_relevant = np.arange(self.num_clusters)
        self.ind_relevant = np.delete(self.ind_relevant, self._index)

        self._center_positions = 0.5 * (
            np.tile(
                self._kmeans.cluster_centers_[self._index, :],
                (self.ind_relevant.shape[0], 1),
            )
            + self._kmeans.cluster_centers_[self.ind_relevant, :]
        )

        # Since obstacle are convex -> intersection needs to be part of one or the other
        labels = self._kmeans.predict(self._center_positions)
        ind_close = np.logical_or(labels == self.ind_relevant, labels == self._index)
        self.ind_relevant = self.ind_relevant[ind_close]
        self._center_positions = self._center_positions[ind_close, :]

        normal_directions = np.zeros((self.num_clusters, self.dimension))
        normal_directions[self.ind_relevant, :] = self._kmeans.cluster_centers_[
            self.ind_relevant, :
        ] - np.tile(
            self._kmeans.cluster_centers_[self._index, :],
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

    @property
    def normal_directions(self) -> VectorArray:
        """Returns the full array of normal directions."""
        # TODO: this seems a redundant step -> check if it can be avoided
        normal_directions = np.zeros((self.num_clusters, self.dimension))
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
        return self._kmeans.cluster_centers_[self._index, :]

    @center_position.setter
    def center_position(self, value) -> None:
        """Returns global center point."""
        warnings.warn("Position is not being set.")

    @property
    def reference_point(self) -> Vector:
        """Returns global reference point."""
        return self._kmeans.cluster_centers_[self._index, :]

    @reference_point.setter
    def reference_point(self, value) -> None:
        """Returns global reference point."""
        if LA.norm(value):  # Nonzero value
            raise NotImplementedError(
                "Reference point is not reset for KMeans-Obstacle."
            )

    @property
    def dimension(self) -> int:
        return self._kmeans.cluster_centers_.shape[1]

    @property
    def num_clusters(self) -> int:
        return self._kmeans.cluster_centers_.shape[0]

    def get_gamma(self, position: Vector, in_global_frame: bool = False) -> float:
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

    def get_normal_direction(self, position, in_global_frame: bool = False) -> Vector:
        """Returns smooth-weighted normal direction around the obstacles."""
        if not in_global_frame:
            position = self.pose.transform_position_from_relative(position)

        relative_position = position - self.center_position

        if not (position_norm := LA.norm(relative_position)):
            # Some direction (normed).
            position[0] = 1
            return position

        normal_directions = self.normal_directions
        normal_directions[self._index, :] = relative_position / position_norm

        distances_surface = self._get_normal_distances(position, is_boundary=False)
        distances_surface[self._index] = position_norm - self.radius

        if not LA.norm(distances_surface):
            # Directly on the boundary
            if not in_global_frame:
                self.pose.transform_position_from_relative(
                    normal_directions[self._index, :]
                )
            return normal_directions[self._index, :]

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
            # Point is exactly on the surface -> return normal
            if in_global_frame:
                return relative_position
            else:
                return self.pose.transform_direction_to_relative(relative_position)

        # Only consider positive ones for the weight
        weights = np.maximum(distances_surface, 0)
        weights = weights / np.sum(weights)

        weighted_direction = get_directional_weighted_sum(
            null_direction=normal_directions[self._index, :],
            directions=normal_directions.T,
            weights=weights,
        )

        if in_global_frame:
            return weighted_direction
        else:
            return self.pose.transform_direction_to_relative(weighted_direction)

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

        center_dists = np.zeros(self._kmeans.cluster_centers_.shape[0])
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

        angle = np.linspace(0, 2 * pi, n_points)
        self.surface_points = np.vstack(np.cos(angle), np.sin(angle))

        for ii in range(n_points):
            self.surface_points[:, ii] = self.get_point_on_surface(
                self.surface_points[:, ii], in_global_frame=True
            )

        return self.surface_points


def test_four_cluster_kmean():
    """Test the intersection and surface points"""
    data = np.array([[-1, 0], [1, 0], [3, 0], [3, 2]])

    dimension = 2
    kmeans = KMeans(init="k-means++", n_clusters=4, n_init=2)
    kmeans.fit(data)

    kmeans.n_features_in_ = dimension
    kmeans.cluster_centers_ = np.array(data).copy(order="C").astype(np.double)

    region_obstacle = KmeansObstacle(radius=1.5, kmeans=kmeans, index=0)

    # Test 1
    position = np.array([2, 1])
    surface_position = region_obstacle.get_point_on_surface(
        position, in_global_frame=True
    )
    assert np.isclose(surface_position[0], 0)

    normal_direction = region_obstacle.get_normal_direction(
        position, in_global_frame=True
    )
    # Is in between the two vectors
    assert np.cross([1, 0], normal_direction) > 0
    assert np.cross([0, 1], normal_direction) < 0

    # Test 2
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
    position = np.array([-1, 2])
    surface_position = region_obstacle.get_point_on_surface(
        position, in_global_frame=True
    )
    assert np.allclose(surface_position, [-1, 1.5])

    normal_direction = region_obstacle.get_normal_direction(
        position, in_global_frame=True
    )
    assert np.allclose(normal_direction, [0, 1])

    # Test gammas
    position = np.array([-0.4, 0.1])
    gamma = region_obstacle.get_gamma(position, in_global_frame=True)
    assert gamma > 1

    position = np.array([-3.0, 1.6])
    gamma = region_obstacle.get_gamma(position, in_global_frame=True)
    assert gamma < 1

    position = np.array([0.2, 0.1])
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
        # Test normal
        n_points = 10

        ff = 1.2
        x_min = (
            main_learner.kmeans.cluster_centers_[ii, 0]
            - main_learner.region_radius_ * ff
        )
        x_max = (
            main_learner.kmeans.cluster_centers_[ii, 0]
            + main_learner.region_radius_ * ff
        )
        y_min = (
            main_learner.kmeans.cluster_centers_[ii, 1]
            - main_learner.region_radius_ * ff
        )
        y_max = (
            main_learner.kmeans.cluster_centers_[ii, 1]
            + main_learner.region_radius_ * ff
        )

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, n_points),
            np.linspace(y_min, y_max, n_points),
        )

        positions = np.array([xx.flatten(), yy.flatten()])
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

    if save_figure:
        fig_name = "kmeans_obstacles_multiplot_normal"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    # test_four_cluster_kmean()
    _test_a_matrix_loader(save_figure=True)

    print("Tests finished.")
