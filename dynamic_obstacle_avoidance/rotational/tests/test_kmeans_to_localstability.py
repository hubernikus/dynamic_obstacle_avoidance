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
import copy

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import scipy

import networkx as nx

from sklearn.cluster import KMeans

from vartools.dynamical_systems import LinearSystem
from vartools.directional_space import get_angle_space_of_array

from dynamic_obstacle_avoidance.obstacle import Obstacle

from dynamic_obstacle_avoidance.rotational.datatypes import Vector


NodeType = int


class HandwrittingHandler:
    def __init__(self, file_name, directory_name: str = None, dimension: float = 2):
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

    def get_gamma(self, ind):
        pass

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


def find_intersection_line_and_plane(
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
            init="k-means++", n_clusters=self.n_clusters, n_init=4
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
        self.region_radius_ = 0.6 * np.max(self.distances_parent)

    def _evaluate_mean_dynamics(self):
        """Assigns constant-value-dynamics to all but the first DS."""

        self._dynamics = []

        for ii, label in enumerate(self.get_feature_labels()):
            if self._graph.nodes[label]["parent"] < 0:
                # Parent is root
                self._dynamics.append(
                    LinearSystem(attractor_position=self.data.attractor)
                )

            ind = np.arange(self.get_n_features)[self.kmeans.labels_ == label]
            direction = np.mean(self.data.position[:, ind], axis=0)

            if norm_dir := LA.norm(direction):
                direction = direction / LA.norm(direction)
            else:
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

        intersection_position = find_intersection_line_and_plane(
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
            raise NotImplementedError("Automatically update the label.")

    def _evaluate_cluster_hierarchy(self):
        """Evaluates the sequence of each cluster along the trajectory.
        -> this can only be used for demonstration which always take the same path."""
        # TODO generalize for multiple (inconsistent sequences) learning
        average_sequence = np.zeros(self.get_n_features())
        self.distances_parent = average_sequence

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
            direction = (
                self.kmeans.cluster_centers_[sorted_list[ind_node], :]
                - self.kmeans.cluster_centers_[sorted_list[ind_node - 1], :]
            )

            if dir_norm := LA.norm(direction):
                direction = direction / dir_norm
            else:
                raise ValueError("Two kmeans are aligned - check the cluster.")

            self.distances_parent[jj] = dir_norm

            ind_parent = sorted_list[jj - 1]
            self._graph.add_node(
                ind_node,
                level=self._graph.nodes[ind_parent]["level"] + 1,
                direction=direction,
            )

            self._graph.add_edge(ind_node, ind_parent)

    def plot_kmeans(self, mesh_distance: float = 0.01, limit_to_radius=True):
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
        plt.figure(1)
        plt.clf()
        plt.imshow(
            Z,
            interpolation="nearest",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            cmap=plt.cm.Paired,
            aspect="auto",
            origin="lower",
        )

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
        # Plot the centroids as a white X
        centroids = self.kmeans.cluster_centers_
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="x",
            s=169,
            linewidths=3,
            color="w",
            zorder=10,
        )

        # Plot attractor
        plt.scatter(
            self.data.attractor[0],
            self.data.attractor[1],
            marker="*",
            s=200,
            color="white",
            zorder=10,
        )


class KmeansObstacle(Obstacle):
    def __init__(self, radius: float, kmeans: KMeans, index: int, **kwargs):
        super().__init__(**kwargs)

        self._kmeans = kmeans
        self._index = index

        self.radius = radius

    def get_gamma(self, position: Vector, in_globacl_frame: bool = False) -> float:
        if not in_global_frame:
            position = self.pose.transform_from_rlative(position)

        self._get_plane_weights(position, in_globacl_frame=True)
        
        pass

    def get_normal_direction(
        self, position, in_global_frame: bool = False
    ) -> normal:
        if not in_global_frame:
            position = self.pose.transform_from_rlative(position)

        if self.is_boundary:
            pass

        # Find all indexes with neighbours
        center_dists = np.zeros(self._kmeans.get_n_features())

        for ind in self._kmeans.get_feature_labels():
            if ind == self.index:
                continue

            center_position = 0.5 * (
                self._kmeans.cluster_centers_[ind, :]
                + self._kmeans.cluster_centers_[self._index, :]
            )

            label = self._kmeans.predict(center_position)

            if not (label == ind or label == self._index):
                continue

            normal_directions[:, ind] = (
                self._kmeans.cluster_centers_[ind, :]
                + self._kmeans.cluster_centers_[self._index, :]
            )

            if not (normal_norm := LA.norm(normal_directions[:, ind])):
                # Zero distance
                continue

            normal_direction[:, ind] = normal_direction[:, ind] / normal_norm
            center_dists[ind] = max(
                0, np.dot(normal_direction, position - center_position)
            )

        # Store the distance to the radius in the original hull
        center_dists[ind] = LA.norm(position) - self.radius
        normal_directions[:, ind] = position / LA.norm(position)

        if not (dist_sum := np.sum(center_dists)):
            normal = np.zeros(self.dimension)
            normal[0] = 1
            return normal
        
        weights = center_dists / dists_sum

        return get_directional_weighted_sum(
            reference_direction=normal_directions[:, ind],
            directions=normal_directions
            weigths=weights,
        )


class GMR():
    pass



def test_a_matrix_loader():
    data = HandwrittingHandler(file_name="2D_Ashape.mat")

    main_learner = MotionLearnerThrougKMeans(data)

    plt.ion()
    plt.close("all")
    main_learner.plot_kmeans()


if (__name__) == "__main__":
    test_a_matrix_loader()
