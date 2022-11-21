"""
Class which allows learning of transition regions / funnels for locally stable regions which ensure
global attraction to final goal

TODO / method:
> K-means learning of transition; ensure that
>> there exists / is a smoothly defined Lyapunov function / contraction metric which ensures
converging flow even at 'passage regions', the Lyapunov function is the sum of the previous
ones

>> Allow for 'friendly-neighbour' => a cluster which shares the direction,
  and allows full flow through (!) /

>> Transition Region: 
"""

import copy
import warnings
import math
from typing import Optional

import numpy as np
from numpy import linalg as LA

import networkx as nx

from sklearn.cluster import KMeans

from vartools.directional_space import get_directional_weighted_sum
from vartools.dynamical_systems import LinearSystem

from vartools.handwritting_handler import HandwrittingHandler

# from vartools.math import get_intersection_between_line_and_plane

from dynamic_obstacle_avoidance.rotational.base_logger import logger
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

from dynamic_obstacle_avoidance.rotational.tests import helper_functions

# from dynamic_obstacle_avoidance.rotational.base_logger import logger
from dynamic_obstacle_avoidance.rotational.datatypes import Vector

# from dynamic_obstacle_avoidance.rotational.datatypes import VectorArray

NodeType = int


class KMeansMotionLearner:
    """
    This class learns and predicts highly non-linear motion (!)

    Attributes
    ----------
    region_radius_: The radius-value of all obstacles (for now we consider
        uniform size of the obstacles

    """

    def __init__(
        self,
        data: HandwrittingHandler = None,
        n_clusters: int = 4,
        radius_factor: float = 0.7,
        repulsive_boundary: bool = True,
    ):
        self.n_clusters_fit = n_clusters

        self._graph = nx.DiGraph()
        self._directions = VectorRotationTree()

        self.region_radius_ = 0
        self.radius_factor = radius_factor

        # Finally
        if data is not None:
            self.fit(data)

        self.repulsive_boundary = repulsive_boundary
        self.convergence_radius = math.pi if self.repulsive_boundary else math.pi
        # self.convergence_cos

    @classmethod
    def from_centers(cls, cluster_centers, data):
        """Alternative contstructor."""
        n_clusters = cluster_centers.shape[1]
        new_instance = cls(n_clusters=n_clusters)
        new_instance.data = data

        new_instance.kmeans = KMeans(
            n_clusters=n_clusters, n_init=1, init=cluster_centers.T, max_iter=1
        )
        new_instance.kmeans.fit(data.position)

        new_instance._fit_cluster_hierarchy()
        new_instance._fit_region_radius()
        new_instance._fit_region_obstacles()
        new_instance._fit_local_dynamics()

        return new_instance

    @property
    def attractor_position(self) -> Vector:
        return self.data.attractor_position

    @property
    def n_clusters(self) -> int:
        return self.kmeans.n_clusters

    @property
    def dimension(self) -> int:
        return self.data.dimension

    def get_distance_from_centers(self, position: Vector) -> np.ndarray:
        return LA.norm(
            np.tile(position, (self.kmeans.cluster_centers_.shape[0], 1))
            - self.kmeans.cluster_centers_,
            axis=1,
        )

    def get_feature_labels(self) -> np.ndarray:
        return np.arange(self.kmeans.cluster_centers_.shape[0])

    def get_number_of_features(self) -> int:
        """Returns number of features."""
        return self.kmeans.cluster_centers_.shape[0]

    def get_predecessors(self, index: int) -> list[int]:
        return list(self._graph.predecessors(index))

    def get_successors(self, index: int) -> list[int]:
        return list(self._graph.successors(index))

    def fit(self, data) -> None:
        self.data = data

        # Evaluate hierarchy and get the 'minimum' distance
        # Get hierarchy just from existing 'sequence label'
        self._fit_kmeans()

        # Iteratively improve the clustering
        it_fit_max = 1
        for ii in range(it_fit_max):
            self._fit_remove_sparse_clusters()
            self._fit_cluster_hierarchy()

        self._fit_region_radius()
        self._fit_region_obstacles()
        self._fit_local_dynamics()

        # Set 'general'/ global velocity (more complex motion could be approximated...)
        self.outside_dynamics = LinearSystem(
            attractor_position=self.data.attractor_position,
            maximum_velocity=1.0,
        )

    def _fit_region_obstacles(self):
        # Create kmeans-obstacles
        self.region_obstacles = []
        for ii in range(self.n_clusters):
            self.region_obstacles.append(create_kmeans_obstacle_from_learner(self, ii))

            # Assumption of only one predecessor (!)
            # TODO: several predecessors and successors (?!)
            self.region_obstacles[ii].successor_index = self.get_successors(ii)

    def _fit_region_radius(self):
        max_dist_clusters = self.radius_factor * np.max(self.distances_parent)

        dists = LA.norm(
            np.swapaxes(
                np.tile(
                    self.kmeans.cluster_centers_, (self.data.position.shape[0], 1, 1)
                ),
                axis1=0,
                axis2=1,
            )
            - np.tile(self.data.position, (self.n_clusters, 1, 1)),
            axis=2,
        )

        max_dist_data = 2 * self.radius_factor * np.max(np.min(dists, axis=0))

        self.region_radius_ = max(max_dist_data, max_dist_clusters)

    def _fit_kmeans(self, normalize_init: bool = False):
        """Fits kmeans on data."""
        self.full_kmeans = KMeans(
            init="k-means++", n_clusters=self.n_clusters_fit, n_init=5
        )
        if normalize_init:
            self.full_kmeans.fit(self.data.get_normalized_data())
        else:
            self.full_kmeans.fit(self.data.X)

        # Update with assigned points -> this is not just and 'update step' but uses
        # information of velocity and sequence, too.
        cluster_centers = np.zeros((self.full_kmeans.n_clusters, self.dimension))
        for ii in range(self.full_kmeans.n_clusters):
            cluster_centers[ii, :] = np.mean(
                self.data.position[self.full_kmeans.labels_ == ii, :], axis=0
            )

        # Initialize-kMeans, but only do few steps since the iteration is position only (!)
        self.kmeans = KMeans(
            n_clusters=self.n_clusters_fit, n_init=1, init=cluster_centers, max_iter=10
        )
        self.kmeans.fit(self.data.position)

    def _fit_remove_sparse_clusters(self, _cluster_removal_factor: float = 0.2) -> None:
        """Removes clusters which are very empty."""
        min_num_of_samples = (
            _cluster_removal_factor * self.data.n_samples / self.n_clusters
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

                self.kmeans.labels_[ind_label] = self.kmeans.predict(
                    self.data.position[ind_label, :]
                )

            elif n_delta_label:
                # Update labels to have sequence of numbers
                self.kmeans.labels_[ind_label] = (
                    self.kmeans.labels_[ind_label] - n_delta_label
                )

    def _fit_new_clusters_to_missfitting_datapoints(self):
        # TODO:

        # -> get directions (without caring about parent)
        # -> remove very small clusters
        # -> Check how much local velocities are aligned
        # --> maybe a split (?)
        # ---> check again, that the locals are now good
        #
        # -> do the hierarchy
        # -> get direction, but has to be aligned with the superiors (!)
        # --> adapt the hieararchy if needed (!)
        breakpoint()

        data = self.data.get_normalized_data()
        # TODO: check sequence
        for ii, label in enumerate(self.get_feature_labels()):
            indexes = self.kmeans.labels_ == ii
            velocity_vars = np.vars(self.data.velocity[indexes, :])

            if velocity_vars < velocity_margin_max:
                continue

            # The points are not really aligned well -> we re-cluster the sub-system

            # TODO: more general -> so far only two
            tmp_kmeans = KMeans(n_clusters=2)
            tmp_kmeans.fit(data[indexes, :])

        pass

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
            # breakpoint()
        sorted_list = np.argsort(average_sequence)[::-1]

        # Set attractor first
        # parent_id = -1
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
        self._dynamics = []

        for ii, label in enumerate(self.get_feature_labels()):
            if self._graph.nodes[label]["level"] == 0:
                # Zero level => Parent is root
                self._dynamics.append(
                    PerpendicularDeviatoinOfLinearDS(
                        attractor_position=self.data.attractor,
                        regressor=MultiOutputSVR(kernel="rbf", gamma=0.1),
                    )
                )

            else:
                ind = np.arange(self.kmeans.labels_.shape[0])[
                    self.kmeans.labels_ == label
                ]

                direction = np.mean(self.data.velocity[ind, :], axis=0)

                # if norm_dir := LA.norm(direction):
                #     direction = direction / norm_dir
                # else:
                #     # Use the K-Means dynamics as default
                #     direction = self._graph.nodes[ind_node]["direction"]

                direction = self._enforces_direction_is_towards_parent(ii, direction)

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

            self._dynamics[label].fit_from_velocities(
                self.data.position[indexes_local, :],
                self.data.velocity[indexes_local, :],
            )

    def _enforces_direction_is_towards_parent(
        self, ind_node: NodeType, direction: Vector, it_max: int = 10
    ) -> Vector:
        """Checks that the main direction point towards the intersection between
        parent and node"""
        if norm_dir := LA.norm(direction):
            direction = direction / norm_dir
        else:
            # Use the K-Means dynamics as default
            direction = self._graph.nodes[ind_node]["direction"]

        if len(preds := self.get_successors(ind_node)) != 1:
            breakpoint()

        ind_parent = preds[0]

        if self._is_pointing_towards_parent(direction, ind_node, ind_parent):
            return direction

        parent_direction = (
            self.kmeans.cluster_centers_[ind_parent, :]
            - self.kmeans.cluster_centers_[ind_node, :]
        )

        if not (par_norm := LA.norm(parent_direction)):
            # This probably does not even have to be tested, as kmeans ensures it.
            raise ValueError("Parent and node are aligned.")

        parent_direction = parent_direction / par_norm

        if np.isclose(np.dot(parent_direction, direction), -1):
            warnings.warn("Direction is far away from expected direction.")

            if self._is_pointing_towards_parent(parent_direction, ind_node, ind_parent):
                return parent_direction
            else:
                raise NotImplementedError(
                    "No intersection direction towards parent found."
                    + " Different method is needed."
                )
        mean_direction = get_directional_weighted_sum(
            direction,
            weights=np.array([0.5, 0.5]),
            directions=np.vstack((direction, parent_direction)).T,
        )

        if self._is_pointing_towards_parent(parent_direction, ind_node, ind_parent):
            # If the parent is not pointing, do another (last) check
            direction = mean_direction
        else:

            if not self._is_pointing_towards_parent(
                mean_direction, ind_node, ind_parent
            ):
                # TODO (!)
                return direction
                # raise NotImplementedError(
                #     "Improved strategy to find a parent has to be developped."
                # )
            parent_direction = mean_direction

        # TODO: to the it_max based on the angle bettween -> reach a minimum angle
        # delta_angle = 0.05
        for ii in range(it_max):
            mean_diection = get_directional_weighted_sum(
                direction,
                weights=np.array([0.5, 0.5]),
                directions=np.vstack((direction, parent_direction)).T,
            )
            if self._is_pointing_towards_parent(parent_direction, ind_node, ind_parent):
                parent_direction = mean_diection
            else:
                direction = mean_diection

        return direction

    def _is_pointing_towards_parent(
        self, direction: Vector, ii_node: int, ii_parent: int
    ) -> bool:
        """Check if a position is part of the boundary between cluster1 and cluster2."""
        surf_point_of_direction = self.region_obstacles[ii_node].get_point_on_surface(
            self.kmeans.cluster_centers_[ii_node, :] + direction, in_global_frame=True
        )

        return np.isclose(
            LA.norm(surf_point_of_direction - self.kmeans.cluster_centers_[ii_node, :]),
            LA.norm(
                surf_point_of_direction - self.kmeans.cluster_centers_[ii_parent, :]
            ),
        )

    def evaluate(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, position: Vector) -> Vector:
        # Get k-means-weights
        position = np.array(position)

        cluster_label = self.kmeans.predict(position.reshape(1, -1))[0]
        if (
            self.region_obstacles[cluster_label].get_gamma(
                position, in_global_frame=True
            )
            < 1
        ):
            return self._predict_outside_of_obstacle(position)

        weights = self._predict_sequence_weights(position, cluster_label)
        ind_rel = np.arange(self.n_clusters)[weights > 0]
        if np.sum(ind_rel) > 1:
            lyapunov_direction = self._predict_averaged_lyapunov(
                position,
                weights=weights,
                indexes_relative=ind_rel,
            )

            initial_velocity = self._predict_initial_direction(
                position,
                averaged_lyapunov=lyapunov_direction,
                weights=weights,
                indexes_relative=ind_rel,
            )

        else:
            # Ensure stability with respect to function-lyapunov
            initial_velocity = self._dynamics[cluster_label]._predict(position)
            lyapunov_direction = self._dynamics[
                cluster_label
            ].get_lyapunov_gradient_direction(position)

        # Modulate only the one which we are currently in
        rotated_velocity = obstacle_avoidance_rotational(
            position,
            initial_velocity,
            [self.region_obstacles[cluster_label]],
            convergence_velocity=lyapunov_direction,
            sticky_surface=False,
            convergence_radius=self.convergence_radius * weights[cluster_label],
        )

        if np.sum(weights) < 1:
            # TODO: allow for 'partial' weight, for e.g.,:
            # - in between two non-neighboring ellipses
            # - to transition from the outside to the inside (!)
            # => create a 'transition margin' to allow for this!
            # (make sure invariance of the region)
            raise NotImplementedError()

        return rotated_velocity

    def _predict_averaged_lyapunov(
        self,
        position: np.ndarray,
        indexes_relative: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> Vector:
        indexes_absolute = np.arange(indexes_relative.shape[0])[indexes_relative]
        lyap_dirs = np.zeros((self.dimension, indexes_absolute.shape[0]))
        for i_rel, i_abs in enumerate(indexes_absolute):
            lyap_dirs[:, i_rel] = self._dynamics[i_abs].get_lyapunov_gradient_direction(
                position
            )

        averaged_lyapunov = get_directional_weighted_sum(
            lyap_dirs[:, 0],
            weights=weights[indexes_relative],
            directions=lyap_dirs,
        )

        return averaged_lyapunov

    def _predict_initial_direction(
        self,
        position: np.ndarray,
        averaged_lyapunov: np.ndarray,
        indexes_relative: Optional[np.ndarray] = None,
        weights: Optional[np.ndarray] = None,
    ) -> Vector:
        """Returns initial velocity and the convergence velocity"""
        indexes_absolute = np.arange(indexes_relative.shape[0])[indexes_relative]

        # Start root below index (otherwise naming conflict may occur)
        root_id = -1
        temp_tree = VectorRotationTree(
            root_id=root_id, root_direction=averaged_lyapunov
        )
        for i_abs in indexes_absolute:
            # TODO: there is an issue if the 'linear attractor'
            temp_tree.add_node(
                parent_id=root_id,
                node_id=i_abs,
                direction=lyap_dirs[:, i_rel],
            )
            temp_tree.add_node(
                parent_id=i_abs,
                node_id=i_abs + self.n_clusters,
                direction=self._dynamics[i_abs].evaluate_without_lyapunov_check(
                    position
                ),
            )
        averaged_direction = temp_tree.get_weighted_mean(
            node_list=indexes_absolute + self.n_clusters,
            weights=weights[indexes_absolute],
        )
        return averaged_direction

    def _predict_outside_of_obstacle(
        self,
        position: Vector,
        cut_off_ratio: float = 2.0,
        power_factor: float = 1,
    ) -> Vector:
        """Returns the velocity vector if we are outside of all position.

        Outside velocity has to be designed such that it does NOT cancel, i.e., when the closest
        cluster is at the back, the transition to it has to be minimized -> adapt the weighting
        function accordingly.

        Arguments
        ---------
        cut_off_ratio (> 1): The distance at which the radius does not have any effect anymore
        """

        # Find the two clusters which are the closest
        center_dists = self.get_distance_from_centers(position)
        # arg_sort = np.argsort(center_dists)[-1:-3:-1]  # Get last two elements
        arg_sort = np.argsort(center_dists)[:2]  # Get last two elements
        close_dists = center_dists[arg_sort]

        max_radius = self.region_radius_ * cut_off_ratio
        if close_dists[0] > max_radius:
            # Far away
            return self.outside_dynamics.evaluate(position)

        surface_distance = max_radius - close_dists[0]
        if close_dists[1] > max_radius:
            relative_dist = surface_distance
        else:
            relative_dist = close_dists[1] - close_dists[0]

        max_dist = max_radius - self.region_radius_

        if surface_distance > max_dist:
            # TODO: only for sanity/temp-debugging [remove if never called]
            breakpoint()

        weight_surf = relative_dist / max_dist
        # print(f"1: {weight_surf=}")
        # print(f"power: {(surface_distance / max_dist * power_factor)}")
        weight_surf = weight_surf ** (max_dist / surface_distance * power_factor)
        # print(f"2: {weight_surf=}")

        # Predict surface velocity
        index = arg_sort[0]
        # Modulate only the one which we are currently in
        outside_velocity = self.outside_dynamics.evaluate(position)

        if weight_surf <= 0:
            # Return velocity directly
            return outside_velocity

        center_dir = surface_velocity = (
            self.region_obstacles[arg_sort[0]].center_position - position
        )
        center_dir = center_dir / LA.norm(center_dir)

        dot_prod = np.dot(center_dir, outside_velocity) / LA.norm(outside_velocity)
        pow_weight = dot_prod - math.cos(self.convergence_radius)
        if pow_weight <= 0:
            # Pointing away from the closest center -> turn around
            return outside_velocity
        weight_surf = weight_surf ** (
            (1 - math.cos(self.convergence_radius)) / pow_weight
        )

        if math.isclose(self.convergence_radius, math.pi):
            # Since all velocities are towards the center on the surface of a cluster
            surface_velocity = center_dir

        else:
            projected_position = self.region_obstacles[
                arg_sort[0]
            ].get_point_on_surface(position, in_global_frame=True)

            surface_velocity = obstacle_avoidance_rotational(
                position,
                self._dynamics[index].evaluate(projected_position),
                [self.region_obstacles[index]],
                convergence_velocity=self._dynamics[
                    index
                ].evaluate_convergence_velocity(projected_position),
                sticky_surface=False,
            )

        mean_direction = get_directional_weighted_sum(
            outside_velocity,
            weights=np.array([weight_surf, (1 - weight_surf)]),
            directions=np.vstack((surface_velocity, outside_velocity)).T,
        )

        # breakpoint()
        return mean_direction

    def _predict_directional_sum(self, position: Vector, weights, velocities) -> Vector:
        # TODO:
        # -> sorted hierarchy
        # -> additionaly add modulated velocities / directions
        # -> return weighted evaluation
        pass
        # tmp_directions = VectorRotationTree()

    def _predict_lyaponuv_gradient_direction(
        self, position: Vector, sequence_weights: Optional[np.ndarray] = None
    ) -> Vector:
        if sequence_weights is None:
            sequence_weights = self._predict_sequence_weights(position)

        ind_nonzero = sequence_weights > 0

        directions = np.zeros((self.dimension, np.sum(ind_nonzero)))
        for ii_rel, ii_abs in enumerate(np.arange(ind_nonzero.shape[0])[ind_nonzero]):
            directions[:, ii_rel] = self._dynamics[
                ii_abs
            ].get_lyapunov_gradient_direction(position)

        if directions[1].shape[0] == 1:
            return directions[:, 0]

        mean_direction = get_directional_weighted_sum(
            directions[np.argmax(sequence_weights[ind_nonzero])],
            weights=sequence_weights[ind_nonzero],
            directions=directions,
        )

        return mean_direction

    def _predict_sequence_weights(
        self,
        position: Vector,
        cluster_label: int = None,
        parent_factor: float = 0.5,
        gamma_cutoff: float = 2.0,
    ) -> np.ndarray:
        """Returns the weights whith which each of the superior clusters is considered

        Arguments
        ---------
        parent_factor in ]0, 1[: determines far into the new obstacle one can enter.
        gamma_cutoff: ensure local convergences through impenetrability of walls

        Returns
        -------
        weights: A (np.ndarray of floats) vector containing the weights of importance of all
            kmeans clusters
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

            # TODO: test that it is reaally parent vs. child index ?!
            gamma = self.region_obstacles[cluster_label].get_gamma(
                position, in_global_frame=True, ind_transparent=parent
            )

            tmp_weight *= 1 - 1 / (1 + max(gamma - gamma_cutoff, 0))

        weights = np.zeros((self.n_clusters))
        weights[parent] = tmp_weight
        weights[cluster_label] = 1 - np.sum(weights)
        return weights

    def plot_kmeans(self, *args, **kwargs) -> None:
        return helper_functions.plot_kmeans(self, *args, **kwargs)

    def plot_boundaries(self, *args, **kwargs) -> None:
        # TODO: this does not need to be here..
        return helper_functions.plot_boundaries(self, *args, **kwargs)


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
