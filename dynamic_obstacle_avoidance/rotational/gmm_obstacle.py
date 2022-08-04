""" (Single) Obstacle created from multiple ellipses (as a GMM description). """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-06-08

# Use python 3.10 [annotations / typematching]
from __future__ import annotations  # Not needed from python 3.10 onwards

import logging

import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt

# TODO: use networkx instead of the built-in library
# import networkx as nx

# from vartools.states import ObjectPose
from vartools.math import get_intersection_with_circle
from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import UnitDirection
from vartools.directional_space import get_directional_weighted_sum

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from dynamic_obstacle_avoidance.rotational.graph_handler import GraphHandler
from dynamic_obstacle_avoidance.rotational.rotational_avoider import RotationalAvoider

from dynamic_obstacle_avoidance.rotational.datatypes import Vector


class GmmObstacle:
    """Obstacle which learns to create multiple ellipses.

    Everything is with respect to the global frame (since the
    data originates from the global frame).

    This obstacle is learned from datapoints.

    The ellipses are defined such that the eigenvalues of the gmm-gaussian corresponding
    to the"""

    def __init__(self, n_gmms: int, variance_factor: float = 2.0) -> None:
        self.n_gmms = n_gmms
        self.gamma_list = None
        self._gmm = None
        self.variance_factor = variance_factor

        self._reference_points = None
        self.gmm_index_graph = None

        self._axes_lengths = None
        self._axes_directions = None

    @staticmethod
    def from_container(cls, environment: ObstacleContainer) -> GmmObstacle:
        n_gmms = len(environment)
        dimension = environment[0].dimension

        new_instance = cls(n_gmms=n_gmms)

        new_instance._gmm = GaussianMixture(
            n_components=new_instance.n_gmms, covariance_type="full"
        )

        # Weights are assigned equally
        new_instance._gmm.weights_ = 1.0 / n_gmms * np.ones(n_gmms)
        new_instance._gmm.means_ = np.zeros((n_gmms, dimension))
        new_instance._gmm.covarinaces_ = np.zeros((n_gmms, dimension, dimension))
        new_instance._gmm.precisions_cholesky_ = np.zeros(
            (n_gmms, dimension, dimension)
        )

        for ii, ellipse in enumerate(environment):
            if not isinstance(ellipse, Ellipse):
                logging.warning(
                    "Obstacle is not of type 'Ellipse', it will be ignored."
                )
                continue
            new_instance._gmm.means_[ii, :] = ellipse.center_position
            # new_instance._gmm.covariances_[ii, :, :] =
            # new_instance._gmm.precisions_cholesky_ = LA.pinv(
            #     new_instance._gmm.covariances_[ii, :, :])
            raise NotImplementedError("Finish implementation.")

        return new_instance

    @property
    def dimension(self) -> int:
        try:
            return self._gmm.means_.shape[1]
        except AttributeError:
            logging.warning(
                "Object-Shape has not been defined yet, returns 0 - dimension."
            )
            return 0

    @property
    def kernel_points(self) -> np.ndarray:
        return self._reference_points

    @kernel_points.setter
    def kernel_points(self, value: npt.ArrayLike) -> None:
        self._reference_points = value

    def get_kernel_point(self, index: int) -> Vector:
        return self._reference_points[:, index]

    def set_kernel_point(self, value: Vector, index: int) -> None:
        self._reference_points[:, index]

    def get_center_position(self, index: int) -> Vector:
        return self._gmm.means_[index, :]

    def fit(self, datapoints: Vector) -> None:
        self._gmm = GaussianMixture(
            n_components=self.n_gmms, covariance_type="full"
        ).fit(datapoints.T)

    def evaluate_hirarchy_and_reference_points(self) -> None:
        """This hirarchy checker is very simple and does not further investigate
        if objects are within a loop."""
        self.gmm_index_graph = GraphHandler()
        self._reference_points = np.zeros(self._gmm.means_.shape).T

        mean_center = np.mean(self._gmm.means_, axis=0)

        ind_closest = np.argmin(
            LA.norm(self._gmm.means_ - np.tile(mean_center, (self.n_gmms, 1)), axis=0)
        )

        self._reference_points[:, ind_closest] = self._gmm.means_[ind_closest, :]
        self.gmm_index_graph.set_root(ind_closest)
        assigned_list = [ind_closest]

        # Generate a consistent 'top-down'
        remaining_list = np.arange(self.n_gmms).tolist()
        del remaining_list[ind_closest]

        if not remaining_list:
            # Only has one element
            return
        proba_matrix = self._gmm.predict_proba(self._gmm.means_[remaining_list, :])

        # Top down adding of new values to the graph
        while len(remaining_list):
            reduced_propa = proba_matrix[:, assigned_list]
            ind = np.unravel_index(
                np.argmax(reduced_propa, axis=None), reduced_propa.shape
            )

            value = remaining_list[ind[0]]
            parent_value = assigned_list[ind[1]]
            self.gmm_index_graph.add_element_with_parent(
                value=value,
                parent_value=parent_value,
            )

            assigned_list.append(value)
            del remaining_list[ind[0]]
            proba_matrix = np.delete(proba_matrix, (ind[0]), axis=0)

            # Update reference point
            self._reference_points[:, value] = self.get_intersection_of_ellipses(
                indices=[value, parent_value]
            )

            # By default also update axes length and direction
            self.evaluate_axes_length_and_direction()

    #     for ii in range(self.n_gmms):
    #         if ii == ind_closest:
    #             continue

    #         # Chose second highest prediction value at the center of a Gaussian
    #         # as the parent-index, since highest one will be the Gaussian itself
    #         # the `predict_proba` function does not apply the weights (yet)
    #         proba_vals = self._gmm.predict_proba(self._gmm.means_[ii, :].reshape(1, -1))
    #         ind_parent = np.argsort(proba_vals[0])[1]

    #         self.gmm_index_graph.add_element_with_parent(value=ii, parent_value=ind_parent)
    #         self._reference_points[:, ii] = self.get_intersection_of_ellipses(
    #             indices=[ii, ind_parent]
    #         )

    def avoid(self, position: Vector, velocity: Vector) -> Vector:
        self.evaluate_gamma_weights(position)
        projected_tangents = np.zeros(())

        descending_index = self.get_nodes_hirarchy_descending()
        for index in descending_index:
            # Get reference-basis
            normal_vector = self.get_normal_direction(position, index)
            modulation_basis = get_orthogonal_basis(normal_vector)
            modulation_basis[:, 0] = self.get_reference_direction(position, index)

            parent = self.get_parent(descending_index)
            if parent is None:
                # Is a root -> get normal modulation
                ref_frame_velocity = LA.pinv(modulation_basis) @ velocity
                projected_tangents[:, index] = (
                    modulation_basis[:, 1:] @ ref_frame_velocity[1:]
                )

    def get_root_rotation_direction(self, position: Vector, velocity: Vector) -> Vector:
        ind_root = self.gmm_index_graph.get_root_indices()[0]
        kernel_direction = position - self.get_kernel_point(ind_root)
        kernal_norm = LA.norm(kernel_direction)
        if not kernal_norm:
            logging.warning("Evaluation at obstacle-center. Something has gone wrong.")
            return velocity

        kernel_direction = kernel_direction / kernal_norm
        if np.dot(kernel_direction, velocity) < 1:
            # Velocity is already pointing away - the main kernel does need to guide
            return velocity

        # Project to be orthogonal to the 'kernel_direction'
        kernel_basis = get_orthogonal_basis(kernel_direction)
        trafo_vector = kernel_basis.T @ velocity
        trafo_vector[0] = 0
        return kernel_basis @ trafo_vector

    def get_relative_orientation_of_ellipses(self, position):
        gmm_directions = []

        for ii in range(self.n_gmms):
            ind_parent = self.gmm_index_graph.get_parent(ii)
            if ind_parent is None:
                gmm_directions.append(None)

            delta_dir_parent = self.get_kernel_position(ii) - self.get_center_position(
                ind_parent
            )
            delta_dir = self.get_center_position(ii) - self.get_kernel_point(ii)

            d_parent_norm = LA.norm(delta_dir_parent)
            d_dir_norm = LA.norm(delta_dir)

            if not (d_parent_norm and d_dir_norm):
                # Directly overlapping parent and child
                if d_dir_norm:
                    delta_dir = delta_dir / d_dir_norm
                    delta_dir_parent = delta_dir

                elif d_parent_norm:
                    delta_dir_parent = delta_dir_parent / d_parent_norm
                    delta_dir = delta_dir_parent

                else:
                    gmm_directions.append(
                        UnitDirection(
                            np.ones(self.dimension) / self.dimension
                        ).from_angle(np.zeros(self.dimension - 1))
                    )

                    continue
            else:
                delta_dir = delta_dir / d_dir_norm
                delta_dir_parent = delta_dir_parent / d_parent_norm

            gmm_directions.append(
                UnitDirection(delta_dir_parent).from_vector(delta_dir)
            )

        return gmm_directions

    def evalute_weighted_reference_and_normal_offset(self, position: Vector):
        """Assumption of all children-nodes being at an total angle of
        (root -> node) < pi

        Summed normal with respect to reference similar to
        # 'Fast Obstacle Avoidance Based on Real-Time Sensing' .
        """
        delta_normal = np.zeros(self.dimension)
        reference_directions = np.zeros((self.dimension, self.n_gmms))

        # Loop over all non-zeros-weights
        for index in np.arange(self.n_gmms)[self.relative_weights.astype(bool)]:
            reference_directions[:, index] = self.get_reference_direction(
                position, index
            )

            delta_normal += (
                self.get_normal_direction(position, index)
                - reference_directions[:, index]
            )

        # Get root
        ind_root = self.gmm_index_graph.get_root_indices()[0]
        if self.relative_weights[ind_root]:
            base_vector = reference_directions[:, ind_root]
        else:
            base_vector = self.get_reference_direction(position, ind_root)

        # Mean reference is normalized, hence this does not need to be checked in
        # further calculations
        self.mean_reference = get_directional_weighted_sum(
            null_direction=base_vector,
            weights=self.relative_weights,
            directions=reference_directions,
        )

        if not LA.norm(delta_normal):
            # Trivial case
            self.mean_normal = self.mean_reference

        dot_prod = (-1) * (
            np.dot(delta_normal, self.mean_reference) / LA.norm(delta_normal)
        )

        if dot_prod < np.sqrt(2) / 2:
            normal_scaling = 1
        else:
            normal_scaling = np.sqrt(2) * dot_prod

        self.mean_normal = normal_scaling * self.mean_reference + delta_normal
        self.mean_normal /= LA.norm(self.mean_normal)

    def evaluate_gamma_weights(
        self,
        position: Vector,
        gamma_min_factor: float = 0.9,
        gamma_min_summand: float = 0.1,
    ) -> np.ndarray:
        """Get the importance weight of the different sub-obstacles.

        Arguments
        ---------
        gamma_relative_cutoff: float in (0, 1) - defines at which value the relative cut-off
        is applied, such that within obstacles are not too far
        """
        self.gamma_list = np.zeros(self.n_gmms)
        self.relative_weights = np.zeros(self.n_gmms)

        for index in range(self.n_gmms):
            self.gamma_list[index] = self.get_gamma_proportional(position, index)

        self.relative_weights = self.gamma_list - 1

        ind_zeros = self.relative_weights <= 0
        if any(ind_zeros):
            self.relative_weights = np.zeros(self.relative_weights.shape)
            self.relative_weights[ind_zeros] = 1.0 / np.sum(ind_zeros)
            return

        # The weights are the inverse of the gammas
        self.relative_weights = 1 / self.relative_weights
        self.relative_weights *= self.get_obstacle_occlusion_factor(position)

        sum_weights = np.sum(self.relative_weights)
        if sum_weights:
            self.relative_weights = self.relative_weights / sum_weights

        # self.gamma_weights = 1.0 / self.gamma_list

    def get_obstacle_occlusion_factor(
        self, position: Vector, gamma_relative_cutoff: float = 0.7
    ) -> np.array:
        """Check if the projected-surface-point is within the other obstacles
        for now this evaluation is only done for direct parents and children
        Adapt weight if the point lies behind the intersection with parent or child"""
        factors = np.ones(self.n_gmms)

        for index in range(self.n_gmms):
            # surface_point_gammas = np.ones((self.n_gmms))

            surface_point = self.project_point_on_surface_with_offcenter_point(
                position, self._reference_points[:, index], index
            )
            ind_parent = self.gmm_index_graph.get_parent(index)

            # breakpoint()
            if ind_parent is None:
                continue

            surface_point_gamma = self.get_gamma_proportional(surface_point, ind_parent)

            if surface_point_gamma >= 1:
                continue

            position_rel = position - self._reference_points[:, index]
            center_rel = (
                self._gmm.means_[ind_parent, :] - self._reference_points[:, index]
            )
            dot_prod = np.dot(position_rel, center_rel) / (
                LA.norm(position_rel) * LA.norm(center_rel)
            )

            if dot_prod <= 0:
                continue

            elif dot_prod >= 1:
                factors[index] = 0
                continue

            factors[index] = surface_point_gamma ** (1 / (1 - dot_prod))

        return factors

    def evaluate_axes_length_and_direction(self) -> None:
        self._axes_lengths = np.zeros((self.dimension, self.n_gmms))
        self._axes_directions = np.zeros((self.dimension, self.dimension, self.n_gmms))

        if not self._gmm.covariance_type == "full":
            raise NotImplementedError("Assumption of 'full' covariances.")

        for ii in range(self.n_gmms):
            eig_vals, self._axes_directions[:, :, ii] = LA.eigh(
                self._gmm.covariances_[ii, :, :]
            )
            self._axes_lengths[:, ii] = 2.0 * np.sqrt(eig_vals) * self.variance_factor

    def transform_to_analytic_ellipses(self) -> ObstacleContainer:
        """Returns ObstacleContainer with n_gmm Ellipse obstacles with
        pose-axes description."""
        if self.dimension != 2:
            raise NotImplementedError("Not implemented for dimension > 2.")

        if self._axes_lengths is None:
            self.evaluate_axes_length_and_direction()

        obstacle_environment = ObstacleContainer()
        for ii in range(self.n_gmms):
            uu = self._axes_directions[:, :, ii][0] / LA.norm(
                self._axes_directions[:, :, ii][0]
            )
            angle = np.arctan2(uu[1], uu[0])

            obstacle_environment.append(
                Ellipse(
                    center_position=self._gmm.means_[ii, :2],
                    orientation=angle,
                    axes_length=self._axes_lengths[:, ii],
                )
            )

        return obstacle_environment

    def plot_obstacle(
        self,
        ax=None,
        alpha_obstacle: float = 0.8,
    ) -> None:
        obstacles = self.transform_to_analytic_ellipses()

        if ax is None:
            fig, ax = plt.subplots()

        plot_obstacles(
            obstacle_container=obstacles,
            ax=ax,
            alpha_obstacle=alpha_obstacle,
        )
        # draw_reference=True

        if self._reference_points is not None:
            ax.plot(self._reference_points[0, :], self._reference_points[1, :], "k+")

            for ii in range(self.n_gmms):
                ind_parent = self.gmm_index_graph.get_parent(ii)
                if ind_parent is None:
                    continue

                ax.plot(
                    self._reference_points[0, [ii, ind_parent]],
                    self._reference_points[1, [ii, ind_parent]],
                    "k--",
                )

    def _get_gauss_derivative(
        self, position: Vector, index: int, powerfactor: float = 1
    ):
        """The additional powerfactor allows"""
        fraction_value = 1 / np.sqrt(
            (2 * np.pi) ** self.dimension * LA.det(self._gmm.covariances_[index, :, :])
        )
        delta_dist = position - self._gmm.means_[index, :, :]
        exp_value = np.exp(
            (-0.5)
            * delta_dist.T
            @ self._gmm.precisions_cholesky_[index, :, :]
            @ delta_dist
            * powerfactor
        )
        deriv_factor = (-0.5) * self._precisions_cholesky_[index, :, :] @ delta_dist
        return (powerfactor * deriv_factor) * fraction_value * exp_value

    def get_gamma_proportional(self, position: Vector, index: int) -> float:
        delta_dist = position - self._gmm.means_[index, :]
        gamma = delta_dist.T @ self._gmm.precisions_cholesky_[index, :, :] @ delta_dist
        return np.sqrt(gamma) / self.variance_factor

    def get_gamma_derivative(
        self, position: Vector, index: int, powerfactor: float = 1
    ) -> float:
        """Returns the derivative of the proportional gamma."""
        delta_dist = position - self._gmm.means_[index, :]

        d_gamma = (
            powerfactor
            / 2.0
            * (delta_dist.T @ self._gmm.precisions_cholesky_[index, :, :] @ delta_dist)
            ** (powerfactor / 2.0 - 1)
            * self.variance_factor ** (-1 * powerfactor)
            * self._gmm.precisions_cholesky_[index, :, :]
            @ delta_dist
        )

        return d_gamma

    def get_gamma(self, position: Vector, index: int) -> int:
        gamma_prop = self.get_gamma_proportional(position, index)
        if not gamma_prop:
            return 0
        return (
            LA.norm(position - self._gmm.means_[index, :]) * (1 - 1.0 / gamma_prop) + 1
        )

    def project_point_on_surface(self, position: Vector, index: int) -> Vector:
        gamma = self.get_gamma_proportional(position, index)
        return (position - self._gmm.means_[index, :]) / gamma + self._gmm.means_[
            index, :
        ]

    def transform_position_to_unitcircle_frame(
        self, position: Vector, index: int
    ) -> Vector:
        """Returns position-vector in the frame of the corresponding unit-(diamaeter)-circle
        to the ellipse at index.

        Inverse function of 'transform_position_from_unitcircle_frame'
        """
        return (
            self._axes_directions[:, :, index].T
            @ (position - self._gmm.means_[index, :])
        ) / self._axes_lengths[:, index]

    def transform_position_from_unitcircle_frame(
        self, position: Vector, index: int
    ) -> Vector:
        """Returns position-vector in the original frame of
        the corresponding unit-(diamaeter)-circle to the ellipse at index.

        Inverse function of 'transform_position_to_unitcircle_frame'
        """
        return (
            self._axes_directions[:, :, index]
            @ (position * self._axes_lengths[:, index])
            + self._gmm.means_[index, :]
        )

    def project_point_on_surface_with_offcenter_point(
        self, position: Vector, offcenter_point: Vector, index: int
    ) -> Vector:
        rel_pos = self.transform_position_to_unitcircle_frame(position, index)
        rel_offset = self.transform_position_to_unitcircle_frame(offcenter_point, index)
        intersection = get_intersection_with_circle(
            rel_offset,
            direction=(rel_pos - rel_offset),
            radius=0.5,
        )
        return self.transform_position_from_unitcircle_frame(intersection, index)

    def get_reference_direction(self, position: Vector, index: int) -> Vector:
        """Reference direction."""
        ref_dir = position - self._gmm.means_[index, :]
        ref_norm = LA.norm(ref_dir)
        if ref_norm:
            ref_dir /= ref_norm
        else:
            ref_dir[0] = 1
        return ref_dir

    def get_normal_direction(self, position: Vector, index: int) -> Vector:
        """Get normal direction of obstacle =>"""
        delta_dist = position - self._gmm.means_[index, :]

        d_gamma = (
            (delta_dist.T @ self._gmm.precisions_cholesky_[index, :, :] @ delta_dist)
            ** (1.0 / 2.0 - 1)
            * self.variance_factor ** (-1)
            * self._gmm.precisions_cholesky_[index, :, :]
            @ delta_dist
        )

        dgamma_norm = LA.norm(d_gamma)
        if dgamma_norm:
            d_gamma = d_gamma / dgamma_norm
        else:
            # Feasible default value
            d_gamma[0] = 1

        return d_gamma

    def get_intersection_of_ellipses(
        self,
        indices: list,
        powerfactor: float = 5,
        it_max: int = 100,
        rtol: float = 1e-1,
        step_size: float = 0.05,
    ) -> Vector:
        """Returns the intersection of ellipses using gradient descent of the GMM.

        Arguments
        rtol: Relative tolerance for convergence.
        """
        # TODO: this convergence stepping could be improved by
        # considering the covariances
        # abs_tol = rtol**powerfactor * LA.norm(
        #     self._gmm.means_[indices[0]] - self._gmm.means_[indices[1]]
        # )
        abs_tol = rtol * LA.norm(
            self._gmm.means_[indices[0]] - self._gmm.means_[indices[1]]
        )

        if not abs_tol:  # Zero value
            logging.info("Almost identical center for two Gaussians detected.")
            return self._gmm.means_[indices[0]]

        # Starting point is the center of the shortest connection
        pos_intersect = np.mean(self._gmm.means_[indices], axis=0)

        for ii in range(it_max):
            gradient_step = np.zeros(self.dimension)
            for index in indices:
                gradient_step += self.get_gamma_derivative(
                    pos_intersect, index, powerfactor
                )

            # TODO: maybe reduce power-factor to decrease the step size slightly
            step_norm = LA.norm(gradient_step)
            gradient_step = gradient_step / step_norm
            step_norm = step_norm ** (1.0 / powerfactor)

            if step_norm < abs_tol:
                logging.info(f"Convergence at iteration {ii}")
                break

            pos_intersect = pos_intersect - (gradient_step * step_norm) * step_size

        return pos_intersect
