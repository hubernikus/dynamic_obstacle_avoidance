#!/USSR/bin/python3
""" Sample the space and decide if points are collision-full or free. """
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

# from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from dynamic_obstacle_avoidance.rotational.graph_handler import GraphHandler
from dynamic_obstacle_avoidance.rotational.rotational_avoider import RotationalAvoider

Vector = npt.ArrayLike


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

        self.reference_points = None

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

    def fit(self, datapoints: Vector) -> None:
        self._gmm = GaussianMixture(
            n_components=self.n_gmms, covariance_type="full"
        ).fit(datapoints.T)

    def evaluate_hirarchy_and_reference_points(self) -> None:
        """This hirarchy checker is very simple and does not further investigate
        if objects are within a loop."""
        self.gmm_index_graph = GraphHandler()
        self.reference_points = np.zeros(self._gmm.means_.shape).T

        mean_center = np.mean(self._gmm.means_, axis=0)

        ind_closest = np.argmin(
            LA.norm(self._gmm.means_ - np.tile(mean_center, (self.n_gmms, 1)), axis=0)
        )

        self.reference_points[:, ind_closest] = self._gmm.means_[ind_closest, :]
        self.gmm_index_graph.set_root(ind_closest)
        assigned_list = [ind_closest]

        # Generate a consistent 'top-down'
        remaining_list = np.arange(self.n_gmms).tolist()
        del remaining_list[ind_closest]
        proba_matrix = self._gmm.predict_proba(self._gmm.means_[remaining_list, :])

        # Remove the 'center'-elements
        # for ii in range(proba_matrix.shape[0]):
        # proba_matrix[ii, remaining_list[ii]] = 0

        # Top down adding of new values to the graph
        while len(remaining_list):
            temp_matrix = proba_matrix[:, assigned_list]
            ind = np.unravel_index(np.argmax(temp_matrix, axis=None), temp_matrix.shape)

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
            self.reference_points[:, value] = self.get_intersection_of_ellipses(
                indices=[value, parent_value]
            )

    #     for ii in range(self.n_gmms):
    #         if ii == ind_closest:
    #             continue

    #         # Chose second highest prediction value at the center of a Gaussian
    #         # as the parent-index, since highest one will be the Gaussian itself
    #         # the `predict_proba` function does not apply the weights (yet)
    #         proba_vals = self._gmm.predict_proba(self._gmm.means_[ii, :].reshape(1, -1))
    #         ind_parent = np.argsort(proba_vals[0])[1]

    #         self.gmm_index_graph.add_element_with_parent(value=ii, parent_value=ind_parent)
    #         self.reference_points[:, ii] = self.get_intersection_of_ellipses(
    #             indices=[ii, ind_parent]
    #         )

    #     # Sanity check -> did it work at least partially?
    #     # TODO: in the future => graph creation under constraints
    #     if len(self.gmm_index_graph.roots) != 1:
    #         breakpoint()
    #         raise NotImplementedError(
    #             "The graph has multiple (or no) roots - Behavior is undefined."
    #         )

    def avoid(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        self.evaluate_gamma_weights(position)

        descending_index = self.get_nodes_hirarchy_descending()
        for ii in descending_index:
            parent = self.get_parent(descending_index)

            if parent is None:
                # Is a root -> get normal modulation
                pass

    def _get_projected_tangent(self, position: np.ndarray, index):
        pass

    def evaluate_gamma_weights(
        self, position: np.ndarray, gamma_min_factor: float = 0.9
    ) -> np.array:
        """Get the importance weight of the different sub-obstacles."""
        self.gamma_list = np.zeros(self.n_gmms)
        for index in range(self.n_gmms):
            self.gamma_list[index] = self.get_gamma_proportional(index)

        self.relative_weights = np.maximum(
            0, (self.gamma_list - gamma_min_factor * np.min(self.gamma_list))
        )
        self.relative_weights = self.relative_weights / np.sum(self.relative_weights)

        self.gamma_weights = 1.0 / self.gamma_list

    def transform_to_analytic_ellipses(self):
        """Returns ObstacleContainer with n_gmm Ellipse obstacles with
        pose-axes description."""
        obstacle_environment = ObstacleContainer()

        for ii in range(len(self._gmm.covariances_)):
            # Renamed: n => ii (in case of error) [remove this comment in the future]
            if self._gmm.covariance_type == "full":
                covariances = self._gmm.covariances_[ii][:2, :2]
            elif self._gmm.covariance_type == "tied":
                covariances = self._gmm.covariances_[:2, :2]
            elif self._gmm.covariance_type == "diag":
                covariances = np.diag(self._gmm.covariances_[ii][:2])
            elif self._gmm.covariance_type == "spherical":
                covariances = (
                    np.eye(self._gmm.means_.shape[1]) * self._gmm.covariances_[ii]
                )

            eig_vals, eig_vecs = LA.eigh(covariances)

            uu = eig_vecs[0] / LA.norm(eig_vecs[0])
            angle = np.arctan2(uu[1], uu[0])
            # angle_degrees = 180 * angle / np.pi  # convert to degrees
            # eig_vals = 2.0 * np.sqrt(2.0) * np.sqrt(eig_vals)
            eig_vals = 2.0 * np.sqrt(eig_vals) * self.variance_factor

            obstacle_environment.append(
                Ellipse(
                    center_position=self._gmm.means_[ii, :2],
                    orientation=angle,
                    axes_length=np.array([eig_vals[0], eig_vals[1]]),
                )
            )
            # obstacle_environment[-1].set_reference_point(
            #     self.reference_points[ii, :], in_global_frame=True)

        return obstacle_environment

    def plot_obstacle(
        self,
        ax=None,
    ):
        obstacles = self.transform_to_analytic_ellipses()

        if ax is None:
            fig, ax = plt.subplots()

        plot_obstacles(
            obstacle_container=obstacles,
            ax=ax,
        )
        # draw_reference=True

        if self.reference_points is not None:
            ax.plot(self.reference_points[0, :], self.reference_points[1, :], "k+")

            for ii in range(self.n_gmms):
                ind_parent = self.gmm_index_graph.get_parent(ii)
                if ind_parent is None:
                    continue

                ax.plot(
                    self.reference_points[0, [ii, ind_parent]],
                    self.reference_points[1, [ii, ind_parent]],
                    "k--",
                )

    def _get_gauss_derivative(self, position, index, powerfactor=1):
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

    def get_gamma_proportional(self, position, index):
        delta_dist = position - self._gmm.means_[index, :]
        gamma = delta_dist.T @ self._gmm.precisions_cholesky_[index, :, :] @ delta_dist
        return np.sqrt(gamma) / self.variance_factor

    def get_gamma_derivative(self, position, index, powerfactor: float = 1):
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

    def get_gamma(self, position, index):
        gamma_prop = self.get_gamma_proportional(position, index)
        if not gamma_prop:
            return 0
        return (
            LA.norm(position - self._gmm.means_[index, :]) * (1 - 1.0 / gamma_prop) + 1
        )

    def get_normal_direction(self, position, index):
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
        indices,
        powerfactor: float = 5,
        it_max: int = 100,
        rtol: float = 1e-1,
        step_size: float = 0.05,
    ) -> np.ndarray:
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
