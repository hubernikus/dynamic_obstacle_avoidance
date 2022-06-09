#!/USSR/bin/python3
""" Sample the space and decide if points are collision-full or free. """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-06-08

import logging

import numpy as np
from numpy import linalg as LA

from sklearn.mixture import GaussianMixture

# from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.rotational.graph_handler import GraphElement
from dynamic_obstacle_avoidance.rotational.graph_handler import GraphHandler


class GmmObstacle():
    """ Obstacle which learns to create multiple ellipses.

    Everything is with respect to the global frame (since the
    data originates from the global frame).

    This obstacle is learned from datapoints.

    The ellipses are defined such that the eigenvalues of the gmm-gaussian corresponding
    to the """
    def __init__(self, n_gmms) -> None:
        self.n_gmms = n_gmms
        self.gamma_list = None

    @property
    def dimension(self) -> int:
        try:
            return self._gmm.means_.shape[0]
        except AttributeError:
            logging.warning("Object-Shape has not been defined yet, returns 0 - dimension.")
            return 0
        
    def fit(self, datapoints) -> None:
        self._gmm = GaussianMixture(
            n_components=self.n_gmms, covariance_type="full"
        ).fit(datapoints.T)

    def evaluate_gammas(self, position):
        self.gamma_list = self._gmm.predict_proba(position)
        breakpoint()

    def get_one_level_hirarchy(self):
        """ This hirarchy checker is very simple and does not further invetigate
        if objects are within a loop. """
        mean_center = np.mean(self._gmm.means_, axis=0)

        ind_closest = np.argmin(
            LA.norm(
                self._gmm.means_ - np.tile(mean_center, (self.dimension, 1)), axis=0
            )
        )

        self.gmm_graph = GraphHandler()   #
        self.gmm_graph.root = ind_closest
        # Root does not need an update of the reference point

        self.reference_points = np.zeros(self._gmm.means_.shape)
        
        for ii in range(self.n_gmms):
            if ii == ind_closest:
                continue

            # Chose second highest prediction value at the center of a Gaussian
            # as the parent-index, since highest one will be the Gaussian itself
            # the `predict_proba` function does not apply the weights (yet)
            proba_vals = self._gmm.predict_proba(self._gmm.means_[ii, :].reshape(1, -1))
            ind_parent = np.argsort(proba_vals[0])[1]

            self.gmm_graph.add_element_with_parent(
                child=ii, parent=ind_parent
            )
            self.reference_points[ii, :] = self.get_intersection_of_ellipses(
                indeces=[ii, ind_parent]
            )

        # Sanity check -> did it work at least partially?
        # TODO: in the future => graph creation under constraints
        if not self.gmm_graph.root.children:
            breakpoint()
            raise Exception("The current graph does not have any children.")
            
    def modulate(self, velocity):
        pass

    def transform_to_analytic_ellipses(self, axes_factor=2.0):
        """ Returns ObstacleContainer with n_gmm Ellipse obstacles
        with pose-axes description."""
        obstacle_environment = ObstacleContainer()

        for n in range(len(self._gmm.covariances_)):
            if self._gmm.covariance_type == "full":
                covariances = self._gmm.covariances_[n][:2, :2]
            elif self._gmm.covariance_type == "tied":
                covariances = self._gmm.covariances_[:2, :2]
            elif self._gmm.covariance_type == "diag":
                covariances = np.diag(self._gmm.covariances_[n][:2])
            elif self._gmm.covariance_type == "spherical":
                covariances = np.eye(self._gmm.means_.shape[1]) * self._gmm.covariances_[n]

            eig_vals, eig_vecs = LA.eigh(covariances)

            uu = eig_vecs[0] / LA.norm(eig_vecs[0])
            angle = np.arctan2(uu[1], uu[0])
            angle = 180 * angle / np.pi  # convert to degrees
            eig_vals = 2.0 * np.sqrt(2.0) * np.sqrt(eig_vals)

            obstacle_environment.append(
                Ellipse(
                    center_position=self._gmm.means_[n, :2],
                    orientation=angle,
                    axes_length=axes_factor*np.array([eig_vals[0], eig_vals[1]]),
                )
            )

        return obstacle_environment 

    def get_subobstacle_weights(
        self, position, gamma_min_factor=0.9
    ):
        """ Get the importance weight of the different sub-obstacles."""
        # if any(self.gammas < 1):
            # How should this be handled?!
            # logging.warning("The current position is within the multi-gamma.")

        self.relative_weights = np.maximum(
            0, (self.gammas - gamma_min_factor * np.min(self.gammas))
        )
        self.relative_weights = self.relative_weights / np.sum(self.relative_weights)

        self.gamma_weights = 1.0 / self.gammas

    def _get_gauss_derivative(self, position, index, powerfactor=1):
        """ The additional powerfactor allows"""
        fraction_value = 1 / np.sqrt(
            (2*np.pi)**self.dimension * LA.det(self._gmm.covariances_[ind, :, :])
        )
        delta_dist = position - self._gmm.means_[ind, :, :]
        exp_value = np.exp(
            (-0.5) * delta_dist.T @ self._gmm.precisions_cholesky[ind, :, :] @ delta_dist
            * powerfactor
        )
        deriv_factor = (-0.5) * self._precisions_cholesky[ind, :, :] @ delta_dist
        return  (powerfactor*deriv_factor) * fraction_value * exp_value

    def get_intersection_of_ellipses(
        self, indices, powerfactor: float = 10.0, it_max: int = 100,
        rtol: float = 1e-1, step_size: float = 0.1,
    ) -> np.ndarray:
        """ Returns the intersection of ellipses using gradient descent of the GMM.

        Arguments
        rtol: Relative tolerance for convergence.
        """
        # TODO: this convergence stepping could be improved by
        # considering the covariances
        abs_tol = (
            rtol**powerfactor
            * LA.norm(self._gmm.means_[indices[0]] - self._gmm.means_[indices[1]])
        )
        
        if not abs_tol:  # Zero value
            logging.info("Almost identical center for two Gaussians detected.")
            return self._gmm.means_[indeces[0]]

        # Starting point is the center of the shortest connection
        pos_intersect = np.mean(self._gmm.means_[indices], axis=0)
        
        for ii in range(it_max):
            gradient_step = np.zeros(self.dimension)
            for index in indices:
                gradient_step += self._get_gauss_derivative(
                    pos_intersect, index, powerfactor)

            # TODO: maybe reduce power-factor to decrease the step size slightly
            step_norm = LA.norm(linalg)
            if step_norm < abs_tol:
                logging.info(f"Convergence at iteration {it_max}")

            gradient_step = gradient_step / step_norm * (step_norm)**(1/gradient_step)
            pos_intersect += gradient_step * step_size

        # Gamma-gradient descent
        # gamma_combo = lambda x: gamma1 ** powerfactor + gamma2 ** powerfactor
        return pos_intersect


class MultiVariantGaussian:
# NOT used anymore(?!)
    """
    This class allows the evaluation of multivariant Gaussian distribution
    (and their derivative).
    Attributes
    ----------
    _precisions_cholesky: inverse of the covariance matrix (for faster calculation)
    _fraction_value: similarly - too speed up computation
    """
    def __init__(self, mean, covariance, precision_cholesky=None):
        self._mean = mean
        self._covariance = covariance
        
        if precision_cholesky is None:
            self._precision_cholesky = LA.pinv(self._covariance)
        else:
            self._precision_cholesky = precision_cholesky

        self._fraction_value = 1 / np.sqrt(
            (2*np.pi)**self.dimension * LA.det(self._covariance)
        )

    @property
    def dimension(self):
        try:
            return self._mean.shape[0]
        except AttributeError:
            logging.warning("Object not defined.")
            return 0

    def evaluate(self, position):
        delta_dist = position - self.mean
        exp_value = np.exp( (-0.5)* delta_dist.T @ self._precision_cholesky @ delta_dist)
        return (self._fraction_value * exp_value

    def evaluate_derivative(self, position):
        delta_dist = position - self.mean
        exp_value = np.exp( (-0.5) * delta_dist.T @ self._precision_cholesky @ delta_dist)
                
        return (-0.5 * self._precision_cholesky @ delta_dist) * self._fraction_value * exp_value
    

