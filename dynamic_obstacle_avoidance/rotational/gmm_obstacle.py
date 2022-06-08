#!/USSR/bin/python3
""" Sample the space and decide if points are collision-full or free. """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-06-08

import logging

import numpy as np
from numpy import linalg

from sklearn.mixture import GaussianMixture

# from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.rotational.graph_handler import GraphElement
from dynamic_obstacle_avoidance.rotational.graph_handler import GraphHandler


class MultiGmmObstacle():
    """ Obstacle which learns to create multiple ellipses.

    Everything is with respect to the global frame (since the
    data originates from the global frame).

    This obstacle is learned from datapoints.

    The ellipses are defined such that the eigenvalues of the gmm-gaussian corresponding
    to the """
    def __init__(self, n_gmms) -> None:
        self.n_gmms = n_gmms
        self.gamma_list = None

    def fit(self, datapoints) -> None:
        self._gmm = GaussianMixture(
            n_components=self.n_gmms, covariance_type="full"
        ).fit(datapoints.T)

    def evaluate_gammas(self, position):
        self.gamma_list = self._gmm.predict_proba(position)
        breakpoint()

    def get_one_level_hirarchy(self):
        center_positions = np.zeros((
            self.environment.dimension, len(self.environment)
        ))
        for ii, obs in enumerate(self.environment):
            center_positions[:, ii] = obs.center_position

        mean = np.mean(center_positions, axis=1)

        ind_closest = np.argmin(
            linalg.norm(
                center_positions - np.tile(mean, (center_positions.shape[1], 1)).T, axis=1
            )
        )

        one_level_graph = GraphHandler()   #
        one_level_graph.root = self.environment[ind_closest]
        # Root does not need an update of the reference point
        
        for ii, obs in enumerate(self.environment):
            if ii == ind_closest:
                continue

            one_level_graph.add_element_with_parent(
                child=obs, parent=self.environment[ind_closest]
            )
            ref_point = self.get_intersection_of_ellipses(obs, self.environment[ind_closest])
            obs.set_reference_point(ref_point, in_global_frame=True)
            
        return one_level_graph
    
    def modulate(self, velocity):
        pass

    def transform_to_analytic_ellipses(self):
        """ Returns ObstacleContainer with n_gmm Ellipse obstacles
        with pose-axes description."""
        obstacle_environment = ObstacleContainer()

        for n in range(len(gmm.covariances_)):
            if gmm.covariance_type == "full":
                covariances = gmm.covariances_[n][:2, :2]
            elif gmm.covariance_type == "tied":
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == "diag":
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == "spherical":
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]

            eig_vals, eig_vecs = np.linalg.eigh(covariances)

            uu = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
            angle = np.arctan2(uu[1], uu[0])
            angle = 180 * angle / np.pi  # convert to degrees
            eig_vals = 2.0 * np.sqrt(2.0) * np.sqrt(eig_vals)

            obstacle_environment.append(
                Ellipse(
                    center_position=gmm.means_[n, :2],
                    orientation=angle,
                    axes_length=axes_factor*np.array([eig_vals[0], eig_vals[1]]),
                )
            )

        return obstacle_environment 

    def get_ellipses_from_2d_gmm(self, gmm, axes_factor=1.4, ax=None):
        # Assumption of single GMM per cluster
        self.environment = ObstacleContainer()

        for n in range(len(gmm.covariances_)):
            if gmm.covariance_type == "full":
                covariances = gmm.covariances_[n][:2, :2]
            elif gmm.covariance_type == "tied":
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == "diag":
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == "spherical":
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]

            # v, w = np.linalg.eigh(covariances)
            eig_vals, eig_vecs = np.linalg.eigh(covariances)

            uu = eig_vecs[0] / np.linalg.norm(eig_vecs[0])
            angle = np.arctan2(uu[1], uu[0])
            angle = 180 * angle / np.pi  # convert to degrees
            eig_vals = 2.0 * np.sqrt(2.0) * np.sqrt(eig_vals)

            self.environment.append(
                Ellipse(
                    center_position=gmm.means_[n, :2],
                    orientation=angle,
                    axes_length=axes_factor*np.array([eig_vals[0], eig_vals[1]]),
                )
            )

            if ax is not None:
                color = [0.3, 0.3, 0.3]

                ell = mpl.patches.Ellipse(
                    gmm.means_[n, :2],
                    axes_factor*eig_vals[0],
                    axes_factor*eig_vals[1], 180 + angle, color=color
                )
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(0.5)

                ax.add_artist(ell)

        return self.environment

    def get_subobstacle_weights(
        self, position, gamma_min_factor=0.9
    ):
        """ Get the importance weight of the different sub-obstacles."""
        self.gammas = np.zeros(len(self.environment))
        for ii, obs in enumerate(self.environment):
            self.gammas[ii] = obs.get_gamma(position)

        # if any(self.gammas < 1):
            # How should this be handled?!
            # logging.warning("The current position is within the multi-gamma.")

        self.relative_weights = np.maximum(
            0, (self.gammas - gamma_min_factor * np.min(self.gammas))
        )
        self.relative_weights = self.relative_weights / np.sum(self.relative_weights)

        self.gamma_weights = 1.0 / self.gammas

    def get_intersection_of_ellipses(self, ind1, ind2) -> np.ndarray:
        """ Returns the intersection of ellipses using gradient descent of the gmm."""
        # Initialize for the two specific Gaussians
        temp_gmm = GaussianMixture()
        breakpoint()
        # temp_gmm.means_ = None
        pass


