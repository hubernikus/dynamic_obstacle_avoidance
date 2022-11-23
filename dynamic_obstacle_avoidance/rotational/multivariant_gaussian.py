#!/USSR/bin/python3
""" Multivariant Gaussian and it's derivative. """

# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-06-13

import logging

import numpy as np
from numpy import linalg as LA


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
            (2 * np.pi) ** self.dimension * LA.det(self._covariance)
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
        exp_value = np.exp(
            (-0.5) * delta_dist.T @ self._precision_cholesky @ delta_dist
        )
        return self._fraction_value * exp_value

    def evaluate_derivative(self, position):
        delta_dist = position - self.mean
        exp_value = np.exp(
            (-0.5) * delta_dist.T @ self._precision_cholesky @ delta_dist
        )

        return (
            (-0.5 * self._precision_cholesky @ delta_dist)
            * self._fraction_value
            * exp_value
        )
