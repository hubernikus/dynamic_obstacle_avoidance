#!/USSR/bin/python3.10
""" Test directional orientation system. """
# Author: Lukas Huber
# Created: 2022-09-02
# Github: hubernikus
# License: BSD (c) 2022

from abc import ABC, abstractmethod

from math import pi

# from typing import List, Protocol

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from sklearn.svm import SVR

from vartools.linalg import get_orthogonal_basis
from vartools.dynamical_systems import DynamicalSystem, ConstantValue

# from vartools.directional_space import get_angle_space_of_array
# from vartools.directional_space import get_angle_space
from vartools.directional_space import get_angle_space_inverse
from vartools.directional_space import get_angle_space, get_angle_space_of_array

from dynamic_obstacle_avoidance.rotational.datatypes import Vector, VectorArray

Basis = np.ndarray
DeviationVector = np.ndarray

# TODO:
# - directly incoorporate (rotational + fast)) obstacle avoidance as a function


class MultiOutputSVR:
    """Creates several svr-models to predict multi-dimensional output."""

    def __init__(self, **kwargs):
        # Temporary store keyword arguments for later construction.
        self._kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = y.shape[1]

        self._models = [SVR(**self._kwargs) for _ in range(self.n_features_out_)]

        for ii, svr in enumerate(self._models):
            svr.fit(X, y[:, ii])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([svr.predict(X) for svr in self._models])


class DirectionalSystem(ABC):
    def evaluate(self, position: Vector) -> Vector:
        angle_deviation = self.evaluate_deviation(position)

        return get_angle_space_inverse(angle_deviation, self.base)

    @abstractmethod
    def evaluate_deviation(self, position: Vector) -> DeviationVector:
        pass


class DeviationOfConstantFlow(DirectionalSystem):
    maximal_deviation = 0.99 * pi / 2.0

    def __init__(self, reference_velocity: np.ndarray, regressor) -> None:
        self.reference_velocity = reference_velocity
        self.base = get_orthogonal_basis(
            reference_velocity / LA.norm(reference_velocity)
        )

        self.regressor = regressor

    @property
    def dimension(self):
        return self.reference_velocity.shape[0]

    def fit_from_velocities(self, X: np.ndarray, y_directions: np.ndarray) -> None:
        y = self._clean_input_data(y_directions)
        self.fit(X, y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the internal learner onto the given model."""
        self.n_samples_fit_ = X.shape[0]

        if y.shape[0] != self.n_samples_fit_:
            raise ValueError("Input data is not consistent.")

        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = y.shape[1]

        if self.n_features_out_ != self.dimension - 1:
            raise ValueError("Unexpected number of output features.")

        self.regressor.fit(X, y)

    def _clean_input_data(self, data):
        # TODO: use directly the array -> get_angle_space_of_array()
        directions = np.zeros((data.shape[0], data.shape[1] - 1))
        for ii in range(data.shape[0]):
            directions[ii, :] = get_angle_space(data[ii, :], null_matrix=self.base)
        return directions

    def evaluate_deviation(self, position: Vector) -> np.ndarray:
        """Return deviation, while making sure that the deviation stays within the
        feasible boundaries."""
        deviation = self.regressor.predict(position.reshape(1, -1))[0]
        if (dev_norm := LA.norm(deviation)) > self.maximal_deviation:
            deviation = deviation / dev_norm
        return deviation


def test_learn_sinus_motion(visualize=False, save_figure=False):
    # Data going from right to left
    np.random.seed(1)

    n_points = 1000
    x = np.linspace(0, 4 * pi, n_points)
    y = np.cos(x)

    X = np.vstack((x, y)).T
    X = X + np.random.rand(X.shape[0], X.shape[1]) * 0.1

    # Obtain velocities form position, and make sure they have the same dimensions
    velocities = X[1:, :] - X[:-1, :]
    X = X[:-1, :]

    dynamics = DeviationOfConstantFlow(
        reference_velocity=np.mean(velocities, axis=0),
        regressor=MultiOutputSVR(kernel="rbf", gamma=0.1),
    )

    dynamics.fit_from_velocities(X, velocities)

    if visualize:
        fig, ax = plt.subplots()
        ax.plot(X[:, 0], X[:, 1], ".")


if (__name__) == "__main__":

    test_learn_sinus_motion(visualize=True)
    print("Tests finished.")
