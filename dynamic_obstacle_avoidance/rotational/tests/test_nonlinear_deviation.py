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

from vartools.linalg import get_orthogonal_basis
from vartools.dynamical_systems import DynamicalSystem, ConstantValue

# from vartools.directional_space import get_angle_space_of_array
# from vartools.directional_space import get_angle_space
from vartools.directional_space import get_angle_space_inverse

from dynamic_obstacle_avoidance.rotational.datatypes import Vector, VectorArray

Basis = np.ndarray
DeviationVector = np.ndarray

# TODO:
# - directly incoorporate (rotational + fast)) obstacle avoidance as a function


class DirectionalSystem(ABC):
    def __init__(self):
        self.base = get_orthogonal_basis(
            reference_dynamics.evalute(np.zeros(reference_dynamics.dimensions))
        )

    def dimension(self) -> int:
        return self.reference_dynamics.dimension

    def evaluate(self, position: Vector) -> Vector:
        angle_deviation = self.evaluate_deviation(position)
        
        return get_angle_space_inverse(angle_deviation, self.base)

    @abstractmethod
    def evaluate_deviation(self, position: Vector) -> DeviationVector:
        pass


class DeviationOfConstantFlow(DirectionalSystem):
    maximal_deviation = pi

    def __init__(self, reference_velocity: np.ndarray, regressor) -> None:
        self.reference_velocity = reference_velocity
        self.base = get_orthogonal_basis(
            reference_direction / LA.norm(reference_direction)
        )

        self.regressor = regressor

    @attribute
    def dimension(self):
        return self.reference_velocity.shape[0]

    def fit_from_directions(self, X, np.ndarray, y_directions: np.ndarray) -> None:
        y = self._clean_input_data(y_directions)
        self.fit(X, y)
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the internal learner onto the given model."""
        self.n_samples_fit_ = X.shape[0]

        if y.shape[0] != self.n_samples_fit_:
            raise ValueError("Input data is not consistent.")

        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = y.shape[1]

        if self.n_features_out_ != self.dimension-1:
            raise ValueError("Unexpected number of output features.")

        self.regressor.fit(X, y)

    def _clean_input_data(self, data):
        pass
        
    def evaluate_deviation(self, position: Vector) -> DevationVector:
        return self.regressor.predict(position.reshape(1, -1))[0]


def test_learn_sinus_motion(visualize=True, save_figure=False):

    dynamics= DeviationOfConstantFlow()
    pass


if (__name__) == "__main__":

    test_learn_sinus_motion()
    print("Tests finished.")
