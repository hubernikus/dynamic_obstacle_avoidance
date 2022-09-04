#!/USSR/bin/python3.10
""" Test directional orientation system. """
# Author: Lukas Huber
# Created: 2022-09-02
# Github: hubernikus
# License: BSD (c) 2022

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


class DirectionalLearningSystem(DynamicalSystem):
    def __init__(self, reference_dynamics: ConstantValue):
        self.reference_dynamics = ConstantValue

        self.base = get_orthogonal_basis(
            reference_dynamics.evalute(np.zeros(reference_dynamics.dimensions))
        )

    def dimension(self) -> int:
        return self.reference_dynamics.dimension

    def evaluate(self, position: Vector) -> Vector:
        angle_deviation = self.evaluate_deviation(position)

        return get_angle_space_inverse(angle_deviation, self.base)

    def evaluate_deviation(self, position: Vector) -> DeviationVector:
        return np.zeros(self.dimension - 1)


def test_simple_circle_avoidance(visualize=True, save_figure=False):
    pass


if (__name__) == "__main__":

    test_simple_circle_avoidance()
    print("Tests finished.")
