"""
Base QP-class
"""
import numpy as np
from numpy import linalg as LA
from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.obstacles import Obstacle


class ControllerQP:
    def __init__(self, f_x, g_x, barrier_function: Obstacle):
        self.f_x = f_x
        self.g_x = g_x
        self.barrier_function = barrier_function

    def evaluate(self, position):
        """Evaluate with 'temporary control'."""
        control = self.get_optimal_control(position)
        return self.evaluate_with_control(position, control=control)

    def evaluate_base_dynamics(self, position):
        return self.f_x.evaluate(position)

    def evaluate_control_dynamics(self, position):
        return self.g_x.evaluate(position)

    def get_derivative_of_dynamics(self, position, matrix_function):
        if isinstance(matrix_function, LinearSystem):
            dim = matrix_function.dimension
            return np.zeros((dim, dim, dim))
        else:
            raise NotImplementedError("Not implemented for other types...")
