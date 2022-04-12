"""
Introduce Barrier functions for further calculation.
"""
from abc import ABC, abstractmethod

import numpy as np
from numpy import linalg as LA

from dynamic_obstacle_avoidance.obstacles import Sphere, DoubleBlob

from vartools.math import get_numerical_gradient, get_numerical_hessian

# def get_barrier_from_gamma():


class BarrierFunction(ABC):
    def evaluate(self, position):
        return self.get_barrier_value(position)

    @abstractmethod
    def get_barrier_value(self, position):
        pass

    def evaluate_gradient(self, position):
        return self.get_gradient(position)

    def get_gradient(self, position):
        """Default numerical function. Replace them with the analytical function if possible."""
        return get_numerical_gradient(
            function=self.get_barrier_value, position=position
        )

    def get_hessian(self, position):
        """Default numerical function. Replace them with the analytical function if possible."""
        return get_numerical_hessian(function=self.get_barrier_value, position=position)

    def draw_barrier_safe_value(self, ax, polygon_color="#00ff00ff"):
        if self._obs is None:
            self.create_obs()
        self._obs.plot_obstacle(ax=ax, polygon_color=polygon_color)


class CirclularBarrier(BarrierFunction):
    def __init__(self, radius=1, center_position=None, *args, **kwargs):
        self.dimension = 2
        if center_position is None:
            self.center_position = np.zeros(self.dimension)
        else:
            self.center_position = center_position

        self.radius = radius

        self._obs = None

    def get_barrier_value(self, position):
        relative_position = position - self.center_position
        # return 0.5*LA.norm(relative_position)**2 - 0.5*self.radius**2
        return LA.norm(relative_position) ** 2 - self.radius**2

    # def get_gradient(self, position):
    # relative_position = position - self.center_position
    # return super().get_gradient(relative_position)

    def get_hessian(self, position):
        # return np.eye(self.dimension)
        return 2 * np.eye(self.dimension)

    def create_obs(self):
        self._obs = Sphere(center_position=self.center_position, radius=self.radius)


class DoubleBlobBarrier(BarrierFunction):
    def __init__(self, blob_matrix, center_position=None, *args, **kwargs):
        self.blob_matrix = blob_matrix
        self.dim = 2
        if center_position is None:
            self.center_position = np.zeros(self.dim)
        else:
            self.center_position = center_position

        self._obs is None

    def get_barrier_value(self, position):
        """Out of the book double-blob hull value."""
        relative_position = position - self.center_position
        hull_value = LA.norm(relative_position) ** 4 - relative_position.T.dot(
            self.blob_matrix
        ).dot(relative_position)
        return hull_value

    def evaluate_gradient(self, position):
        relative_position = position - self.center_position
        gradient = 4 * LA.norm(
            relative_position
        ) ** 2 * relative_position - 2 * self.blob_matrix.dot(relative_position)
        return gradient

    def create_obs(self):
        self._obs = DoubleBlob(
            a_value=self.blob_matrix[0],
            b_value=self.blob_matrix[0],
        )


class BarrierFromObstacleList(BarrierFunction):
    def __init__(self, obstacle_list):
        self._obstacle_list = obstacle_list

    def get_barrier_value(self, position):
        """Transform the gamma-function [1, infinity] to a barrier function [0, infinity]
        Assumption of proportional-gamma"""
        n_obs = len(self._obstacle_list)

        barrier_values = np.zeros(n_obs)
        for ii in range(n_obs):
            pos_local = self._obstacle_list[ii].transform_global2relative(position)
            norm_pos = LA.norm(pos_local)
            gamma = self._obstacle_list[ii].get_gamma(pos_local)

            rad_local = norm_pos / gamma

            if self._obstacle_list[ii].is_boundary:
                barrier_values[ii] = rad_local - rad_local
            else:
                barrier_values[ii] = norm_pos - rad_local

        return np.prod(barrier_values)
