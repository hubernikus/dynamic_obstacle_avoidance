"""
Introduce Barrier functions for further calculation.
"""
from abc import ABC, abstractmethod

import numpy as np
from numpy import linalg as LA

from vartools.math import get_numerical_gradient, get_numerical_hessian


class BarrierFunction(ABC):
    def evaluate(self, position):
        return self.get_barrier_value(position)
    
    @abstractmethod
    def get_barrier_value(self, position):
        pass

    def evaluate_gradient(self, position):
        return self.get_gradient(position)
    
    def get_gradient(self, position):
        """ Default numerical function. Replace them with the analytical function if possible."""
        return get_numerical_gradient(function=self.get_barrier_value, position=position)

    def get_hessian(self, position):
        """ Default numerical function. Replace them with the analytical function if possible."""
        return get_numerical_hessian(function=self.get_barrier_value, position=position)


class CirclularBarrier(BarrierFunction):
    def __init__(self, radius=1, center_position=None, *args, **kwargs):
        self.dimension = 2
        if center_position is None:
            self.center_position = np.zeros(self.dimension)
        else:
            self.center_position = center_position

        self.radius = radius
    
    def get_barrier_value(self, position):
        relative_position = position - self.center_position
        # return 0.5*LA.norm(relative_position)**2 - 0.5*self.radius**2
        return LA.norm(relative_position)**2 - self.radius**2

    def get_hessian(self, position):
        # return np.eye(self.dimension)
        return 2*np.eye(self.dimension)
    

class DoubleBlobBarrier(BarrierFunction):
    def __init__(self, blob_matrix, center_position=None, *args, **kwargs):
        self.blob_matrix = blob_matrix
        self.dim = 2
        if center_position is None:
            self.center_position = np.zeros(self.dim)
        else:
            self.center_position = center_position

    def get_barrier_value(self, position):
        """ Out of the book double-blob hull value."""
        relative_position = position - self.center_position
        hull_value = (
            LA.norm(relative_position)**4
            - relative_position.T.dot(self.blob_matrix).dot(relative_position))
        return hull_value

    def evaluate_gradient(self, position):
        relative_position = position - self.center_position
        gradient = (4*LA.norm(relative_position)**2*relative_position
                      - 2*self.blob_matrix.dot(relative_position))
        return gradient

class BarrierFromObstacleList(BarrierFunction):
    def __init__(self):
        pass

    def get_barrier_value(self, position):
        breakpoint()
    
    
