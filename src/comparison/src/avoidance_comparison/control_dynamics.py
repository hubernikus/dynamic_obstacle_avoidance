"""
Control dynamics for static case
"""

from abc import ABC

import numpy as np
from numpy import linalg as LA


class ControlDynamics(ABC):
    pass


class StaticControlDynamics(ControlDynamics):
    def __init__(self, A_matrix):
        self.A_matrix = np.array(A_matrix)

    @property
    def dimension(self):
        return self.A_matrix.shape[1]

    def evaluate(self, position):
        return self.A_matrix
