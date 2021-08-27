"""

"""
import numpy as np
from numpy import linalg as LA


class RobotArmAvoider():
    def __ini__(self):
        self.jacobian = None
        self.obstacle_environment = None

    def set_evaluation_points(self, n_points):
        """ Set points """
        self.evaluation_points = None
        # self.evaluation_point_margins = None
        raise NotImplementedError()

    def get_gamma_at_evaluation_points(self):
        raise NotImplementedError()

    def evaluate_motino(self):
        pass

    def update(self):
        pass
