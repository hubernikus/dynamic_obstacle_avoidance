"""
This is an example of how to structure a new Obstacle
"""
import numpy as np

from dynamic_obstacle_avoidance.obstacles import Obstacle


class NewObstacle(Obstacle):
    def __init__(self, dimension, is_jointspace=True):
        super().__init__(dimension=dimension)

        self.is_jointspace = is_jointspace

    def get_gamma(self, position):
        return 1

    def get_normal_direction(self, position):
        return np.ones(self.dimension) / self.dimension

    def get_reference_direction(self, position, **kwargs):
        # If there is a reference point (mostly in task space)
        # the reference-direction and normal-direction can differ

        if self.is_jointspace:
            return self.get_normal_direction(position)
        else:
            return super().get_reference_direction(position, **kwargs)
