""" (Single) Obstacle created from multiple ellipses (as a GMM description). """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-08-03

import numpy as np

from dynamic_obstacle_avoidance.rotational.datatypes import Vector


class BaseMulitObstacle:
    def _evaluate_weights(
        self,
        position: Vector,
        mult_power_weight: float = 3.0,
        max_power_weight: float = 5.0,
    ) -> None:
        """Position input is in local-frame."""
        self.gamma_list = np.zeros(self.n_elements)
        self.weights = np.zeros(self.n_elements)

        for ii, obs_ii in enumerate(self.all_obstacles):
            self.gamma_list[ii] = obs_ii.get_gamma(position, in_global_frame=True)
        self.weights = np.maximum(self.gamma_list - 1, 0)

        # self.gamma_weight = 1 - 1 / self.gamma_list

        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            # At least one weight has to be bigger than one
            self.weights /= weight_sum
