"""
Examples of how to use an obstacle-boundary mix,
i.e., an obstacle which can be entered

This could be bag in task-space, or a complex-learned obstacle.
"""
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-06-21

import copy

import numpy as np
import numpy.typing as npt

from dynamic_obstacle_avoidance.obstacles import Obstacle

from dynamic_obstacle_avoidance.visualization import plot_obstacles

Vector = npt.ArrayLike


class MultiHullAndObstacle(Obstacle):
    """This is of type obstacle."""

    def __init__(
        self,
        boundary_list: list,
        obstacle_list: list,
        center_position: np.ndarray = None,
        **kwargs
    ):
        if center_position is None:
            center_position = np.zeros(obstacle_list[0].center_position.shape)

        super().__init__(center_position=center_position, **kwargs)

        # This obstacle is duality of boundary - obstacle (depending on position)
        self.is_boundary = None

        self.obstacle_list = obstacle_list
        self.boundary_list = boundary_list

    @property
    def total_list(self):
        return self.obstacle_list + self.boundary_list

    @property
    def n_elements(self):
        return len(self.obstacle_list) + len(self.boundary_list)

    @property
    def _indices_obstacles(self):
        return np.arange(0, len(self.obstacle_list))

    @property
    def _indices_boundaries(self):
        return np.arange(len(self.obstacle_list), self.n_elements)

    def _evaluate_weights(
        self,
        position: Vector,
        mult_power_weight: float = 3.0,
        max_power_weight: float = 5.0,
    ):
        """Position input is in local-frame."""
        self.gamma_list = np.zeros(self.n_elements)
        self.weights = np.zeros(self.n_elements)

        for ii, obs_ii in enumerate(self.total_list):
            self.gamma_list[ii] = obs_ii.get_gamma(position, in_global_frame=True)
        self.weights = np.maximum(self.gamma_list - 1, 0)

        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            # At least one weight has to be bigger than one
            self.weights /= weight_sum

    def get_normal_direction(self, position: Vector, in_global_frame=True):
        pass

    def plot_obstacle(self, ax=None, x_lim: list = None, y_lim: list = None) -> None:
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        plot_obstacles(
            obstacle_container=self.obstacle_list,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            # alpha_obstacle=1.0,
        )

        temp_list = copy.deepcopy(self.boundary_list)
        for obs in temp_list:
            obs.is_boundary = False

        plot_obstacles(
            obstacle_container=temp_list,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            alpha_obstacle=1.0,
            obstacle_color="white",
            draw_reference=True,
        )

    def get_gamma(self, position: Vector, in_global_frame: bool = False) -> float:
        """Get distance value with respect to the obstacle."""
        if in_global_frame:
            position = self.transform_global2relative(position)
        breakpoint()
        # return gamma

    def is_inside(self):
        """ """
        pass

    def get_intersection_of_ellipses(self):
        raise NotImplementedError()
