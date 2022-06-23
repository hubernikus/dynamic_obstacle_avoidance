"""
Examples of how to use an obstacle-boundary mix,
i.e., an obstacle which can be entered

This could be bag in task-space, or a complex-learned obstacle.
"""
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-06-21

import copy
import logging

import numpy as np
import numpy.linalg as LA
import numpy.typing as npt

import networkx as nx

from dynamic_obstacle_avoidance.obstacles import Obstacle

from dynamic_obstacle_avoidance.visualization import plot_obstacles

Vector = npt.ArrayLike


class MultiHullAndObstacle(Obstacle):
    """This is of type obstacle.

    Attributes
    ----------
    inner_obstacles: List of obstacle which form the inner hull.
    outer_obstacles: For now the outer_obstacle is limited to a single obstacle (concave).
    """

    def __init__(
        self,
        inner_obstacles: list,
        outer_obstacle: list,
        center_position: np.ndarray = None,
        **kwargs,
    ):
        if center_position is None:
            center_position = np.zeros(outer_obstacle.center_position.shape)

        super().__init__(center_position=center_position, **kwargs)

        # This obstacle is duality of boundary - obstacle (depending on position)
        self.is_boundary = None

        self.outer_obstacle = outer_obstacle
        self.inner_obstacles = inner_obstacles

    @property
    def total_list(self):
        return self.inner_obstacles + [self.outer_obstacle]

    @property
    def n_elements(self):
        return len(self.inner_obstacles) + 1

    @property
    def _indices_outer(self):
        # return np.arange(0, len(self.outer_obstacles))
        return len(self.inner_obstacles)

    @property
    def _indices_inner(self):
        return np.arange(self.n_elements - 1)

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
            obstacle_container=[self.outer_obstacle],
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            # alpha_obstacle=1.0,
        )

        temp_list = copy.deepcopy(self.inner_obstacles)
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
        pass

    def _intersections_with_outer_obstacle(self):
        """Get intersection with the 'free' space"""
        for ii, obs_ii in enumerate(self.inner_obstacles):
            if LA.norm(obs_ii.center_position - self.outer_obstacle.center_position) < (
                np.min(self.outer_obstacle.axes_length)
                - np.max(self.outer_obstacle.axes_length)
            ):
                continue

            position = self._gamma_normal_gradient_descent(
                [obs_ii, self.outer_obstacle],
                powers=[-2, -2],  # -> both in free space, i.e. < 0
                factors=[1, -1],
            )

            if (
                obs_ii.get_gamma(position) < 1
                or self.outer_obstacle.get_gamma(position) < 1
            ):
                continue

    def get_intersection_of_ellipses(self):
        raise NotImplementedError()

    def _evaluate_hirarchy_and_reference_points(self) -> None:
        self._graph = nx.Graph()
        self._graph.add_nodes_from(self.inner_obstacles)

        # Define entering & exit points
        self._exit_obstacles = []
        self._exit_references = []

        for ii, obs_ii in enumerate(self.inner_obstacles):
            for jj in range(len(self.inner_obstacles)):
                # Speedy distance check
                obs_jj = self.total_list[jj]
                dist_center = LA.norm(obs_ii.center_position - obs_jj.center_position)
                if dist_center < np.max(obs_ii.axes_length) + np.max(
                    obs_jj.axes_length
                ):
                    continue

                close_position = self._gamma_normal_gradient_descent(obs_ii, obs_jj)

                distance = LA.norm(obs_ii.center_position - close_position) + LA.norm(
                    obs_jj.center_position - close_position
                )

                self._graph.add_edge(
                    obs_jj, obs_jj, weight=distance, reference_point=close_position
                )

        if not self._graph.is_conected:
            # breakpoint()
            raise NotImplementedError()

    def _gamma_normal_gradient_descent(
        self,
        obstacles: Obstacle,
        factors: npt.ArrayLike = None,
        powers: npt.ArrayLike = None,
        it_max: int = 50,
        step_factor: float = 0.1,
        convergence_error: float = 1e-1,
    ) -> np.ndarray:
        """Returns the intersection-position (or a point if it does not exists),
        for two convex input obstacles.

        Arguments
        ---------
        factors: The factor of the direction;
            > 0 if outside obstacle OR inside boundary
            < 0 otherwise
        powers: Power of the weights, chose |powers| > 1 for better convergence; good
            choice for the values is |powers[ii]|=2. Furthermore:
            > 0 if inside obstacle OR inside boundary
            < 0 otherwise
        """
        position = 0.5 * (obstacles[0].center_position + obstacles[1].center_position)

        if powers is None:
            powers = (2 for _ in range(len(obstacles)))

        if factors is None:
            factors = (1 for _ in range(len(obstacles)))

        for ii in range(it_max):
            step = np.zeros(self.dimension)

            # Gamma Gradient and Normal direction (?)
            for ii, obs_ii in enumerate(obstacles):
                stepsize = (
                    obs_ii.get_gamma(position, in_global_frame=True) ** powers[ii]
                )
                step += (
                    stepsize
                    * factors[ii]
                    * obs_ii.get_normal_direction(position, in_global_frame=True)
                )

            if LA.norm(step) < convergence_error:
                logging.info(f"Gamma gradient converged at it={ii}")
                break

            position += step_factor * step

        # if (obs_ii.get_gamma(position, in_global_frame=True) < 1
        #     or obs_jj.get_gamma(position, in_global_frame=True)
        # ):
        #     # The points are not actually intersecting

        # else:
        return position
