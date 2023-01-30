""" Obstacle Avoider Virtual Base # from abc import ABC, abstractmethod. """
# Author Lukas Huber
# Github
# Created 2022-05-20
# License: BSD (c) 2022

from abc import ABC, abstractmethod
import warnings
from typing import Optional

import numpy as np
from numpy import linalg as LA

from vartools.dynamical_systems import DynamicalSystem

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import GammaType
from dynamic_obstacle_avoidance.containers import BaseContainer
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving

from .obstacle_avoider import ObstacleAvoiderWithInitialDynamcis


def obstacle_environment_slicer(
    environment: BaseContainer, obs_index: int
) -> list[Obstacle]:
    return environment[0:obs_index] + environment[obs_index + 1 :]


class DynamicCrowdAvoider(ObstacleAvoiderWithInitialDynamcis):
    def __init__(
        self,
        initial_dynamics: DynamicalSystem,
        obstacle_environment: BaseContainer,
        maximum_speed: Optional[float] = None,
        obs_multi_agent=None,
    ):
        super().__init__(
            initial_dynamics=initial_dynamics,
            obstacle_environment=obstacle_environment,
            maximum_speed=maximum_speed,
        )
        self.obs = None
        self.obs_multi_agent = obs_multi_agent

    def environment_slicer(self, obs_index):
        temp_env = (
            self.obstalce_environment[0:obs_index]
            + self.obstalce_environment[obs_index + 1 :]
        )
        return temp_env

    @staticmethod
    def get_gamma_product_crowd(
        position, env: BaseContainer, gamma_type=GammaType.EUCLEDIAN
    ):
        if not len(env):
            # Very large number
            return 1e20

        gamma_list = np.zeros(len(env))
        for ii, obs in enumerate(env):
            # if not isinstance(obs, Obstacle):
            #     # TODO: remove... This is only for debugging purposes
            #     breakpoint()
            gamma_list[ii] = obs.get_gamma(position, in_global_frame=True)

        n_obs = len(gamma_list)
        # Total gamma [1, infinity]
        # Take root of order 'n_obs' to make up for the obstacle multiple
        if any(gamma_list < 1):
            warnings.warn("Collision detected.")
            return 0

        # gamma = np.prod(gamma_list-1)**(1.0/n_obs) + 1
        gamma = np.min(gamma_list)

        if np.isnan(gamma):
            breakpoint()
        return gamma

    def get_gamma_at_control_point(
        self, control_points: np.ndarray, obs_eval: Obstacle, env: BaseContainer
    ):
        gamma_values = np.zeros(len(control_points))

        for cp in range(len(self.obs_multi_agent[obs_eval])):
            gamma_values[cp] = self.get_gamma_product_crowd(
                control_points[cp, :], env=env
            )

        return gamma_values

    @staticmethod
    def get_weight_from_gamma(
        gammas, cutoff_gamma, n_points, gamma0=1.0, frac_gamma_nth=0.5
    ):
        weights = (gammas - gamma0) / (cutoff_gamma - gamma0)
        weights = weights / frac_gamma_nth
        weights = 1.0 / weights
        weights = (weights - frac_gamma_nth) / (1 - frac_gamma_nth)
        weights = weights / n_points
        return weights

    def get_influence_weight_at_ctl_points(
        self, control_points, cutoff_gamma=5, return_gamma: bool = False
    ):
        # TODO
        ctl_weight_list = []
        gamma_values_list = np.empty(shape=0)
        ctl_weight_save = np.empty(shape=0)
        for obs in self.obs_multi_agent:
            if not self.obs_multi_agent[obs]:
                break
            # temp_env = self.env_slicer(obs)
            temp_env = obstacle_environment_slicer(
                self.obstacle_environment, obs_index=obs
            )
            gamma_values = self.get_gamma_at_control_point(
                control_points[self.obs_multi_agent[obs]],
                obs_eval=obs,
                env=temp_env,
            )

            ctl_point_weight = np.zeros(gamma_values.shape)
            ind_nonzero = gamma_values < cutoff_gamma
            if not any(ind_nonzero):
                # ctl_point_weight[-1] = 1
                ctl_point_weight = np.full(
                    gamma_values.shape, 1 / len(self.obs_multi_agent[obs])
                )
            # for index in range(len(gamma_values)):
            ctl_point_weight[ind_nonzero] = self.get_weight_from_gamma(
                gamma_values[ind_nonzero],
                cutoff_gamma=cutoff_gamma,
                n_points=len(self.obs_multi_agent[obs]),
            )

            ctl_point_weight_sum = np.sum(ctl_point_weight)
            if ctl_point_weight_sum > 1:
                ctl_point_weight = ctl_point_weight / ctl_point_weight_sum
            else:
                ctl_point_weight[-1] += 1 - ctl_point_weight_sum

            ctl_weight_list.append(ctl_point_weight)

            if return_gamma:
                gamma_values_list = np.append(gamma_values_list, gamma_values)
                ctl_weight_save = np.append(ctl_weight_save, ctl_point_weight)

        if return_gamma:
            return ctl_weight_list, gamma_values_list, ctl_weight_save

        return ctl_weight_list

    def evaluate_for_crowd_agent(
        self, position: np.ndarray, selected_agent, env
    ) -> np.ndarray:
        """DynamicalSystem compatible 'evaluate' method that returns the velocity at a
        given input position."""
        return self.compute_dynamics_for_crowd_agent(position, selected_agent, env)

    def compute_dynamics_for_crowd_agent(
        self, position: np.ndarray, selected_agent, env
    ) -> np.ndarray:
        """DynamicalSystem compatible 'compute_dynamics' method that returns the velocity at a
        given input position."""
        initial_velocity = self.initial_dynamics[selected_agent].evaluate(position)

        return self.avoid_for_crowd_agent(
            position=position,
            initial_velocity=initial_velocity,
            env=env,
        )

    def avoid_for_crowd_agent(
        self,
        position: np.ndarray,
        initial_velocity: np.ndarray,
        env,
        const_speed: bool = True,
    ) -> np.ndarray:

        vel = obs_avoidance_interpolation_moving(
            position=position, initial_velocity=initial_velocity, obs=env
        )

        # Adapt speed if desired
        if const_speed:
            vel_mag = LA.norm(vel)
            if vel_mag:
                vel = vel / vel_mag * LA.norm(initial_velocity)

        elif self.maximum_speed is not None:
            vel_mag = LA.norm(vel)
            if vel_mag > self.maximum_speed:
                vel = vel / vel_mag * self.maximum_speed

        return vel

    def avoid(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        pass

    def get_attractor_position(self, control_point):
        return self.initial_dynamics[control_point].attractor_position

    def set_attractor_position(self, position: np.ndarray, control_point):
        self.initial_dynamics[control_point].attractor_position = position

    def get_gamma_at_pts(self, control_points, obstacle):
        gamma_values_list = np.empty(shape=0)

        for obs in self.obs_multi_agent:
            if not self.obs_multi_agent[obs]:
                break
            gamma_values = self.get_gamma_at_control_point(
                control_points[self.obs_multi_agent[obs]], obs, obstacle
            )
            gamma_values_list = np.append(gamma_values_list, gamma_values)

        return gamma_values_list
