""" Obstacle Avoider Virtual Base # from abc import ABC, abstractmethod. """
# Author Lukas Huber
# Mail lukas.huber@epfl.ch
# Created 2021-09-13
# License: BSD (c) 2021
# from abc import ABC, abstractmethod

from abc import ABC, abstractmethod
import warnings

import numpy as np
from numpy import linalg as LA

from vartools.dynamical_systems import DynamicalSystem

from dynamic_obstacle_avoidance.containers import BaseContainer
from dynamic_obstacle_avoidance.obstacles import GammaType

from .modulation import obs_avoidance_interpolation_moving


class ObstacleAvoiderWithInitialDynamcis:
    def __init__(
        self,
        initial_dynamics: DynamicalSystem,
        environment: BaseContainer,
        maximum_speed: float = None,
    ) -> None:
        self.initial_dynamics = initial_dynamics
        self.environment = environment

        self.maximum_speed = maximum_speed

    def get_gamma_product(self, position, gamma_type=GammaType.EUCLEDIAN):
        if not len(self.environment):
            # Very large number
            return 1e20

        gamma_list = np.zeros(len(self.environment))
        for ii, obs in enumerate(self.environment):
            # gamma_type needs to be implemented for all obstacles
            gamma_list[ii] = obs.get_gamma(
                position, in_global_frame=True, gamma_type=gamma_type
            )

        n_obs = len(gamma_list)
        # Total gamma [1, infinity]
        # Take root of order 'n_obs' to make up for the obstacle multiple
        if any(gamma_list < 1):
            warnings.warn("Collision detected.")
            # breakpoint()
            return 0

        # gamma = np.prod(gamma_list-1)**(1.0/n_obs) + 1
        gamma = np.min(gamma_list)

        if np.isnan(gamma):
            breakpoint()
        return gamma

    def evaluate(self, position: np.ndarray) -> np.ndarray:
        """DynamicalSystem compatible 'evaluate' method that returns the velocity at
        a given input position."""
        return self.compute_dynamics(position)

    def compute_dynamics(self, position: np.ndarray) -> np.ndarray:
        """DynamicalSystem compatible 'compute_dynamics' method that returns the
        velocity at a given input position."""
        initial_velocity = self.initial_dynamics.evaluate(position)
        return self.avoid(position=position, initial_velocity=initial_velocity)

    @abstractmethod
    def avoid(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        pass


class DynamicModulationAvoider(ObstacleAvoiderWithInitialDynamcis):
    def __init__(
        self, input_output_speed_constant: bool = False, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.input_output_speed_constant = input_output_speed_constant

    def avoid(
        self,
        position: np.ndarray,
        initial_velocity: np.ndarray,
    ) -> np.ndarray:
        vel = obs_avoidance_interpolation_moving(
            position=position,
            initial_velocity=initial_velocity,
            obs=self.environment,
        )

        # Adapt speed if desired
        if self.input_output_speed_constant:
            vel_mag = LA.norm(vel)
            if vel_mag:
                vel = vel / vel_mag * LA.norm(initial_velocity)

        elif self.maximum_speed is not None:
            vel_mag = LA.norm(vel)
            if vel_mag > self.maximum_speed:
                vel = vel / vel_mag * self.maximum_speed

        return vel


class DynamicCrowdAvoider(ObstacleAvoiderWithInitialDynamcis):
    def __init__(
            self,
            initial_dynamics: DynamicalSystem,
            environment: BaseContainer,
            maximum_speed: float = None,
            obs_multi_agent=None,
    ):
        super().__init__(initial_dynamics, environment, maximum_speed)
        self.obs = None
        self.obs_multi_agent = obs_multi_agent

    def env_slicer(self, obs_index):
        temp_env = self.environment[0:obs_index] + self.environment[obs_index + 1:]
        return temp_env

    @staticmethod
    def get_gamma_product_crowd(position, env, gamma_type=GammaType.EUCLEDIAN):
        if not len(env):
            # Very large number
            return 1e20

        gamma_list = np.zeros(len(env))
        for ii, obs in enumerate(env):
            # gamma_type needs to be implemented for all obstacles
            gamma_list[ii] = obs.get_gamma(
                position, in_global_frame=True, gamma_type=gamma_type
            )

        n_obs = len(gamma_list)
        # Total gamma [1, infinity]
        # Take root of order 'n_obs' to make up for the obstacle multiple
        if any(gamma_list < 1):
            warnings.warn("Collision detected.")
            # breakpoint()
            return 0

        # gamma = np.prod(gamma_list-1)**(1.0/n_obs) + 1
        gamma = np.min(gamma_list)

        if np.isnan(gamma):
            breakpoint()
        return gamma

    def get_gamma_at_control_point(self, control_points, obs_eval, env):
        # TODO
        gamma_values = np.zeros(len(self.obs_multi_agent[obs_eval]))

        for cp in self.obs_multi_agent[obs_eval]:
            gamma_values[cp] = self.get_gamma_product_crowd(control_points[cp, :], env)

        return gamma_values

    @staticmethod
    def get_weight_from_gamma(gammas, cutoff_gamma, n_points, gamma0=1.0, frac_gamma_nth=0.5):
        weights = (gammas - gamma0) / (cutoff_gamma - gamma0)
        weights = weights / frac_gamma_nth
        weights = 1.0 / weights
        weights = (weights - frac_gamma_nth) / (1 - frac_gamma_nth)
        weights = weights / n_points
        return weights

    def get_influence_weight_at_ctl_points(self, control_points, cutoff_gamma=5):
        # TODO
        ctl_weight_list = []
        for obs in self.obs_multi_agent:
            if not self.obs_multi_agent[obs]:
                break
            temp_env = self.env_slicer(obs)
            gamma_values = self.get_gamma_at_control_point(control_points[self.obs_multi_agent[obs]], obs, temp_env)

            ctl_point_weight = np.zeros(gamma_values.shape)
            ind_nonzero = gamma_values < cutoff_gamma
            if not any(ind_nonzero):
                ctl_point_weight[-1] = 1
            # for index in range(len(gamma_values)):
            ctl_point_weight[ind_nonzero] = self.get_weight_from_gamma(
                gamma_values[ind_nonzero],
                cutoff_gamma=cutoff_gamma,
                n_points=len(self.obs_multi_agent[obs])
            )

            ctl_point_weight_sum = np.sum(ctl_point_weight)
            if ctl_point_weight_sum > 1:
                ctl_point_weight = ctl_point_weight / ctl_point_weight_sum
            else:
                ctl_point_weight[-1] += 1 - ctl_point_weight_sum

            ctl_weight_list.append(ctl_point_weight)

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
