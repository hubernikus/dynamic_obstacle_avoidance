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

from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving


class ObstacleAvoiderWithInitialDynamcis:
    def __init__(
        self,
        obstacle_environment: BaseContainer,
        initial_dynamics: DynamicalSystem = None,
        maximum_speed: float = None,
    ) -> None:
        self.initial_dynamics = initial_dynamics
        self.obstacle_environment = obstacle_environment

        self.maximum_speed = maximum_speed

    def get_gamma_product(self, position, gamma_type=GammaType.EUCLEDIAN):
        if not len(self.obstacle_environment):
            # Very large number
            return 1e20

        gamma_list = np.zeros(len(self.obstacle_environment))
        for ii, obs in enumerate(self.obstacle_environment):
            # gamma_type needs to be implemented for all obstacles
            gamma_list[ii] = obs.get_gamma(
                position, in_global_frame=True, gamma_type=gamma_type
            )

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

    def evaluate(self, position: np.ndarray) -> np.ndarray:
        """DynamicalSystem compatible 'evaluate' method that returns the velocity at
        a given input position."""
        initial_velocity = self.initial_dynamics.evaluate(position)
        return self.avoid(position=position, initial_velocity=initial_velocity)

    def compute_dynamics(self, position: np.ndarray) -> np.ndarray:
        """DynamicalSystem compatible 'compute_dynamics' method that returns the
        velocity at a given input position."""
        initial_velocity = self.initial_dynamics.evaluate(position)
        return self.avoid(position=position, initial_velocity=initial_velocity)

    @abstractmethod
    def avoid(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        pass
