""" Obstacle Avoider Virtual Base # from abc import ABC, abstractmethod. """
# Author Lukas Huber 
# Mail lukas.huber@epfl.ch
# Created 2021-09-13
# License: BSD (c) 2021
# from abc import ABC, abstractmethod

from abc import ABC, abstractmethod

import numpy as np

from vartools.dynamical_systems import DynamicalSystem

from dynamic_obstacle_avoidance.avoidance.containers import BaseContainer

from .modulation import obs_avoidance_interpolation_moving


class ObstacleAvoiderWithInitialDynamcis():
    def __init__(self, initial_dynamics: DynamicalSystem, obstacle_environment: BaseContainer):
        self._initial_dynamics = initial_dynamics
        self._obstacle_environment = _obstacle_environment
    
    def evaluate(self, position np.ndarray) ->  np.ndarray:
        """ DynamicalSystem compatible 'evaluate' method that returns the velocity at a
        given input position. """
        initial_velocity = self._initial_dynamics(position)
        return self.avoid(position, initial_velocity)

    @abstractmethod
    def avoid(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        pass


class DynamicModulationAvoider(ObstacleAvoiderWithInitialDynamcis):
    def __init__(self, initial_dynamics=None, obstacle_avoider=None):
        super().__init__(initial_dynamics=initial_dynamics, obstacle_avoider=obstacle_avoider)

    def avoid(self, position: np.ndarray, initial_velocity: np.ndarray) -> np.ndarray:
        # TODO include avoid function directly.
        return obs_avoidance_interpolation_moving(
            position=position, initial_velocity=initial_velocity, obs=self._obstacle_environment)
