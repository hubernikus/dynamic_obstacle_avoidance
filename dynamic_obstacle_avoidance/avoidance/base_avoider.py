"""
Rotational Avoider
"""
# Author: Lukas Huber
# GitHub: hubernikus
# Created: 2022-04-21

from abc import ABC, abstractmethod


class BaseAvoider(ABC):
    """BaseAvoider which Allow the Evaluate"""

    def __init__(self, obstacle_environment, initial_dynamics=None):
        self.initial_dynamics = initial_dynamics
        self.obstacle_environment = obstacle_environment

    @property
    def attractor(self):
        return self.initial_dynamics.attractor

    def evaluate(self, position):
        if self.initial_dynamics is None:
            raise NotImplementedError("You have to define 'initial_dynamics' first.")
        
        velocity = self.initial_dynamics.evaluate(position)
        return self.avoid(position, velocity, self.obstacle_list)

    @abstractmethod
    def avoid(self, position, velocity):
        pass
