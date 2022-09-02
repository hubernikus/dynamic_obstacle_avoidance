"""
Rotational Avoider
"""
# Author: Lukas Huber
# GitHub: hubernikus
# Created: 2022-04-21

from abc import ABC, abstractmethod


class BaseAvoider(ABC):
    """BaseAvoider which Allow the Evaluate"""

    def __init__(self, initial_dynamics, obstacle_list):
        self.initial_dynamics = initial_dynamics
        self.obstacle_list = obstacle_list

    @property
    def attractor(self):
        return self.initial_dynamics.attractor

    def evaluate(self, position):
        initial_velocity = self.initial_dynamics.evaluate(position)
        return self.avoid(position, initial_velocity, self.obstacle_list)

    @abstractmethod
    def avoid(self, position, initial_velocity, obstacle_list):
        pass
