"""
Container to describe obstacles & wall environemnt.
"""
# Author Lukas Huber
# Mail lukas.huber@epfl.ch
# Created 2021-06-22
# License: BSD (c) 2021
from abc import ABC, abstractmethod

import numpy as np
import warnings

from dynamic_obstacle_avoidance.utils import *


class BaseContainer(ABC):
    def __init__(self, obs_list=None):
        self._obstacle_list = []

        if isinstance(obs_list, (list, BaseContainer)):
            self._obstacle_list = obs_list

    def __getitem__(self, key):
        """List-like or dictionarry-like access to obstacle"""
        # TODO: can this be done more efficiently?
        if isinstance(key, (str)):
            for ii in range(len(self._obstacle_list)):
                if self._obstacle_list[ii].name == key:
                    return self._obstacle_list[ii]
            raise ValueError("Obstacle <<{}>> not in list.".format(key))
        else:
            return self._obstacle_list[key]

    def __setitem__(self, key, value):
        self._obstacle_list[key] = value

    def append(self, value):  # Compatibility with normal list.
        """Add new elements to obstacles list. The wall obstacle is placed last."""
        self._obstacle_list.append(value)

    def __delitem__(self, key):
        """Obstacle is not part of the workspace anymore."""
        del self._obstacle_list[key]

    def add_obstacle(self, value):
        self.append(value)

    def __iter__(self):
        return iter(self._obstacle_list)

    def __repr__(self):
        return "ObstacleContainer of length #{}".format(len(self))

    def __str__(self):
        return "ObstacleContainer of length #{}".format(len(self))

    def __len__(self):
        return len(self._obstacle_list)

    @property
    def boundary(self):
        return self._obstacle_list[-1]

    @property
    def number(self):
        return len(self)

    @property
    def n_obstacles(self):
        return len(self)

    @property
    def dimension(self):
        # Dimension of all obstacles is expected to be equal
        return self._obstacle_list[0].dim

    @property
    def dim(self):
        # Dimension of all obstacles is expected to be equal
        return self._obstacle_list[0].dim

    @property
    def list(self):
        return self._obstacle_list

    @property
    def has_environment(self):
        return bool(len(self))

    def check_collision(self, position: np.ndarray) -> bool:
        """Returns collision with environment (type Bool)

        Convention for this model is that:
        > Obstacles are mutually additive, i.e. no collision with any obstacle
        > Boundaries are mutually subractive, i.e. collision free with at least one boundary.
        """
        gamma_list_boundary = []
        for oo in range(self.n_obstacles):
            gamma = self[oo].get_gamma(position, in_global_frame=True)

            if self[oo].is_boundary:
                gamma_list_boundary.append(gamma)

            elif gamma <= 1:
                # Collided with an obstacle
                return True

        if len(gamma_list_boundary):
            # At least one boundary
            return all(np.array(gamma_list_boundary) <= 1)
        else:
            return False

    def check_collision_array(self, positions: np.ndarray) -> np.ndarray:
        """Return array of checked collisions of type bool."""
        collision_array = np.zeros(positions.shape[1], dtype=bool)
        for it in range(positions.shape[1]):
            collision_array[it] = self.check_collision(positions[:, it])
        return collision_array
