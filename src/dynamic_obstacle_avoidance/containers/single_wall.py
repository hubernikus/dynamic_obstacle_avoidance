# Author Lukas Huber
# Mail lukas.huber@epfl.ch
# Created 2021-06-22
# License: BSD (c) 2021

import time
import numpy as np
import copy
from math import pi
import warnings, sys

from vartools.angle_math import *

from dynamic_obstacle_avoidance.utils import *
from dynamic_obstacle_avoidance.avoidance.obs_common_section import (
    Intersection_matrix,
)
from dynamic_obstacle_avoidance.avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.avoidance.obs_dynamic_center_3d import *

from dynamic_obstacle_avoidance.containers import BaseContainer


class SingleWallContainer(BaseContainer):
    def __init__(self, obs_list=None):
        # self.index_wall = None

        self._obstacle_list = []

        self.contains_wall_obstacle = False

        if isinstance(obs_list, (list, BaseContainer)):
            for ii in range(len(obs_list)):
                self.append(obs_list[ii])
                # if not self.index_wall is None:
                # warnings.warn("Several boundary obstacles in one container.")
                # self.index_wall = ii

    def __delitem__(self, key):
        """Obstacle is not part of the workspace anymore."""
        if key == len(self) - 1:
            self.contains_wall_obstacle = False

        del self._obstacle_list[key]

        # if not self.index_wall is None:
        # if self.index_wall>key:
        # self.index_wall -= 1
        # elif self.index_wall==key:
        # self.index_wall = None

    def append(self, value):  # Compatibility with normal list.
        """Add new elements to obstacles list. The wall obstacle is placed last."""
        if self.contains_wall_obstacle:
            if value.is_boundary:
                raise RuntimeError("Obstacles container already has a wall!.")

            self._obstacle_list.insert(len(self._obstacle_list) - 1, value)
        else:
            if value.is_boundary:
                self.contains_wall_obstacle = True
            self._obstacle_list.append(value)

    @property
    def index_wall(self):
        """Wall obstacles are placed at the end of the list."""
        if self.contains_wall_obstacle:
            return len(self._obstacle_list) - 1
        else:
            return None

    @property
    def has_wall(self):
        return self.has_environment and self._obstacle_list[-1].is_boundary

    def delete_boundary(self):
        boundary_succesfully_deleted = False

        if self.has_wall:
            del self._obstacle_list[-1]

            boundary_succesfully_deleted = True

        return boundary_succesfully_deleted
