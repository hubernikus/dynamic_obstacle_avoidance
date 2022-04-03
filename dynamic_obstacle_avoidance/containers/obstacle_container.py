"""
Container which Contains OBstacles
"""
# Author Lukas Huber
# Mail lukas.huber@epfl.ch
# Created 2021-06-22
# License: BSD (c) 2021

import time
import numpy as np
import copy
from math import pi
import warnings, sys

import matplotlib.pyplot as plt

from vartools.angle_math import *

from dynamic_obstacle_avoidance.utils import *
from dynamic_obstacle_avoidance.avoidance.obs_common_section import (
    Intersection_matrix,
)
from dynamic_obstacle_avoidance.avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.avoidance.obs_dynamic_center_3d import *

from dynamic_obstacle_avoidance.containers import BaseContainer


class ObstacleContainer(BaseContainer):
    # class ObstacleContainer(list): # ?? inherit from list?
    # List like obstacle container
    # Contains properties of the obstacle environment. Which require centralized handling

    # TODO: Update this in a smart way
    # TODO: how much information/treatement in the obstacle class, how much in container?
    def __init__(self, obs_list=None):
        if sys.version_info > (3, 0):
            super().__init__(obs_list)
        else:  # Python 2
            super(ObstacleContainer, self).__init__(
                obs_list
            )  # works for python < 3.0?!

        self._family_label = None
        self._unique_families = None
        self._rotation_direction = None

        if len(self) == 0:
            self._boundary_reference_points = None
            self._distance_matrix = None
        else:
            self._boundary_reference_points = np.zeros((2, len(self), len(self)))
            self._distance_matrix = DistanceMatrix(n_obs=self.number)

    def do_velocity_step(self, delta_time: float) -> None:
        for obs in self._obstacle_list:
            obs.do_velocity_step(delta_time)

    def reset_clusters(self):
        self.get_sibling_groups()
        # self.get_sibling_groups()
        self.reset_rotation_direction()

    def reset_rotation_direction(self):
        self._rotation_direction = np.zeros(self._unique_families.shape)
        # self._outside_influence_region = np.ones(self._unique_families.shape, dtype=bool)
        self._outside_influence_region = np.ones(self._family_label.shape, dtype=bool)

    def __getitem__(self, key):
        """List-like or dictionarry-like access to obstacle"""
        if isinstance(key, (str)):
            for ii in range(len(self._obstacle_list)):
                if self._obstacle_list[ii].name == key:
                    return self._obstacle_list[ii]
            raise ValueError("Obstacle <<{}>> not in list.".format(key))
        else:
            return self._obstacle_list[key]

    def __setitem__(self, key, value):
        # Is this useful?
        self._obstacle_list[key] = value

        for jj in range(self.number):
            if jj == key:
                continue
            self._distance_matrix[jj, key] = None

    def reset_intersections(self, index=None):
        if index is None:
            self._distance_matrix = Intersection_matrix(
                len(self._obstacle_list), self.dimension
            )
        else:
            for ii in range(self.number):
                if ii == index:
                    continue
                self._distance_matrix[index, ii] = None

    def get_distance(self, index1=None, index2=None):
        if index1 is None:
            return self._distance_matrix
        elif not index2 is None:
            return self._distance_matrix[index1, index2]
        else:
            raise ValueError("Wrong number of arguments.")

    def find_root(self):
        ind_parents = np.array([self[ii].ind_parent for ii in range(len(self))])
        hirarchy_max = np.max([self[ii].ind_parent for ii in range(len(self))])

        ind_roots = ind_parents < 0
        ind_parents[ind_roots] = (np.arange(np.sum(ind_roots), dtype=int) + 1) * (-1)

        for ii in range(hirarchy_max):
            ind_nonRoot = ind_parents >= 0

            ind_parents[ind_nonRoot] = ind_parents[ind_parents[ind_nonRoot]]

        self._family_label = ind_parents * (-1) - 1  # Range from 0 to n

    def get_sibling_groups(self):
        (intersecting_obs, self.intersection_matrix,) = obs_common_section_hirarchy(
            self, update_reference_point=False, get_intersection_matrix=True
        )
        self.assign_sibling_groups(intersecting_obs)

    def assign_sibling_groups(self, intersecting_obs):
        """Assign each obstacle to a group label & calculate corresponding label"""
        if True:
            self._family_label = np.ones(len(self)) * (-1)

            it_lablel = 0
            for ii in range(len(intersecting_obs)):
                self._family_label[intersecting_obs[ii]] = ii

            ind_singles = self._family_label == (-1)

            if np.sum(ind_singles):  # nonzero
                self._family_label[ind_singles] = np.arange(
                    len(intersecting_obs),
                    len(intersecting_obs) + np.sum(ind_singles),
                )

            self._family_label = self._family_label.astype(int)

        self.find_root()
        self._unique_families = np.unique(self._family_label)
        self._family_centers = np.zeros((self.dim, self._unique_families.shape[0]))

        center_list = np.array([self[jj].center_position for jj in range(len(self))]).T
        for ii in range(self._family_centers.shape[1]):
            self._family_centers[:, ii] = np.mean(
                center_list[:, (self._family_label == ii)], axis=1
            )

    def get_family_index(self, index):
        # Assumption: _unique_families is sorted
        return self._family_label[index]

    # @property
    # def family_center(self, index):
    # return self._family_centers

    @property
    def family_center(self):
        return self._family_centers

    def get_family_center(self, index):
        return self._family_centers[:, self.get_family_index(index)]

    @property
    def num_families(self):
        return len(self._family_label)

    @property
    def index_family(self):
        # TODO: remove since depreciated
        return self.family_label

    @property
    def family_label(self):
        if isinstance(self._family_label, type(None)):
            self.get_sibling_groups()
        return self._family_label

    def get_siblings_boolIndex(self, index):
        label = self._family_label[index]
        return label == self._family_label

    def get_siblings_number(self, index):
        # if isinstance(self._family_label, type(None)):
        # self.get_sibling_groups()
        return np.arange(len(self), dtype=int)[self.get_siblings_boolIndex(index)]

    def is_outside_influence_region(self, index):
        # TODO: make getter and setter of sub-list
        # if not hasattr(self, '_outside_influence_region'):
        # self._outside_influence_region = np.ones(self._unique_families.shape, dtype=bool)
        return self._outside_influence_region[index]

    def set_is_outside_influence_region(self, index, value=None):
        # if not hasattr(self, '_outside_influence_region'):
        # self._outside_influence_region = np.ones(self._unique_families.shape, dtype=bool)
        # self._outside_influence_region[self._unique_families==self._family_label[index]] = value
        self._outside_influence_region[index] = value

    def set_rotation_direction(self, index, value):
        # if not hasattr(self, '_rotation_direction'):
        # self._rotation_direction = np.zeros(self._unique_families.shape)
        # self._outside_influence_region = np.ones(self._unique_families.shape, dtype=bool)
        # self._rotation_direction[self._unique_families==self._family_label[index]] = value

        self._rotation_direction[self.get_family_index(index)] = value
        # self._outside_influence_region[self._unique_families==self._family_label[index]] = False

    def get_rotation_direction(self, index):
        # if isinstance(self._rotation_direction, type(None)):
        # self._rotation_direction = np.zeros(self._unique_families.shape)
        # value = self._rotation_direction[self._unique_families==self._family_label[index]]
        return self._rotation_direction[self.get_family_index(index)]

    # @property
    # def intersection_position(self, index):
    # if index[0] == index[1]

    def get_relative_angle_to_family(self, ind_newObstacle, position):
        # Angle Windup to Family Member
        # TODO: Debug and verify
        ind_family = self._family_label == self._family_label[ind_newObstacle]

        short_connection = [ind_newObstacle]
        for ii in range(ind_family.shape[0]):
            if not self.is_outside_influence_region(ind_family[ii]):
                ind_short_connection = self.find_shortes_connection(
                    ind_family[ii], ind_newObstacle
                )
                break

        angle_space_difference = 0
        # for ii in range(len(short_connection)-1):
        # for ii in range(1, len(short_connection)-1)):
        # Under the assumption that everything is star-shaped

        basis_direction = np.zeros(self.dim)
        basis_direction[0] = 1
        transform_direction = (
            self[ind_short_connection[0]].global_reference_point - position
        )
        angle_space_difference += get_directional_space(ref_dir, transform_direction)

        for ii in range(0, len(short_connection) - 1):
            basis_direction = transform_direction
            transform_direction = (
                self.intesection_points[
                    ind_short_connection[ii + 1], ind_short_connection[ii]
                ]
                - self[ind_short_connection[ii]].global_reference_point
            )

            angle_space_difference += get_directional_space(
                basis_direction, transform_direction
            )

            basis_direction_ = transform_direction
            transform_dir = (
                self[ind_short_connection[ii]].global_reference_point
                - self.intersection_points[
                    ind_short_connection[ii + 1], ind_short_connection[ii]
                ]
            )
            angle_space_difference += get_directional_space(
                basis_direction, transform_direction
            )

        return angle_space_difference

    def find_shortes_connection(self, ind_start, ind_end):
        # Exploration search
        # TODO: DEBUG AND VERIFY
        # CREATE TESTING FUNCTION!!!!
        it_list = []  # Search it list
        siblings_list = [get_siblings(ind_start)]
        flat_list = [siblings_list]

        path_not_found = True
        it_list = [0 for ii in range(len(it_list) + 1)]  # increment of one
        siblings_list.append(copy.deepcopy(siblings_list[-1]))

        increment_level = -1

        while path_not_found:
            increment_in_process = True
            # while(increment_in_process):

            eval_list = siblings_list[-2]
            # for ii in range(len(it_list)):
            # Increment get evaluation-list
            ii = 0
            while ii < range(len(it_list)):
                # while len(it_list[ii]): #nonzero
                # it_list[ii] += 1
                if ii == increment_level:
                    it_list[ii] += 1

                if it_list[ii] > len(eval_list):
                    it_list[ii] = 0
                    increment_level = ii - 1
                    if increment_level == 0:  # all siblings found
                        siblings_list.append(copy.deepcopy(siblings_list[-1]))
                        it_list.apppend(0)
                        increment_level = len(it_list) - 1
                        ii = 0
                        continue

                    eval_list = siblings_list[-2]
                    ii = 0
                    continue

                eval_list = eval_list[it_list[ii]]
                ii += 1

            increment_level = len(it_list) - 1

            fill_list = siblings_list[-1]
            for ii in range(len(it_list)):
                fill_list = fill_list[it_list[ii]]

            # Fill the list
            for ii in range(1, len(it_list)):
                fill_list = fill_list[it_list[ii]]

                if ind_end in fill_list[ii]:
                    it_list.append(np.array(fill_list[ii] == ind_end)[0])
                    return it_list

                # Check for Duplicates
                jj = 0
                while jj < len(fill_list[-1]):
                    if fill_list[ii] in flat_list:
                        del fill_list[jj]
                        continue
                    jj += 1
                flat_list.append(fill_list[-1])

    def get_siblings(self, ind):
        # TODO: use intersection matrix instead / maybe not
        if self[ind].parent >= 0:
            return np.array([self[ind].parent] + self.ind_children)
        else:
            return np.array(self.ind_children)
