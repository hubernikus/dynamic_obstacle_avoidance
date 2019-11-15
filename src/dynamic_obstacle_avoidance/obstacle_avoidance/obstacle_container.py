#!/USSR/bin/python3
'''
@date 2019-10-15
@author Lukas Huber 
@mail lukas.huber@epfl.ch
'''

import time
import numpy as np
from math import sin, cos, pi, ceil
import warnings, sys

import numpy.linalg as LA

from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import *

from dynamic_obstacle_avoidance.obstacle_avoidance.state import *
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_dynamic_center_3d import *

import matplotlib.pyplot as plt

visualize_debug = False

class ObstacleContainer(State):
    # Contains properties of the obstacle environment. Which require centralized handling
    # TODO: Update this in a smart way
    # TODO: how much in obstacle class, how much in container?
    
    def __init__(self, obs_list=None):
        if isinstance(obs_list, list):
            self._obstacle_list = obs_list
        else:
            self._obstacle_list = []
            
        self._index_families = None
        self._unique_families = None
        self._rotation_direction = None

    def reset_clusters(self):
        self.find_hirarchy()
        # self.get_sibling_groups()
        self.reset_rotation_direction()
        
    def reset_rotation_direction(self):
        self._rotation_direction =  np.zeros(self._unique_families.shape)
        # self._outside_influence_region = np.ones(self._unique_families.shape, dtype=bool)
        self._outside_influence_region = np.ones(self._index_families.shape, dtype=bool)

    def __repr__(self):
        return "ObstacleContainer of length #{}".format(len(self))
        
    def __len__(self):
        return len(self._obstacle_list)

    @ property
    def number(self):
        return len(self._obstacle_list)

    def __getitem__(self, key):
        return self._obstacle_list[key]

    def __setitem__(self, key, value):
        self._obstacle_list[key] = value
        
    def __delitem__(self, key):
        del self._obstacle_list[key]

    def append(self, value):
        self._obstacle_list.append(value)

    def find_hirarchy(self):
        intersecting_obs = obs_common_section(self, update_reference_point=False)

        self._index_families = np.ones(len(self))*(-1)
        
        for ii in range(len(intersecting_obs)):
            self._index_families[intersecting_obs[ii]] = ii

        ind_singles = self._index_families == (-1)

        if np.sum(ind_singles): # nonzero
            self._index_families[ind_singles] = np.arange(len(intersecting_obs),
                                                          len(intersecting_obs)+np.sum(ind_singles))

        self._unique_families = np.unique(self.index_families)
        
        # dynamic_center_3d(self, intersecting_obs)

    @property
    def dim(self):
        return self._obstacle_list[0].dim
    
    @property
    def list(self):
        return self._obstacle_list

    @property
    def index_families(self):
        if isinstance(self._index_families, type(None)):
            self.get_sibling_groups()
        return self._index_families

    def get_siblings(self, index):
        if isinstance(self._index_families, type(None)):
            self.get_sibling_groups()

        index_family_ii = (index==self._index_families)
        return np.arange(len(self))[index_family_ii]
    
    def get_sibling_groups(self):
        # TODO: evaluate what to store in container
        hirarchy_obs = np.array([self._obstacle_list[ii].hirarchy for ii in range(len(self))])
        index_to_parents = np.array([self._obstacle_list[ii].ind_parent for ii in range(len(self))])

        import pdb; pdb.set_trace() ## DEBUG ##
        
        ind_roots = (hirarchy_obs==0)
        num_families = np.sum(ind_roots)
        
        self._index_families = index_to_parents
        if not num_families:
            raise ValueError("No obstacle root detected.")
            
        self._index_families[ind_roots] = np.arange(num_families)

        for ii in range(max(hirarchy_obs)-1, 0, -1):
            self._index_families[hiarchy_obs==ii] = _index_families[_index_families[hiarchy_obs==ii]]
        self._unique_families = np.unique(self._index_families)
        self._family_references = np.zeros((self.dim, self._unique_families.shape[0]))

        return self._index_families

    def get_rotation_direction(self, index):
        if isinstance(self._rotation_direction, type(None)):
            self._rotation_direction = np.zeros(self._unique_families.shape)
        value = self._rotation_direction[self._unique_families==self._index_families[index]]
        return value

    # TODO: make getter and setter of sub-list
    def is_outside_influence_region(self, index):
        if not hasattr(self, '_outside_influence_region'):
            self._outside_influence_region = np.ones(self._unique_families.shape, dtype=bool)
        return self._outside_influence_region[self._unique_families==self._index_families[index]]
    
    def set_is_outside_influence_region(self, index, value=None):
        if not hasattr(self, '_outside_influence_region'):
            self._outside_influence_region = np.ones(self._unique_families.shape, dtype=bool)
        # self._outside_influence_region[self._unique_families==self._index_families[index]] = value
        self._outside_influence_region[index] = value
            
    def set_rotation_direction(self, index, value):
        if not hasattr(self, '_rotation_direction'):
            self._rotation_direction = np.zeros(self._unique_families.shape)
            self._outside_influence_region = np.ones(self._unique_families.shape, dtype=bool)
            
        self._rotation_direction[self._unique_families==self._index_families[index]] = value
        # self._outside_influence_region[self._unique_families==self._index_families[index]] = False
