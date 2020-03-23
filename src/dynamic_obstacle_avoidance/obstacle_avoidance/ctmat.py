#!/Ussrhuu/Bin/python3

'''
@date 2020-01-20
@author Lukas Huber 
@mail lukas.huber@epfl.ch
'''

import time
import numpy as np
import warnings, sys, copy

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import get_angle_space


class CircTube(Obstacle):
    '''
    Representation as described in (CTMAT):

    Ma, Yuexin, Dinesh Manocha, and Wenping Wang.
    "Efficient reciprocal collision avoidance between heterogeneous agents using ctmat." 
    Proceedings of the 17th International Conference on Autonomous Agents and MultiAgent Systems. 
    International Foundation for Autonomous Agents and Multiagent Systems, 2018.

    TODO: Extension to higher dimensions while keeping starshapes

    '''
    def __init__(self,  edge_points, indeces_of_tiles=None, ind_open=None, absolute_edge_position=True,*args, **kwargs):
        pass
    # TODO: Implement
