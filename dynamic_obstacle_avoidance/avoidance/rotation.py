""" Library for the Rotation (Modulation Imitation) of Linear Systems
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

# TODO: Speed up using cython / cpp e.g. eigen?
# TODO: list / array stack for lru_cache to speed

import warnings
import copy
from math import pi

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt  # For debugging only (!)

from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_directional_weighted_sum
from vartools.directional_space import (
    get_directional_weighted_sum_from_unit_directions,
)
from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import UnitDirection, DirectionBase
from vartools.dynamical_systems import DynamicalSystem

from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.utils import get_weight_from_inv_of_gamma
from dynamic_obstacle_avoidance.utils import get_relative_obstacle_velocity

from .rotational_avoider import RotationalAvoider


def obstacle_avoidance_rotational(*args, **kwargs):
    warnings.warn("Depreciated - switch to Avoiders.")

    _avoider  = RotationalAvoider()
    return _avoider.avoid(*args, **kwargs)
