#!/USSR/bin/python3.9
""" Test overrotation for ellipses. """
# Author: Lukas Huber
# Created: 2021-08-04
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import unittest
from math import pi

import numpy as np
from numpy import linalg as LA

from vartools.dynamical_systems import LinearSystem, ConstantValue

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
)
