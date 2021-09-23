#!/USSR/bin/python3.9
""" Test overrotation for ellipses. """
# Author: Lukas Huber
# Created: 2021-08-04
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import unittest
from math import pi

import numpy as np
# from numpy import linalg as LA

from vartools.dynamical_systems import LinearSystem, SinusAttractorSystem
from vartools.dynamical_systems import ConstVelocityDecreasingAtAttractor

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.containers import MultiBoundaryContainer

# Rename when moving to different library...
from dynamic_obstacle_avoidance.avoidance.multihull_convergence import get_desired_radius, multihull_attraction


class TestMultiHullBehvior(unittest.TestCase):
    def test_creation(self):
        dim = 2
            
        limiter = ConstVelocityDecreasingAtAttractor(const_velocity=1.0, distance_decrease=1.0)
        initial_dynamics = SinusAttractorSystem(trimmer=limiter, attractor_position=np.zeros(dim))

        obstacle_list = MultiBoundaryContainer()
        obstacle_list.append(
            Ellipse(
            center_position=np.array([-1, -1]), 
            axes_length=np.array([2.5, 1.5]),
            orientation=40./180*pi,
            is_boundary=True,
            )
        )
        obstacle_list.append(
            Ellipse(
            center_position=np.array([-3, -1]), 
            axes_length=np.array([2.5, 1.5]),
            orientation=-40./180*pi,
            is_boundary=True,
            )
        )

        convering_dynamics = LinearSystem(
            attractor_position=initial_dynamics.attractor_position, maximum_velocity=0.5)

        # breakpoint()
        obstacle_list.set_convergence_directions(convering_dynamics)

        # position = np.array([-7.06896552,  1.24137931])
        position = np.array([-5.80, 0.96])

        initial_velocity = initial_dynamics.evaluate(position)
        rotated_velocity = multihull_attraction(position, initial_velocity, obstacle_list)
        

if (__name__) == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

    manual_test = False
    if manual_test:
        my_tester = TestMultiHullBehvior()
        my_tester.test_creation()
