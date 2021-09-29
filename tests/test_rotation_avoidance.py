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
from dynamic_obstacle_avoidance.containers import RotationContainer

from dynamic_obstacle_avoidance.avoidance import obstacle_avoidance_rotational


class TestRotationAvoidance(unittest.TestCase):
    def test_single_circle(self):
        obstacle_list = RotationContainer()
        obstacle_list.append(
            Ellipse(
            center_position=np.array([0, 0]), 
            axes_length=np.array([1, 1]),
            ))

        # Arbitrary constant velocity
        initial_velocity = np.array([1, 1])
        
        obstacle_list.set_convergence_directions(
            ConvergingDynamics=ConstantValue(initial_velocity))
        
        # Velocity on surface is tangent after modulation
        position = np.array([-1, 0])
        modualted_velocity = obstacle_avoidance_rotational(
            position, initial_velocity, obstacle_list)
        
        normal_dir = obstacle_list[0].get_normal_direction(position, in_global_frame=True)
        self.assertTrue(np.isclose(np.dot(modualted_velocity, normal_dir), 0))

        # No effect when already pointing away (save circle)
        position = np.array([1, 0])
        modualted_velocity = obstacle_avoidance_rotational(
            position, initial_velocity, obstacle_list)
        self.assertTrue(np.allclose(initial_velocity, modualted_velocity))

        # Point far away has no/little influence
        position = np.array([1e10, 0])
        modualted_velocity = obstacle_avoidance_rotational(
            position, initial_velocity, obstacle_list)
        self.assertTrue(np.allclose(initial_velocity, modualted_velocity))

        # Decreasing influence with decreasing distance
        position = np.array([-1, 0.1])
        mod_vel = obstacle_avoidance_rotational(position, initial_velocity, obstacle_list)
        mod_vel1 = mod_vel/LA.norm(mod_vel)
            
        position = np.array([-2, 0.1])
        mod_vel = obstacle_avoidance_rotational(position, initial_velocity, obstacle_list)
        mod_vel2 = mod_vel/LA.norm(mod_vel)
        
        position = np.array([-5, 0.1])
        mod_vel = obstacle_avoidance_rotational(position, initial_velocity, obstacle_list)
        mod_vel3 = mod_vel/LA.norm(mod_vel)
        
        position = np.array([-10, 0.1])
        mod_vel = obstacle_avoidance_rotational(position, initial_velocity, obstacle_list)
        mod_vel4 = mod_vel/LA.norm(mod_vel)

        # Decreasing influence -> closer to 0 [without magnitude]
        velocity = initial_velocity
        self.assertTrue(np.dot(mod_vel1, velocity) < np.dot(mod_vel2, velocity))
        self.assertTrue(np.dot(mod_vel2, velocity) < np.dot(mod_vel3, velocity))
        self.assertTrue(np.dot(mod_vel3, velocity) < np.dot(mod_vel4, velocity))


    def test_double_ellipse(self):
        obstacle_list = RotationContainer()
        obstacle_list.append(
            Ellipse(
            center_position=np.array([0, 0]), 
            axes_length=np.array([1, 1]),
            name="center_ellipse"
            ))

        obstacle_list.append(
            Ellipse(
            center_position=np.array([1, 0]), 
            axes_length=np.array([2, 1]),
            orientation=30/180.*pi
            ))

        # Arbitrary constant velocity
        initial_velocity = np.array([1, 1])
        
        obstacle_list.set_convergence_directions(
            ConvergingDynamics=ConstantValue(initial_velocity))

        # Random evaluation
        position = np.array([-4, 2])
        modualted_velocity = obstacle_avoidance_rotational(
            position, initial_velocity, obstacle_list)

        # Normal in either case
        position = np.array([-1, 0])
        modualted_velocity = obstacle_avoidance_rotational(
            position, initial_velocity, obstacle_list)
        
        normal_dir = obstacle_list[0].get_normal_direction(position, in_global_frame=True)
        self.assertTrue(np.isclose(np.dot(modualted_velocity, normal_dir), 0))
    
        
if (__name__) == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
