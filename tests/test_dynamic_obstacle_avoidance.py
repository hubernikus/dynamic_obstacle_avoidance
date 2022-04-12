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


class TestRotationAvoidance(unittest.TestCase):
    def test_single_circle(self, visualize=False):
        obstacle_list = ObstacleContainer()
        obstacle_list.append(
            Ellipse(
                center_position=np.array([0, 0]),
                axes_length=np.array([1, 1]),
            )
        )

        # Arbitrary constant velocity
        initial_velocity = np.array([1, 1])

        if visualize:
            dynamical_system = ConstantValue(velocity=initial_velocity)

            x_lim = [-3, 3]
            y_lim = [-3, 3]
            n_resolution = 10

            Simulation_vectorFields(
                x_lim,
                y_lim,
                n_resolution,
                obstacle_list,
                saveFigure=False,
                noTicks=True,
                showLabel=False,
                draw_vectorField=True,
                dynamical_system=dynamical_system.evaluate,
                obs_avoidance_func=obs_avoidance_interpolation_moving,
                automatic_reference_point=False,
                pos_attractor=None,
                show_streamplot=False,
            )

        # Velocity on surface is tangent after modulation
        position = np.array([-1, 0])
        modualted_velocity = obs_avoidance_interpolation_moving(
            position, initial_velocity, obstacle_list
        )

        normal_dir = obstacle_list[0].get_normal_direction(
            position, in_global_frame=True
        )
        self.assertTrue(np.isclose(np.dot(modualted_velocity, normal_dir), 0))

        # Point far away has no/little influence
        position = np.array([1e10, 0])
        modualted_velocity = obs_avoidance_interpolation_moving(
            position, initial_velocity, obstacle_list
        )
        self.assertTrue(np.allclose(initial_velocity, modualted_velocity))

        # Decreasing influence with decreasing distance
        position = np.array([-1, 0.1])
        mod_vel = obs_avoidance_interpolation_moving(
            position, initial_velocity, obstacle_list
        )
        mod_vel1 = mod_vel / LA.norm(mod_vel)

        position = np.array([-2, 0.1])
        mod_vel = obs_avoidance_interpolation_moving(
            position, initial_velocity, obstacle_list
        )
        mod_vel2 = mod_vel / LA.norm(mod_vel)

        position = np.array([-5, 0.1])
        mod_vel = obs_avoidance_interpolation_moving(
            position, initial_velocity, obstacle_list
        )
        mod_vel3 = mod_vel / LA.norm(mod_vel)

        position = np.array([-10, 0.1])
        mod_vel = obs_avoidance_interpolation_moving(
            position, initial_velocity, obstacle_list
        )
        mod_vel4 = mod_vel / LA.norm(mod_vel)

        # Decreasing influence -> closer to 0 [without magnitude]
        velocity = initial_velocity
        self.assertTrue(np.dot(mod_vel1, velocity) < np.dot(mod_vel2, velocity))
        self.assertTrue(np.dot(mod_vel2, velocity) < np.dot(mod_vel3, velocity))
        self.assertTrue(np.dot(mod_vel3, velocity) < np.dot(mod_vel4, velocity))

    def test_double_ellipse(self):
        obstacle_list = ObstacleContainer()
        obstacle_list.append(
            Ellipse(
                center_position=np.array([0, 0]),
                axes_length=np.array([1, 1]),
                name="center_ellipse",
            )
        )

        obstacle_list.append(
            Ellipse(
                center_position=np.array([1, 0]),
                axes_length=np.array([2, 1]),
                orientation=30 / 180.0 * pi,
            )
        )

        # Arbitrary constant velocity
        initial_velocity = np.array([1, 1])

        # Random evaluation
        position = np.array([-4, 2])
        modualted_velocity = obs_avoidance_interpolation_moving(
            position, initial_velocity, obstacle_list
        )

        # Normal in either case
        position = np.array([-1, 0])
        modualted_velocity = obs_avoidance_interpolation_moving(
            position, initial_velocity, obstacle_list
        )

        normal_dir = obstacle_list[0].get_normal_direction(
            position, in_global_frame=True
        )
        self.assertTrue(np.isclose(np.dot(modualted_velocity, normal_dir), 0))


if (__name__) == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

    visualize = False
    if visualize:
        my_tester = TestRotationAvoidance()
        my_tester.test_single_circle(visualize=True)
