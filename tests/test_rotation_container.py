#!/USSR/bin/python3.9
"""
Testing script for (python) obstacle rotation
"""
# Author: Lukas Huber
# License: BSD (c) 2021

import unittest
from math import pi

import numpy as np
import matplotlib.pyplot as plt

from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.containers import RotationContainer
from dynamic_obstacle_avoidance.obstacles import Ellipse

class TestRotational(unittest.TestCase):
    def test_zero_rotation_container(
        self, visualize_plot=False, assert_check=True, num_resolution=30,
        x_range=[-10, 10], y_range=[-10, 10], dim=2):

        obstacle_list = RotationContainer()
        obstacle_list.append(
            Ellipse(
            center_position=np.array([0, 0]), 
            axes_length=np.array([2, 5]),
            orientation=0./180*pi,
            tail_effect=False,
            )
        )
        InitialSystem = LinearSystem(attractor_position=np.array([8, 0]))
        # Convergence direction of a Linear System is equal to 0
        obstacle_list.set_convergence_directions(InitialSystem)
        
        x_vals = np.linspace(x_range[0], x_range[1], num_resolution)
        y_vals = np.linspace(y_range[0], y_range[1], num_resolution)

        positions = np.zeros((dim, num_resolution, num_resolution))
        conv_dir = np.zeros((dim, num_resolution, num_resolution))
        
        for ix in range(num_resolution):
            for iy in range(num_resolution):
                positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]

                conv_dir[:, ix, iy] = obstacle_list.get_convergence_direction(positions[:, ix, iy], it_obs=0)

                # TODO: check edge / boundary case
                if assert_check:
                    dir_pos = InitialSystem.attractor_position-positions[:, ix, iy]
                    dot_prod = np.dot(conv_dir[:, ix, iy], dir_pos)
                    dot_prod = dot_prod / (np.linalg.norm(conv_dir[:, ix, iy]) * np.linalg.norm(dir_pos))
                    # print(f'{dot_prod=}')
                    self.assertTrue(abs(dot_prod - 1) < 1e-6, "DS not equal not linear one.")

        if visualize_plot:
            plt.quiver(positions[0, :, :], positions[1, :, :],
                       conv_dir[0, :, :], conv_dir[1, :, :], color="blue")


if (__name__)=="__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    plot_results = False
    if plot_results:
        MyTester = TestRotational()
        MyTester.test_zero_rotation_container(visualize_plot=True, assert_check=True)


