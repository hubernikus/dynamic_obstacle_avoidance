"""
Test script for obstacle avoidance algorithm
Test normal formation
"""
import unittest

import numpy as np
from math import pi

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import FlatPlane
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from scipy.spatial.transform import Rotation as Rotation


class TestFlatPlane(unittest.TestCase):
    def test_flat_plane(self, visualize=False):
        """Visualize FlatPlane with reference point far away"""
        x_lim = [-3, 3]
        y_lim = [-3, 3]

        # Create Ellipse
        obs_container = ObstacleContainer()
        obs_container.append(FlatPlane(center_position=[0.0, 0.0], normal=[0, 1]))

        if visualize:
            fig, ax = plt.subplots(figsize=(8, 10))

            obs_container[-1].draw_obstacle()
            plot_obstacles(ax, obs_container, x_lim, y_lim)


if (__name__) == "__main__":
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)

    visualize = True
    if visualize:
        my_tester = TestFlatPlane()
        my_tester.test_flat_plane(visualize=visualize)
