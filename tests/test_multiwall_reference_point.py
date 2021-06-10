#!/USSR/bin/python3.9
""" Test the directional space. """

__author__ = "LukasHuber"
__date__ = "2021-05-18"
__email__ = "lukas.huber@epfl.ch"

import unittest
from math import pi

import numpy as np

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.obstacles import MultiBoundaryContainer

class TestMultiBoundary(unittest.TestCase):
    def test_creating(self):
        """ Cretion & adapation of MultiWall-Surrounding """
        obs_list = MultiBoundaryContainer()

        obs_list.append(
            Ellipse(
            center_position=np.array([-6, 0]), 
            axes_length=np.array([5, 2]),
            orientation=50./180*pi,
            is_boundary=True,
            )
        )
        obs_list.append(
            Ellipse(
            center_position=np.array([0, 0]), 
            axes_length=np.array([5, 2]),
            orientation=-50./180*pi,
            is_boundary=True,
            )
        )

        # Save not possible for further use (...)

        
    def test_displacement(self):
        """ """
        self.assertTrue(True)

    @classmethod
    def plottest_list(cls):
        """ Additional test for visualization. """
        
        import matplotlib.pyplot as plt
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        obs_list = MultiBoundaryContainer()
        obs_list.append(
            Ellipse(
            center_position=np.array([-6, 0]), 
            axes_length=np.array([5, 2]),
            orientation=50./180*pi,
            is_boundary=True,
            )
        )
        obs_list.append(
            Ellipse(
            center_position=np.array([0, 0]), 
            axes_length=np.array([5, 2]),
            orientation=-50./180*pi,
            is_boundary=True,
            )
        )

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        plot_obstacles(ax=ax, obs=obs_list, x_range=[-10, 5], y_range=[-5, 5])

        position = np.array([-8, -2])

        obs_list.update_relative_reference_point(position)

        plt.plot(position[0], position[1], 'ko')
        for oo in range(len(obs_list)):
            ref_point = obs_list[oo].global_relative_reference_point
            plt.plot(ref_point[0], ref_point[1], 'k*')

if __name__ == '__main__':
    # Allow running in ipython (!)
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)
    # unittest.main()

    visualize = True
    if visualize:
        TestMultiBoundary.plottest_list()
        
