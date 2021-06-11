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
    def plottest_list_simple(cls):
        """ Additional test for visualization. """
        
        import matplotlib.pyplot as plt
        from dynamic_obstacle_avoidance.visualization import plot_obstacles
        plt.close('all')

        obs_list = MultiBoundaryContainer()
        obs_list.append(
            Ellipse(
            center_position=np.array([0, 0]), 
            axes_length=np.array([5, 2]),
            orientation=-50./180*pi,
            is_boundary=True,
            )
        )
        obs_list.append(
            Ellipse(
            center_position=np.array([6, 0]), 
            axes_length=np.array([5, 2]),
            orientation=50./180*pi,
            is_boundary=True,
            )
        )

        position_list = [
            np.array([2, -2]),
            np.array([0, -2]),
            np.array([3, -2]),
            np.array([-2, 2]),
            ]
        n_tests = len(position_list)
        
        fig, axs = plt.subplots(1, n_tests, figsize=(14, 5))
        
        for ii in range(n_tests):
            ax = axs[ii]
            plot_obstacles(ax=ax, obs=obs_list, x_range=[-6, 10], y_range=[-6, 6], showLabel=False)
            position = position_list[ii]
            obs_list.update_relative_reference_point(position)
            
            ax.plot(position[0], position[1], 'ko')

            for oo in range(len(obs_list)):
                abs_ref_point = obs_list[oo].global_reference_point
                ax.plot(abs_ref_point[0], abs_ref_point[1], 'k+')

                ref_point = obs_list[oo].global_relative_reference_point

                ax.plot(ref_point[0], ref_point[1], 'k*')
                ax.plot([abs_ref_point[0], ref_point[0]],
                         [abs_ref_point[1], ref_point[1]], 'k--')

    @classmethod
    def plottest_list_advanced(cls):
        """ Additional test for visualization. """
        
        import matplotlib.pyplot as plt
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        obs_list = MultiBoundaryContainer()
        obs_list.append(
            Ellipse(
            center_position=np.array([0, 0]), 
            axes_length=np.array([6, 2]),
            orientation=-40./180*pi,
            is_boundary=True,
            )
        )
        obs_list.append(
            Ellipse(
            center_position=np.array([5, 0]), 
            axes_length=np.array([6, 2]),
            orientation=40./180*pi,
            is_boundary=True,
            )
        )

        position_list = [
            np.array([4, -2]),
            np.array([4, -4]),
            np.array([3.458, -3.1948]),
            ]
        n_tests = len(position_list)
        
        fig, axs = plt.subplots(1, n_tests, figsize=(14, 5))
        
        for ii in range(n_tests):
            ax = axs[ii]
            plot_obstacles(ax=ax, obs=obs_list, x_range=[-6, 10], y_range=[-6, 6], showLabel=False)
            position = position_list[ii]
            obs_list.update_relative_reference_point(position)
            
            ax.plot(position[0], position[1], 'ko')

            for oo in range(len(obs_list)):
                abs_ref_point = obs_list[oo].global_reference_point
                ax.plot(abs_ref_point[0], abs_ref_point[1], 'k+')

                ref_point = obs_list[oo].global_relative_reference_point

                ax.plot(ref_point[0], ref_point[1], 'k*')
                ax.plot([abs_ref_point[0], ref_point[0]],
                         [abs_ref_point[1], ref_point[1]], 'k--')


    @classmethod
    def plottest_list_intersect(cls):
        """ Additional test for visualization. """
        
        import matplotlib.pyplot as plt
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        obs_list = MultiBoundaryContainer()
        obs_list.append(
            Ellipse(
            center_position=np.array([0, 0]), 
            axes_length=np.array([6, 3]),
            orientation=-45./180*pi,
            is_boundary=True,
            )
        )
        obs_list.append(
            Ellipse(
            center_position=np.array([0, 0]), 
            axes_length=np.array([6, 3]),
            orientation=45./180*pi,
            is_boundary=True,
            )
        )

        position_list = [
            # np.array([2.57, -1.20]),
            np.array([-2.57, -1.20]),
            # np.array([-1.20, 2.57]),
            # np.array([4, -2]),
            np.array([0.0245, -3.31]),
            ]
        n_tests = len(position_list)
        
        fig, axs = plt.subplots(1, n_tests, figsize=(14, 5))
        
        for ii in range(n_tests):
            try:
                ax = axs[ii]
            except TypeError:
                # If it's a single element; not a list.
                ax = axs 
                
            plot_obstacles(ax=ax, obs=obs_list, x_range=[-7, 7], y_range=[-6, 6], showLabel=False)
            position = position_list[ii]
            obs_list.update_relative_reference_point(position)
            
            ax.plot(position[0], position[1], 'ko')

            for oo in range(len(obs_list)):
                abs_ref_point = obs_list[oo].global_reference_point
                ax.plot(abs_ref_point[0], abs_ref_point[1], 'k+')

                ref_point = obs_list[oo].global_relative_reference_point

                ax.plot(ref_point[0], ref_point[1], 'k*')
                ax.plot([abs_ref_point[0], ref_point[0]],
                         [abs_ref_point[1], ref_point[1]], 'k--')


if __name__ == '__main__':
    # Allow running in ipython (!)
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)
    # unittest.main()

    visualize = True
    if visualize:
        TestMultiBoundary.plottest_list_simple()
        TestMultiBoundary.plottest_list_advanced()
        TestMultiBoundary.plottest_list_intersect()
