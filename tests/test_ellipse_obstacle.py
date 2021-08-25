"""
Test script for obstacle avoidance algorithm
Test normal formation
"""
import unittest

import numpy as np
from math import pi

from dynamic_obstacle_avoidance.containers import GradientContainer
from dynamic_obstacle_avoidance.obstacles import Ellipse, CircularObstacle
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import Simulation_vectorFields

from scipy.spatial.transform import Rotation as Rotation


class TestEllipses(unittest.TestCase):
    def test_ellipse_reference_point_inside(self, visualize=False):
        """ Visualize ellipse with reference point far away """
        # Create Ellipse
        obs = Ellipse(
            axes_length=[1.2, 2.0],
            center_position=[0.0, 0.0],
            orientation=30./180*pi,
        )

        # Reset reference point
        obs.set_reference_point(np.array([1, 0.3]), in_global_frame=True)

        self.assertTrue(obs.get_gamma(obs.reference_point)<1)
                            

        obs_list = GradientContainer()
        obs_list.append(obs)
    
        if visualize:
            Simulation_vectorFields(
                x_range=[-3, 3], y_range=[-3, 3],
                point_grid=100,
                obs=obs_list,
                draw_vectorField=False,
                automatic_reference_point=False,
            )
        
    def test_visualization_ellipse_with_ref_point_outside(self, visualize=False):
        """ Visualize ellipse with reference point far away """
        # Create Ellipse
        obs = Ellipse(
            axes_length=[1.2, 2.0],
            center_position=[0.0, 0.0],
            orientation=30./180*pi,
        )

        # Set reference point outside
        obs.set_reference_point(np.array([2, 1]), in_global_frame=True)

        obs_list = GradientContainer()
        obs_list.append(obs)

        if visualize:
            Simulation_vectorFields(
                x_range=[-3, 3], y_range=[-3, 3],
                point_grid=0,
                obs=obs_list,
                draw_vectorField=False,
                automatic_reference_point=False,
            )

    def test_visualization_circular_reference_point_outside(self, visualize=False):
        """ Visualize circular-obstacle with reference point far away """
        obs = CircularObstacle(
            radius=1.5,
            center_position=[0.0, 0.0],
            orientation=0./180*pi,
        )

        obs.set_reference_point(np.array([1.2, 1.9]), in_global_frame=True)

        obs_list = GradientContainer()
        obs_list.append(obs)

        if visualize:
            Simulation_vectorFields(
                x_range=[-3, 3], y_range=[-3, 3],
                point_grid=0,
                obs=obs_list,
                draw_vectorField=False,
                automatic_reference_point=False,        
            )

    def test_creation_3d(self):
        ObstacleEnvironment = GradientContainer()
        ObstacleEnvironment.append(
            Ellipse(
            center_position=np.array([0.5, -1, 0.3]), 
            axes_length=np.array([0.3, 0.3, 0.3]),
            orientation=Rotation.from_rotvec([0, 0, 0]),
            tail_effect=False,
            )
        )

    
if (__name__) == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

    visualize = False
    if visualize:
        Tester = TestEllipses()
        Tester.test_visualization_ellipse_with_ref_point_outside(visualize=visualize)
        Tester.test_ellipse_reference_point_inside(visualize=visualize)
        Tester.test_visualization_circular_reference_point_outside(
            visualize=visualize)
