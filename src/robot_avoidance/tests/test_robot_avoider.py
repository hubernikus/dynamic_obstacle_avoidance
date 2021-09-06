#!/USSR/bin/python3.9
"""
Test script for obstacle avoidance algorithm - specifically the normal function evaluation
"""
import unittest
from math import pi

import numpy as np

from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import Sphere
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from robot_avoidance.model_robot import RobotArm2D
from robot_avoidance.robot_arm_avoider import RobotArmAvoider



class TestRobotAvoider(unittest.TestCase):
    def test_evaluation_points(self, visualize=False):
        # Assumption of already created jacobian
        from robot_avoidance.jacobians.robot_arm_3link import _get_jacobian
        
        my_robot = RobotArm2D(link_lengths=np.array([1, 1, 1]))
        my_robot.name = "robot_arm_3link"
        my_robot.set_jacobian(function=_get_jacobian)

        obstacle_environment = ObstacleContainer()
        obstacle_environment.append(
            Sphere(radius=1.0, center_position=np.array([0, 5]))
            )
            
        initial_dynamics = LinearSystem(dimension=2)
        
        my_avoider = RobotArmAvoider(initial_dynamics=initial_dynamics,
                        robot_arm=my_robot,
                        obstacle_environment=obstacle_environment)

        my_robot.set_joint_state(
            np.array([90, 0, 0]),
            input_unit='deg')
        
        evaluation_points, gamma_values = my_avoider.get_influence_weight_evaluation_points()
        
        for pp in range(evaluation_points.shape[1]):
            for jj in range(evaluation_points.shape[2]):
                self.assertTrue(np.isclose(evaluation_points[0, pp, jj], 0))
                self.assertTrue(evaluation_points[1, pp, jj] >= 0)

        # Check if gamma is decreasing in a 'simple' circle sceneario
        gamma_values = gamma_values.T.flatten()
        for ii in range(1, gamma_values.shape[0]):
            self.assertTrue(gamma_values[ii-1] > gamma_values[ii])

        if visualize:
            x_lim = [-0.3, 7]
            y_lim = [-3, 3]
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 7.5))
            
            my_robot.draw_robot(ax=ax)
            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)
            
            ax.set_xlim(y_lim)
            ax.set_ylim(x_lim)
            
            ax.set_aspect('equal', adjustable='box')
            ax.grid()
            # for pp in range(evaluation_points.shape[1]):
            ax.plot(evaluation_points[0, :], evaluation_points[1, :], 'k.')
            

    def test_evaluation_weight(self, visualize=False):
        # Assumption of already created jacobian
        from robot_avoidance.jacobians.robot_arm_3link import _get_jacobian
        
        my_robot = RobotArm2D(link_lengths=np.array([1, 1, 1]))
        my_robot.name = "robot_arm_3link"
        my_robot.set_jacobian(function=_get_jacobian)

        obstacle_environment = ObstacleContainer()
        obstacle_environment.append(
            Sphere(radius=1.0, center_position=np.array([1.5, 2]))
            )
            
        initial_dynamics = LinearSystem(dimension=2)
        
        my_avoider = RobotArmAvoider(initial_dynamics=initial_dynamics,
                        robot_arm=my_robot,
                        obstacle_environment=obstacle_environment)

        my_robot.set_joint_state(
            np.array([90, 0, 0]),
            input_unit='deg')

        points, gamma = my_avoider.get_influence_weight_evaluation_points()
        joint_weight, point_weight_list = my_avoider.evaluate_velocity_at_points(points, gamma)

        for ii, weight in enumerate(joint_weight):
            if weight: # nonzero
                point_weight_list[ii] = point_weight_list[ii]*weight

        if visualize:
            x_lim = [-0.3, 4]
            y_lim = [-3, 3]
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 7.5))
            
            my_robot.draw_robot(ax=ax)
            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

            ax.set_xlim(y_lim)
            ax.set_ylim(x_lim)

            ax.set_aspect('equal', adjustable='box')
            ax.grid()

            # breakpoint()
            for pp, weights in enumerate(point_weight_list):
                if weights is None:
                    continue
                
                # plt.scatter(points[0, :, pp], points[1, :, pp], s=(weights*10)**5, zorder=3)
                plt.scatter(points[0, :, pp], points[1, :, pp], s=(weights*100), zorder=3)
                # pp_weights
                
            # for pp in range(evaluation_points.shape[1]):
            # ax.plot(evaluation_points[0, :], evaluation_points[1, :], 'k.')

            breakpoint()


if (__name__) == "__main__":
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    visualize = True
    if visualize:
        import matplotlib.pyplot as plt
        plt.close('all')
        
        my_tester = TestRobotAvoider()
        # my_tester.test_evaluation_points(visualize=True)
        my_tester.test_evaluation_weight(visualize=True)
