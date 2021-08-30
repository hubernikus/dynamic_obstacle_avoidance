"""
Example to stage an robot-arm obstacle avoidance.
"""
# Author: Lukas Hubero

from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import FlatPlane, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from robot_arm_avoider import RobotArmAvoider
from model_robot import ModelRobot2D


def dummy_robot_avoidance():
    x_lim = [-4, 4]
    y_lim = [-0.2, 3.5]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    my_robot = ModelRobot2D()
    my_robot.set_joint_state(np.array([30, -10, -20]), input_unit='deg')
                             
    my_robot.draw_robot(ax=ax)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
    ax.set_aspect('equal', adjustable='box')

    # Creat environment
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        FlatPlane(center_position=np.array([0, 0]),
                  normal=np.array([0, 1]),
                  width=10, height=1,
                  ))

    obstacle_environment.append(
        Ellipse(center_position=np.array([0, 2.1]),
                axes_length=np.array([0.5, 0.5]),
                ))
    
    for obs in obstacle_environment:
        obs.draw_obstacle()
        
    plot_obstacles(ax, obstacle_environment, x_lim, y_lim)
    plt.show()


if (__name__) == "__main__":
    plt.close('all')
    plt.ion()
    # dummy_robot_avoidance()
    jacobian_comparison()
    

