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


def jacobian_comparison(max_it=100):
    """ Use Jacobian for evluation of velocity """
    x_lim = [-0.2, 4.5]
    y_lim = [-4, 4]

    # fig, ax = plt.subplots(figsize=(12, 7.5))
    fig, axs = plt.subplots(1, 2, figsize=(12, 7.5))
    
    my_robot = ModelRobot2D()

    delta_time = 0.01
    initial_pos = np.array([30, -10, -20])
    # initial_pos = np.array([0, 0, 0])
    joint_velocity = np.array([10, 20, 10])
    
    my_robot.set_joint_state(initial_pos, input_unit='deg')
    ee_pos0 = my_robot.get_ee_in_base()
    ee_vel = my_robot.get_ee_velocity(joint_velocity, input_unit='deg')

    ax = axs[0]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.arrow(ee_pos0[0], ee_pos0[1], ee_vel[0], ee_vel[1], color='#808080', head_width=0.1)
    ax.set_aspect('equal', adjustable='box')
    my_robot.draw_robot(ax=ax)

    
    my_robot.update_state(joint_velocity_control=joint_velocity, delta_time=delta_time)
    ee_pos1 = my_robot.get_ee_in_base()

    print('ee_vel', ee_vel)
    print('ee_pos', ee_pos0)
    print('ee_pos', ee_pos1)

    ax = axs[1]
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal', adjustable='box')
    my_robot.draw_robot(ax=ax)



if (__name__) == "__main__":
    plt.close('all')
    plt.ion()
    # dummy_robot_avoidance()
    # jacobian_comparison()
    
    test_similarity_of_analytic_and_numerical_rotation_matr()

