"""
Dummy robot models for cluttered obstacle environment + testing
"""
# Author: Lukas Huber

from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import FlatPlane
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from robot_arm_avoider import RobotArmAvoider


class RobotArm():
    pass


class ModelRobot(RobotArm):
    """
    Velocity base controller.
    """
    def __init__(self):
        self.n_joints = 3
        self.joint_lengths = np.array([1, 1, 1])

        # In radiaon
        self._joint_state = np.zeros(self.n_joints)
        self._joint_velocity = np.zeros(self.n_joints)
        
        self.base_position = np.array([0, 0])

    def set_joint_state(self, value, input_unit='rad'):
        if value.shape[0] != self.n_joints:
            raise Exception("Wrong dimension of joint input.")
        
        if input_unit=='rad':
            self._joint_state = value
        elif input_unit=='deg':
            self._joint_state = value*pi/180.0
        else:
            raise Exception(f"Unpexpected input_unit argument: '{input_unit}'")

    def set_velocity(self, value, input_unit='rad'):
        if input_unit=='rad':
            self._joint_velocity = value
        elif input_unit=='deg':
            self._joint_velocity = value*pi/180.0
        else:
            raise Exception(f"Unpexpected input_unit argument: '{input_unit}'")
        
    def draw_robot(self, ax, link_color='orange', joint_color='black'):
        # Base
        ax.plot(self.base_position[0], self.base_position[1], 'o',
                markersize=16, color=joint_color)

        pos_joint_low = self.base_position
        state_joint_low = 0
        for ii in range(self.n_joints):
            state_joint_low += self._joint_state[ii]
            pos_joint_high = (pos_joint_low
                              + np.array([-np.sin(state_joint_low), np.cos(state_joint_low)])
                              * self.joint_lengths[ii])
            
            ax.plot([pos_joint_low[0], pos_joint_high[0]],
                    [pos_joint_low[1], pos_joint_high[1]], '-',
                    linewidth=8,
                    color=link_color, zorder=1)
            
            ax.plot(pos_joint_high[0], pos_joint_high[1], 'o',
                    # markeredgewidth=2,
                    markersize=12,
                    color=joint_color, zorder=2)

            pos_joint_low = pos_joint_high

    @property
    def jacobian(self):
        return None

    def update(self, delta_time=0.01):
        self._joint_state = self._joint_state + self._joint_velocity*delta_time


def dummy_robot_avoidance():
    x_lim = [-4, 4]
    y_lim = [-0.2, 3.5]
    fig, ax = plt.subplots(figsize=(12, 6))
    
    my_robot = ModelRobot()
    my_robot.set_joint_state(np.array([30, -60, 30]), input_unit='deg')
                             
    my_robot.draw_robot(ax=ax)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    
    ax.set_aspect('equal', adjustable='box')

    # Creat environment
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        FlatPlane(position=np.array([0, 0]),
                  normal=np.array([0, 1]),
                  width=10, height=1,
                  ))
    for obs in obstacle_environment:
        obs.draw_obstacle()
        
    plot_obstacles(ax, obstacle_environment, x_lim, y_lim)
    plt.show()


if (__name__) == "__main__":
    plt.close('all')
    plt.ion()
    dummy_robot_avoidance()
    
        
