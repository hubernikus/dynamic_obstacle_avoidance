"""
Dummy robot models for cluttered obstacle environment + testing


"""
# Author: Lukas Huber
from math import pi

from scipy.spatial.transform import Rotation

import numpy as np
from numpy import linalg as LA

from dynamic_obstacle_avoidance.obstacles import FlatPlane
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from robot_arm_avoider import RobotArmAvoider

try:
    from jacobians.model_robot_2d import _get_jacobian
except ImportError:
    print("Jacobian-file found. -- Limited functionality.")


class RobotArm():
    def update_state(self, joint_velocity_control, delta_time=0.01, input_unit='rad'):
        if input_unit == 'deg':
            joint_velocity_control = joint_velocity_control*pi/180
        elif input_unit != 'rad':
            raise Exception("Unkown joint-control input.")
        
        self._joint_state = self._joint_state + joint_velocity_control*delta_time

    def get_jacobian(self):
        """ Returns end-effector velocity based on current joint state. """
        return _get_jacobian(
            ll=self._link_lengths, qq=self._joint_state)


class ModelRobot2D(RobotArm):
    """
    Model Robot in 2D with Various Joints.
    """
    def __init__(self):
        self.n_joints = 4
        self._link_lengths = np.array([0.5, 1, 1, 1])

        # In radiaon
        self._joint_state = np.zeros(self.n_joints)
        self._joint_axes_of_rotation = [2, 2, 2, 2]   # important for 3D
        self._joint_velocity = np.zeros(self.n_joints)
        
        self.base_position = np.array([0, 0])

        self.dimension = 2

        self.name = "model_robot_2d"

        # self.transformation_matrices = self.get_transformation_matrices()
        # self.total_transformation = self.get_total_transformation(self.transformation_matrices)

    @property
    def n_links(self):
        return self.n_joints
    
    def get_joint_stat_plus0(self):
        """ Allows for easier iteration, since 0-angle for base link is added."""
        return np.hstack((self._joint_state, 0))

    def get_ee_in_base(self):
        """ Transform ee to base. """
        transformation_matrices = self.get_transformation_matrices()
        
        dim = 3 
        position = np.zeros((dim+1))
        position[-1] = 1
        
        for ii in reversed(range(transformation_matrices.shape[2])):
            pos_old = position
            position = transformation_matrices[:, :, ii] @ position
        return position[:2]
    
    def get_transformation_matrices(self):
        """ Transformation matrices.
        Note, they are expressed in 3D to have compatibility. """
        dim = 3
        
        transformation_matrices = np.zeros((dim+1, dim+1, self.n_links+1))
        transformation_matrices[-1, -1, :] = 1
        transformation_matrices[:dim, -1, -1] = [self._link_lengths[-1], 0, 0]
        transformation_matrices[:dim, :dim, -1] = np.eye(dim)
        
        for ii in range(1, self.n_joints):
            rot_matr = Rotation.from_euler('xyz', [0, 0, self._joint_state[ii]]).as_matrix()
            transformation_matrices[:dim, :dim, ii] = rot_matr
            transformation_matrices[:dim, -1, ii] = [self._link_lengths[ii-1], 0, 0]

        rot_matr = Rotation.from_euler('xyz', [0, 0, self._joint_state[0]]).as_matrix()
        transformation_matrices[:dim, :dim, 0] = rot_matr
        
        return transformation_matrices

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

    def get_ee_velocity(self, joint_velocity, input_unit='rad'):
        """ Return end-effectory velocity by evaluating the jacobian at current position."""
        if input_unit=='deg':
            joint_velocity = joint_velocity*pi/180.0
        elif input_unit=='rad':
            pass
        else:
            raise Exception(f"Unpexpected input_unit argument: '{input_unit}'")

        jacobian = self.get_jacobian()
        return jacobian @ joint_velocity
        
    def draw_robot(self, ax, link_color='orange', joint_color='black'):
        # Base
        ax.plot(self.base_position[0], self.base_position[1], 'o',
                markersize=16, color=joint_color)
        # joint_state_plus_0 = self.get_joint_stat_plus0()
        joint_state = self._joint_state
        pos_joint_low = self.base_position
        state_joint_low = 0
        
        # breakpoint()
        for ii in range(self.n_links):
            # state_joint_low += self._joint_state[ii]
            state_joint_low += joint_state[ii]
            pos_joint_high = (pos_joint_low
                              + np.array([np.cos(state_joint_low), np.sin(state_joint_low)])
                              * self._link_lengths[ii])
            
            ax.plot([pos_joint_low[0], pos_joint_high[0]],
                    [pos_joint_low[1], pos_joint_high[1]], '-',
                    linewidth=8,
                    color=link_color, zorder=1)
            
            ax.plot(pos_joint_high[0], pos_joint_high[1], 'o',
                    # markeredgewidth=2,
                    markersize=12,
                    color=joint_color, zorder=2)

            pos_joint_low = pos_joint_high

    
