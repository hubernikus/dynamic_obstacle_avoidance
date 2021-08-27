"""
Dummy robot models for cluttered obstacle environment + testing

NOTE: this class requires the sympy module for jacobian evaluation (if desired)
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


class RobotArm():
    pass


class ModelRobot2D(RobotArm):
    """
    Model Robot in 2D with Various Joints.
    """
    def __init__(self):
        self.n_joints = 3
        self._joint_lengths = np.array([1, 1, 1])

        # In radiaon
        self._joint_state = np.zeros(self.n_joints)
        self._joint_axes_of_rotation = [2, 2, 2]   # important for 3D
        self._joint_velocity = np.zeros(self.n_joints)
        
        self.base_position = np.array([0, 0])

        self.dimension = 2

        self.transformation_matrices = self.get_transformation_matrices()
        self.total_transformation = self.get_total_transformation(self.transformation_matrices)


    def _sympy_get_rotation_matrix_from_euler(self, val):
        from sympy import cos, sin, Matrix
        rot = Matrix([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        
        if val[0]:
            ang = val[0]
            rot1 = Matrix([[ cos(ang), sin(ang), 0],
                           [-sin(ang), cos(ang), 0],
                           [0, 0, 1]])

            rot = rot @ rot1
            
        if val[1]:
            ang = val[1]
            rot1 = Matrix([[ cos(ang), 0, -sin(ang)],
                           [0, 1, 0],
                           [sin(ang), 0, cos(ang)]])
                           
            rot = rot @ rot1

        if val[2]:
            ang = val[2]
            rot1 = Matrix([[1, 0, 0],
                           [0, cos(ang), sin(ang)],
                           [0, -sin(ang), cos(ang)]])

            rot = rot @ rot1
        return rot
        
    def analytic_evaluation_jacobian(self):
        """ Symbolic evaluation of the jacobian."""
        from sympy import Matrix, symbols, eye, diff

        # base_rot_matr = _sympy_get_rotation_matrix_from_euler(alpha, beta, gamma)
        joint_lengths = []
        angles = []
        for ii in range(self.n_joints):
            # joint_lengths.append(symbols("ll["+str(ii)+"]"))
            # angles.append(symbols("qq["+str(ii)+"]"))
            joint_lengths.append(symbols("l"+str(ii)))
            angles.append(symbols("q"+str(ii)))
                
        dim = 3
        tot_trafo = eye((dim+1))
        
        for ii in range(self.n_joints):
            tranformation_matrix = eye((dim+1))
            tranformation_matrix[:dim, :dim] = self._sympy_get_rotation_matrix_from_euler(
                [0, 0, angles[ii]])
            tranformation_matrix[:dim, -1] = Matrix([[joint_lengths[ii], 0, 0]]).T
            tot_trafo = tot_trafo @ tranformation_matrix

        from sympy import pprint
        pprint(tot_trafo)
        jacx_dq0 = diff(tot_trafo[:, -1] , angles[0])
        print(jacx_dq0)
        
    def get_total_transformation(self, transformation_matrices):
        self.total_transformation = np.zeros(transformation_matrices.shape[:2])
        for ii in range( transformation_matrices.shape[2]):
            self.total_transformation = (self.total_transformation
                                         @ transformation_matrices[:, :, ii])
        return self.total_transformation
    
    def get_transformation_matrices(self):
        """ Transformation matrices.
        Note, they are expressed in 3D to have compatibility. """
        dim = 3
        
        self.transformation_matrices = np.zeros((dim+1, dim+1, self.n_joints))
        self.transformation_matrices[-1, -1, :] = 1
        
        for ii in range(self.n_joints):
            rot_matr = Rotation.from_euler('xyz', [0, 0, self._joint_state[ii]]).as_matrix()
            self.transformation_matrices[:dim, :dim, ii] = rot_matr
            lin_disp = [self._joint_lengths[ii], 0, 0]
            self.transformation_matrices[:dim, :dim, ii] = lin_disp

        return self.transformation_matrices


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
                              * self._joint_lengths[ii])
            
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



if (__name__) == "__main__":
    my_robot = ModelRobot2D()
    my_robot.analytic_evaluation_jacobian()
