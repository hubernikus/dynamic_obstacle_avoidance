"""
Allows obstacle avoidance based on the dynamical-system method for a robot arm

1. Get the desired motion of the end-effector
   > movement towards attractor
   > over-simplification of critical environment
   
2. Inv kinematics for joint-motor-commands

3. Get weights of critical / evaluation points

4. Forward kinematics for critical points
   [different margins(?)-> maybe not in the beginning]
   
5. DS based modulation for critical points
   > increase avoidance / repulsion
   > mostly consider local environment
   > Move up / down the 'link' chain to ensure base goes in the right direction
   >> Start desired direction for end-effector
   >> move down the link [IK] to get velocity at next point
   >> Modulate, repeat
   >> Outcome: desired velocity at each link
   >> Bottom / base up: Desired velocity

   # Maybe: find closes gamma point alongt links

6. Inverse kinematics of motion at critical points

7. Weighted average of motion
   [additional / increased weight the further it is ahead for end-effector]
   # OR NOT TRUE(?!) since lower joints have more certainty of going the right path (...)
"""
import numpy as np
from numpy import linalg as LA

from dynamic_obstacle_avoidance.avoidance import ObstacleAvoiderWithInitialDynamcis


class RobotArmAvoider():
    def __init__(self, obstacle_avoider: ObstacleAvoiderWithInitialDynamcis, robot_arm) -> None:
        self.obstacle_avoider = obstacle_avoider
        # self.initial_dynamics = initial_dynamics
        self.robot_arm = robot_arm
        # self.obstacle_environment = obstacle_environment

        # Evaluations points per link
        self.n_eval = 4
        
        self.dim = 2

    def get_relative_joint_distance(self, pp, jj):
        """ Array with displacement of each joint along desired direction."""
        return (pp+1)/self.n_eval*self.robot_arm.link_lengths[jj]
        
    def get_influence_weight_evaluation_points(self):
        """ Get  """
        evaluation_points = np.zeros((
            self.dim, self.n_eval, self.robot_arm.n_links))
        gamma_values = np.zeros((
            self.n_eval, self.robot_arm.n_links))

        for pp in range(self.n_eval):
            for jj in range(self.robot_arm.n_links):
                evaluation_points[:, pp, jj] = self.robot_arm.get_joint_in_base(
                    level=jj,
                    relative_point_position=self.get_relative_joint_distance(pp, jj))
                gamma_values[pp, jj] = self.get_gamma_product(evaluation_points[:, pp, jj])
        return evaluation_points, gamma_values

    def evaluate_velocity_at_points(self, evaluation_points, gamma_values, cutoff_gamma=5.0):
        # Cut-off far away points to reduce amount of dynamical system evaluation
        evaluation_points, gamma_values = self.get_influence_weight_evaluation_points()

        # 1. Caclulate weight across gamma's first and inbetweend links
        min_gammas_joints = np.min(gamma_values, axis=0)
        ind_nonzero = min_gammas_joints < cutoff_gamma

        joint_weight = np.zeros(min_gammas_joints.shape)
        joint_weight[ind_nonzero] = ((cutoff_gamma - min_gammas_joints[ind_nonzero])
                                      / (cutoff_gamma - 1))
        weight_sum = np.sum(joint_weight)

        if weight_sum > 1:
            joint_weight /= weight_sum
        else:
            # Give more importance to end-effector link when far away from everything.
            joint_weight[0] = joint_weight[0] + (1-weight_sum)

        # 2. Caclulate weight across one link & the 'relative' value
        point_weight_list = [None] * self.robot_arm.n_joints
        cutoff_weight = 1.0/(self.n_eval + 1)
        for jj in range(self.robot_arm.n_joints):
            if not joint_weight[jj]:
                continue
            
            ind_nonzero = gamma_values[:, jj] < cutoff_gamma
            point_weight = np.zeros(gamma_values.shape[0])
            point_weight[ind_nonzero] = ((cutoff_gamma - gamma_values[ind_nonzero, jj])
                                         / (cutoff_gamma - 1))
            # Additionally cut of low weights
            point_weight = np.maximum(point_weight - cutoff_weight, 0)
            point_weight = point_weight / np.sum(point_weight)

            point_weight_list[jj] = point_weight
            
        return joint_weight, point_weight_list

    # def get_ee_velocity(self, position):
        # return self.obstacle_avoider.evaluate(position=position_list[:, ii])
        # desired_velocity = initial_dynamics.evaluate(position=position_list[:, ii])
        
        # desired_velocity1 = obs_avoidance_interpolation_moving(
            # position_list[:, ii],
            # desired_velocity, obs=obstacle_environment)

    def get_joint_velocity(self, position):
        velocity_ee = self.get_ee_velocity(position)
        joint_control = robot_arm.get_inverse_kinematics(velocity_ee)

        # Weight of each joint & corresponding weights of 'sub-samples'
        joint_weight_list, point_weight_list = self.evaluate_velocity_at_points()

        # Get velocity of link with respect to obstacle-envirnment & corresponding weights
        for jj, point_weight in enumerate(point_weight_list):
            if point_weight is None:
                continue

            # ik_velocities = np.zeros((self.dim, point_weight.shape[0]))
            # ik_modulated_velocities = np.zeros((self.dim, point_weight.shape[0]))
            jonit_control_ik_modulated = np.zeros((self.dim, point_weight.shape[0]))
            
            for pp in range(point_weight.shape[0]):
                if not point_weight[pp]:
                    continue

                ik_velocities = robot_arm.get_joint_vel_at_linklevel_and_position(
                    joint_velocity=joint_control, level=jj,
                    relative_point_position=self.get_relative_point_position(pp, jj))
            
                
                
    def get_gamma_product(self, position):
        gamma_list = np.zeros(len(self.obstacle_environment))
        for ii, obs in enumerate(self.obstacle_environment):
            gamma_list[ii] = obs.get_gamma(position, in_global_frame=True)

        n_obs = len(gamma_list)
        # Total gamma [1, infinity]
        # Take root of order 'n_obs' to make up for the obstacle multiple
        gamma = np.prod(gamma_list-1)**(1.0/n_obs) + 1
        return gamma

    def evaluate_motion(self):
        pass

    def update(self):
        pass

