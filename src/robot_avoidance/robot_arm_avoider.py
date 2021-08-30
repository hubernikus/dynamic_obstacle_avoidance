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

class RobotArmAvoider():
    def __ini__(self, attractor, robot_arm, obstacle_environment):
        self.attractor = attractor
        self.robot_arm = robot_arm
        self.obstacle_environment = obstacle_environment

    def set_evaluation_points(self, n_points):
        """ Set points """
        self.evaluation_points = None
        # self.evaluation_point_margins = None
        raise NotImplementedError()

    def get_gamma_at_evaluation_points(self, position):
        raise NotImplementedError()

    def get_gamma_product(self, position):
        gamma_list = np.zeros(len(self.environment))
        for ii in range(gamma_list):
            gamma_list[ii] = self.environment[ii].get_gamma(position, in_global_frame=True)

        # Total gamma [1, infinity]
        gamma = np.prod(gamma_list-1) + 1
        return gamma

    def evaluate_motion(self):
        pass

    def update(self):
        pass

