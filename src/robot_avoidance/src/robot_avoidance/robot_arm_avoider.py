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

from dynamic_obstacle_avoidance.avoidance import (
    ObstacleAvoiderWithInitialDynamcis,
)


class RobotArmAvoider:
    """
    Class which avoids obstacles based on robot-arm
    """

    def __init__(
        self,
        obstacle_avoider: ObstacleAvoiderWithInitialDynamcis,
        robot_arm,
        n_eval: int = 4,
    ) -> None:
        self.obstacle_avoider = obstacle_avoider
        # self.initial_dynamics = initial_dynamics
        self.robot_arm = robot_arm
        # self.obstacle_environment = obstacle_environment

        # Evaluations points per link
        self.n_eval = n_eval
        self.dim = 2

    def get_relative_joint_distance(self, pp, jj):
        """Array with displacement of each joint along desired direction."""
        pp = pp % self.n_eval
        return (pp + 1) / self.n_eval * self.robot_arm.link_lengths[jj]

    def get_evaluation_points(self):
        evaluation_points = np.zeros((self.dim, self.n_eval, self.robot_arm.n_links))

        for pp in range(self.n_eval):
            for jj in range(self.robot_arm.n_links):
                evaluation_points[:, pp, jj] = self.robot_arm.get_joint_in_base(
                    level=jj,
                    relative_point_position=self.get_relative_joint_distance(pp, jj),
                )

        return evaluation_points

    def get_gamma_at_points(self, evaluation_points):
        """Get gamma-product at evaluation-points"""
        gamma_values = np.zeros((self.n_eval, self.robot_arm.n_links))

        for pp in range(self.n_eval):
            for jj in range(self.robot_arm.n_links):
                gamma_values[pp, jj] = self.obstacle_avoider.get_gamma_product(
                    evaluation_points[:, pp, jj]
                )
        # breakpoint()
        return gamma_values

    def get_weight_from_gamma(
        self, gammas, cutoff_gamma, n_points, gamma0=1.0, frac_gamma_nth=0.5
    ):
        """
        Arguments
        ---------
        cutoff_gamma: Gamma value at which the weights is equal to 1
        gamma0: Gamma value at which weight is maximal (equal to 1 & overpowering all other weights
        frac_gamma_nth: Fraction of the gamma span (difference between the previous two values)
            at which it reaches 1/n_joint or 1/n_eval respectively
        """
        weights = (gammas - gamma0) / (cutoff_gamma - gamma0)
        weights = weights / frac_gamma_nth
        weights = 1.0 / weights
        weights = (weights - frac_gamma_nth) / (1 - frac_gamma_nth)
        weights = weights / n_points
        return weights

    def get_influence_weight_at_points(self, evaluation_points=None, cutoff_gamma=1.5):
        """Get weights for evaluation of joint & corresponding joint positions."""
        if evaluation_points is None:
            evaluation_points = self.get_evaluation_points()

        # Cut-off far away points to reduce amount of dynamical system evaluation
        gamma_values = self.get_gamma_at_points(evaluation_points)

        # 1. Caclulate across several links
        min_gammas_joints = np.min(gamma_values, axis=0)
        ind_nonzero = min_gammas_joints < cutoff_gamma

        joint_weights = np.zeros(min_gammas_joints.shape)
        joint_weights[ind_nonzero] = self.get_weight_from_gamma(
            min_gammas_joints[ind_nonzero],
            cutoff_gamma=cutoff_gamma,
            n_points=self.robot_arm.n_joints,
        )

        # Leverage weight: higher weight at end of effector
        joint_weights = joint_weights * np.arange(1, self.robot_arm.n_links + 1)

        weight_sum = np.sum(joint_weights)
        if weight_sum > 1:
            # Only normalize when larger than 1
            joint_weights /= weight_sum

            # [?! -> is the next comment still valid]
            # remaining weight will always be assigned to 'standard' IK-velocity
        # else:
        # joint_weights[-1] += 1 - weight_sum

        # else:
        # Give more importance to end-effector link when far away from everything.
        # joint_weights[-1] = joint_weights[-1] + (1-weight_sum)

        # 2. Caclulate weight across points on an individual link & the 'relative' value
        point_weight_list = [None] * self.robot_arm.n_joints
        for jj in range(self.robot_arm.n_joints):
            if not joint_weights[jj]:
                continue

            point_weight_list[jj] = np.zeros(gamma_values.shape[0])

            ind_nonzero = gamma_values[:, jj] < cutoff_gamma
            if not any(ind_nonzero):
                # This can be the case when the last-joint does not have any important weights
                # in this case, make the end-effector matter (!)
                point_weight_list[jj][-1] = 1

            point_weight_list[jj][ind_nonzero] = self.get_weight_from_gamma(
                gamma_values[ind_nonzero, jj],
                cutoff_gamma=cutoff_gamma,
                n_points=self.n_eval,
            )

            # Leverage weight: higher weight at end of effector
            point_weight_list[jj] = point_weight_list[jj] * np.arange(
                1, point_weight_list[jj].shape[0] + 1
            )

            # Normalize
            point_weight_sum = np.sum(point_weight_list[jj])
            if point_weight_sum > 1:
                point_weight_list[jj] = point_weight_list[jj] / point_weight_sum
            else:
                # Favour the last & guiding point
                point_weight_list[jj][-1] += 1 - point_weight_sum

        # print()
        # print(f"joint_position={repr(self.robot_arm._joint_state)}")
        # print(f'min_gammas_joints={repr(min_gammas_joints)}')
        # print(f'joint_weight={repr(joint_weights)}')
        # breakpoint()

        return joint_weights, point_weight_list

    def get_joint_avoidance_velocity(
        self, max_joint_vel=1.0, max_cart_vel=0.3, ax=None
    ):
        """
        Arguments
        ----------
        max_joint_vel: Crop after IK output each individual joint velocity (singularity...)
        max_cart_vel: Scale modulated velocity at
        ax: if ax-input is given -> draw arrow
        """
        dim = 2
        # Weight of each joint & corresponding weights of 'sub-samples'
        evaluation_points = self.get_evaluation_points()
        (
            joint_weight_list,
            point_weight_list,
        ) = self.get_influence_weight_at_points(evaluation_points)

        # Actual command after considering each of the joints
        joint_control_weighted = np.zeros(self.robot_arm.n_joints)

        # if sum(joint_weight_list) < 1:
        # Base IK - with end-effector following a DS
        velocity_mod = self.obstacle_avoider.evaluate(evaluation_points[:, -1, -1])

        joint_control_ik = self.robot_arm.get_inverse_kinematics(velocity_mod)

        # Adapt weights to only act when actually moving
        # -> since weight in [0, 1], we know that increased weight will have increased influence
        only_dynamic_active = True
        if only_dynamic_active:
            # TODO: put this into 'get_influence_weight_at_points'-function
            # breakpoint()
            # print(joint_weight_list)

            joint_vel_weight_base = 1.0
            power_weight = np.zeros(joint_control_ik.shape)
            ind_nonzero = joint_control_ik != 0
            power_weight[ind_nonzero] = joint_vel_weight_base / np.abs(
                joint_control_ik[ind_nonzero]
            )

            joint_weight_list = joint_weight_list ** power_weight
            # print(joint_weight_list)

        joint_control_weighted = (1 - np.sum(joint_weight_list)) * joint_control_ik
        # print('start weighted', joint_control_weighted)

        # Get velocity of link with respect to obstacle-envirnment & corresponding weights
        for jj, point_weight in enumerate(point_weight_list):
            # Assign velocity coming from previous IK / wegihts
            rel_pos = self.get_relative_joint_distance(self.n_eval - 1, jj)
            joint_control_up_till_now = np.zeros(joint_control_weighted.shape)
            joint_control_up_till_now[:jj] = joint_control_weighted[:jj]

            velocity_control = (
                self.robot_arm.get_cartesian_vel_from_joint_velocity_on_link(
                    joint_velocity=joint_control_up_till_now,
                    level=jj,
                    relative_point_position=rel_pos,
                )
            )

            velocity_ik = self.robot_arm.get_cartesian_vel_from_joint_velocity_on_link(
                joint_velocity=joint_control_ik,
                level=jj,
                relative_point_position=rel_pos,
            )

            diff_velocity = velocity_ik - velocity_control

            if jj == 0:
                # Get the base of the robot
                start_link_pos = self.robot_arm.get_joint_in_base(
                    level=jj, relative_point_position=0
                )
            else:
                start_link_pos = evaluation_points[:, -1, jj - 1]

            control_velocity = np.cross(
                evaluation_points[:, -1, jj] - start_link_pos, diff_velocity
            )

            # TODO: is this control velocity really correct (!?)
            # Compare with paper...

            # print('jj', jj)
            # print('control_velocity', control_velocity)
            # breakpoint()
            # joint_control_weighted[jj] += (1-np.sum(joint_weight_list[jj+1:]))*control_velocity
            # joint_control_weighted[jj] += (np.sum(joint_weight_list[:jj+1]))*control_velocity
            joint_control_weighted[jj] += (
                np.sum(joint_weight_list[:jj])
            ) * control_velocity

            if point_weight is None:
                continue

            # orientation = self.robot_arm.get_joint_orientation_in_base(level=jj)
            # joint_dir = np.array([np.cos(orientation), np.sin(orientation)])
            # joint_direction_perp = np.array([-np.sin(orientation), np.cos(orientation)])

            # Linear and angular velocity at joint end
            velocity_linear = np.zeros(dim)
            velocity_angular = 0

            if ax is not None:
                fac_arrow = 0.1

                arr_mod = ax.arrow(
                    evaluation_points[0, -1, jj],
                    evaluation_points[1, -1, jj],
                    fac_arrow * velocity_ik[0],
                    fac_arrow * velocity_ik[1],
                    color="b",
                    width=0.01,
                    zorder=10,
                )

            for pp in range(point_weight.shape[0]):
                if not point_weight[pp]:
                    continue

                velocity_mod_weighted = self.obstacle_avoider.evaluate(
                    evaluation_points[:, pp, jj]
                )

                velocity_mod_weighted = velocity_mod_weighted * point_weight[pp]
                velocity_linear += velocity_mod_weighted

                dir_from_joint_base = (
                    evaluation_points[:, pp, jj] - evaluation_points[:, 0, jj]
                )
                velocity_angular += np.cross(dir_from_joint_base, velocity_mod_weighted)

                if ax is not None:
                    fac_arrow = 0.1

                    # arr_ik = ax.arrow(
                    # evaluation_points[0, pp, jj], evaluation_points[1, pp, jj],
                    # fac_arrow*velocity_ik[0], fac_arrow*velocity_ik[1],
                    # color='b', width=0.01, zorder=10)

                    arr_mod = ax.arrow(
                        evaluation_points[0, pp, jj],
                        evaluation_points[1, pp, jj],
                        fac_arrow * velocity_mod_weighted[0],
                        fac_arrow * velocity_mod_weighted[1],
                        color="r",
                        width=0.01,
                        zorder=10,
                    )

                    # arr_ctrl = ax.arrow(
                    # evaluation_points[0, pp, jj], evaluation_points[1, pp, jj],
                    # fac_arrow*velocity_control[0], fac_arrow*velocity_control[1],
                    # color='r', width=0.01, zorder=10)

            joint_control_mod_ik = self.robot_arm.get_inverse_kinematics_at_level(
                velocity_linear, velocity_angular, level=jj
            )

            # Only assign joint_control_modulated after the pp-loop to not affect
            # the 'velocity_control' measurement.
            # weight = np.sum(joint_weight_list[:jj+1])
            # joint_control_weighted[jj] += weight * joint_control_ik[jj]

            # Assign previous joints too
            joint_control_weighted[: jj + 1] += (
                joint_weight_list[jj] * joint_control_mod_ik
            )

            # print("jj", jj)
            # print(f"control ik ", joint_control_ik)
            # print(f'control mod', joint_control_mod_ik)
            # print(f'control total', joint_control_weighted)

            # print('jj', jj)

            # arr_ik.set_label("Initial Ctrl")
            # arr_mod.set_label("Initial Ctrl")
            # arr_ctrl.set_label("Initial Ctrl")

        # print()
        # print(f"joint_state = {self.robot_arm._joint_state}")
        # print(f"joint_weight_list = {joint_weight_list}")

        # print(f"joint_control_ik = {joint_control_ik}")
        # print(f"joint_control_weighted = {joint_control_weighted}")

        # print()
        # breakpoint()

        return joint_control_weighted

    def get_joint_avoidance_velocity_old(self, max_joint_vel=1.0, max_cart_vel=0.3):
        position = self.robot_arm.get_ee_in_base()
        velocity_ee = self.obstacle_avoider.evaluate(position=position)

        vel_norm = LA.norm(velocity_ee)
        if vel_norm > max_cart_vel:
            velocity_ee /= vel_norm * max_cart_vel

        joint_control_ik = self.robot_arm.get_inverse_kinematics(velocity_ee)

        # Already limit here, to avoid 'jittering' around singularity
        ind_large = np.abs(joint_control_ik) > max_joint_vel
        joint_control_ik[ind_large] = np.copysign(
            max_joint_vel, joint_control_ik[ind_large]
        )

        # Actual command after considering each of the joints
        joint_control_modulated = np.zeros(joint_control_ik.shape)

        # Weight of each joint & corresponding weights of 'sub-samples'
        evaluation_points = self.get_evaluation_points()
        (
            joint_weight_list,
            point_weight_list,
        ) = self.get_influence_weight_at_points(evaluation_points)

        # Get velocity of link with respect to obstacle-envirnment & corresponding weights
        for jj, point_weight in enumerate(point_weight_list):
            orientation = self.robot_arm.get_joint_orientation_in_base(level=jj)
            joint_direction_perp = np.array([-np.sin(orientation), np.cos(orientation)])

            if point_weight is None:
                rel_pos = self.get_relative_joint_distance(-1, jj)
                velocity_ik = (
                    self.robot_arm.get_cartesian_vel_from_joint_velocity_on_link(
                        joint_velocity=joint_control_ik,
                        level=jj,
                        relative_point_position=rel_pos,
                    )
                )

                velocity_control = (
                    self.robot_arm.get_cartesian_vel_from_joint_velocity_on_link(
                        joint_velocity=joint_control_modulated,
                        level=jj,
                        relative_point_position=rel_pos,
                    )
                )

                diff_velocity_ik = velocity_ik - velocity_control
                control_velocity_ik = (
                    np.dot(diff_velocity_ik, joint_direction_perp) / rel_pos
                )

                joint_control_modulated[jj] = control_velocity_ik
                continue

            control_velocity = 0
            control_velocity_ik = 0

            for pp in range(point_weight.shape[0]):
                if not point_weight[pp]:
                    continue
                rel_pos = self.get_relative_joint_distance(pp, jj)

                velocity_ik = (
                    self.robot_arm.get_cartesian_vel_from_joint_velocity_on_link(
                        joint_velocity=joint_control_ik,
                        level=jj,
                        relative_point_position=rel_pos,
                    )
                )

                # velocity_modulated = self.obstacle_avoider.avoid(
                # evaluation_points[:, pp, jj], velocity_ik)
                velocity_modulated = self.obstacle_avoider.evaluate(
                    evaluation_points[:, pp, jj]
                )

                velocity_control = (
                    self.robot_arm.get_cartesian_vel_from_joint_velocity_on_link(
                        joint_velocity=joint_control_modulated,
                        level=jj,
                        relative_point_position=rel_pos,
                    )
                )

                diff_velocity = velocity_modulated - velocity_control
                control_velocity += (
                    np.dot(diff_velocity, joint_direction_perp)
                    / rel_pos
                    * point_weight[pp]
                )

                diff_velocity_ik = velocity_ik - velocity_control
                control_velocity_ik += (
                    np.dot(diff_velocity_ik, joint_direction_perp)
                    / rel_pos
                    * point_weight[pp]
                )

                # if jj == 2:
                # print(f"CONTROL ik = {control_velocity_ik}")
                # print(f"CONTROL mo = {control_velocity}")

                if True:
                    # print("")
                    # print(f'points[{pp}, {jj}] = {evaluation_points[:, pp, jj]}')
                    # print(f'velocity_con = {np.round(velocity_control, 2)}')
                    # print(f'velocity_mod = {np.round(velocity_modulated, 2)}')

                    # print(f'diff vel ik = {diff_velocity_ik}')
                    # print(f'diff vel mo= {diff_velocity}')

                    fac_arrow = 0.1

                    arr_ik = ax.arrow(
                        evaluation_points[0, pp, jj],
                        evaluation_points[1, pp, jj],
                        fac_arrow * velocity_ik[0],
                        fac_arrow * velocity_ik[1],
                        color="b",
                        width=0.01,
                        zorder=10,
                    )

                    arr_mod = ax.arrow(
                        evaluation_points[0, pp, jj],
                        evaluation_points[1, pp, jj],
                        fac_arrow * velocity_modulated[0],
                        fac_arrow * velocity_modulated[1],
                        color="g",
                        width=0.01,
                        zorder=10,
                    )

                    arr_ctrl = ax.arrow(
                        evaluation_points[0, pp, jj],
                        evaluation_points[1, pp, jj],
                        fac_arrow * velocity_control[0],
                        fac_arrow * velocity_control[1],
                        color="r",
                        width=0.01,
                        zorder=10,
                    )

            # print("Weight points", point_weight)
            # Only assign joint_control_modulated after the pp-loop to not affect
            # the 'velocity_control' measurement.
            weight = np.sum(joint_weight_list[: jj + 1])
            joint_control_modulated[jj] = (
                weight * control_velocity + (1 - weight) * control_velocity_ik
            )

            # if np.abs(joint_control_modulated[jj]) > max_joint_vel:
            # joint_control_modulated[jj] = np.copysign(
            # max_joint_vel, joint_control_modulated[jj])

            # print('joint control', joint_control_modulated)
            # breakpoint()
            # _ = None
            arr_ik.set_label("Initial Ctrl")
            arr_mod.set_label("Initial Ctrl")
            arr_ctrl.set_label("Initial Ctrl")
            # ax.arrow(0, 0, 0, 0, color='b', width=0.01, zorder=10, label="IK velocity")
            # ax.arrow(0, 0, 0, 0, color='g', width=0.01, zorder=10, label="Modulated")

            # ax.legend(True)

        print(f"joint_position={self.robot_arm._joint_state}")
        # print(f"joint_weight: {joint_weight_list}")
        print()
        # print(f"control_ik= {np.round(joint_control_ik, 2)}")
        # print(f"control_mo= {np.round(joint_control_modulated, 2)}")

        return joint_control_modulated

    def evaluate_motion(self):
        pass

    def update(self):
        pass
