#!/USSR/bin/python3.9
"""
Test script for obstacle avoidance algorithm - specifically the normal function evaluation
"""
import unittest

from math import pi

import numpy as np
import sympy

from robot_avoidance.model_robot import RobotArm2D, ModelRobot2D
from robot_avoidance.analytic_evaluation_jacobian import (
    _get_sympy_transformation_matrix,
)


class TestAnalyticalFunctionEvaluation(unittest.TestCase):
    def test_similarity_of_analytic_and_numerical_rotation_matr(self, visualize=False):
        """Test the forward transform-matrix for different joint-state configurations."""
        my_robot = ModelRobot2D()

        initial_pos_list = [
            [30, -10, -20, 30],
            [10, 20, 30, 30],
            [10, -60, 130, 30],
            [90, 120, 170, -30],
        ]

        for initial_pos in initial_pos_list:
            initial_pos = np.array(initial_pos)
            my_robot.set_joint_state(initial_pos, input_unit="deg")

            trafo_matr = _get_sympy_transformation_matrix(my_robot)
            init_pose = my_robot._joint_state
            for ii in range(init_pose.shape[0]):
                qq = sympy.symbols("qq[" + str(ii) + "]")
                trafo_matr = trafo_matr.subs(qq, init_pose[ii])
                ll = sympy.symbols("ll[" + str(ii) + "]")
                trafo_matr = trafo_matr.subs(ll, my_robot._link_lengths[ii])

            trafo_matr_eval = np.round(np.array(trafo_matr.evalf()).astype(float), 3)
            position_analytical = trafo_matr_eval[:2, -1]

            my_robot.set_joint_state(initial_pos, input_unit="deg")
            ee_pos0 = my_robot.get_ee_in_base()

            if visualize:
                import matplotlib.pyplot as plt

                x_lim = [-0.2, 4.5]
                y_lim = [-4, 4]
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                my_robot.draw_robot(ax=ax)

            # print(f"{position_analytical=}")
            # print(f"{ee_pos0=}")
            self.assertTrue(
                np.allclose(ee_pos0, position_analytical, rtol=1e-2),
                "Analytical & numerical Transformation are not close to each other...",
            )
        # print("Test for analyitical vs. numerical transform is done.")

    def test_jacobian_comparison(self, visualize=False):
        """Use Jacobian for evluation of velocity"""
        x_lim = [-0.2, 4.5]
        y_lim = [-4, 4]

        my_robot = ModelRobot2D()

        delta_time = 0.00001
        initial_pos_list = [
            [0, -10, -20, 10],
            [10, -80, 110, 30],
        ]

        joint_velocity_list = [[0, 20, 10, 10], [31, 90, -10, 30]]

        for initial_pos, joint_velocity in zip(initial_pos_list, joint_velocity_list):
            initial_pos = np.array(initial_pos)
            joint_velocity = np.array(joint_velocity)

            my_robot.set_joint_state(initial_pos, input_unit="deg")
            ee_pos0 = my_robot.get_ee_in_base()
            ee_vel_and_rotation = my_robot.get_ee_velocity(joint_velocity)

            ee_delta_dist = delta_time * ee_vel_and_rotation[:2]

            if visualize:
                fig, axs = plt.subplots(1, 2, figsize=(12, 7.5))
                ax = axs[0]
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.arrow(
                    ee_pos0[0],
                    ee_pos0[1],
                    ee_delta_dist[0],
                    ee_delta_dist[1],
                    color="#808080",
                    head_width=0.1,
                )
                ax.set_aspect("equal", adjustable="box")
                my_robot.draw_robot(ax=ax)

            my_robot.update_state(
                joint_velocity_control=joint_velocity,
                delta_time=delta_time,
                # input_unit='deg',
            )
            ee_pos1 = my_robot.get_ee_in_base()

            self.assertTrue(
                np.allclose(ee_delta_dist, ee_pos1 - ee_pos0, rtol=1e-2),
                "jacobian result unexpectedly large.",
            )
            # print('ee_delta_dist', ee_delta_dist)
            # print('ee_pos -> delta_dist', ee_pos1-ee_pos0)

            if visualize:
                ax = axs[1]
                ax.set_xlim(x_lim)
                ax.set_ylim(y_lim)
                ax.set_aspect("equal", adjustable="box")
                my_robot.draw_robot(ax=ax)

    def test_forward_kinematics(self, visualize=False):
        # Assumption of already created jacobian
        from robot_avoidance.jacobians.robot_arm_3link import _get_jacobian

        my_robot = RobotArm2D(link_lengths=np.array([1, 1, 1]))
        my_robot.set_joint_state(np.array([90, 0, 0]), input_unit="deg")

        my_robot.name = "robot_arm_3link"
        my_robot.set_jacobian(function=_get_jacobian)

        for level in range(my_robot.n_joints + 1):
            joint_vel_0 = pi
            joint_velocity = np.array([joint_vel_0, 0, 0])
            relative_point_position = 0.0

            pos = my_robot.get_joint_in_base(level, relative_point_position)
            vel = my_robot.get_joint_vel_at_linklevel_and_position(
                joint_velocity, level, relative_point_position
            )

            perp_vel = np.array([-pos[1], pos[0]]) * joint_vel_0

            self.assertTrue(np.allclose(perp_vel, vel))

        if visualize:
            fig, ax = plt.subplots(1, 1, figsize=(12, 7.5))
            my_robot.draw_robot(ax=ax)

            ax.set_xlim([-3, 3])
            ax.set_ylim([-0.3, 4])

            ax.set_aspect("equal", adjustable="box")
            ax.grid()

            pos = my_robot.get_joint_in_base(level, relative_point_position)
            vel = my_robot.get_cartesian_vel_from_joint_velocity_on_link(
                joint_velocity, level, relative_point_position
            )

            vel = my_robot.get_joint_vel_at_linklevel_and_position(
                joint_velocity, level, relative_point_position
            )

            vel = vel * 0.1

            ax.plot(pos[0], pos[1], "ko")
            # ax.arrow(pos[0], pos[1], vel[0], vel[1], color='k')
            ax.plot(pos[0] + vel[0], pos[1] + vel[1], "k*")
            # breakpoint()


if (__name__) == "__main__":
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)

    visualize = True
    if visualize:
        import matplotlib.pyplot as plt

        plt.close("all")

        my_tester = TestAnalyticalFunctionEvaluation()
        # my_tester.test_forward_kinematics(visualize=True)
