#!/USSR/bin/python3.9
"""
Test script for obstacle avoidance algorithm - specifically the normal function evaluation
"""
import unittest
from math import pi

import numpy as np
from numpy import linalg as LA

from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.obstacles import Sphere, Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.avoidance import DynamicModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from robot_avoidance.robot_arm_avoider import RobotArmAvoider
from robot_avoidance.model_robot import RobotArm2D


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

        dynamic_avoider = DynamicModulationAvoider(
            initial_dynamics=initial_dynamics, environment=obstacle_environment
        )

        my_avoider = RobotArmAvoider(
            obstacle_avoider=dynamic_avoider, robot_arm=my_robot
        )

        my_robot.set_joint_state(np.array([90, 0, 0]), input_unit="deg")

        evaluation_points = my_avoider.get_evaluation_points()
        gamma_values = my_avoider.get_gamma_at_points(evaluation_points)

        for pp in range(evaluation_points.shape[1]):
            for jj in range(evaluation_points.shape[2]):
                self.assertTrue(np.isclose(evaluation_points[0, pp, jj], 0))
                self.assertTrue(evaluation_points[1, pp, jj] >= 0)

        # Check if gamma is decreasing in a 'simple' circle sceneario
        gamma_values = gamma_values.T.flatten()
        for ii in range(1, gamma_values.shape[0]):
            self.assertTrue(gamma_values[ii - 1] > gamma_values[ii])

        if visualize:
            x_lim = [-0.3, 7]
            y_lim = [-3, 3]

            fig, ax = plt.subplots(1, 1, figsize=(12, 7.5))

            my_robot.draw_robot(ax=ax)
            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

            ax.set_xlim(y_lim)
            ax.set_ylim(x_lim)

            ax.set_aspect("equal", adjustable="box")
            ax.grid()
            # for pp in range(evaluation_points.shape[1]):
            ax.plot(evaluation_points[0, :], evaluation_points[1, :], "k.")

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

        dynamic_avoider = DynamicModulationAvoider(
            initial_dynamics=initial_dynamics, environment=obstacle_environment
        )

        my_avoider = RobotArmAvoider(
            obstacle_avoider=dynamic_avoider, robot_arm=my_robot
        )

        my_robot.set_joint_state(np.array([90, 0, 0]), input_unit="deg")

        points = my_avoider.get_evaluation_points()
        # gamma = my_avoider.get_gamma_at_points(points)

        # joint_weight, point_weight_list = my_avoider.evaluate_velocity_at_points(points, gamma)
        (
            joint_weight,
            point_weight_list,
        ) = my_avoider.get_influence_weight_at_points(points)

        for ii, weight in enumerate(joint_weight):
            if weight:  # nonzero
                point_weight_list[ii] = point_weight_list[ii] * weight

        if visualize:
            x_lim = [-0.3, 4]
            y_lim = [-3, 3]

            fig, ax = plt.subplots(1, 1, figsize=(12, 7.5))

            my_robot.draw_robot(ax=ax)
            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

            ax.set_xlim(y_lim)
            ax.set_ylim(x_lim)

            ax.set_aspect("equal", adjustable="box")
            ax.grid()

            # breakpoint()
            for pp, weights in enumerate(point_weight_list):
                if weights is None:
                    continue

                weights = (weights ** 5) * 1e7
                # plt.scatter(points[0, :, pp], points[1, :, pp], s=(weights*10)**5, zorder=3)
                plt.scatter(points[0, :, pp], points[1, :, pp], s=(weights), zorder=3)
                # pp_weights

            # for pp in range(evaluation_points.shape[1]):
            # ax.plot(evaluation_points[0, :], evaluation_points[1, :], 'k.')
            # breakpoint()

        # Check if all the sum of weights is equal to 1
        sum_weights = 0
        for weights in point_weight_list:
            if weights is not None:
                sum_weights += np.sum(weights)
        self.assertTrue((sum_weights <= 1), f"Sum weights: {sum_weights} instead  <1.")

    def test_3link_robot_arm(self, visualize=False, save_figure=False):
        my_robot = RobotArm2D(link_lengths=np.array([1, 1, 1]))
        my_robot.set_joint_state(np.array([70 + 90, -30, -30]), input_unit="deg")
        # my_robot.set_joint_state(np.array([ 2.84362939, -1.92357885, -0.91727301]

        my_robot.name = "robot_arm_3link"

        from robot_avoidance.jacobians.robot_arm_3link import _get_jacobian

        my_robot.set_jacobian(function=_get_jacobian)

        margin_absolut = 0.1
        obstacle_environment = ObstacleContainer()
        obstacle_environment.append(
            Cuboid(
                center_position=np.array([0.2, 2.4]),
                axes_length=[0.4, 2.4],
                margin_absolut=margin_absolut,
                orientation=-30 * pi / 180,
                tail_effect=False,
                repulsion_coeff=1.5,
            )
        )

        obstacle_environment.append(
            Cuboid(
                center_position=np.array([1.2, 0.25]),
                axes_length=[0.4, 1.45],
                margin_absolut=margin_absolut,
                orientation=0 * pi / 180,
                tail_effect=False,
                repulsion_coeff=1.5,
            )
        )

        # attractor_position = np.array([2, 2])
        attractor_position = np.array([2, 0.7])
        initial_dynamics = LinearSystem(attractor_position=attractor_position)

        dynamic_avoider = DynamicModulationAvoider(
            initial_dynamics=initial_dynamics, environment=obstacle_environment
        )

        main_avoider = RobotArmAvoider(
            obstacle_avoider=dynamic_avoider,
            robot_arm=my_robot,
            n_eval=4,
        )

        position = my_robot.get_ee_in_base()
        # joint_control = main_avoider.get_joint_avoidance_velocity(position)
        evaluation_points = main_avoider.get_evaluation_points()

        max_joint_vel = 0.5
        max_cart_vel = 0.5

        velocity_ee = main_avoider.obstacle_avoider.evaluate(position=position)
        # print(f"{velocity_ee=}")
        vel_norm = LA.norm(velocity_ee)
        if vel_norm > max_cart_vel:
            velocity_ee /= vel_norm * max_cart_vel

        joint_control_ik = main_avoider.robot_arm.get_inverse_kinematics(velocity_ee)
        joint_control_ik = np.minimum(joint_control_ik, max_joint_vel)
        # joint_control_ik = main_avoider

        jj = 0
        pp = 3

        rel_pos = main_avoider.get_relative_joint_distance(pp, jj)

        velocity_ik = (
            main_avoider.robot_arm.get_cartesian_vel_from_joint_velocity_on_link(
                joint_velocity=joint_control_ik,
                level=jj,
                relative_point_position=rel_pos,
            )
        )

        # dir_link = np.
        vel_perp = np.array([-0.17101007, -0.46984631])
        self.assertTrue(np.allclose(vel_perp, velocity_ik))

        # Specific state evaluation
        # my_robot.set_joint_state(np.array([ 3.19710803, -2.05862884, -1.49232649]))
        my_robot.set_joint_state(np.array([1.65, -0.7, -1.23929867]))

        if visualize:
            # x_lim = [-3.5, 3.5]
            # y_lim = [-0.5, 4]
            x_lim = [-1.0, 2.0]
            y_lim = [-0.03, 2.5]

            fig, ax = plt.subplots(1, 1, figsize=(12, 7.5))

            my_robot.draw_robot(ax=ax)
            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

            ax.set_aspect("equal", adjustable="box")
            ax.grid()

            ax.tick_params(
                axis="both",
                which="major",
                labelbottom=False,
                labelleft=False,
                bottom=False,
                top=False,
                left=False,
                right=False,
            )

            main_avoider.n_eval = 3

            joint_control = main_avoider.get_joint_avoidance_velocity(ax=ax)
            if save_figure:
                figure_name = "robot_arm_with_avoidance_arrows"
                plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

            plt.plot(
                initial_dynamics.attractor_position[0],
                initial_dynamics.attractor_position[1],
                "k*",
                # linewidth=18.0,
                markersize=10,
            )

        else:
            joint_control = main_avoider.get_joint_avoidance_velocity()


if (__name__) == "__main__":
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)
    visualize = True
    if visualize:
        import matplotlib.pyplot as plt

        plt.close("all")

        my_tester = TestRobotAvoider()
        # my_tester.test_evaluation_points(visualize=True)
        # my_tester.test_evaluation_weight(visualize=True)
        my_tester.test_3link_robot_arm(visualize=True, save_figure=True)
