"""
Replcication of 2D-animation of paper
"""
# Author: Lukas Huber
import time
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer

# from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving
from dynamic_obstacle_avoidance.avoidance import DynamicModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from vartools.dynamical_systems import ConstVelocityDecreasingAtAttractor

from robot_avoidance.robot_arm_avoider import RobotArmAvoider
from robot_avoidance.model_robot import RobotArm2D
from robot_avoidance.analytic_evaluation_jacobian import (
    analytic_evaluation_jacobian,
)


class RobotAnimation:
    def __init__(self):
        self.animation_paused = False

    def start_animator(
        self,
        my_robot,
        initial_dynamics,
        obstacle_environment,
        x_lim=[-1.5, 2],
        y_lim=[-0.5, 2.5],
        it_max=1000,
        delta_time=0.03,
        dt_sleep=0.1,
    ):
        vel_trimmer = ConstVelocityDecreasingAtAttractor(
            const_velocity=1.0,
            distance_decrease=0.1,
            attractor_position=initial_dynamics.attractor_position,
        )

        dynamic_avoider = DynamicModulationAvoider(
            initial_dynamics=initial_dynamics, environment=obstacle_environment
        )

        main_avoider = RobotArmAvoider(
            obstacle_avoider=dynamic_avoider, robot_arm=my_robot
        )

        # plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)
        # dt_sleep = 0.001

        dim = 2
        position_list = np.zeros((dim, it_max))
        trajectory_joint = np.zeros((dim, it_max, my_robot.n_joints))
        joint_control = np.zeros(my_robot.n_joints)

        fig, ax = plt.subplots(figsize=(10, 8))
        my_robot.draw_robot(ax=ax)
        cid = fig.canvas.mpl_connect("button_press_event", self.on_click)

        ii = 0
        while ii < it_max:
            # for ii in range(it_max):
            if self.animation_paused:
                plt.pause(dt_sleep)
                if not plt.fignum_exists(fig.number):
                    print("Stopped animation on closing of the figure..")
                    break
                continue

            ii += 1

            if ii > it_max:
                break
            # if ii > 0:
            # main_avoider.get_influence_weight_evaluation_points()

            position_list[:, ii] = my_robot.get_ee_in_base()

            for jj in range(my_robot.n_joints):
                trajectory_joint[:, ii, jj] = my_robot.get_joint_in_base(
                    level=jj + 1, relative_point_position=0.0
                )

            # desired_velocity = main_avoider.obstacle_avoider.evaluate(position=position_list[:, ii])

            # desired_velocity = initial_dynamics.evaluate(position=position_list[:, ii])

            # Modulate
            # desired_velocity = obs_avoidance_interpolation_moving(
            # position_list[:, ii],
            # desired_velocity, obs=obstacle_environment)

            # desired_velocity = vel_trimmer.limit(position=position_list[:, ii],
            # velocity=desired_velocity)

            # joint_control = my_robot.get_inverse_kinematics(desired_velocity)
            # print(f"iks: {joint_control=}")

            ax.clear()
            my_robot.draw_robot(ax=ax)

            my_robot.update_state(
                joint_velocity_control=joint_control,
                delta_time=delta_time,
                check_max_velocity=False,
            )

            joint_control = main_avoider.get_joint_avoidance_velocity(ax=ax)
            # print(f"mod: {joint_control=}")

            # if ii <= 5:
            # ax.legend()
            # t = time.process_time()

            my_robot.draw_robot(ax=ax)
            # plt.plot(position_list[0, :ii], position_list[1, :ii], '--', color='k')
            plt.plot(
                position_list[0, : ii + 1],
                position_list[1, : ii + 1],
                ".",
                color="k",
            )
            for jj in range(my_robot.n_joints):
                plt.plot(
                    trajectory_joint[0, : ii + 1, jj],
                    trajectory_joint[1, : ii + 1, jj],
                    ".",
                    color="k",
                )

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

            ax.plot(
                initial_dynamics.attractor_position[0],
                initial_dynamics.attractor_position[1],
                "k*",
            )
            ax.grid()

            ax.set_aspect("equal", adjustable="box")
            # elapsed_time = time.process_time() - t
            # print('time_plt', elapsed_time)

            if np.sum(np.abs(joint_control)) < 1e-2:
                print(f"Converged at it={ii}")
                break

            if not ii % 10:
                print(f"it={ii}")

            plt.pause(dt_sleep)
            if not plt.fignum_exists(fig.number):
                print("Stopped animation on closing of the figure..")
                break

        print("Done")

    def on_click(self, event):
        if self.animation_paused:
            self.animation_paused = False
        else:
            self.animation_paused = True


def multi_robot_picture(
    my_robot,
    initial_dynamics,
    obstacle_environment,
    x_lim=[-1.5, 2],
    y_lim=[-0.5, 2.5],
    it_max=1000,
    delta_time=0.03,
    dt_sleep=0.1,
    save_figure=False,
    it_draw_list=[],
    figure_name=None,
):

    vel_trimmer = ConstVelocityDecreasingAtAttractor(
        const_velocity=1.0,
        distance_decrease=0.1,
        attractor_position=initial_dynamics.attractor_position,
    )

    dynamic_avoider = DynamicModulationAvoider(
        initial_dynamics=initial_dynamics, environment=obstacle_environment
    )

    main_avoider = RobotArmAvoider(obstacle_avoider=dynamic_avoider, robot_arm=my_robot)

    dim = 2
    position_list = np.zeros((dim, it_max))
    trajectory_joint = np.zeros((dim, it_max, my_robot.n_joints))
    joint_control = np.zeros(my_robot.n_joints)

    fig, ax = plt.subplots(figsize=(10, 8))

    ii = 0
    while ii < it_max:
        ii += 1
        if ii > it_max:
            break

        position_list[:, ii] = my_robot.get_ee_in_base()

        for jj in range(my_robot.n_joints):
            trajectory_joint[:, ii, jj] = my_robot.get_joint_in_base(
                level=jj + 1, relative_point_position=0.0
            )

        my_robot.update_state(
            joint_velocity_control=joint_control,
            delta_time=delta_time,
            check_max_velocity=False,
        )

        joint_control = main_avoider.get_joint_avoidance_velocity()

        if np.sum(np.abs(joint_control)) < 1e-2:
            print(f"Converged at it={ii}")
            break

        if not ii % 10:
            print(f"it={ii}")

        if ii in it_draw_list:
            my_robot.draw_robot(
                ax=ax,
                link_line_width=2,
                link_color="black",
                joint_marker_size=4,
                joint_color="black",
            )

    plt.plot(
        position_list[0, 1 : ii + 1],
        position_list[1, 1 : ii + 1],
        ":",
        color="k",
    )
    # for jj in range(my_robot.n_joints):
    # plt.plot(trajectory_joint[0, :ii+1, jj],
    # trajectory_joint[1, :ii+1, jj], '.', color='k')

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

    ax.plot(
        initial_dynamics.attractor_position[0],
        initial_dynamics.attractor_position[1],
        "k*",
    )
    ax.grid()

    if save_figure:
        if figure_name is None:
            figure_name = "2d_robot_arm_edge_avoidance"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

    print("Done")


def simple_2link_robot(evaluate_jacobian=False):
    my_robot = RobotArm2D(link_lengths=np.array([1, 1]))
    my_robot.set_joint_state(np.array([30, 60]), input_unit="deg")

    if evaluate_jacobian:
        analytic_evaluation_jacobian(my_robot)

    from robot_avoidance.jacobians.robot_arm_2d import _get_jacobian

    my_robot.set_jacobian(function=_get_jacobian)

    wall_width = 0.02
    margin_absolut = 0.05
    edge_points = [
        [0.5 - wall_width, 1 + margin_absolut],
        [0.5 + wall_width, 1 + margin_absolut],
        [0.5 + wall_width, 2.0 - wall_width],
        [1.5, 2.0 - wall_width],
        [1.5, 2.0 + wall_width],
        [0.5 - wall_width, 2.0 + wall_width],
    ]

    center_position = np.array([0.5, 2.0])
    attractor_position = np.array([-1, 0])

    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Polygon(
            edge_points=np.array(edge_points).T,
            center_position=center_position,
            margin_absolut=margin_absolut,
            absolute_edge_position=True,
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    obstacle_environment.append(
        Cuboid(
            axes_length=[0.9, wall_width * 2],
            center_position=np.array([1, (-1) * (margin_absolut + wall_width)]),
            margin_absolut=margin_absolut,
            tail_effect=False,
        )
    )

    initial_dynamics = LinearSystem(attractor_position=attractor_position)

    # my_robot.set_joint_state(np.array([0.05009573, 2.27274197]))
    # my_robot.set_joint_state(np.array([0.07828263, 2.4501339 ]))

    # RobotAnimation().start_animator(
    # my_robot, initial_dynamics, obstacle_environment,
    # x_lim=[-1.5, 2], y_lim=[-0.5, 2.5],
    # delta_time=0.05,
    # )

    multi_robot_picture(
        my_robot,
        initial_dynamics,
        obstacle_environment,
        x_lim=[-1.5, 2],
        y_lim=[-0.5, 2.5],
        delta_time=0.05,
        save_figure=True,
        it_draw_list=[1, 25, 60, 150],
    )


def three_link_robot_around_block(evaluate_jacobian=False):
    my_robot = RobotArm2D(link_lengths=np.array([1, 1, 1]))
    my_robot.set_joint_state(np.array([70 + 90, -30, -30]), input_unit="deg")

    # (!)
    # my_robot.set_joint_state(np.array([ 1.84782286, -1.72426276, -0.2256878 ]))
    my_robot.name = "robot_arm_3link"

    if evaluate_jacobian:
        analytic_evaluation_jacobian(my_robot)

    from robot_avoidance.jacobians.robot_arm_3link import _get_jacobian

    my_robot.set_jacobian(function=_get_jacobian)

    margin_absolut = 0.1
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Cuboid(
            axes_length=[0.4, 2.4],
            center_position=np.array([0.2, 2.4]),
            # center_position=np.array([-0.2, 2.4]),
            margin_absolut=margin_absolut,
            orientation=-30 * pi / 180,
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    obstacle_environment.append(
        Cuboid(
            axes_length=[0.4, 1.3],
            center_position=np.array([1.2, 0.25]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=margin_absolut,
            orientation=10 * pi / 180,
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    # obstacle_environment = ObstacleContainer()

    attractor_position = np.array([1.5, 1.4])
    # attractor_position = np.array([1.7, 0.7])
    initial_dynamics = LinearSystem(
        attractor_position=attractor_position,
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    # RobotAnimation().start_animator(
    # my_robot, initial_dynamics, obstacle_environment,
    # x_lim=[-3, 3], y_lim=[-0.5, 3.5],
    # delta_time=0.05)

    multi_robot_picture(
        my_robot,
        initial_dynamics,
        obstacle_environment,
        x_lim=[-3, 3],
        y_lim=[-0.5, 3.5],
        delta_time=0.05,
        save_figure=True,
        it_draw_list=[1, 25, 40, 60, 100],
        figure_name="three_link_robot_around_block",
    )


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    simple_2link_robot()
    # three_link_robot_around_block()
