"""
Example to stage an robot-arm obstacle avoidance.
"""
# Author: Lukas Hubero
import time
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import FlatPlane, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem

from robot_arm_avoider import RobotArmAvoider
from model_robot import ModelRobot2D


def dummy_robot_movement():
    x_lim = [-0.2, 4.5]
    y_lim = [-4, 4]

    fig, ax = plt.subplots(figsize=(8, 10))

    my_robot = ModelRobot2D()
    my_robot.set_joint_state(np.array([0, 30, -10, -20]), input_unit="deg")

    # position = my_robot.get_ee_in_base()

    # attractor_position = np.array([3.0, -1.0])
    attractor_position = np.array([1.0, -1.0])
    # attractor_position = np.array([3.0, 3.0])

    linear_ds = LinearSystem(attractor_position=attractor_position)

    it_max = 1000
    dt_sleep = 0.001
    delta_time = 0.03

    dim = 2

    position_list = np.zeros((dim, it_max))
    for ii in range(it_max):
        position_list[:, ii] = my_robot.get_ee_in_base()
        desired_velocity = linear_ds.evaluate(position=position_list[:, ii])

        if LA.norm(desired_velocity) < 1e-1:
            print(f"Converged at it={ii}")
            break

        joint_control = my_robot.get_inverse_kinematics(desired_velocity)

        my_robot.update_state(
            joint_velocity_control=joint_control, delta_time=delta_time
        )

        # t = time.process_time()
        ax.clear()
        my_robot.draw_robot(ax=ax)
        plt.plot(position_list[0, :ii], position_list[1, :ii], "-", color="k")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        ax.set_aspect("equal", adjpustable="box")
        # elapsed_time = time.process_time() - t
        # print('time_plt', elapsed_time)

        plt.pause(dt_sleep)

        if ii % 10:
            print(f"it={ii}")

        if not plt.fignum_exists(fig.number):
            print("Stopped animation on closing of the figure..")
            break

    print("Done.")


def dummy_robot_avoidance():
    x_lim = [-0.2, 4.5]
    y_lim = [-4, 4]
    # x_lim = [-10, 10]
    # y_lim = [-10, 10]

    fig, ax = plt.subplots(figsize=(8, 10))

    my_robot = ModelRobot2D()
    my_robot.set_joint_state(np.array([0, 30, -10, -20]), input_unit="deg")

    my_robot.draw_robot(ax=ax)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_aspect("equal", adjustable="box")

    # Creat environment
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        FlatPlane(
            center_position=np.array([0, 0]),
            normal=np.array([1, 0]),
            width=10,
            height=1,
        )
    )

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([2.1, 0]),
            axes_length=np.array([0.5, 0.5]),
        )
    )

    for obs in obstacle_environment:
        obs.draw_obstacle()

    plot_obstacles(ax, obstacle_environment, x_lim, y_lim)
    plt.show()


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()
    dummy_robot_movement()

    # dummy_robot_avoidance()
