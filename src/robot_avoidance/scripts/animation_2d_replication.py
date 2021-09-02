"""
Replcication of 2D-animation of paper
"""
# Author: Lukas Hubero
import time
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from vartools.dynamical_systems import ConstVelocityDecreasingAtAttractor

from robot_avoidance.robot_arm_avoider import RobotArmAvoider
from robot_avoidance.model_robot import RobotArm2D
from robot_avoidance.analytic_evaluation_jacobian import analytic_evaluation_jacobian


def simple_2d_environment(my_robot):
    x_lim = [-1.5, 2]
    y_lim = [-0.5, 2.5]

    wall_width = 0.02
    margin_absolut = 0.05
    edge_points = [[0.5-wall_width, 1+margin_absolut],
                   [0.5+wall_width, 1+margin_absolut],
                   [0.5+wall_width, 2.0-wall_width],
                   [1.5, 2.0-wall_width],
                   [1.5, 2.0+wall_width],
                   [0.5-wall_width, 2.0+wall_width],
                   ]
    center_position = np.array([0.5, 2.0])
    attractor_position = np.array([-1, 0])
    
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Polygon(edge_points=np.array(edge_points).T,
                center_position=center_position,
                margin_absolut=margin_absolut,
                absolute_edge_position=True,
                tail_effect=False,
                ))

    obstacle_environment.append(
        Cuboid(axes_length=[1, wall_width*2],
               center_position=np.array(
        [1, (-1)*(margin_absolut+wall_width)]),
               margin_absolut=margin_absolut,
               tail_effect=False,
               ))
    linear_ds = LinearSystem(attractor_position=attractor_position)
    
    vel_trimmer = ConstVelocityDecreasingAtAttractor(
        const_velocity=1.0, distance_decrease=0.1,
        attractor_position=linear_ds.attractor_position)
    
    # plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)
    # my_robot.draw_robot(ax=ax)

    it_max = 1000
    dt_sleep = 0.001
    delta_time = 0.03
    
    dim = 2
    
    position_list = np.zeros((dim, it_max))
    trajectory_joint = np.zeros((dim, it_max))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for ii in range(it_max):
        position_list[:, ii] = my_robot.get_ee_in_base()
        trajectory_joint[:, ii] = my_robot.get_joint_in_base(level=2)
        
        desired_velocity =  linear_ds.evaluate(position=position_list[:, ii])

        # Modulate
        desired_velocity = obs_avoidance_interpolation_moving(
            position_list[:, ii],
            desired_velocity, obs=obstacle_environment)
        
        desired_velocity = vel_trimmer.limit(position=position_list[:, ii],
                                             velocity=desired_velocity)

        if LA.norm(desired_velocity) < 1e-1:
            print(f"Converged at it={ii}")
            break
        
        joint_control = my_robot.get_inverse_kinematics(desired_velocity)
        
        my_robot.update_state(joint_velocity_control=joint_control, delta_time=delta_time)
        
        # t = time.process_time()
        ax.clear()
        my_robot.draw_robot(ax=ax)
        # plt.plot(position_list[0, :ii], position_list[1, :ii], '--', color='k')
        plt.plot(position_list[0, :ii], position_list[1, :ii], '.', color='k')
        plt.plot(trajectory_joint[0, :ii], trajectory_joint[1, :ii], '.', color='k')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

        ax.plot(attractor_position[0], attractor_position[1], 'k*')
        ax.grid()
    
        ax.set_aspect('equal', adjustable='box')
        # elapsed_time = time.process_time() - t
        # print('time_plt', elapsed_time)
        
        plt.pause(dt_sleep)

        if ii % 10:
            print(f"it={ii}")
            
        if not plt.fignum_exists(fig.number):
            print("Stopped animation on closing of the figure..")
            break

    print("Done")


if (__name__) == "__main__":
    plt.close('all')
    plt.ion()

    my_robot = RobotArm2D(link_lengths=np.array([1, 1]))
    my_robot.set_joint_state(np.array([30, 60]),
                             input_unit='deg')

    evaluate_jacobian = False
    if evaluate_jacobian:
        analytic_evaluation_jacobian(my_robot)
        
    from robot_avoidance.jacobians.robot_arm_2d import _get_jacobian
    my_robot.set_jacobian(function=_get_jacobian)
    
    simple_2d_environment(my_robot=my_robot)
    
    # dummy_robot_avoidance()
