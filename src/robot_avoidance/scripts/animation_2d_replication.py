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
# from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving
from dynamic_obstacle_avoidance.avoidance import DynamicModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from vartools.dynamical_systems import ConstVelocityDecreasingAtAttractor

from robot_avoidance.robot_arm_avoider import RobotArmAvoider
from robot_avoidance.model_robot import RobotArm2D
from robot_avoidance.analytic_evaluation_jacobian import analytic_evaluation_jacobian


def run_animation(
    my_robot, initial_dynamics, obstacle_environment,
    x_lim=[-1.5, 2], y_lim=[-0.5, 2.5],
    it_max=1000, delta_time=0.03,
    ):
    main_avoider = RobotArmAvoider(initial_dynamics, my_robot, obstacle_environment)

    vel_trimmer = ConstVelocityDecreasingAtAttractor(
        const_velocity=1.0, distance_decrease=0.1,
        attractor_position=initial_dynamics.attractor_position)

    dynamic_avoider = DynamicModulationAvoider(
        initial_dynamics=initial_dynamics, environment=obstacle_environment)
    
    # plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)
    # my_robot.draw_robot(ax=ax)
    # dt_sleep = 0.001
    dt_sleep = 0.001
    dim = 2
    
    position_list = np.zeros((dim, it_max))
    trajectory_joint = np.zeros((dim, it_max, my_robot.n_joints))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    for ii in range(it_max):
        # if ii > 0:
            # main_avoider.get_influence_weight_evaluation_points()
        
        position_list[:, ii] = my_robot.get_ee_in_base()
        
        for jj in range(my_robot.n_joints):
            trajectory_joint[:, ii, jj] = my_robot.get_joint_in_base(
                level=jj+1, relative_point_position=0.0)

        dynamic_avoider = dynamic_avoider.evaluate(position=position_list[:, ii])
        
        # desired_velocity = initial_dynamics.evaluate(position=position_list[:, ii])

        # Modulate
        # desired_velocity = obs_avoidance_interpolation_moving(
            # position_list[:, ii],
            # desired_velocity, obs=obstacle_environment)
        
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
        for jj in range(my_robot.n_joints):
            plt.plot(trajectory_joint[0, :ii, jj], trajectory_joint[1, :ii, jj], '.', color='k')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

        ax.plot(initial_dynamics.attractor_position[0],
                initial_dynamics.attractor_position[1], 'k*')
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

def simple_2link_robot(evaluate_jacobian=False):
    my_robot = RobotArm2D(link_lengths=np.array([1, 1]))
    my_robot.set_joint_state(np.array([30, 60]),
                             input_unit='deg')
    
    if evaluate_jacobian:
        analytic_evaluation_jacobian(my_robot)
        
    from robot_avoidance.jacobians.robot_arm_2d import _get_jacobian
    
    my_robot.set_jacobian(function=_get_jacobian)
    
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
    
    initial_dynamics = LinearSystem(attractor_position=attractor_position)

    run_animation(
        my_robot, initial_dynamics, obstacle_environment,
        x_lim=[-1.5, 2], y_lim=[-0.5, 2.5])


def three_link_robot_around_block(evaluate_jacobian=False):
    my_robot = RobotArm2D(link_lengths=np.array([1, 1, 1]))
    my_robot.set_joint_state(np.array([70+90, -30, -30]),
                             input_unit='deg')
    my_robot.name = "robot_arm_3link"

    if evaluate_jacobian:
        analytic_evaluation_jacobian(my_robot)
        
    from robot_avoidance.jacobians.robot_arm_3link import _get_jacobian
    my_robot.set_jacobian(function=_get_jacobian)

    margin_absolut = 0.1
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Cuboid(center_position=np.array([0.2, 2.4]),
               axes_length=[0.4, 2.4],
               margin_absolut=margin_absolut,
               orientation=-30*pi/180,
               tail_effect=False,
               ))
    
    obstacle_environment.append(
        Cuboid(center_position=np.array([1.2, 0.25]),
               axes_length=[0.4, 1.45],
               margin_absolut=margin_absolut,
               orientation=0*pi/180,
               tail_effect=False,
               ))
    
    # attractor_position = np.array([2, 2])
    attractor_position = np.array([2, 0.7])
    initial_dynamics = LinearSystem(attractor_position=attractor_position) 

    run_animation(
        my_robot, initial_dynamics, obstacle_environment,
        x_lim=[-3, 3], y_lim=[-0.5, 3.5],
        delta_time=0.05)


if (__name__) == "__main__":
    plt.close('all')
    plt.ion()
    
    simple_2link_robot()
    three_link_robot_around_block()
