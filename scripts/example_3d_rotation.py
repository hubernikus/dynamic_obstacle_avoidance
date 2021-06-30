#!/USSR/bin/python3
""" Script to show lab environment on computer """
# Author: Lukas Huber
# License: BSD (c) 2021

import warnings
import copy

import numpy as np
from numpy import pi

from scipy.spatial.transform import Rotation # scipy rotation

from vartools.dynamical_systems import LinearSystem, SpiralStable

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.avoidance import obstacle_avoidance_rotational
from dynamic_obstacle_avoidance.containers import MultiBoundaryContainer, RotationContainer
from dynamic_obstacle_avoidance.visualization.plot_3d_trajectory import plot_obstacles_and_trajectory_3d


def evaluate_single_position(
    position,
    InitialDynamics,
    ObstacleContainer,
    func_obstacle_avoidance):
    
    initial_velocity = InitialDynamics.evaluate(position)
    modulated_velocity = func_obstacle_avoidance(position, initial_velocity, ObstacleContainer)
    
    return modulated_velocity

def example_single_ellipse_3d():
    ObstacleEnvironment = RotationContainer()
    # ObstacleEnvironment.append(
    #     Ellipse(
    #     center_position=np.array([0, 1, -0.4]), 
    #     axes_length=np.array([0.1, 0.3, 0.2]),
    #     orientation=Rotation.from_rotvec([0.1, 0.4, 0.3]),
    #     tail_effect=False,
    #     )
    # )
    ObstacleEnvironment.append(
        Ellipse(
        center_position=np.array([0.5, -1, 0.3]), 
        axes_length=np.array([0.3, 0.3, 0.3]),
        orientation=Rotation.from_rotvec([0, 0, 0]),
        tail_effect=False,
        )
    )

    # ObstacleEnvironment.append(
    #     Ellipse(
    #     center_position=np.array([-0.5, 1, 1]), 
    #     axes_length=np.array([0.4, 0.3, 0.2]),
    #     orientation=Rotation.from_rotvec([0, 0, 0]),
    #     tail_effect=False,
    #     )
    # )
    # ObstacleEnvironment = RotationContainer()
    
    InitialDynamics = SpiralStable()
    ObstacleEnvironment.set_convergence_directions(
        ConvergingDynamics=LinearSystem(InitialDynamics.attractor_position))

    # test_pos = np.array([ 0.24685765, -0.49288152, -0.35809686])
    # evaluate_single_position(test_pos,
    #                          InitialDynamics=InitialDynamics,
    #                          ObstacleContainer=ObstacleEnvironment,
    #                          func_obstacle_avoidance=obstacle_avoidance_rotational)
    # if True:
        # return 

    # start_pos1 = [0.1, 0.1, 0.1]
    start_pos1 = [1.4, 1.4, 1.0]
    start_positions = np.array([start_pos1]).T
    
    xyz_lim=[-1.5, 1.5]
    # plot_obstacles_and_trajectory_3d(ObstacleEnvironment)
    plot_obstacles_and_trajectory_3d(
        ObstacleEnvironment,
        func_obstacle_avoidance=obstacle_avoidance_rotational,
        delta_time=0.001,
        InitialDynamics=InitialDynamics,
        start_positions=start_positions,
        x_lim=xyz_lim, y_lim=xyz_lim, z_lim=xyz_lim)
                          # x_lim=[-2, 2], y_lim=[-2, 2], z_lim=[-2, 2])
    
if (__name__)=="__main__":
    example_single_ellipse_3d()
