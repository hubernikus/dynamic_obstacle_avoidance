#!/USSR/bin/python3
""" Script to show lab environment on computer """
# Author: Lukas Huber
# License: BSD (c) 2021

import warnings
import copy

import numpy as np
from numpy import pi

from scipy.spatial.transform import Rotation # scipy rotation

from vartools.dynamical_systems import LinearSystem, SpiralStable, CircularLinear

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.avoidance import obstacle_avoidance_rotational
from dynamic_obstacle_avoidance.containers import MultiBoundaryContainer, RotationContainer
from dynamic_obstacle_avoidance.visualization.plot_3d_trajectory import plot_obstacles_and_trajectory_3d
from dynamic_obstacle_avoidance.visualization.plot_3d_trajectory import plot_obstacles_and_vector_levelz_3d


def evaluate_single_position(
    position,
    InitialDynamics,
    ObstacleContainer,
    func_obstacle_avoidance):
    
    initial_velocity = InitialDynamics.evaluate(position)
    modulated_velocity = func_obstacle_avoidance(position, initial_velocity, ObstacleContainer)
    
    print(f"pos={position}")
    # print(f'{initial_velocity=}')
    print(f"vel={modulated_velocity}")
    print("")
    
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
        center_position=np.array([0.8, 0.8, 0.5]), 
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

    
    InitialDynamics = SpiralStable(complexity_spiral=5)
    # InitialDynamics = CircularLinear(attractor_position=np.array([0, 0, 0]),
                                     # factor_circular=100, factor_linear=10,
                                     # maximum_velocity=1)
    InitialDynamics = LinearSystem(attractor_position=np.array([0, 0, 0]))
    ObstacleEnvironment.set_convergence_directions(
        ConvergingDynamics=LinearSystem(InitialDynamics.attractor_position))
    

    test_pos = np.array([ 0.24685765, -0.49288152, -0.35809686])
    # vel1=[-0.427 -0.272  0.214] to vel2=[-0.034  0.068 -0.543] at [0.941 0.279 0.412] to [0.941 0.28  0.407]
    test_poses = []
    # test_poses.append(np.array([0.941, 0.279, 0.412]))
    # test_poses.append(np.array([0.941, 0.28, 0.407]))
    test_poses.append(np.array([0.9453695848565005, 0.2816314518962237, 0.4098719224093203]))
    test_poses.append(np.array([0.9411042172145309, 0.2789087614555541, 0.41201641691364266]))
    if False:
    # for test_pos in test_poses:
        # np.set_printoptions(precision=8)
        evaluate_single_position(test_pos,
                                 InitialDynamics=InitialDynamics,
                                 ObstacleContainer=ObstacleEnvironment,
                                 func_obstacle_avoidance=obstacle_avoidance_rotational)

    # if True:
        # return
    
    start_pos1 = [1.2, 1.4, 0.7]
    start_positions = np.array([start_pos1]).T
    
    start_pos1 = [1.4, 1.4, 0.7]
    start_pos2 = [1.2, 1.4, 0.7]
    start_pos3 = [1.0, 1.4, 0.7]
    start_pos4 = [0.8, 1.4, 0.7]
    # start_positions = np.array([start_pos1, start_pos2, start_pos3, start_pos4, ]).T
    # n_points = 4
    # start_positions = np.array(np.linspace(start_pos1
    
    # xyz_lim=[-0.8, 0.8]
    xyz_lim=[-1.5, 1.5]
    # plot_obstacles_and_trajectory_3d(ObstacleEnvironment)

    # if False:
    if False:
        # ObstacleEnvironment = RotationContainer()
        plot_obstacles_and_trajectory_3d(
            ObstacleEnvironment,
            func_obstacle_avoidance=obstacle_avoidance_rotational,
            delta_time=0.01,
            # InitialDynamics=ObstacleEnvironment._ConvergenceDynamics[0],
            InitialDynamics=InitialDynamics,
            # start_positions=start_positions,
            n_points=10,
            n_max_it=400,
            # start_positions=start_positions,
            x_lim=xyz_lim, y_lim=xyz_lim, z_lim=[-1.0, 2.0])
                              # x_lim=[-2, 2], y_lim=[-2, 2], z_lim=[-2, 2])

    # if True:
    if True:
        # ObstacleEnvironment = RotationContainer()
        plot_obstacles_and_vector_levelz_3d(
            ObstacleEnvironment,
            func_obstacle_avoidance=obstacle_avoidance_rotational,
            InitialDynamics=InitialDynamics,
            n_grid=10,
            # z_value=0.5,
            x_lim=xyz_lim, y_lim=xyz_lim, z_lim=xyz_lim)

if (__name__)=="__main__":
    from datetime import datetime

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Starting @", current_time)
    example_single_ellipse_3d()
