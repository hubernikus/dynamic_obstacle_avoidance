"""
Test script for obstacle avoidance algorithm
Test normal formation
"""
# Author: Lukas Huber
# Created: 2022-05-19
# Email: lukas.huber@epfl.ch

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid

from vartools.dynamical_systems import LinearSystem
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider


def test_triple_obstacles():

    radius_length = 1.2

    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Ellipse(
            axes_length=[radius_length, radius_length, radius_length],
            center_position=np.array([1, 0, 0]),
            linear_velocity=np.zeros(3),
            margin_absolut=0,
            tail_effect=False,
            repulsion_coeff=1.0,
        )
    )

    obstacle_environment.append(
        Ellipse(
            axes_length=[radius_length, radius_length, radius_length],
            center_position=np.array([-1, 0, 0]),
            linear_velocity=np.zeros(3),
            margin_absolut=0,
            tail_effect=False,
            repulsion_coeff=1.0,
        )
    )

    axes_length = 2.2
    obstacle_environment.append(
        Cuboid(
            axes_length=[axes_length, axes_length, axes_length],
            center_position=np.array([0.1, 2.2, 0]),
            linear_velocity=np.zeros(3),
            margin_absolut=0,
            tail_effect=False,
            repulsion_coeff=1.0,
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([1.0, 1.0, 0.1]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    eval_position = np.array([0, 0, 1])

    avoider = ModulationAvoider(obstacle_environment=obstacle_environment)

    init_vel = initial_dynamics.evaluate(eval_position)
    mod_vel = avoider.avoid(eval_position, velocity=init_vel)

    # breakpoint()
    print("Done")


if (__name__) == "__main__":
    test_triple_obstacles()
