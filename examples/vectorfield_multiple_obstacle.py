#!/USSR/bin/python3
"""
Vector fields of different setups
"""
# Author: LukasHuber
# Email: lukas.huber@epfl.ch
# Created:  2021-09-23
import time
from math import pi

import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes, CuboidXd

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
)

from vartools.dynamical_systems import LinearSystem
from vartools.dynamical_systems import ConstVelocityDecreasingAtAttractor


def simple_vectorfield():
    obstacle_environment = ObstacleContainer()

    obstacle_environment.append(
        EllipseWithAxes(
            axes_length=[0.6, 1.3],
            center_position=np.array([-0.2, 2.4]),
            margin_absolut=0,
            orientation=-30 * pi / 180,
            tail_effect=False,
        )
    )

    obstacle_environment.append(
        CuboidXd(
            axes_length=[0.4, 1.3],
            center_position=np.array([1.2, 0.25]),
            margin_absolut=0.5,
            orientation=10 * pi / 180,
            tail_effect=False,
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([2.0, 1.8]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    x_lim = [-3.2, 3.2]
    y_lim = [-0.2, 4.4]

    Simulation_vectorFields(
        x_lim,
        y_lim,
        point_grid=100,
        obs=obstacle_environment,
        pos_attractor=initial_dynamics.attractor_position,
        dynamical_system=initial_dynamics.evaluate,
        noTicks=True,
        automatic_reference_point=False,
        show_streamplot=True,
        draw_vectorField=True,
        normalize_vectors=False,
    )

    plt.grid()
    plt.show()


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    simple_vectorfield()
