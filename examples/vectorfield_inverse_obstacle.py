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

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
)

from vartools.dynamical_systems import LinearSystem
from vartools.dynamical_systems import ConstVelocityDecreasingAtAttractor


def simple_vectorfield_inside():
    """Simple robot avoidance."""
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Cuboid(
            center_position=np.array([0, 0]),
            axes_length=np.array([7, 7]),
            is_boundary=True,
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([1.5, 1.5]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    x_lim = [-4.1, 4.1]
    y_lim = [-4.1, 4.1]

    Simulation_vectorFields(
        x_range=x_lim,
        y_range=y_lim,
        point_grid=100,
        obstacle_list=obstacle_environment,
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

    simple_vectorfield_inside()
