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

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
)

from vartools.dynamical_systems import LinearSystem
from vartools.dynamical_systems import ConstVelocityDecreasingAtAttractor


def simple_vectorfield_around_circle_zoom():
    """Simple vectorfield around robot"""
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        EllipseWithAxes(
            axes_length=[1.0, 1.0],
            center_position=np.array([0.2, -0.2]),
            margin_absolut=0,
            orientation=0 * pi / 180,
            tail_effect=False,
            repulsion_coeff=1.0,
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([3.0, 1.4]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    x_lim = [-3.5, 0.5]
    y_lim = [-1.0, 2.0]

    n_resolution = 100
    dim = 2

    fig, ax = Simulation_vectorFields(
        x_lim,
        y_lim,
        point_grid=n_resolution,
        obs=obstacle_environment,
        pos_attractor=initial_dynamics.attractor_position,
        dynamical_system=initial_dynamics.evaluate,
        noTicks=True,
        showLabel=False,
        automatic_reference_point=False,
        show_streamplot=True,
        draw_vectorField=True,
        normalize_vectors=False,
        streamColor="black",
    )

    # Gamma Level
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], n_resolution),
        np.linspace(y_lim[0], y_lim[1], n_resolution),
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    gammas = np.zeros((positions.shape[1]))

    for ii in range(gammas.shape[0]):
        gammas[ii] = obstacle_environment[0].get_gamma(
            positions[:, ii], in_global_frame=True
        )

    cs = ax.contourf(
        positions[0, :].reshape(n_resolution, n_resolution),
        positions[1, :].reshape(n_resolution, n_resolution),
        gammas.reshape(n_resolution, n_resolution),
        np.arange(1.0, 5.0, 0.2),
        cmap=plt.get_cmap("hot"),
        extend="max",
        alpha=0.6,
        zorder=-3,
    )

    cbar = fig.colorbar(cs, fraction=0.033, pad=0.05)
    # plt.colorbar(im,

    plt.savefig("figures/simple_obstacle.png")
    plt.show()


def vectorfield_single_obstacle():
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        EllipseWithAxes(
            axes_length=[2.0, 4.0],
            center_position=np.array([0.0, -0.0]),
            margin_absolut=0,
            orientation=0 * pi / 180,
            tail_effect=False,
            repulsion_coeff=1.0,
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([2.0, 0.0]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    x_lim = [-2, 4]
    y_lim = [-3, 3.0]

    n_resolution = 100
    dim = 2

    fig, ax = Simulation_vectorFields(
        x_lim,
        y_lim,
        point_grid=n_resolution,
        obs=obstacle_environment,
        pos_attractor=initial_dynamics.attractor_position,
        dynamical_system=initial_dynamics.evaluate,
        noTicks=True,
        showLabel=False,
        automatic_reference_point=False,
        show_streamplot=True,
        draw_vectorField=True,
        normalize_vectors=False,
        streamColor="blue",
    )

    pass


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # simple_vectorfield_around_circle_zoom()
    vectorfield_single_obstacle()
