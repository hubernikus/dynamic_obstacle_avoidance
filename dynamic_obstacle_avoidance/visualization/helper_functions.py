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

from dynamic_obstacle_avoidance.avoidance import DynamicModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
)

from vartools.dynamical_systems import LinearSystem
from vartools.dynamical_systems import ConstVelocityDecreasingAtAttractor


def vectorfield(
    obstacle_environment, x_lim=[-3.2, 3.2], y_lim=[-1.4, 4.4], n_resolution=30
):
    dim = 2
    nx = n_resolution
    ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    fig, axs = plt.subplots(1, 2, figsize=(16, 7))

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    n_obs = len(obstacle_environment)

    vel_init = np.zeros((positions.shape))
    vel_mod = np.zeros((positions.shape))

    initial_dynamics = LinearSystem(
        attractor_position=np.array([0.0, 0.0]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )
    dynamic_avoider = DynamicModulationAvoider(
        initial_dynamics=initial_dynamics, environment=obstacle_environment
    )

    for it_obs, obs in enumerate(obstacle_environment):
        for ii in range(positions.shape[1]):
            vel_mod[:, ii] = dynamic_avoider.evaluate(positions[:, ii - 1])

        axs[it_obs].quiver(
            positions[0, :],
            positions[1, :],
            vel_mod[0, :],
            vel_mod[1, :],
            color="black",
        )

        obs.draw_obstacle()

        x_obs_sf = obs.boundary_points_margin_global_closed
        axs[it_obs].plot(x_obs_sf[0, :], x_obs_sf[1, :], "--", color="k")

        axs[it_obs].axis("equal")
        axs[it_obs].grid(True)


def normal_and_vector_field(
    obstacle_environment, x_lim=[-3.2, 3.2], y_lim=[-1.4, 4.4], n_resolution=30
):
    dim = 2
    nx = n_resolution
    ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    fig, axs = plt.subplots(1, 2, figsize=(16, 7))

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    n_obs = len(obstacle_environment)

    gamma_vals = np.zeros((positions.shape[1], n_obs))
    normals = np.zeros((dim, positions.shape[1], n_obs))

    for it_obs, obs in enumerate(obstacle_environment):
        for ii in range(positions.shape[1]):
            gamma_vals[ii, it_obs] = obs.get_gamma(
                positions[:, ii], in_global_frame=True
            )

            normals[:, ii, it_obs] = obs.get_normal_direction(
                positions[:, ii], in_global_frame=True
            )

        axs[it_obs].contourf(
            x_vals, y_vals, gamma_vals[:, it_obs].reshape(n_resolution, n_resolution)
        )

        axs[it_obs].quiver(
            positions[0, :],
            positions[1, :],
            normals[0, :, it_obs],
            normals[1, :, it_obs],
            color="black",
        )

        obs.draw_obstacle()

        x_obs_sf = obs.boundary_points_margin_global_closed
        axs[it_obs].plot(x_obs_sf[0, :], x_obs_sf[1, :], "--", color="k")

        axs[it_obs].axis("equal")
        axs[it_obs].grid(True)


def simple_vectorfield(obstacle_environment, x_lim=[-3.2, 3.2], y_lim=[-0.2, 4.4]):
    """Simple robot avoidance."""
    initial_dynamics = LinearSystem(
        attractor_position=np.array([0.0, 0.0]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    Simulation_vectorFields(
        x_lim,
        y_lim,
        point_grid=50,
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

    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Cuboid(
            axes_length=[1.6, 0.7],
            center_position=np.array([0.0, 0.0]),
            margin_absolut=0,
            orientation=-0,
            tail_effect=False,
            repulsion_coeff=1.0,
        )
    )

    obstacle_environment.append(
        Cuboid(
            axes_length=[0.5, 0.5],
            center_position=np.array([0.0, 0.6]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0,
            orientation=0,
            tail_effect=False,
            repulsion_coeff=1.0,
        )
    )

    # simple_vectorfield(obstacle_environment)
    # gamma_function(obstacle_environment)
    vectorfield(obstacle_environment)
