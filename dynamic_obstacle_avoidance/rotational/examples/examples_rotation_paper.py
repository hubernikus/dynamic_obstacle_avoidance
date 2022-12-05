#!/USSR/bin/python3
""" Script to show lab environment on computer """
# Author: Lukas Huber
# License: BSD (c) 2021

import warnings
import copy
from functools import partial
import math

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

from vartools.dynamical_systems import LinearSystem, QuadraticAxisConvergence
from dynamic_obstacle_avoidance.rotational.dynamics import WavyLinearDynamics

# from vartools.dynamical_systems import SinusAttractorSystem
# from vartools.dynamical_systems import BifurcationSpiral

from vartools.dynamical_systems import plot_dynamical_system_streamplot

from dynamic_obstacle_avoidance.obstacles import Ellipse, StarshapedFlower
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving
from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

from dynamic_obstacle_avoidance.rotational.multiboundary_container import (
    MultiBoundaryContainer,
)

from dynamic_obstacle_avoidance.rotational.rotation_container import RotationContainer
from dynamic_obstacle_avoidance.rotational.rotational_avoidance import (
    obstacle_avoidance_rotational,
)
from dynamic_obstacle_avoidance.rotational.rotational_avoider import RotationalAvoider

from dynamic_obstacle_avoidance.visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)

from vartools.visualization import VectorfieldPlotter

# plt.close('all')
plt.ion()


def inverted_star_obstacle_avoidance(visualize=False, save_figure=False):
    obstacle_list = RotationContainer()

    obstacle_list.append(
        StarshapedFlower(
            center_position=np.array([0, 0]),
            radius_magnitude=2,
            number_of_edges=5,
            radius_mean=7,
            orientation=0.0 / 180 * pi,
            tail_effect=False,
            is_boundary=True,
        )
    )
    # initial_dynamics = WavyLinearDynamics(attractor_position=np.array([0, 0]))
    initial_dynamics = WavyLinearDynamics(attractor_position=np.array([6.8, -1]))
    convergence_dynamics = LinearSystem(
        attractor_position=initial_dynamics.attractor_position
    )

    obstacle_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
        convergence_system=convergence_dynamics,
    )

    if visualize:
        plt.close("all")

        x_lim = [-10, 10]
        y_lim = [-10, 10]

        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

        Simulation_vectorFields(
            x_lim,
            y_lim,
            n_resolution=100,
            # n_resolution=20,
            obstacle_list=obstacle_list,
            # obstacle_list=[],
            saveFigure=False,
            noTicks=True,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=initial_dynamics.evaluate,
            obs_avoidance_func=obstacle_avoider.avoid,
            automatic_reference_point=False,
            pos_attractor=initial_dynamics.attractor_position,
            fig_and_ax_handle=(fig, ax),
            # Quiver or Streamplot
            show_streamplot=True,
            # show_streamplot=False,
        )

        if save_figure:
            fig_name = "wavy_nonlinear_ds_in_star_obstacle"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)


def test_single_circle_linear_inverted(visualize=False, save_figure=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([3.5, 2]),
            is_boundary=True,
            orientation=30 / 180.0 * math.pi,
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([2.2, 1]))
    convergence_dynamics = initial_dynamics

    # main_avoider = RotationalAvoider()
    # my_avoider = partial(main_avoider.avoid, convergence_radius=math.pi)

    obstacle_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
        convergence_system=convergence_dynamics,
    )
    obstacle_avoider.convergence_radius = math.pi

    if visualize:
        plt.close("all")

        x_lim = [-3.4, 3.4]
        y_lim = [-3.4, 3.4]

        fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

        Simulation_vectorFields(
            x_lim=x_lim,
            y_lim=y_lim,
            n_resolution=100,
            # n_resolution=20,
            obstacle_list=obstacle_avoider.obstacle_environment,
            # obstacle_list=[],
            saveFigure=False,
            noTicks=True,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=obstacle_avoider.initial_dynamics.evaluate,
            obs_avoidance_func=obstacle_avoider.avoid,
            automatic_reference_point=False,
            pos_attractor=initial_dynamics.attractor_position,
            fig_and_ax_handle=(fig, ax),
            # Quiver or Streamplot
            show_streamplot=True,
            # show_streamplot=False,
        )

        if save_figure:
            fig_name = "linear_dynamics_in_repulsive_ellipse_wall"
            fig.savefig("figures/" + fig_name + figtype, bbox_inches="tight", dpi=300)


if (__name__) == "__main__":
    figtype = ".pdf"
    # figtype = ".png"

    plt.ion()

    # inverted_star_obstacle_avoidance(visualize=True, save_figure=True)
    test_single_circle_linear_inverted(visualize=True, save_figure=True)

    print("--- done ---")
