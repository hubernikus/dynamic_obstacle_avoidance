#!/usr/bin/python3
""" Script to show lab environment on computer """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-04-09

# import warnings
import numpy as np

from vartools.visualization import VectorfieldPlotter
from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.rotational.rotation_container import RotationContainer

from dynamic_obstacle_avoidance.rotational.rotational_avoider import RotationalAvoider


def single_ellipse_nonlinear(n_resolution=10, save_figure=False):
    figure_name = "comparison_nonlinear_vectorfield"

    initial_dynamics = LinearSystem(
        stretching_factor=3,
        maximum_velocity=1.0,
        dimension=2,
        attractor_position=np.array([8, 0]),
    )

    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([3, 3]),
            orientation=0.0 / 180 * pi,
            is_boundary=False,
            tail_effect=False,
        )
    )
    obstacle_list.set_convergence_directions(initial_dynamics)

    my_plotter = VectorfieldPlotter(
        x_lim=[-10, 10],
        y_lim=[-10, 10],
        figsize=(4.5, 4.0),
        attractor_position=initial_dynamics.attractor_position,
    )

    my_plotter.obstacle_alpha = 1

    my_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list,
    )

    my_plotter.plot(
        my_avoider.evaluate,
        obstacle_list=obstacle_list,
        check_functor=obstacle_list.is_collision_free,
        n_resolution=n_resolution,
    )

    if save_figure:
        my_plotter.save(figure_name + "_rotated")


if (__name__) == "__main__":
    single_ellipse_nonlinear()
