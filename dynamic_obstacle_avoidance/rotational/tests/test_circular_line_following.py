#!/USSR/bin/python3.10
""" Test / visualization of line following. """
# Author: Lukas Huber
# Created: 2022-11-25
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2025

import warnings
from functools import partial
import unittest
from math import pi
import math

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

# from vartools.linalg import get_orthogonal_basis
# from vartools.dynamical_systems import LinearSystem, ConstantValue
from vartools.dynamical_systems import CircularStable

# from vartools.directional_space import UnitDirection

# DirectionBase
from vartools.dynamical_systems import plot_vectorfield

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.rotational.rotational_avoider import (
    get_intersection_with_circle,
)
from dynamic_obstacle_avoidance.rotational.rotation_container import RotationContainer
from dynamic_obstacle_avoidance.rotational.rotational_avoidance import (
    obstacle_avoidance_rotational,
)
from dynamic_obstacle_avoidance.rotational.rotational_avoider import RotationalAvoider

from dynamic_obstacle_avoidance.visualization import Simulation_vectorFields

from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics


def _test_circle_following_rotational_avoidance(visualize=False):
    global_ds = CircularStable(radius=10, maximum_velocity=1)

    if visualize:
        plt.close("all")
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=global_ds.evaluate,
            x_lim=[-15, 15],
            y_lim=[-15, 15],
            n_grid=30,
            ax=ax,
        )
        ax.scatter(
            0,
            0,
            marker="*",
            s=200,
            color="black",
            zorder=5,
        )

    container = RotationContainer()


if (__name__) == "__main__":
    figtype = ".png"

    _test_circle_following_rotational_avoidance(visualize=True)
    print("Tests done")
