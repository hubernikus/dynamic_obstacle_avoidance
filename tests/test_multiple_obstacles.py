"""
UNIT TESTING

Test script for obstacle avoidance algorithm
Test normal formation
"""
# TODO: TEST on: moving general creation, moving, gamma values, obstacle container
import pytest

import numpy as np
from math import pi

from dynamic_obstacle_avoidance.obstacles import Ellipse, CircularObstacle
from dynamic_obstacle_avoidance.obstacles import Cuboid
from dynamic_obstacle_avoidance.containers import GradientContainer

from dynamic_obstacle_avoidance.visualization import Simulation_vectorFields


def test_two_intersecting_circles(visualize=False):
    """Appending one obstacle."""
    obs_list = GradientContainer()  # create empty obstacle list
    obs_list.append(
        CircularObstacle(
            radius=1.5,
            center_position=[-1.0, 0.0],
            orientation=0.0 / 180 * pi,
        )
    )  #

    obs_list.append(
        CircularObstacle(
            radius=1.5,
            center_position=[1.0, 0.0],
            orientation=0.0 / 180 * pi,
        )
    )  #

    obs_list.update_reference_points()

    if visualize:
        Simulation_vectorFields(
            x_range=[-4, 4],
            y_range=[-4, 4],
            point_grid=0,
            obs=obs_list,
            automatic_reference_point=False,
        )

    for obs in obs_list:
        assert (
            obs.get_gamma(obs.reference_point) < 1
        ), "Warning reference point outside obstacle"

        assert (
            obs.get_gamma(obs.global_reference_point, in_global_frame=True) < 1
        ), "Warning reference point outside obstacle"


def test_three_intersecting_circles(visualize=False):
    """Appending one obstacle."""
    obs_list = GradientContainer()  # create empty obstacle list
    obs_list.append(
        CircularObstacle(
            radius=1.0,
            center_position=[-1.2, 0.5],
            orientation=0.0 / 180 * pi,
        )
    )  #

    obs_list.append(
        CircularObstacle(
            radius=1.0,
            center_position=[0.0, 0.0],
            orientation=0.0 / 180 * pi,
        )
    )  #

    obs_list.append(
        CircularObstacle(
            radius=1.0,
            center_position=[1.2, 1.5],
            orientation=0.0 / 180 * pi,
        )
    )  #

    obs_list.update_reference_points()

    if visualize:
        Simulation_vectorFields(
            x_range=[-4, 4],
            y_range=[-4, 4],
            point_grid=0,
            obs=obs_list,
            automatic_reference_point=False,
        )

    for obs in obs_list:
        # assert obs.get_gamma(obs.reference_point)<1
        # assert obs.get_gamma(obs.global_reference_point, in_global_frame=True)<1
        pass


def test_two_intersecting_ellipses(visualize=False):
    """Appending one obstacle."""
    obs_list = GradientContainer()  # create empty obstacle list
    obs_list.append(
        Ellipse(
            axes_length=np.array([1, 2]),
            center_position=[-1.2, 0.0],
            orientation=-40.0 / 180 * pi,
        )
    )  #

    obs_list.append(
        Ellipse(
            axes_length=np.array([1, 2]),
            center_position=[1.5, 0.0],
            orientation=40.0 / 180 * pi,
        )
    )  #
    # obs_list.append(CircularObstacle(
    # radius=1.0,
    # center_position=[1.2, 1.5],
    # orientation=0./180*pi,
    # ))                          #

    obs_list.update_reference_points()

    if visualize:
        Simulation_vectorFields(
            x_range=[-4, 4],
            y_range=[-4, 4],
            point_grid=0,
            obs=obs_list,
            automatic_reference_point=False,
        )
    for obs in obs_list:
        assert (
            obs.get_gamma(obs.reference_point) < 1
        ), "Warning reference point outside obstacle"

        assert (
            obs.get_gamma(obs.global_reference_point, in_global_frame=True) < 1
        ), "Warning reference point outside obstacle"


if (__name__) == "__main__":
    plot_results = False

    test_two_intersecting_circles(plot_results)
    test_three_intersecting_circles(plot_results)
    test_two_intersecting_ellipses(plot_results)

    print("Selected tests complete.")
