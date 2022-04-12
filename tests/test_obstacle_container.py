#!/USSR/bin/python3
"""
Test script for obstacle avoidance algorithm
Test normal formation
"""
import numpy as np
from math import pi

from dynamic_obstacle_avoidance.obstacles import Ellipse, Cuboid
from dynamic_obstacle_avoidance.containers import GradientContainer

import pytest


def test_obstacle_container_appending():
    """Appending one obstacle."""
    obs = GradientContainer()  # create empty obstacle list
    obs.append(
        Ellipse(
            axes_length=[2, 1.2],
            center_position=[0.0, 0.0],
            orientation=0.0 / 180 * pi,
        )
    )


def test_obstacle_container_deleting():
    """Appending & deleting obstacles"""
    obs = GradientContainer()  # create empty obstacle list

    # Static obstacles at center
    obs.append(
        Cuboid(
            axes_length=np.array([2, 2.0]),
            center_position=np.array([1, 1.4]),
        )
    )

    # Wall
    obs.append(
        Cuboid(
            axes_length=np.array([10.8, 10.2]),
            center_position=np.array([0, 0]),
            is_boundary=True,
        )
    )

    x_range = [-5, 5]
    y_range = [-5, 5]
    axes_min, axes_max = 0.1, 1

    # Iterate five times
    for ii in range(5):
        # print("Getting sensory input number={}".format(ii))
        # Remove all dynamic obstacles
        it_cont = 0
        while it_cont < len(obs):
            if obs[it_cont].is_dynamic:
                del obs[it_cont]
            else:
                it_cont += 1

        # Add a couple of new ellipses
        num_new_obs = np.random.randint(low=1, high=6)
        for jj in range(num_new_obs):
            # print('Add obstacle number{}'.format(jj))
            axes_length = np.random.rand(2) * (axes_max - axes_min) + axes_min

            x_val = np.random.rand() * (x_range[1] - x_range[0]) + x_range[0]
            y_val = np.random.rand() * (y_range[1] - y_range[0]) + y_range[0]

            orient = np.random.rand() * 2 * pi

            obs.append(
                Ellipse(
                    axes_length=axes_length,
                    center_position=np.array([x_val, y_val]),
                    orientation=orient,
                    is_dynamic=True,
                )
            )


if (__name__) == "__main__":
    test_obstacle_container_appending()
    test_obstacle_container_deleting()

    print("Done all.")
