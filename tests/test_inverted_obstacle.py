"""
Testing script to ensure that 
"""
import pytest

import numpy as np

from dynamic_obstacle_avoidance.obstacles import Cuboid, Ellipse


def test_inverted_obstacle_ellipse():
    my_obstacle = Ellipse(
        center_position=np.array([0, 0]),
        axes_length=np.array([2, 2]),
        is_boundary=True,
        )

    point = np.array([0, 1])
    gamma = my_obstacle.get_gamma(point, in_global_frame=True)
    assert gamma > 1, "Gamma is not greater than 1 for ellipse."


def test_inverted_obstacle_cuboid():
    my_obstacle = Cuboid(
        center_position=np.array([0, 0]),
        axes_length=np.array([2, 2]),
        is_boundary=True,
        )
    point = np.array([1, 10])
    # Local radius is needed for this evalaution
    print(f"rad = {my_obstacle.get_local_radius(point)} at pos {point}")
    
    # Once this is working, one can move on
    gamma = my_obstacle.get_gamma(point, in_global_frame=True)
    assert gamma > 1, "Gamma is not greater than 1 for cuboid inside the wall."
    

if (__name__) == "__main__":
    test_inverted_obstacle_ellipse()
    test_inverted_obstacle_cuboid()
    
    print("Done testing")
