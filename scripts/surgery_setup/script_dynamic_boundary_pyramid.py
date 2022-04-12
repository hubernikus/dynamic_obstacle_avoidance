"""
UNIT TESTING

Test script for obstacle avoidance algorithm
Test normal formation
"""

# TODO: TEST on: moving general creation, moving, gamma values, obstacle container

import numpy as np

from dynamic_obstacle_avoidance.obstacle_avoidance.dynamic_boundaries_polygon import (
    DynamicBoundariesPolygon,
)
from dynamic_obstacle_avoidance.container import GradientContainer


def test_obstacle_list_creation():
    # obs = GradientContainer()  # create empty obstacle list
    pass


def test_gamma_proportional():
    """Ensure that with increasing distance gamma increases (or equal)."""
    obs = DynamicBoundariesPolygon(is_surgery_setup=True)

    # Check 10 random points
    x_range = [obs.x_min, obs.x_max]
    y_range = [obs.y_min, obs.y_max]
    z_range = [obs.z_min, obs.z_max]

    ii = 0
    while ii < 100:
        pos = np.random.rand(3)
        pos[0] = pos[0] * (x_range[1] - x_range[0]) + x_range[0]
        pos[1] = pos[1] * (y_range[1] - y_range[0]) + y_range[0]
        pos[2] = pos[2] * (z_range[1] - z_range[0]) + z_range[0]

        # Factor in [0, 1]
        fac = np.random.rand()
        if not fac:  # zero value
            continue

        # Check two gammas
        pos1 = pos
        pos2 = np.hstack((fac * pos[:2], pos[2]))

        gamma1 = obs.get_gamma(pos1)
        gamma2 = obs.get_gamma(pos2)

        if obs.is_boundary:  # fac is nonzero
            if gamma2 < gamma1:
                print("fac={} // pos1={} --- pos2{}".format(fac, pos1, pos2))
            assert (
                gamma2 >= gamma1
            ), "Gamma not increasing with decreasing distance from origin."
        else:
            assert (
                gamma2 <= gamma1
            ), "Gamma not decreasing with increasing distance from origin."

            # Only obstacle (not for wall)
            assert not (
                gamma1 ^ np.linalg.norm(pos)
            ), "Zero gamma value outside origin."

        assert gamma1 >= 0, "Negative Gamma value."
        ii += 1


def test_normal_direction():
    """Normal has to point alongside reference"""
    obs = DynamicBoundariesPolygon(is_surgery_setup=True)

    # Check 10 random points
    x_range = [obs.x_min, obs.x_max]
    y_range = [obs.y_min, obs.y_max]
    z_range = [obs.z_min, obs.z_max]

    ii = 0
    while ii < 100:
        pos = np.random.rand(3)
        pos[0] = pos[0] * (x_range[1] - x_range[0]) + x_range[0]
        pos[1] = pos[1] * (y_range[1] - y_range[0]) + y_range[0]
        pos[2] = pos[2] * (z_range[1] - z_range[0]) + z_range[0]

        # Only defined outside the obstacle
        if obs.get_gamma(pos) <= 1:
            continue

        vector_normal = obs.get_normal_direction(pos, in_global_frame=True)
        vector_reference = obs.get_reference_direction(pos, in_global_frame=True)

        # print('position', pos)
        # print('vector normal', vector_normal)
        # print('vector reference', vector_reference)

        assert (
            vector_normal.dot(vector_reference) >= 0
        ), "Tangent and Normal and reference for cuboid not in same direction."

        ii += 1


if (__name__) == "__main__":
    test_gamma_proportional()
    # test_normal_direction()

    print("Selected tests complete.")
