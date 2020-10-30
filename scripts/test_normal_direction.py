'''
UNIT TESTING

Test script for obstacle avoidance algorithm
Test normal formation
'''

# TODO: TEST on: moving general creation, moving, gamma values, obstacle container

import numpy as np
from math import pi

from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import Ellipse
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import Cuboid



def test_obstacle_list_creation():
    obs = GradientContainer() # create empty obstacle list

    pass


def test_normal_ellipse():
    ''' Normal has to point alongside reference'''
    obs = Ellipse(
            axes_length=[2, 1.2],
            center_position=[0.0, 0.0],
            orientation=0./180*pi,
    )

    # Check 10 random points
    x_range = [-10, 10]
    y_range = [-10, 10]

    ii = 0
    while(ii < 10):
        pos = np.random.rand(2)
        pos[0] = pos[0]*(x_range[1] - x_range[0]) + x_range[0]
        pos[1] = pos[1]*(y_range[1] - y_range[0]) + y_range[0]

        # Only defined outside the obstacle
        if obs.get_gamma(pos) <= 1:
            continue

        vector_normal = obs.get_normal_direction(pos, in_global_frame=True)
        vector_reference = obs.get_reference_direction(pos, in_global_frame=True)

        assert vector_normal.dot(vector_reference)>=0, "Tangent and Normal and reference for cuboid not in same direction."
                
        ii += 1


def test_normal_cuboid():
    ''' Normal has to point alongside reference'''
    obs = Cuboid(
            axes_length=[2, 1.2],
            center_position=[0.0, 0.0],
            orientation=0./180*pi,
    )

    # Check 10 random points
    x_range = [-10, 10]
    y_range = [-10, 10]

    ii = 0
    while(ii < 10):
        pos = np.random.rand(2)
        pos[0] = pos[0]*(x_range[1] - x_range[0]) + x_range[0]
        pos[1] = pos[1]*(y_range[1] - y_range[0]) + y_range[0]
        

        # Only defined outside the obstacle
        if obs.get_gamma(pos) <= 1:
            continue

        vector_normal = obs.get_normal_direction(pos, in_global_frame=True)
        vector_reference = obs.get_reference_direction(pos, in_global_frame=True)

        # print('pos', pos)
        
        assert vector_normal.dot(vector_reference)>=0, "Reference and Normal for Cuboid not in same direction."
        ii += 1


def test_normal_cuboid_with_margin():
    ''' Normal has to point alongside reference'''
    obs = Cuboid(
        axes_length=[2, 1.2],
        center_position=[0.0, 0.0],
        orientation=0./180*pi,
        margin_absolut=1.0
    )

    # Check 10 random points
    x_range = [-10, 10]
    y_range = [-10, 10]

    ii = 0
    while(ii < 100):
        pos = np.random.rand(2)
        pos[0] = pos[0]*(x_range[1] - x_range[0]) + x_range[0]
        pos[1] = pos[1]*(y_range[1] - y_range[0]) + y_range[0]
        

        # Only defined outside the obstacle
        if obs.get_gamma(pos) <= 1:
            continue

        vector_normal = obs.get_normal_direction(pos, in_global_frame=True)
        vector_reference = obs.get_reference_direction(pos, in_global_frame=True)

        # print('pos', pos)
        
        assert vector_normal.dot(vector_reference)>=0, "Reference and Normal for Cuboid not in same direction."
        ii += 1

    # print('All normals are pointing away from cuboids with_margins')

if (__name__)=="__main__":
    # test_obstacle_list_creation
    # test_normal_ellipse()
    # test_normal_cuboid()
    test_normal_cuboid_with_margin()

    print("Selected tests complete.")
