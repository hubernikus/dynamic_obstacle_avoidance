'''
UNIT TESTING

Test script for obstacle avoidance algorithm
Test normal formation
'''

# TODO: TEST on: moving general creation, moving, gamma values, obstacle container

import numpy as np
from math import pi

from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import Ellipse, CircularObstacle
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import Cuboid
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import GradientContainer

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import Simulation_vectorFields


def test_two_intersecting_circles():
    ''' Appending one obstacle. '''
    
    obs_list = GradientContainer() # create empty obstacle list
    obs_list.append(CircularObstacle(
            radius=1.5,
            center_position=[-1.0, 0.0],
            orientation=0./180*pi,
    ))                          #

    obs_list.append(CircularObstacle(
            radius=1.5,
            center_position=[1.0, 0.0],
            orientation=0./180*pi,
    ))                          #

    obs_list.update_reference_points()

    Simulation_vectorFields(
        x_range=[-4, 4], y_range=[-4, 4],
        point_grid=0,
        obs=obs_list,
        automatic_reference_point=False,        
    )



def test_three_intersecting_circles():
    ''' Appending one obstacle. '''
    
    obs_list = GradientContainer() # create empty obstacle list
    obs_list.append(CircularObstacle(
            radius=1.0,
            center_position=[-1.2, 0.5],
            orientation=0./180*pi,
    ))                          #

    obs_list.append(CircularObstacle(
            radius=1.0,
            center_position=[0.0, 0.0],
            orientation=0./180*pi,
    ))                          #

    obs_list.append(CircularObstacle(
            radius=1.0,
            center_position=[1.2, 1.5],
            orientation=0./180*pi,
    ))                          #

    obs_list.update_reference_points()
    
    Simulation_vectorFields(
        x_range=[-4, 4], y_range=[-4, 4],
        point_grid=0,
        obs=obs_list,
        automatic_reference_point=False,        
    )
    

if (__name__)=="__main__":
    test_two_intersecting_circles()

    test_three_intersecting_circles()

    print("Selected tests complete.")
