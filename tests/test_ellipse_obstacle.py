'''
UNIT TESTING

Test script for obstacle avoidance algorithm
Test normal formation
'''

# TODO: TEST on: moving general creation, moving, gamma values, obstacle container

import numpy as np
from math import pi

from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import GradientContainer
from dynamic_obstacle_avoidance.obstacles import Ellipse, CircularObstacle
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import Simulation_vectorFields

def visualize_ellipse_with_ref_point_inside():
    ''' Visualize ellipse with reference point far away '''

    # Create Ellipse
    obs = Ellipse(
        axes_length=[1.2, 2.0],
        center_position=[0.0, 0.0],
        orientation=30./180*pi,
    )

    # Reset reference point
    obs.set_reference_point(np.array([1, 0.3]), in_global_frame=True)

    obs_list = GradientContainer()
    obs_list.append(obs)
    
    Simulation_vectorFields(
        x_range=[-3, 3], y_range=[-3, 3],
        point_grid=100,
        obs=obs_list,
        draw_vectorField=False,
        automatic_reference_point=False,
    )

        
def visualization_ellipse_with_ref_point_outside():
    ''' Visualize ellipse with reference point far away '''
    # Create Ellipse
    obs = Ellipse(
        axes_length=[1.2, 2.0],
        center_position=[0.0, 0.0],
        orientation=30./180*pi,
    )


    # Set reference point outside
    obs.set_reference_point(np.array([2, 1]), in_global_frame=True)
    
    obs_list = GradientContainer()
    obs_list.append(obs)

    Simulation_vectorFields(
        x_range=[-3, 3], y_range=[-3, 3],
        point_grid=0,
        obs=obs_list,
        draw_vectorField=False,
        automatic_reference_point=False,
    )

    
def visualization_circular_reference_point_outside():
    ''' Visualize circular-obstacle with reference point far away '''

    obs = CircularObstacle(
        radius=1.5,
        center_position=[0.0, 0.0],
        orientation=0./180*pi,
    )

    obs.set_reference_point(np.array([1.2, 1.9]), in_global_frame=True)

    obs_list = GradientContainer()
    obs_list.append(obs)
    
    Simulation_vectorFields(
        x_range=[-3, 3], y_range=[-3, 3],
        point_grid=0,
        obs=obs_list,
        draw_vectorField=False,
        automatic_reference_point=False,        
    )

    
if (__name__) == "__main__":
    # test_normal_ellipse()
    # visualize_ellipse_with_ref_point_inside()
    # visualization_ellipse_with_ref_point_outside()
    visualization_circular_reference_point_outside()
    
    print("Selected tests complete.")
