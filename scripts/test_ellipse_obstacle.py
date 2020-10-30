'''
UNIT TESTING

Test script for obstacle avoidance algorithm
Test normal formation
'''

# TODO: TEST on: moving general creation, moving, gamma values, obstacle container

import numpy as np
from math import pi

from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import GradientContainer
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import Ellipse, CircularObstacle
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import Simulation_vectorFields

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
        
    

    
if (__name__)=="__main__":
    # test_normal_ellipse()

    # visualize_ellipse_with_ref_point_inside()

    # visualization_ellipse_with_ref_point_outside()
    
    visualization_circular_reference_point_outside()
    
    print("Selected tests complete.")
