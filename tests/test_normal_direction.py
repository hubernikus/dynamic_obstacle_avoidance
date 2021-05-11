#!/USSR/bin/python3
'''
UNIT TESTING

Test script for obstacle avoidance algorithm
Test normal formation
'''

# TODO: TEST on: moving general creation, moving, gamma values, obstacle container
import sys
import os
from math import pi

import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import CircularObstacle, Ellipse
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import Cuboid

def test_obstacle_list_creation():
    obs = GradientContainer() # create empty obstacle list
    pass


def test_normal_circle():
    ''' Normal has to point alongside reference'''
    obs = CircularObstacle(
        radius=0.5,
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

        assert vector_normal.dot(vector_reference)>=0, "Normal and reference for circle not in same direction."
                
        ii += 1

        
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

        assert vector_normal.dot(vector_reference)>=0, "Normal and reference for ellipse not in same direction."
                
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

    
def test_normal_of_boundary_with_gaps(plot_normals=False):
    ''' Normal has to point alongside reference'''
    # TODO: this will potentially be moved
    # Add specific library (still in prototopye phase)
    rel_path = os.path.join(".", "scripts")
    if rel_path not in sys.path:
        sys.path.append(rel_path)
        
    from animation_inflating_obstacle import BoundaryCuboidWithGaps

    # Dimension of space is 2D
    dim = 2
    
    # Check predefined points
    x_range = [-1, 11]
    y_range = [-6, 6]

    obs = BoundaryCuboidWithGaps(
            name='RoomWithDoor',
            axes_length=[10, 10],
            center_position=[5, 0],
            gap_points_relative=np.array([[-5, -1], [-5, 1]]).T
        )

    attractor_position = obs.get_global_gap_center() 
    
    num_resolution = 30

    x_vals = np.linspace(x_range[0], x_range[1], num_resolution)
    y_vals = np.linspace(y_range[0], y_range[1], num_resolution)

    positions = np.zeros((dim, num_resolution, num_resolution))
    normal_vectors = np.zeros((dim, num_resolution, num_resolution))
    reference_vectors = np.zeros((dim, num_resolution, num_resolution))
    for ix in range(num_resolution):
        for iy in range(num_resolution):
            positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]
            
            if obs.get_gamma(positions[:, ix, iy], in_global_frame=True) <= 1:
                continue

            normal_vectors[:, ix, iy] = obs.get_normal_direction(
                position=positions[:, ix, iy], in_global_frame=True)
            reference_vectors[:, ix, iy] = obs.get_reference_direction(
                position=positions[:, ix, iy], in_global_frame=True)

            # TODO: check edge / boundary case
            assert normal_vectors[:, ix, iy].dot(reference_vectors[:, ix, iy]) >= 0, \
                "Reference and Normal for Cuboid-Wall not in same direction."

    if plot_normals:
        from dynamic_obstacle_avoidance.visualization.vector_field_visualization import Simulation_vectorFields  #
        from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import GradientContainer

        obs_list = GradientContainer()
        obs_list.append(obs)
        
        fig, ax = plt.subplots()
        Simulation_vectorFields(
            x_range, y_range,  obs=obs_list,
            xAttractor=attractor_position,
            saveFigure=False,
            noTicks=False, showLabel=True,
            show_streamplot=False, draw_vectorField=False,
            fig_and_ax_handle=(fig, ax),
            normalize_vectors=False,
        )

        ax.quiver(positions[0, :, :], positions[1, :, :],
                  normal_vectors[0, :, :], normal_vectors[1, :, :], color='green')

        ax.quiver(positions[0, :, :], positions[1, :, :],
                  reference_vectors[0, :, :], reference_vectors[1, :, :], color='blue')
                  

if (__name__)=="__main__":
    # test_normal_circle()
    # test_normal_ellipse()
    # test_normal_cuboid()
    # test_normal_cuboid_with_margin()
    # test_normal_of_boundary_with_gaps()

    test_normal_of_boundary_with_gaps(plot_normals=True)
    
    print("Selected tests complete.")
