#!/USSR/bin/python3
'''
Script which creates a variety of examples of local modulation of a vector field with obstacle avoidance. 
'''
__author__ =  "LukasHuber"
__email__ = "lukas.huber@epfl.ch"
__date__ =  "2018-02-15"

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt

# from dynamic_obstacle_avoidance.dynamical_system import linearAttractor
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import obs_avoidance_interpolation_moving
                
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import *


def get_outside_enviornment_simplified(robot_margin = 0.6):
    obs = GradientContainer() # create empty obstacle list
    obs.append(Polygon(
        edge_points=[[-1.8, 5.4, 5.4, -1.8],
                     [-1.8,-1.8, 0.9, 0.9]],
        center_position=[3.0, -1],
        orientation=0./180*pi,
        margin_absolut=robot_margin,
        is_boundary=True,
    ))

    obs.append(Cuboid(
        axes_length=[0.4, 0.4],
        center_position=[3.0, -0.5],
        orientation=90./180*pi,
        margin_absolut=robot_margin,
        is_boundary=False,
        repulsion_coeff=2.0
    ))

    return obs

    
def get_outside_enviornment(robot_margin = 0.6):
    obs = GradientContainer() # create empty obstacle list
    obs.append(Polygon(
        edge_points=[[-1.8, 4.5, 4.5, 5.4, 5.4,-1.8 ],
                     [-3.6,-3.6, -1.8, -1.8, 0.9, 0.9]],
        center_position=[3.0, -1],
        orientation=0./180*pi,
        margin_absolut=robot_margin,
        is_boundary=True,
    ))

    obs.append(Cuboid(
        axes_length=[0.4, 0.4],
        center_position=[3.0, 0.0],
        orientation=90./180*pi,
        margin_absolut=robot_margin,
        is_boundary=False,
        repulsion_coeff=2.0
    ))

    return obs


def get_outside_enviornment_lshape(robot_margin=0.6):
    
    obs = GradientContainer() # create empty obstacle list
    
    obs.append(Polygon(
        edge_points=[[0.0, 4.0, 4.0, 2.0, 2.0, 0.0],
                     [0.0, 0.0, 2.0, 2.0, 4.0, 4.0]],
        center_position=[0.35, 0.35],
        orientation=0./180*pi,
        margin_absolut=robot_margin,
        is_boundary=True,
    ))


    obs.append(Ellipse(
        axes_length=[0.2, 0.3],
        center_position=[1.0, 1.4],
        orientation=20./180*pi,
        margin_absolut=robot_margin,
    ))
    
    
    return obs


def get_outside_enviornment_lshape_hack(robot_margin=0.6):
    
    obs = GradientContainer() # create empty obstacle list
    
    obs.append(Polygon(
        edge_points=[[0.0, 4.0, 4.0, 0.0],
                     [0.0, 0.0, 4.0, 4.0]],
        orientation=0./180*pi,
        margin_absolut=robot_margin,
        is_boundary=True,
    ))

    obs.append(Cuboid(
        axes_length=np.array([3.0, 3.0]),
        center_position=[3.5, 3.5],
        orientation=0./180*pi,
        margin_absolut=robot_margin,
    ))

    obs.append(Ellipse(
        axes_length=[0.2, 0.3],
        center_position=[1.0, 1.4],
        orientation=20./180*pi,
        margin_absolut=robot_margin,
        repulsion_coeff=2.0
    ))
    
    return obs



if (__name__)=="__main__":
    num_resolution=80
    saveFigures=False

    x_lim = [-0.3, 4.5]
    y_lim = [-2.5, 0.5]

    xAttractor=[3.5, 1.0]

    robot_margin = 0.3

    # obs = get_outside_enviornment(robot_margin)
    obs = get_outside_enviornment_simplified(robot_margin)
    # obs = get_outside_enviornment_lshape(robot_margin)
    # obs = get_outside_enviornment_lshape_hack(robot_margin)

    vectorfield = True
    if vectorfield:
        # fig_mod, ax_mod = Simulation_vectorFields(
        #     x_lim, y_lim, obs=obs, xAttractor=xAttractor, saveFigure=saveFigures,
        #     figName='linearSystem_boundaryCuboid', noTicks=False, draw_vectorField=True,
        #     show_streamplot=True,
        #     automatic_reference_point=True, point_grid=N_resol,
        #     # figureSize=(6., 4.25),
        # )

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim, y_lim, obs=obs, xAttractor=xAttractor, point_grid=num_resolution
        )

    else:
        # Specific value
        position = np.array([1.06, 1.07])

        ds_init = linearAttractor(position, x0=xAttractor)
        ds_mod = obs_avoidance_interpolation_moving(position, ds_init, obs)

        print('ds_mod', ds_mod)
        import pdb; pdb.set_trace()

# Run function
