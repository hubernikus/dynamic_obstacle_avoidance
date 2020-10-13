# !/USSR/bin/python3

'''
Script which creates a variety of examples of local modulation of a vector field with obstacle avoidance. 
'''

import sys
import os

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt

# Add obstacle avoidance without 'setting' up
# directory_path = rospack.get_path('qolo_modulation')
# directory_path = "/home/lukas/Code/ObstacleAvoidance/dynamic_obstacle_avoidance/"
# path_avoidance = os.path.join(directory_path, "src")
# if not path_avoidance in sys.path:
    # sys.path.append(path_avoidance)


# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import *

__author__ =  "LukasHuber"
__email__ = "lukas.huber@epfl.ch"
__date__ =  "2018-02-15"


########################################################################
# Chose the option you want to run as a number in the option list (integer from -2 to 10)
# options = [0, 1, 2]

options = [0]

n_resolution = 30
saveFigures=False

########################################################################


def visualize_simple_ellipse(n_resolution=n_resolution):
    n_resolution = 10
    
    obs = GradientContainer() # create empty obstacle list
    x_lim = [-0.6, 5.1]
    y_lim = [-2.1, 2.1]

    xAttractor=[0, 0]
    save_figure = True
    figsize = (6,5)

    obs.append(Ellipse(
        axes_length=[1.0, 1.0],
        center_position=[2.5, 0.0],
        p=[1,1],
        orientation=0./180*pi,
        margin_absolut=0.0,
        is_boundary=False,
        has_sticky_surface=False,
        reactivity=1,
        tail_effect=True,
    ))
    
    
    fig_mod, ax_mod = Simulation_vectorFields(
        x_lim, y_lim,  obs=obs, xAttractor=xAttractor,
        saveFigure=save_figure, figName='circular_sticky_surface',
        noTicks=False, draw_vectorField=True,
        automatic_reference_point=True, point_grid=n_resolution, show_streamplot=False,
        figureSize=figsize,
        reference_point_number=False, showLabel=False,
        normalize_vectors=False, dynamicalSystem=linearAttractor_const)


    obs.has_sticky_surface=False
    
    fig_mod, ax_mod = Simulation_vectorFields(
        x_lim, y_lim,  obs=obs, xAttractor=xAttractor,
        saveFigure=save_figure, figName='circular_nonsticky_surface',
        noTicks=False, draw_vectorField=True,
        automatic_reference_point=True, point_grid=n_resolution, show_streamplot=False,
        figureSize=figsize,
        reference_point_number=False, showLabel=False,
        normalize_vectors=False, dynamicalSystem=linearAttractor_const)
    
    MAX_SPEED = 3.0
    
    # pos  = np.array([1, 1])
    # xd_init = linear_ds_max_vel(pos, attractor=xAttractor, vel_max=MAX_SPEED)
    # xd_mod = obs_avoidance_interpolation_moving(pos, xd_init, obs)
    # print('pos', pos)
    # print('xd init', xd_init)
    # print('xd init mag', np.linalg.norm(xd_init))
    # print('xd mod', xd_mod)
    # print('xd mod mag', np.linalg.norm(xd_mod))

    # pos  = np.array([2.5-1+2.5, 1])
    # xd_init = linear_ds_max_vel(pos, attractor=xAttractor, vel_max=MAX_SPEED)
    # xd_mod = obs_avoidance_interpolation_moving(pos, xd_init, obs)
    # print('pos', pos)
    # print('xd init', xd_init)
    # print('xd init mag', np.linalg.norm(xd_init))
    # print('xd mod', xd_mod)
    # print('xd mod mag', np.linalg.norm(xd_mod))

    # import pdb; pdb.set_trace()



def visualize_intersecting_ellipse(n_resolution=n_resolution):
    obs = GradientContainer() # create empty obstacle list
    x_lim = [-0.6, 5.1]
    y_lim = [-2.1, 2.1]

    xAttractor=[0, 0]
    figure_size = (5, 3.0)
    n_resolution = 15
    saveFigures=True
    

    obs.append(Ellipse(
        axes_length=[0.8, 0.4],
        center_position=[2.0, 0.5],
        p=[1,1],
        orientation=20./180*pi,
        margin_absolut=0.5,
        is_boundary=False
    ))

    obs.append(Ellipse(
        axes_length=[0.8, 0.4],
        center_position=[2.0, -0.5],
        p=[1,1],
        orientation=-40./180*pi,
        margin_absolut=0.5,
        is_boundary=False,
        reactivity=1,
    ))

    fig_mod, ax_mod = Simulation_vectorFields(
        x_lim, y_lim,  obs=obs, xAttractor=xAttractor,
        saveFigure=saveFigures, figName='intersecting_ellipses_sticky_surfaces',
        noTicks=True,
        draw_vectorField=True,  automatic_reference_point=True,
        point_grid=n_resolution, show_streamplot=False,
        normalize_vectors=False, dynamicalSystem=linearAttractor_const,
        figureSize=figure_size,
        reference_point_number=False, showLabel=False,
    )

    for oo in range(len(obs)):
        obs[oo].has_sticky_surface=False

    
    # Why does 'automatic reference point' create a mess?
    fig_mod, ax_mod = Simulation_vectorFields(
        x_lim, y_lim,  obs=obs, xAttractor=xAttractor,
        saveFigure=saveFigures, figName='intersecting_ellipses_nonsticky_surfaces',
        noTicks=True, draw_vectorField=True,  automatic_reference_point=False, point_grid=n_resolution, show_streamplot=False,
        normalize_vectors=False, dynamicalSystem=linearAttractor_const,
        figureSize=figure_size,
        reference_point_number=False, showLabel=False,
    )

    
def visualize_repulsive_cube(
        robot_margin=0.35,
        n_resolution=40,
        save_figure=False,
):
    x_lim = [-2, 6]
    y_lim = [-2, 2]

    figure_size = (15, 6.0)
    pos_attractor = [-1, 0]
    
    
    obs = GradientContainer() # create empty obstacle list

    obs.append(Polygon(
          edge_points=[[-1.8, 5.4, 5.4, -1.8],
                       [-1.8,-1.8, 1.8, 1.8]],
        center_position=[3.0, -1],
        orientation=0./180*pi,
        margin_absolut=robot_margin,
        is_boundary=True,
    ))

    obs.append(Cuboid(
        axes_length=[0.4, 0.4],
        center_position=[2.0, -0.0],
        orientation=90./180*pi,
        margin_absolut=robot_margin,
        is_boundary=False,
        repulsion_coeff=2.0
    ))


    obs.append(Cuboid(
        axes_length=[0.4, 0.4],
        center_position=[2.0, -0.0],
        orientation=90./180*pi,
        margin_absolut=robot_margin,
        is_boundary=False,
        repulsion_coeff=5.0
    ))


    # obs.append(Polygon(
    #     edge_points=[[-1.8, 5.4, 5.4, -1.8],
    #                  [-1.8,-1.8, 1.8, 1.8]],
    #     # center_position=[3.0, -1],
    #     orientation=0./180*pi,
    #     margin_absolut=robot_margin,
    #     is_boundary=True,
    #     tail_effect=False,
    # ))

    # obs.append(Cuboid(
    #     axes_length=[0.4, 0.4],
    #     center_position=[2.0, 0.0],
    #     orientation=0./180*pi,
    #     margin_absolut=robot_margin,
    #     is_boundary=False,
    #     repulsion_coeff=2.0,
    #     tail_effect=False,
    # ))

    fig_mod, ax_mod = Simulation_vectorFields(
        x_lim, y_lim,  obs=obs, xAttractor=pos_attractor,
        saveFigure=False, figName='repulsive_cube_in_hallway',
        noTicks=True, draw_vectorField=True,  automatic_reference_point=False, point_grid=n_resolution, show_streamplot=True,
        normalize_vectors=False, dynamicalSystem=linearAttractor_const,
        figureSize=figure_size,
        reference_point_number=False, showLabel=False,
    )


def visualize_nonrepulsive_cube(
        robot_margin=0.35,
        n_resolution=20,
        save_figure=False,
):
    x_lim = [-2, 6]
    y_lim = [-2, 2]

    figure_size = (15, 6.0)
    pos_attractor = [-1, 0]
    
    
    obs = GradientContainer() # create empty obstacle list
    obs.append(Polygon(
        edge_points=[[-1.8, 5.4, 5.4, -1.8],
                     [-1.8,-1.8, 1.8, 1.8]],
        # center_position=[3.0, -1],
        orientation=0./180*pi,
        margin_absolut=robot_margin,
        is_boundary=True,
        tail_effect=False,
    ))

    obs.append(Cuboid(
        axes_length=[0.4, 0.4],
        center_position=[2.0, 0.0],
        orientation=0./180*pi,
        margin_absolut=robot_margin,
        is_boundary=False,
        repulsion_coeff=1.0,
        tail_effect=False,
    ))

    fig_mod, ax_mod = Simulation_vectorFields(
        x_lim, y_lim,  obs=obs, xAttractor=pos_attractor,
        saveFigure=False, figName='nonrepulsive_cube_in_hallway',
        noTicks=True, draw_vectorField=True,  automatic_reference_point=False, point_grid=n_resolution, show_streamplot=True,
        normalize_vectors=False, dynamicalSystem=linearAttractor_const,
        figureSize=figure_size,
        reference_point_number=False, showLabel=False,
    )
    # import pdb; pdb.set_trace()



if (__name__)=="__main__":
    # visualize_simple_ellipse(n_resolution=20)
    # visualize_intersecting_ellipse()

    visualize_repulsive_cube()
    visualize_nonrepulsive_cube()
# Run function

