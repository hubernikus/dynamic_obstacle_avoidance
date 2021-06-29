#!/USSR/bin/python3
'''
Demonstration script on how to use multiple obstacles
'''
# Author: Lukas Huber
# Created: 2021-06-29
# Email: lukas.huber@epfl.ch

from math import pi
import numpy as np
# import matplotlib.pyplot as plt

# Import dynamical systems
from vartools.dynamicalsys import LinearSystem

# Custom libraries
from dynamic_obstacle_avoidance.obstacles import Ellipse, Polygon, Cuboid
from dynamic_obstacle_avoidance.containers import GradientContainer
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import Simulation_vectorFields


def single_obstacle_environment(n_resolution=100, save_figure=False):
    Environment = GradientContainer() # create empty obstacle list
    Environment.append(
        Ellipse(
        center_position=[3.5, 0.4],
        orientation=30./180.*pi,
        axes_length=[2.0, 1.5]
        ))

    InitialSytem = LinearSystem(attractor_position=np.array([0, 0 ]))

    x_lim = [-10, 10]
    y_lim = [-10, 10]

    Simulation_vectorFields(
        x_lim, y_lim, n_resolution, Environment,
        saveFigure=False, 
        noTicks=True, showLabel=False,
        draw_vectorField=True,
        dynamical_system=InitialSytem.evaluate,
        obs_avoidance_func=obs_avoidance_interpolation_moving,
        automatic_reference_point=False,
        pos_attractor=InitialSytem.attractor_position,
        show_streamplot=True,
        )


def multiple_obstacles(n_resolution=100, save_figure=False):
    Environment = GradientContainer() # create empty obstacle list
    Environment.append(
        Ellipse(
        center_position=[3.5, -4.4],
        orientation=40./180.*pi,
        axes_length=[2.0, 3.0]
        ))

    Environment.append(
        Cuboid(
        center_position=[-4.0, -1.4],
        orientation=40./180.*pi,
        axes_length=[3.0, 4.0]
        ))

    Environment.append(
        Ellipse(
        center_position=[4.0, 4.4],
        orientation=40./180.*pi,
        axes_length=[3.0, 1.0]
        ))

    Environment.append(
        Polygon(
        center_position=[0, 0],
        orientation=40./180.*pi,
        edge_points = np.array([[-2, 2, 0],
                                [-1,-1, 2]]),
        ))

    InitialSytem = LinearSystem(attractor_position=np.array([8, 0 ]))

    x_lim = [-10, 10]
    y_lim = [-10, 10]

    Simulation_vectorFields(
        x_lim, y_lim, n_resolution, Environment,
        saveFigure=False, 
        noTicks=False, showLabel=False,
        draw_vectorField=True,
        dynamical_system=InitialSytem.evaluate,
        obs_avoidance_func=obs_avoidance_interpolation_moving,
        automatic_reference_point=False,
        pos_attractor=InitialSytem.attractor_position,
        show_streamplot=True,
        )


def obstacle_and_hull_environment(n_resolution=100, save_figure=False):
    Environment = GradientContainer() # create empty obstacle list
    Environment.append(
        Ellipse(
        center_position=[3.5, 0.4],
        orientation=30./180.*pi,
        axes_length=[2.0, 1.5],
        ))

    Environment.append(
        Cuboid(
        center_position=[0, 0.0],
        orientation=0./180.*pi,
        axes_length=[19.0, 19.0],
        is_boundary=True,
        ))

    InitialSytem = LinearSystem(attractor_position=np.array([-8, -4.0 ]))

    x_lim = [-10, 10]
    y_lim = [-10, 10]

    Simulation_vectorFields(
        x_lim, y_lim, n_resolution, Environment,
        saveFigure=False, 
        noTicks=False, showLabel=False,
        draw_vectorField=True,
        dynamical_system=InitialSytem.evaluate,
        obs_avoidance_func=obs_avoidance_interpolation_moving,
        automatic_reference_point=False,
        pos_attractor=InitialSytem.attractor_position,
        show_streamplot=True,
        )

    
if "__main__"==__name__:
    # single_obstacle_environment(n_resolution=100)
    # multiple_obstacles(n_resolution=100)
    obstacle_and_hull_environment(n_resolution=100)
