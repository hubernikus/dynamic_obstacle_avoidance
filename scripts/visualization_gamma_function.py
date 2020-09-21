#!/USSR/bin/python3

'''
Script which creates a variety of examples of local modulation of a vector field with obstacle avoidance. 

'''

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import *

__author__ =  "LukasHuber"
__email__ = "lukas.huber@epfl.ch"
__date__ =  "2018-02-15"


# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt


########################################################################
# Chose the option you want to run as a number in the option list (integer from -2 to 10)
# options = [0, 1, 2]

options = [0]

n_resolution = 20
saveFigures=False

########################################################################


def visualize_simple_ellipse(n_resolution=n_resolution):
    obs = GradientContainer() # create empty obstacle list
    x_lim = [-0.6, 5.1]
    y_lim = [-2.1, 2.1]

    xAttractor=[0, 0]

    obs.append(Ellipse(
        axes_length=[0.4, 0.4],
        center_position=[2.0, 0.0],
        p=[1,1],
        orientation=0./180*pi,
        margin_absolut=0.5,
        is_boundary=False
    ))

    fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_boundaryCuboid', noTicks=False, draw_vectorField=True,  automatic_reference_point=True, point_grid=n_resolution, show_streamplot=False, normalize_vectors=False, dynamicalSystem=linearAttractor_const)


if (__name__)=="__main__":
    
    visualize_simple_ellipse()


# Run function

