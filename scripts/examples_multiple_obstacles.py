#!/USSR/bin/python3

'''
Script which creates a variety of examples of local modulation of a vector field with obstacle avoidance. 
'''

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt
import sys
from math import pi

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import Simulation_vectorFields  #
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import Polygon, Cuboid
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import Ellipse
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import ObstacleContainer

__author__ = "LukasHuber"
__date__ = "2018-02-15"
__email__ = "lukas.huber@epfl.ch"


########################################################################
# Chose the option you want to run as a number in the option list (integer from -2 to 10)

options = [0]

N_resol = 80

saveFigures=False
########################################################################

def main(options=[0], N_resol=100, saveFigures=False):
    for option in options:
        obs = ObstacleContainer() # create empty obstacle list
        if option==0:
            xAttractor = [-2., 0]
            
            x_lim, y_lim = [-3, 3], [-3.1, 3.1]
            
            # obs.append(Ellipse(center_position=[0, 0.9],
                               # orientation=60/180.*pi,
                               # axes_length=[1.2, 0.6]))

            obs.append(Ellipse(center_position=[0, -0.5],
                               orientation=-60/180.*pi,
                               axes_length=[1.3, 0.5]))
            
            Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='twoEllispoidsIntersecting', noTicks=False, draw_vectorField=True,  automatic_reference_point=True, point_grid=N_resol)

        if option==1:
            xAttractor = [1., 0]
            
            x_lim, y_lim = [-1.1, 1.1], [-1.1, 1.1]

            edge_points = np.array([[1, 0.5],
                                    [1, 1],
                                    [-1, 1],
                                    [-1, -1],
                                    [1, -1],
                                    [1, -0.5]]).T

            ind_open=5
            obs.append(Polygon(edge_points, is_boundary=True, ind_open=5, center_position=[0, 0]))
            
            Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='twoEllispoidsIntersecting', noTicks=False, draw_vectorField=True,  automatic_reference_point=True, point_grid=N_resol)
            

if (("__main__")==str(__name__)):
    if len(sys.argv) > 1 and not sys.argv[1]=='-i':
        options = [int(sys.argv[1])]
        if len(sys.argv) > 2:
            N_resol = int(sys.argv[2])
            if len(sys.argv) > 3:
                saveFigures = bool(sys.argv[3])

    main(options=options, N_resol=N_resol, saveFigures=saveFigures)

