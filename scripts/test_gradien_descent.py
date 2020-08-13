#!/USSR/bin/python3
'''
Script which creates a variety of examples of local modulation of a vector field with obstacle avoidance. 
'''

__author__ = "LukasHuber"
__date__ = "2018-02-15"

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt
 
# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import *
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import *

from dynamic_obstacle_avoidance.settings import DEBUG_FLAG
from dynamic_obstacle_avoidance import settings

########################################################################

# Chose the option you want to run as a number in the option list (integer from -2 to 10)
options = [0]
N_resol = 10
saveFigures=False

########################################################################

def main(options=[0], N_resol=100, saveFigures=False):
    if -1 in options:
        obs = GradientContainer() # create empty obstacle list
        x_lim, y_lim = [-3, 3],[-2, 2]

        obs.append(Ellipse(
            axes_length=[0.8, 0.9],
            # p=[1,1],
            x0=[0, 0],
            orientation=30./180*pi
        ))
        
        obs.append(Ellipse(
            axes_length=[0.9, 0.9],
            # p=[1,1],
            x0=[2, 0],
            orientation=-65./180*pi,
        ))

        xAttractor = [-2., 0.2]

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim, y_lim,  obs=obs, xAttractor=xAttractor,
            saveFigure=saveFigures, figName='linearSystem_avoidanceCube',
            noTicks=False, draw_vectorField=True,  automatic_reference_point=True, 
            point_grid=N_resol, show_streamplot=False)

        for ii in range(len(obs)):
            for jj in range(len(obs)):
                if ii==jj:
                    continue
                boundary_ref_point = obs.get_boundary_reference_point(ii, jj)
                plt.plot(boundary_ref_point[0], boundary_ref_point[1],
                         'r+', linewidth=18, markeredgewidth=4, markersize=13)

    if 0 in options:
        obs = GradientContainer() # create empty obstacle list
        x_lim, y_lim = [-3, 3],[-2, 2]

        obs.append(Ellipse(
            axes_length=[0.4, 0.7],
            # p=[1,1],
            x0=[0, 0],
            orientation=30./180*pi
        ))
        obs.append(Ellipse(
            axes_length=[0.3, 0.8],
            # p=[1,1],
            x0=[2, 0],
            orientation=-65./180*pi,
        ))

        xAttractor = [-2., 0.2]

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim, y_lim,  obs=obs, xAttractor=xAttractor,
            saveFigure=saveFigures, figName='linearSystem_avoidanceCube',
            noTicks=False, draw_vectorField=True,  automatic_reference_point=True, 
            point_grid=N_resol, show_streamplot=False)

        for ii in range(len(obs)):
            for jj in range(len(obs)):
                if ii==jj:
                    continue
                boundary_ref_point = obs.get_boundary_reference_point(ii, jj)
                plt.plot(boundary_ref_point[0], boundary_ref_point[1],
                         'r+', linewidth=18, markeredgewidth=4, markersize=13)


    if 1 in options:
        obs = GradientContainer() # create empty obstacle list
        x_lim, y_lim = [-3, 3],[-2, 2]

        obs.append(Ellipse(
            axes_length=[0.4, 0.7],
            # p=[1,1],
            x0=[0, 0],
            orientation=30./180*pi
        ))
        
        obs.append(Cuboid(
            axes_length=[0.8, 1.8],
            # p=[1,1],
            x0=[2, 0],
            orientation=-65./180*pi,
        ))

        xAttractor = [-2., 0.2]

        if 'DEBUG_FLAG' in globals() and DEBUG_FLAG:
            settings.init()
            # import pdb; pdb.set_trace()
        
        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim, y_lim,  obs=obs, xAttractor=xAttractor,
            saveFigure=saveFigures, figName='linearSystem_avoidanceCube',
            noTicks=False, draw_vectorField=True,  automatic_reference_point=True, 
            point_grid=N_resol, show_streamplot=False)

        for ii in range(len(obs)):
            for jj in range(len(obs)):
                if ii==jj:
                    continue
                boundary_ref_point = obs.get_boundary_reference_point(ii, jj)
                plt.plot(boundary_ref_point[0], boundary_ref_point[1],
                         'r+', linewidth=18, markeredgewidth=4, markersize=13)

        
        if 'DEBUG_FLAG' in globals() and DEBUG_FLAG:
            # import pdb; pdb.set_trace()
            # global settings.boundary_ref_point_list

            settings.position0 = np.array(settings.position0)
            plt.plot(settings.position0[:, 0], settings.position0[:, 1], marker='x')
                     
            settings.position1 = np.array(settings.position1)
            plt.plot(settings.position1[:, 0], settings.position1[:, 1], marker='x')

            # import pdb; pdb.set_trace()

            settings.boundary_ref_point_list = np.array(settings.boundary_ref_point_list)
            plt.figure()
            plt.plot(settings.dist_ref_points)

            plt.figure()
            plt.plot(settings.boundary_ref_point_list[:, 0],
                     settings.boundary_ref_point_list[:, 1])

            # import pdb; pdb.set_trace()
            

        # dist_ref_points

    if 2 in options:
        obs = GradientContainer() # create empty obstacle list
        x_lim, y_lim = [-3, 3],[-2, 2]

        obs.append(Cuboid(
            axes_length=[1.3, 1.3],
            # p=[1,1],
            x0=[-1, 0],
            orientation=0./180*pi
        ))
        
        obs.append(Cuboid(
            axes_length=[1.3, 1.3],
            # p=[1,1],
            x0=[1, 0],
            orientation=0./180*pi,
        ))

        xAttractor = [-2., 0.2]

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim, y_lim,  obs=obs, xAttractor=xAttractor,
            saveFigure=saveFigures, figName='linearSystem_avoidanceCube',
            noTicks=False, draw_vectorField=True,  automatic_reference_point=True, 
            point_grid=N_resol, show_streamplot=False)

        for ii in range(len(obs)):
            for jj in range(len(obs)):
                if ii==jj:
                    continue
                boundary_ref_point = obs.get_boundary_reference_point(ii, jj)
                plt.plot(boundary_ref_point[0], boundary_ref_point[1],
                         'r+', linewidth=18, markeredgewidth=4, markersize=13)

    if 3 in options:
        obs = GradientContainer() # create empty obstacle list
        x_lim, y_lim = [-3, 3],[-2, 2]

        obs.append(Cuboid(
            axes_length=[1.3, 1.3],
            x0=[-1, 0.5],
            orientation=0./180*pi
        ))
        
        obs.append(Cuboid(
            axes_length=[1.3, 1.3],
            # p=[1,1],
            x0=[1, -0.5],
            orientation=0./180*pi,
        ))

        xAttractor = [-2., 0.2]

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim, y_lim,  obs=obs, xAttractor=xAttractor,
            saveFigure=saveFigures, figName='linearSystem_avoidanceCube',
            noTicks=False, draw_vectorField=True,  automatic_reference_point=True, 
            point_grid=N_resol, show_streamplot=False)

        for ii in range(len(obs)):
            for jj in range(len(obs)):
                if ii==jj:
                    continue
                boundary_ref_point = obs.get_boundary_reference_point(ii, jj)
                plt.plot(boundary_ref_point[0], boundary_ref_point[1],
                         'r+', linewidth=18, markeredgewidth=4, markersize=13)


    if 4 in options:
        obs = GradientContainer() # create empty obstacle list
        x_lim, y_lim = [-3, 3],[-2, 2]

        obs.append(Cuboid(
            axes_length=[1.3, 1.3],
            margin_absolut=0.4,
            center_position=[-1, -0.5],
            orientation=0./180*pi
        ))
        
        obs.append(Cuboid(
            axes_length=[1.3, 1.3],
            margin_absolut=0.4,
            center_position=[1, 0.5],
            orientation=0./180*pi,
        ))

        xAttractor = [-2., 0.2]

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim, y_lim,  obs=obs, xAttractor=xAttractor,
            saveFigure=saveFigures, figName='linearSystem_avoidanceCube',
            noTicks=False, draw_vectorField=True,  automatic_reference_point=True, 
            point_grid=N_resol, show_streamplot=False)

        for ii in range(len(obs)):
            for jj in range(len(obs)):
                if ii==jj:
                    continue
                boundary_ref_point = obs.get_boundary_reference_point(ii, jj)
                plt.plot(boundary_ref_point[0], boundary_ref_point[1],
                         'r+', linewidth=18, markeredgewidth=4, markersize=13)


    if 5 in options:
        obs = GradientContainer() # create empty obstacle list
        x_lim, y_lim = [-3, 3],[-2, 2]

        obs.append(Cuboid(
            axes_length=[1.3, 1.3],
            # margin_absolut=0.4,
            center_position=[-1, -0.5],
            orientation=0./180*pi
        ))
        
        obs.append(Cuboid(
            axes_length=[1.3, 2.6],
            # margin_absolut=0.4,
            center_position=[1, 0.0],
            orientation=0./180*pi,
        ))

        xAttractor = [-2., 0.2]

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim, y_lim,  obs=obs, xAttractor=xAttractor,
            saveFigure=saveFigures, figName='linearSystem_avoidanceCube',
            noTicks=False, draw_vectorField=True,  automatic_reference_point=True, 
            point_grid=N_resol, show_streamplot=False)

        for ii in range(len(obs)):
            for jj in range(len(obs)):
                if ii==jj:
                    continue
                boundary_ref_point = obs.get_boundary_reference_point(ii, jj)
                plt.plot(boundary_ref_point[0], boundary_ref_point[1],
                         'r+', linewidth=18, markeredgewidth=4, markersize=13)

if (__name__)=="__main__":
    main(options=options, N_resol=N_resol, saveFigures=saveFigures)

# Run function

