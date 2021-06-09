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

N_resol = 10
saveFigures=False

########################################################################


def main(options=[], N_resol=100, saveFigures=False):
    for option in options:
        obs = GradientContainer() # create empty obstacle list
        if option==0:
            x_lim = [-1.1,8.1]
            y_lim = [-3.9,6.3]

            xAttractor=[1,0]
            
            obs.append(Cuboid(
                axes_length=[8, 9.6],
                center_position=[3, 1],
                orientation=0./180*pi,
                margin_absolut=0.5,
                is_boundary=True,
            ))
            
            # obs.append(Ellipse(
            #     axes_length=[0.4, 1.2],
            #     center_position=[5, 2.1],
            #     p=[1,1],
            #     orientation=150./180*pi,
            #     margin_absolut=0.5,
            #     is_boundary=False
            # ))

            obs.append(Ellipse(
                axes_length=[0.4, 1.2],
                center_position=[3, 5.0],
                p=[1,1],
                orientation=-70./180*pi,
                margin_absolut=0.5,
                is_boundary=False
            ))
                                
            fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_boundaryCuboid', noTicks=False, draw_vectorField=True,  automatic_reference_point=True, point_grid=N_resol)

            # import pdb; pdb.set_trace()

            # fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=[], xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_initial', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=N_resol)

            
        if option==1:
            x_lim = [-1.1,8.1]
            y_lim = [-3.9,6.3]

            xAttractor=[1,0]
            
            obs.append(Cuboid(
                axes_length=[8, 9.6],
                center_position=[3, 1],
                orientation=0./180*pi,
                margin_absolut=0.0,
                is_boundary=True))
            
            fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_boundaryCuboid_twoEllipses', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=N_resol)

        if option==2:
            x_lim = [-1.1,8.1]
            y_lim = [-3.9,6.1]

            xAttractor=[1,0]
            
            obs.append(Ellipse(axes_length=[3.5, 4.0], center_position=[4, 2.0], p=[1,1], orientation=-70./180*pi, sf=1, is_boundary=True))
            
            fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_boundaryEllipse', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=N_resol)

        if option==3:
            x_lim = [-1.1,8.1]
            y_lim = [-3.9,6.1]

            xAttractor=[1,0]
            
            obs.append(Ellipse(axes_length=[3.5, 4.0], center_position=[4, 2.0], p=[1,1], orientation=-70./180*pi, sf=1, is_boundary=True))
            
            fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_boundaryEllipse', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=N_resol)

        if option==4:
            # Three intersecting obstacles
            x_lim = [-0.1,10.1]
            y_lim = [-0.1,10.1]

            xAttractor=[1,1]
            
            obs.append(Ellipse(axes_length=[0.7, 2.],
                               center_position=[5, 1.0], p=[1,1],
                               orientation=0./180*pi, sf=1))

            obs.append(Ellipse(axes_length=[0.6, 2.],
                               center_position=[5, 4.0], p=[1,1],
                               orientation=0./180*pi, sf=1))

            obs.append(Ellipse(axes_length=[0.7, 2.],
                               center_position=[7, 4.0], p=[1,1],
                               orientation=90./180*pi, sf=1))

            # obs.append(Ellipse(axes_length=[0.7, 2.],
                               # center_position=[5, 4.0], p=[1,1],
                               # orientation=0./180*pi, sf=1))

            # for oo in range(len(obs)):
                # obs[oo].set_reference_point(np.array([5,-0.05]), in_global_frame=True)
            

            obs.append(Cuboid(axes_length=[10, 10],
                              center_position=[5, 5], orientation=0./180*pi, margin_absolut=0.0,
                              is_boundary=True))
            
            fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='threeEllipses_inCube', noTicks=False, draw_vectorField=True,  automatic_reference_point=True, point_grid=N_resol)

        if option==5:
            # Three intersecting obstacles
            x_lim = [-0.1,10.1]
            y_lim = [-0.1,10.1]

            xAttractor=[1,1]
            
            obs.append(Ellipse(axes_length=[0.7, 2.],
                               center_position=[5, 1.0], p=[1,1],
                               orientation=0./180*pi, sf=1))

            obs.append(Ellipse(axes_length=[0.6, 2.],
                               center_position=[5, 4.0], p=[1,1],
                               orientation=0./180*pi, sf=1))

            obs.append(Ellipse(axes_length=[0.7, 2.],
                               center_position=[7, 4.0], p=[1,1],
                               orientation=90./180*pi, sf=1))

            obs.append(Cuboid(axes_length=[10, 10],
                              center_position=[5, 5], orientation=0./180*pi, margin_absolut=0.0,
                              is_boundary=True))
            
            fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='threeEllipses_inCube_bad', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=N_resol)


if (__name__)=="__main__":
    argv_copy = copy.deepcopy(sys.argv)
    
    if argv_copy[2] == '-i':
        del argv_copy[2]
        del argv_copy[1]
        
    if len(argv_copy) > 1:
        options = argv_copy[1]

    if len(argv_copy) > 2:
        N_resol = argv_copy[2]

    if len(argv_copy) > 3:
        saveFigures = argv_copy[3]

    # options = [0,1,2,3,4,5]
    main(options=options, N_resol=N_resol, saveFigures=saveFigures)

    # input("\nPress enter to continue...")

# Run function

