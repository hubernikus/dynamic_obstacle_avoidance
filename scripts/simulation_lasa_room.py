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


# Custom libraries
# from dynamic_obstacle_avoidance.dynamical_system import linearAttractor
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import obs_avoidance_interpolation_moving
                
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import *


########################################################################
# Chose the option you want to run as a number in the option list (integer from -2 to 10)
# options = [0, 1, 2]
options = [0]

N_resol = 40
saveFigures=False
########################################################################

def main(options=[], N_resol=100, saveFigures=False):
    for option in options:
        obs = GradientContainer() # create empty obstacle list
        if option==0:
            x_lim = [-0.5, 6.5]
            y_lim = [-0.5, 6.0]
            # x_lim = [-5.5, 12.5]
            # y_lim = [-5.5, 12.5]

            xAttractor=[5.0, 5.0]
            # xAttractor=[1, 1]

            robot_margin = 0.6
            
            # obs.append(Cuboid(
                # axes_length=[8, 7],
                # center_position=[3, 3.5],
                # orientation=0./180*pi,
                # margin_absolut=robot_margin,
                # is_boundary=True,
            # ))

            # obs.append(Polygon(
                # edge_points=[[ 0.0, 6.5, 6.5, 0.0, 0.0, 0.8, 0.8, 0.0],
                             # [-0.6,-0.6, 5.5, 5.5, 4.2, 3.8, 2.0, 1.6]],
                # center_position=[5, 3.5],
                # orientation=0./180*pi,
                # margin_absolut=robot_margin,
                # is_boundary=True,
            # ))

            obs.append(Polygon(
                edge_points=[[ 0.0, 6.0, 6.0, 0.0],
                             [0.0, 0.0, 5.5, 5.5]],
                # center_position=[, 3.5],
                orientation=0./180*pi,
                margin_absolut=robot_margin,
                is_boundary=True,
            ))
            

            obs.append(Cuboid(
                axes_length=[1.6, 0.8],
                center_position=[0.4, 3.0],
                orientation=90./180*pi,
                margin_absolut=robot_margin,
                is_boundary=False
            ))

            obs.append(Cuboid(
                axes_length=[1.6, 1.6],
                center_position=[3.4, 2.6],
                orientation=0./180*pi,
                margin_absolut=robot_margin,
                is_boundary=False
            ))

            obs.append(Ellipse(
                axes_length=[0.2, 0.5],
                center_position=[1.8, 2.5],
                p=[1,1],
                orientation=30./180*pi,
                margin_absolut=robot_margin,
                is_boundary=False
            ))

            vectorfield = True
            if vectorfield:
                fig_mod, ax_mod = Simulation_vectorFields(
                    x_lim, y_lim, obs=obs, xAttractor=xAttractor, saveFigure=saveFigures,
                    figName='linearSystem_boundaryCuboid', noTicks=False, draw_vectorField=True,
                    show_streamplot=False,
                    automatic_reference_point=True, point_grid=N_resol,
                    # figureSize=(6., 4.25),
                )
                
            else:
                
                # Specific value
                position = np.array([1.06, 1.07])

                ds_init = linearAttractor(position, x0=xAttractor)
                ds_mod = obs_avoidance_interpolation_moving(position, ds_init, obs)

                print('ds_mod', ds_mod)
                import pdb; pdb.set_trace()


if (__name__)=="__main__":
    if len(sys.argv) > 1 and not sys.argv[-1]== '-i':
        options = sys.argv[1]
        
        if len(sys.argv) > 2:
            N_resol = sys.argv[2]

            if len(sys.argv) > 3:
                saveFigures = sys.argv[3]

    main(options=options, N_resol=N_resol, saveFigures=saveFigures)

    # input("\nPress enter to continue...")

# Run function

