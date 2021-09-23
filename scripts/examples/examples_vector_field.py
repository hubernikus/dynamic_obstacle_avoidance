#!/USSR/bin/python3
"""
Script which creates a variety of examples of local modulation of a vector field with obstacle avoidance. 
"""
# Author: Lukas Huber
# Created: 2018-02-15
from math import pi

import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.containers import GradientContainer
from dynamic_obstacle_avoidance.obstacles import Ellipse, Polygon
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import Simulation_vectorFields

########################################################################

# Chose the option you want to run as a number in the option list (integer from -6 to 10)
options = [0]
N_resol = 80
saveFigures=False

########################################################################

def main(options=[0], N_resol=100, saveFigures=False):
    for option in options:
        obs = GradientContainer() # create empty obstacle list
        if option==-6:
            x_lim = [-3,3]
            y_lim = [-0.1,5]
            
            obs.append(Ellipse(a=[0.7, 0.7], p=[1,1], x0=[0, 1.5], th_r=0, w=10, sf=1, xd=[0,0]))
            obs.append(Ellipse(a=[0.7, 0.7], p=[1,1], x0=[1, 1.5], th_r=0, w=10, sf=1, xd=[0,0]))
        # obs.append(Ellipse(a=[0.7, 0.7], p=[1,1], x0=[-0.7, 2.2], th_r=0, w=10, sf=1, xd=[0,0]))
        # obs.append(Ellipse(a=[0.7, 0.7], p=[1,1], x0=[1.7, 2.2], th_r=0, w=10, sf=1, xd=[0,0]))

            pos_attractor = [0., 0]
            
            fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='linearSystem_avoidanceCube', noTicks=False, draw_vectorField=True,  automatic_reference_point=True, point_grid=N_resol, show_streamplot=False)

        if option==-5:
            x_lim = [-3,6.1]
            y_lim = [-5,5]
            
            cuboid_obs = Cuboid(
                axes_length=[3, 3],
                center_position=[1, 0],
                orientation=0./180*pi, 
                absolut_margin=0.0)

            obs.append(cuboid_obs)
            pos_attractor = [4., -2]
            
            fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='linearSystem_avoidanceCube', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=10)

            
        if option==-4:
            x_lim = [-3,6.1]
            y_lim = [-5,5]

            # x_lim = [-10,100]
            # y_lim = [-40,40]

            cuboid_obs = Cuboid(
                axes_length=[3, 3],
                center_position=[1, 0],
                orientation=0./180*pi,
                absolut_margin=0.0)

            obs.append(cuboid_obs)

            # cuboid_obs = Cuboid(
                # axes_length=[3, 1.3],
                # center_position=[0, 2],
                # orientation=-40./180*pi,
                # absolut_margin=0.1)
            # obs.append(cuboid_obs)
            
            # obs[0].set_reference_point([0.3, 3], in_global_frame=False)o

            # Simulation_vectorFields(x_lim, y_lim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='linearSystem_avoidanceCube', noTicks=False, figureSize=(6.,5))

            n_points = 12
            points_init = np.vstack((np.ones(n_points)*x_lim[1], np.linspace(y_lim[0], y_lim[1], n_points)))
            points_attr1 = points_init[:, :int(n_points/2)]
            points_attr2 = points_init[:, int(n_points/2):]

            pos_attractor1 = [1., 1.5]
            pos_attractor2 = [1., -1.5]
            
            points_init = points_init[:, 1:-1]
            fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim, N_resol, obs, pos_attractor=pos_attractor1, saveFigure=saveFigures, figName='linearSystem_avoidanceCube', noTicks=False, draw_vectorField=False, points_init=points_attr2, automatic_reference_point=False)

            plot_streamlines(points_attr1, ax_mod, obs=obs, attractorPos=pos_attractor2)
            ax_mod.plot(pos_attractor2[0],pos_attractor2[1], 'k*',linewidth=18.0, markersize=18)

            figName = "dual_attracor_box_nonSmooth"
            plt.savefig('../figures/' + figName + '.eps', bbox_inches='tight')
            

        if option==-3:
            x_lim = [-6,4.1]
            y_lim = [-2,8]

            # x_lim = [-10,100]
            # y_lim = [-40,40]

            pos_attractor = [-3,0.1]

            a=[0.5, 1.5]
            p=[1,1]
            # x0=[5.5, -0.8]
            x0=[0, 0]
            # th_r=40/180*pi
            th_r=0
            sf=1
            vel = [0, 0]
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=vel))


            a=[0.5, 1.5]
            p=[1,1]
            x0=[0.3, 4]
            th_r=-40/180*pi
            sf=1
            vel = [0, 0]
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=vel))

            obs[0].set_reference_point([0.3, 3], in_global_frame=False)
            obs[1].set_reference_point(obs[0].get_reference_point(in_global_frame=True), in_global_frame=True)
            # Simulation_vectorFields(x_lim, y_lim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='linearSystem_avoidanceCircle', noTicks=False, figureSize=(6.,5))

            n_points = 10
            points_init = np.vstack((np.ones(n_points)*x_lim[1],
                                     np.linspace(y_lim[0], y_lim[1], n_points)))
            
            points_init = points_init[:, 1:-1]
            Simulation_vectorFields(x_lim, y_lim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='linearSystem_avoidanceCircle', noTicks=False, draw_vectorField=False, points_init=points_init, automatic_reference_point=False)


        if option==-2:
            xlim = [-0.8,4.2]
            ylim = [-2,2]

            pos_attractor = [0,0]

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='linearSystem', noTicks=True)

            a=[1, 1]
            p=[1,1]
            x0=[1.5,0]
            th_r=0/180*pi
            sf=1
            vel = [0, 0]
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=vel))

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='linearSystem_avoidanceCircle', noTicks=True)


        if option==-1:
            # Two ellipses placed at x1=0 with dynamic center diplaced and center line in gray
            a=[0.4, 1]
            p=[1, 1]
            x0=[1.5, 0]
            th_r=0/180*pi
            sf=1

            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

            xlim = [-0.5,4]
            ylim = [-2,2]

            pos_attractor = [0,0]

            obs[0].center_dyn = x0

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=False, figName='ellipse_centerMiddle', noTicks=True)


        if option==0:
            # Two ellipses placed at x1=0 with dynamic center diplaced and center line in gray
            a=[0.4, 1]
            p=[1,1]
            x0=[1.5,0]
            th_r=30/180*pi
            sf=1
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf,
                               tail_effect=False, repulsion_coeff=2.0))

            xlim = [-0.5,4]
            ylim = [-2,2]

            pos_attractor = [0,0]

            # obs[0].center_dyn = x0

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=False, figName='ellipse_centerMiddle', noTicks=True)

            # pltLines(pos_attractor, obs[0].center_dyn)
            # if saveFigures:
                # plt.savefig('fig/' + 'ellipseCenterMiddle_centerLine' + '.eps', bbox_inches='tight')       
            # rat = 0.6
            # obs[0].center_dyn = [x0[0] - rat*np.sin(th_r)*a[1],
                                 # x0[1] - rat*np.cos(th_r)*a[1]]
            # Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=False, figName='ellipse_centerNotMiddle', noTicks=True)
            # pltLines(pos_attractor, obs[0].center_dyn)

            # if saveFigures:
                # plt.savefig('fig/' + 'ellipseCenterNotMiddle_centerLine' + '.eps', bbox_inches='tight')


        if option==1:
            # Two ellipses combined to represent robot arm -- remove outlier plotting when saving!!
            obs.append(Ellipse(
                a=[1.1, 1.4],
                p=[2,2],
                x0=[3.0, 1.3],
                th_r=0/180*pi,
                sf=1))

            obs.append(Ellipse(
                a=[2, 0.4],
                p=[2,2],
                x0=[2.5, 3],
                th_r=-20/180*pi,
                sf=1))

            xlim = [-1.0, 6.5]
            ylim = [-0.3, 5.2]

            pos_attractor = [0,0]

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=False, figName='convexRobot', noTicks=True)

        if option==2:
            # Decomposition several obstacles - obstacle 1, obstacle 2, both obstacles
            pos_attractor = np.array([0,0])
            centr = [2, 2.5]

            obs = []
            a = [0.5,1.2]
            p = [1,1]
            x0 = [-1.0, 3.2]
            th_r = -60/180*pi
            sf = 1.0
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

            a = [1.4,1.0]
            p = [3,3]
            x0 = [1.2, 1.5]
            th_r = -30/180*pi
            sf = 1.0
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

            xlim = [-2.5,3.2]
            ylim = [-0.3, 5.2]

            col1= [0.7,0.3,0]
            fig, ax =Simulation_vectorFields(xlim, ylim, N_resol, [obs[0]], pos_attractor=pos_attractor, saveFigure=saveFigures, figName='linearCombination_obstacle0', noTicks=True, streamColor=col1, obstacleColor=[col1], alphaVal=0.7)

            col2 = [0.05,0.3,0.05]
            Simulation_vectorFields(xlim, ylim, N_resol, [obs[1]], pos_attractor=pos_attractor, saveFigure=saveFigures, figName='linearCombination_obstacle_overlay', noTicks=True, streamColor=col2, obstacleColor=[col2], figHandle=[fig, ax], alphaVal=0.7)

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='linearCombination_obstaclesBoth', noTicks=True)

        if option==3:
            # Three obstacles touching, with and without common center

            xlim = [-1,7]
            xlim = [xlim[ii]*0.8 for ii in range(2)]

            ylim = [-3.5,3.5]
            ylim = [ylim[ii]*0.8 for ii in range(2)]

            ### Three obstacles touching - convergence
            pos_attractor = np.array([0,0])
            centr = [2, 0]
            dx = 0.5
            dy = 2.5

            a = [0.6,0.6]
            p = [1,1]
            x0 = [2.+dx, 3.2-dy]
            th_r = -60/180*pi
            sf = 1.2
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
            #obs[0].center_dyn = centr

            a = [1,0.4]
            p = [1,3]
            x0 = [1.5+dx, 1.6-dy]
            th_r = +60/180*pi
            sf = 1.2
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
            #obs[1].center_dyn = centr

            a = [1.2,0.2]
            p = [2,2]
            x0 = [3.3+dx,2.1-dy]
            th_r = -20/180*pi
            sf = 1.2
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))
            #obs[2].center_dyn = centr


            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='three_obstacles_touching')

            pos_attractor = np.array([0,0])
            centr = [1.5, 0]

            dx = 0.5
            dy = 3.0

            ### Three obstacles touching -- no common center, no convergence
            obs = []
            a = [0.6,0.6]
            p = [1,1]
            x0 = [2.5+dx, 4.1-dy]
            th_r = 0*pi
            sf = 1.2
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

            a = [1.2,0.2]
            p = [2,2]
            x0 = [2.3+dx,3.1-dy]
            th_r = 20/180*pi
            sf = 1.2
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

            a = [1,0.4]
            p = [1,4]
            x0 = [2.5+dx, 2.0-dy]
            th_r= -25/180*pi
            sf = 1.2
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='three_obstacles_touching_noConvergence')

        if option==4:
            # Ellipses being overlapping an being perpendicular or parallel to flow 
            xlim = [-1,7]
            ylim = [-3.5,3.5]

            ### Three obstacles touching - convergence
            pos_attractor = np.array([0,0])
            centr = [2, 2.5]

            a = [1.4,0.3]
            p = [1,1]
            x0 = [3, 0.9]
            th_r = +40/180*pi
            sf = 1.2
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

            a = [1.2,0.4]
            p = [1,3]
            x0 = [3, -1.0]
            th_r = -40/180*pi
            sf = 1.2
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='twoEllipses_concaveRegion_front')

            a = [1.8,0.3]
            p = [1,1]
            x0 = [4.0, -0.4]
            th_r = +60/180*pi
            sf = 1.2
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

            a = [1.2,0.4]
            p = [1,3]
            x0 = [2.3, -0.0]
            th_r = -40/180*pi
            sf = 1.2
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='twoEllipses_concaveRegion_top')

        if option==5:
            # TOOD -- radial displacement algorithm
            # Ellipses being overlapping an being perpendicular or parallel to flow 
            xlim = [-1,4]
            ylim = [-2,2]

            ### Three obstacles touching - convergence
            pos_attractor = np.array([0,0])

            a = [0.4,1.5]
            p = [1,1]
            x0 = [1.6, 0.0]
            th_r = +0/180*pi
            sf = 1.0
            xd = [-3,3]
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='movingEllipse_notMoving', obs_avoidance_func=obs_avoidance_radialDisplace)

        if option==7:
            # Ellipse Avoidance with and without tail effect
            xlim = [-0.1,4]
            ylim = [-2,2]

            ### Three obstacles touching - convergence
            pos_attractor = np.array([0,0])

            a = [0.35,1.4]
            p = [1,1]
            x0 = [1.9, 0.0]
            th_r = -15/180*pi
            sf = 1.0
            xd = [-3,3]
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf, tail_effect=False))


            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='ellipse_tailEffectOff', obs_avoidance_func=obs_avoidance_interpolation_moving, showLabel=False)

            obs[0].tail_effect = True

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='ellipse_tailEffectOn', obs_avoidance_func=obs_avoidance_interpolation_moving, showLabel=False)

        if option==7:
            figSize = (14,12)

            xlim = [-10,10]
            ylim = [-8,8]

            pos_attractor = np.array([0,0])
            centr = [2, 2.5]

            N_obs = 12
            R = 5
            th_r0 = 38/180*pi
            rCent=2.4
            for n in range(N_obs):
                obs.append(Ellipse(
                    a = [0.4,3],
                    p = [1,1],
                    x0 = [R*cos(2*pi/N_obs*n), R*sin(2*pi/N_obs*n)],
                    th_r = th_r0 + 2*pi/N_obs*n,
                    sf = 1.0,
                    tail_effect=False))

                obs[n].center_dyn=[obs[n].x0[0]-rCent*sin(obs[n].th_r),
                                   obs[n].x0[1]+rCent*cos(obs[n].th_r)]

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='ellipseCircle_tailEffectOff', obs_avoidance_func=obs_avoidance_interpolation_moving, showLabel=False, figureSize=figSize)

            for n in range(N_obs):
                obs[n].tail_effect = True

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='ellipseCircle_tailEffectOn', obs_avoidance_func=obs_avoidance_interpolation_moving, showLabel=False, figureSize=figSize)

        if option==9:
            figSize = (5*2.5,3*2)

            xlim = [1,11]
            ylim = [1,7]

            pos_attractor = np.array([2,1.5])

            N_obs = 8

            x0_list = [[2.95,6],
                       [4.3, 2.7],
                       [4.35, 4.3],
                       [6.9, 1.6],
                       [7.0, 3.95],
                       [6.8, 6.2],
                       [8.5, 3.5],
                       [9.0, 6.25]
            ]

            for n in range(len(x0_list)):
                obs.append(Ellipse(
                    a = [0.5,0.5],
                    p = [1,1],
                    x0 = x0_list[n],
                    th_r = 0,
                    sf = 2.0))

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=saveFigures, figName='wheelchairSimulation', obs_avoidance_func=obs_avoidance_interpolation_moving, showLabel=False, figureSize=figSize)

        if option==8:
            # Two ellipses placed at x1=0 with dynamic center diplaced and center line in gray -- with color Code for velocity
            a=[0.5, 1.4]
            p=[1,1]
            x0=[2,0]
            th_r=0/180*pi
            sf=1
            obs.append(Ellipse(a=a, p=p, x0=x0,th_r=th_r, sf=sf))


            xlim = [-0.5,4]
            ylim = [-2,2]

            pos_attractor = [0,0]

            obs[0].center_dyn = x0

            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=False, figName='ellipse_centerMiddle', noTicks=True, showLabel=False, colorCode=True)
            pltLines(pos_attractor, obs[0].center_dyn)
            if saveFigures:
                plt.savefig('fig/' + 'ellipseCenterMiddle_centerLine_pres_colMap' + '.eps', bbox_inches='tight')

            rat = -0.9
            obs[0].center_dyn = [x0[0] - rat*np.sin(th_r)*a[1],
                                 x0[1] - rat*np.cos(th_r)*a[1]]
            Simulation_vectorFields(xlim, ylim, N_resol, obs, pos_attractor=pos_attractor, saveFigure=False, figName='ellipse_centerNotMiddle', noTicks=True, showLabel=False, colorCode=True)
            pltLines(pos_attractor, obs[0].center_dyn)
            if saveFigures:
                plt.savefig('fig/' + 'ellipseCenterNotMiddle_centerLine_pres_colMap' + '.eps', bbox_inches='tight')

if (__name__)=="__main__":
    if False:
        if len(sys.argv) > 1:
            options = sys.argv[1]

            if len(sys.argv) > 2:
                N_resol = sys.argv[2]

                if len(sys.argv) > 3:
                    saveFigures = sys.argv[3]

    main(options=options, N_resol=N_resol, saveFigures=saveFigures)

# Run function

