'''
Obstacle Avoidance Algorithm script with vecotr field

@author LukasHuber
@date 2018-02-15
'''


# General classes
import numpy as np
from numpy import pi
import copy
import time

import matplotlib.pyplot as plt
import matplotlib

from dynamic_obstacle_avoidance.dynamical_system import *
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import *
from dynamic_obstacle_avoidance.obstacle_avoidance.nonlinear_modulation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_dynamic_center_3d import dynamic_center_3d


def pltLines(pos0, pos1, xlim=[-100,100], ylim=[-100,100]):
    if pos1[0]-pos0[0]: # m < infty
        m = (pos1[1] - pos0[1])/(pos1[0]-pos0[0])
        
        ylim=[0,0]
        ylim[0] = pos0[1] + m*(xlim[0]-pos0[0])
        ylim[1] = pos0[1] + m*(xlim[1]-pos0[0])
    else:
        xlim = [pos1[0], pos1[0]]
    
    plt.plot(xlim, ylim, '--', color=[0.3, 0.3, 0.3], linewidth=2)


def plot_streamlines(points_init, ax, obs=[], attractorPos=[0,0],
                     dim=2, dt=0.01, max_simu_step=300, convergence_margin=0.03):
    
    n_points = np.array(points_init).shape[1]

    x_pos = np.zeros((dim, max_simu_step+1, n_points))
    x_pos[:,0,:] = points_init

    it_count = 0
    for iSim in range(max_simu_step):
        for j in range(n_points):
            x_pos[:, iSim+1,j] = obs_avoidance_rk4(
                dt, x_pos[:,iSim, j], obs, x0=attractorPos,
                obs_avoidance=obs_avoidance_interpolation_moving)

         # Check convergence
        if (np.sum((x_pos[:, iSim+1, :]-np.tile(attractorPos, (n_points,1)).T)**2)
            < convergence_margin):
            x_pos = x_pos[:, :iSim+2, :]

            print("Convergence reached after {} iterations.".format(it_count))
            break

        it_count += 1
    for j in range(n_points):
        ax.plot(x_pos[0, :, j], x_pos[1, :, j], '--', lineWidth=4)
        ax.plot(x_pos[0, 0, j], x_pos[1, 0, j], 'k*', markeredgewidth=4, markersize=13)
        
    # return x_pos

    
def Simulation_vectorFields(x_range=[0,10], y_range=[0,10], point_grid=10, obs=[], sysDyn_init=False, xAttractor = np.array(([0,0])), saveFigure=False, figName='default', noTicks=True, showLabel=True, figureSize=(12.,9.5), obs_avoidance_func=obs_avoidance_interpolation_moving, attractingRegion=False, drawVelArrow=False, colorCode=False, streamColor=[0.05,0.05,0.7], obstacleColor=[], plotObstacle=True, plotStream=True, figHandle=[], alphaVal=1, dynamicalSystem=linearAttractor, draw_vectorField=True, points_init=[], show_obstacle_number=False, automatic_reference_point=True, nonlinear=True, show_streamplot=True):
    
    dim = 2

    # Numerical hull of ellipsoid 
    for n in range(len(obs)): 
        obs[n].draw_obstacle(numPoints=50) # 50 points resolution 


    # Adjust dynamic center
    if automatic_reference_point:
        intersection_obs = get_intersections_obstacles(obs)
        get_dynamic_center_obstacles(obs, intersection_obs)

    # Numerical hull of ellipsoid 
    for n in range(len(obs)): 
        obs[n].draw_obstacle(numPoints=50) # 50 points resolution 


    if len(figHandle): 
        fig_ifd, ax_ifd = figHandle[0], figHandle[1] 
    else:
        fig_ifd, ax_ifd = plt.subplots(figsize=figureSize) 
        
    if plotObstacle:
        obs_polygon = []
        obs_polygon_sf = []

        for n in range(len(obs)):
            # plt.plot([x_obs_sf[i][0] for i in range(len(x_obs_sf))], [x_obs_sf[i][1] for i in range(len(x_obs_sf))], 'k--')
            x_obs = obs[n].boundary_points_global_closed
            x_obs_sf = obs[n].boundary_points_margin_global_closed
            plt.plot(x_obs_sf[0, :], x_obs_sf[1, :], 'k--')
            
            if obs[n].is_boundary:
                outer_boundary = np.array([[x_range[0], x_range[1], x_range[1], x_range[0]],
                                           [y_range[0], y_range[0], y_range[1], y_range[1]]]).T
                
                boundary_polygon = plt.Polygon(outer_boundary, alpha=0.8, zorder=-2)
                boundary_polygon.set_color(np.array([176,124,124])/255.)
                plt.gca().add_patch(boundary_polygon)

                obs_polygon.append( plt.Polygon(x_obs.T, alpha=1.0, zorder=-1))
                obs_polygon[n].set_color(np.array([1.0,1.0,1.0]))

            else:
                obs_polygon.append( plt.Polygon(x_obs.T, alpha=0.8, zorder=2))
                
                if len(obstacleColor)==len(obs):
                    obs_polygon[n].set_color(obstacleColor[n])
                else:
                    obs_polygon[n].set_color(np.array([176,124,124])/255)
            
            obs_polygon_sf.append( plt.Polygon(x_obs_sf.T, zorder=1, alpha=0.2))
            obs_polygon_sf[n].set_color([1,1,1])

            plt.gca().add_patch(obs_polygon_sf[n])
            plt.gca().add_patch(obs_polygon[n])

            if show_obstacle_number:
                ax_ifd.annotate('{}'.format(n+1), xy=np.array(obs[n].center_position)+0.16, textcoords='data', size=16, weight="bold")
            
            ax_ifd.plot(obs[n].center_position[0], obs[n].center_position[1], 'k.')
            
            # automatic adaptation of center
            reference_point = obs[n].get_reference_point(in_global_frame=True)
            
            if not any(reference_point==None):
                ax_ifd.plot(reference_point[0],reference_point[1], 'k+', linewidth=18, markeredgewidth=4, markersize=13)
                # ax_ifd.annotate('{}'.format(obs[n].hirarchy), xy=reference_point+0.08, textcoords='data', size=16, weight="bold")  #
                ax_ifd.annotate('{}'.format(n), xy=reference_point+0.08, textcoords='data', size=16, weight="bold")  #
                # add group, too
                
            if drawVelArrow and np.linalg.norm(obs[n].xd)>0:
                col=[0.5,0,0.9]
                fac=5 # scaling factor of velocity
                ax_ifd.arrow(obs[n].center_position[0], obs[n].center_position[1], obs[n].xd[0]/fac, obs[n].xd[1]/fac, head_width=0.3, head_length=0.3, linewidth=10, fc=col, ec=col, alpha=1)

    plt.gca().set_aspect('equal', adjustable='box')

    ax_ifd.set_xlim(x_range)
    ax_ifd.set_ylim(y_range)

    if noTicks:
        plt.tick_params(axis='both', which='major',bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    if showLabel:
        plt.xlabel(r'$\xi_1$', fontsize=16)
        plt.ylabel(r'$\xi_2$', fontsize=16)

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    ax_ifd.plot(xAttractor[0],xAttractor[1], 'k*',linewidth=18.0, markersize=18)

    # Show certain streamlines
    if np.array(points_init).shape[0]:
        plot_streamlines(points_init, ax_ifd, obs, xAttractor)

    if not draw_vectorField:
        plt.ion()
        plt.show()
        return fig_ifd, ax_ifd
        # return

    start_time = time.time()

    # Create meshrgrid of points
    if type(point_grid)==int:
        N_x = N_y = point_grid
        YY, XX = np.mgrid[y_range[0]:y_range[1]:N_y*1j, x_range[0]:x_range[1]:N_x*1j]

    else:
        N_x = N_y = 1
        XX, YY = np.array([[point_grid[0]]]), np.array([[point_grid[1]]])

    # TODO: DEBUGGING Only for Development and testing
    ########## START REMOVE ##########
    # N_x = N_y = 1
    # XX = np.zeros((N_x, N_y))
    # YY = np.zeros((N_x, N_y))

    it_start = 0
    n_samples = 0
    
    pos1 = [-1.516, -1.100]
    pos2 = [-1.4, -1.1]

    x_sample_range = [pos1[0], pos2[0]]
    y_sample_range = [pos1[1], pos2[1]]

    x_sample = np.linspace(x_sample_range[0], x_sample_range[1], n_samples)
    y_sample = np.linspace(y_sample_range[0], y_sample_range[1], n_samples)

    ii = 0
    for ii in range(n_samples):
        iy = (ii+it_start) % N_y
        ix = int((ii+it_start) /N_x)
        
        XX[ix, iy] = x_sample[ii]
        YY[ix, iy] = y_sample[ii]
    ########## STOP REMOVE ###########
    

    if attractingRegion: # Forced to attracting Region
        def obs_avoidance_temp(x, xd, obs):
            return obs_avoidance_func(x, xd, obs, xAttractor)
        
        obs_avoidance= obs_avoidance_temp
    else:
        obs_avoidance = obs_avoidance_func
        
    xd_init = np.zeros((2,N_x,N_y))
    xd_mod  = np.zeros((2,N_x,N_y))

    for ix in range(N_x):
        for iy in range(N_y):
            pos = np.array([XX[ix,iy],YY[ix,iy]])

            xd_init[:,ix,iy] = dynamicalSystem(pos, x0=xAttractor) # initial DS
            xd_mod[:,ix,iy] = obs_avoidance(pos, xd_init[:,ix,iy], obs) # DEBUGGING: remove

    if sysDyn_init:
        fig_init, ax_init = plt.subplots(figsize=(5,2.5))
        res_init = ax_init.streamplot(XX, YY, xd_init[0,:,:], xd_init[1,:,:], color=[(0.3,0.3,0.3)])
        
        ax_init.plot(xAttractor[0], xAttractor[1], 'k*')
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlim(x_range)
        plt.ylim(y_range)

    indOfnoCollision = obs_check_collision_2d(obs, XX, YY)
    # indOfnoCollision = np.zeros(np.squeeze(xd_mod[0,:,:]).shape) # DEBUGGING: remove
    dx1_noColl = np.squeeze(xd_mod[0,:,:]) * indOfnoCollision
    dx2_noColl = np.squeeze(xd_mod[1,:,:]) * indOfnoCollision

    end_time = time.time()

    print('Number of points: {}'.format(point_grid*point_grid))
    print('Average time: {} ms'.format(np.round((end_time-start_time)/(N_x*N_y)*1000),5) )
    print('Modulation calulcation total: {} s'.format(np.round(end_time-start_time), 4))

    if plotStream:
        if colorCode:
            velMag = np.linalg.norm(np.dstack((dx1_noColl, dx2_noColl)), axis=2 )/6*100

            strm = res_ifd = ax_ifd.streamplot(XX, YY,dx1_noColl, dx2_noColl, color=velMag, cmap='winter', norm=matplotlib.colors.Normalize(vmin=0, vmax=10.) )
        else:
            # Normalize
            normVel = np.sqrt(dx1_noColl**2 + dx2_noColl**2)
            ind_nonZero = normVel>0
            dx1_noColl[ind_nonZero] = dx1_noColl[ind_nonZero]/normVel[ind_nonZero]
            dx2_noColl[ind_nonZero] = dx2_noColl[ind_nonZero]/normVel[ind_nonZero]

            if show_streamplot:
                res_ifd = ax_ifd.streamplot(XX, YY,dx1_noColl, dx2_noColl, color=streamColor, zorder=0)
            else:
                res_ifd = ax_ifd.quiver(XX, YY, dx1_noColl, dx2_noColl, color=streamColor, zorder=0)
                # res_ifd = ax_ifd.quiver(XX, YY, xd_init[0,:,:], xd_init[1,:,:], color=[0.8, 0.2, 0.2], zorder=0)

    plt.ion()
    plt.show()

    if saveFigure:
        # plt.savefig('figures/' + figName + '.eps', bbox_inches='tight')
        try:
            plt.savefig('figures/' + figName + '.png', bbox_inches='tight')
        except:
            plt.savefig('../figures/' + figName + '.png', bbox_inches='tight')
    return fig_ifd, ax_ifd
