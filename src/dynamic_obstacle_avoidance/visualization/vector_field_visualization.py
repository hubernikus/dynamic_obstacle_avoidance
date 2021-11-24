#!/USSR/bin/python3
'''
Obstacle Avoidance Algorithm script with vecotr field
'''

from dynamic_obstacle_avoidance.dynamical_system import *
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import *
from dynamic_obstacle_avoidance.obstacle_avoidance.nonlinear_modulation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_dynamic_center_3d import get_dynamic_center_obstacles

# General classes
import copy
import time
import os
import warnings

import numpy as np
from numpy import pi

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection

from scipy import ndimage
import matplotlib.image as mpimg

# Show plot in a reactive manner

__author__ = "Lukas Huber"
__date__ =  "2018-02-15"
__email__ = "lukas.huber@epfl.ch"


def plt_speed_line_and_qolo(points_init, attractorPos, obs, max_simu_step=500, dt=0.01, convergence_margin=1e-4, fig_and_ax_handle=None, normalize_magnitude=True, line_color=None, min_value=0, max_value=None):
    ''' Draw line where qolo has been moving along. '''

    if fig_and_ax_handle is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_and_ax_handle
    
    dim = 2 # 2-D problem
    if (len(points_init.shape)==1):
        points_init = np.array([points_init]).T

    n_points = points_init.shape[1]
    for j in range(n_points):
        x_pos = np.zeros((dim, max_simu_step+1))
        x_pos[:, 0] = points_init[:, j]

        it_count = 0
        
        for it_count in range(max_simu_step):
            x_pos[:, it_count+1] = obs_avoidance_rk4(
                dt, x_pos[:, it_count], obs, x0=attractorPos,
                obs_avoidance=obs_avoidance_interpolation_moving)

            # Check convergence
            if (np.linalg.norm(x_pos[:, it_count+1] - attractorPos) < convergence_margin):
                
                x_pos = x_pos[:, :it_count+2]
                
                print("Convergence reached after {} iterations.".format(it_count))
                break

            if (np.linalg.norm(x_pos[:, it_count+1]-x_pos[:, it_count]) < convergence_margin):
                x_pos = x_pos[:, :it_count+2]
                
                print("Stopping at local minimum after {} iterations.".format(it_count))
                break

            it_count += 1

        if line_color is None:
            magnitude = np.linalg.norm(x_pos[:, :-1] - x_pos[:, 1:], axis=0)
            if normalize_magnitude:
                magnitude = magnitude/magnitude.max()
                min_value, max_value = 0, 1
            else:
                if min_value is None:
                    max_value = magnitude.max()

                if max_value is None:
                    min_value = magnitude.min()

            points = x_pos.T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # norm = plt.Normalize(min_value, max_value)
            norm = ax.Normalize(min_value, max_value)
            lc = LineCollection(segments, cmap='jet', norm=norm)

            # Set the values used for colormapping
            lc.set_array(magnitude)
            lc.set_linewidth(2)
            line = ax.add_collection(lc)
            # ax.axis('equal')
            ax.axis('equal')

            if False:
                fig.colorbar(line, ax=ax)
        else:
            # line = plt.plot(x_pos[0, :], x_pos[1, :], color=line_color, linewidth=3)
            # line = ax.plot(x_pos[0, :], x_pos[1, :], color=line_color, linewidth=3, zorder=6)
            line = ax.plot(x_pos[0, :], x_pos[1, :], color=line_color, linewidth=3)
        
        arr_img = mpimg.imread(os.path.join('data', 'Qolo_T_CB_top_bumper.png'))
    
        length_x = 1.2
        length_y = (1.0)*arr_img.shape[0]/arr_img.shape[1] * length_x

        dx0 = x_pos[0, 1] - x_pos[0, 0]
        dy0 = x_pos[1, 1] - x_pos[1, 0]
        rot = np.arctan2(dy0, dx0) 
        arr_img_rotated = ndimage.rotate(arr_img, rot*180.0/pi, cval=255)
    
        length_x_rotated = (np.abs(np.cos(rot))*length_x + np.abs(np.sin(rot))*length_y)
    
        length_y_rotated = (np.abs(np.sin(rot))*length_x + np.abs(np.cos(rot))*length_y)

        ax.imshow(arr_img_rotated, extent=[
            x_pos[0, 0]-length_x_rotated/2.0, x_pos[0, 0]+length_x_rotated/2.0,
            x_pos[1, 0]-length_y_rotated/2.0, x_pos[1, 0]+length_y_rotated/2.0 ], zorder=-2)

    return line


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
    ''' Plot streamlines. '''
    # 
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
        ax.plot(x_pos[0, :, j], x_pos[1, :, j], '--', lineWidth=4, color='r')
        ax.plot(x_pos[0, 0, j], x_pos[1, 0, j], 'k*', markeredgewidth=4, markersize=13, zorder=5)
        
    # return x_pos

    
def plot_obstacles(ax, obs, x_range, y_range, xAttractor=None, obstacleColor=None, show_obstacle_number=False, reference_point_number=False, drawVelArrow=True, noTicks=False, showLabel=True, draw_wall_reference=False, border_linestyle='--'):
    ''' Plot all obstacles & attractors '''

    if not xAttractor is None:
        ax.plot(xAttractor[0], xAttractor[1], 'k*',linewidth=18.0, markersize=18, zorder=5)

    obs_polygon = []
    obs_polygon_sf = []

    for n in range(len(obs)):
        x_obs = obs[n].boundary_points_global_closed
        x_obs_sf = obs[n].boundary_points_margin_global_closed
        ax.plot(x_obs_sf[0, :], x_obs_sf[1, :], color='k', linestyle=border_linestyle)

        if obs[n].is_boundary:
            outer_boundary = None
            if hasattr(obs[n], 'global_outer_edge_points'):
                outer_boundary = obs[n].global_outer_edge_points
                
            if outer_boundary is None:
                outer_boundary = np.array([[x_range[0], x_range[1], x_range[1], x_range[0]],
                                           [y_range[0], y_range[0], y_range[1], y_range[1]]])
            outer_boundary = outer_boundary.T
            boundary_polygon = plt.Polygon(outer_boundary, alpha=1.0, zorder=-4)
            boundary_polygon.set_color(np.array([176, 124, 124])/255.)
            ax.add_patch(boundary_polygon)

            obs_polygon.append(plt.Polygon(x_obs.T, alpha=1.0, zorder=-3))
            obs_polygon[n].set_color(np.array([1.0, 1.0, 1.0]))

        else:
            obs_polygon.append(plt.Polygon(x_obs.T, alpha=0.8, zorder=2))

            if obstacleColor is None:
                obs_polygon[n].set_color(np.array([176,124,124])/255)
            else:
                obs_polygon[n].set_color(obstacleColor[n])
            
        obs_polygon_sf.append(plt.Polygon(x_obs_sf.T, zorder=1, alpha=0.2))
        obs_polygon_sf[n].set_color([1, 1, 1])

        ax.add_patch(obs_polygon_sf[n])
        ax.add_patch(obs_polygon[n])


        if show_obstacle_number:
            ax.annotate('{}'.format(n+1), xy=np.array(obs[n].center_position)+0.16, textcoords='data', size=16, weight="bold")

        if not obs[n].is_boundary or draw_wall_reference:
            ax.plot(obs[n].center_position[0], obs[n].center_position[1], 'k.')

        # automatic adaptation of center
        reference_point = obs[n].get_reference_point(in_global_frame=True)

        if not reference_point is None:
            if not obs[n].is_boundary or draw_wall_reference:
                ax.plot(reference_point[0],reference_point[1], 'k+', linewidth=18, markeredgewidth=4, markersize=13)
            if reference_point_number:
                ax.annotate('{}'.format(n), xy=reference_point+0.08, textcoords='data', size=16, weight="bold")  #
            # add group, too

        if drawVelArrow and np.linalg.norm(obs[n].xd)>0:
            # col=[0.5,0,0.9]
            col = [255/255., 51/255., 51/255.]
            fac=5 # scaling factor of velocity
            ax.arrow(obs[n].center_position[0], obs[n].center_position[1],
                         obs[n].xd[0]/fac, obs[n].xd[1]/fac,
                         # head_width=0.3, head_length=0.3, linewidth=10,
                         head_width=0.1, head_length=0.1, linewidth=3,
                         fc=col, ec=col, alpha=1)

    ax.set_aspect('equal', adjustable='box')

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    if noTicks:
        ax.tick_params(axis='both', which='major',bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    if showLabel:
        plt.xlabel(r'$\xi_1$', fontsize=16)
        plt.ylabel(r'$\xi_2$', fontsize=16)

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tick_params(axis='both', which='minor', labelsize=12)

    return 
    

def Simulation_vectorFields(x_range=[0,10], y_range=[0,10], point_grid=10, obs=[], sysDyn_init=False, xAttractor=None, saveFigure=False, figName='default', noTicks=True, showLabel=True, figureSize=(12.,9.5), obs_avoidance_func=obs_avoidance_interpolation_moving, attractingRegion=False, drawVelArrow=False, colorCode=False, streamColor=[0.05,0.05,0.7], obstacleColor=None, plotObstacle=True, plotStream=True, fig_and_ax_handle=None, alphaVal=1, dynamicalSystem=linearAttractor, draw_vectorField=True, points_init=[], show_obstacle_number=False, automatic_reference_point=True, nonlinear=True, show_streamplot=True, reference_point_number=False, normalize_vectors=True, tangent_eigenvalue_isometric=True, draw_wall_reference=False, gamma_distance=None, vector_field_only_outside=True, print_info=False, **kwargs):
    ''' 
    Draw obstacle and vectorfield. Several parameters and defaults 
    allow easy customization of plot.
    '''
    # TODO: gamma ditance does not fit as paramtere here (since not visual)...

    dim = 2

    # Numerical hull of ellipsoid 
    for n in range(len(obs)): 
        obs[n].draw_obstacle(numPoints=50) # 50 points resolution

    # Adjust dynamic center
    if automatic_reference_point:
        tt = time.time()
        obs.update_reference_points()
        dt = time.time() - tt

        if print_info:
            print("Time for dynamic_center: {}ms".format(np.round(dt*1000, 2)))

    # Numerical hull of ellipsoid 
    for n in range(len(obs)): 
        obs[n].draw_obstacle(numPoints=50) # 50 points resolution 

    if fig_and_ax_handle is None:
        fig, ax = plt.subplots(figsize=figureSize)
    else:
        fig, ax = fig_and_ax_handle

    if not plotObstacle:
        warnings.warn('x_label & ticks not implemented')
    
    plot_obstacles(ax, obs, x_range, y_range, xAttractor, obstacleColor, show_obstacle_number, reference_point_number, drawVelArrow, noTicks, showLabel, draw_wall_reference=draw_wall_reference, **kwargs)

    # Show certain streamlines
    if np.array(points_init).shape[0]:
        plot_streamlines(points_init, ax, obs, xAttractor)

    if not draw_vectorField or xAttractor is None:
        return fig, ax
        # return

    start_time = time.time()

    # Create meshrgrid of points
    if type(point_grid)==int:
        N_x = N_y = point_grid
        YY, XX = np.mgrid[y_range[0]:y_range[1]:N_y*1j, x_range[0]:x_range[1]:N_x*1j]
        
    else:
        N_x = N_y = 1
        XX, YY = np.array([[point_grid[0]]]), np.array([[point_grid[1]]])

    
    ########## DEBUGGING ONLY ##########
    # TODO: DEBUGGING Only for Development and testing
    # N_x = N_y = 1
    # XX = np.zeros((N_x, N_y))
    # YY = np.zeros((N_x, N_y))
    it_start = 0
    n_samples = 5

    pos1 = [4.38, -4.52]
    pos2 = [4.40, -4.52]
    # pos1 = [-5.0, 2.5]
    # pos2 = [5.0, 2.5]
    
    x_sample_range = [pos1[0], pos2[0]]
    y_sample_range = [pos1[1], pos2[1]]

    x_sample = np.linspace(x_sample_range[0], x_sample_range[1], n_samples)
    y_sample = np.linspace(y_sample_range[0], y_sample_range[1], n_samples)

    ii = 0
    for ii in range(n_samples):
        iy = (ii+it_start) % N_y
        ix = int((ii+it_start) / N_x)
        
        XX[ix, iy] = x_sample[ii]
        YY[ix, iy] = y_sample[ii]

    # import pdb; pdb.set_trace()
    ########## STOP REMOVE ###########

    # Forced to attracting Region
    if attractingRegion:      
        def obs_avoidance_temp(x, xd, obs):
            return obs_avoidance_func(x, xd, obs, xAttractor)
        
        obs_avoidance = obs_avoidance_temp
    else:
        obs_avoidance = obs_avoidance_func
        
    xd_init = np.zeros((2, N_x, N_y))
    xd_mod = np.zeros((2, N_x, N_y))

    if vector_field_only_outside:
        indOfNoCollision = obs_check_collision_2d(obs, XX, YY)
    else:
        indOfNoCollision = np.ones((N_x, N_y))

    for ix in range(N_x):
        for iy in range(N_y):
            if not indOfNoCollision[ix, iy]:
                continue
            pos = np.array([XX[ix, iy], YY[ix, iy]])

            #xd_init[:,ix,iy] = dynamicalSystem(pos, x0=xAttractor) # initial DS
            MAX_SPEED = 3.0
            xd_init[:, ix, iy] = linear_ds_max_vel(pos, attractor=xAttractor, vel_max=MAX_SPEED)

            # print('pos', pos)
            xd_mod[:, ix, iy] = obs_avoidance(pos, xd_init[:,ix,iy], obs,
            # gamma_distance=gamma_distance # DEBUGGING: remove
            )     

    dx1_noColl, dx2_noColl = np.squeeze(xd_mod[0,:,:]), np.squeeze(xd_mod[1,:,:])

    if sysDyn_init:
        fig_init, ax_init = plt.subplots(figsize=(5,2.5))
        res_init = ax_init.streamplot(XX, YY, xd_init[0,:,:], xd_init[1,:,:], color=[(0.3,0.3,0.3)])
        
        ax_init.plot(xAttractor[0], xAttractor[1], 'k*', zorder=5)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlim(x_range)
        plt.ylim(y_range)


    end_time = time.time()

    n_calculations = np.sum(indOfNoCollision)
    if print_info:
        print('Number of free points: {}'.format(n_calculations))
        print('Average time: {} ms'.format(np.round((end_time-start_time)/(n_calculations)*1000),5) )
        print('Modulation calulcation total: {} s'.format(np.round(end_time-start_time), 4))

    if plotStream:
        if colorCode:
            # velMag = np.linalg.norm(np.dstack((dx1_noColl, dx2_noColl)), axis=2 )/6*100
            strm = res_ifd = ax.streamplot(XX, YY,dx1_noColl, dx2_noColl, color=velMag, cmap='winter', norm=matplotlib.colors.Normalize(vmin=0, vmax=10.) )
            
        else:
            # Normalize
            if normalize_vectors:
                normVel = np.sqrt(dx1_noColl**2 + dx2_noColl**2)

                max_vel = 0.3
                ind_nonZero = normVel>0
                dx1_noColl[ind_nonZero] = dx1_noColl[ind_nonZero]/normVel[ind_nonZero]
                dx2_noColl[ind_nonZero] = dx2_noColl[ind_nonZero]/normVel[ind_nonZero]

            if show_streamplot:
                res_ifd = ax.streamplot(XX[0, :], YY[:, 0], dx1_noColl, dx2_noColl, color=streamColor, zorder=3)
                
            else:
                quiver_factor=1.0

                if not normalize_vectors:
                    normVel = np.sqrt(dx1_noColl**2 + dx2_noColl**2)
                    ind_nonZero = normVel>0
                # Only display non-collision quiver-arrows
                ind_flatten = ind_nonZero.flatten()
                XX = XX.flatten()[ind_flatten]
                YY = YY.flatten()[ind_flatten]
                dx1_noColl = dx1_noColl.flatten()[ind_flatten]
                dx2_noColl = dx2_noColl.flatten()[ind_flatten]

                res_ifd = ax.quiver(XX, YY,
                                    dx1_noColl*quiver_factor, dx2_noColl*quiver_factor,
                                    color=streamColor, zorder=3)

    plt.show()

    start_time = time.time()

    if saveFigure:
        # Save as png
        try:
            plt.savefig('figures/' + figName + '.png', bbox_inches='tight')
        except:
            plt.savefig('../figures/' + figName + '.png', bbox_inches='tight')
    return fig, ax
