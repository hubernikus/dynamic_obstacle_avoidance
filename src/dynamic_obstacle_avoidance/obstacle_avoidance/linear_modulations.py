'''
Library for the Modulation of Linear Systems
Copyright (c)2019 under GPU license
'''

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *

__author__ = "Lukas Huber"
__date__ =  "2019-11-29"
__info__ = "Obstacle avoidance for star-shaped obstacle in linear DS"

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import warnings
import sys


def obs_avoidance_interpolation_moving(position, xd, obs=[], attractor='none', weightPow=2, repulsive_gammaMargin=0.01, repulsive_obstacle=True, velocicity_max=None, evaluate_in_global_frame=True, zero_vel_inside=False, cut_off_gamma=1e6, x=None):
    '''
    This function modulates the dynamical system at position x and dynamics xd such that it avoids all obstacles obs. It can furthermore be forced to converge to the attractor. 
    
    INPUT
    x [dim]: position at which the modulation is happening
    xd [dim]: initial dynamical system at position x
    obs [list of obstacle_class]: a list of all obstacles and their properties, which present in the local environment
    attractor [list of [dim]]]: list of positions of all attractors
    weightPow [int]: hyperparameter which defines the evaluation of the weight
    
    OUTPUT
    xd [dim]: modulated dynamical system at position x
    '''

    # import pdb; pdb.set_trace()

    if not x is None:
        warnings.warn("Depreciated, don't use x as position argument.")
        position = x
    else:
        x = position

    # print('position', position)
    N_obs = len(obs) #number of obstacles
    if not N_obs: # No obstacle
        return xd

    dim = obs[0].dimension

    xd_norm = np.linalg.norm(xd)
    if xd_norm:
        xd_normalized = xd/xd_norm
    else:
        return xd # Trivial solution

    if type(attractor)==str:
        if attractor=='default': # Define attractor position
            attractor = np.zeros((d))
            N_attr = 1
        else:
            N_attr = 0            
    else:
        N_attr = 1

    if evaluate_in_global_frame:
        pos_relative = np.tile(position, (N_obs, 1)).T
    else:
        pos_relative = np.zeros((dim, N_obs))
        for n in range(N_obs):
            # Move to obstacle centered frame
            pos_relative[:, n] = obs[n].transform_global2relative(position) 

    Gamma = np.zeros((N_obs))
    for n in range(N_obs):
        Gamma[n] = obs[n].get_gamma(pos_relative[:, n], in_global_frame=evaluate_in_global_frame)

    if zero_vel_inside:
        if any(Gamma < 1):
            return np.zeros(dim)

    ind_obs = Gamma<cut_off_gamma
    if any(~ind_obs):
        return xd

    # pos_relative = pos_relative[:, ind_obs]
    # N_obs = np.sum(ind_obs)
    
    if N_attr:
        d_a = LA.norm(x - np.array(attractor)) # Distance to attractor
        weight = compute_weights(np.hstack((Gamma, [d_a])), N_obs+N_attr)
    else:
        weight = compute_weights(Gamma, N_obs)

    # Linear and angular roation of velocity
    # TODO: transform to global/relative frame!
    xd_dx_obs = np.zeros((dim, N_obs))
    xd_w_obs = np.zeros((dim, N_obs))  #velocity due to the rotation of the obstacle

    # Modulation matrices
    E = np.zeros((dim, dim, N_obs))
    D = np.zeros((dim, dim, N_obs))
    E_orth = np.zeros((dim, dim, N_obs))
    # M = np.zeros((dim, dim, N_obs))

    for n in np.arange(N_obs)[ind_obs]:
        # x_t = obs[n].transform_global2relative(x) # Move to obstacle centered frame
        D[:, :, n] = compute_diagonal_matrix(Gamma[n], dim, repulsion_coeff=obs[n].repulsion_coeff)
        # import pdb; pdb.set_trace()
        E[:, :, n], E_orth[:, :, n] = compute_decomposition_matrix(obs[n], pos_relative[:, n], in_global_frame=evaluate_in_global_frame)
            
        
    xd_obs = np.zeros((dim))
    
    for n in np.arange(N_obs)[ind_obs]:
        if dim==2:
            xd_w = np.cross(np.hstack(([0,0], obs[n].angular_velocity)),
                            np.hstack((x-np.array(obs[n].center_position),0)))
            xd_w = xd_w[0:2]
        elif dim==3:
            xd_w = np.cross(obs[n].orientation, x-obs[n].center_position)
        else:
            xd_w = np.zeros(dim)
            # raise ValueError('NOT implemented for d={}'.format(d))
            warnings.warn('Angular velocity is not defined for={}'.format(d))

        #The Exponential term is very helpful as it help to avoid the crazy rotation of the robot due to the rotation of the object
        exp_weight = np.exp(-1/obs[n].sigma*(np.max([Gamma[n],1])-1))
        xd_obs_n = exp_weight*(np.array(obs[n].linear_velocity) + xd_w)
        
        xd_obs = xd_obs + xd_obs_n*weight[n]

    xd = xd-xd_obs #computing the relative velocity with respect to the obstacle
    
    xd_hat = np.zeros((dim, N_obs))
    xd_hat_magnitude = np.zeros((N_obs))

    n = 0
    # plt.quiver(x[0], x[1], E_orth[0, 0, n], E_orth[1, 0, n])

    
    for n in np.arange(N_obs)[ind_obs]:
        if ((obs[n].is_boundary and E_orth[:, 0, n].T.dot(xd)>0)
            or (obs[n].repulsion_coeff>1 and E_orth[:, 0, n].T.dot(xd)>0)):
            # Only consider boundary when moving towards (normal direction)
            # OR if the object has positive repulsion-coefficient (only consider it at front)
            xd_hat[:, n] = xd
            
        else:
            # Matrix inversion cost between O(n^2.373) - O(n^3)
            if not evaluate_in_global_frame:
                xd_temp = obs[n].transform_global2relative_dir(xd)
            else:
                xd_temp = np.copy(xd)

            # Modulation with M = E @ D @ E^-1
            xd_trafo = LA.pinv(E[:,:,n]).dot(xd_temp)
            
            if not obs[n].tail_effect and xd_trafo[0]>0:
                xd_hat[:, n] = xd
            else:
                xd_hat[:, n] = E[:,:,n].dot(D[:,:,n]).dot(xd_trafo)
            
                if not evaluate_in_global_frame:
                    xd_hat[:, n] = obs[n].transform_relative2global_dir(xd_hat[:, n])
                
                    
                
        if repulsive_obstacle:
        # if False:
            # Move away from center in case of a collision
            if Gamma[n] < (1+repulsive_gammaMargin): 
                repulsive_power = 5
                repulsive_factor = 5
                repulsive_gamma = (1+repulsive_gammaMargin)
                
                repulsive_speed =  ((repulsive_gamma/Gamma[n])**repulsive_power-
                                       repulsive_gamma)*repulsive_factor
                if not obs[n].is_boundary:
                    repulsive_speed *= (-1)
                    
                # xd_hat[:,n] += R[:,:,n] .dot(E[:, 0, n]) * repulsive_velocity
                # x_t = R[:,:,n].T.dot(x-obs[n].center_position)
                pos_rel = obs[n].get_reference_direction(position, in_global_frame=True)

                # norm_xt = np.linalg.norm(pos_rel)
                
                # if (norm_xt): # nonzero
                repulsive_velocity = pos_rel*repulsive_speed
                # else:
                    # repulsive_velocity = np.zeros(dim)
                    # repulsive_velocity[0] = 1*repulsive_speed

                # if obs[n].is_boundary:
                    # repulsive_velocity = (-1)*repulsive_velocity
                    
                xd_hat[:, n] = repulsive_velocity

                # import pdb; pdb.set_trace()

        xd_hat_magnitude[n] = np.sqrt(np.sum(xd_hat[:,n]**2))

    xd_hat_normalized = np.zeros(xd_hat.shape)
    ind_nonzero = (xd_hat_magnitude>0)
    if np.sum(ind_nonzero):
        xd_hat_normalized[:, ind_nonzero] = xd_hat[:, ind_nonzero]/np.tile(xd_hat_magnitude[ind_nonzero], (dim, 1))

    if N_attr:
        # IMPLEMENT PROPERLY & TEST
        k_ds = np.hstack((k_ds, np.zeros((dim-1, N_attr)) )) # points at the origin
        xd_hat_magnitude = np.hstack((xd_hat_magnitude, LA.norm((xd))*np.ones(N_attr) ))
        
        total_weight = 1-weight_attr # Does this work
    else:
        total_weight = 1

    weighted_direction = get_directional_weighted_sum(null_direction=xd_normalized, directions=xd_hat_normalized, weights=weight, total_weight=total_weight)

    xd_magnitude = np.sum(xd_hat_magnitude*weight)
    vel_final = xd_magnitude*weighted_direction.squeeze()

    vel_final = vel_final + xd_obs
    if not velocicity_max is None:
        xd_norm = np.linalg.norm(vel_final)
        if xd_norm > velocicity_max:
            vel_final = vel_final/xd_norm * velocicity_max

    # import pdb; pdb.set_trace()

    # transforming back from object frame of reference to inertial frame of reference
    return vel_final


def obs_avoidance_rk4(dt, x, obs, obs_avoidance=obs_avoidance_interpolation_moving, ds=linearAttractor, x0=False):
    ''' Fourth order integration of obstacle avoidance differential equation '''
    # NOTE: The movement of the obstacle is considered as small, hence position and movement changed are not considered. This will be fixed in future iterations.
    # TODO: More General Implementation (find library)

    if type(x0)==bool:
        x0 = np.zeros(np.array(x).shape[0])

    # k1
    xd = ds(x, x0)
    xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x, xd, obs)
    k1 = dt*xd

    # k2
    xd = ds(x+0.5*k1, x0)
    xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x+0.5*k1, xd, obs)
    k2 = dt*xd

    # k3
    xd = ds(x+0.5*k2, x0)
    xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x+0.5*k2, xd, obs)
    
    k3 = dt*xd

    # k4
    xd = ds(x+k3, x0)
    xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x+k3, xd, obs)
    k4 = dt*xd

    # x final
    # Maybe: directional sum? Can this be done?
    x = x + 1./6*(k1+2*k2+2*k3+k4) # + O(dt^5)


    return x
