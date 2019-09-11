'''
# Library for the Modulation of Linear Systems

@author Lukas Huber

Copyright (c) 2019 under GPU license. 
'''

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *

import warnings

import sys


def obs_avoidance_interpolation_moving(x, xd, obs=[], attractor='none', weightPow=2, repulsive_gammaMargin=0.01):
    # This function modulates the dynamical system at position x and dynamics xd such that it avoids all obstacles obs. It can furthermore be forced to converge to the attractor. 
    # 
    # INPUT
    # x [dim]: position at which the modulation is happening
    # xd [dim]: initial dynamical system at position x
    # obs [list of obstacle_class]: a list of all obstacles and their properties, which present in the local environment
    # attractor [list of [dim]]]: list of positions of all attractors
    # weightPow [int]: hyperparameter which defines the evaluation of the weight
    #
    # OUTPUT
    # xd [dim]: modulated dynamical system at position x
    #

    # print('x', x)
    # print('xd', xd)
    # print('obs', obs)
    
    # Initialize Variables
    N_obs = len(obs) #number of obstacles
    if N_obs ==0:
        return xd
    
    d = x.shape[0]
    Gamma = np.zeros((N_obs))

    if type(attractor)==str:
        if attractor=='default': # Define attractor position
            attractor = np.zeros((d))
            N_attr = 1
        else:
            N_attr = 0            
    else:
        N_attr = 1                 

    # Linear and angular roation of velocity
    xd_dx_obs = np.zeros((d,N_obs))
    xd_w_obs = np.zeros((d,N_obs)) #velocity due to the rotation of the obstacle

    # Modulation matrices
    E = np.zeros((d,d,N_obs))
    D = np.zeros((d,d,N_obs))
    M = np.zeros((d,d,N_obs))
    E_orth = np.zeros((d,d,N_obs))

    # Rotation matrix
    R = np.zeros((d,d,N_obs))


    for n in range(N_obs):
        # Move the position into the obstacle frame of reference
        if obs[n].th_r: # Nonzero value
            R[:,:,n] = compute_R(d,obs[n].th_r)
        else:
            R[:,:,n] = np.eye(d)

        # Move to obstacle centered frame
        x_t = R[:,:,n].T @ (x-obs[n].center_position)
        E[:,:,n], D[:,:,n], Gamma[n], E_orth[:,:,n] = compute_modulation_matrix(x_t, obs[n], R[:,:,n])
        
    if N_attr:
        d_a = LA.norm(x - np.array(attractor)) # Distance to attractor
        weight = compute_weights(np.hstack((Gamma, [d_a])), N_obs+N_attr)

    else:
        weight = compute_weights(Gamma, N_obs)
    xd_obs = np.zeros((d))

    
    for n in range(N_obs):
        if d==2:
            xd_w = np.cross(np.hstack(([0,0], obs[n].w)),
                            np.hstack((x-np.array(obs[n].center_position),0)))
            xd_w = xd_w[0:2]
        elif d==3:
            xd_w = np.cross( obs[n].w, x-obs[n].center_position )
        else:
            warnings.warn('NOT implemented for d={}'.format(d))

        #the exponential term is very helpful as it help to avoid the crazy rotation of the robot due to the rotation of the object
        exp_weight = np.exp(-1/obs[n].sigma*(np.max([Gamma[n],1])-1))
        xd_obs_n = exp_weight*(np.array(obs[n].xd) + xd_w)

        # xd_obs_n = E_orth[:,:,n].T @ xd_obs_n
        # xd_obs_n[0] = np.max([xd_obs_n[0], 0]) # Onl use orthogonal part 
        # xd_obs_n = E_orth[:,:,n] @ xd_obs_n
        
        xd_obs = xd_obs + xd_obs_n*weight[n]

    xd = xd-xd_obs #computing the relative velocity with respect to the obstacle

    # Create orthogonal matrix
    xd_norm = LA.norm(xd)
    
    if xd_norm: # nonzero
        xd_normalized = xd/xd_norm
    else:
        xd_normalized=xd

    xd_t = np.array([xd_normalized[1], -xd_normalized[0]])

    Rf = np.array([xd_normalized, xd_t]).T

    
    xd_hat = np.zeros((d, N_obs))
    xd_hat_magnitude = np.zeros((N_obs))
    k_ds = np.zeros((d-1, N_obs))
        
    for n in range(N_obs):
        # xd_R = LA.pinv(E[:,:,n]) @ R[:,:,n].T @ xd
        
        M[:,:,n] = R[:,:,n] @ E[:,:,n] @ D[:,:,n] @ LA.pinv(E[:,:,n]) @ R[:,:,n].T
        
        xd_hat[:,n] = M[:,:,n] @ xd # velocity modulation
        
        # if False:
        if Gamma[n] < (1+repulsive_gammaMargin): # Safety for implementation (Remove for pure algorithm)
            repulsive_power = 5
            repulsive_factor = 5
            repulsive_gamma = (1+repulsive_gammaMargin)
            
            repulsive_velocity =  ((repulsive_gamma/Gamma[n])**repulsive_power-
                                    repulsive_gamma)*repulsive_factor 
            # print("\n\n Add repulsive vel: {} \n\n".format(repulsive_velocity))
            xd_hat[:,n] += R[:,:,n] @ E[:,0,n] * repulsive_velocity

        xd_hat_magnitude[n] = np.sqrt(np.sum(xd_hat[:,n]**2)) 
        if xd_hat_magnitude[n]: # Nonzero hat_magnitude
            xd_hat_normalized = xd_hat[:,n]/xd_hat_magnitude[n] # normalized direction
        else:
            xd_hat_normalized = xd_hat[:,n]
        
        if not d==2:
            warnings.warn('not implemented for d neq 2')

        xd_hat_normalized_velocityFrame = Rf @ xd_hat_normalized

        # Kappa space - directional space
        k_fn = xd_hat_normalized_velocityFrame[1:]
        kfn_norm = LA.norm(k_fn) # Normalize
        if kfn_norm:# nonzero
            k_fn = k_fn/ kfn_norm
            
        sumHat = np.sum(xd_hat_normalized*xd_normalized)
        if sumHat > 1 or sumHat < -1:
            sumHat = max(min(sumHat, 1), -1)
            warnings.warn('cosinus out of bound!')
            
        # !!!! ??? is this the same as 
        # np.arccos(sumHat) == np.arccos(xd_hat_normalized_velocityFrame[0])
        
        k_ds[:,n] = np.arccos(sumHat)*k_fn.squeeze()

    # xd_hat_magnitude = np.sqrt(np.sum(xd_hat**2, axis=0) ) # TODO - remove as already caclulated
    if N_attr: #nonzero
        k_ds = np.hstack((k_ds, np.zeros((d-1, N_attr)) )) # points at the origin
        xd_hat_magnitude = np.hstack((xd_hat_magnitude, LA.norm((xd))*np.ones(N_attr) ))
        
    # Weighted interpolation for several obstacles
    # weight = weight**weightPow
    # if not LA.norm(weight,2):
        # warnings.warn('trivial weight.')
    # weight = weight/LA.norm(weight,2)
    
    xd_magnitude = np.sum(xd_hat_magnitude*weight)
    k_d = np.sum(k_ds*np.tile(weight, (d-1, 1)), axis=1)

    # print('k_d', k_d)
    norm_kd = LA.norm(k_d)
    
    if norm_kd: # Nonzero
        n_xd = Rf.T @ np.hstack((np.cos(norm_kd), np.sin(norm_kd)/norm_kd*k_d ))
    else:
        n_xd = Rf.T @ np.hstack((1, k_d ))

    xd = xd_magnitude*n_xd.squeeze()

    # transforming back from object frame of reference to inertial frame of reference
    xd = xd + xd_obs

    return xd


def obs_avoidance_rk4(dt, x, obs, obs_avoidance=obs_avoidance_interpolation_moving, ds=linearAttractor, x0=False):
    # Fourth order integration of obstacle avoidance differential equation
    # NOTE: The movement of the obstacle is considered as small, hence position and movement changed are not considered. This will be fixed in future iterations.

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
    x = x + 1./6*(k1+2*k2+2*k3+k4) # + O(dt^5)

    return x
