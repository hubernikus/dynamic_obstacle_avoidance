'''
# Library for the Modulation of Linear Systems

@author Lukas Huber

Copyright (c) 2019 under GPU license. 

'''

import numpy as np
import numpy.linalg as LA

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *

import warnings
import copy 

import sys

def obs_avoidance_nonlinear_hirarchy(x, ds_init, obs, attractor=True, gamma_limit=1.0, weight_pow=2, repulsive_gammaMargin=0.01):
    # Gamma_limit [float] - defines the minimum gamma, where one function take the whole value
    # TODO -- speed up for multiple obstacles, not everything needs to be safed...

    # Initialize Variables
    N_obs = len(obs) #number of obstacles
    if N_obs ==0:
        return ds_init(x), x
    
    d = x.shape[0] #TODO remove
    dim = x.shape[0]
    Gamma = np.zeros((N_obs))
    
    max_hirarchy = 0
    hirarchy_array = np.zeros(N_obs)
    for oo in range(N_obs):
        hirarchy_array[oo] = obs[oo].hirarchy
        if obs[oo].hirarchy>max_hirarchy:
            max_hirarchy = obs[oo].hirarchy
    
    if type(attractor)==bool:
        if attractor:
            attractor = np.zeros((d,1))
            N_attr = 1
        else:
            attractor = np.zeros((d,0))
            N_attr = 0
    else:
        attractor = np.array(attractor)
        if len(attractor.shape)==1:
            attractor = np.array(([attractor])).T

        N_attr = attractor.shape[1]
    
    # Linear and angular roation of velocity
    xd_dx_obs = np.zeros((d,N_obs))
    xd_w_obs = np.zeros((d,N_obs)) #velocity due to the rotation of the obstacle

    R = np.zeros((d,d,N_obs))
    E = np.zeros((d,d,N_obs))
    D = np.zeros((d,d,N_obs))
    M = np.zeros((d,d,N_obs))
    E_orth = np.zeros((d,d,N_obs))
    
    reference_points = np.zeros((d, N_obs))

    x_t = np.zeros((d, N_obs))

    # Radial displacement position with the first element being the original one
    m_x = np.zeros((d, max_hirarchy+2)) 
    m_x[:,-1] = x

    # TODO - x_t only be computated later?
    for n in range(N_obs):
        # rotating the query point into the obstacle frame of reference
        if obs[n].th_r: # Greater than 0
            R[:,:,n] = compute_R(d,obs[n].th_r)
        else:
            R[:,:,n] = np.eye(d)
            
        reference_points[:,n] = np.array(obs[n].reference_point)

    Gamma_a = []
    for a in range(N_attr):
        # Eucledian distance -- other options possible
        Gamma_a = LA.norm(x-attractor[:,a])+1
        
    weights_hirarchy = np.zeros((N_obs+N_attr, max_hirarchy+1))
    Gammas_hirarchy = np.zeros((N_obs, max_hirarchy+1))
    radius_hirarchy = np.zeros((N_obs, max_hirarchy+1)) # TODO -- remove

    for hh in range(max_hirarchy, -1, -1): # backward loop
        
        ind_hirarchy = (hirarchy_array==hh)
        ind_hirarchy_low = (hirarchy_array<hh)

        Gamma = -np.ones((N_obs))
        
        # Loop to find new DS-evaluation point
        for o in np.arange(N_obs)[ind_hirarchy]:
            # rotating the query point into the obstacle frame of reference
            x_t[:,o] = R[:,:,o].T @ (m_x[:,hh+1]-obs[o].center_position)

            # TODO compute weight separately to avoid unnecessary computation
            E[:,:,o], D[:,:,o], Gamma[o], E_orth[:,:,o] = compute_modulation_matrix(x_t[:,o],obs[o], R[:,:,o])

        for o in np.arange(N_obs)[ind_hirarchy_low]:
            # TODO only evaluate GAMMA and not matrices
            x_t[:,o] = R[:,:,o].T @ (m_x[:,hh+1]-obs[o].center_position)
            E_, D_, Gamma[o],E_orth_  = compute_modulation_matrix(x_t[:,o],obs[o], R[:,:,o])

        Gammas_hirarchy[:,hh] = Gamma
        ind_lowEq = hirarchy_array<=hh
        
        weights_hirarchy[np.hstack((ind_lowEq, np.tile(True, N_attr) )), hh ] = compute_weights(np.hstack((Gamma[ind_lowEq],Gamma_a)), np.sum(hirarchy_array<=hh)+N_attr)
        weight = weights_hirarchy[:,hh]

        delta_x = np.zeros((d))
        for o in np.arange(N_obs)[ind_hirarchy]:
            if obs[o].is_boundary:
                continue

            distToRef = LA.norm((reference_points[:,o]-m_x[:,hh+1]))
            if distToRef > 0: 
                directionX = (reference_points[:,o]-m_x[:,hh+1])/distToRef
            else:
                directionX = np.zeros((d))
            
            dir_p2ref =  (m_x[:,hh+1] - obs[o].reference_point)
            rad_obs = get_radius(vec_point2ref=dir_p2ref, obs=obs[o]) # with reference point

            if obs[o].hirarchy>0:
                dir_p2ref_parent = (m_x[:,hh+1] - obs[obs[o].ind_parent].reference_point)
                rad_parent = get_radius(dir_p2ref_parent, obs=obs[obs[o].ind_parent])
                maximal_displacement = np.max([LA.norm(dir_p2ref_parent)-rad_parent, 0])
            else:
                maximal_displacement = LA.norm(dir_p2ref)

            if Gamma[o] <= 1: # Any point inside the obstacle is mapped to the center
                warnings.warn("Point inside obstacle")
                # print("WARNING -- Point inside obstacle")
                Gamma[o] = 1
                # delta_x = (reference_points[:,o]-m_x[:,hh+1])*weight[o]
                # break
                
            delta_r = np.min([rad_obs, maximal_displacement])
            p = 1 # hyperparameter
            delta_r = delta_r*(1/Gamma[o])**p
            delta_x = delta_x + weight[o]*delta_r*directionX

            
        m_x[:, hh] = m_x[:,hh+1]+delta_x # For each hirarchy level, there is one mean radial displacement

    xd = ds_init(m_x[:,0])

    # if Gamma<=1:
        # print('Gamma', Gamma)
        # print('weight', weight)
    
    #adding the influence of the rotational and cartesian velocity of the
    #obstacle to the velocity of the robot
    xd_obs = np.zeros((d))
    
    # Relative velocity
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
        xd_obs_n = np.exp(-1/obs[n].sigma*(max([Gammas_hirarchy[n,-1],1])-1))*  ( np.array(obs[n].xd) + xd_w )
        
        # Only consider velocity of the obstacle in direction
        xd_obs_n = LA.inv(E_orth[:,:,n]) @ xd_obs_n
        xd_obs_n[0] = np.max(xd_obs_n[0], 0)
        xd_obs_n = E_orth[:,:,n] @ xd_obs_n

        xd_obs = xd_obs + xd_obs_n*weight[n]

    # compute velocity of to the obstacle with respect to the obstacle frame of reference
    xd = xd-xd_obs

    xd_list = np.zeros((d, max_hirarchy+1))

    for hh in range(0, max_hirarchy+1, 1): # forward loop
        # weight_hirarchy = np.zeros((N_obs + N_attr))
        # ind_w = np.hstack((ind_hirarchy_low, np.ones(N_attr)))>0 # TODO convert to bool
        # weight_hirarchy[ind_w] = weight[ind_w] / LA.norm(weight[ind_w])
        ind_hirarchy = (hirarchy_array==hh)
        
        Gamma = Gammas_hirarchy[:,hh]
        weight = weights_hirarchy[:,hh]

        if not np.sum(weight[:N_obs][ind_hirarchy]): # everything far; no modulation
            xd_list[:, hh] = xd
            continue
        
        # if np.sum(weight>0)==1:
            # M[:,:,n] = (R[:,:,n] @ E[:,:,n] @ D[:,:,n] @ LA.pinv(E[:,:,n]) @ R[:,:,n].T)
            # xd_hat[:,n] = M[:,:,n] @ xd #velocity modulation
        
        # Create orthogonal matrix
        xd_norm = LA.norm(xd)
        if xd_norm:#nonzero
            xd_n = xd/xd_norm
        else:
            xd_n=xd

        xd_t = np.array([xd_n[1], -xd_n[0]])

        Ff = np.array([xd_n, xd_t])

        Rf = Ff.T # ?! True???

        M = np.zeros((d,d,N_obs))
        xd_hat = np.zeros((d, N_obs))
        xd_mags = np.zeros((N_obs))
        k_ds = np.zeros((d-1, N_obs))

        weight_ind = weight.astype(bool)
        
        for n in np.arange(N_obs)[ind_hirarchy]:
            if not weight_ind[n]:
                # print('zero weight', weight_ind[n])
                continue
            
            M[:,:,n] = (R[:,:,n] @ E[:,:,n] @ D[:,:,n] @ LA.pinv(E[:,:,n]) @ R[:,:,n].T)
            xd_hat[:,n] = M[:,:,n] @ xd #velocity modulation
            xd_mags[n] = np.sqrt(np.sum(xd_hat[:,n]**2))
            if xd_mags[n]: # Nonzero magnitude ---
                xd_hat_n = xd_hat[:,n]/xd_mags[n]
            else:
                xd_hat_n = xd_hat[:,n]

            if not d==2:
                warnings.warn('not implemented for d neq 2')

            Rfn = Rf @ xd_hat_n
            k_fn = Rfn[1:]
            kfn_norm = LA.norm(k_fn) # Normalize
            if kfn_norm:#nonzero
                k_fn = k_fn/ kfn_norm

            sumHat = np.sum(xd_hat_n*xd_n)
            if sumHat > 1 or sumHat < -1:
                sumHat = max(min(sumHat, 1), -1)
                warnings.warn(' cosinus out of bound!')
    
            k_ds[:,n] = np.arccos(sumHat)*k_fn.squeeze()

        xd_mags = np.sqrt(np.sum(xd_hat**2, axis=0) )

        if N_attr:
            # Enforce convergence in the region of the attractor
            #d_a = np.linalg.norm(x - np.array(attractor)) # Distance to attractor
            #w = compute_weights(np.hstack((Gamma, [d_a])), N_obs+N_attr)
            k_ds = np.hstack((k_ds, np.zeros((d-1, N_attr)) )) # points at the origin
            xd_mags = np.hstack((xd_mags, LA.norm((xd))*np.ones(N_attr) ))

        # Weighted interpolation
        weightPow = 1 # Hyperparameter for several obstacles !!!!
        weight_hirarchy = weight**weightPow
        if not LA.norm(weight,2):
            warnings.warn('trivial weight.')
            
        weight_hirarchy = weight/LA.norm(weight,2)

        xd_mag = np.sum(xd_mags*weight)
        k_d = np.sum(k_ds*np.tile(weight, (d-1, 1)), axis=1)

        norm_kd = LA.norm(k_d)

        # Reverse k_d
        if norm_kd: #nonzero
            n_xd = Rf.T @ np.hstack((np.cos(norm_kd), np.sin(norm_kd)/norm_kd*k_d ))
        else:
            n_xd = Rf.T @ np.hstack((1, k_d ))

        xd = xd_mag*n_xd.squeeze()
 
        xd_list[:, hh] = xd

        # xd = constVelocity_distance(xd, x, center_position=attractor[:,closestAttr], velConst = 10.0, distSlow=0.1)
    # xd = xd + xd_obs # transforming back the velocity into the global coordinate system
    if N_attr:
        for hh in range(max_hirarchy+1):
            xd = xd_list[:,hh]
            closestAttr = np.argmin(LA.norm(np.tile(x, (N_attr,1)).T - attractor, axis=0))
            xd_list[:,hh] = velConst_attr(x, xd, x0=attractor[:,closestAttr], velConst=2.0)
            # print('mag vel', LA.norm(xd))
            # print('mag vel', LA.norm(xd))

    xd_list = xd_list + np.tile(xd_obs, (max_hirarchy+1, 1)).T
    
    return xd_list, m_x


def obs_avoidance_rungeKutta(dt, x, obs, obs_avoidance=obs_avoidance_nonlinear_hirarchy, ds_init=linearAttractor, center_position=False, order=4):
    # Fourth order integration of obstacle avoidance differential equation
    # NOTE: The movement of the obstacle is considered as small, hence position and movement changed are not considered. This will be fixed in future iterations.
    dim = np.array(x).shape[0]
    if type(center_position)==bool:
        # TODO --- no default value
        center_position = np.zeros(dim)

    if order == 1:
        step_fraction = np.array[1]
        rk_fac = np.array[1]
    elif order==4:
        step_fraction = np.array([0, 0.5, 0.5, 1.0])
        rk_fac = np.array([1,2,2,1])/6
    else:
        print('WARNING: implement rk with order {}'.format(order))
        step_fraction = np.array([1])
        rk_fac = np.array([1])
        
    k = np.zeros((dim, len(step_fraction)+1))

    for ii in range(len(step_fraction)):
        # TODO remove after debugging
        xd, m_x = obs_avoidance(x + k[:,ii]*step_fraction[ii], ds_init, obs)
        
        xd = xd[:,-1]
        # xd = velConst_attr(x, xd, center_position)
        k[:,ii+1] = dt*xd

    x = x + np.sum(np.tile(rk_fac,(dim,1))*k[:,1:],axis=1) # + O(dt^5)
 
    return x
