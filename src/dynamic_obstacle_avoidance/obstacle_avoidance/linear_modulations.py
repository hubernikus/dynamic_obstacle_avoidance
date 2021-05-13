#!/USSR/bin/python3
''' Library for the Modulation of Linear Systems
Copyright (c) 2019 under GPU license
'''

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *

__author__ = "Lukas Huber"
__date__ = "2019-11-29"
__info__ = "Obstacle avoidance for star-shaped obstacle in linear DS"

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import warnings
import sys

def obs_avoidance_interpolation_moving(position, initial_velocity, obs=[], attractor='none', weightPow=2, repulsive_gammaMargin=0.01, repulsive_obstacle=False, velocicity_max=None, evaluate_in_global_frame=True, zero_vel_inside=False, cut_off_gamma=1e6, x=None, tangent_eigenvalue_isometric=True, gamma_distance=None, xd=None):
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
    if x is not None:
        warnings.warn("Depreciated, don't use x as position argument.")
        position = x
    else:
        x = position

    if xd is not None:
        warnings.warn('xd is depriciated. Use <<initial_velocity>> instead.')
        initial_velocity = xd

    # number of obstacles
    N_obs = len(obs)       
    if not N_obs:
        # No obstacle
        return initial_velocity

    dim = obs[0].dimension

    initial_velocity_norm = np.linalg.norm(initial_velocity)
    if initial_velocity_norm:
        initial_velocity_normalized = initial_velocity/initial_velocity_norm
    else:
        return initial_velocity      # Trivial solution

    if type(attractor) == str:
        if attractor == 'default':       # Define attractor position
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

    # Two (Gamma) weighting functions lead to better behavior when agent &
    # obstacle size differs largely.
    Gamma = np.zeros((N_obs))
    for n in range(N_obs):
        Gamma[n] = obs[n].get_gamma(pos_relative[:, n], in_global_frame=evaluate_in_global_frame)

        if obs[n].is_boundary:
            pass
        # warnings.warn('Not... Artificially increasing boundary influence.')
        # Gamma[n] = pow(Gamma[n], 1.0/3.0)
        # else:
        # pass

    Gamma_proportional = np.zeros((N_obs))
    for n in range(N_obs):
        Gamma_proportional[n] = obs[n].get_gamma(pos_relative[:, n],
                                                 in_global_frame=evaluate_in_global_frame,
                                                 gamma_distance=gamma_distance,
        )

    # Worst case of being at the center
    if any(Gamma == 0):
        return np.zeros(dim)

    if zero_vel_inside and any(Gamma < 1):
        return np.zeros(dim)

    ind_obs = (Gamma < cut_off_gamma)
    if any(~ind_obs):
        # warnings.warn('Exceeding cut-off gamma. Stopping modulation.')
        return initial_velocity

    if N_attr:
        d_a = LA.norm(x - np.array(attractor))        # Distance to attractor
        weight = compute_weights(np.hstack((Gamma_proportional, [d_a])), N_obs+N_attr)
    else:
        weight = compute_weights(Gamma_proportional, N_obs)

    # Modulation matrices
    E = np.zeros((dim, dim, N_obs))
    D = np.zeros((dim, dim, N_obs))
    E_orth = np.zeros((dim, dim, N_obs))

    for n in np.arange(N_obs)[ind_obs]:
        # x_t = obs[n].transform_global2relative(x) # Move to obstacle centered frame
        D[:, :, n] = compute_diagonal_matrix(
            Gamma[n], dim, repulsion_coeff=obs[n].repulsion_coeff,
            tangent_eigenvalue_isometric=tangent_eigenvalue_isometric,
            rho=obs[n].reactivity,
        )

        E[:, :, n], E_orth[:, :, n] = compute_decomposition_matrix(obs[n], pos_relative[:, n], in_global_frame=evaluate_in_global_frame)

    # Linear and angular roation of velocity
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

        weight_angular = np.exp(-1/obs[n].sigma*(np.max([Gamma_proportional[n],1])-1))
        
        linear_velocity = obs[n].linear_velocity
        velocity_only_in_positive_normal_direction = True
        
        if velocity_only_in_positive_normal_direction:
            lin_vel_local = E_orth[:, :, n].T.dot(obs[n].linear_velocity)
            if lin_vel_local[0]<0 and not obs[n].is_boundary:
                # Obstacle is moving towards the agent
                linear_velocity = np.zeros(lin_vel_local.shape[0])
            else:
                linear_velocity = E_orth[:, 0, n].dot(lin_vel_local[0])

            weight_linear = np.exp(-1/obs[n].sigma*(np.max([Gamma_proportional[n],1])-1))
            # linear_velocity = weight_linear*linear_velocity

        xd_obs_n = weight_linear*linear_velocity + weight_angular*xd_w
        
        # The Exponential term is very helpful as it help to avoid
        # the crazy rotation of the robot due to the rotation of the object
        if obs[n].is_deforming:
            weight_deform = np.exp(-1/obs[n].sigma*(np.max([Gamma_proportional[n], 1])-1))
            vel_deformation = obs[n].get_deformation_velocity(pos_relative[:, n])

            if velocity_only_in_positive_normal_direction:
                vel_deformation_local = E_orth[:, :, n].T.dot(vel_deformation)
                if ((vel_deformation_local[0] > 0 and not obs[n].is_boundary)
                    or (vel_deformation_local[0] < 0 and obs[n].is_boundary)):
                    vel_deformation = np.zeros(vel_deformation.shape[0])
                    
                else:
                    vel_deformation = E_orth[:, 0, n].dot(vel_deformation_local[0])
                    
            xd_obs_n += weight_deform * vel_deformation
            
        xd_obs = xd_obs + xd_obs_n*weight[n]

    relative_velocity = initial_velocity - xd_obs      # Computing the relative velocity with respect to the obstacle
    
    relative_velocity_hat = np.zeros((dim, N_obs))
    relative_velocity_hat_magnitude = np.zeros((N_obs))

    n = 0
    for n in np.arange(N_obs)[ind_obs]:
        if ((obs[n].repulsion_coeff>1 and E_orth[:, 0, n].T.dot(relative_velocity)<0)
            # or(obs[n].is_boundary and E_orth[:, 0, n].T.dot(relative_velocity)>0) or 
            ):
            # Only consider boundary when moving towards (normal direction)
            # OR if the object has positive repulsion-coefficient (only consider it at front)
            relative_velocity_hat[:, n] = relative_velocity

        else:
            # Matrix inversion cost between O(n^2.373) - O(n^3)
            if not evaluate_in_global_frame:
                relative_velocity_temp = obs[n].transform_global2relative_dir(relative_velocity)
            else:
                relative_velocity_temp = np.copy(relative_velocity)

            # Modulation with M = E @ D @ E^-1
            relative_velocity_trafo = np.linalg.pinv(E[:, :, n]).dot(relative_velocity_temp)

            if obs[n].repulsion_coeff<0:
                # Negative Repulsion Coefficient at the back of an obstacle
                if E_orth[:, 0, n].T.dot(relative_velocity)<0:
                    # import pdb; pdb.set_trace()
                    # Adapt in reference direction
                    D[0, 0, n] = 2 - D[0, 0, n]

            # relative_velocity_trafo[0]>0
            elif (not obs[n].tail_effect and
                  ((relative_velocity_trafo[0] > 0 and not obs[n].is_boundary)
                   or (relative_velocity_trafo[0] < 0 and obs[n].is_boundary)
                  )):
                D[0, 0, n] = 1       # No effect in 'radial direction'

            relative_velocity_hat[:, n] = E[:, :, n].dot(D[:, :, n]).dot(relative_velocity_trafo)

            if not evaluate_in_global_frame:
                relative_velocity_hat[:, n] = obs[n].transform_relative2global_dir(relative_velocity_hat[:, n])
                # import pdb; pdb.set_trace()

        # TODO: review sticky surface feature [!]
        if False:
            # if obs[n].has_sticky_surface:
            relative_velocity_norm = np.linalg.norm(relative_velocity)
            
            if relative_velocity_norm:    # Nonzero
                # Normalize relative_velocity_hat
                mag = np.linalg.norm(relative_velocity_hat[:, n])
                if mag:    # nonzero
                    relative_velocity_hat[:, n] = relative_velocity_hat[:, n]/mag
                    
                # Limit maximum magnitude with respect to the tangent value
                sticky_surface_power = 2
                
                # Treat inside obstacle as on the surface
                Gamma_mag = max(Gamma_proportional[n], 1)
                if abs(Gamma_proportional[n]) < 1:
                    # if abs(Gamma_mag) < 1:
                    eigenvalue_magnitude = 0
                else:
                    eigenvalue_magnitude = 1 - 1./abs(Gamma_proportional[n])**sticky_surface_power
                    # eigenvalue_magnitude = 1 - 1./abs(Gamma_mag)**sticky_surface_power

                if not evaluate_in_global_frame:
                    relative_velocity_temp = obs[n].transform_global2relative_dir(relative_velocity_hat[:, n])
                else:
                    relative_velocity_temp = relative_velocity_hat[:, n]
                    
                tang_vel = np.abs(E_orth[:, :, n].T.dot(relative_velocity_temp)[0])
                
                eigenvalue_magnitude = min(eigenvalue_magnitude/tang_vel, 1) if tang_vel else 0

                relative_velocity_hat[:, n] = relative_velocity_hat[:, n]*relative_velocity_norm * eigenvalue_magnitude
                
                if not evaluate_in_global_frame:
                    relative_velocity_hat[:, n] = obs[n].transform_relative2global_dir(relative_velocity_hat[:, n])

                # import pdb; pdb.set_trace()

        if repulsive_obstacle:
            # Emergency move away from center in case of a collision
            # Not the cleanest solution...
            if Gamma[n] < (1+repulsive_gammaMargin): 
                repulsive_power = 5
                repulsive_factor = 5
                repulsive_gamma = (1+repulsive_gammaMargin)
                
                repulsive_speed =  ((repulsive_gamma/Gamma[n])**repulsive_power-
                                    repulsive_gamma)*repulsive_factor
                if not obs[n].is_boundary:
                    repulsive_speed *= (-1)
                    
                pos_rel = obs[n].get_reference_direction(position, in_global_frame=True)
                
                repulsive_velocity = pos_rel*repulsive_speed
                relative_velocity_hat[:, n] = repulsive_velocity

        relative_velocity_hat_magnitude[n] = np.sqrt(np.sum(relative_velocity_hat[:,n]**2))

    relative_velocity_hat_normalized = np.zeros(relative_velocity_hat.shape)
    ind_nonzero = (relative_velocity_hat_magnitude>0)
    if np.sum(ind_nonzero):
        relative_velocity_hat_normalized[:, ind_nonzero] = relative_velocity_hat[:, ind_nonzero]/np.tile(relative_velocity_hat_magnitude[ind_nonzero], (dim, 1))

    if N_attr:
        # TODO: implement properly & test
        # points at the origin
        k_ds = np.hstack((k_ds, np.zeros((dim-1, N_attr)) )) 
        relative_velocity_hat_magnitude = np.hstack((relative_velocity_hat_magnitude, LA.norm((relative_velocity))*np.ones(N_attr) ))
        
        total_weight = 1 - weight_attr 
    else:
        total_weight = 1

    weighted_direction = get_directional_weighted_sum(null_direction=initial_velocity_normalized, 
                                                      directions=relative_velocity_hat_normalized, weights=weight, total_weight=total_weight)

    relative_velocity_magnitude = np.sum(relative_velocity_hat_magnitude*weight)
    vel_final = relative_velocity_magnitude*weighted_direction.squeeze()

    vel_final = vel_final + xd_obs
    
    if velocicity_max is not None:
        # IMPLEMENT MAXIMUM VELOCITY / Not needed with sticky boundary
        velocity_norm = np.linalg.norm(vel_final)
        if velocity_norm > velocicity_max:
            vel_final = vel_final/velocity_norm * velocicity_max
            
    # Transforming back from object frame of reference to inertial frame of reference
    if False:
        print(f'{initial_velocity=}')
        print(f'{relative_velocity=}')
        E0 = E[:, :, 0]
        D0 = D[:, :, 0]
        print(f'{E0=}')
        print(f'{D0=}')
        print(f'{vel_final=}')
        
        plt.plot(obs[-1].center_position[0], obs[-1].center_position[1], 'ko')
        plt.plot(obs[-1].center_position[0], obs[-1].center_position[1], 'k+')
        
        plt.quiver(position[0], position[1], E0[0, 0], E0[1, 0], color='blue', label='Reference')
        plt.quiver(position[0], position[1], E0[0, 1], E0[1, 1], color='red', label='Tangent')
        
        plt.legend()
        plt.ion()
        plt.show()
        
        breakpoint()
        
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
