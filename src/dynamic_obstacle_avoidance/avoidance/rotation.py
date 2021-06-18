""" Library for the Rotation (Modulation Imitation) of Linear Systems
Copyright (c) 2021 under MIT license
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com

import warnings

import numpy as np
import matplotlib.pyplot as plt   # For debugging only (!)

from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_directional_weighted_sum
from vartools.directional_space import directional_convergence_summing
from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system

from dynamic_obstacle_avoidance.avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.avoidance.utils import get_relative_obstacle_velocity

# TODO: Speed up using cython / cpp e.g. eigen?
# TODO: list / array stack for lru_cache to speed

def get_weight_from_gamma(gamma_array, power_value=1.0):
    """ Returns weight-array based on input of gamma_array """
    return 1.0/np.abs(gamma_array)**power_value


def obstacle_avoidance_rotational(
    position, initial_velocity, obstacle_list, cut_off_gamma=1e6, gamma_distance=None,
    get_convergence_direction=None):
    """ Obstacle avoidance based on 'local' rotation and the directional weighted mean.
    
    Parameters
    ----------
    position : array of the position at which the modulation is performed
        float-array of (dimension,)
    initial_velocity : Initial velocity which is modulated 
    obstacle_list :
    gamma_distance : factor for the evaluation of the proportional gamma
    get_convergence_direction : function with which the direction of the convergence can
        be evaluated at position
    
    Return
    ------
    modulated_velocity : array-like of shape (n_dimensions,)
    """
    n_obstacles = len(obstacle_list)
    if not n_obstacles:  # zero length
        return initial_velocity

    dimension = position.shape[0]

    gamma_array = np.zeros((n_obstacles))
    for ii in range(n_obstacles):
        gamma_array[ii] = obstacle_list[ii].get_gamma(position, in_global_frame=True)
        if gamma_array[ii] <= 1 and not obstacle_list[ii].is_boundary:
            # Since boundaries are mutually subtracted,
            raise NotImplementedError()

    ind_obs = np.logical_and(gamma_array < cut_off_gamma, gamma_array > 1)
    
    if not any(ind_obs):
        return initial_velocity

    n_obs_close = np.sum(ind_obs)
    
    gamma_array = gamma_array[ind_obs]    # Only keep relevant ones
    gamma_proportional = np.zeros((n_obs_close))
    normal_orthogonal_matrix = np.zeros((dimension, dimension, n_obstacles))

    for it, it_obs in zip(range(n_obs_close), np.arange(n_obstacles)[ind_obs]):
        gamma_proportional[it] = obstacle_list[it_obs].get_gamma(
           position, in_global_frame=True, gamma_distance=gamma_distance,
        )
        normal_dir = obstacle_list[it_obs].get_normal_direction(
            position, in_global_frame=True)
        normal_orthogonal_matrix[:, :, it] = get_orthogonal_basis(normal_dir)
    
    weights = compute_weights(gamma_array)
    relative_velocity = get_relative_obstacle_velocity(
        position=position, obstacle_list=obstacle_list,
        ind_obstacles=ind_obs, gamma_list=gamma_proportional,
        E_orth=normal_orthogonal_matrix, weights=weights) 
    modulated_velocity = initial_velocity - relative_velocity

    inv_gamma_weight = get_weight_from_gamma(gamma_array)
    rotated_velocities = np.zeros((dimension, n_obs_close))
    for it, oo in zip(range(n_obs_close), np.arange(n_obstacles)[ind_obs]):
        # It is with respect to the close-obstacles -- oo ONLY to use in obstacle_list (whole)
        reference_dir = obstacle_list[oo].get_reference_direction(
            position, in_global_frame=True)

        if obstacle_list[oo].is_boundary:
            reference_dir = (-1)*reference_dir
            null_matrix = normal_orthogonal_matrix[:, :, it] * (-1)
        else:
            null_matrix = normal_orthogonal_matrix[:, :, it]
        
        if (hasattr(obstacle_list, 'get_convergence_direction')):
            convergence_velocity = obstacle_list.get_convergence_direction(position=position,
                                                                           it_obs=oo)
        elif get_convergence_direction is not None:
            convergence_velocity = get_convergence_direction(position=position)
        else:
            raise ValueError("No initial-convergence direction is defined")

        conv_vel_norm = np.linalg.norm(convergence_velocity)
        if conv_vel_norm:   # Zero value
            rotated_velocities[:, it] = initial_velocity

        rotated_velocities[:, it] = directional_convergence_summing(
            convergence_vector=convergence_velocity,
            reference_vector=reference_dir,
            weight=inv_gamma_weight[it],
            nonlinear_velocity=initial_velocity,
            null_matrix=null_matrix)

    rotated_velocity = get_directional_weighted_sum(
        null_direction=initial_velocity,
        directions=rotated_velocities,
        weights=weights,
        )

    # Magnitude such that zero on the surface of an obstacle
    magnitude = np.dot(inv_gamma_weight, weights) * np.linalg.norm(initial_velocity)
    if False: # TODO: remove after DEBUGGING 
        import matplotlib.pyplot as plt
        temp_init = initial_velocity / np.linalg.norm(initial_velocity)
        
        plt.quiver(position[0], position[1], temp_init[0], temp_init[1], label='initial', color='k')

        temp_vel = velocity_perp_proj / np.linalg.norm(velocity_perp_proj)
        plt.quiver(position[0], position[1], velocity_perp_proj[0],
                   velocity_perp_proj[1], label='projected', color='r')

        plt.quiver(position[0], position[1],
                   rotated_velocity[0], rotated_velocity[1], label='rotated', color='b')
        
        plt.quiver(position[0], position[1],
                   convergence_velocity[0], convergence_velocity[1], label='convergence', color='g')
        plt.legend(loc='right')
        plt.show()
        # breakpoint()
        
    rotated_velocity = rotated_velocity * magnitude
    
    rotated_velocity = rotated_velocity - relative_velocity
    # TODO: check maximal magnitude (in dynamic environments); i.e. see paper
    return rotated_velocity
