""" Library for the Rotation (Modulation Imitation) of Linear Systems
Copyright (c) 2021 under MIT license
"""
# author: Lukas Huber
# email: hubernikus@gmail.com

import warnings

import numpy as np
import matplotlib.pyplot as plt   # For debugging only (!)

from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_directional_weighted_sum
from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system

from dynamic_obstacle_avoidance.avoidance.utils import get_relative_obstacle_velocity
from dynamic_obstacle_avoidance.avoidance.utils import compute_weights

# TODO: Speed up using cython / cpp e.g. eigen?
# TODO: list / array stack for lru_cache to speed


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

    inv_gamma_weight = 1 - 1./gamma_array
    rotated_velocities = np.zeros((dimension, n_obs_close))
    for it, oo in zip(range(n_obs_close), np.arange(n_obstacles)[ind_obs]):
        # It is with respect to the close-obstacles -- oo ONLY to use in obstacle_list (whole)
        reference_dir = obstacle_list[oo].get_reference_direction(
            position, in_global_frame=True)
        
        if hasattr(obstacle_list, 'get_convergence_direction'):
            convergence_velocity = obstacle_list.get_convergence_direction(oo)
        elif get_convergence_direction is not None:
            convergence_velocity = get_convergence_direction(position=position)
        else:
            raise ValueError("No initial-convergence direction is defined")

        conv_vel_norm = np.linalg.norm(convergence_velocity)
        if not conv_vel_norm:
            rotated_velocities[:, it] = initial_velocity
            
        convergence_velocity = convergence_velocity / conv_vel_norm
            
        velocity_perp = (convergence_velocity
                         - np.dot(convergence_velocity, reference_dir) * reference_dir)

        # Project on tangent plane
        velocity_perp_proj = normal_orthogonal_matrix[:, :, it].T @ velocity_perp
        velocity_perp_proj[0] = 0
        velocity_perp_proj = normal_orthogonal_matrix[:, :, it] @ velocity_perp_proj

        # Investigate: can this be done for all obstacles together(?)
        rotated_velocities[:, it] = get_directional_weighted_sum(
            null_direction=convergence_velocity,
            directions=np.vstack((initial_velocity, velocity_perp_proj)).T,
            weights=np.array([inv_gamma_weight[it], 1-inv_gamma_weight[it]]),
            )

    rotated_velocity = get_directional_weighted_sum(
        null_direction=initial_velocity,
        directions=rotated_velocities,
        weights=weights,
        )

    # Magnitude such that zero on the surface of an obstacle
    magnitude = np.dot(inv_gamma_weight, weights) * np.linalg.norm(initial_velocity)
    rotated_velocity = rotated_velocity * magnitude
    
    rotated_velocity = rotated_velocity - relative_velocity
    # TODO: check maximal magnitude (in dynamic environments); i.e. see paper
    return rotated_velocity
