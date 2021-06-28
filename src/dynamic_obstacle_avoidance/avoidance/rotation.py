"""
Library for the Rotation (Modulation Imitation) of Linear Systems
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import warnings
import numpy as np
import matplotlib.pyplot as plt   # For debugging only (!)

from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_directional_weighted_sum
from vartools.directional_space import get_angle_space, get_angle_space_inverse

# from vartools.directional_space import directional_convergence_summing
from vartools.dynamicalsys.closedform import evaluate_linear_dynamical_system

from dynamic_obstacle_avoidance.avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.avoidance.utils import get_relative_obstacle_velocity

# TODO: Speed up using cython / cpp e.g. eigen?
# TODO: list / array stack for lru_cache to speed

# TODO: speed up learning through cpp / c / cython(!?)
def get_intersection_with_circle(start_position, direction, radius):
    """Returns intersection with circle of of radius 'radius' and the line defined as
    'start_position + x * direction'
    (!) Only intersection at furthest distance to start_point is returned. """
    # Binomial Formula to solve for x in:
    # || dir_reference + x * (delta_dir_conv) || = pi/2
    AA = np.sum(direction**2) 
    BB = 2 * np.dot(direction, start_position)
    CC = np.sum(start_position**2) - radius**2
    DD = BB**2 - 4*AA*CC

    if DD < 0:
        # raise ValueError("Negative Determinant. No intersection.")
        # No intersection with circle
        return None

    # Only negative direction due to expected negative A (?!) [returns max-direciton]..
    fac_direction = (-BB + np.sqrt(DD)) / (2*AA)
    return fac_direction
    
    
def directional_convergence_summing(
    convergence_vector, reference_vector, weight, nonlinear_velocity=None,
    null_direction=None, null_matrix=None):
    """ Rotatiting / modulating a vector by using directional space.

    Paramters
    ---------
    convergence_vector: a array of floats of size (dimension,)
    reference_vector: a array of floats of size (dimension,)
    weight: float in the range [0, 1] which gives influence on how important vector 2 is.
    nonlinear_velocity: (optional) the vector-field which converges 

    Return
    ------
    converging_velocity: Weighted summing in direction-space to 'emulate' the modulation.
    """
    if null_matrix is None:
        null_matrix = get_orthogonal_basis(null_direction)
    
    dir_reference = get_angle_space(reference_vector, null_matrix=null_matrix)
    dir_convergence = get_angle_space(convergence_vector, null_matrix=null_matrix)

    # Find intersection a with radius of pi/2
    tangent_radius = np.pi/2.0

    if np.linalg.norm(dir_convergence) < tangent_radius:
        # Inside the tangent radius, i.e. vectorfield towards obstacle [no-tail-effect]

        # Do the math in the angle space
        delta_dir_conv = dir_convergence - dir_reference
        norm_dir_conv = np.linalg.norm(delta_dir_conv)
        if not norm_dir_conv: # Zero value
            if nonlinear_velocity is None:
                return convergence_vector
            else:
                return nonlinear_velocity

        fac_tang_conv = get_intersection_with_circle(
            start_position=dir_reference, direction=delta_dir_conv, radius=tangent_radius)

        dir_tangent = dir_reference + fac_tang_conv*delta_dir_conv
        norm_tangent_dist = np.linalg.norm(dir_tangent - dir_reference)

        # Weight to ensure that:
        # weight=1 => w_conv=1  AND norm_dir_conv=0 => w_conv=0
        weight_deviation = norm_dir_conv / norm_tangent_dist

        # TODO: allow 'tail-effect' after obstacle (is it useful?)
        # Allow finding of 'closest' 
        if weight_deviation <= 0:
            warnings.warn("Weight negative. This should be cautght by the (+square problem.")
            # print("Weight deviation", weight_deviation)
            dir_conv_rotated = dir_convergence

        elif weight_deviation > 1:
            warnings.warn("Weight negative. This should be excluded radius test.")
            # print("Weight deviation", weight_deviation)
            dir_conv_rotated = dir_convergence

        else:
            # If both negative, no change needed
            power_factor = 5.0
            w_conv = weight**(1./(weight_deviation*power_factor))

            dir_conv_rotated = w_conv*dir_tangent + (1-w_conv)*dir_convergence
    else:
        # Initial velocity 'dir_convergecne' already pointing away from obstacle
        dir_conv_rotated = dir_convergence
        

    if False:
    # if True:
        # TODO: Remove after DEBUG
        import matplotlib.pyplot as plt
        # plt.close('all')
        plt.figure()
        plt.plot([-pi, pi], [0, 0], '--', color='k')
        plt.plot(0, 0, 'k.', label='normal / tangent')
        plt.plot(-pi/2.0, 0, 'k.')
        plt.plot(+pi/2.0, 0, 'k.')
        
        plt.plot(dir_reference[0], 0, 'ro', label="ref")
        plt.plot(dir_convergence[0], 0, 'bo', label="converg")
        plt.plot(dir_conv_rotated[0], 0, 'go', label="rotated")
        plt.plot(dir_tangent[0], 0, 'o', color='magenta', label="tangent")
        plt.legend()
        plt.show()
        breakpoint()
        
    if nonlinear_velocity is None:
        return get_angle_space_inverse(dir_conv_rotated, null_matrix=null_matrix)
    
    else:
        # TODO: only do until pi/2, not full circle [i.e. circle cutting]
        # TODO: expand for more general nonlinear velocities
        dir_nonlinearvelocity = get_angle_space(nonlinear_velocity, null_matrix=null_matrix)

        if np.linalg.norm(dir_nonlinearvelocity - dir_convergence) > np.pi:
            # If it is larger than pi, we assume that it was rotated in the 'wrong direction'
            # i.e. find the same one, which is 2*pi in the other direction.
            norm_nonlinear = np.linalg.norm(dir_nonlinearvelocity)
            dirdir_nonlinear = dir_nonlinearvelocity / norm_nonlinear
            
            dir_nonlinearvelocity = (norm_nonlinear-2*pi) * dirdir_nonlinear

        dir_nonlinear_rotated = weight*dir_conv_rotated + (1-weight)*dir_nonlinearvelocity
        return get_angle_space_inverse(dir_nonlinear_rotated, null_matrix=null_matrix)
    

def get_weight_from_gamma(gamma_array, power_value=1.0):
    """ Returns weight-array based on input of gamma_array """
    return 1.0/np.abs(gamma_array)**power_value


def obstacle_avoidance_rotational(
    position, initial_velocity, obstacle_list, cut_off_gamma=1e6, gamma_distance=None):
    """ Obstacle avoidance based on 'local' rotation and the directional weighted mean.
    
    Parameters
    ----------
    position : array of the position at which the modulation is performed
        float-array of (dimension,)
    initial_velocity : Initial velocity which is modulated 
    obstacle_list :
    gamma_distance : factor for the evaluation of the proportional gamma
    
    Return
    ------
    modulated_velocity : array-like of shape (n_dimensions,)
    """
    n_obstacles = len(obstacle_list)
    if not n_obstacles:  # zero length
        return initial_velocity

    if hasattr(obstacle_list, 'update_relative_reference_point'):
        obstacle_list.update_relative_reference_point(position=position)

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

    # breakpoint()
    rotated_velocity = get_directional_weighted_sum(
        null_direction=initial_velocity,
        directions=rotated_velocities,
        weights=weights,
        )

    # Magnitude such that zero on the surface of an obstacle
    magnitude = np.dot(inv_gamma_weight, weights) * np.linalg.norm(initial_velocity)
    if False: # TODO: remove after DEBUGGING
        breakpoint()
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
