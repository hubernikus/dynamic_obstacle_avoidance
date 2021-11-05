"""
Library for the Modulation of Linear Systems
Obstacle avoidance for star-shaped obstacle in linear DS
"""
# Author: Lukas Huber
# Created: 2021-20-01
# License BSD (c) 2021

import warnings

import numpy as np
import matplotlib.pyplot as plt  # TODO: remove for production

from .modulation import compute_diagonal_matrix, compute_decomposition_matrix
from dynamic_obstacle_avoidance.utils import compute_weights


def obs_avoidance_potential_field(
    position,
    velocity,
    obs=[],
    xd=None,
    factor_repulsion=0.1,
    min_distance=0.001,
    constant_gain_repulsion=2.0,
    limit_distance_repulsion=2.0,
    virtual_mass_time_factor=1.0,
    evaluate_with_relative_minimum=True,
):
    """Potential field method.
    Based on: khatib1986real
    Not that the artificial potential field algorithm is acting in the force space."""

    # Trivial enrionment check
    if not len(obs):
        return obs

    if xd is not None:
        warnings.warn("Name xd is depreciated")
        velocity = xd

    obs_position = np.array([oo.center_position for oo in obs]).T
    dir_obstacles = np.tile(position, (obs_position.shape[1], 1)).T - obs_position

    # dist_to_center = np.linalg.norm(dir_obstacles, axis=0)

    # local_radius = np.array([oo.get_local_radius_ellipse(position, in_global_frame=True) for oo in obs]).T

    # dist_to_obstacles = dist_to_center - local_radius

    dist_to_obstacles = np.zeros(len(obs))
    for oo in range(len(obs)):
        gamma = obs[oo].get_gamma(position, in_global_frame=True)
        dist_to_ref = np.linalg.norm(obs[oo].global_reference_point - position)

        # if obs[oo].is_boundary:
        # gamma = 1.0/gamma
        # dist_to_obstacles[oo] = dist_to_ref*(1-gamma)
        # else:
        dist_to_obstacles[oo] = dist_to_ref * (gamma - 1)

        # import pdb; pdb.set_trace()

    # Cut-off at minimum distance
    if evaluate_with_relative_minimum:
        limit_distance_repulsion = (
            np.array([oo.get_maximal_distance() for oo in obs]).T
            * limit_distance_repulsion
        )
    repulsive_force = constant_gain_repulsion * (
        1.0 / dist_to_obstacles - 1.0 / limit_distance_repulsion
    )
    repulsive_force = np.maximum(repulsive_force, np.zeros(repulsive_force.shape))

    # Potential field acts on force. Since instantanious velocity of 2D, we choose 1.0 factor
    repulsive_velocity = repulsive_force * virtual_mass_time_factor
    repulsive_dir = np.array(
        [oo.get_normal_direction(position, in_global_frame=True) for oo in obs]
    ).T

    repulsive_velocity = (
        (-1)
        * repulsive_dir
        * np.tile((repulsive_velocity), (dir_obstacles.shape[0], 1))
    )

    for oo in range(len(obs)):
        if obs[oo].is_boundary:
            # pass
            repulsive_velocity[:, oo] = repulsive_velocity[:, oo] * (-1)

    repulsive_velocity = np.sum(repulsive_velocity, axis=1)

    # import pdb; pdb.set_trace()
    velocity = velocity + repulsive_velocity
    # velocity = repulsive_velocity

    # print('velocity', np.linalg.norm(velocity))
    return velocity


def obs_avoidance_orthogonal_moving(
    position,
    xd,
    obs=[],
    attractor="none",
    weightPow=2,
    repulsive_gammaMargin=0.01,
    repulsive_obstacle=False,
    velocicity_max=None,
    evaluate_in_global_frame=True,
    zero_vel_inside=False,
    cut_off_gamma=1e6,
    x=None,
    tangent_eigenvalue_isometric=True,
    gamma_distance=None,
):
    """
    This function modulates the dynamical system at position x and dynamics xd such that it avoids all obstacles obs. It can furthermore be forced to converge to the attractor.

    INPUT
    x [dim]: position at which the modulation is happening
    xd [dim]: initial dynamical system at position x
    obs [list of obstacle_class]: a list of all obstacles and their properties, which present in the local environment
    attractor [list of [dim]]]: list of positions of all attractors
    weightPow [int]: hyperparameter which defines the evaluation of the weight

    OUTPUT
    xd [dim]: modulated dynamical system at position x
    """

    if x is not None:
        warnings.warn("Depreciated, don't use x as position argument.")
        position = x
    else:
        x = position

    N_obs = len(obs)  # number of obstacles
    if not N_obs:  # No obstacle
        return xd

    dim = obs[0].dimension

    xd_norm = np.linalg.norm(xd)
    if xd_norm:
        xd_normalized = xd / xd_norm
    else:
        return xd  # Trivial solution

    if type(attractor) == str:
        if attractor == "default":  # Define attractor position
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
        Gamma[n] = obs[n].get_gamma(
            pos_relative[:, n], in_global_frame=evaluate_in_global_frame
        )

    ind_sort = np.argsort(Gamma)
    ind_sort = np.flip(ind_sort)
    Gamma = Gamma[ind_sort]
    pos_relative = pos_relative[:, ind_sort]
    # Create obstacle list
    obs_container = obs
    obs = []
    for oo in range(len(obs_container)):
        obs.append(obs_container[ind_sort[oo]])

    Gamma_proportional = np.zeros((N_obs))
    for n in range(N_obs):
        Gamma_proportional[n] = obs[n].get_gamma(
            pos_relative[:, n],
            in_global_frame=evaluate_in_global_frame,
            gamma_distance=gamma_distance,
        )

    # Gamma_proportional = np.copy(Gamma)

    # Worst case of being at the center
    if any(Gamma == 0):
        return np.zeros(dim)

    if zero_vel_inside and any(Gamma < 1):
        return np.zeros(dim)

    ind_obs = Gamma < cut_off_gamma
    if any(~ind_obs):
        return xd

    if N_attr:
        d_a = np.linalg.norm(x - np.array(attractor))  # Distance to attractor
        weight = compute_weights(np.hstack((Gamma_proportional, [d_a])), N_obs + N_attr)
    else:
        weight = compute_weights(Gamma_proportional, N_obs)

    # Modulation matrices
    D = np.zeros((dim, dim, N_obs))
    E_orth = np.zeros((dim, dim, N_obs))

    for n in np.arange(N_obs)[ind_obs]:
        D[:, :, n] = compute_diagonal_matrix(
            Gamma[n],
            dim,
            repulsion_coeff=obs[n].repulsion_coeff,
            tangent_eigenvalue_isometric=tangent_eigenvalue_isometric,
            rho=obs[n].reactivity,
        )

        temp, E_orth[:, :, n] = compute_decomposition_matrix(
            obs[n],
            pos_relative[:, n],
            in_global_frame=evaluate_in_global_frame,
        )

    # Linear and angular roation of velocity
    xd_obs = np.zeros((dim))

    for n in np.arange(N_obs)[ind_obs]:
        if dim == 2:
            xd_w = np.cross(
                np.hstack(([0, 0], obs[n].angular_velocity)),
                np.hstack((x - np.array(obs[n].center_position), 0)),
            )
            xd_w = xd_w[0:2]
        elif dim == 3:
            xd_w = np.cross(obs[n].orientation, x - obs[n].center_position)
        else:
            xd_w = np.zeros(dim)
            # raise ValueError('NOT implemented for d={}'.format(d))
            warnings.warn("Angular velocity is not defined for={}".format(d))

        weight_angular = np.exp(
            -1 / obs[n].sigma * (np.max([Gamma_proportional[n], 1]) - 1)
        )

        linear_velocity = obs[n].linear_velocity

        weight_linear = np.exp(
            -1 / obs[n].sigma * (np.max([Gamma_proportional[n], 1]) - 1)
        )

        xd_obs_n = weight_linear * linear_velocity + weight_angular * xd_w

        # The Exponential term is very helpful as it help to avoid the crazy rotation of the robot due to the rotation of the object

        xd_obs = xd_obs + xd_obs_n * weight[n]

    # Computing the relative velocity with respect to the obstacle
    xd = xd - xd_obs

    xd_hat = xd

    n = 0
    for n in np.arange(N_obs)[ind_obs]:
        if not evaluate_in_global_frame:
            xd_temp = obs[n].transform_global2relative_dir(xd_hat)
        else:
            xd_temp = np.copy(xd)

        # Modulation with M = E @ D @ E^-1
        xd_hat = np.linalg.pinv(E_orth[:, :, n]).dot(xd_temp)
        xd_hat = E_orth[:, :, n].dot(D[:, :, n]).dot(xd_hat)

        if not evaluate_in_global_frame:
            xd_hat = obs[n].transform_relative2global_dir(xd_hat)

    vel_final = xd_hat
    # Transforming back from object frame of reference to inertial frame of reference
    return vel_final
