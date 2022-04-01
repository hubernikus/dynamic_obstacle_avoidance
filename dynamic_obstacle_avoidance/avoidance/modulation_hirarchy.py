"""
Library for the Modulation of Linear Systems

@author Lukas Huber
@date 2019-11-29
@info Obstacle avoidance for star-shaped obstacle in linear DS

Copyright (c)2019 under GPU license
"""

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *

import warnings
import sys


def obs_avoidance_interpolation_moving(
    x,
    xd,
    obs=[],
    attractor="none",
    weightPow=2,
    repulsive_gammaMargin=0.01,
    repulsive_obstacle=True,
    velocicity_max=None,
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

    N_obs = len(obs)  # number of obstacles
    if not N_obs:  # No obstacle
        return xd

    dim = obs.dimension

    # Create variables
    Gamma = np.zeros((N_obs))

    xd_norm = np.linalg.norm(xd)
    if xd_norm == 0:  # Trivial solution
        return xd

    xd_normalized = xd / xd_norm

    if type(attractor) == str:
        if attractor == "default":  # Define attractor position
            attractor = np.zeros((d))
            N_attr = 1
        else:
            N_attr = 0
    else:
        N_attr = 1

    # Linear and angular roation of velocity
    xd_dx_obs = np.zeros((dim, N_obs))
    xd_w_obs = np.zeros((dim, N_obs))  # velocity due to the rotation of the obstacle

    # Modulation matrices
    E = np.zeros((dim, dim, N_obs))
    D = np.zeros((dim, dim, N_obs))
    M = np.zeros((dim, dim, N_obs))
    E_orth = np.zeros((dim, dim, N_obs))

    for n in range(N_obs):
        x_t = obs[n].transform_global2relative(x)  # Move to obstacle centered frame
        (
            E[:, :, n],
            D[:, :, n],
            Gamma[n],
            E_orth[:, :, n],
        ) = compute_modulation_matrix(x_t, obs[n])

    if N_attr:
        d_a = LA.norm(x - np.array(attractor))  # Distance to attractor
        weight = compute_weights(np.hstack((Gamma, [d_a])), N_obs + N_attr)
    else:
        weight = compute_weights(Gamma, N_obs)

    # if np.sum(weight)==0:
    # return xd

    xd_obs = np.zeros((dim))

    for n in range(N_obs):
        if dim == 2:
            xd_w = np.cross(
                np.hstack(([0, 0], obs[n].w)),
                np.hstack((x - np.array(obs[n].center_position), 0)),
            )
            xd_w = xd_w[0:2]
        elif d == 3:
            xd_w = np.cross(obs[n].w, x - obs[n].center_position)
        else:
            raise ValueError("NOT implemented for d={}".format(d))

        # The Exponential term is very helpful as it help to avoid the crazy rotation of the robot due to the rotation of the object
        exp_weight = np.exp(-1 / obs[n].sigma * (np.max([Gamma[n], 1]) - 1))
        xd_obs_n = exp_weight * (np.array(obs[n].xd) + xd_w)

        xd_obs = xd_obs + xd_obs_n * weight[n]
    xd = xd - xd_obs  # computing the relative velocity with respect to the obstacle

    # Create orthogonal matrix
    xd_norm = LA.norm(xd)

    xd_normalized = xd / xd_norm if xd_norm else xd  # check zero division

    xd_hat = np.zeros((dim, N_obs))
    xd_hat_magnitude = np.zeros((N_obs))

    for n in range(N_obs):
        if obs[n].is_boundary and E_orth[:, 0, n].T.dot(xd) < 0:
            # Only consider boundary when moving towards (normal direction)
            xd_hat[:, n] = xd
        else:
            # Matrix inversion cost between O(n^2.373) - O(n^3)
            # M[:,:,n] = dot(E[:,:,n]).dot(D[:,:,n]).dot(LA.pinv(E[:,:,n]))
            M[:, :, n] = E[:, :, n].dot(D[:, :, n]).dot(LA.pinv(E[:, :, n]))

            xd = obs[n].transform_global2relative_dir(xd)
            xd = M[:, :, n].dot(xd)
            xd_hat[:, n] = obs[n].transform_relative2global_dir(xd)
            # xd_hat[:,n] = M[:,:,n].dot(xd) # velocity modulation

        if repulsive_obstacle:
            if Gamma[n] < (
                1 + repulsive_gammaMargin
            ):  # Safety for implementation (Remove for pure algorithm)
                repulsive_power = 5
                repulsive_factor = 5
                repulsive_gamma = 1 + repulsive_gammaMargin

                repulsive_speed = (
                    (repulsive_gamma / Gamma[n]) ** repulsive_power - repulsive_gamma
                ) * repulsive_factor
                if obs[n].is_boundary:
                    repulsive_speed *= -1
                # xd_hat[:,n] += R[:,:,n] .dot(E[:, 0, n]) * repulsive_velocity
                # x_t = R[:,:,n].T.dot(x-obs[n].center_position)
                norm_xt = np.linalg.norm(x_t)
                if norm_xt:  # nonzero
                    repulsive_velocity = x_t / norm_xt * repulsive_speed
                else:
                    repulsive_velocity = np.zeros(3.0)
                    repulsive_velocity[0] = 1 * repulsive_speed

                xd_hat[:, n] = repulsive_velocity

        xd_hat_magnitude[n] = np.sqrt(np.sum(xd_hat[:, n] ** 2))

    xd_hat_normalized = np.zeros(xd_hat.shape)
    ind_nonzero = xd_hat_magnitude > 0
    if np.sum(ind_nonzero):
        xd_hat_normalized[:, ind_nonzero] = xd_hat[:, ind_nonzero] / np.tile(
            xd_hat_magnitude[ind_nonzero], (dim, 1)
        )

    if N_attr:
        # IMPLEMENT PROPERLY & TEST
        k_ds = np.hstack((k_ds, np.zeros((d - 1, N_attr))))  # points at the origin
        xd_hat_magnitude = np.hstack(
            (xd_hat_magnitude, LA.norm((xd)) * np.ones(N_attr))
        )

        total_weight = 1 - weight_attr  # Does this work
    else:
        total_weight = 1

    weighted_direction = get_directional_weighted_sum(
        reference_direction=xd_normalized,
        directions=xd_hat_normalized,
        weights=weight,
        total_weight=total_weight,
    )

    xd_magnitude = np.sum(xd_hat_magnitude * weight)
    xd = xd_magnitude * weighted_direction.squeeze()

    # if not velocicity_max is None:
    # xd = constVelocity(xd, position, position_attractor)
    # transforming back from object frame of reference to inertial frame of reference
    xd = xd + xd_obs

    return xd


def obs_avoidance_rk4(
    dt,
    x,
    obs,
    obs_avoidance=obs_avoidance_interpolation_moving,
    ds=linearAttractor,
    x0=False,
):
    # Fourth order integration of obstacle avoidance differential equation
    # NOTE: The movement of the obstacle is considered as small, hence position and movement changed are not considered. This will be fixed in future iterations.
    # TODO: More General Implementation (find library)

    if type(x0) == bool:
        x0 = np.zeros(np.array(x).shape[0])

    # k1
    xd = ds(x, x0)
    xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x, xd, obs)
    k1 = dt * xd

    # k2
    xd = ds(x + 0.5 * k1, x0)
    xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x + 0.5 * k1, xd, obs)
    k2 = dt * xd

    # k3
    xd = ds(x + 0.5 * k2, x0)
    xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x + 0.5 * k2, xd, obs)

    k3 = dt * xd

    # k4
    xd = ds(x + k3, x0)
    xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x + k3, xd, obs)
    k4 = dt * xd

    # x final
    # Maybe: directional sum? Can this be done?
    x = x + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # + O(dt^5)

    return x
