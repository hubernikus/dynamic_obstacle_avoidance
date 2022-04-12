"""
# Library for the Modulation of Linear Systems

@author Lukas Huber
Copyright (c) 2019 under GPU license. 
"""

import warnings
import copy
import sys

import numpy as np
import numpy.linalg as LA

from dynamic_obstacle_avoidance.utils import *


def obs_avoidance_nonlinear_hirarchy(
    position_absolut,
    ds_init,
    obs,
    attractor=True,
    gamma_limit=1.0,
    weight_pow=2,
    repulsive_gammaMargin=0.01,
):
    # Gamma_limit [float] - defines the minimum gamma, where one function take the whole value
    # TODO -- speed up for multiple obstacles, not everything needs to be safed...

    # Initialize Variables
    N_obs = len(obs)  # number of obstacles
    if N_obs == 0:
        return ds_init(x), x

    d = position_absolut.shape[0]  # TODO remove
    dim = position_absolut.shape[0]
    Gamma = np.zeros((N_obs))

    max_hirarchy = 0
    hirarchy_array = np.zeros(N_obs)
    for oo in range(N_obs):
        hirarchy_array[oo] = obs[oo].hirarchy
        if obs[oo].hirarchy > max_hirarchy:
            max_hirarchy = obs[oo].hirarchy

    if type(attractor) == bool:
        if attractor:
            attractor = np.zeros((d, 1))
            N_attr = 1
        else:
            attractor = np.zeros((d, 0))
            N_attr = 0
    else:
        attractor = np.array(attractor)
        if len(attractor.shape) == 1:
            attractor = np.array(([attractor])).T

        N_attr = attractor.shape[1]

    # Linear and angular roation of velocity
    xd_dx_obs = np.zeros((d, N_obs))
    xd_w_obs = np.zeros((d, N_obs))  # velocity due to the rotation of the obstacle

    R = np.zeros((d, d, N_obs))
    E = np.zeros((d, d, N_obs))
    D = np.zeros((d, d, N_obs))
    M = np.zeros((d, d, N_obs))
    E_orth = np.zeros((d, d, N_obs))

    reference_points = np.zeros((d, N_obs))

    position_relative = np.zeros((d, N_obs))

    # Radial displacement position with the first element being the original one
    m_x = np.zeros((d, max_hirarchy + 2))
    m_x[:, -1] = position_absolut

    # TODO - position_relative only be computated later?
    for n in range(N_obs):
        # rotating the query point into the obstacle frame of reference
        if obs[n].th_r:  # Greater than 0
            R[:, :, n] = compute_R(d, obs[n].th_r)
        else:
            R[:, :, n] = np.eye(d)

        reference_points[:, n] = np.array(
            obs[n].get_reference_point(in_global_frame=True)
        )

    Gamma_a = []
    for a in range(N_attr):
        # Eucledian distance -- other options possible
        Gamma_a = LA.norm(position_absolut - attractor[:, a]) + 1

    weights_hirarchy = np.zeros((N_obs + N_attr, max_hirarchy + 1))
    Gammas_hirarchy = np.zeros((N_obs, max_hirarchy + 1))
    radius_hirarchy = np.zeros((N_obs, max_hirarchy + 1))  # TODO -- remove

    for hh in range(max_hirarchy, -1, -1):  # backward loop
        ind_hirarchy = hirarchy_array == hh
        ind_hirarchy_low = hirarchy_array < hh

        Gamma = -np.ones((N_obs))

        # Loop to find new DS-evaluation point
        for o in np.arange(N_obs)[ind_hirarchy]:
            # rotating the query point into the obstacle frame of reference
            position_relative[:, o] = R[:, :, o].T.dot(
                m_x[:, hh + 1] - obs[o].center_position
            )

            # TODO compute weight separately to avoid unnecessary computation
            (
                E[:, :, o],
                D[:, :, o],
                Gamma[o],
                E_orth[:, :, o],
            ) = compute_modulation_matrix(position_relative[:, o], obs[o], R[:, :, o])

        for o in np.arange(N_obs)[ind_hirarchy_low]:
            # TODO only evaluate GAMMA and not matrices
            position_relative[:, o] = R[:, :, o].T.dot(
                m_x[:, hh + 1] - obs[o].center_position
            )
            E_, D_, Gamma[o], E_orth_ = compute_modulation_matrix(
                position_relative[:, o], obs[o], R[:, :, o]
            )

        Gammas_hirarchy[:, hh] = Gamma
        ind_lowEq = hirarchy_array <= hh

        weights_hirarchy[
            np.hstack((ind_lowEq, np.tile(True, N_attr))), hh
        ] = compute_weights(
            np.hstack((Gamma[ind_lowEq], Gamma_a)),
            np.sum(hirarchy_array <= hh) + N_attr,
        )
        weight = weights_hirarchy[:, hh]

        delta_x = np.zeros((d))
        for o in np.arange(N_obs)[ind_hirarchy]:
            if obs[o].is_boundary:
                continue

            distToRef = LA.norm((reference_points[:, o] - m_x[:, hh + 1]))
            if distToRef > 0:
                directionX = (reference_points[:, o] - m_x[:, hh + 1]) / distToRef
            else:
                directionX = np.zeros((d))

            vec_point2ref = m_x[:, hh + 1] - reference_points[:, o]
            rad_obs = get_radius(
                vec_point2ref=vec_point2ref, obs=obs[o]
            )  # with reference point

            if obs[o].hirarchy > 0:
                vec_point2ref_parent = m_x[:, hh + 1] - obs[
                    obs[o].ind_parent
                ].get_reference_point(in_global_frame=True)
                rad_parent = get_radius(
                    vec_point2ref_parent, obs=obs[obs[o].ind_parent]
                )
                maximal_displacement = np.max(
                    [LA.norm(vec_point2ref_parent) - rad_parent, 0]
                )
            else:
                maximal_displacement = LA.norm(vec_point2ref)

            # import pdb; pdb.set_trace() ## DEBUG ##

            # Check intersection with other obstacles!
            # if ind_hirarchy>=1:
            # warnings.warn("Implement differently for interconnected obstacles.")

            weight_obs_product = 1
            # weights_displacement_limit = np.zeros(N_obs)
            angle2displacement = np.ones(N_obs) * pi
            radius_displacement = np.ones(N_obs) * (-1)

            for pp in np.arange(N_obs)[ind_hirarchy]:
                if pp == o:
                    continue
                if not isinstance(obs[pp], Ellipse):
                    raise TypeError(
                        "Not defined (yet) for obstacle not of type Ellipse."
                    )

                # cos_angle2reference = np.arccos(-vec_point2ref, obs[pp].reference_direction-obs[o].reference_direction)

                intersections = get_intersectionWithEllipse(
                    obs[o].reference_point, -vec_point2ref, obs[pp].axes
                )

                if not isinstance(intersections, type(None)):  # intersection exists
                    fac_direction = (intersections[:, 0] - obs[o].reference_point) / (
                        -vec_point2ref
                    )

                    # TODO test if equal and then remove
                    if not fac_direction[0] == fac_direction[1]:
                        warnings.warn("not equal. TODO - remove after debugging.")

                    if fac_direction[0] > 0:
                        angle2displacement[pp] = 0
                        radius_displacement[pp] = np.min(
                            LA.norm(
                                intersections
                                - np.tile(
                                    obs[o].get_reference_point(in_global_frame=True)
                                ).T,
                                axis=0,
                            )
                        )
                        continue

                tangents, tangent_points = get_tangents2ellipse(
                    obs[o].get_reference_point(in_global_frame=True),
                    obs[pp].axes,
                )
                # sin_tangentAndReference = np.zeros(2)
                # for ii in range(2):
                # sin_tangentAndReference[ii] = np.cross(tangents[:, ii], obs[o].reference_point)

                cos_tangentAndReference = np.cross(
                    tangents[:, ii],
                    obs[o].get_reference_direction(
                        m_x[:, hh + 1], in_global_frame=True
                    ),
                )
                angles = np.arccos(cos_tangentAndReference)

                if np.sum(np.abs(angles) < pi / 8):  # any true
                    min_ind = np.argmin(np.abs(angles))
                    angle2displacement[pp] = angles[min_ind]
                    radius_displacement[pp] = LA.norm(
                        obs[o].get_reference_point(in_global_frame=True)
                        - tangent_points[pp]
                    )
                else:
                    min_ind = np.argmin(np.abs(angles))

                weight_counter_obstacle = 1 - distToRef / radius_displacement[pp]
                slider_angle = max(0, 1 - angle2displacement[pp])
                weight_sliding = (
                    slider_angle * weight_counter_obstacle + (1 - slider_angle) * 1
                )

                weight_obs_product *= weight_sliding

            if Gamma[o] <= 1:  # Any point inside the obstacle is mapped to the center
                warnings.warn("Point inside obstacle")
                # print("WARNING -- Point inside obstacle")
                Gamma[o] = 1
                # delta_x = (reference_points[:,o]-m_x[:,hh+1])*weight[o]
                # break

            delta_r = np.min([rad_obs, maximal_displacement])
            power_factor = 1
            weight_delta_r = (1 / Gamma[o]) ** power_factor
            delta_r = delta_r * weight_delta_r
            delta_x = delta_x + weight[o] * weight_obs_product * delta_r * directionX

        m_x[:, hh] = (
            m_x[:, hh + 1] + delta_x
        )  # For each hirarchy level, there is one mean radial displacement

    xd = ds_init(m_x[:, 0])

    # if Gamma<=1:
    # print('Gamma', Gamma)
    # print('weight', weight)

    # adding the influence of the rotational and cartesian velocity of the
    # obstacle to the velocity of the robot
    xd_obs = np.zeros((d))

    # Relative velocity
    for n in range(N_obs):
        if d == 2:
            xd_w = np.cross(
                np.hstack(([0, 0], obs[n].w)),
                np.hstack((position_absolut - np.array(obs[n].center_position), 0)),
            )
            xd_w = xd_w[0:2]
        elif d == 3:
            xd_w = np.cross(obs[n].w, position_absolut - obs[n].center_position)
        else:
            warnings.warn("NOT implemented for d={}".format(d))

        # the exponential term is very helpful as it help to avoid the crazy rotation of the robot due to the rotation of the object
        xd_obs_n = np.exp(
            -1 / obs[n].sigma * (max([Gammas_hirarchy[n, -1], 1]) - 1)
        ) * (np.array(obs[n].xd) + xd_w)

        # Only consider velocity of the obstacle in direction
        xd_obs_n = LA.inv(E_orth[:, :, n]).dot(xd_obs_n)
        xd_obs_n[0] = np.max(xd_obs_n[0], 0)
        xd_obs_n = E_orth[:, :, n].dot(xd_obs_n)

        xd_obs = xd_obs + xd_obs_n * weight[n]

    # compute velocity of to the obstacle with respect to the obstacle frame of reference
    xd = xd - xd_obs

    xd_list = np.zeros((d, max_hirarchy + 1))

    for hh in range(0, max_hirarchy + 1, 1):  # forward loop
        # weight_hirarchy = np.zeros((N_obs + N_attr))
        # ind_w = np.hstack((ind_hirarchy_low, np.ones(N_attr)))>0 # TODO convert to bool
        # weight_hirarchy[ind_w] = weight[ind_w] / LA.norm(weight[ind_w])
        ind_hirarchy = hirarchy_array == hh

        Gamma = Gammas_hirarchy[:, hh]
        weight = weights_hirarchy[:, hh]

        if not np.sum(weight[:N_obs][ind_hirarchy]):  # everything far; no modulation
            xd_list[:, hh] = xd
            continue

        # if np.sum(weight>0)==1:
        # M[:,:,n] = (R[:,:,n] @ E[:,:,n] @ D[:,:,n] @ LA.pinv(E[:,:,n]) @ R[:,:,n].T)
        # xd_hat[:,n] = M[:,:,n] @ xd #velocity modulation

        # Create orthogonal matrix
        xd_norm = LA.norm(xd)
        if xd_norm:  # nonzero
            xd_n = xd / xd_norm
        else:
            xd_n = xd

        xd_t = np.array([xd_n[1], -xd_n[0]])

        Ff = np.array([xd_n, xd_t])

        Rf = Ff.T  # ?! True???

        M = np.zeros((d, d, N_obs))
        xd_hat = np.zeros((d, N_obs))
        xd_mags = np.zeros((N_obs))
        k_ds = np.zeros((d - 1, N_obs))

        weight_ind = weight.astype(bool)

        for n in np.arange(N_obs)[ind_hirarchy]:
            if not weight_ind[n]:
                # print('zero weight', weight_ind[n])
                continue

            M[:, :, n] = (
                R[:, :, n]
                .dot(E[:, :, n])
                .dot(D[:, :, n])
                .dot(LA.pinv(E[:, :, n]))
                .dot(R[:, :, n].T)
            )
            xd_hat[:, n] = M[:, :, n].dot(xd)  # velocity modulation
            xd_mags[n] = np.sqrt(np.sum(xd_hat[:, n] ** 2))
            if xd_mags[n]:  # Nonzero magnitude ---
                xd_hat_n = xd_hat[:, n] / xd_mags[n]
            else:
                xd_hat_n = xd_hat[:, n]

            if not d == 2:
                warnings.warn("not implemented for d neq 2")

            Rfn = Rf.dot(xd_hat_n)
            k_fn = Rfn[1:]
            kfn_norm = LA.norm(k_fn)  # Normalize
            if kfn_norm:  # nonzero
                k_fn = k_fn / kfn_norm

            sumHat = np.sum(xd_hat_n * xd_n)
            if sumHat > 1 or sumHat < -1:
                sumHat = max(min(sumHat, 1), -1)
                warnings.warn(" cosinus out of bound!")

            k_ds[:, n] = np.arccos(sumHat) * k_fn.squeeze()

        xd_mags = np.sqrt(np.sum(xd_hat**2, axis=0))

        if N_attr:
            # Enforce convergence in the region of the attractor
            # d_a = np.linalg.norm(x - np.array(attractor)) # Distance to attractor
            # w = compute_weights(np.hstack((Gamma, [d_a])), N_obs+N_attr)
            k_ds = np.hstack((k_ds, np.zeros((d - 1, N_attr))))  # points at the origin
            xd_mags = np.hstack((xd_mags, LA.norm((xd)) * np.ones(N_attr)))

        # Weighted interpolation
        weightPow = 1  # Hyperparameter for several obstacles !!!!
        weight_hirarchy = weight**weightPow
        if not LA.norm(weight, 2):
            warnings.warn("trivial weight.")

        weight_hirarchy = weight / LA.norm(weight, 2)

        xd_mag = np.sum(xd_mags * weight)
        k_d = np.sum(k_ds * np.tile(weight, (d - 1, 1)), axis=1)

        norm_kd = LA.norm(k_d)

        # Reverse k_d
        if norm_kd:  # nonzero
            n_xd = Rf.T.dot(
                np.hstack((np.cos(norm_kd), np.sin(norm_kd) / norm_kd * k_d))
            )
        else:
            n_xd = Rf.T.dot(np.hstack((1, k_d)))

        xd = xd_mag * n_xd.squeeze()

        xd_list[:, hh] = xd

        # xd = constVelocity_distance(xd, x, center_position=attractor[:,closestAttr], velConst = 10.0, distSlow=0.1)
    # xd = xd + xd_obs # transforming back the velocity into the global coordinate system
    if N_attr:
        for hh in range(max_hirarchy + 1):
            xd = xd_list[:, hh]
            closestAttr = np.argmin(
                LA.norm(
                    np.tile(position_absolut, (N_attr, 1)).T - attractor,
                    axis=0,
                )
            )
            xd_list[:, hh] = velConst_attr(
                position_absolut,
                xd,
                x0=attractor[:, closestAttr],
                velConst=2.0,
            )
            # print('mag vel', LA.norm(xd))
            # print('mag vel', LA.norm(xd))

    xd_list = xd_list + np.tile(xd_obs, (max_hirarchy + 1, 1)).T

    return xd_list, m_x
