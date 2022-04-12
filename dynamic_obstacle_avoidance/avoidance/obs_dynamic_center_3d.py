import numpy as np

from math import pi, floor

import matplotlib.pyplot as plt  # only for debugging
import warnings

# from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import compute_weights

# TODO: Rewrite with faster & more advanced library //


def get_gamma_weight(Gammas, gamma_threshold, Gamma_min=1, pow_gamma=3):
    """Sum of weight gives importance of this object
    Each individual weight gives importance of vector"""

    ind_nonzero = Gammas < gamma_threshold

    point_weights = np.zeros(Gammas.shape)
    object_weight = 0

    if any(ind_nonzero):
        ind_bellowmin = Gammas < 1
        if any(ind_bellowmin):
            point_weights[ind_bellowmin] = 1.0 / np.sum(ind_bellowmin)
            object_weight = 1
        else:
            point_weights[ind_nonzero] = 1 / (Gammas[ind_nonzero] - Gamma_min) - 1 / (
                gamma_threshold - Gamma_min
            )
            point_weights = point_weights / np.sum(point_weights)  # normalize n_1
            object_weight = (1 / np.min(Gammas)) ** pow_gamma  # in [0, 1]

    return point_weights, object_weight


def get_object_weight(weights, weight_max=1, weight_pow=1):
    new_weights = np.zeros(weights.shape)

    ind_maxweights = weights == weight_max
    if any(ind_maxweights):
        new_weights[ind_maxweights] = 1.0 / np.sum(ind_maxweights)
        return new_weights

    new_weights = 1 / (weight_max - weights) - 1
    new_weights = new_weights**weight_pow
    new_weights = new_weights / np.sum(new_weights)
    return new_weights


def dynamic_center_3d(*args, **kwargs):
    # TODO: depreciated, remove
    return get_dynamic_center_obstacles(*args, **kwargs)


def get_dynamic_center_obstacles(
    obs,
    intersection_lists,
    gamma_threshold=3,
    gamma_matr=None,
    default_kernel_point_weight=0.5,
):
    # def dynamic_center_3d
    n_obs = len(obs)
    if n_obs == 1:
        return

    intersection_list = []
    for ii in range(len(intersection_lists)):
        intersection_list += intersection_lists[ii]
    intersection_list = set(intersection_list)
    # print('int list', intersection_list)

    dimension = obs[0].center_position.shape[0]

    # reference_point_list =
    Gamma_doubleList = [[None for ii in range(n_obs)] for ii in range(n_obs)]

    for it1 in range(n_obs):
        if it1 in intersection_list or obs[it1].is_boundary:
            continue
        # x_obs1 = np.unique(obs[it1].x_obs, axis=0)

        x_obs1 = np.zeros((0, 0, 0))

        object_weights = np.zeros(n_obs)
        kernel_points = np.zeros((dimension, n_obs))

        # boundary_points = obs[it1].boundary_points_margin_global

        for it2 in range(n_obs):
            if it2 == it1:
                # for it2 in range(it1+1, n_obs):
                # if it2 in intersection_obs:
                continue
            if Gamma_doubleList[it1][it2] is None:
                Gamma_doubleList[it1][it2] = obs[it2].get_gamma(
                    obs[it1].boundary_points_margin_global,
                    in_global_frame=True,
                )
            if Gamma_doubleList[it2][it1] is None:
                Gamma_doubleList[it2][it1] = obs[it1].get_gamma(
                    obs[it2].boundary_points_margin_global,
                    in_global_frame=True,
                )

            Gamma_list = np.hstack(
                (Gamma_doubleList[it1][it2], Gamma_doubleList[it2][it1])
            )

            if any(Gamma_list < gamma_threshold):
                point_weights, object_weights[it2] = get_gamma_weight(
                    Gamma_list, gamma_threshold=gamma_threshold
                )

                if object_weights[it2]:  # nonzero
                    boundary_list = np.hstack(
                        (
                            obs[it1].boundary_points_margin_global,
                            obs[it1].transform_relative2global(
                                obs[it1].transform_global2relative(
                                    obs[it2].boundary_points_margin_global
                                )
                                / (
                                    np.tile(
                                        Gamma_doubleList[it2][it1] ** 2,
                                        (dimension, 1),
                                    )
                                )
                            ),
                        )
                    )

                    kernel_points[:, it2] = np.sum(
                        boundary_list * np.tile(point_weights, (dimension, 1)),
                        axis=1,
                    )

        if any(object_weights):
            # Put weight to initial kernel_point
            kernel_points[:, it1] = obs[it1].global_reference_point
            object_weights[it1] = default_kernel_point_weight

            weights = get_object_weight(object_weights)
            # print('reset for it', it1)

            obs[it1].set_reference_point(
                np.sum(kernel_points * np.tile(weights, (dimension, 1)), axis=1),
                in_global_frame=True,
            )


def dynamic_center_3d_old(
    obs,
    intersection_obs,
    marg_dynCenter=1.3,
    N_distStep=3,
    resol_max=1000,
    N_resol=16,
    numbFactor_closest=2,
):

    N_obs = len(obs)
    if N_obs < 2:
        return  # no intersction possible

    # Convert to single list
    intersection_temp = []
    for ii_list in intersection_obs:
        for jj in ii_list:
            if jj not in intersection_temp:
                intersection_temp.append(jj)
    intersection_obs = intersection_temp

    for ii in range(N_obs):  # Default value for dynamic center
        if ii in intersection_obs:
            continue
        else:
            obs[ii].center_dyn = np.copy(obs[ii].x0)

    if N_obs <= 1 or np.array(intersection_obs).shape[0] == N_obs:
        return []

    # Resolution of outside plot
    # MAYBE - Change to user fixed size -- replot first oneh
    x_obs_sf = []
    for ii in range(N_obs):
        x_obs_sf.append(np.array(obs[ii].x_obs_sf))

    N_closest = (
        obs[0].d * numbFactor_closest
    )  # Number of close points which are considered for interpolation

    resolution = 30
    # resolution = len(obs[0].x_obs) # ... or noot?

    # Calculate distance between obstacles
    weight_obs_temp = np.zeros((N_obs, N_obs))
    x_cyn_temp = np.zeros((obs[0].d, N_obs, N_obs))

    rotMatrices = []
    for ii in range(N_obs):
        rotMatrices.append(np.array(obs[ii].rotMatrix))

    # Iterate over obstacles
    for it1 in range(N_obs):
        if it1 in intersection_obs:
            continue
        # x_obs1 = np.unique(obs[it1].x_obs, axis=0)

        x_obs1 = np.zeros((0, 0, 0))

        for it2 in range(it1 + 1, N_obs):
            if it2 in intersection_obs:
                continue
            if not x_obs1.shape[1] == obs[it2].boundary_points_local.shape[1]:
                x_obs1 = np.array(obs[it1].x_obs)
                # x_obs1 = x_obs1[0:-1,:]
                x_obs1 = np.tile(
                    (x_obs1), (obs[it2].boundary_points_local.shape[1], 1, 1)
                )
                x_obs1 = np.swapaxes(x_obs1, 0, 1)

            # x_obs2 = np.unique(obs[it2].x_obs, axis=0)
            x_obs2 = np.array(obs[it2].x_obs)
            # x_obs2 = x_obs2[0:-1, :]
            x_obs2 = np.tile((x_obs2), (obs[it1].boundary_points_local.shape[1], 1, 1))
            x_obs2 = np.swapaxes(x_obs1, 0, 1)
            x_obs2 = np.swapaxes(x_obs1, 0, 2)
            import pdb

            pdb.set_trace()  ## DEBUG ##

            # ref_dist = marg_dynCenter*0.5*np.sqrt(0.25*(np.sqrt(np.sum(obs[it1].a)**2+max(obs[it2].a)**2)))

            # For ellipses:
            # dist_contact = 0.5*(np.sqrt(np.sum(np.array(obs[it1].a)**2))) + np.sqrt(np.sum(np.array(obs[it2].a)**2))
            # ref_dist = dist_contact*marg_dynCenter
            min_margin_scaling = 0.9
            max_margin_scaling = 1.1
            dist_contact = min_margin_scaling * (
                obs[it1].get_minimal_distance() + obs[it2].get_minimal_distance()
            )

            ref_dist = max_margin_scaling * (
                obs[it1].get_maximal_distance() + obs[it2].get_maximal_distance()
            )

            # Inside consideration region -- are obstacles close to each other
            ind = (
                np.sum(
                    (
                        x_obs_sf[it2]
                        - np.tile(obs[it1].x0, (x_obs_sf[it2].shape[1], 1)).T
                    )
                    ** 2,
                    axis=0,
                )
                < ref_dist**2
            )

            if not sum(ind):
                delta_dist = ref_dist + 1  # greater than reference distance
                continue  # Obstacle too far away

            # Get minimal distance between all points
            # Computationally expensive (N^2*M^2) // is there an alternative
            distSqr = np.sum((x_obs2 - x_obs1) ** 2, axis=2).reshape(1, -1).squeeze()
            minDist_ind_vect = np.argsort((distSqr))

            # minDist_jj = floor(minDist_ind_vect[0]/resolution)
            # minDist_ll = minDist_ind_vect[0] - floor(minDist_ind_vect[0]/resolution)*resolution

            # For matrix form
            minDist_ind = -np.ones((2, N_closest))
            minDist = np.zeros((2, N_closest))
            jj, ll = 0, 0
            for ii in range(N_closest):
                minDist_jj = floor(minDist_ind_vect[jj] / resolution)
                while minDist_jj in minDist_ind[0, :]:
                    jj += 1
                    minDist_jj = floor(minDist_ind_vect[jj] / resolution)
                minDist_ind[0, ii] = minDist_jj
                jj += 1
                minDist[0, ii] = np.sqrt(distSqr[minDist_ind_vect[jj]])

                minInd_ll = (
                    minDist_ind_vect[ll]
                    - floor(minDist_ind_vect[ll] / resolution) * resolution
                )
                while minInd_ll in minDist_ind[1, :]:
                    ll += 1
                    minInd_ll = (
                        minDist_ind_vect[ll]
                        - floor(minDist_ind_vect[ll] / resolution) * resolution
                    )
                minDist_ind[1, ii] = minInd_ll
                ll += 1
                minDist[1, ii] = np.sqrt(distSqr[minDist_ind_vect[ll]])

            # get corresponding weights
            weights = np.zeros((2, N_closest))
            weights[0, :] = compute_weights(minDist[0, :], distMeas_lowerLimit=0)
            weights[1, :] = compute_weights(minDist[1, :], distMeas_lowerLimit=0)

            # Desired Gamma in (0,1) to be on obstacle
            # Gamma_dynCenter = max([1-delta_dist/(ref_dist-dist_contact),0])
            powerCent = 0.5
            Gamma_dynCenter = (
                np.maximum(1 - minDist / (ref_dist - dist_contact), 0) ** powerCent
            )

            x_cyn_temp[:, it1, it2] = np.zeros((obs[it1].d))
            x_cyn_temp[:, it2, it1] = np.zeros((obs[it1].d))
            # Desired position of dynamic_center if only one obstacle existed
            for ww in range(N_closest):
                x_cyn_temp[:, it1, it2] = x_cyn_temp[:, it1, it2] + weights[0, ww] * (
                    obs[it1].x_obs[int(minDist_ind[0, ww])] - np.array(obs[it1].x0)
                ) * np.tile(Gamma_dynCenter[0, ww], (obs[it1].d))

                x_cyn_temp[:, it2, it1] = x_cyn_temp[:, it2, it1] + weights[1, ww] * (
                    obs[it2].x_obs[int(minDist_ind[1, ww])] - np.array(obs[it2].x0)
                ) * np.tile(Gamma_dynCenter[1, ww], (obs[it2].d))

            # Weight to include all obstacles
            delta_dist = np.min(minDist)
            # Negative step
            if (
                delta_dist == 0
            ):  # Obstacles are touching: weight is only assigned to one obstacle
                weight_obs_temp[it1, it2] = -1
            elif delta_dist >= ref_dist:  # Obstacle is far away
                weight_obs_temp[it1, it2] = 0
                continue
            else:
                weight_obs_temp[it1, it2] = max(
                    1 / delta_dist - 1 / (ref_dist - dist_contact), 0
                )  # if too far away/
            weight_obs_temp[it2, it1] = weight_obs_temp[it1, it2]
            # Desired Gamma in (0,1) to be on obstacle

    for it1 in range(N_obs):  # Assign dynamic center
        if it1 in intersection_obs:
            continue  # Don't reasign dynamic center if intersection exists

        if np.sum(
            abs(weight_obs_temp[it1, :])
        ):  # Some obstacles are close to each other
            # Check if there are points on the surface of the obstacle
            pointOnSurface = weight_obs_temp == -1
            if np.sum(pointOnSurface):
                weight_obs = 1 * pointOnSurface  # Bool to float
            else:
                weight_obs = weight_obs_temp[:, it1] / np.sum(weight_obs_temp[:, it1])

            # Linear interpolation if at least one close obstacle --- MAYBE
            # change to nonlinear
            x_centDyn = np.squeeze(x_cyn_temp[:, it1, :])

            obs[it1].center_dyn = (
                np.sum(x_centDyn * np.tile(weight_obs, (obs[it1].d, 1)), axis=1)
                + obs[it1].x0
            )
            # plt.plot(obs[it1].center_dyn[0], obs[it1].center_dyn[1], 'ro')

        else:  # default center otherwise
            obs[it1].center_dyn = obs[it1].x0
