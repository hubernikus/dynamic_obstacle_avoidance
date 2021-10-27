"""
Library for the Modulation of Linear Systems
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021
import warnings

import numpy as np
import numpy.linalg as LA

from vartools.directional_space import get_directional_weighted_sum

from dynamic_obstacle_avoidance.utils import *


def compute_diagonal_matrix(
    Gamma,
    dim,
    is_boundary=False,
    rho=1,
    repulsion_coeff=1.0,
    tangent_eigenvalue_isometric=True,
    tangent_power=5,
    treat_obstacle_special=True,
):
    """Compute diagonal Matrix"""
    if Gamma <= 1 and treat_obstacle_special:
        # Point inside the obstacle
        delta_eigenvalue = 1
    else:
        delta_eigenvalue = 1.0 / abs(Gamma) ** (1.0 / rho)
    eigenvalue_reference = 1 - delta_eigenvalue * repulsion_coeff

    if tangent_eigenvalue_isometric:
        eigenvalue_tangent = 1 + delta_eigenvalue
    else:
        # Decreasing velocity in order to reach zero on surface
        eigenvalue_tangent = 1 - 1.0 / abs(Gamma) ** tangent_power
    return np.diag(
        np.hstack((eigenvalue_reference, np.ones(dim - 1) * eigenvalue_tangent))
    )


def compute_decomposition_matrix(obs, x_t, in_global_frame=False, dot_margin=0.02):
    """Compute decomposition matrix and orthogonal matrix to basis"""
    normal_vector = obs.get_normal_direction(
        x_t, normalize=True, in_global_frame=in_global_frame
    )
    reference_direction = obs.get_reference_direction(
        x_t, in_global_frame=in_global_frame
    )

    dot_prod = np.dot(normal_vector, reference_direction)
    if obs.is_non_starshaped and np.abs(dot_prod) < dot_margin:
        # Adapt reference direction to avoid singularities
        # WARNING: full convergence is not given anymore, but impenetrability
        if not np.linalg.norm(normal_vector):  # zero
            normal_vector = -reference_direction
        else:
            weight = np.abs(dot_prod) / dot_margin
            dir_norm = np.copysign(1, dot_prod)
            reference_direction = get_directional_weighted_sum(
                reference_direction=normal_vector,
                directions=np.vstack((reference_direction, dir_norm * normal_vector)).T,
                weights=np.array([weight, (1 - weight)]),
            )

    E_orth = get_orthogonal_basis(normal_vector, normalize=True)
    E = np.copy((E_orth))
    E[:, 0] = -reference_direction

    return E, E_orth


def compute_modulation_matrix(
    x_t, obs, matrix_singularity_margin=pi / 2.0 * 1.05, angular_vel_weight=0
):
    # TODO: depreciated remove
    """
    The function evaluates the gamma function and all necessary components needed to construct
    the modulation function, to ensure safe avoidance of the obstacles.
    Beware that this function is constructed for ellipsoid only, but the algorithm is applicable
    to star shapes.

    Input
    x_t [dim]: The position of the robot in the obstacle reference frame
    obs [obstacle class]: Description of the obstacle with parameters

    Output
    E [dim x dim]: Basis matrix with rows the reference and tangent to the obstacles surface
    D [dim x dim]: Eigenvalue matrix which is responsible for the modulation
    Gamma [dim]: Distance function to the obstacle surface (in direction of the reference vector)
    E_orth [dim x dim]: Orthogonal basis matrix with rows the normal and tangent
    """
    if True:
        raise NotImplementedError("Depreciated ---- remove")
    warnings.warn("Depreciated ---- remove")
    dim = obs.dim

    if hasattr(obs, "rho"):
        rho = np.array(obs.rho)
    else:
        rho = 1

    Gamma = obs.get_gamma(x_t, in_global_frame=False)  # function for ellipsoids

    E, E_orth = compute_decomposition_matrix(obs, x_t, dim)
    import pdb

    pbd.set_trace()
    D = compute_diagonal_matrix(
        Gamma,
        dim=dim,
        is_boundary=obs.is_boundary,
        repulsion_coeff=obs.repulsion_coeff,
    )

    return E, D, Gamma, E_orth


def obs_avoidance_interpolation_moving(
    position,
    initial_velocity,
    obs=[],
    attractor=None,
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
    xd=None,
):
    """
    This function modulates the dynamical system at position x and dynamics xd such that it
    avoids all obstacles obs. It can furthermore be forced to converge to the attractor.

    Parameters
    ----------
    x [dim]: position at which the modulation is happening
    xd [dim]: initial dynamical system at position x
    obs [list of obstacle_class]: a list of all obstacles and their properties, which
        present in the local environment
    attractor [list of [dim]]]: list of positions of all attractors
    weightPow [int]: hyperparameter which defines the evaluation of the weight

    Return
    ------
    xd [dim]: modulated dynamical system at position x

    """
    if x is not None:
        warnings.warn("Depreciated, don't use x as position argument.")
        position = x
    else:
        x = position

    if xd is not None:
        warnings.warn("xd is depriciated. Use <<initial_velocity>> instead.")
        initial_velocity = xd

    N_obs = len(obs)
    if not N_obs:  # No obstacle
        return initial_velocity

    dim = obs[0].dimension

    initial_velocity_norm = np.linalg.norm(initial_velocity)
    if initial_velocity_norm:
        initial_velocity_normalized = initial_velocity / initial_velocity_norm
    else:
        return initial_velocity  # Trivial solution

    if attractor is not None:
        N_attr = 1
    else:
        N_attr = 0

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

        if obs[n].is_boundary:
            pass
        # warnings.warn('Not... Artificially increasing boundary influence.')
        # Gamma[n] = pow(Gamma[n], 1.0/3.0)
        # else:
        # pass

    Gamma_proportional = np.zeros((N_obs))
    for n in range(N_obs):
        Gamma_proportional[n] = obs[n].get_gamma(
            pos_relative[:, n],
            in_global_frame=evaluate_in_global_frame,
            gamma_distance=gamma_distance,
        )

    # Worst case of being at the center
    if any(Gamma == 0):
        return np.zeros(dim)

    if zero_vel_inside and any(Gamma < 1):
        return np.zeros(dim)

    ind_obs = Gamma < cut_off_gamma
    if any(~ind_obs):
        # warnings.warn('Exceeding cut-off gamma. Stopping modulation.')
        return initial_velocity

    if N_attr:
        d_a = np.linalg.norm(x - np.array(attractor))  # Distance to attractor
        weight = compute_weights(np.hstack((Gamma_proportional, [d_a])), N_obs + N_attr)
    else:
        weight = compute_weights(Gamma_proportional, N_obs)

    # Modulation matrices
    E = np.zeros((dim, dim, N_obs))
    D = np.zeros((dim, dim, N_obs))
    E_orth = np.zeros((dim, dim, N_obs))

    for n in np.arange(N_obs)[ind_obs]:
        # x_t = obs[n].transform_global2relative(x) # Move to obstacle centered frame
        D[:, :, n] = compute_diagonal_matrix(
            Gamma[n],
            dim,
            repulsion_coeff=obs[n].repulsion_coeff,
            tangent_eigenvalue_isometric=tangent_eigenvalue_isometric,
            rho=obs[n].reactivity,
        )

        E[:, :, n], E_orth[:, :, n] = compute_decomposition_matrix(
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
        velocity_only_in_positive_normal_direction = True

        if velocity_only_in_positive_normal_direction:
            lin_vel_local = E_orth[:, :, n].T.dot(obs[n].linear_velocity)
            if lin_vel_local[0] < 0 and not obs[n].is_boundary:
                # Obstacle is moving towards the agent
                linear_velocity = np.zeros(lin_vel_local.shape[0])
            else:
                linear_velocity = E_orth[:, 0, n].dot(lin_vel_local[0])

            weight_linear = np.exp(
                -1 / obs[n].sigma * (np.max([Gamma_proportional[n], 1]) - 1)
            )
            # linear_velocity = weight_linear*linear_velocity

        xd_obs_n = weight_linear * linear_velocity + weight_angular * xd_w

        # The Exponential term is very helpful as it help to avoid
        # the crazy rotation of the robot due to the rotation of the object
        if obs[n].is_deforming:
            weight_deform = np.exp(
                -1 / obs[n].sigma * (np.max([Gamma_proportional[n], 1]) - 1)
            )
            vel_deformation = obs[n].get_deformation_velocity(pos_relative[:, n])

            if velocity_only_in_positive_normal_direction:
                vel_deformation_local = E_orth[:, :, n].T.dot(vel_deformation)
                if (vel_deformation_local[0] > 0 and not obs[n].is_boundary) or (
                    vel_deformation_local[0] < 0 and obs[n].is_boundary
                ):
                    vel_deformation = np.zeros(vel_deformation.shape[0])

                else:
                    vel_deformation = E_orth[:, 0, n].dot(vel_deformation_local[0])

            xd_obs_n += weight_deform * vel_deformation

        xd_obs = xd_obs + xd_obs_n * weight[n]

    # Computing the relative velocity with respect to the obstacle
    relative_velocity = initial_velocity - xd_obs

    relative_velocity_hat = np.zeros((dim, N_obs))
    relative_velocity_hat_magnitude = np.zeros((N_obs))

    n = 0
    for n in np.arange(N_obs)[ind_obs]:
        if (
            obs[n].repulsion_coeff > 1
            and E_orth[:, 0, n].T.dot(relative_velocity) < 0
            # or(obs[n].is_boundary and E_orth[:, 0, n].T.dot(relative_velocity)>0) or
        ):
            # Only consider boundary when moving towards (normal direction)
            # OR if the object has positive repulsion-coefficient (only consider it at front)
            relative_velocity_hat[:, n] = relative_velocity

        else:
            # Matrix inversion cost between O(n^2.373) - O(n^3)
            if not evaluate_in_global_frame:
                relative_velocity_temp = obs[n].transform_global2relative_dir(
                    relative_velocity
                )
            else:
                relative_velocity_temp = np.copy(relative_velocity)

            # Modulation with M = E @ D @ E^-1
            relative_velocity_trafo = np.linalg.pinv(E[:, :, n]).dot(
                relative_velocity_temp
            )

            if obs[n].repulsion_coeff < 0:
                # Negative Repulsion Coefficient at the back of an obstacle
                if E_orth[:, 0, n].T.dot(relative_velocity) < 0:
                    # import pdb; pdb.set_trace()
                    # Adapt in reference direction
                    D[0, 0, n] = 2 - D[0, 0, n]

            # relative_velocity_trafo[0]>0
            elif not obs[n].tail_effect and (
                (relative_velocity_trafo[0] > 0 and not obs[n].is_boundary)
                or (relative_velocity_trafo[0] < 0 and obs[n].is_boundary)
            ):
                D[0, 0, n] = 1  # No effect in 'radial direction'

            stretched_velocity = D[:, :, n].dot(relative_velocity_trafo)

            if D[0, 0, n] < 0:
                # Repulsion in tangent direction, too, have really active repulsion
                factor_tangent_repulsion = 2
                tang_vel_norm = LA.norm(relative_velocity_trafo[1:])
                stretched_velocity[0] += (
                    (-1) * D[0, 0, n] * tang_vel_norm * factor_tangent_repulsion
                )

            relative_velocity_hat[:, n] = E[:, :, n].dot(stretched_velocity)

            if not evaluate_in_global_frame:
                relative_velocity_hat[:, n] = obs[n].transform_relative2global_dir(
                    relative_velocity_hat[:, n]
                )
                # import pdb; pdb.set_trace()

        # TODO: review sticky surface feature [!]
        if False:
            # if obs[n].has_sticky_surface:
            relative_velocity_norm = np.linalg.norm(relative_velocity)

            if relative_velocity_norm:  # Nonzero
                # Normalize relative_velocity_hat
                mag = np.linalg.norm(relative_velocity_hat[:, n])
                if mag:  # nonzero
                    relative_velocity_hat[:, n] = relative_velocity_hat[:, n] / mag

                # Limit maximum magnitude with respect to the tangent value
                sticky_surface_power = 2

                # Treat inside obstacle as on the surface
                Gamma_mag = max(Gamma_proportional[n], 1)
                if abs(Gamma_proportional[n]) < 1:
                    # if abs(Gamma_mag) < 1:
                    eigenvalue_magnitude = 0
                else:
                    eigenvalue_magnitude = (
                        1 - 1.0 / abs(Gamma_proportional[n]) ** sticky_surface_power
                    )
                    # eigenvalue_magnitude = 1 - 1./abs(Gamma_mag)**sticky_surface_power

                if not evaluate_in_global_frame:
                    relative_velocity_temp = obs[n].transform_global2relative_dir(
                        relative_velocity_hat[:, n]
                    )
                else:
                    relative_velocity_temp = relative_velocity_hat[:, n]

                tang_vel = np.abs(E_orth[:, :, n].T.dot(relative_velocity_temp)[0])

                eigenvalue_magnitude = (
                    min(eigenvalue_magnitude / tang_vel, 1) if tang_vel else 0
                )

                relative_velocity_hat[:, n] = (
                    relative_velocity_hat[:, n]
                    * relative_velocity_norm
                    * eigenvalue_magnitude
                )

                if not evaluate_in_global_frame:
                    relative_velocity_hat[:, n] = obs[n].transform_relative2global_dir(
                        relative_velocity_hat[:, n]
                    )

        if repulsive_obstacle:
            # Emergency move away from center in case of a collision
            # Not the cleanest solution...
            if Gamma[n] < (1 + repulsive_gammaMargin):
                repulsive_power = 5
                repulsive_factor = 5
                repulsive_gamma = 1 + repulsive_gammaMargin

                repulsive_speed = (
                    (repulsive_gamma / Gamma[n]) ** repulsive_power - repulsive_gamma
                ) * repulsive_factor
                if not obs[n].is_boundary:
                    repulsive_speed *= -1

                pos_rel = obs[n].get_reference_direction(position, in_global_frame=True)

                repulsive_velocity = pos_rel * repulsive_speed
                relative_velocity_hat[:, n] = repulsive_velocity

        relative_velocity_hat_magnitude[n] = np.sqrt(
            np.sum(relative_velocity_hat[:, n] ** 2)
        )

    relative_velocity_hat_normalized = np.zeros(relative_velocity_hat.shape)
    ind_nonzero = relative_velocity_hat_magnitude > 0
    if np.sum(ind_nonzero):
        relative_velocity_hat_normalized[:, ind_nonzero] = relative_velocity_hat[
            :, ind_nonzero
        ] / np.tile(relative_velocity_hat_magnitude[ind_nonzero], (dim, 1))

    if N_attr:
        # TODO: implement properly & test
        # points at the origin
        k_ds = np.hstack((k_ds, np.zeros((dim - 1, N_attr))))
        relative_velocity_hat_magnitude = np.hstack(
            (
                relative_velocity_hat_magnitude,
                np.linalg.norm((relative_velocity)) * np.ones(N_attr),
            )
        )

        total_weight = 1 - weight_attr
    else:
        total_weight = 1

    weighted_direction = get_directional_weighted_sum(
        null_direction=initial_velocity_normalized,
        directions=relative_velocity_hat_normalized,
        weights=weight,
        total_weight=total_weight,
    )

    relative_velocity_magnitude = np.sum(relative_velocity_hat_magnitude * weight)
    vel_final = relative_velocity_magnitude * weighted_direction.squeeze()

    vel_final = vel_final + xd_obs

    if velocicity_max is not None:
        # IMPLEMENT MAXIMUM VELOCITY / Not needed with sticky boundary
        velocity_norm = np.linalg.norm(vel_final)
        if velocity_norm > velocicity_max:
            vel_final = vel_final / velocity_norm * velocicity_max

    return vel_final
