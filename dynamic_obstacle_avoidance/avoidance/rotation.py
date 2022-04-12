""" Library for the Rotation (Modulation Imitation) of Linear Systems
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import warnings
import copy
from math import pi

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt  # For debugging only (!)

from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_directional_weighted_sum
from vartools.directional_space import (
    get_directional_weighted_sum_from_unit_directions,
)
from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import UnitDirection, DirectionBase

from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.utils import get_weight_from_inv_of_gamma
from dynamic_obstacle_avoidance.utils import get_relative_obstacle_velocity


class WeightOutOfBoundError(Exception):
    def __init__(self, weight):
        self.weight = weight
        super().__init__()

    def __str__(self):
        return (
            f"weight={self.weight} -> Weight out of bounded, expected to be in [0, 1]."
        )


# TODO: Speed up using cython / cpp e.g. eigen?
# TODO: list / array stack for lru_cache to speed

# TODO: speed up learning through cpp / c / cython(!?)
def get_intersection_with_circle(
    start_position: np.ndarray,
    direction: np.ndarray,
    radius: float,
    only_positive: bool = True,
) -> np.ndarray:
    """Returns intersection with circle of of radius 'radius' and the line defined as
    'start_position + x * direction'
    (!) Only intersection at furthest distance to start_point is returned."""
    if not radius:  # Zero radius
        return None

    # Binomial Formula to solve for x in:
    # || dir_reference + x * (delta_dir_conv) || = radius
    AA = np.sum(direction ** 2)
    BB = 2 * np.dot(direction, start_position)
    CC = np.sum(start_position ** 2) - radius ** 2
    DD = BB ** 2 - 4 * AA * CC

    if DD < 0:
        # No intersection with circle
        return None

    if only_positive:
        # Only negative direction due to expected negative A (?!) [returns max-direciton]..
        fac_direction = (-BB + np.sqrt(DD)) / (2 * AA)
        point = start_position + fac_direction * direction
        return point

    else:
        factors = (-BB + np.array([-1, 1]) * np.sqrt(DD)) / (2 * AA)
        points = (
            np.tile(start_position, (2, 1)).T
            + np.tile(factors, (start_position.shape[0], 1))
            * np.tile(direction, (2, 1)).T
        )
        return points


def _get_directional_deviation_weight(
    weight: float, weight_deviation: float, power_factor: float = 3.0
) -> float:
    """This 'smooth'-weighting needs to be done, in order to have a smooth vector-field
    which can be approximated by the nonlinear DS."""
    if weight_deviation <= 0:
        return 0
    elif weight_deviation >= 1 or weight >= 1:
        return 1
    else:
        return weight ** (1.0 / (weight_deviation * power_factor))


def _get_nonlinear_inverted_weight(
    inverted_conv_rotated_norm: float,
    inverted_nonlinear_norm: float,
    inv_convergence_radius: float,
    weight: float,
) -> float:

    if inverted_nonlinear_norm <= inv_convergence_radius:
        return 0

    elif inverted_conv_rotated_norm > inv_convergence_radius:
        delta_nonl = inverted_nonlinear_norm - inv_convergence_radius
        delta_conv = inverted_conv_rotated_norm - inv_convergence_radius
        # weight_nonl = weight * delta_nonl/(delta_nonl + delta_conv)
        return weight * delta_nonl / (delta_nonl + delta_conv)

    else:
        # weight_nonl = weight
        return weight


def _get_projection_of_inverted_convergence_direction(
    inv_conv_rotated: UnitDirection,
    inv_nonlinear: UnitDirection,
    # delta_dir_conv: UnitDirection,
    inv_convergence_radius: UnitDirection,
) -> UnitDirection:
    """Returns projected converenge direction based in the (normal)-direction space.

    Parameters
    ----------
    dir_conv_rotated
    # delta_dir_conv: Difference between dir_conv_rotated & delta_dir_conv (in normal space
    inv_convergence_radius: the inverted convergence radius

    Returns
    -------
    Convergence value rotated.
    """
    if inv_nonlinear.norm() <= inv_convergence_radius:
        # Already converging
        return inv_conv_rotated

    inv_conv_rotated = inv_conv_rotated
    points = get_intersection_with_circle(
        start_position=inv_conv_rotated.as_angle(),
        direction=(inv_conv_rotated - inv_nonlinear).as_angle(),
        radius=inv_convergence_radius,
        only_positive=False,
    )

    if inv_conv_rotated.norm() <= inv_convergence_radius:
        # weight=1 -> set to point far-away
        inv_conv_proj = UnitDirection(inv_conv_rotated.base).from_angle(points[:, 0])

    elif points is not None:
        # 2 points are returned
        nonl_angle = inv_nonlinear.as_angle()
        convrot_angle = inv_conv_rotated.as_angle()
        for ii in range(points.shape[0]):
            nonl_dir = points[ii, 1] - nonl_angle[ii]
            if not np.isclose(nonl_dir, 0):  # Nonzero
                point_dir = convrot_angle[ii] - nonl_angle[ii]
                break

        # Check if direction of points are aligned!
        if point_dir / nonl_dir > 1:
            dist = LA.norm(np.tile(inv_nonlinear.as_angle(), (2, 1)).T - points, axis=0)
            it_max = np.argmax(dist)
            w_cp = LA.norm(points[:, 0] - points[:, 1]) / LA.norm(
                inv_conv_rotated.as_angle() - points[:, it_max]
            )

            inv_conv_proj = (
                w_cp
                * UnitDirection(inv_conv_rotated.base).from_angle(points[:, it_max])
                + (1 - w_cp) * inv_conv_rotated
            )

        else:
            inv_conv_proj = inv_conv_rotated

    else:
        # Points is none
        inv_conv_proj = inv_conv_rotated

    if len(inv_conv_proj.as_angle()) > pi:  # DEBUG
        breakpoint()

    return inv_conv_proj


def _get_projected_nonlinear_velocity(
    dir_conv_rotated: UnitDirection,
    dir_nonlinear: UnitDirection,
    weight: float,
    convergence_radius: float = pi / 2,
) -> UnitDirection:
    """Invert the directional-circle-space and project the nonlinear velocity to approach
    the linear counterpart.

    Parameters
    ----------
    dir_conv_rotated: rotated convergence direction which guides the nonlinear
    dir_nonlinear: nonlinear direction which is pulled towards the convergence direction.
    weight: initial weight - which is taken into the calculation
    convergence_radius: radius of circle to which direction is pulled towards (e.g. pi/2 = tangent)

    Returns
    -------
    nonlinar_conv: Projected nonlinear velocity to be aligned with convergence velocity
    """
    # Invert matrix to get smooth summing.
    inv_nonlinear = dir_nonlinear.invert_normal()

    # Only project when 'outside the radius'
    inv_convergence_radius = pi - convergence_radius
    if inv_nonlinear.norm() <= inv_convergence_radius:
        # return dir_nonlinear.as_vector()
        return dir_nonlinear

    else:
        inv_conv_rotated = dir_conv_rotated.invert_normal()
        weight_nonl = _get_nonlinear_inverted_weight(
            inv_conv_rotated.norm(),
            inv_nonlinear.norm(),
            inv_convergence_radius,
            weight=weight,
        )

        if not weight_nonl:  # zero value
            return inv_nonlinear.invert_normal()

        inv_conv_proj = _get_projection_of_inverted_convergence_direction(
            inv_conv_rotated=inv_conv_rotated,
            inv_nonlinear=inv_nonlinear,
            inv_convergence_radius=inv_convergence_radius,
        )

        inv_nonlinear_conv = (
            weight_nonl * inv_conv_proj + (1 - weight_nonl) * inv_nonlinear
        )
        return inv_nonlinear_conv.invert_normal()


def directional_convergence_summing(
    convergence_vector: np.ndarray,
    reference_vector: np.ndarray,
    base: DirectionBase,
    weight: float,
    nonlinear_velocity: np.ndarray = None,
    convergence_radius: float = pi / 2,
) -> UnitDirection:
    """Rotating / modulating a vector by using directional space.

    Paramters
    ---------
    convergence_vector: a array of floats of size (dimension,)
    reference_vector: a array of floats of size (dimension,)
    weight: float in the range [0, 1] which gives influence on how important vector 2 is.
    nonlinear_velocity: (optional) the vector-field which converges

    Returns
    -------
    converging_velocity: Weighted summing in direction-space to 'emulate' the modulation.
    """
    # if weight > 1 or weight < 0:
    # raise WeightOutOfBoundError(weight=weight)
    if weight >= 1:
        weight = 1
    elif weight < 0:
        weight = 0

    dir_reference = UnitDirection(base).from_vector(reference_vector)
    dir_convergence = UnitDirection(base).from_vector(convergence_vector)

    # Find intersection a with radius of pi/2
    # convergence_is_inside_tangent = np.linalg.norm(dir_convergence) < convergence_radius
    if dir_convergence.norm() < convergence_radius:
        # Inside the tangent radius, i.e. vectorfield towards obstacle [no-tail-effect]
        # Do the math in the angle space
        delta_dir_conv = dir_convergence - dir_reference

        norm_dir_conv = delta_dir_conv.norm()
        if not norm_dir_conv:  # Zero value
            if nonlinear_velocity is None:
                return UnitDirection(base).from_vector(convergence_vector)
            else:
                return UnitDirection(base).from_vector(nonlinear_velocity)

        angle_tangent = get_intersection_with_circle(
            start_position=dir_reference.as_angle(),
            direction=delta_dir_conv.as_angle(),
            radius=convergence_radius,
        )
        dir_tangent = UnitDirection(base).from_angle(angle_tangent)

        norm_tangent_dist = (dir_tangent - dir_reference).norm()

        # Weight to ensure that:
        weight_deviation = norm_dir_conv / norm_tangent_dist
        w_conv = _get_directional_deviation_weight(
            weight, weight_deviation=weight_deviation
        )
        dir_conv_rotated = w_conv * dir_tangent + (1 - w_conv) * dir_convergence

    else:
        # Initial velocity 'dir_convergecne' already pointing away from obstacle
        dir_conv_rotated = dir_convergence

    if nonlinear_velocity is None:
        return dir_conv_rotated

    else:
        dir_nonlinear = UnitDirection(base).from_vector(nonlinear_velocity)

        nonlinear_conv = _get_projected_nonlinear_velocity(
            dir_conv_rotated=dir_conv_rotated,
            dir_nonlinear=dir_nonlinear,
            convergence_radius=convergence_radius,
            weight=weight,
        )

        return nonlinear_conv


def obstacle_avoidance_rotational(
    position: np.ndarray,
    initial_velocity: np.ndarray,
    obstacle_list: list,
    # obstacle_list: ObstacleContainer,
    cut_off_gamma: float = 1e6,
    gamma_distance: float = None,
) -> np.ndarray:
    """Obstacle avoidance based on 'local' rotation and the directional weighted mean.

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
    if not n_obstacles:  # No obstacles in the environment
        return initial_velocity

    if hasattr(obstacle_list, "update_relative_reference_point"):
        # TODO: directly return gamma_array
        obstacle_list.update_relative_reference_point(position=position)

    dimension = position.shape[0]

    gamma_array = np.zeros((n_obstacles))
    for ii in range(n_obstacles):
        # try:
        gamma_array[ii] = obstacle_list[ii].get_gamma(position, in_global_frame=True)
        # except:
        # breakpoint()

        if gamma_array[ii] < 1 and not obstacle_list[ii].is_boundary:
            # Since boundaries are mutually subtracted,
            # breakpoint()
            raise NotImplementedError()

    ind_obs = np.logical_and(gamma_array < cut_off_gamma, gamma_array >= 1)

    if not any(ind_obs):
        return initial_velocity

    n_obs_close = np.sum(ind_obs)

    gamma_array = gamma_array[ind_obs]  # Only keep relevant ones
    gamma_proportional = np.zeros((n_obs_close))
    normal_orthogonal_matrix = np.zeros((dimension, dimension, n_obs_close))

    for it, it_obs in zip(range(n_obs_close), np.arange(n_obstacles)[ind_obs]):
        gamma_proportional[it] = obstacle_list[it_obs].get_gamma(
            position,
            in_global_frame=True,
            gamma_distance=gamma_distance,
        )
        normal_dir = obstacle_list[it_obs].get_normal_direction(
            position, in_global_frame=True
        )
        normal_orthogonal_matrix[:, :, it] = get_orthogonal_basis(normal_dir)

    weights = compute_weights(gamma_array)
    relative_velocity = get_relative_obstacle_velocity(
        position=position,
        obstacle_list=obstacle_list,
        ind_obstacles=ind_obs,
        gamma_list=gamma_proportional,
        E_orth=normal_orthogonal_matrix,
        weights=weights,
    )
    modulated_velocity = initial_velocity - relative_velocity

    inv_gamma_weight = get_weight_from_inv_of_gamma(gamma_array)
    # rotated_velocities = np.zeros((dimension, n_obs_close))
    rotated_directions = [None] * n_obs_close
    for it, oo in zip(range(n_obs_close), np.arange(n_obstacles)[ind_obs]):
        # It is with respect to the close-obstacles -- oo ONLY to use in obstacle_list (whole)
        reference_dir = obstacle_list[oo].get_reference_direction(
            position, in_global_frame=True
        )

        if obstacle_list[oo].is_boundary:
            reference_dir = (-1) * reference_dir
            null_matrix = normal_orthogonal_matrix[:, :, it] * (-1)
        else:
            null_matrix = normal_orthogonal_matrix[:, :, it]

        convergence_velocity = obstacle_list.get_convergence_direction(
            position=position, it_obs=oo
        )

        conv_vel_norm = np.linalg.norm(convergence_velocity)
        if conv_vel_norm:  # Zero value
            base = DirectionBase(matrix=null_matrix)
            # rotated_velocities[:, it] = UnitDirection(base).from_vector(initial_velocity)
            rotated_directions[it] = UnitDirection(base).from_vector(initial_velocity)

        # Note that the inv_gamma_weight was prepared for the multiboundary environment through
        # the reference point displacement (see 'loca_reference_point')
        rotated_directions[it] = directional_convergence_summing(
            convergence_vector=convergence_velocity,
            reference_vector=reference_dir,
            weight=inv_gamma_weight[it],
            nonlinear_velocity=initial_velocity,
            base=DirectionBase(matrix=null_matrix),
        )
        # rotated_velocities[:, it] = rotated_directionsrotated_velocity[it].as_vector()

    # rotated_velocities = np.array([r_dir.as_vector() for r_dir in rotated_directions]).T

    base = DirectionBase(vector=initial_velocity)
    rotated_velocity = get_directional_weighted_sum_from_unit_directions(
        base=base, weights=weights, unit_directions=rotated_directions
    )

    # rotated_velocity = get_directional_weighted_sum(
    # null_direction=initial_velocity,
    # directions=rotated_velocities,
    # weights=weights,
    # )

    # Magnitude such that zero on the surface of an obstacle
    magnitude = np.dot(inv_gamma_weight, weights) * np.linalg.norm(initial_velocity)

    rotated_velocity = rotated_velocity * magnitude
    rotated_velocity = rotated_velocity - relative_velocity
    # TODO: check maximal magnitude (in dynamic environments); i.e. see paper
    return rotated_velocity
