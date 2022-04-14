"""
Library for the Rotation (Modulation Imitation) of Linear Systems
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import warnings
import copy
from math import pi

from abc import ABC, abstractmethod

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
from vartools.dynamical_systems import DynamicalSystem

from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.utils import get_weight_from_inv_of_gamma
from dynamic_obstacle_avoidance.utils import get_relative_obstacle_velocity


def get_intersection_with_circle(
    start_position: np.ndarray,
    direction: np.ndarray,
    radius: float,
    only_positive: bool = True,
) -> np.ndarray:
    """Returns intersection with circle with center at 0
    of of radius 'radius' and the line defined as
    'start_position + x * direction'
    (!) Only intersection at furthest distance to start_point is returned."""
    if not radius:  # Zero radius
        return None

    # Binomial Formula to solve for x in:
    # || dir_reference + x * (delta_dir_conv) || = radius
    AA = np.sum(direction**2)
    BB = 2 * np.dot(direction, start_position)
    CC = np.sum(start_position**2) - radius**2
    DD = BB**2 - 4 * AA * CC

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


class BaseAvoider(ABC):
    @abstractmethod
    def avoid(self, *args, **kwargs):
        pass


class RotationalAvoider(BaseAvoider):
    """
    RotationalAvoider -> Obstacle Avoidance based on local avoider.
    """

    def __init__(
        self,
        initial_dynamics: DynamicalSystem = None,
        convergence_system: DynamicalSystem = None,
        obstacle_environment=None,
        cut_off_gamma: float = 1e6,
    ):
        """Initial dynamics, convergence direction and obstacle list are used."""
        self.initial_dynamics = initial_dynamics
        if convergence_system is None:
            self.convergence_system = self.initial_dynamics
        else:
            self.convergence_system = convergence_system

        self.obstacle_list = obstacle_environment

        self.cut_off_gamma = cut_off_gamma

    def avoid(
        self,
        position: np.ndarray,
        initial_velocity: np.ndarray,
        obstacle_list: list,
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
        if initial_velocity is None:
            initial_velocity = self.initial_dynamics.evaluate(position)

        if obstacle_list is None:
            # TODO: depreciated
            obstacle_list = self.obstacle_environment

        n_obstacles = len(obstacle_list)
        if not n_obstacles:  # No obstacles in the environment
            return initial_velocity

        if hasattr(obstacle_list, "update_relative_reference_point"):
            # TODO: directly return gamma_array
            obstacle_list.update_relative_reference_point(position=position)

        dimension = position.shape[0]

        gamma_array = np.zeros((n_obstacles))
        for ii in range(n_obstacles):
            gamma_array[ii] = obstacle_list[ii].get_gamma(
                position, in_global_frame=True
            )

            if gamma_array[ii] < 1 and not obstacle_list[ii].is_boundary:
                # Since boundaries are mutually subtracted,
                raise NotImplementedError()

        ind_obs = np.logical_and(gamma_array < self.cut_off_gamma, gamma_array >= 1)

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
        # modulated_velocity = initial_velocity - relative_velocity

        inv_gamma_weight = get_weight_from_inv_of_gamma(gamma_array)
        # rotated_velocities = np.zeros((dimension, n_obs_close))

        rotated_directions = [None] * n_obs_close
        for it, it_obs in zip(range(n_obs_close), np.arange(n_obstacles)[ind_obs]):
            # It is with respect to the close-obstacles
            # -- it_obs ONLY to use in obstacle_list (whole)
            # => the null matrix should be based on the normal
            # direction (not reference, right?!)
            reference_dir = obstacle_list[it_obs].get_reference_direction(
                position, in_global_frame=True
            )

            # Null matrix is pointing towards the object
            if obstacle_list[it_obs].is_boundary:
                reference_dir = (-1) * reference_dir
                # null_matrix = normal_orthogonal_matrix[:, :, it] * (-1)

            else:
                # null_matrix = normal_orthogonal_matrix[:, :, it]
                null_matrix = normal_orthogonal_matrix[:, :, it] * (-1)

            # Convergence direcctions can be local for certain obstacles
            # / convergence environments
            convergence_velocity = obstacle_list.get_convergence_direction(
                position=position, it_obs=it_obs
            )

            conv_vel_norm = np.linalg.norm(convergence_velocity)
            if conv_vel_norm:
                # Zero value
                base = DirectionBase(matrix=null_matrix)

                # rotated_velocities[:, it] = UnitDirection(base).from_vector(initial_velocity)
                rotated_directions[it] = UnitDirection(base).from_vector(
                    initial_velocity
                )

            # Note that the inv_gamma_weight was prepared for the multiboundary
            # environment through the reference point displacement (see 'loca_reference_point')
            rotated_directions[it] = self.directional_convergence_summing(
                convergence_vector=convergence_velocity,
                reference_vector=reference_dir,
                weight=inv_gamma_weight[it],
                nonlinear_velocity=initial_velocity,
                base=DirectionBase(matrix=null_matrix),
            )

        base = DirectionBase(vector=initial_velocity)
        rotated_velocity = get_directional_weighted_sum_from_unit_directions(
            base=base, weights=weights, unit_directions=rotated_directions
        )

        # Magnitude such that zero on the surface of an obstacle
        magnitude = np.dot(inv_gamma_weight, weights) * np.linalg.norm(initial_velocity)

        rotated_velocity = rotated_velocity * magnitude
        rotated_velocity = rotated_velocity - relative_velocity
        # TODO: check maximal magnitude (in dynamic environments); i.e. see paper
        return rotated_velocity

    def _get_directional_deviation_weight(
        self, weight: float, weight_deviation: float, power_factor: float = 3.0
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
        self,
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
        self,
        inv_conv_rotated: UnitDirection,
        inv_nonlinear: UnitDirection,
        inv_convergence_radius: UnitDirection,
    ) -> UnitDirection:
        """Returns projected converenge direction based in the (normal)-direction space.

        The method only projects when crossing is actually needed.
        It checks the connection points on the direction-circle from dir_nonlinear to dir_convergence,
        and does following:
        - [dir_nolinear in convergence_radius] => no rotation => return dir_nonlinear
        - [No Intersection] => [weight=0] => no rotation => return dir_nonlinear
        - [dir_convergence in convergence_radius] => return the intersection point
        - [two intersection points] => return relative sectant

        Note: This method has the danger that the modulation only ever happens if the rotation is going
        in the correct direction, i.e., it is in correct part of the circle.
        It is hence important to activate the rotation early enough.

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

        # Both points are returned since we need to measure the 'sequente'
        if inv_conv_rotated.norm() <= inv_convergence_radius:
            point = get_intersection_with_circle(
                start_position=inv_conv_rotated.as_angle(),
                direction=(inv_conv_rotated - inv_nonlinear).as_angle(),
                radius=inv_convergence_radius,
                only_positive=True,
            )
            # sectant_dist = LA.norm(point - inv_conv_rotated.as_angle())
            # Since $w_cp = 1$ it we directly return the value
            return UnitDirection(inv_conv_rotated.base).from_angle(point)

        points = get_intersection_with_circle(
            start_position=inv_conv_rotated.as_angle(),
            direction=(inv_conv_rotated - inv_nonlinear).as_angle(),
            radius=inv_convergence_radius,
            only_positive=False,
        )

        if points is None:
            # No intersection => we are outside
            return inv_conv_rotated

        # Check if the two intersection points are inbetween the two directions
        # Any of the points can be chosen for the u check, since either both
        # or none points are between / outside
        dir_vector_conv = (inv_conv_rotated - inv_nonlinear).as_angle()
        dir_vector_point = points[:, 1] - inv_nonlinear.as_angle()

        if np.dot(dir_vector_conv, dir_vector_point) < 0 or np.dot(
            dir_vector_conv, dir_vector_point
        ) > np.dot(dir_vector_conv, dir_vector_conv):
            # Intersections are behind or in front of both points with respect to the two angles
            return inv_conv_rotated

        w_cp = LA.norm(points[:, 0] - points[:, 1]) / LA.norm(
            inv_conv_rotated.as_angle() - points[:, 1]
        )

        inv_conv_proj = (
            w_cp * UnitDirection(inv_conv_rotated.base).from_angle(points[:, 1])
            + (1 - w_cp) * inv_conv_rotated
        )

        if LA.norm(inv_conv_proj.as_angle()) > np.pi:
            # -> DEBUG
            raise NotImplementedError(
                f"Unexpected value of {LA.norm(inv_conv_proj.as_angle())}"
            )

        return inv_conv_proj

    def _get_projected_nonlinear_velocity(
        self,
        dir_conv_rotated: UnitDirection,
        dir_nonlinear: UnitDirection,
        weight: float,
        convergence_radius: float = np.pi / 2,
    ) -> UnitDirection:
        """Invert the directional-circle-space and project the nonlinear velocity to approach
        the linear counterpart.

        Parameters
        ----------
        dir_conv_rotated: rotated convergence direction which guides the nonlinear
        dir_nonlinear: nonlinear direction which is pulled towards the convergence direction.
        weight: initial weight - which is taken into the calculation
        convergence_radius: radius of circle to which direction is pulled
            towards (e.g. pi/2 = tangent)

        Returns
        -------
        nonlinar_conv: Projected nonlinear velocity to be aligned with convergence velocity
        """
        # Invert matrix to get smooth summing.
        inv_nonlinear = dir_nonlinear.invert_normal()

        # Only project when 'outside the radius'
        inv_convergence_radius = np.pi - convergence_radius
        if inv_nonlinear.norm() <= inv_convergence_radius:
            return dir_nonlinear

        inv_conv_rotated = dir_conv_rotated.invert_normal()
        weight_nonl = self._get_nonlinear_inverted_weight(
            inv_conv_rotated.norm(),
            inv_nonlinear.norm(),
            inv_convergence_radius,
            weight=weight,
        )

        if not weight_nonl:  # zero value
            return inv_nonlinear.invert_normal()

        # TODO: integrate this function here
        inv_conv_proj = self._get_projection_of_inverted_convergence_direction(
            inv_conv_rotated=inv_conv_rotated,
            inv_nonlinear=inv_nonlinear,
            inv_convergence_radius=inv_convergence_radius,
        )

        inv_nonlinear_conv = (
            weight_nonl * inv_conv_proj + (1 - weight_nonl) * inv_nonlinear
        )
        return inv_nonlinear_conv.invert_normal()

    def _get_rotated_convergence_direction(
        self,
        weight: float,
        convergence_radius: float,
        convergence_vector: np.ndarray,
        reference_vector: np.ndarray,
        base: DirectionBase,
    ) -> UnitDirection:
        """Rotates the convergence vector according to given input and basis"""

        dir_reference = UnitDirection(base).from_vector(reference_vector)
        dir_convergence = UnitDirection(base).from_vector(convergence_vector)

        if dir_convergence.norm() >= convergence_radius:
            # Initial velocity 'dir_convergecne' already pointing away from obstacle
            return dir_convergence

        # Find intersection a with radius of pi/2
        # Inside the tangent radius,
        # i.e. vectorfield towards obstacle [no-tail-effect]
        # Do the math in the angle space
        delta_dir_conv = dir_convergence - dir_reference

        norm_dir_conv = delta_dir_conv.norm()
        if not norm_dir_conv:  # Zero value
            return None

        angle_tangent = get_intersection_with_circle(
            start_position=dir_reference.as_angle(),
            direction=delta_dir_conv.as_angle(),
            radius=convergence_radius,
        )

        dir_tangent = UnitDirection(base).from_angle(angle_tangent)

        norm_tangent_dist = (dir_tangent - dir_reference).norm()

        # Weight to ensure that:
        weight_deviation = norm_dir_conv / norm_tangent_dist
        w_conv = self._get_directional_deviation_weight(
            weight, weight_deviation=weight_deviation
        )

        return w_conv * dir_tangent + (1 - w_conv) * dir_convergence

    def directional_convergence_summing(
        self,
        convergence_vector: np.ndarray,
        reference_vector: np.ndarray,
        base: DirectionBase,
        weight: float,
        nonlinear_velocity: np.ndarray = None,
        convergence_radius: float = np.pi / 2,
    ) -> UnitDirection:
        """Rotating / modulating a vector by using directional space.

        Parameters
        ---------
        convergence_vector: a array of floats of size (dimension,)
        reference_vector: a array of floats of size (dimension,)
        weight: float in the range [0, 1] which gives influence on how important vector 2 is.
        nonlinear_velocity: (optional) the vector-field which converges

        Returns
        -------
        converging_velocity: Weighted summing in direction-space to 'emulate' the modulation.
        """
        if weight >= 1:
            # TODO return special
            weight = 1
        elif weight <= 0:
            # TODO return special
            weight = 0

        dir_conv_rotated = self._get_rotated_convergence_direction(
            weight=weight,
            convergence_radius=convergence_radius,
            convergence_vector=convergence_vector,
            reference_vector=reference_vector,
            base=base,
        )

        if dir_conv_rotated is None:
            if nonlinear_velocity is None:
                return UnitDirection(base).from_vector(convergence_vector)
            else:
                return UnitDirection(base).from_vector(nonlinear_velocity)

        if nonlinear_velocity is None:
            return dir_conv_rotated

        dir_nonlinear = UnitDirection(base).from_vector(nonlinear_velocity)

        nonlinear_conv = self._get_projected_nonlinear_velocity(
            dir_conv_rotated=dir_conv_rotated,
            dir_nonlinear=dir_nonlinear,
            convergence_radius=convergence_radius,
            weight=weight,
        )

        return nonlinear_conv
