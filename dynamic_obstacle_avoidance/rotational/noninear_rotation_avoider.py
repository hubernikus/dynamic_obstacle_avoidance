"""
Library for the Rotation (Modulation Imitation) of Linear Systems
"""
# Author: Lukas Huber
# GitHub: hubernikus
# Created: 2021-09-01

import warnings
import copy
import math

import numpy as np
from numpy import linalg as LA

from vartools.math import get_intersection_with_circle
from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_directional_weighted_sum
from vartools.directional_space import (
    get_directional_weighted_sum_from_unit_directions,
)
from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import UnitDirection

# from vartools.directional_space DirectionBase
from vartools.dynamical_systems import DynamicalSystem
from vartools.dynamical_systems import AxesFollowingDynamics
from vartools.dynamical_systems import ConstantValue

from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.utils import get_weight_from_inv_of_gamma
from dynamic_obstacle_avoidance.utils import get_relative_obstacle_velocity

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.avoidance import BaseAvoider
from dynamic_obstacle_avoidance.rotational.rotational_avoider import (
    RotationalAvoider,
)
from dynamic_obstacle_avoidance.rotational.rotation_container import RotationContainer


class NonlinearRotationalAvoider(BaseAvoider):
    """
    NonlinearRotationalAvoider -> Rotational Obstacle Avoidance by additionally considering initial dynamics
    """

    # TODO:
    #   - don't use UnitDirection (as I assume it has a large overhead)

    def __init__(
        self,
        initial_dynamics: DynamicalSystem = None,
        obstacle_environment=None,
        convergence_system: DynamicalSystem = None,
        **kwargs,
    ) -> None:
        """Initial dynamics, convergence direction and obstacle list are used."""
        self._rotation_avoider = RotationalAvoider(
            initial_dynamics=initial_dynamics,
            obstacle_environment=obstacle_environment,
            convergence_system=convergence_system,
            **kwargs,
        )

    @property
    def dimension(self):
        return self._rotation_avoider.initial_dynamics.dimension

    @property
    def cut_off_gamma(self):
        return self._rotation_avoider.cut_off_gamma

    @property
    def n_obstacles(self):
        return len(self._rotation_avoider.obstacle_environment)

    def evaluate_initial_dynamics(self, position: np.ndarray) -> np.ndarray:
        return self._rotation_avoider.initial_dynamics.evaluate(position)

    def evaluate_convergence_dynamics(self, position: np.ndarray) -> np.ndarray:
        return self._rotation_avoider.convergence_dynamics.evaluate(position)

    def avoid(self, position, initial_velocity):
        convergence_velocity = self.evaluate_convergence_dynamics(position)
        return self._rotation_avoider.avoid(
            position=position,
            initial_velocity=initial_velocity,
            convergence_velocity=convergence_velocity,
        )

    def evaluate(self, position, **kwargs):
        initial_velocity = self.evaluate_initial_dynamics(position)
        local_convergence_velocity = self.evaluate_weighted_dynamics(
            position, initial_velocity
        )

        return self._rotation_avoider.avoid(
            position=position,
            initial_velocity=initial_velocity,
            convergence_velocity=local_convergence_velocity,
            **kwargs,
        )

    def evaluate_weighted_dynamics(
        self, position: np.ndarray, initial_velocity: np.ndarray
    ) -> np.ndarray:
        convergence_velocity = self.evaluate_convergence_dynamics(position)

        # TODO: this gamma/weight calculation could be shared...
        gamma_array = np.zeros((self.n_obstacles))
        for ii in range(self.n_obstacles):
            gamma_array[ii] = self._rotation_avoider.obstacle_environment[ii].get_gamma(
                position, in_global_frame=True
            )

        gamma_min = 1

        ind_obs = gamma_array <= gamma_min
        if sum_close := np.sum(ind_obs):
            # Dangerously close..
            weights = np.zeros(ind_obs) * 1.0 / sum_close

        else:
            ind_obs = gamma_array < self._rotation_avoider.cut_off_gamma

            if not np.sum(ind_obs):
                # return self.evaluate_convergence_dynamics(position)
                return convergence_velocity

            # w = 1 / (gamma-gamma_min) - 1 / (gamma_max - gamma_min)
            weights = 1.0 / (gamma_array[ind_obs] - gamma_min) - 1 / (
                self.cut_off_gamma - gamma_min
            )
            if (weight_sum := np.sum(weights)) > 1:
                # Normalize weight, but leave possibility to be smaller than one (!)
                weights = weights / weight_sum

        # Remaining convergence is the linear system, if it is far..
        initial_norm = LA.norm(initial_velocity)
        if weight_sum < 1:
            local_velocities = np.zeros((self.dimension, np.sum(ind_obs) + 1))

            weights = np.append(weights, 1 - weight_sum)

            if not initial_norm:
                return initial_velocity

            local_velocities[:, -1] = initial_velocity / initial_norm

        else:
            local_velocities = np.zeros((self.dimension, np.sum(ind_obs)))

        # Evaluate center directions for the relevant obstacles
        for ii, it_obs in enumerate(np.arange(self.n_obstacles)[ind_obs]):
            local_velocities[:, ii] = self.evaluate_initial_dynamics(
                # TODO: could also be reference point...
                self._rotation_avoider.obstacle_environment[ii].center_position
            )

            # local_magnitudes[ii] = LA.norm(local_velocities[:, ii])
            if not LA.norm(local_velocities[:, ii]):
                # What should be done here (?) <-> smoothly reduce the weight as we approach the center(?)
                raise NotImplementedError()

            # local_velocities[:, ii] = local_velocities[:, ii] / local_magnitudes[ii]

        # convergence_dynamics = self.evaluate_convergence_dynamics(position)
        if not (convergence_norm := LA.norm(convergence_velocity)):
            return convergence_velocity
        convergence_velocity = convergence_velocity / convergence_norm

        # Weighted sum -> should have the same result as 'the graph summing' (but for now slightly more stable)

        try:
            averaged_direction = get_directional_weighted_sum(
                null_direction=convergence_velocity,
                weights=weights,
                directions=local_velocities,
            )
        except:
            breakpoint()

        # Magnitude should stay the one of the initial vel
        # averaged_magnitudes = (
        #     np.sum(local_magnitudes * weights / np.sum(weights))
        #         + (1 - np.sum(weights)) * convergence_norm
        # )

        return initial_norm * averaged_direction


def test_nonlinear_avoider(visualize: bool = False) -> None:

    # initial dynamics
    main_direction = np.array([1, 0])
    convergence_dynamics = ConstantValue(velocity=main_direction)
    initial_dynamics = AxesFollowingDynamics(
        center_position=np.zeros(2),
        maximum_velocity=1.0,
        main_direction=main_direction,
    )
    obstacle_environment = obstacle_list = RotationContainer()
    obstacle_environment.append(
        Ellipse(
            center_position=np.array([5, 2]),
            axes_length=np.array([1.1, 1.1]),
            margin_absolut=0.9,
        )
    )

    obstacle_avoider = NonlinearRotationalAvoider(
        initial_dynamics=initial_dynamics,
        convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
    )

    if visualize:
        x_lim = [0, 20]
        y_lim = [-5, 5]
        # y_lim = [-1, 24]

        figsize = (12, 6)

        # nx = n_resolution
        # ny = n_resolution
        # x_vals, y_vals = np.meshgrid(
        #     np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        # )

        from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
            plot_obstacle_dynamics,
        )
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=obstacle_environment,
            dynamics=obstacle_avoider.evaluate_convergence_dynamics,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
        )
        plot_obstacles(
            ax=ax, obstacle_container=obstacle_environment, x_lim=x_lim, y_lim=y_lim
        )

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=obstacle_environment,
            dynamics=obstacle_avoider.evaluate_initial_dynamics,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
        )
        plot_obstacles(
            ax=ax, obstacle_container=obstacle_environment, x_lim=x_lim, y_lim=y_lim
        )

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=obstacle_environment,
            dynamics=lambda x: obstacle_avoider.evaluate_weighted_dynamics(
                x, initial_dynamics.evaluate(x)
            ),
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
        )
        plot_obstacles(
            ax=ax, obstacle_container=obstacle_environment, x_lim=x_lim, y_lim=y_lim
        )

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=obstacle_environment,
            dynamics=obstacle_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
        )
        plot_obstacles(
            ax=ax, obstacle_container=obstacle_environment, x_lim=x_lim, y_lim=y_lim
        )

        # positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        # velocities = np.zeros(positions.shape)
        # for it in range(positions.shape[1]):
        #     velocities[:, it] = dynamical_system.evaluate(positions[:, it])

    # Close to obstacle
    pos = np.array([4.25, 3.25])
    init_vel = initial_dynamics.evaluate(pos)
    convergence_velocity = obstacle_avoider.evaluate_weighted_dynamics(pos, init_vel)

    center_vel = initial_dynamics.evaluate(obstacle_environment[0].center_position)
    assert np.allclose(convergence_velocity, center_vel)

    # Rotation to the right (due to linearization...)
    rotated_velocity = obstacle_avoider.evaluate(pos)
    assert rotated_velocity[0] > 0

    # Far away -> very close to initial
    pos = np.array([20.0, 1.0])
    init_vel = initial_dynamics.evaluate(pos)
    convergence_velocity = obstacle_avoider.evaluate_weighted_dynamics(pos, init_vel)
    assert convergence_velocity[0] > 0 and convergence_velocity[1] < 0

    rotated_velocity = obstacle_avoider.evaluate(pos)
    assert rotated_velocity[0] > 0 and rotated_velocity[1] < 0


if (__name__) == "__main__":
    # Import visualization libraries here
    import matplotlib.pyplot as plt  # For debugging only (!)

    plt.close("all")
    plt.ion()

    test_nonlinear_avoider(visualize=True)
