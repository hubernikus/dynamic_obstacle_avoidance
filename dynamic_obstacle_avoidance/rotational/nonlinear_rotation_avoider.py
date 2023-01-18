"""
Library for the Rotation (Modulation Imitation) of Linear Systems
"""
# Author: Lukas Huber
# GitHub: hubernikus
# Created: 2021-09-01

import warnings
import copy
import math
from functools import partial
from typing import Protocol, Optional

import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

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
from dynamic_obstacle_avoidance.obstacles import Obstacle

from dynamic_obstacle_avoidance.avoidance import BaseAvoider
from dynamic_obstacle_avoidance.rotational.rotational_avoider import (
    RotationalAvoider,
)
from dynamic_obstacle_avoidance.rotational.rotation_container import RotationContainer
from dynamic_obstacle_avoidance.rotational.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)


class ObstacleConvergenceDynamics(Protocol):
    def evaluate_convergence_around_obstacle(
        self, position: npt.ArrayLike, obstacle: Obstacle
    ) -> np.ndarray:
        ...

    def get_base_convergence(self, position: npt.ArrayLike) -> np.ndarray:
        ...


class LinearConvergenceDynamics:
    def __init__(
        self, convergence_dynamics: DynamicalSystem, initial_dynamics: DynamicalSystem
    ) -> None:
        self.initial_dynamics = initial_dynamics
        self.convergence_dynamics = convergence_dynamics

    def evaluate_convergence_around_obstacle(
        self, position: npt.ArrayLike, obstacle: Obstacle
    ) -> np.ndarray:
        return self.initial_dynamics.evaluate(
            # TODO: could also be reference point...
            obstacle.center_position
        )

    def get_base_convergence(self, position: npt.ArrayLike) -> np.ndarray:
        return self.convergence_dynamics.evaluate(position)


class NonlinearRotationalAvoider(BaseAvoider):
    """
    NonlinearRotationalAvoider -> Rotational Obstacle Avoidance by additionally considering initial dynamics
    """

    # TODO:
    #   - don't use UnitDirection (as I assume it has a large overhead)

    def __init__(
        self,
        initial_dynamics: DynamicalSystem,
        obstacle_environment: RotationContainer,
        obstacle_convergence: ObstacleConvergenceDynamics,
        **kwargs,
    ) -> None:
        """Initial dynamics, convergence direction and obstacle list are used."""
        self._rotation_avoider = RotationalAvoider(
            initial_dynamics=initial_dynamics,
            obstacle_environment=obstacle_environment,
            # convergence_system=convergence_system,
            cut_off_gamma=10,
            **kwargs,
        )

        self.obstacle_convergence = obstacle_convergence

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

    # def evaluate_convergence_dynamics(self, position: np.ndarray) -> np.ndarray:
    #     return self._rotation_avoider.convergence_dynamics.evaluate(position)

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

    def _compute_gamma_weights(self, position: np.ndarray):
        pass

    def evaluate_weighted_dynamics(
        self, position: np.ndarray, initial_velocity: np.ndarray
    ) -> np.ndarray:
        # convergence_velocity = self.evaluate_convergence_dynamics(position)
        convergence_velocity = self.obstacle_convergence.get_base_convergence(position)

        # TODO: this gamma/weight calculation could be shared...
        gamma_array = np.zeros((self.n_obstacles))
        for ii in range(self.n_obstacles):
            gamma_array[ii] = self._rotation_avoider.obstacle_environment[ii].get_gamma(
                position, in_global_frame=True
            )

        gamma_min = 1
        # Store weights -> mostly for visualization
        self.weights = np.zeros(self.n_obstacles)

        ind_obs = gamma_array <= gamma_min
        if sum_close := np.sum(ind_obs):
            # Dangerously close..
            weights = np.ones(sum_close) * 1.0 / sum_close
            weight_sum = 1

        else:
            ind_obs = gamma_array < self._rotation_avoider.cut_off_gamma

            if not np.sum(ind_obs):
                return initial_velocity

            weights = 1.0 / (gamma_array[ind_obs] - gamma_min) - 1 / (
                self.cut_off_gamma - gamma_min
            )
            if (weight_sum := np.sum(weights)) > 0:
                # Normalize weight, but leave possibility to be smaller than one (!)
                weights = weights / weight_sum

            # Influence of each obstacle -> but better mapping to [0, 1]
            ww_weights = (
                1 / gamma_array[ind_obs] - 1 / self._rotation_avoider.cut_off_gamma
            )
            ww_weights = ww_weights / (1 - 1 / self._rotation_avoider.cut_off_gamma)
            weights = weights * np.minimum(1, ww_weights)

        self.weights[ind_obs] = weights

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
            # local_velocities[:, ii] = self.evaluate_initial_dynamics(
            #     # TODO: could also be reference point...
            #     self._rotation_avoider.obstacle_environment[ii].center_position
            # )

            local_velocities[
                :, ii
            ] = self.obstacle_convergence.evaluate_convergence_around_obstacle(
                position, obstacle=self._rotation_avoider.obstacle_environment[ii]
            )

            if not LA.norm(local_velocities[:, ii]):
                # What should be done here (?)
                # <-> smoothly reduce the weight as we approach the center(?)
                raise NotImplementedError()

        if not (convergence_norm := LA.norm(convergence_velocity)):
            return convergence_velocity
        convergence_velocity = convergence_velocity / convergence_norm

        # Weighted sum -> should have the same result as 'the graph summing' (but current implementation is more stable)
        averaged_direction = get_directional_weighted_sum(
            null_direction=convergence_velocity,
            weights=weights,
            directions=local_velocities,
        )

        return initial_norm * averaged_direction


def test_nonlinear_avoider(
    visualize: bool = False,
    savefig: bool = False,
    n_resolution: int = 20,
) -> None:
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

    obstacle_convergence = LinearConvergenceDynamics(
        initial_dynamics=initial_dynamics, convergence_dynamics=convergence_dynamics
    )

    obstacle_avoider = NonlinearRotationalAvoider(
        initial_dynamics=initial_dynamics,
        # convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=obstacle_convergence,
    )

    if visualize:
        x_lim = [0, 15]
        y_lim = [-2, 5]

        do_quiver = False
        vf_color = "blue"
        # vf_color = "black"

        figsize = (6.5, 3.0)
        from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
            plot_obstacle_dynamics,
        )
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=[],
            # dynamics=obstacle_avoider.evaluate_convergence_dynamics,
            dynamics=convergence_dynamics.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            do_quiver=do_quiver,
            vectorfield_color=vf_color,
        )
        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            noTicks=True,
            # show_ticks=False,
        )
        # fig.tight_layout()

        if savefig:
            figname = "base_convergence"
            plt.savefig(
                "figures/" + "nonlinear_infinite_dynamics_" + figname + figtype,
                bbox_inches="tight",
            )

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=obstacle_avoider.evaluate_initial_dynamics,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=do_quiver,
            n_grid=n_resolution,
            vectorfield_color=vf_color,
        )
        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            noTicks=True,
            alpha_obstacle=0.6,
        )
        # ax.plot(x_lim, [0, 0], "--", color="#696969", linewidth=2)
        ax.plot(x_lim, [0, 0], "--", color="#5F021F", linewidth=4)
        if savefig:
            figname = "initial"
            plt.savefig(
                "figures/" + "nonlinear_infinite_dynamics_" + figname + figtype,
                bbox_inches="tight",
            )

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=lambda x: obstacle_avoider.evaluate_weighted_dynamics(
                x, initial_dynamics.evaluate(x)
            ),
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            show_ticks=False,
            do_quiver=do_quiver,
            n_grid=n_resolution,
            vectorfield_color=vf_color,
        )

        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            noTicks=True,
            alpha_obstacle=0.0,
        )

        n_grid = n_resolution
        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_grid),
            np.linspace(y_lim[0], y_lim[1], n_grid),
        )
        positions = np.array([xx.flatten(), yy.flatten()])
        weights = np.zeros(positions.shape[1])

        for pp in range(positions.shape[1]):
            obstacle_avoider.evaluate_weighted_dynamics(
                positions[:, pp], initial_dynamics.evaluate(positions[:, pp])
            )
            weights[pp] = obstacle_avoider.weights[0]

        cf = ax.contourf(
            xx,
            yy,
            weights.reshape(n_grid, n_grid),
            # cmap="hot_r",
            cmap="binary",
            levels=np.linspace(0, 1, 6),
            zorder=-3,
            alpha=0.5,
        )

        cax = fig.add_axes([0.6, 0.75, 0.28, 0.05])
        # cax = fig.add_axes([0.6, 0.8, 0.2, 0.05])
        cticks = [0, 1]
        cbar = fig.colorbar(cf, cax=cax, orientation="horizontal", ticks=cticks)
        cbar.ax.set_xticklabels(
            labels=cticks,
            # weight="bold",
            fontsize=12,
        )
        # cbar.ax.set_xticklabels(labels=cbar.ax.get_yticklabels(), weight="bold")

        if savefig:
            figname = "convergence"
            plt.savefig(
                "figures/" + "nonlinear_infinite_dynamics_" + figname + figtype,
                bbox_inches="tight",
            )

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=obstacle_environment,
            dynamics=obstacle_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=do_quiver,
            n_grid=n_resolution,
            vectorfield_color=vf_color,
        )
        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            noTicks=True,
        )

        if savefig:
            figname = "avoidance"
            plt.savefig(
                "figures/" + "nonlinear_infinite_dynamics_" + figname + figtype,
                bbox_inches="tight",
            )

    # Close to obstacle
    pos = np.array([4.25, 3.25])
    init_vel = initial_dynamics.evaluate(pos)
    convergence_velocity = obstacle_avoider.evaluate_weighted_dynamics(pos, init_vel)

    # center_vel = initial_dynamics.evaluate(obstacle_environment[0].center_position)
    # assert np.allclose(convergence_velocity, center_vel)

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


def test_multiobstacle_nonlinear_avoider(visualize=True):
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

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([15, -0.1]),
            axes_length=np.array([1.1, 1.1]),
            margin_absolut=0.9,
        )
    )

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([3, -4]),
            axes_length=np.array([1.1, 1.1]),
            margin_absolut=0.9,
        )
    )

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([10, -3]),
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
        x_lim = [-2, 22]
        y_lim = [-5, 5]

        figsize = (12, 6)

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=obstacle_environment,
            dynamics=obstacle_avoider.evaluate_convergence_dynamics,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
        )
        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
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

    # Velocity on the surface is perpendicular to circle-center
    dir_surf = np.array([-1, -1]) / math.sqrt(2)
    position = obstacle_environment[-1].center_position + dir_surf * 1.001
    velocity = obstacle_avoider.evaluate(position)

    assert math.isclose(np.dot(dir_surf, velocity), 0, abs_tol=1e-4)


def test_circular_single_obstacle(visualize=False):
    from vartools.dynamical_systems import CircularStable

    circular_ds = CircularStable(radius=2.5, maximum_velocity=2.0)

    obstacle_environment = RotationContainer()
    obstacle_environment.append(
        Ellipse(
            center_position=np.array([2.5, 0.0]),
            axes_length=np.array([1.4, 1.4]),
            margin_absolut=0.3,
        )
    )

    rotation_projector = ProjectedRotationDynamics(
        attractor_position=circular_ds.pose.position,
        initial_dynamics=circular_ds,
        reference_velocity=lambda x: x - center_velocity.center_position,
    )

    if visualize:
        x_lim = [-4, 4]
        y_lim = [-4, 4]
        n_grid = 15

        figsize = (4.0, 3.5)

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=circular_ds.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_grid,
        )
        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            alpha_obstacle=1.0,
            noTicks=True,
        )

        linearised_ds = partial(
            rotation_projector.evaluate_convergence_around_obstacle,
            obstacle=obstacle_environment[0],
        )
        # Linearized Dynamics
        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=linearised_ds,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_grid,
            show_ticks=False,
        )
        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            alpha_obstacle=0.0,
            noTicks=True,
        )

        # plot_obstacles(
        #     ax=ax,
        #     obstacle_container=obstacle_environment,
        #     x_lim=x_lim,
        #     y_lim=y_lim,
        # )

    rotation_projector.obstacle = obstacle_environment[0]

    # Position within obstacle
    obstacle = obstacle_environment[0]
    position = obstacle.center_position + obstacle.axes_length * 0.4
    projected_position = rotation_projector.get_projected_position(position)
    center_velocity = circular_ds.evaluate(obstacle_environment[0].center_position)
    velocity = rotation_projector.evaluate_convergence_around_obstacle(
        position, obstacle=obstacle_environment[0]
    )
    assert np.allclose(velocity, center_velocity)

    # Position close to opposite
    position = np.array([-2, 1e-3])
    projected_position = rotation_projector.get_projected_position(position)

    velocity = rotation_projector.evaluate_convergence_around_obstacle(
        position=position,
        obstacle=obstacle_environment[0],
    )

    initial_velocity = circular_ds.evaluate(position)
    assert np.allclose(
        initial_velocity, velocity, atol=1e-3
    ), "No influence (close) opposite."

    # There is still some influence at a specific position
    position = np.array([-1.5, 6])
    velocity = rotation_projector.evaluate_convergence_around_obstacle(
        position=position,
        obstacle=obstacle_environment[0],
    )
    initial_velocity = circular_ds.evaluate(position)
    assert not np.allclose(initial_velocity, velocity)

    # Opposite of attractor
    position = np.array([-1, 0])
    velocity = rotation_projector.evaluate_convergence_around_obstacle(
        position=position,
        obstacle=obstacle_environment[0],
    )
    initial_velocity = circular_ds.evaluate(position)
    assert np.allclose(velocity, initial_velocity)

    # At the top - since the rotation at the center is equal to the obstacle's
    # the modulated is equal to the initial
    position = obstacle.center_position
    position = np.array([position[1], position[0]])

    initial_velocity = circular_ds.evaluate(position)
    velocity = rotation_projector.evaluate_convergence_around_obstacle(
        position=position,
        obstacle=obstacle_environment[0],
    )

    assert velocity[0] < -1, "Circle value is wrong"
    assert velocity[1] > 0, "The linearization should move outwards."
    assert abs(velocity[0]) > abs(velocity[1]), "Not circling enough"

    # Evaluation at center of obstacle
    velocity = rotation_projector.evaluate_convergence_around_obstacle(
        position=obstacle_environment[0].center_position,
        obstacle=obstacle_environment[0],
    )
    initial_velocity = circular_ds.evaluate(obstacle_environment[0].center_position)
    assert np.allclose(velocity, initial_velocity)

    # Position just on the surface
    position = copy.deepcopy(obstacle_environment[0].center_position)
    position[1] = (
        position[1]
        + obstacle_environment[0].axes_length[1] / 2.0
        + obstacle_environment[0].margin_absolut
        + 1e-3
    )
    velocity = rotation_projector.evaluate_convergence_around_obstacle(
        position=position,
        obstacle=obstacle_environment[0],
    )
    assert np.allclose(velocity, initial_velocity, atol=1e-2)


def test_circular_multiple(
    visualize=False,
    n_resolution: int = 20,
    savefig: bool = False,
):
    from vartools.dynamical_systems import CircularStable

    circular_ds = CircularStable(radius=2.5, maximum_velocity=2.0)

    obstacle_environment = RotationContainer()
    obstacle_environment.append(
        Ellipse(
            center_position=np.array([-1.0, 2.0]),
            axes_length=np.array([1.0, 1.0]),
            # margin_absolut=0.3,
        )
    )

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([1.0, 2.0]),
            # axes_length=np.array([1.4, 1.4]),
            axes_length=np.array([1.0, 1.0]),
            # margin_absolut=0.3,
        )
    )

    rotation_projector = ProjectedRotationDynamics(
        attractor_position=circular_ds.pose.position,
        initial_dynamics=circular_ds,
        reference_velocity=lambda x: x - center_velocity.center_position,
    )

    obstacle_avoider = NonlinearRotationalAvoider(
        initial_dynamics=circular_ds,
        # convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=rotation_projector,
    )

    if visualize:
        x_lim = [-3.5, 3.5]
        y_lim = [-2.8, 2.8]

        vf_color = "blue"
        figname = "nonlinear_infinite_dynamics"
        # vf_color = "black"

        figsize = (8.0, 8.0)

        from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
            plot_obstacle_dynamics,
        )
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=lambda x: obstacle_avoider.evaluate_weighted_dynamics(
                x, circular_ds.evaluate(x)
            ),
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            do_quiver=True,
            vectorfield_color=vf_color,
        )
        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            # noTicks=True,
            # show_ticks=False,
        )
        # fig.tight_layout()

        if savefig:
            figspec = "base_convergence"
            plt.savefig(
                "figures/" + figname + "_" + figspec + figtype,
                bbox_inches="tight",
            )

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=obstacle_environment,
            # dynamics=obstacle_avoider.evaluate_convergence_dynamics,
            dynamics=obstacle_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            do_quiver=True,
            vectorfield_color=vf_color,
        )
        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            # noTicks=True,
            # show_ticks=False,
        )
        # fig.tight_layout()

        if savefig:
            figspec = "full_avoidance"
            plt.savefig(
                "figures/" + figname + "_" + figspec + figtype,
                bbox_inches="tight",
            )


if (__name__) == "__main__":
    # Import visualization libraries here
    import matplotlib.pyplot as plt  # For debugging only (!)
    from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
        plot_obstacle_dynamics,
    )
    from dynamic_obstacle_avoidance.visualization import plot_obstacles

    # plt.close("all")
    plt.ion()

    figtype = ".pdf"
    # figtype = ".png"

    # test_nonlinear_avoider(visualize=True, savefig=False)
    # test_nonlinear_avoider(visualize=True, savefig=False, n_resolution=10)
    # test_multiobstacle_nonlinear_avoider(visualize=False)

    # test_circular_single_obstacle(visualize=True)
    test_circular_multiple(visualize=True, n_resolution=20)
