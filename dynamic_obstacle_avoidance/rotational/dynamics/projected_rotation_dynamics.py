""" Class to Deviate a DS based on an underlying obtacle.
"""
# %%
import sys
import math
import copy
import os

# from enum import Enum

import numpy as np
from numpy import linalg as LA
import warnings

from vartools.dynamical_systems import DynamicalSystem
from vartools.linalg import get_orthogonal_basis

# from vartools.linalg import get_orthogonal_basis
# from vartools.directional_space import get_angle_space

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from dynamic_obstacle_avoidance.rotational.rotational_avoidance import (
    obstacle_avoidance_rotational,
)

from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationXd
from dynamic_obstacle_avoidance.rotational.datatypes import Vector


class ProjectedRotationDynamics(DynamicalSystem):
    """
    A dynamical system which locally modulates

    Properties
    ----------
    obstacle: The obstacle around which shape the DS is deformed
    attractor_position: Position of the attractor
    center_direction: The direction of the DS at the center of the obstacle

    (Optional)
    min_gamma (> 1): The position at which the DS has 'maximum' rotation
    max_gamma (> min_gamma): The gamma-distance at which the influence stops.
    """

    # TODO: include Lyapunov function which checks avoidance.
    # this might effect the 'single-saddle point' on the surface'

    def __init__(
        self,
        obstacle: Obstacle,
        attractor_position: np.ndarray,
        reference_velocity: np.ndarray,
        min_gamma: float = 1,
        max_gamma: float = 10,
    ) -> None:
        super().__init__(dimension=obstacle.dimension)

        self.obstacle = obstacle
        self.attractor_position = attractor_position

        self.maximum_velocity = LA.norm(reference_velocity)
        if not self.maximum_velocity:
            raise ValueError("Zero velocity was obtained.")

        reference_velocity = reference_velocity / self.maximum_velocity

        attractor_dir = self.attractor_position - obstacle.center_position
        if not (attr_norm := LA.norm(attractor_dir)):
            warnings.warn("Obstacle is at attractor - zero deviation")
            return

        self.rotation = VectorRotationXd.from_directions(
            vec_init=attractor_dir / attr_norm, vec_rot=reference_velocity
        )
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        # Modify if needed
        self.attractor_influence = 3
        self.dotprod_projection_power = 2

        # self.base = get_orthogonal_basis()
        # self.deviation = get_angle_space(reference_velocity, null_matrix=self.base)

    def get_projected_gamma(self, position: Vector) -> float:
        # Get gamma
        gamma = self.obstacle.get_gamma(position, in_global_frame=True)
        if gamma >= self.max_gamma:
            return self.gamma_max

        elif gamma >= self.min_gamma:
            # Weight is additionally based on dot-product
            attractor_dir = self.attractor_position - position
            if dist_attractor := LA.norm(attractor_dir):
                attractor_dir = attractor_dir / dist_attractor
                dot_product = np.dot(attractor_dir, self.rotation.base0)
                gamma = gamma ** (2 / (dot_product + 1))

                if dist_obs := LA.norm(self.obstacle.center_position):
                    dist_stretching = LA.norm(position) / LA.norm(
                        self.obstacle.center_position
                    )
                    gamma = gamma**dist_stretching
                else:
                    gamma = self.gamma_max

            else:
                gamma = self.gamma_max

    def _get_position_after_deflating_obstacle(
        self, position: Vector, in_attractor_frame: bool = True
    ) -> Vector:
        """Returns position in the environment where the obstacle is shrunk."""
        radius = self.obstacle.get_local_radius(
            position, in_global_frame=in_attractor_frame
        )
        relative_position = position - self.obstacle.center_position
        pos_norm = LA.norm(relative_position)

        if pos_norm < radius:
            if in_attractor_frame:
                return np.copy(self.obstacle.center_position)
            else:
                return np.zeros_like(position)

        deflated_position = ((pos_norm - radius) / pos_norm) * relative_position

        if in_attractor_frame:
            return deflated_position + self.obstacle.center_position
        else:
            return deflated_position

    def _get_position_after_inflating_obstacle(
        self, position: Vector, in_attractor_frame: bool = True
    ) -> Vector:
        """Returns position in the environment where the obstacle is shrunk."""
        radius = self.obstacle.get_local_radius(
            position, in_global_frame=in_attractor_frame
        )

        relative_position = position - self.obstacle.center_position
        if not (pos_norm := LA.norm(relative_position)):
            # Needs a tiny value
            relative_position[0] = 1e6
            pos_norm = relative_position[0]

        inflated_position = ((pos_norm + radius) / pos_norm) * relative_position
        if in_attractor_frame:
            return inflated_position + self.obstacle.center_position
        else:
            return inflated_position

    def _get_folded_position_opposite_kernel_point(
        self, relative_position: Vector, relative_attractor: Vector
    ) -> Vector:
        """Returns the relative position folded with respect to the dynamics center."""

        # Copy just in case - but probably not needed
        relative_position = np.copy(relative_position)

        # 'Unfold' the circular plane into an infinite -y/+y-plane
        if not (dist_attr_obs := LA.norm(relative_attractor)):
            raise NotImplementedError("Implement for position at center.")
        dir_attractor_to_obstacle = relative_attractor / dist_attr_obs
        vec_attractor_to_position = relative_position # ?! this is position-obstacle?!

        basis = get_orthogonal_basis(dir_attractor_to_obstacle)
        transformed_position = basis.T @ vec_attractor_to_position

        # Stretch x-values along x-axis in order to have a x-weight at the attractor
        if dist_attr_pos := LA.norm(vec_attractor_to_position):
            transformed_position[0] = dist_attr_obs * math.log(
                dist_attr_pos / dist_attr_obs
            )

        else:
            transformed_position[0] = (-1) * sys.float_info.max

        vec_attractor_to_position = vec_attractor_to_position / dist_attr_pos
        dot_prod = np.dot(vec_attractor_to_position, dir_attractor_to_obstacle)

        if dot_prod <= -1:
            # Put it very far awa
            transformed_position[1] = sys.float_info.max

        elif dot_prod < 1:
            dotprod_factor = 2 / (1 + dot_prod) - 1
            dotprod_factor = dotprod_factor ** (1.0 / self.dotprod_projection_power)
            transformed_position[1:] = (
                transformed_position[1:]
                / LA.norm(transformed_position[1:])
                * dotprod_factor
            )
        transformed_position = basis @ transformed_position

        return transformed_position

    def _get_unfolded_position_opposite_kernel_point(
        self, transformed_position: Vector, relative_attractor: Vector
    ) -> Vector:
        """Returns UNfolded rleative position folded with respect to the dynamic center."""

        dir_attractor_to_obstacle = self.obstacle.center_position - relative_attractor
        if not (dist_attr_obs := LA.norm(dir_attractor_to_obstacle)):
            raise NotImplementedError()

        dir_attractor_to_obstacle = dir_attractor_to_obstacle / dist_attr_obs

        # Everything with resepect to attractor

        # Dot product is sufficient, as we only need first element.
        # Rotation is performed with VectorRotationXd
        vec_attractor_to_position = transformed_position - relative_attractor
        radius = np.dot(dir_attractor_to_obstacle, vec_attractor_to_position)

        # Ensure that the square root stays positive close to singularities
        transform_norm = LA.norm(vec_attractor_to_position)
        dot_prod = math.sqrt(max(transform_norm**2 - radius**2, 0))
        dot_prod = dot_prod**self.dotprod_projection_power
        dot_prod = 2.0 / (dot_prod + 1) - 1

        if dot_prod < 1:
            dir_perp = (
                dir_attractor_to_obstacle
                - vec_attractor_to_position / transform_norm * dot_prod
            )

            rotation_ = VectorRotationXd.from_directions(
                vec_init=dir_attractor_to_obstacle,
                vec_rot=dir_perp / LA.norm(dir_perp),
            )
            rotation_.rotation_angle = math.acos(dot_prod)

            # Initially a unit vector
            relative_position = rotation_.rotate(dir_attractor_to_obstacle)
        else:
            relative_position = dir_attractor_to_obstacle

        relative_position = (
            relative_position
            * math.exp((radius - dist_attr_obs) / dist_attr_obs)
            * dist_attr_obs
        )

        # Move from attractor-frame to obstacle-frame
        relative_position = relative_position + relative_attractor
        return relative_position

    def get_projected_point(self, position: Vector) -> Vector:
        """Projected point in 'linearized' environment

        Assumption of the point being outside of the obstacle."""

        # Do the evaluation in local frame
        relative_position = self.obstacle.transform_position_to_relative(position)
        relative_attractor = self.obstacle.transform_position_to_relative(
            self.attractor_position
        )

        if not (pos_norm := LA.norm(relative_position)):
            raise NotImplementedError()

        # Shrunk position
        relative_position = self._get_position_after_deflating_obstacle(
            relative_position
        )
        relative_attractor = self._get_position_after_deflating_obstacle(
            relative_attractor
        )

        relative_position = self._get_unfoled_opposite_kernel_point(
            relative_position, relative_attractor
        )

        relative_position = self._get_position_after_inflating_obstacle(
            relative_position
        )
        breakpoint()

        return self.obstacle.transform_position_from_relative(relative_position)

    def _get_lyapunov_gradient(self, position: Vector) -> Vector:
        """Returns the Gradient of the Lyapunov function.
        For now, we assume a quadratic Lyapunov function."""
        # Weight is additionally based on dot-product
        attractor_dir = self.attractor_position - position
        if not (dist_attractor := LA.norm(attractor_dir)):
            return np.zeros_like(position)

        attractor_dir = attractor_dir / dist_attractor
        return attractor_dir

    def _get_projected_lyapunov_gradient(self, position: Vector) -> Vector:
        """Returns projected lyapunov gradient function.

        It is assumed that z-axis is the base gradient."""
        attractor_dir = self.attractor_position - self.obstacle.center_position

        if not (dist_attractor := LA.norm(attractor_dir)):
            return np.zeros_like(attractor_dir)

        return attractor_dir / dist_attractor

    def _get_vector_rotation_of_modulation(
        self, position: Vector, velocity: Vector
    ) -> VectorRotationXd:
        """Returns the rotation of the modulation close to an obstacle."""
        if not (vel_norm := LA.norm(velocity)):
            return VectorRotationXd(np.eye(self.dimension, 2), rotation_angle=0.0)

        modulated_velocity = obstacle_avoidance_rotational(
            position,
            velocity,
            obstacle_list=[self.obstacle],
            convergence_velocity=velocity,
        )
        if not (mod_vel_norm := LA.norm(modulated_velocity)):
            return VectorRotationXd(np.eye(self.dimension, 2), rotation_angle=0.0)

        return VectorRotationXd.from_directions(
            velocity / vel_norm, modulated_velocity / mod_vel_norm
        )

    def evaluate(self, position: Vector) -> Vector:
        attractor_dir = self.get_lyapunov_gradient(position)

        dot_product = np.dot(attractor_dir, self.rotation.base0)

        weight = 1.0
        if not dot_product or not weight:
            return attractor_dir * min(dist_attractor, self.maximum_velocity)

        position = self.get_projected_point(position)

        global_rotation = VectorRotationXd.from_directions(
            self.rotation.base0, attractor_dir
        )
        local_rotation = global_rotation.rotate_vector_rotation(self.rotation)

        final_dir = local_rotation.rotate(attractor_dir, rot_factor=weight)
        velocity = final_dir * min(dist_attractor, self.maximum_velocity)

        return velocity


def get_environment_obstacle_top_right():
    # attractor_position = np.array([1.0, -1.0])
    attractor_position = np.array([0.0, 0.0])
    obstacle = Ellipse(
        # center_position=np.array([.0, 0.0]),
        center_position=np.array([3.0, 0.0]),
        axes_length=np.array([2, 3.0]),
        orientation=0 * math.pi / 180.0,
    )

    reference_velocity = np.array([2, 0.1])

    dynamics = ProjectedRotationDynamics(
        obstacle=obstacle,
        attractor_position=attractor_position,
        reference_velocity=reference_velocity,
    )

    return dynamics


def _test_base_gamma(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    visualize_vectors: bool = False,
    n_vectors: int = 8,
    save_figure: bool = False,
    **kwargs,
):
    # No explicit test in here, since only getting the gamma value.
    dynamics = get_environment_obstacle_top_right()

    if visualize:
        # x_lim = [-10, 10]
        # y_lim = [-10, 10]

        nx = ny = n_resolution

        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gammas = np.zeros(positions.shape[1])

        ### Basic Obstacle Transformation ###
        for pp in range(positions.shape[1]):
            gammas[pp] = dynamics.obstacle.get_gamma(
                positions[:, pp], in_global_frame=True
            )

        fig, ax = plt.subplots(figsize=figsize)
        cs = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas.reshape(nx, ny),
            cmap="binary_r",
            vmin=1.0,
            levels=np.linspace(1, 10, 9),
        )
        # cbar = fig.colorbar(cs, ticks=np.linspace(1, 11, 6))

        ax.plot(
            dynamics.attractor_position[0],
            dynamics.attractor_position[1],
            "*",
            color=kwargs["attractor_color"],
            linewidth=12,
            markeredgewidth=1.2,
            markersize=15,
            zorder=3,
        )

        # Opposite point
        ax.plot(
            [dynamics.attractor_position[0], x_lim[0]],
            [dynamics.attractor_position[1], dynamics.attractor_position[1]],
            "--",
            color=kwargs["opposite_color"],
            linewidth=3,
            zorder=2,
        )
        # ax.plot(
        #     dynamics.obstacle.center_position[0],
        #     dynamics.obstacle.center_position[1],
        #     "+",
        #     color=kwargs[]
        #     linewidth=12,
        #     markeredgewidth=3.0,
        #     markersize=14,
        #     zorder=3,
        # )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        plot_obstacles(
            ax=ax, obstacle_container=[dynamics.obstacle], alpha_obstacle=1.0
        )

        if visualize_vectors:
            # Plot the vectors
            nx = ny = n_vectors

            x_vals, y_vals = np.meshgrid(
                np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
            )
            positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

            for pp in range(positions.shape[1]):
                pos = dynamics._get_position_after_inflating_obstacle(positions[:, pp])
                velocity = dynamics._get_lyapunov_gradient(pos)
                ax.quiver(
                    pos[0],
                    pos[1],
                    velocity[0],
                    velocity[1],
                    color=kwargs["initial_color"],
                    scale=10.0,
                    width=0.01,
                    zorder=1,
                )

                velocity_rotation = dynamics._get_vector_rotation_of_modulation(
                    pos, velocity
                )

                velocity_mod = velocity_rotation.rotate(velocity)
                ax.quiver(
                    pos[0],
                    pos[1],
                    velocity_mod[0],
                    velocity_mod[1],
                    color=kwargs["final_color"],
                    scale=10.0,
                    width=0.01,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if save_figure:
                figure_name = "obstacle_original_space"
                fig.savefig(
                    os.path.join(
                        # os.path.dirname(__file__),
                        "figures",
                        figure_name + figtype,
                    ),
                    bbox_inches="tight",
                )


def test_obstacle_inflation(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    n_vectors: int = 8,
    save_figure: bool = False,
    **kwargs,
):
    dynamics = get_environment_obstacle_top_right()

    if visualize:
        # x_lim = [-10, 10]
        # y_lim = [-10, 10]
        nx = ny = n_resolution

        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gammas = np.zeros(positions.shape[1])

        ### Do before the trafo ###
        gammas_shrink = np.zeros_like(gammas)
        for pp in range(positions.shape[1]):
            # Do the reverse operation to obtain an 'even' grid
            pos_shrink = dynamics._get_position_after_inflating_obstacle(
                positions[:, pp]
            )
            gammas_shrink[pp] = dynamics.obstacle.get_gamma(
                pos_shrink, in_global_frame=True
            )

        # Transpose attractor
        attractor_position = dynamics._get_position_after_deflating_obstacle(
            dynamics.attractor_position
        )

        fig, ax = plt.subplots(figsize=figsize)
        cs = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas_shrink.reshape(nx, ny),
            cmap="binary_r",
            vmin=1.0,
            levels=np.linspace(1, 10, 9),
        )

        # cbar = fig.colorbar(cs, ticks=np.linspace(1, 11, 6))

        ax.plot(
            attractor_position[0],
            attractor_position[1],
            "*",
            color=kwargs["attractor_color"],
            linewidth=12,
            markeredgewidth=1.2,
            markersize=15,
            zorder=3,
        )

        ax.plot(
            dynamics.obstacle.center_position[0],
            dynamics.obstacle.center_position[1],
            "+",
            color=kwargs["obstacle_color"],
            linewidth=12,
            markeredgewidth=3.0,
            markersize=14,
            zorder=3,
        )

        # Opposite point
        ax.plot(
            [attractor_position[0], x_lim[0]],
            [attractor_position[1], dynamics.attractor_position[1]],
            "--",
            color=kwargs["opposite_color"],
            linewidth=3,
            zorder=2,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        if n_vectors:
            # Plot the vectors
            nx = ny = n_vectors

            x_vals, y_vals = np.meshgrid(
                np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
            )
            positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

            for pp in range(positions.shape[1]):
                pos = dynamics._get_position_after_inflating_obstacle(positions[:, pp])
                velocity = dynamics._get_lyapunov_gradient(pos)
                ax.quiver(
                    positions[0, pp],
                    positions[1, pp],
                    velocity[0],
                    velocity[1],
                    color=kwargs["initial_color"],
                    scale=10.0,
                    width=0.01,
                    zorder=1,
                )

                velocity_rotation = dynamics._get_vector_rotation_of_modulation(
                    pos, velocity
                )

                velocity_mod = velocity_rotation.rotate(velocity)
                ax.quiver(
                    positions[0, pp],
                    positions[1, pp],
                    velocity_mod[0],
                    velocity_mod[1],
                    color=kwargs["final_color"],
                    scale=10.0,
                    width=0.01,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if save_figure:
                figure_name = "obstacle_deflated_space"
                fig.savefig(
                    os.path.join(
                        # os.path.dirname(__file__),
                        "figures",
                        figure_name + figtype,
                    ),
                    bbox_inches="tight",
                )
        return fig, ax

    # Attractor is outside the obstacle
    position = np.array([0, 0])
    new_position = dynamics._get_position_after_inflating_obstacle(position)
    assert dynamics.obstacle.get_gamma(new_position, in_global_frame=True) > 1

    restored_position = dynamics._get_position_after_deflating_obstacle(new_position)
    assert np.allclose(position, restored_position)

    # Deflating close to the obstacle
    position = dynamics.obstacle.center_position + 1e-1
    deflated_position = dynamics._get_position_after_deflating_obstacle(position)
    assert np.allclose(deflated_position, dynamics.obstacle.center_position)

    # Position relatively close
    position = np.copy(dynamics.obstacle.center_position)
    position[0] = position[0] + 4
    new_position = dynamics._get_position_after_inflating_obstacle(position)
    restored_position = dynamics._get_position_after_deflating_obstacle(new_position)
    assert np.allclose(position, restored_position)


def test_inverse_projection_around_obstacle(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    n_vectors=10,
    save_figure=False,
    **kwargs,
):
    dynamics = get_environment_obstacle_top_right()

    if visualize:
        nx = ny = n_resolution

        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gammas = np.zeros(positions.shape[1])

        ### Do before trafo ###
        attractor_position = dynamics._get_position_after_deflating_obstacle(
            dynamics.attractor_position
        )

        gammas_shrink = np.zeros_like(gammas)
        for pp in range(positions.shape[1]):
            # Do the reverse operation to obtain an 'even' grid
            pos_shrink = dynamics._get_unfolded_position_opposite_kernel_point(
                positions[:, pp], attractor_position
            )
            pos_shrink = dynamics._get_position_after_inflating_obstacle(pos_shrink)
            gammas_shrink[pp] = dynamics.obstacle.get_gamma(
                pos_shrink, in_global_frame=True
            )

        fig, ax = plt.subplots(figsize=figsize)
        cs = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas_shrink.reshape(nx, ny),
            cmap="binary_r",
            vmin=1.0,
            levels=np.linspace(1, 10, 9),
        )

        # cbar = fig.colorbar(cs, ticks=np.linspace(1, 11, 6))

        # Attractor line
        ax.plot(
            [x_lim[0], x_lim[0]],
            y_lim,
            "--",
            color=kwargs["attractor_color"],
            linewidth=7,
            zorder=3,
        )
        # Split lines
        ax.plot(
            x_lim,
            [y_lim[0], y_lim[0]],
            "--",
            color=kwargs["opposite_color"],
            linewidth=7,
            zorder=3,
        )
        ax.plot(
            x_lim,
            [y_lim[1], y_lim[1]],
            "--",
            color=kwargs["opposite_color"],
            linewidth=7,
            zorder=3,
        )

        ax.plot(
            dynamics.obstacle.center_position[0],
            dynamics.obstacle.center_position[1],
            "+",
            color=kwargs["obstacle_color"],
            linewidth=12,
            markeredgewidth=3.0,
            markersize=14,
            zorder=3,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # Plot vectors
        if n_vectors:
            # plot the vectors
            nx = ny = n_vectors

            x_vals, y_vals = np.meshgrid(
                np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
            )
            positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
            for pp in range(positions.shape[1]):
                pos = dynamics._get_position_after_deflating_obstacle(positions[:, pp])
                velocity = dynamics._get_projected_lyapunov_gradient(pos)
                ax.quiver(
                    positions[0, pp],
                    positions[1, pp],
                    velocity[0],
                    velocity[1],
                    color=kwargs["initial_color"],
                    scale=10.0,
                    width=0.01,
                    zorder=1,
                )

                velocity_rotation = dynamics._get_vector_rotation_of_modulation(
                    pos, velocity
                )

                velocity_mod = velocity_rotation.rotate(velocity)
                ax.quiver(
                    positions[0, pp],
                    positions[1, pp],
                    velocity_mod[0],
                    velocity_mod[1],
                    color=kwargs["final_color"],
                    scale=10.0,
                    width=0.01,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if save_figure:
                figure_name = "obstacle_projection_deflation"
                fig.savefig(
                    os.path.join(
                        # os.path.dirname(__file__),
                        "figures",
                        figure_name + figtype,
                    ),
                    bbox_inches="tight",
                )

    attractor_position = dynamics._get_position_after_deflating_obstacle(
        dynamics.attractor_position
    )

    # center of the obstacle is the 'stable' point
    position = dynamics.obstacle.center_position
    pos_shrink = dynamics._get_unfolded_position_opposite_kernel_point(
        position, attractor_position
    )
    assert np.allclose(pos_shrink, position)

    # position very south of the obstacle gets projected to the attractor (and vice-versa)
    position_start = (
        dynamics.obstacle.center_position
        + (attractor_position - dynamics.obstacle.center_position) * 100
    )
    pos_shrink = dynamics._get_unfolded_position_opposite_kernel_point(
        position_start, attractor_position
    )
    assert np.allclose(pos_shrink, attractor_position)


def test_inverse_projection_and_deflation_around_obstacle(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    n_vectors=10,
    save_figure=False,
    **kwargs,
):
    dynamics = get_environment_obstacle_top_right()

    if visualize:
        nx = ny = n_resolution

        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gammas = np.zeros(positions.shape[1])

        ### do before trafo ###
        attractor_position = dynamics._get_position_after_deflating_obstacle(
            dynamics.attractor_position
        )

        gammas_shrink = np.zeros_like(gammas)
        for pp in range(positions.shape[1]):
            # do the reverse operation to obtain an 'even' grid
            pos_shrink = positions[:, pp]
            # pos_shrink = np.array([3.0, 1.0])
            pos_shrink = dynamics._get_position_after_deflating_obstacle(pos_shrink)

            if np.allclose(pos_shrink, dynamics.obstacle.center_position):
                gammas_shrink[pp] = 1
                continue

            pos_shrink = dynamics._get_unfolded_position_opposite_kernel_point(
                pos_shrink, attractor_position
            )
            pos_shrink = dynamics._get_position_after_inflating_obstacle(pos_shrink)

            gammas_shrink[pp] = dynamics.obstacle.get_gamma(
                pos_shrink, in_global_frame=True
            )

        fig, ax = plt.subplots(figsize=figsize)
        cs = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas_shrink.reshape(nx, ny),
            cmap="binary_r",
            vmin=1.0,
            levels=np.linspace(1, 10, 9),
            zorder=-1,
        )

        # cbar = fig.colorbar(cs, ticks=np.linspace(1, 11, 6))

        # attractor line
        ax.plot(
            [x_lim[0], x_lim[0]],
            y_lim,
            "--",
            color=kwargs["attractor_color"],
            linewidth=7,
            zorder=3,
        )
        # split lines
        ax.plot(
            x_lim,
            [y_lim[0], y_lim[0]],
            "--",
            color=kwargs["opposite_color"],
            linewidth=7,
            zorder=3,
        )
        ax.plot(
            x_lim,
            [y_lim[1], y_lim[1]],
            "--",
            color=kwargs["opposite_color"],
            linewidth=7,
            zorder=3,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        plot_obstacles(
            ax=ax, obstacle_container=[dynamics.obstacle], alpha_obstacle=1.0
        )

        # Plot vectors
        if n_vectors:
            # plot the vectors
            nx = ny = n_vectors

            x_vals, y_vals = np.meshgrid(
                np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
            )
            positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
            for pp in range(positions.shape[1]):
                pos = dynamics._get_position_after_inflating_obstacle(positions[:, pp])
                velocity = dynamics._get_projected_lyapunov_gradient(pos)
                ax.quiver(
                    pos[0],
                    pos[1],
                    velocity[0],
                    velocity[1],
                    color=kwargs["initial_color"],
                    scale=10.0,
                    width=0.01,
                )

                velocity_rotation = dynamics._get_vector_rotation_of_modulation(
                    pos, velocity
                )

                velocity_mod = velocity_rotation.rotate(velocity)
                ax.quiver(
                    pos[0],
                    pos[1],
                    velocity_mod[0],
                    velocity_mod[1],
                    color=kwargs["final_color"],
                    scale=10.0,
                    width=0.01,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if save_figure:
                figure_name = "obstacle_projection_inflated"
                fig.savefig(
                    os.path.join(
                        # os.path.dirname(__file__),
                        "figures",
                        figure_name + figtype,
                    ),
                    bbox_inches="tight",
                )


def test_obstacle_partially_rotated():
    # attractor_position = np.array([1.0, -1.0])
    attractor_position = np.array([0.0, 0.0])
    obstacle = Ellipse(
        center_position=np.array([-5.0, 4.0]),
        axes_length=np.array([2, 3.0]),
        orientation=0 * math.pi / 180.0,
    )

    reference_velocity = np.array([2, 0.1])

    dynamics = ProjectedRotationDynamics(
        obstacle=obstacle,
        attractor_position=attractor_position,
        reference_velocity=reference_velocity,
    )

    # test projection
    dist_surf = 1e-6
    pos_close_to_center = copy.deepcopy(dynamics.obstacle.center_position)
    pos_close_to_center[0] = pos_close_to_center[0] + 1 + dist_surf

    pos = dynamics._get_position_after_deflating_obstacle(pos_close_to_center)
    assert np.allclose(pos, dynamics.obstacle.center_position, atol=dist_surf / 2.0)


def test_obstacle_on_x_transformation():
    """tests if the folding / unfolding are bijective, i.e., same start and end point."""
    # simplified environment
    attractor_position = np.array([0.0, 0.0])
    obstacle = Ellipse(
        center_position=np.array([5.0, 0.0]),
        axes_length=np.array([2, 2.0]),
        orientation=0 * math.pi / 180.0,
    )

    reference_velocity = np.array([-1, 0.0])

    dynamics = ProjectedRotationDynamics(
        obstacle=obstacle,
        attractor_position=attractor_position,
        reference_velocity=reference_velocity,
    )

    # check that the folding / unfolding is bijective
    position = np.array([0, 5])
    relative_position = dynamics.obstacle.pose.transform_position_to_relative(position)
    relative_attr_pos = dynamics.obstacle.pose.transform_position_to_relative(
        attractor_position
    )

    trafo_pos = dynamics._get_folded_position_opposite_kernel_point(
        relative_position, relative_attractor=relative_attr_pos
    )
    assert np.allclose(trafo_pos, [0, 1])

    # breakpoint()
    reconstructed_pos = dynamics._get_unfolded_position_opposite_kernel_point(
        trafo_pos, relative_attractor=relative_attr_pos
    )

    assert np.allclose(relative_position, reconstructed_pos)


def test_transformation_bijection_for_rotated():
    # rotated obstacle
    relative_attr_pos = np.array([0.0, -4.0])
    obstacle = Ellipse(
        center_position=np.array([0.0, 0.0]),
        axes_length=np.array([1.0, 2.0]),
        orientation=30 * math.pi / 180.0,
    )

    reference_velocity = np.array([0, -1])
    dynamics = ProjectedRotationDynamics(
        obstacle=obstacle,
        attractor_position=np.zeros(2),
        reference_velocity=reference_velocity,
    )
    relative_position = np.array([-4.0, -4.0])

    trafo_pos = dynamics._get_folded_position_opposite_kernel_point(
        relative_position, relative_attractor=relative_attr_pos
    )
    assert np.allclose(trafo_pos, [-1, 0])

    reconstructed_pos = dynamics._get_unfolded_position_opposite_kernel_point(
        trafo_pos, relative_attractor=relative_attr_pos
    )
    assert np.allclose(relative_position, reconstructed_pos)


if (__name__) == "__main__":
    setup = {
        "attractor_color": "#db6e14",
        "opposite_color": "#96a83d",
        "obstacle_color": "#b35f5b",
        "initial_color": "#a430b3",
        "final_color": "#30a0b3",
        "figsize": (5, 4),
        "x_lim": [-3, 9],
        "y_lim": [-6, 6],
        "n_resolution": 100,
        "n_vectors": 8,
    }
    figtype = "png"

    import matplotlib.pyplot as plt

    plt.ion()
    plt.close("all")

    _test_base_gamma(visualize=True, visualize_vectors=True, save_figure=True, **setup)
    test_obstacle_inflation(visualize=True, **setup, save_figure=True)

    # test_obstacle_partially_rotated()
    # test_obstacle_on_x_transformation()
    # test_transformation_bijection_for_rotated()
    # test_transformation(visualize=False)

    # test_inverse_projection_around_obstacle(visualize=False)
    test_inverse_projection_around_obstacle(visualize=True, **setup, save_figure=True)

    test_inverse_projection_and_deflation_around_obstacle(
        visualize=1, **setup, save_figure=True
    )
    print("Tests done.")
