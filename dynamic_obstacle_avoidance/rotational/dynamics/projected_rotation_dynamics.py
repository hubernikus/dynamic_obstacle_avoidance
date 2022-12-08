"""
Class to Deviate a DS based on an underlying obtacle.
"""

import math
import copy

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
        self, relative_position: Vector
    ) -> Vector:
        """Returns position in the environment where the obstacle is shrunk."""
        radius = self.obstacle.get_local_radius(
            relative_position, in_global_frame=False
        )
        pos_norm = LA.norm(relative_position)

        if pos_norm < radius:
            return np.zeros_like(relative_position)

        return ((pos_norm - radius) / pos_norm) * relative_position

    def _get_position_after_inflating_obstacle(
        self, relative_position: Vector
    ) -> Vector:
        """Returns position in the environment where the obstacle is shrunk."""
        radius = self.obstacle.get_local_radius(
            relative_position, in_global_frame=False
        )

        if not (pos_norm := LA.norm(relative_position)):
            # Needs a tiny value
            relative_position[0] = 1e6
            pos_norm = relative_position[0]

        return ((pos_norm + radius) / pos_norm) * relative_position

    def _get_folded_position_opposite_kernel_point(
        self, relative_position: Vector, relative_attractor: Vector
    ) -> Vector:
        """Returns the relative position folded with respect to the dynamics center."""

        # Copy just in case - but probably no needed
        relative_position = np.copy(relative_position)

        # 'Unfold' the circular plane into an infinite yy-plane
        # basis = get_orthogonal_basis(relative_position / LA.norm(relative_position))
        if not (dist_attr_obs := LA.norm(relative_attractor)):
            raise NotImplementedError("Implement for position at center.")
        dir_attractor_to_obstacle = (-1) * relative_attractor / dist_attr_obs

        basis = get_orthogonal_basis(dir_attractor_to_obstacle)
        # transformed_position = np.zeros_like(relative_position)
        # transformed_position[1:] = basis[:, 1:].T @ relative_position
        transformed_position = basis.T @ relative_position

        # Stretch x-values along x-axis in order to have a x-weight at the attractor
        vec_attractor_to_position = relative_position - relative_attractor
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
            transformed_position[1:] = (
                relative_position[1:]
                / LA.norm(relative_position[1:])
                * (2 / (1 + dot_prod) - 1)
            )
        transformed_position[1:] = basis[1:, :] @ transformed_position
        return transformed_position

    def _get_unfolded_position_opposite_kernel_point(
        self, transformed_position: Vector, relative_attractor: Vector
    ) -> Vector:
        """Returns UNfolded rleative position folded with respect to the dynamic center."""
        # basis = get_orthogonal_basis(
        #     transformed_position / LA.norm(transformed_position)
        # )

        if not (dist_attr_obs := LA.norm(relative_attractor)):
            raise NotImplementedError()

        dir_attractor_to_obstacle = (-1) * relative_attractor / dist_attr_obs
        vec_attractor_to_position = transformed_position - relative_attractor
        # relative_position = basis @ (-1) * relative_attractor

        # Dot product is sufficient, as we only need first element.
        # Rotation is performed with VectorRotationXd
        radius = np.dot(dir_attractor_to_obstacle, transformed_position)
        dot_prod = math.sqrt(1 - radius**2)

        # relative_position = np.zeros_like(transformed_position[1:])
        dot_prod = 2.0 / (dot_prod + 1) - 1

        if not (dist_attr_pos := LA.norm(vec_attractor_to_position)):
            breakpoint()

        rotation_ = VectorRotationXd.from_directions(
            vec_init=dir_attractor_to_obstacle,
            vec_rot=vec_attractor_to_position / dist_attr_pos,
        )
        rotation_.rotation_angle = math.acos(dot_prod)

        # Initially a unit vector
        relative_position = rotation_.rotate(dir_attractor_to_obstacle)
        relative_position = (
            relative_position * math.exp(radius / dist_attr_obs) * dist_attr_obs
        )

        # Move from attractor-frame to obstacle-frame
        relative_position = relative_position + relative_attractor
        # relative_position[1:] = basis[1:, :].T @ relative_position

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

    def get_lyapunov_gradient(self, position: Vector) -> Vector:
        """Returns the Gradient of the Lyapunov function.
        For now, we assume a quadratic Lyapunov function."""
        # Weight is additionally based on dot-product
        attractor_dir = self.attractor_position - position
        if not (dist_attractor := LA.norm(attractor_dir)):
            return np.zeros_like(position)

        attractor_dir = attractor_dir / dist_attractor

    def evaluate(self, position: Vector) -> Vector:
        attractor_dir = self.get_lyapunov_gradient(position)

        dot_product = np.dot(attractor_dir, self.rotation.base0)

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


def test_transformation(visualize=False):
    attractor_position = np.array([1.0, -1.0])
    obstacle = Ellipse(
        center_position=np.array([-3.0, 2.0]),
        axes_length=np.array([2, 3.0]),
        orientation=0 * math.pi / 180.0,
    )

    reference_velocity = np.array([2, 0.1])

    dynamics = ProjectedRotationDynamics(
        obstacle=obstacle,
        attractor_position=attractor_position,
        reference_velocity=reference_velocity,
    )

    if visualize:
        import matplotlib.pyplot as plt

        plt.close("all")

        # x_lim = [-10, 10]
        # y_lim = [-10, 10]
        x_lim = [-6, 6]
        y_lim = [-6, 6]
        n_resolution = 30
        figsize = (5, 4)

        nx = ny = n_resolution

        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gammas = np.zeros(positions.shape[1])

        for pp in range(positions.shape[1]):
            # pos_rot = dynamics._get_position_after_shrinking_obstacle(pos)
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

        cbar = fig.colorbar(cs, ticks=np.linspace(1, 11, 6))

        ax.plot(
            dynamics.attractor_position[0],
            dynamics.attractor_position[1],
            "k*",
            linewidth=12,
            markeredgewidth=1.2,
            markersize=15,
            zorder=3,
        )
        ax.plot(
            dynamics.obstacle.center_position[0],
            dynamics.obstacle.center_position[1],
            "k+",
            linewidth=12,
            markeredgewidth=3.0,
            markersize=14,
            zorder=3,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # Do before the trafo
        gammas_shrink = np.zeros_like(gammas)
        for pp in range(positions.shape[1]):
            # Do the reverse operation to obtain an 'even' grid
            pos_shrink = dynamics.obstacle.pose.transform_position_to_relative(
                positions[:, pp]
            )
            pos_shrink = dynamics._get_position_after_inflating_obstacle(pos_shrink)
            gammas_shrink[pp] = dynamics.obstacle.get_gamma(pos_shrink)

        # Transpose attractor
        attractor_position = dynamics.obstacle.pose.transform_position_to_relative(
            dynamics.attractor_position
        )
        attractor_position = dynamics._get_position_after_deflating_obstacle(
            attractor_position
        )
        attractor_position = dynamics.obstacle.pose.transform_position_from_relative(
            attractor_position
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

        cbar = fig.colorbar(cs, ticks=np.linspace(1, 11, 6))

        ax.plot(
            attractor_position[0],
            attractor_position[1],
            "k*",
            linewidth=12,
            markeredgewidth=1.2,
            markersize=15,
            zorder=3,
        )

        ax.plot(
            dynamics.obstacle.center_position[0],
            dynamics.obstacle.center_position[1],
            "k+",
            linewidth=12,
            markeredgewidth=3.0,
            markersize=14,
            zorder=3,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    # Test Projection
    dist_surf = 1e-6
    pos_close_to_center = copy.deepcopy(dynamics.obstacle.center_position)
    pos_close_to_center[0] = pos_close_to_center[0] + 1 + dist_surf

    pos = dynamics.obstacle.pose.transform_position_to_relative(pos_close_to_center)
    # assert np.allclose(pos, np.zeros_like(pos))
    pos = dynamics._get_position_after_deflating_obstacle(pos)
    pos = dynamics.obstacle.pose.transform_position_from_relative(pos)
    assert np.allclose(pos, dynamics.obstacle.center_position, atol=dist_surf / 2.0)


def test_obstacle_on_x_transformation():
    """Tests if the folding / unfolding are bijective, i.e., same start and end point."""
    # Simplified environment
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

    # Check that the folding / unfolding is bijective
    position = np.array([0, 5])
    relative_position = dynamics.obstacle.pose.transform_position_to_relative(position)
    relative_attr_pos = dynamics.obstacle.pose.transform_position_to_relative(
        attractor_position
    )

    trafo_pos = dynamics._get_folded_position_opposite_kernel_point(
        relative_position, relative_attractor=relative_attr_pos
    )
    assert np.allclose(trafo_pos, [0, 1])

    reconstructed_pos = dynamics._get_unfolded_position_opposite_kernel_point(
        trafo_pos, relative_attractor=relative_attr_pos
    )
    assert np.allclose(relative_position, reconstructed_pos)


if (__name__) == "__main__":

    # test_transformation(visualize=True)
    test_obstacle_on_x_transformation()
    # test_transformation(visualize=False)
