"""
Class to Deviate a DS based on an underlying obtacle.
"""

import math

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


class LocallyRotatedFromObtacle(DynamicalSystem):
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

    # def get_projectd_gamma(self, position: Vector) -> float:
    #     # Get gamma
    #     gamma = self.obstacle.get_gamma(position, in_global_frame=True)
    #     if gamma >= self.max_gamma:
    #         return self.gamma_max

    #     elif gamma >= self.min_gamma:
    #         # Weight is additionally based on dot-product
    #         attractor_dir = self.attractor_position - position
    #         if dist_attractor := LA.norm(attractor_dir):
    #             attractor_dir = attractor_dir / dist_attractor
    #             dot_product = np.dot(attractor_dir, self.rotation.base0)
    #             gamma = gamma ** (2 / (dot_product + 1))

    #             if dist_obs := LA.norm(self.obstacle.center_position):
    #                 dist_stretching = LA.norm(position) / LA.norm(
    #                     self.obstacle.center_position
    #                 )
    #                 gamma = gamma**dist_stretching
    #             else:
    #                 gamma = self.gamma_max

    #         else:
    #             gamma = self.gamma_max

    # def _get_position_after_shrinking_obstacle(
    #     self, relative_position: Vector
    # ) -> Vector:
    #     """Returns position in the environment where the obstacle is shrunk."""
    #     radius = self.obstacle.get_local_radius(
    #         relative_position, in_global_frame=False
    #     )
    #     pos_norm = LA.norm(relative_position)

    #     if pos_norm < radius:
    #         return np.zeros_like(relative_position)

    #     return ((pos_norm - radius) / pos_norm) * relative_position

    # def _get_position_after_inflating_obstacle(
    #     self, relative_position: Vector
    # ) -> Vector:
    #     """Returns position in the environment where the obstacle is shrunk."""
    #     radius = self.obstacle.get_local_radius(
    #         relative_position, in_global_frame=False
    #     )

    #     if not (pos_norm := LA.norm(relative_position)):
    #         # Needs a tiny value
    #         relative_position[0] = 1e6
    #         pos_norm = relative_position[0]

    #     return ((pos_norm + radius) / pos_norm) * relative_position

    # def _get_unfoled_opposite_kernel_point(
    #     self, relative_position: Vector, relative_attractor: Vector
    # ) -> Vector:
    #     """Returns the relative_position with respect to the relative center"""

    #     if not (attr_norm := LA.norm(relative_attractor)):
    #         raise NotImplementedError("Implement for position at center.")

    #     transformed_position = np.zeros_like(relative_position)

    #     # Stretch x-values along x-axis in order to have a x-weight at the attractor
    #     if dist_attractor_to_position := LA.norm(
    #         relative_position - relative_attractor
    #     ):
    #         relative_position[0] = math.log(dist_attractor_to_position / attr_norm)

    #     else:
    #         transformed_position[0] = (-1) * sys.float_info.max

    #     # 'Unfold' the circular plane into an infinite yy-plane
    #     # basis = get_orthogonal_basis(relative_position / LA.norm(relative_position))

    #     # relative_position[1:] = (basis.T @ (relative_position - relative_attractor))[1:]
    #     dot_prod = np.dot(relative_position, relative_attractor)

    #     if dot_prod == -1:
    #         transformed_position[1:] = relative_position[1:] * sys.float_info.max
    #     else:
    #         transformed_position[1:] = (
    #             relative_position[1:]
    #             / LA.norm(relative_position[1:])
    #             * (2 / (1 + dot_prod) - 1)
    #         )

    #     return relative_position

    # def get_projected_point(self, position: Vector) -> Vector:
    #     """Projected point in 'linearized' environment

    #     Assumption of the point being outside of the obstacle."""

    #     # Do the evaluation in local frame
    #     relative_position = self.obstacle.transform_position_to_relative(position)
    #     relative_attractor = self.obstacle.transform_position_to_relative(
    #         self.attractor_position
    #     )

    #     if not (pos_norm := LA.norm(relative_position)):
    #         raise NotImplementedError()

    #     # Shrunk position
    #     relative_position = self._get_position_after_shrinking_obstacle(
    #         relative_position
    #     )
    #     relative_attractor = self._get_position_after_shrinking_obstacle(
    #         relative_attractor
    #     )

    #     relative_position = self._get_unfoled_opposite_kernel_point(
    #         relative_position, relative_attractor
    #     )

    #     relative_position = self._get_position_after_inflating_obstacle(
    #         relative_position
    #     )

    #     return self.obstacle.transform_position_from_relative(relative_position)

    def evaluate(self, position: Vector) -> Vector:
        # Weight is based on gamma
        gamma = self.obstacle.get_gamma(position, in_global_frame=True)
        if gamma <= self.min_gamma:
            weight = 1
        elif gamma > self.max_gamma:
            weight = 0
        else:
            weight = (self.max_gamma - gamma) / (self.max_gamma - self.min_gamma)

        # Weight is additionally based on dot-product
        attractor_dir = self.attractor_position - position
        if not (dist_attractor := LA.norm(attractor_dir)):
            return np.zeros_like(position)

        attractor_dir = attractor_dir / dist_attractor
        dot_product = np.dot(attractor_dir, self.rotation.base0)

        if not dot_product or not weight:
            return attractor_dir * min(dist_attractor, self.maximum_velocity)

        # And attractor
        if weight < 1:
            tmp_weight = 1.0 / (1 - weight)
            attr_weight = max(self.attractor_influence / dist_attractor, 0)

            attr_weight = tmp_weight / (tmp_weight + attr_weight)
        else:
            attr_weight = 1

        weight = weight ** (2.0 / (1 + dot_product)) * attr_weight

        global_rotation = VectorRotationXd.from_directions(
            self.rotation.base0, attractor_dir
        )
        local_rotation = global_rotation.rotate_vector_rotation(self.rotation)

        final_dir = local_rotation.rotate(attractor_dir, rot_factor=weight)
        velocity = final_dir * min(dist_attractor, self.maximum_velocity)

        return velocity


def plot_obstacle_of_dynamics(dynamical_system: LocallyRotatedFromObtacle, ax):
    """2D plotting"""
    boundary_points = np.array(dynamical_system.obstacle.get_boundary_xy())
    ax.plot(
        boundary_points[0, :],
        boundary_points[1, :],
        color="black",
        linestyle="--",
        zorder=3,
        linewidth=2,
    )

    global_ref = dynamical_system.obstacle.get_reference_point(in_global_frame=True)
    ax.plot(
        global_ref[0],
        global_ref[1],
        "k+",
        linewidth=12,
        markeredgewidth=2.4,
        markersize=8,
        zorder=3,
    )

    ax.plot(
        dynamical_system.obstacle.center_position[0],
        dynamical_system.obstacle.center_position[1],
        "ko",
        linewidth=12,
        markeredgewidth=2.4,
        markersize=8,
        zorder=3,
    )

    ax.plot(
        dynamical_system.attractor_position[0],
        dynamical_system.attractor_position[1],
        "k*",
        linewidth=12,
        markeredgewidth=1.2,
        markersize=15,
        zorder=3,
    )


def test_ellipse_ds(visualize=False):
    import math

    # fig, ax = plt.subplots()
    attractor_position = np.array([1, -1])
    obstacle = Ellipse(
        center_position=np.array([5, 4]),
        axes_length=np.array([4, 6]),
        orientation=90 * math.pi / 180.0,
    )

    reference_velocity = np.array([2, 0.1])
    local_ds = LocallyRotatedFromObtacle(
        obstacle=obstacle,
        attractor_position=attractor_position,
        reference_velocity=reference_velocity,
    )

    # Test opposite the obstacle-center
    position = np.array([0, 0])
    opposite_ds = local_ds.evaluate(position)
    dir_attr = attractor_position - position
    dot_prod = np.dot(opposite_ds, dir_attr) / (
        LA.norm(opposite_ds) * LA.norm(dir_attr)
    )

    # Same sign and angle in the same direction
    assert local_ds.rotation.rotation_angle * dot_prod > 0
    assert np.arccos(dot_prod) < local_ds.rotation.rotation_angle
    # breakpoint()

    # Test opposite the obstacle-center
    position = np.array([-5, -4])
    opposite_ds = local_ds.evaluate(position)
    assert np.isclose(LA.norm(opposite_ds), LA.norm(reference_velocity))

    dir_attr = attractor_position - position
    assert np.allclose(opposite_ds / LA.norm(opposite_ds), dir_attr / LA.norm(dir_attr))

    # Tests at Attractor
    origin_ds = local_ds.evaluate(attractor_position)
    assert LA.norm(origin_ds) == 0

    # Tests at ellipse center
    center_ds = local_ds.evaluate(obstacle.center_position)
    assert np.allclose(center_ds, reference_velocity)

    # # Tests at ellipse surface
    # center_ds = local_ds.evaluate(obstacle.center_position)
    # assert np.allclose(center_ds, reference_velocity)

    if visualize:
        import matplotlib.pyplot as plt
        from vartools.dynamical_systems import plot_dynamical_system_quiver

        _, ax = plot_dynamical_system_quiver(
            dynamical_system=local_ds, x_lim=[-10, 10], y_lim=[-9, 9], axes_equal=True
        )

        plot_obstacle_of_dynamics(local_ds, ax=ax)


if (__name__) == "__main__":
    test_ellipse_ds(visualize=True)
