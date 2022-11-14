"""
Class to Deviate a DS based on an underlying obtacle.
"""
import numpy as np
from numpy import linalg as LA
import warnings

from vartools.dynamical_systems import DynamicalSystem
from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_angle_space

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationXd


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

    def plot_obstacle(self, ax) -> None:
        boundary_points = np.array(self.obstacle.get_boundary_xy())
        ax.plot(
            boundary_points[0, :],
            boundary_points[1, :],
            color="black",
            linestyle="--",
            zorder=3,
            linewidth=2,
        )

        global_ref = self.obstacle.get_reference_point(in_global_frame=True)
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
            self.obstacle.center_position[0],
            self.obstacle.center_position[1],
            "ko",
            linewidth=12,
            markeredgewidth=2.4,
            markersize=8,
            zorder=3,
        )

        ax.plot(
            self.attractor_position[0],
            self.attractor_position[1],
            "k*",
            linewidth=12,
            markeredgewidth=1.2,
            markersize=15,
            zorder=3,
        )

    def evaluate(self, position: np.ndarray) -> np.ndarray:
        # Weight is based on gamma
        gamma = self.obstacle.get_gamma(position, in_global_frame=True)
        if gamma < self.min_gamma:
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
            attr_weight = max(self.attractor_influence / dist_attractor - 1, 0)

            weight = weight * tmp_weight / (tmp_weight + attr_weight)

        weight = weight ** (2.0 / (1 + dot_product))

        global_rotation = VectorRotationXd.from_directions(
            self.rotation.base0, attractor_dir
        )
        local_rotation = global_rotation.rotate_vector_rotation(self.rotation)

        final_dir = local_rotation.rotate(attractor_dir, rot_factor=weight)
        velocity = final_dir * min(dist_attractor, self.maximum_velocity)

        return velocity


def test_ellipse_ds(visualize=False):
    # import matplotlib.pyplot as plt
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
        from vartools.dynamical_systems import plot_dynamical_system_quiver

        _, ax = plot_dynamical_system_quiver(
            dynamical_system=local_ds, x_lim=[-10, 10], y_lim=[-9, 9], axes_equal=True
        )

        local_ds.plot_obstacle(ax=ax)


if (__name__) == "__main__":
    test_ellipse_ds(visualize=True)
