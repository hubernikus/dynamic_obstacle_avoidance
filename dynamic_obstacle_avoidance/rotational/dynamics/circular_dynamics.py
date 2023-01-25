"""
Circular Limit Cycle Field based on VectorRotationXd
"""
import math
from typing import Optional

import numpy as np
from numpy import linalg as LA

from vartools.dynamical_systems import DynamicalSystem
from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationXd
from dynamic_obstacle_avoidance.rotational.datatypes import Vector


class DynamicDynamics:
    """

    Attributes
    ----------
    decent_factor: Radius at which it stops: influence_radius*decent_factor
        -> centeral dynamics are not moving
    """

    def __init__(
        self,
        dynamics_local: DynamicalSystem,
        dynamics_of_base: DynamicalSystem,
        influence_radius: float = 2.0,
    ) -> None:
        self.dynamics_local = dynamics_local
        self.dynamics_of_base = dynamics_of_base

        self.influence_radius = influence_radius
        self.decent_factor = 2.0

        self.time_step_of_base_movement = 1e-2

    @property
    def position(self) -> Vector:
        return self.dynamics_local.pose.position

    @position.setter
    def position(self, value: Vector) -> None:
        self.dynamics_local.pose.position = value

    def update_base(self, position: Vector) -> None:
        dist_center = LA.norm(position - self.position)
        if dist_center < self.influence_radius:
            weight = 1
        elif dist_center < self.influence_radius * self.decent_factor:
            max_radius = self.influence_radius * self.decent_factor
            weight = (dist_center - self.influence_radius) / (
                (self.decent_factor - 1) * self.influence_radius
            )
        else:
            return

        base_velocity = self.dynamics_of_base.evaluate(position)

        self.position = (
            base_velocity * weight * self.time_step_of_base_movement + self.position
        )

    def evaluate(self, position: Vector) -> Vector:
        self.update_base(position)
        return self.dynamics_local.evaluate(position)


class CircularRotationDynamics(DynamicalSystem):
    def __init__(
        self,
        radius: float = 1.0,
        constant_velocity: float = 1.0,
        outside_approaching_factor: float = 1.0,
        inside_approaching_factor: float = 1.0,
        dimension: int = 2,
        direction: int = 1,
        pose: Optional[ObjectPose] = None,
        base_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ):
        """
        Arguments
        ---------
        direction: either -1 or +1 => the direction or orientation
        """
        if pose is None:
            pose = ObjectPose(position=np.zeros(dimension))
        # else:
        #     pose: ObjectPose = pose

        super().__init__(dimension=dimension, pose=pose, **kwargs)

        self.radius = radius
        self.constant_velocity = constant_velocity
        self.outside_approaching_factor = outside_approaching_factor
        self.inside_approaching_factor = inside_approaching_factor

        if dimension != 2:
            raise ValueError("Base vectors for rotation are needed")

        self._rotation = VectorRotationXd.from_directions(
            np.array([1, 0]), np.array([0, np.copysign(1, direction)])
        )

        if base_matrix is None:
            self.base_matrix = np.eye(self.dimension)
        else:
            self.base_matrix = base_matrix

    @classmethod
    def from_rotation(cls, VectorRotationXd: Vector):
        raise NotImplementedError()

    def get_initial_direction(self, position) -> Vector:
        """Initial direction pointing outwards"""
        if not (pos_norm := LA.norm(position)):
            return position
        else:
            return position / pos_norm

    def sigmoid(
        self,
        x,
        logistic_growth: float = 1.0,
        max_exponent: float = 50.0,
        margin_value=1e-9,
    ):
        exponent = logistic_growth * x

        # breakpoint()
        # Avoid numerical error
        if exponent > max_exponent:
            return 1 - margin_value
        elif exponent < (-1) * max_exponent:
            return margin_value

        value = 1 / (1 + math.exp((-1) * exponent))
        return max(min(value, 1 - margin_value), margin_value)

    def get_rotation_weight(self, position: Vector) -> float:
        """Returns a weight between ]0, 2[ , the weight is 1 on the radius"""
        pos_norm = LA.norm(position)

        if pos_norm < self.radius:
            distance_surface = 1 - self.radius / pos_norm
            distance_surface = distance_surface * self.inside_approaching_factor
        else:
            distance_surface = (pos_norm - self.radius) * (1.0 / self.radius)
            distance_surface = distance_surface * self.outside_approaching_factor

        return 2 * self.sigmoid(distance_surface)

    def evaluate(self, position):
        if not LA.norm(position):
            return position

        local_position = self.base_matrix.T @ position

        rot_weight = self.get_rotation_weight(local_position)
        init_direction = self.get_initial_direction(local_position)

        rot_dir = self._rotation.rotate(init_direction, rot_factor=rot_weight)
        final_dir = rot_dir * self.constant_velocity

        return self.base_matrix @ final_dir


class SimpleCircularDynamics(DynamicalSystem):
    """Creates circular 2D or 3D dynamics."""

    def __init__(
        self, base_matrix: Optional[np.ndarray] = None, dimension: int = 2, **kwargs
    ):
        if base_matrix is None:
            self.base_matrix = np.eye(dimension)
        else:
            self.base_matrix = base_matrix
            dimension = base_matrix.shape[0]

        super().__init__(dimension=dimension, **kwargs)

        self.k1 = 1.0
        self.k2 = 1.0

        self._E = np.array([[0, -1], [1, 0]])
        self.radius = 2.0

    def transform_position_to_relative(self, position: Vector) -> Vector:
        return self.base_matrix.T @ (position - self.pose.position)

    def transform_position_from_relative(self, position: Vector) -> Vector:
        return (self.base_matrix @ position) + self.pose.position

    def transform_direction_to_relative(self, position: Vector) -> Vector:
        return self.base_matrix.T @ position

    def transform_direction_from_relative(self, position: Vector) -> Vector:
        return self.base_matrix @ position

    def get_phi(self, position: Vector) -> float:
        return np.sum(position[:2] ** 2) - self.radius**2

    def get_grad(self, position: Vector) -> np.ndarray:
        """Returns 2D gradient (in the circulation plane)"""
        return 2 * position[:2]

    def evaluate(self, position):
        relative_position = self.transform_position_to_relative(position)

        grad = self.get_grad(relative_position)  # or n11
        phi = self.get_phi(relative_position)  # or e11

        direction = np.zeros((self.dimension))
        direction[:2] = self._E @ grad - self.k1 * phi * grad
        if self.dimension > 2:
            direction[2:] = (-1) * relative_position[2:] * self.k2

        if not (dir_norm := LA.norm(direction)):
            return direction
        normalized_direction = direction / dir_norm

        return self.transform_direction_from_relative(normalized_direction)


def test_rotation_circle(visualize=False):
    circular_ds = CircularRotationDynamics(
        radius=1.0,
        constant_velocity=1.0,
        outside_approaching_factor=12.0,
        inside_approaching_factor=6.0,
    )

    if visualize:
        import matplotlib.pyplot as plt
        from vartools.dynamical_systems import plot_dynamical_system

        x_lim = [-2, 2]
        y_lim = [-2, 2]

        figsize = (8, 6)

        fig, ax = plt.subplots(figsize=figsize)
        plot_dynamical_system(
            dynamical_system=circular_ds,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_resolution=17,
        )

    # Test on the circle
    position = np.array([-1, 0])
    weight = circular_ds.get_rotation_weight(position)
    assert math.isclose(weight, 1.0)
    inital_direction = circular_ds.get_initial_direction(position)
    assert np.allclose(inital_direction, [-1, 0])
    final_velocity = circular_ds.evaluate(position)
    assert np.allclose(final_velocity, [0, -1])

    # Test close far away
    position = np.array([0, 1e9])
    weight = circular_ds.get_rotation_weight(position)
    assert math.isclose(weight, 2)
    inital_direction = circular_ds.get_initial_direction(position)
    assert np.allclose(inital_direction, [0, 1])
    final_velocity = circular_ds.evaluate(position)
    assert np.allclose(final_velocity, [0, -1], atol=1e-3)

    # Test close to the center
    position = np.array([1e-3, 0])
    weight = circular_ds.get_rotation_weight(position)
    assert weight < 1e-3
    inital_direction = circular_ds.get_initial_direction(position)
    assert np.allclose(inital_direction, [1, 0])
    final_velocity = circular_ds.evaluate(position)
    assert np.allclose(final_velocity, [1, 0], atol=1e-3)


def _test_simple_dynamcis(visualize=False):
    initial_ds = SimpleCircularDynamics()

    if visualize:
        import matplotlib.pyplot as plt
        from vartools.dynamical_systems import plot_dynamical_system

        x_lim = [-4, 4]
        y_lim = [-4, 4]

        figsize = (8, 6)

        fig, ax = plt.subplots(figsize=figsize)
        plot_dynamical_system(
            dynamical_system=initial_ds,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_resolution=17,
        )


def test_3d_simple_dynamics(visualize=False):
    base_matrix = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ]
    )
    pose = ObjectPose(position=np.array([1.5, -1, 3]))
    initial_ds = SimpleCircularDynamics(base_matrix=base_matrix, pose=pose)

    if visualize:
        # from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt

        # plt.close("all")
        plt.ion()

        ax = plt.figure().add_subplot(projection="3d")
        # X, Y, Z = axes3d.get_test_data(0.05)

        it_max = 1000
        dt = 0.05
        dimension = 3
        # start_postion = np.array([4, 3, 3])

        starting_array = [
            [4, 3, 3],
            [5, 2, 1],
            [-1, 3, 4],
            [0.1, 0.3, 0.1],
            [-4, 3, 3],
            [-4, -3, 3],
            [4, -3, 3],
        ]

        for start_postion in starting_array:
            positions = np.zeros((dimension, it_max + 1))
            positions[:, 0] = start_postion

            for ii in range(it_max):
                velocity = initial_ds.evaluate(positions[:, ii])
                positions[:, ii + 1] = positions[:, ii] + velocity * dt

            ax.plot(positions[0, :], positions[1, :], positions[2, :])
            ax.set_aspect("equal", "box")

    # Random position
    position = np.array([4, 3, 3])
    velocity = initial_ds.evaluate(position)
    assert velocity[0] < 0 and velocity[1] < 0 and velocity[2] > 0

    # Center position
    position = pose.position
    velocity = initial_ds.evaluate(position)
    assert np.allclose(velocity, np.zeros_like(velocity))


if (__name__) == "__main__":
    # test_rotation_circle(visualize=True)
    # _test_simple_dynamcis(visualize=True)
    test_3d_simple_dynamics(visualize=True)
    print("Done tests")
