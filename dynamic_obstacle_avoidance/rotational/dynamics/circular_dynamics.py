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

        rot_weight = self.get_rotation_weight(position)
        init_direction = self.get_initial_direction(position)

        rot_dir = self._rotation.rotate(init_direction, rot_factor=rot_weight)

        return rot_dir * self.constant_velocity


class SimpleCircularDynamics(DynamicalSystem):
    def __init__(self):
        super().__init__(dimension=2)

        self.k1 = 1
        self.E = np.array([[0, -1], [1, 0]])
        self.R = 2

    def get_phi(self, position):
        return np.sum(position**2) - self.R**2

    def get_grad(self, position):
        return 2 * position

    def evaluate(self, position):
        grad = self.get_grad(position)
        phi = self.get_phi(position)

        direction = self.E @ grad - self.k1 * phi * grad

        if not (dir_norm := LA.norm(direction)):
            return direction

        return direction / dir_norm


def test_rotation_circle(visualize=False):
    circular_ds = CircularRotationDynamics(
        radius=1.0,
        constant_velocity=1.0,
        outside_approaching_factor=12.0,
        inside_approaching_factor=6.0,
    )

    if visualize:
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


if (__name__) == "__main__":
    # test_rotation_circle(visualize=True)
    _test_simple_dynamcis(visualize=True)
    print("Done tests")
