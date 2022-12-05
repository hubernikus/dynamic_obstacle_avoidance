"""
Dynamical System with Convergence towards attractor_position
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import math

import numpy as np
from numpy import linalg as LA

from vartools.dynamical_systems._base import DynamicalSystem
from vartools.dynamical_systems.linear import LinearSystem

from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationXd
from dynamic_obstacle_avoidance.rotational.datatypes import Vector


class WavyLinearDynmics(DynamicalSystem):
    def __init__(
        self,
        attractor_position: Vector,
        distance_strech_factor: float = 1,
        maximum_deviation: float = math.pi * 0.3,  # Just below pi / 2
        **kwargs
    ):

        super().__init__(attractor_position=attractor_position)
        if self.dimension != 2:
            raise NotImplementedError("This vectorfield is limited to 2-D.")

        self._linear_dynamics = LinearSystem(self.attractor_position, **kwargs)

        self.distance_strech_factor = distance_strech_factor
        self.maximum_deviation = maximum_deviation

    def get_rotation_matrix(self, angle: float) -> np.ndarray:
        sin_ = math.sin(angle)
        cos_ = math.cos(angle)
        return np.array([[cos_, -sin_], [sin_, cos_]])

    def evaluate(self, position: Vector) -> Vector:
        distance = LA.norm(position - self.attractor_position)

        deviation = math.sin(distance * self.distance_strech_factor)
        rot_matrix = self.get_rotation_matrix(deviation * self.maximum_deviation)
        linear_velocity = self._linear_dynamics.evaluate(position)

        return rot_matrix @ linear_velocity


def test_wavy_dynamics(visualize=False):
    attractor_position = np.array([1, -1])
    dynamics = WavyLinearDynmics(
        attractor_position=attractor_position,
        # maximum_deviation=0,
    )

    if visualize:
        import matplotlib.pyplot as plt
        from vartools.dynamical_systems import plot_dynamical_system_quiver

        _, ax = plot_dynamical_system_quiver(
            dynamical_system=dynamics, x_lim=[-8, 8], y_lim=[-8, 8], axes_equal=True
        )
        ax.plot(
            attractor_position[0],
            attractor_position[1],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

    # Zero velocity at attractor
    velocity = dynamics.evaluate(dynamics.attractor_position)
    assert math.isclose(LA.norm(velocity), 0), "Non-converging at attractor"

    # Zero velocity at attractor
    position = np.array([3, 4])
    velocity = dynamics.evaluate(position)
    velocity_linear = dynamics._linear_dynamics.evaluate(position)
    assert np.dot(velocity, velocity_linear) > 0, "System is (locally) unstable."


if (__name__) == "__main__":
    test_wavy_dynamics(visualize=True)
