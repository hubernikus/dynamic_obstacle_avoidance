"""
Replicating the paper of
'Safety of Dynamical Systems With MultipleNon-Convex Unsafe Sets UsingControl Barrier Functions'
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
from matplotlib import cm

from vartools.dynamical_systems import DynamicalSystem, LinearSystem
from vartools.dynamical_systems import plot_dynamical_system_streamplot
from vartools.math import get_numerical_gradient, get_numerical_hessian
from vartools.math import get_numerical_hessian_fast
from vartools.math import get_scaled_orthogonal_projection

from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.obstacles import Obstacle

from barrier_functions import (
    BarrierFunction,
    CirclularBarrier,
    DoubleBlobBarrier,
)

# from cvxopt.modeling import variable
from cvxopt import solvers, matrix


def compare_gradients():
    """Similar to the testing-script from the 'various-tools' library."""
    blob_obstacle = DoubleBlobBarrier(
        blob_matrix=np.array([[10, 0], [0, -1]]),
        center_position=np.array([0, 3]),
    )

    positions = np.array([[2, -3], [0, 0], [1, 1]]).T

    for it_pos in range(positions.shape[1]):
        position = positions[:, it_pos]

        gradient_analytic = blob_obstacle.evaluate_gradient(position=position)
        gradient_numerical = get_numerical_gradient(
            position=position,
            function=blob_obstacle.get_barrier_value,
            delta_magnitude=1e-6,
        )

        assert np.allclose(gradient_analytic, gradient_numerical, rtol=1e-5)

        print(f"Gradient Analytic: {gradient_analytic}")
        print(f"Gradient Numerical: {gradient_numerical}")


def compare_get_numerical_hessian():
    barrier_function = CirclularBarrier(radius=1.5)

    position = np.array([0, 0])
    hessian_numerical = get_numerical_hessian(
        function=barrier_function.get_barrier_value, position=position
    )
    hessian_numerical_fast = get_numerical_hessian_fast(
        function=barrier_function.get_barrier_value, position=position
    )
    hessian_analytical = barrier_function.get_hessian(position=position)
    hessian_analytical = barrier_function.get_hessian(position=position)

    print(
        "is close here",
        np.allclose(hessian_numerical, hessian_analytical, rtol=1e-4),
    )

    print(f"{hessian_numerical=}")
    print(f"{hessian_analytical=}")
    print(f"{hessian_numerical_fast=}")
    # compare speed


if (__name__) == "__main__":
    plt.ion()
    compare_gradients()
    compare_get_numerical_hessian()

    plt.show()
