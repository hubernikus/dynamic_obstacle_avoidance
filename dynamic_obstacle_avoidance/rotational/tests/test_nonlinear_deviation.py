#!/USSR/bin/python3.10
""" Test directional orientation system. """
# Author: Lukas Huber
# Created: 2022-09-02
# Github: hubernikus
# License: BSD (c) 2022

from abc import ABC, abstractmethod

import copy
from math import pi

# from typing import List, Protocol

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import train_test_split

from vartools.linalg import get_orthogonal_basis
from vartools.dynamical_systems import DynamicalSystem, ConstantValue

# from vartools.directional_space import get_angle_space_of_array
# from vartools.directional_space import get_angle_space
from vartools.directional_space import get_angle_space_inverse
from vartools.directional_space import get_angle_space, get_angle_space_of_array
from vartools.directional_space import get_directional_weighted_sum

from dynamic_obstacle_avoidance.rotational.datatypes import Vector, VectorArray

Basis = np.ndarray
DeviationVector = np.ndarray

# TODO:
# - directly incoorporate (rotational + fast)) obstacle avoidance as a function


class ConstantRegressor:
    def __init__(self, value: np.ndarray):
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.value, (X.shape[0], 1))


class MultiOutputSVR:
    """Creates several svr-models to predict multi-dimensional output."""

    def __init__(self, **kwargs):
        # Temporary store keyword arguments for later construction.
        self._kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = y.shape[1]

        self._models = [SVR(**self._kwargs) for _ in range(self.n_features_out_)]

        for ii, svr in enumerate(self._models):
            svr.fit(X, y[:, ii])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([svr.predict(X) for svr in self._models]).T


class DirectionalSystem(ABC):
    @abstractmethod
    def evaluate(self, position: Vector) -> Vector:
        pass

    @abstractmethod
    def evaluate_convergence_velocity(position: Vector) -> Vector:
        pass

    # @abstractmethod
    # def evaluate_deviation(self, position: Vector) -> DeviationVector:
    #     pass


class DeviationOfConstantFlow(DirectionalSystem):
    maximal_deviation = 0.99 * pi / 2.0

    def __init__(self, reference_velocity: Vector, regressor) -> None:
        self.reference_velocity = reference_velocity
        self.velocity_magnitude = LA.norm(self.reference_velocity)

        self.base = get_orthogonal_basis(
            reference_velocity / LA.norm(reference_velocity)
        )

        self.regressor = regressor

    def get_lyapunov_gradient_direction(self, position: Vector) -> Vector:
        """Returns the reference direction / gradient of the Lyapunov function.
        The value is normalize (!), since only the direction is relevant."""
        return self.reference_velocity

    @property
    def dimension(self):
        return self.reference_velocity.shape[0]

    def fit_from_velocities(self, X: np.ndarray, y_directions: np.ndarray) -> None:
        y = self._clean_input_data(y_directions)
        self.fit(X, y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the internal learner onto the given model."""
        self.n_samples_fit_ = X.shape[0]

        if y.shape[0] != self.n_samples_fit_:
            raise ValueError("Input data is not consistent.")

        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = y.shape[1]

        if self.n_features_out_ != self.dimension - 1:
            raise ValueError("Unexpected number of output features.")

        self.regressor.fit(X, y)

    def _clean_input_data(self, data):
        # TODO: use directly the array -> get_angle_space_of_array()
        directions = np.zeros((data.shape[0], data.shape[1] - 1))
        for ii in range(data.shape[0]):
            directions[ii, :] = get_angle_space(data[ii, :], null_matrix=self.base)
        return directions

    def evaluate(self, position: Vector) -> Vector:
        angle_deviation = self.evaluate_deviation(position)
        return (
            get_angle_space_inverse(angle_deviation, null_matrix=self.base)
            * self.velocity_magnitude
        )

    def evaluate_deviation(self, position: Vector) -> np.ndarray:
        """Return deviation, while making sure that the deviation stays within the
        feasible boundaries."""
        deviation = self.regressor.predict(position.reshape(1, -1))[0]
        if (dev_norm := LA.norm(deviation)) > self.maximal_deviation:
            deviation = deviation / dev_norm * self.maximal_deviation
        return deviation

    def evaulate_without_lyapunov_check(self, position: Vector) -> np.ndarray:
        angle_deviation = self.regressor.predict(position.reshape(1, -1))[0]
        return (
            get_angle_space_inverse(angle_deviation, null_matrix=self.base)
            * self.velocity_magnitude
        )

    def predict(self, position: np.ndarray) -> np.ndarray:
        return self.regressor.predict(position)

    def evaluate_convergence_velocity(self, position: Vector) -> Vector:
        return self.reference_velocity


class DeviationOfLinearDS(DirectionalSystem):
    # >> Try to obtain similar behavior with modified GMM / GMR
    # TODO: modify scikit.learn GMM in order to allow for in depth

    weight_orientation = 0.1
    # self.angle_inf = 0.5 * math.pi

    def __init__(
        self,
        attractor_position: Vector,
        n_components: int = 5,
        covariance_type: str = "full",
    ):
        self.attractor_position = attractor_position

    def get_lyapunov_gradient_direction(self, position: Vector) -> Vector:
        """Returns the reference direction / gradient of the Lyapunov function.
        Note that this is only the direction but not the magnitude (!)."""
        attr_dir = self.attractor_position - position
        if not (attr_norm := LA.norm(attr_dir)):
            return np.zeros_like(position)

        return attr_dir / attr_norm

    def get_distance(self, positions: VectorArray, factor_dp: float = 1.0) -> float:
        """Returns 'angle' distance, i.e., a combination of dot-product and local-radius

        Current tuning is such that at dot_prod <= 0 -> distance is infinite"""

        rel_dists = position - np.tile(self.attractor_position, (2, 1)).T
        dist_norms = LA.norm(rel_dists, axis=0)

        if any(dist_norms == 0):
            return sys.float_info.max

        rel_dists = dist_norms / np.tile(dist_norms, (dist_norms.shape[0], 1))
        dot_prod = np.dot(rel_dists[:, 0], rel_dists[:, 1])

        if dot_prod <= 0:
            return sys.float_info.max

        return factor_dp * (1.0 / dot_prod - 1) + abs(rel_dists[0] - rel_dists[1])


class PerpendicularDeviatoinOfLinearDS(DirectionalSystem):
    """The linear dynamics is a sum of ()2 * D), with D = the number of dimensions of linear systems."""

    # TODO:
    # >> This could be further extended to optimize / learn the distribution weight (!)
    # In such a scenario, the proposed algorithm could learn complex(er) motions
    # BUT: would it actually stay stable?

    def __init__(self, attractor_position: Vector, regressor):
        self.attractor_position = attractor_position

        self.regressors = [
            [copy.deepcopy(regressor) for _ in range(2)] for _ in range(self.dimension)
        ]

        self.prox_distance = 1
        self.max_velocity = 1

        self.max_angle = np.pi * 0.49
        # self.unbelievable_angle = np.pi * 0.8

    @property
    def dimension(self):
        return self.attractor_position.shape[0]

    def fit_from_velocities(self, X, y):
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = y.shape[1]

        X_relative = X - np.tile(self.attractor_position, (X.shape[0], 1))

        self._null_matrices = np.zeros(
            (self.dimension, self.dimension, 2, self.dimension)
        )

        for dd in range(self.dimension):
            for ii, sign in enumerate([-1, 1]):
                # Create null_axis which is pointing towards centers
                null_axis = np.zeros(self.dimension)
                null_axis[dd] = (-1) * sign
                self._null_matrices[:, :, dd, ii] = get_orthogonal_basis(null_axis)

                ind_good = (X_relative[:, dd] * sign) >= 0

                if not np.sum(ind_good):
                    self.regressors[dd][ii] = ConstantRegressor(
                        np.zeros(self.dimension - 1)
                    )
                    continue

                X_tmp = X_relative[ind_good, :]
                y_tmp = self.velocity_to_direction(
                    y[
                        ind_good,
                    ],
                    self._null_matrices[:, :, dd, ii],
                )

                self.regressors[dd][ii].fit(X_tmp, y_tmp)

    def velocity_to_direction(self, data, null_matrix):
        directions = np.zeros((data.shape[0], data.shape[1] - 1))
        for ii in range(data.shape[0]):
            directions[ii, :] = get_angle_space(data[ii, :], null_matrix=null_matrix)
        return directions

    def evaluate(self, position: Vector):
        relative_position = position - self.attractor_position

        if not (distance := LA.norm(relative_position)):
            return np.zeros_like(position)

        direction = relative_position / distance

        local_predictions = np.zeros((self.dimension, self.dimension))
        for dd in np.arange(self.dimension):
            ii = 0 if relative_position[dd] < 0 else 1
            angle = self.regressors[dd][ii].predict(relative_position.reshape(1, -1))[0]

            if (angle_norm := LA.norm(angle)) > self.max_angle:
                angle = angle / angle_norm * self.max_angle
            # angle = np.zeros(self.dimension - 1)

            local_predictions[:, dd] = get_angle_space_inverse(
                angle, null_matrix=self._null_matrices[:, :, dd, ii]
            )
            # breakpoint()

        # breakpoint()
        weights = self.get_weights(relative_position)

        # TODO: transfer of direction would allow for higher directions
        # i.e. less conservative cut-off
        final_direction = get_directional_weighted_sum(
            null_direction=(-1) * direction,
            directions=local_predictions,
            weights=weights,
        )

        return final_direction * self.get_velocity_factor(distance)

    def get_velocity_factor(self, distance):
        return min(1, distance / self.prox_distance) * self.max_velocity

    def evaluate_convergence_velocity(self, position: Vector) -> Vector:
        direction = self.attractor_position - position

        if dir_norm := LA.norm(direction):
            dir_norm = direction / dir_norm * self.get_velocity_factor(dir_norm)

        return direction

    def get_weights(self, relative_position):
        # The further along an axis, the higher the weight
        # weights = position**2
        weights = abs(relative_position)
        return weights / np.sum(weights)

    def get_reference_system(self, position: Vector):
        pass

    def predict(self, positions: VectorArray) -> np.ndarray:
        """Predicts the deviation."""
        deviations = np.zeros((positions.shape[0], self.dimension - 1))
        for pp in range(positions.shape[0]):
            if not (norm_pos := LA.norm(positions[pp, :])):
                continue

            velocity = self.evaluate(positions[pp, :])
            norm_vel = LA.norm(velocity)

            deviations[pp, :] = np.cross(positions[pp, :], velocity) / (
                norm_pos * norm_vel
            )

        return deviations


def test_learning_deviation_of_linear_DS(visualize=False, save_figure=False):
    dynamics = PerpendicularDeviatoinOfLinearDS(
        attractor_position=np.zeros(2),
        regressor=MultiOutputSVR(kernel="rbf", gamma=0.05),
    )

    # Create data towards attractor with different inputs
    n_points = 100
    x1 = np.linspace(-np.pi, 0, n_points)
    y1 = np.sin(x1)

    x2 = (-1) * x1
    y2 = y1.copy()

    x3 = np.zeros(n_points)
    y3 = np.linspace(np.pi, 0, n_points)

    X = np.vstack((np.hstack((x1, x2, x3)), np.hstack((y1, y2, y3)))).T
    y = np.vstack((X[1:, :] - X[:-1, :], np.zeros((1, 2))))
    y[n_points - 1, :] = 0
    y[2 * n_points - 2, :] = 0

    tt_ratio = 2 / 3
    train_index, test_index = train_test_split(
        np.arange(X.shape[0]), test_size=(1 - tt_ratio)
    )

    X_train = X[train_index, :]
    X_test = X[test_index, :]

    y_train = y[train_index, :]
    y_test = y[test_index, :]

    dynamics.fit_from_velocities(X=X_train, y=y_train)

    if visualize:
        # direction_predict = dynamics.predict(X_test)
        # direction_test = dynamics._clean_input_data(y_test)
        # MSE = np.mean((direction_predict - direction_test) ** 2)
        # print(f"Mean squared error of test-set: {round(MSE, 4)}")

        plt.close("all")
        plt.ion()

        fig, ax = plt.subplots()
        x_lim = [-5, 5]
        y_lim = [-5, 5]

        n_grid = 40
        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_grid),
            np.linspace(y_lim[0], y_lim[1], n_grid),
        )
        positions = np.array([xx.flatten(), yy.flatten()])
        velocities = np.zeros_like(positions)
        angles = np.zeros(positions.shape[1])

        for pp in range(positions.shape[1]):
            velocities[:, pp] = dynamics.evaluate(positions[:, pp])
            if not (pos_norm := LA.norm(positions[:, :])):
                continue

            vel_norm = LA.norm(velocities[:, pp])
            angles[pp] = (
                (-1)
                * np.cross(positions[:, pp], velocities[:, pp])
                / (pos_norm * vel_norm)
            )

        ax.plot(X[:, 0], X[:, 1], "r.")

        ax.quiver(
            positions[0, :],
            positions[1, :],
            velocities[0, :],
            velocities[1, :],
            # scale=5,
        )

        # levels = np.linspace(-pi / 2, pi / 2, 21)
        # cntr = ax.contourf(
        #     positions[0, :].reshape(n_grid, n_grid),
        #     positions[1, :].reshape(n_grid, n_grid),
        #     angles.reshape(n_grid, n_grid),
        #     levels=levels,
        #     # cmap="cool",
        #     cmap="seismic",
        #     # alpha=0.7,
        #     extend="both",
        # )
        # fig.colorbar(cntr)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.axis("equal")

    #
    position = np.array([0, 1])
    velocity = dynamics.evaluate(position)
    assert velocity[1] < 0 and abs(velocity[0]) < abs(velocity[1])

    position = np.array([np.pi, 0])
    velocity = dynamics.evaluate(position)
    assert velocity[0] < 0 and velocity[1] < 0

    position = np.array([-np.pi, 0])
    velocity = dynamics.evaluate(position)
    assert velocity[0] > 0 and velocity[1] < 0


def test_learn_sinus_motion(visualize=False, save_figure=False):
    # Data going from right to left
    np.random.seed(1)

    n_points = 1000
    x = np.linspace(0, 4 * pi, n_points)
    y = np.cos(x)

    X = np.vstack((x, y)).T
    X = X + np.random.rand(X.shape[0], X.shape[1]) * 0.1

    # Obtain velocities form position, and make sure they have the same dimensions
    velocities = X[1:, :] - X[:-1, :]
    X = X[:-1, :]

    tt_ratio = 2 / 3
    train_index, test_index = train_test_split(
        np.arange(X.shape[0]), test_size=(1 - tt_ratio)
    )

    X_train = X[train_index, :]
    X_test = X[test_index, :]

    y_train = velocities[train_index, :]
    y_test = velocities[test_index, :]

    dynamics = DeviationOfConstantFlow(
        reference_velocity=np.mean(velocities, axis=0),
        regressor=MultiOutputSVR(kernel="rbf", gamma=0.05),
    )

    dynamics.fit_from_velocities(X_train, y_train)

    if visualize:
        direction_predict = dynamics.predict(X_test)
        direction_test = dynamics._clean_input_data(y_test)
        MSE = np.mean((direction_predict - direction_test) ** 2)
        print(f"Mean squared error of test-set: {round(MSE, 4)}")

        plt.close("all")
        plt.ion()

        fig, ax = plt.subplots()
        ax.plot(X[:, 0], X[:, 1], ".", color="black")

        x_lim = [x[0], x[-1]]
        y_lim = [-1.3, 1.3]

        n_grid = 40
        xx, yy = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], n_grid),
            np.linspace(y_lim[0], y_lim[1], n_grid),
        )
        positions = np.array([xx.flatten(), yy.flatten()])

        predictions = dynamics.predict(positions.T).T
        levels = np.linspace(-pi / 2, pi / 2, 21)

        cntr = ax.contourf(
            positions[0, :].reshape(n_grid, n_grid),
            positions[1, :].reshape(n_grid, n_grid),
            predictions.reshape(n_grid, n_grid),
            levels=levels,
            # cmap="cool",
            cmap="seismic",
            # alpha=0.7,
            extend="both",
        )
        fig.colorbar(cntr)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    #
    position = np.array([4.8, 0])
    velocity = dynamics.evaluate(position)
    assert velocity[1] > 0

    position = np.array([7.8, 0])
    velocity = dynamics.evaluate(position)
    assert velocity[1] < 0


if (__name__) == "__main__":
    # test_learn_sinus_motion(visualize=True)
    # test_learning_deviation_of_linear_DS(visualize=True)
    print("Tests finished.")
