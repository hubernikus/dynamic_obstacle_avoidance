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

from dynamic_obstacle_avoidance.rotational.dynamics import ConstantRegressor
from dynamic_obstacle_avoidance.rotational.dynamics import MultiOutputSVR
from dynamic_obstacle_avoidance.rotational.dynamics import DirectionalSystem
from dynamic_obstacle_avoidance.rotational.dynamics import DeviationOfConstantFlow

from dynamic_obstacle_avoidance.rotational.dynamics import DeviationOfLinearDS
from dynamic_obstacle_avoidance.rotational.dynamics import (
    PerpendicularDeviatoinOfLinearDS,
)


from dynamic_obstacle_avoidance.rotational.datatypes import Vector, VectorArray

Basis = np.ndarray
DeviationVector = np.ndarray

# TODO:
# - directly incoorporate (rotational + fast)) obstacle avoidance as a function


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
