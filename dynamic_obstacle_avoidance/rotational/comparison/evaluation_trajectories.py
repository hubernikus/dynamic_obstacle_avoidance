"""
Library for the Rotation (Modulation Imitation) of Linear Systems
"""
# Author: Lukas Huber
# GitHub: hubernikus
# Created: 2022-01-20

import json
from dataclasses import dataclass

import math
from typing import Optional

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.rotational.dynamics.circular_dynamics import (
    # CircularRotationDynamics,
    SimpleCircularDynamics,
)


# IDEAS: Metrics to use
# - Distance to circle (desired path)
# - Average acceleration
# - Std acceleration
# - # of switching
# - # of stuck in local minima
# - Deviation / error to desired velocity


def normalize_velocities(velocities):
    vel_norm = LA.norm(velocities, axis=0)

    ind_nonzero = vel_norm > 0
    if not any(ind_nonzero):
        raise ValueError()

    velocities = velocities[:, ind_nonzero] / np.tile(
        vel_norm[ind_nonzero], (velocities.shape[0], 1)
    )
    return velocities


def mean_squared_error_to_path(trajectory, center_position, radius):
    # Assumption of circular dynamics
    traj_centered = trajectory - np.tile(center_position, (trajectory.shape[1], 1)).T
    err_squared = np.sum(traj_centered**2, axis=0) - radius**2

    return np.mean(err_squared)


def mean_squared_acceleration(trajectory, delta_time):
    velocities = (trajectory[:, 1:] - trajectory[:, :-1]) / delta_time
    acceleration = (velocities[:, 1:] - velocities[:, :-1]) / delta_time

    acceleration_squared = np.sum(acceleration**2, axis=0)
    return np.mean(acceleration_squared)


def mean_squared_velocity_deviation(trajectory, dynamic_functor, delta_time):
    velocities = (trajectory[:, 1:] - trajectory[:, :-1]) / delta_time
    velocities = normalize_velocities(velocities)

    init_velocities = np.zeros_like(velocities)
    for ii in range(init_velocities.shape[1]):
        init_velocities[:, ii] = dynamic_functor(trajectory[:, ii])

    delta_vel = velocities - init_velocities
    deviations = np.sum(delta_vel**2, axis=0)

    return np.mean(deviations)


def mean_squared_velocity_change_dot_product(trajectory, delta_time):
    velocities = (trajectory[:, 1:] - trajectory[:, :-1]) / delta_time
    velocities = normalize_velocities(velocities)

    velocity_dotprod = np.sum(velocities[:, 1:] * velocities[:, :-1], axis=0)
    velocity_dot_prod_fact = (1.0 - velocity_dotprod) / 2.0
    return np.mean(velocity_dot_prod_fact**2)


def mean_squared_velocity_deviation_dot_product(
    trajectory, dynamic_functor, delta_time
):
    velocities = (trajectory[:, 1:] - trajectory[:, :-1]) / delta_time
    velocities = normalize_velocities(velocities)

    init_velocities = np.zeros_like(velocities)
    for ii in range(init_velocities.shape[1]):
        init_velocities[:, ii] = dynamic_functor(trajectory[:, ii])
        if not (init_norm := LA.norm(init_velocities)):
            continue
        init_velocities = init_velocities / init_norm

    velocity_dotprod = np.sum(velocities * init_velocities, axis=0)
    velocity_dotprod_factor = (1.0 - velocity_dotprod) / 2.0

    return np.mean(velocity_dotprod_factor**2)


@dataclass
class TrajectoryEvaluator:
    data_folder: "str"
    data_path: "str" = "/home/lukas/Code/dynamic_obstacle_avoidance/dynamic_obstacle_avoidance/rotational/comparison/data"

    n_runs: int = 0

    dist_to_path: float = 0

    squared_acceleration: float = 0
    squared_error_velocity: float = 0

    dotprod_err_velocity: float = 0
    dotprod_acceleration: float = 0

    n_local_minima: int = 0
    n_converged: int = 0

    def run(self):
        with open(
            os.path.join(datapath, "..", "comparison_parameters.json")
        ) as user_file:
            simulation_parameters = json.load(user_file)

        delta_time = simulation_parameters["delta_time"]
        it_max = simulation_parameters["it_max"]

        initial_dynamics = SimpleCircularDynamics()

        datafolder_path = os.path.join(self.data_path, self.data_folder)
        files_list = os.listdir(datafolder_path)

        if self.n_runs <= 0:
            self.n_runs = len(files_list)
        else:
            files_list = files_list[: self.n_runs]

        print(f"Evaluating #{self.n_runs} runs.")

        for ii, filename in enumerate(files_list):
            trajectory = np.loadtxt(
                os.path.join(datafolder_path, filename),
                delimiter=",",
                dtype=float,
                skiprows=0,
            )

            if not len(trajectory):
                warnings.warn("Empty trajectory file.")
                continue

            trajectory = trajectory.T

            if trajectory.shape[1] < it_max:
                self.n_local_minima += 1
            else:
                self.n_converged += 1

            self.dist_to_path += (
                mean_squared_error_to_path(
                    trajectory, center_position=np.zeros(2), radius=2.0
                )
                / self.n_runs
            )

            self.squared_acceleration += (
                mean_squared_acceleration(trajectory, delta_time) / self.n_runs
            )

            self.squared_error_velocity += (
                mean_squared_velocity_deviation(
                    trajectory, initial_dynamics.evaluate, delta_time
                )
                / self.n_runs
            )

            self.dotprod_err_velocity += (
                mean_squared_velocity_change_dot_product(trajectory, delta_time)
                / self.n_runs
            )

            self.dotprod_acceleration += (
                mean_squared_velocity_deviation_dot_product(
                    trajectory, initial_dynamics.evaluate, delta_time
                )
                / self.n_runs
            )


def print_table(evaluation_list):
    # print(" & & \\\\ \hline")

    value = [
        f"{ee.n_converged / ee.n_runs * 100:.0f}" + "\\%" for ee in evaluation_list
    ]
    print(" & ".join(["$N^c$"] + value) + " \\\\ \hline")

    value = [f"{ee.dist_to_path:.2f}" for ee in evaluation_list]
    print(" & ".join(["$\\Delta R^2$"] + value) + " \\\\ \hline")

    value = [f"{ee.squared_acceleration:.2f}" for ee in evaluation_list]
    print(" & ".join(["$a$"] + value) + " \\\\ \hline")

    value = [f"{ee.squared_error_velocity:.2f}" for ee in evaluation_list]
    print(" & ".join(["$\Delta v$"] + value) + " \\\\ \hline")

    value = [f"{ee.dotprod_err_velocity * 1e6:.2f}" for ee in evaluation_list]
    print(" & ".join(["$\\langle v \\rangle [1e6]$"] + value) + " \\\\ \hline")

    value = [f"{ee.dotprod_acceleration * 1e1:.2f}" for ee in evaluation_list]
    print(" & ".join(["$\\langle a \\rangle [1e1m/s]$"] + value) + " \\\\ \hline")


if (__name__) == "__main__":

    if False:
        # n_runs = 5
        n_runs = -1

        nonlinear_evaluation = TrajectoryEvaluator(
            n_runs=n_runs, data_folder="nonlinear_avoidance"
        )
        nonlinear_evaluation.run()

        modulation_evaluation = TrajectoryEvaluator(
            n_runs=n_runs, data_folder="modulation_avoidance"
        )
        modulation_evaluation.run()

        gfield_evaluation = TrajectoryEvaluator(
            n_runs=n_runs, data_folder="guiding_field"
        )
        gfield_evaluation.run()

    print_table([nonlinear_evaluation, modulation_evaluation, gfield_evaluation])
