"""
Setup file of conference room
"""
__author__ = "Lukas Huber"
__date__ = "2021-02-01"
__email__ = "lukas.huber@epfl.ch"

import json

# import yaml

import sys
import os

import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib


# Import custom libraries
from dynamic_obstacle_avoidance.obstacle_avoidance.metric_evaluation import (
    MetricEvaluator,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.metric_evaluation import (
    filter_moving_average,
)

plt.close("all")

MainEvaluator = MetricEvaluator()


def import_data_raw(metric_key="linear_velocity"):
    global_directory = "/home/lukas/Code/dynamic_obstacle_avoidance"
    import_directory = os.path.join(global_directory, "data", "metrics_recording")

    all_recordings = os.listdir(import_directory)

    data = MainEvaluator.import_data_from_file(
        filename=os.path.join(import_directory, all_recordings[1])
    )

    density_key_dict = {
        "1": "g",
        "5": "g",
        "10": "g",
        "20": "g",
        "30": "g",
        "50": "g",
        "100": "g",
    }

    for ii in range(10):
        filename = copy.deepcopy(all_recordings[ii])
        split_filenames = filename.split("_")
        num_people = int(split_filenames[2])

        MainEvaluator.import_data_from_file(
            filename=os.path.join(import_directory, all_recordings[ii])
        )

        time = MainEvaluator.time_list
        time = np.array(time) - time[0]
        dt = np.mean(time[1:] - time[:-1])

        position = np.array(MainEvaluator.position_list)

        # Sub-sample position
        induced_vel = (position[:, 1:] - position[:, :-1]) / dt

        # filtered_vel = filter_moving_average(induced_vel, width=6)
        # weights = np.ones(6)
        # vel_averaged = np.convolve(induced_vel, np.ones(width), 'valid') / width
        # induced_vel = np.ma.average(induced_vel, axis=1, weights=weights)

        linear_velocity_magnitude = np.linalg.norm(induced_vel, axis=0)
        # vel_averaged = np.convolve(induced_vel, np.ones(width), 'valid') / width
        filtered_vel = filter_moving_average(linear_velocity_magnitude, width=30)

        # linear_velocity_magnitude = np.sum(MainEvaluator.velocity_list, axis=0)
        # breakpoint()
        plt.plot(time[: filtered_vel.shape[0]], filtered_vel, label=num_people)

    plt.ion()
    plt.xlabel("Time [s]")
    plt.legend()
    plt.show()


dict_metrics = import_data_raw()

# plt.ion()

# plt.close('all')
# fig, ax = plt.subplots(figsize=(5, 4))

# for ii in range(1):
# pass
# plt.plot(dict_metrics)


plt.show()
