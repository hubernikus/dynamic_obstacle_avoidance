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
import matplotlib.pyplot as plt
import matplotlib

# Import custom libraries
from dynamic_obstacle_avoidance.obstacle_avoidance.metric_evaluation import (
    MetricEvaluator,
)

plt.close("all")

MainEvaluator = MetricEvaluator()


def import_data():
    # global_directory = ""
    global_directory = "/home/lukas/Code/dynamic_obstacle_avoidance"
    import_directory = os.path.join(global_directory, "data", "metrics_recording")
    all_recordings = os.listdir(import_directory)

    # Take random one to get all the keys & construct the dictionary
    data = MainEvaluator.import_data_from_file(
        filename=os.path.join(import_directory, all_recordings[1])
    )

    dict_metrics = {}
    dict_metrics["num_people"] = []
    dict_metrics["direction"] = []

    for key in data.keys():
        if type(data[key]) is dict:
            dict_metrics[key] = {}
            for subkey in data[key]:
                dict_metrics[key][subkey] = []
        else:
            dict_metrics[key] = []

    fig, ax = plt.subplots()
    for filename in all_recordings:
        data = MainEvaluator.import_data_from_file(
            filename=os.path.join(import_directory, filename)
        )

        if data["duration"] > 100:
            if (
                abs(MainEvaluator.position_list[0, -1]) > 5
                or abs(MainEvaluator.position_list[1, -1]) > 25
            ):
                print("Found file out of range")
                print(filename)
                print("")
                continue

        if data["duration"] < 30:
            print("Found file out of range")
            print(filename)
            print("")
            continue

        for key in data.keys():
            if type(data[key]) is dict:
                for subkey in data[key]:
                    dict_metrics[key][subkey].append(data[key][subkey])
            else:
                dict_metrics[key].append(data[key])

        split_filenames = filename.split("_")
        dict_metrics["num_people"].append(int(split_filenames[2]))

        # MainEvaluator.plot_position(ax=ax)

    return convert_to_numpy(dict_metrics)


def convert_to_numpy(data):
    """Convert dictionary of list to numpy array"""
    for key in data.keys():
        if type(data[key]) is dict:
            for subkey in data[key]:
                data[key][subkey] = np.array(data[key][subkey])
        else:
            data[key] = np.array(data[key])

    return data


def get_mean_and_variance_at_speeds(value_list, crowd_density, simulator_area=10 * 50):
    """Get the mean and variance from list"""
    crowd_sizes = np.sort(np.unique(crowd_density))
    mean_value = np.zeros(crowd_sizes.shape)
    variance_value = np.zeros(crowd_sizes.shape)

    ii = 0
    for cc in crowd_sizes:

        value_ind = crowd_density == cc

        mean_value[ii] = np.mean(value_list[value_ind])
        variance_value[ii] = np.var(value_list[value_ind])

        ii += 1

    crowd_density = crowd_sizes / simulator_area * 1000
    return crowd_density, mean_value, variance_value


dict_metrics = import_data()

plt.ion()

pos_dir = dict_metrics["direction"] > 0.0
neg_dir = dict_metrics["direction"] < 0.0

# pos_dir = np.arange(pos_dir.shape[0], dtype=int)[pos_dir]
subplot_it = 0
# n_plots = 4
# plt.figure(figsize=(5, 4))

# subplot_it += 1
# plt.subplot(n_plots, 1, subplot_it)
plt.close("all")
fig, ax = plt.subplots(figsize=(5, 4))

subplot_it += 1
plt.subplot(n_plots, 1, subplot_it)
# import pdb; pdb.set_trace()
crowd_density, value_mean, value_var = get_mean_and_variance_at_speeds(
    dict_metrics["distance"][pos_dir], dict_metrics["num_people"][pos_dir]
)
plt.errorbar(crowd_density, value_mean, yerr=np.sqrt(value_var), fmt="o", color="blue")

crowd_density, value_mean, value_var = get_mean_and_variance_at_speeds(
    dict_metrics["distance"][neg_dir], dict_metrics["num_people"][neg_dir]
)
plt.errorbar(crowd_density, value_mean, yerr=np.sqrt(value_var), fmt="o", color="red")
plt.xscale("log")
plt.xticks(ticks=[])
# plt.grid('True')
# plt.xlabel('Acceleration in Crowd')
plt.ylabel("D [m]")

subplot_it += 1
plt.subplot(n_plots, 1, subplot_it)
crowd_density, value_mean, value_var = get_mean_and_variance_at_speeds(
    dict_metrics["linear_velocity"]["mean"][pos_dir],
    dict_metrics["num_people"][pos_dir],
)
plt.errorbar(crowd_density, value_mean, yerr=np.sqrt(value_var), fmt="o", color="blue")

crowd_density, value_mean, value_var = get_mean_and_variance_at_speeds(
    dict_metrics["linear_velocity"]["mean"][neg_dir],
    dict_metrics["num_people"][neg_dir],
)
plt.errorbar(crowd_density, value_mean, yerr=np.sqrt(value_var), fmt="o", color="red")
plt.xscale("log")
# plt.xlabel('Acceleration in Crowd')
plt.xticks(ticks=[])
plt.ylabel("V [m/s] \n (Mean)")


subplot_it += 1
plt.subplot(n_plots, 1, subplot_it)
crowd_density, value_mean, value_var = get_mean_and_variance_at_speeds(
    np.sqrt(dict_metrics["linear_velocity"]["variance"][pos_dir]),
    dict_metrics["num_people"][pos_dir],
)
plt.errorbar(crowd_density, value_mean, yerr=np.sqrt(value_var), fmt="o", color="blue")

crowd_density, value_mean, value_var = get_mean_and_variance_at_speeds(
    np.sqrt(dict_metrics["linear_velocity"]["variance"][neg_dir]),
    dict_metrics["num_people"][neg_dir],
)
plt.errorbar(crowd_density, value_mean, yerr=np.sqrt(value_var), fmt="o", color="red")
plt.xscale("log")
# plt.xlabel('Acceleration in Crowd')
plt.xticks(ticks=[])
plt.ylabel("V [m/s] \n (Std.)")


subplot_it += 1
ax = plt.subplot(n_plots, 1, subplot_it)
metric_type = "duration"
crowd_density, value_mean, value_var = get_mean_and_variance_at_speeds(
    dict_metrics[metric_type][pos_dir], dict_metrics["num_people"][pos_dir]
)
plt.errorbar(crowd_density, value_mean, yerr=np.sqrt(value_var), fmt="o", color="blue")

crowd_density, value_mean, value_var = get_mean_and_variance_at_speeds(
    dict_metrics[metric_type][neg_dir], dict_metrics["num_people"][neg_dir]
)
plt.errorbar(crowd_density, value_mean, yerr=np.sqrt(value_var), fmt="o", color="red")
plt.xscale("log")
# plt.xticks(ticks=crowd_density.tolist())
ax.set_xticks(crowd_density.tolist())
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# plt.xlabel('Number of people in Crowd')
plt.ylabel("T [s]")

# subplot_it += 1
# plt.subplot(n_plots, 1, subplot_it)
# crowd_density, value_mean, value_var = get_mean_and_variance_at_speeds(
#     dict_metrics['acceleration']['mean'][pos_dir], dict_metrics['num_people'][pos_dir])
# plt.errorbar(crowd_density, value_mean, yerr=np.sqrt(value_var), fmt='o', color='blue')

# crowd_density, value_mean, value_var = get_mean_and_variance_at_speeds(
#     dict_metrics['acceleration']['mean'][neg_dir], dict_metrics['num_people'][neg_dir])
# plt.errorbar(crowd_density, value_mean, yerr=np.sqrt(value_var), fmt='o', color='red')
# plt.ylabel('Acceleration [m/s2]')
plt.xlabel(r"Crowd Density [Agents / 1000 $m^2$]")
plt.savefig("figures/" + "simulation_evaluation" + ".png", bbox_inches="tight")

# plt.figure()
# plt.scatter(dict_metrics['num_people'][pos_dir], dict_metrics['duration'][pos_dir], color='b')
# plt.scatter(dict_metrics['num_people'][pos_dir], dict_metrics['duration'][pos_dir],
# color='b', label='Positive Direction')

# plt.scatter(dict_metrics['num_people'][neg_dir], dict_metrics['duration'][neg_dir],
# color='r', label='Negative Direction')
# plt.legend()

# plt.subplots(4,1,2)

# plt.subplots(4,1,3)

# plt.subplots(4,1,4)

plt.show()
