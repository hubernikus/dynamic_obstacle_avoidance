# !/usr/bin/env python3
"""
Class to evaluate different metrics during evaluatin
"""

import sys
import numpy as np

# Only one should be relevant
import json
import yaml

import warnings


def filter_moving_average(input_array, width, axis=1):
    """Along axis=0."""
    if len(input_array.shape) == 1:
        return np.convolve(input_array, np.ones(width), "valid") / width

    output_array = np.zeros((input_array.shape[0], input_array.shape[1] - width + 1))
    for dd in range(input_array.shape[0]):
        output_array[dd, :] = (
            np.convolve(input_array[dd, :], np.ones(width), "valid") / width
        )

    return output_array


class MetricEvaluator:
    def __init__(
        self,
        position=None,
        velocity=None,
        crowd_density=None,
        time=None,
        closest_dist=None,
        file_name=None,
        **kwargs
    ):

        # Initialize Metric measures
        self._acceleration_squared = []
        self._distance = []

        # Initialize lists
        self.position_list = [] if position is None else [position]
        self.velocity_list = [] if velocity is None else [velocity]

        self.closest_dist = [] if closest_dist is None else [closest_dist]

        self.time_list = [] if time is None else [time]

        # TODO: more 'metrics' as general data_list
        self.data_lists = {}
        for key in kwargs.keys():
            self.data_lists[key] = [kwargs[key]]

        self.file_name = file_name

        self.distance = 0
        self.duration = 0

        self.linear_velocity_sum = 0
        self.angular_velocity_sum = 0

        self.linear_velocity_variance = 0
        self.angular_velocity_variance = 0

        self.acceleration_summed = 0
        self.acceleration_std = 0

        self.saver_reset = False

        self.converged = True

    @property
    def start_time(self):
        return self.time_list[0]

    def reset_saver(self, position=None, velocity=None, closest_dist=None, time=None):
        self.saver_reset = True
        # Initialize lists
        self.position_list = [] if position is None else [position]
        self.velocity_list = [] if velocity is None else [velocity]
        self.closestdist_list = [] if closest_dist is None else [closest_dist]
        self.time_list = [] if time is None else [time]

        for key in self.args_list.keys():
            self.args_list[key] = []

        self.converged = True

    def update_list(self, position, velocity, closest_dist=None, time=None, **kwargs):
        self.position_list.append(position)
        self.velocity_list.append(velocity)

        for key in kwargs.keys():
            if key in self.args_list:
                self.data_lists[key].append(kwargs[key])
            else:
                self.data_lists[key] = [kwargs[key]]

        if closest_dist is not None:
            self.closestdist_list.append(closest_dist)

        if time is not None:
            self.time_list.append(time)

    def save_saver(self, file_name):
        if not self.saver_reset:
            return

        # data = self.evaluate_metrics()

        data = self.convert_all_to_dict()
        self.store_to_file(value=data, file_name=file_name)

        self.saver_reset = False
        # self.reset_saver()

    def convert_all_to_dict(self):
        if len(self.position_list) and (type(self.position_list[0]) is np.ndarray):
            self.position_list = [
                self.position_list[ii].tolist() for ii in range(len(self.position_list))
            ]

        if len(self.velocity_list) and (type(self.velocity_list[0]) is np.ndarray):
            self.velocity_list = [
                self.velocity_list[ii].tolist() for ii in range(len(self.velocity_list))
            ]

        if type(self.closestdist_list) is np.ndarray:
            self.closestdist_list = self.closestdist_list.tolist()

        if type(self.time_list) is np.ndarray:
            self.time_list = self.time_list.tolist()

        storage = {
            "position": self.position_list,
            "velocity": self.velocity_list,
            "closestdist": self.closestdist_list,
            "time": self.time_list,
        }

        # Add general lists
        storage.update(self.data_lists)

        return storage

    def import_data_from_file(self, filename, evaluate_and_return_metrics=True):
        """Import fil from json"""

        if filename[-5:] == ".json":
            with open(filename, "r") as ff:
                data = json.load(ff)

                self.position_list = data["position"]
                data.pop("position")

                self.velocity_list = data["velocity"]
                data.pop("velocity")

                self.closestdist_list = data["closestdist"]
                data.pop("closestdist")

                self.time_list = data["time"]
                data.pop("time")

                # General/special data
                self.data_lists = data

        else:
            warnings.warn("Unexpected filename detected")
            return {}

        if evaluate_and_return_metrics:
            return self.evaluate_metrics()

    def evaluate_metrics(self, delta_time=None):
        """Evaluate the metrics based on the safed position & velocity."""

        if len(self.position_list) <= 1:
            # Emtpy or 1D list -- not enough datapoints
            print("Returning empty list due to insufficient recording")
            return {}

        self.position_list = np.array(self.position_list).T
        self.velocity_list = np.array(self.velocity_list).T

        metrics = {}

        # Assumption of constant time-gap
        if hasattr(self, "time_list") and len(self.time_list):
            self.time_list = np.array(self.time_list)

            if delta_time is None:
                delta_time = self.time_list[1] - self.time_list[0]

            # Duration
            metrics["duration"] = self.time_list[-1] - self.time_list[0]

        # Closest Distance
        if hasattr(self, "closestdist_list"):
            self.closestdist_list = np.array(self.closestdist_list).T
            metrics["closest_dist"] = {}
            metrics["closest_dist"]["mean"] = np.mean(self.closestdist_list)
            metrics["closest_dist"]["variance"] = np.var(self.closestdist_list)

        # Distance
        metrics["distance"] = np.sum(
            np.linalg.norm(
                self.position_list[:, 1:] - self.position_list[:, :-1], axis=0
            )
        )

        # Linear Velocity | Don't include first velocity
        # import pdb; pdb.set_trace()
        linear_velocity = np.linalg.norm(self.velocity_list[1:], axis=0)
        if np.sum(linear_velocity) == 0:
            # Pseudo linear velocity if not saved correctly
            velocity_list = (
                self.position_list[:, 1:] - self.position_list[:, :-1]
            ) / delta_time

            linear_velocity = np.linalg.norm(velocity_list, axis=0)

            # Filter velocity
            linear_velocity = filter_moving_average(linear_velocity, width=30)

        else:
            velocity_list = self.velocity_list

        metrics["linear_velocity"] = {}
        metrics["linear_velocity"]["mean"] = np.mean(linear_velocity)
        metrics["linear_velocity"]["std"] = np.std(linear_velocity)

        # Angular Velocity
        orientation = np.arctan(velocity_list[:, 1], velocity_list[:, 0])
        angular_velocity = (orientation[1:] - orientation[:-1]) / delta_time

        metrics["angular_velocity"] = {}
        metrics["angular_velocity"]["mean"] = np.mean(angular_velocity)
        metrics["angular_velocity"]["variance"] = np.var(angular_velocity)

        # Acceleration
        acceleration = (velocity_list[:, 1:] - velocity_list[:, :-1]) / delta_time

        metrics["acceleration"] = {}
        acceleration = np.linalg.norm(acceleration, axis=0)
        metrics["acceleration"]["mean"] = np.mean(acceleration)
        metrics["acceleration"]["variance"] = np.var(acceleration)

        metrics = self.convert_dict_to_float(metrics)

        # Evaluate main direction of velocity
        mean_vel = np.mean(velocity_list, axis=1)
        metrics["direction"] = np.copysign(1, mean_vel[1])

        # General metrics
        for key in self.data_lists.keys():
            metrics[key] = {}
            metrics[key]["mean"] = np.mean(self.data_lists[key])
            metrics[key]["variance"] = np.var(self.data_lists[key])

        return metrics

    def plot_position(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        # if metrics['distance'] > 100:
        import matplotlib.pyplot as plt

        # plt.figure()
        ax.scatter(self.position_list[0, 0], self.position_list[1, 0])
        ax.plot(self.position_list[0, :], self.position_list[1, :])

        plt.axis("equal")

        plt.xlim([-5, 5])
        plt.ylim([-25, 25])

    def convert_dict_to_float(self, metrics):
        for key in metrics.keys():

            if type(metrics[key]) is dict:
                for subkey in metrics[key].keys():
                    metrics[key][subkey] = float(metrics[key][subkey])
            else:
                metrics[key] = float(metrics[key])

        return metrics

    def store_to_file(self, value, file_name=None):
        """Calculates the comulated energy used."""

        if file_name is None:
            # TODO: make sure this works
            # Open dialog on shutdown to save data

            # Dialog input libraries
            if sys.version_info > (3, 0):  # TODO: remove in future
                import tkinter as tk
                from tkinter import simpledialog
            else:
                import Tkinter as tk
                from Tkinter import simpledialog
            # Self converged

            # ROOT = tk.Tk()
            # ROOT.withdraw()
            # the input dialog
            # USER_INP = simpledialog.askstring(title="Save File",
            # prompt="File Name")
        value["converged"] = self.converged

        # with open(file_name + '.yaml', 'w') as file:
        # documents = yaml.dump(value, file)

        with open(file_name + ".json", "w") as file:
            documents = json.dump(value, file)
