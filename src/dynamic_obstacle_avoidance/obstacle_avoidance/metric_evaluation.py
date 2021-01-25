# !/usr/bin/env python3

'''
Class to evaluate different metrics during evaluatin
'''

# import json
import yaml
import numpy as np

class MetricEvaluator():
    def __init__(self, position=None, velocity=None, time=None, filename=None):

        # Initialize Metric measures
        self._acceleration_squared = []
        self._distance = []

        # Initialize lists
        self.position_list = [] if position is None else [position]
        self.velocity_list = [] if velocity is None else [velocity]
        self.time_list = [] if time is not None else [time]

        self.filename = filename

        self.distance = 0
        self.duration = 0

        self.linear_velocity_sum = 0
        self.angular_velocity_sum = 0

        self.linear_velocity_variance = 0
        self.angular_velocity_variance = 0

        self.acceleration_summed = 0
        self.acceleration_std = 0

        
    def update_list(self, position, velocity, time=None):
        self.position_list.append(position)
        self.velocity_list.append(position)

        if time is not None:
            self.time_list.append(time)


    def shutdown(self, time):
        metrics = self.evaluate_metrics()
        self.store_to_file(value=metrics, filename=self.file_name)

    def evaluate_metrics(self,  dt= None):
        ''' Evaluate the metrics based on the safed position & velocity. '''

        self.position_list = np.array(self.position_list).T
        self.velocity_list = np.array(self.velocity_list).T

        metrics = {}

        # Duration
        metrics['duration'] = self.time_list[-1] - self.time_list[0]

        # Distance
        metrics['distance'] = np.sum(
            np.linalg.norm(self.position_list[:, 1:]
                           - self.position_list[:, 1:], axis=0))

        # Linear Velocity | Don't include first velocity
        linear_velocity = np.linalg.norm(self.velocity_list[1:], axis=0)
        metrics['linear_velocity'] = {}
        metrics['linear_velocity']['mean'] = np.mean(linear_velocity)
        metrics['linear_velocity']['variance'] = np.var(linear_velocity)

        # Angular Velocity
        orientation = np.arctan(self.velocity_list[:, 1], self.velocity_list[:, 0])
        angular_velocity = ((orientation[1:] - orientation[:-1])
                            / (self.time_list[1:] - self.time_list[:-1]) )
        metrics['angular_velocity'] = {}
        metrics['angular_velocity']['mean'] = np.mean(linear_velocity)
        metrics['angular_velocity']['variance'] = np.var(linear_velocity)
        
        # Acceleration
        acceleration = ((self.velocity_list_[:, 1:] - self.velocity_list[:-1])
                        / (self.time_list[1:] - self.time_list[:-1]))
        
        metrics['acceleration'] = {}
        acceleration = np.linalg.norm(acceleration, axis=0)
        metrics['acceleration']['mean'] = np.mean(acceleration)
        metrics['acceleration']['variance'] = np.var(acceleration)

        return metrics

    
    def store_to_file(self, value, filename=None):
        ''' Calculates the comulated energy used. '''
        
        if filename is None:
            # Open dialog on shutdown to save data
            
            # Dialog input libraries
            import tkinter as tk
            from tkinter import simpledialog

            ROOT = tk.Tk()

            ROOT.withdraw()
            # the input dialog
            USER_INP = simpledialog.askstring(title="Save File",
                                              prompt="File Name")

        with open(filename, 'w') as file:
            documents = yaml.dump(value, file)
