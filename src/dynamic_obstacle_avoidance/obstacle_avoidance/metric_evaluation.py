# !/usr/bin/env python3

'''
Class to evaluate different metrics during evaluatin
'''

import sys
import numpy as np

# Only one should be relevant
import json
import yaml


class MetricEvaluator():
    def __init__(self, position=None, velocity=None, time=None, closest_dist=None, file_name=None):

        # Initialize Metric measures
        self._acceleration_squared = []
        self._distance = []

        # Initialize lists
        self.position_list = [] if position is None else [position]
        self.velocity_list = [] if velocity is None else [velocity]

        self.closest_dist = [] if closest_dist is None else [closest_dist]
        
        self.time_list = [] if time is None else [time]

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

        self.converged = True
        
    def update_list(self, position, velocity, closest_dist=None, time=None):
        self.position_list.append(position)
        self.velocity_list.append(velocity)

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
            self.position_list = [self.position_list[ii].tolist() for ii in range(len(self.position_list))]
            
        if len(self.velocity_list) and (type(self.velocity_list[0]) is np.ndarray):
            self.velocity_list = [self.velocity_list[ii].tolist() for ii in range(len(self.velocity_list))]
        
        if type(self.closestdist_list) is np.ndarray:
            self.closestdist_list = self.closestdist_list.tolist()
            
        if type(self.time_list) is np.ndarray:
            self.time_list = self.time_list.tolist()
        
        storage = {
            'position': self.position_list,
            'velocity': self.velocity_list,
            'closestdist': self.closestdist_list,
            'time': self.time_list,
        }

        
        return storage

    def evaluate_metrics(self,  dt=None):
        ''' Evaluate the metrics based on the safed position & velocity. '''

        if not len(self.position_list):
            # Emtpy list -- no datapoints yet.
            return

        self.position_list = np.array(self.position_list).T
        self.velocity_list = np.array(self.velocity_list).T
        self.closestdist_list = np.array(self.closestdist_list).T
        self.time_list = np.array(self.time_list)

        metrics = {}

        # Duration
        metrics['duration'] = self.time_list[-1] - self.time_list[0]

        # Closest Distance
        metrics['closest_dist'] = {}
        metrics['closest_dist']['mean'] = np.mean(self.closestdist_list)
        metrics['closest_dist']['variance'] = np.var(self.closestdist_list)

        # Distance
        metrics['distance'] = np.sum(
            np.linalg.norm(self.position_list[:, 1:]
                           - self.position_list[:, :-1], axis=0))

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
        acceleration = ((self.velocity_list[:, 1:] - self.velocity_list[:,:-1])
                        / (self.time_list[1:] - self.time_list[:-1]))
                       
        metrics['acceleration'] = {}
        acceleration = np.linalg.norm(acceleration, axis=0)
        metrics['acceleration']['mean'] = np.mean(acceleration)
        metrics['acceleration']['variance'] = np.var(acceleration)

        import pdb; pdb.set_trace()
        metrics = self.convert_dict_to_float(metrics)
        
        return metrics

    def convert_dict_to_float(self, metrics):
        for key in metrics.keys():
            
            if type(metrics[key]) is dict:
                for subkey in metrics[key].keys():
                    metrics[key][subkey] = float(metrics[key][subkey])
            else:
                metrics[key] = float(metrics[key])
                    
        return metrics
        
    
    def store_to_file(self, value, file_name=None):
        ''' Calculates the comulated energy used. '''
        
        if file_name is None:
            # TODO: make sure this works
            # Open dialog on shutdown to save data
            
            # Dialog input libraries
            if (sys.version_info > (3, 0)): # TODO: remove in future
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
        value['converged'] = self.converged

        # with open(file_name + '.yaml', 'w') as file:
            # documents = yaml.dump(value, file)

        with open(file_name + '.json', 'w') as file:
            documents = json.dump(value, file)

