#!/USSR/bin/python3
""" Script to show lab environment on computer """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-04-23

import os
# import warnings
# import copy

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from scipy.io import loadmat


class HandwrittingHandler:
    dir_handwritting = "/home/lukas/Code/data/lasahandwritingdataset/DataSet"
    
    def __init__(self, datset_name="Angle.mat"):
        self.datset_name = datset_name
        self.dataset = loadmat(os.path.join(self.dir_handwritting, self.datset_name))

        self.dimension = int( (self.dataset['demos'][0].shape[0] - 1) / 2)

    @property
    def n_demonstrations(self):
        return len(self.dataset['demos'][0])

        # Assign
        self.time = self.dataset['demos'][0][1]
        self.positions = self.dataset['demos'][0][1:(1+self.dimension)]
        self.velocity = self.dataset['demos'][0][(1+self.dimension):(1+2*self.dimension)]

    # @property
    # def positions(self):
        # return

    def visualize(self, it_demos=0):
        fig, ax = plt.subplots()
        
        for it_demo in range(self.dataset['demos'].shape[1]):
            demo = self.dataset['demos'][0, it_demo]
            
            time = demo[0][0]['t'][0]
            # dtime = demo[0][0]['dt'][0]

            positions = np.zeros((0, len(time)))
            for pos in demo[0][0]['pos']:
                positions = np.vstack((positions, pos))

            velocities = np.zeros((0, len(time)))
            for vel in demo[0][0]['vel']:
                velocities = np.vstack((velocities, vel))

            accelerations = np.zeros((0, len(time)))
            for acc in demo[0][0]['acc']:
                accelerations = np.vstack((accelerations, acc))

            ax.plot(positions[0, :], positions[1, :], '.', markersize=1)
            # ax.plot(positions[0, :], positions[1, :], '.' , color='blue')
        ax.set_aspect("equal", adjustable="box")


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    my_visualizer = HandwrittingHandler()
    my_visualizer.visualize()
