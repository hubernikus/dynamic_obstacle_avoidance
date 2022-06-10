#!/USSR/bin/python3
""" Script to show lab environment on computer """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-04-23

import os
import logging

import matplotlib.pyplot as plt

import numpy as np
from numpy import linalg as LA

from vartools.handwritting_handler import HandwrittingDataHandler


class DataPlotter:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.dimensions = self.data_handler.dimensions

    # @property
    # def positions(self):
    # return

    def visualize(self, it_demos=0):
        fig, ax = plt.subplots()

        for it_demo in range(self.data_handler.n_demonstrations):
            # time = self.data_handler.get_times()
            positions = self.data_handler.get_positions(it_demo)

            ax.plot(positions[0, :], positions[1, :], ".", markersize=1)
            # ax.plot(positions[0, :], positions[1, :], '.' , color='blue')
        ax.set_aspect("equal", adjustable="box")


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    data_handler = HandwrittingDataHandler("Angle.mat")

    my_visualizer = DataPlotter(data_handler)
    my_visualizer.visualize()
