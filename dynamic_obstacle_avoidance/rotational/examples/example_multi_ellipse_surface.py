#!/USSR/bin/python3
""" Sample the space and decide if points are collision-full or free. """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-04-23

import logging

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
import matplotlib as mpl

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization import plot_obstacles


def collision_sample_space(obstacle_container, num_samples, x_lim, y_lim):
    """ Returns random points based on collision with
    [ >0 : free space // <0 : within obstacle ] """
    dimension = 2
    rand_points = np.random.rand(dimension, num_samples)
    
    rand_points[0] = rand_points[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
    rand_points[1] = rand_points[1] * (y_lim[1] - y_lim[0]) + y_lim[0]

    value = obstacle_container.get_maximum_gamma(rand_points)
    
    value = value - 1
    return rand_points, value


class EnvironmentLearner:
    def __init__(self, data_points):
        pass


def main():
    pass


def get_three_elipse_learner():
    x_lim = [-10, 10]
    y_lim = [-10, 10]

    environment = ObstacleContainer()
    environment.append(
        Cuboid(
            center_position=[4.5, 0],
            axes_length=[2, 6],
        ))

    environment.append(
        Cuboid(
            center_position=[2, 3],
            axes_length=[5, 2],
        ))

    environment.append(
        Cuboid(
            center_position=[2, -3],
            axes_length=[5, 2],
        ))

    plot_obstacles(obstacle_container=environment, x_lim=x_lim, y_lim=y_lim)
    data_points, label = collision_sample_space(
        obstacle_container=environment,
        num_samples=100,
        x_lim=x_lim,
        y_lim=y_lim
    )
    min_label = np.min(label)
    max_label = np.max(label)
    
    label_range = np.maximum(abs(min_label), max_label)

    _, ax = plt.subplots()
    ax.scatter(
        data_points[0, :],
        data_points[1, :],
        s=10,
        c=label,
        cmap='seismic',
        # norm=mpl.colors.Normalize(vmin=(-1)*label_range, vmax=label_range)
    )
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    
if (__name__) == "__main__":
    main()
    
    environment_learner = get_three_elipse_learner()
    
    pass
