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

from sklearn.mixture import GaussianMixture
from sklearn import svm

from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from dynamic_obstacle_avoidance.rotational.gmm_obstacle import GmmObstacle


def collision_sample_space(obstacle_container, num_samples, x_lim, y_lim):
    """Returns random points based on collision with
    [ >0 : free space // <0 : within obstacle ]"""
    dimension = 2
    rand_points = np.random.rand(dimension, num_samples)

    rand_points[0] = rand_points[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
    rand_points[1] = rand_points[1] * (y_lim[1] - y_lim[0]) + y_lim[0]

    value = obstacle_container.get_minimum_gamma_of_array(rand_points)

    # value = value - 1
    value = (value > 1).astype(int)
    value = 2 * value - 1  # Value is in [-1, 1]

    return rand_points, value


def gaussian_clustering():
    x_lim = [-2, 7]
    y_lim = [-6.5, 6.5]

    environment = ObstacleContainer()
    environment.append(
        Cuboid(
            center_position=[4.5, 0],
            axes_length=[2, 8],
        )
    )

    environment.append(
        Cuboid(
            center_position=[2, 3],
            axes_length=[5, 2],
        )
    )

    environment.append(
        Cuboid(
            center_position=[2, -3],
            axes_length=[5, 2],
        )
    )

    # plot_obstacles(obstacle_container=environment, x_lim=x_lim, y_lim=y_lim)
    data_points, label = collision_sample_space(
        obstacle_container=environment, num_samples=1000, x_lim=x_lim, y_lim=y_lim
    )

    x_lim = [-10, 10]
    y_lim = [-10, 10]

    obs_points = data_points[:, label < 0]
    fig, ax = plt.subplots()
    ax.plot(obs_points[0, :], obs_points[1, :], ".")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal", adjustable="box")

    my_obstacle = GmmObstacle(n_gmms=3)
    my_obstacle.fit(obs_points)
    # my_obstacle.plot_gaussians(x_lim=x_lim, y_lim=y_lim)
    # my_obstacle.plot_probability(x_lim=x_lim, y_lim=y_lim, n_resolution=100)


if (__name__) == "__main__":
    plt.close("all")
    gaussian_clustering()
