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


def collision_sample_space(obstacle_container, num_samples, x_lim, y_lim):
    """Returns random points based on collision with
    [ >0 : free space // <0 : within obstacle ]"""
    dimension = 2
    rand_points = np.random.rand(dimension, num_samples)

    rand_points[0] = rand_points[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
    rand_points[1] = rand_points[1] * (y_lim[1] - y_lim[0]) + y_lim[0]

    value = obstacle_container.get_minimum_gamma(rand_points)

    # value = value - 1
    value = (value > 1).astype(int)
    value = 2 * value - 1  # Value is in [-1, 1]

    return rand_points, value


# class MultiEllipseObstacle(Obstacle):
# pass


class MultiGuassianObstacle(Obstacle):
    def __init__(self, pose=None, n_components=1, dimension=2, **kwargs):
        if pose is None:
            pose = ObjectPose(position=np.zeros(dimension))

        super().__init__(pose=pose, **kwargs)

        self.n_components = n_components
        self._gmm = GaussianMixture(n_components=n_components)

    def fit(self, X):
        self.center_position = np.mean(X, axis=0)
        self._gmm.fit(X)

        # Find core gaussian
        dists = LA.norm(
            self._gmm.means_
            - np.tile(self.center_position, (self._gmm.n_components, 1)),
            axis=1,
        )

        ind_core = np.argmin(dists)

        # =Parent <-> children
        # One child only one parent => how to do connection

        # Define a rule that allow to continously decend the gaussians

        # TODO: more smart/constraint gradient tescent to better align with limitations

    def get_normal_direction(self):
        pass

    def get_gamma(self):
        pass

    def plot_probability(self, x_lim=None, y_lim=None, n_resolution=10):
        nx = ny = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx),
            np.linspace(y_lim[0], y_lim[1], ny),
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

        # proba = self._gmm.predict_proba(positions.T)
        # proba = np.sum(proba * np.tile(self._gmm.weights_, (proba.shape[0], 1)), axis=1)

        proba = np.exp(self._gmm.score_samples(positions.T))

        fig, ax = plt.subplots()
        contourf_ = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            proba.reshape(nx, ny),
        )
        cbar = fig.colorbar(contourf_)

    def plot_gaussians(self, x_lim=None, y_lim=None):
        fig, ax = plt.subplots()

        colors = ["navy", "turquoise", "darkorange"]
        ii = 0
        # breakpoint()
        for jj in range(len(self._gmm.covariances_)):
            covariances = self._gmm.covariances_[jj][:2, :2]

            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])

            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ell = mpl.patches.Ellipse(
                self._gmm.means_[jj, :2], v[0], v[1], 180 + angle, color=colors[jj]
            )

            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        ax.set_aspect("equal", "datalim")


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

    my_obstacle = MultiGuassianObstacle(n_components=3)
    my_obstacle.fit(obs_points.T)
    my_obstacle.plot_gaussians(x_lim=x_lim, y_lim=y_lim)
    my_obstacle.plot_probability(x_lim=x_lim, y_lim=y_lim, n_resolution=100)


if (__name__) == "__main__":
    plt.close("all")
    gaussian_clustering()
