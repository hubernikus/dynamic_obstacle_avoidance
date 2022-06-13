#!/USSR/bin/python3
""" Sample the space and decide if points are collision-full or free. """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-06-08
import logging

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.rotational.gmm_obstacle import GmmObstacle


def plot_gammas_multigamma(gmm_ellipse, simple_ellipse):
    n_resolution = 30
    x_lim = [-4, 4]
    y_lim = [-4, 4]

    nx = ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx),
        np.linspace(y_lim[0], y_lim[1], ny),
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    gamma_vals_gmm = np.zeros(positions.shape[1])
    gamma_vals_gmm_prop = np.zeros(positions.shape[1])
    gamma_vals_analytic = np.zeros(positions.shape[1])

    for ii in range(positions.shape[1]):
        gamma_vals_analytic[ii] = simple_ellipse[-1].get_gamma(
            positions[:, ii], in_obstacle_frame=False
        )

        gamma_vals_gmm[ii] = gmm_ellipse.get_gamma(positions[:, ii], index=0)

        gamma_vals_gmm_prop[ii] = gmm_ellipse.get_gamma_proportional(
            positions[:, ii], index=0
        )

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    levels = np.linspace(1.0, 10.0, 10)

    cs0 = axs[0].contourf(
        x_vals,
        y_vals,
        gamma_vals_gmm_prop.reshape(x_vals.shape),
        levels=levels,
        # cmap=cmap
    )
    axs[0].set_title("Proportional-Gamma Value for GMM-Obstacle")

    cs1 = axs[1].contourf(
        x_vals,
        y_vals,
        gamma_vals_gmm.reshape(x_vals.shape),
        levels=levels,
        # cmap=cmap
    )
    axs[1].set_title("Gamma Value for GMM-Obstacle")

    cs2 = axs[2].contourf(
        x_vals,
        y_vals,
        gamma_vals_analytic.reshape(x_vals.shape),
        levels=levels,
        # cmap=cmap
    )
    axs[2].set_title("Gamma Value for Analytic Obstacle")

    for ax in axs:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(x_lim)

        ax.grid()

    fig.tight_layout()
    plt.colorbar(cs2, ax=axs)


def test_uniradius_obstacle_from_gmm(visualize=False):
    """Test to verify the consistent behavior between the Gmm-obstacle and the
    corresponding analytic one."""
    dimension = 2
    n_gmms = 1
    gmm_ellipse = GmmObstacle(n_gmms=1)
    gmm_ellipse._gmm = GaussianMixture(n_components=n_gmms)
    gmm_ellipse._gmm.means_ = np.ones((n_gmms, dimension))
    gmm_ellipse._gmm.covariances_ = 1.0 * np.eye(dimension).reshape(
        n_gmms, dimension, dimension
    )
    gmm_ellipse._gmm.precisions_cholesky = LA.pinv(gmm_ellipse._gmm.covariances_)
    gmm_ellipse._gmm.weights = np.ones(n_gmms)

    simple_ellipse = gmm_ellipse.transform_to_analytic_ellipses()

    # Test different positions
    position = np.array([4, 1])
    gamma_analytic = simple_ellipse[-1].get_gamma(position, in_obstacle_frame=False)
    gamma_vals_gmm_prop = gmm_ellipse.get_gamma_proportional(position, index=0)
    gamma_vals_gmm = gmm_ellipse.get_gamma(position, index=0)

    assert np.isclose(gamma_analytic, gamma_vals_gmm), "Gamma values are not close"
    assert np.isclose(gamma_analytic, gamma_vals_gmm_prop), "Gamma values are not close"

    if visualize:
        plot_gammas_multigamma(gmm_ellipse, simple_ellipse)


def test_obstacle_with_radius_3_from_gmm(visualize=False):
    """Test to verify the consistent behavior between the Gmm-obstacle and the
    corresponding analytic one."""
    dimension = 2
    n_gmms = 1

    gmm_ellipse = GmmObstacle(n_gmms=1)
    gmm_ellipse._gmm = GaussianMixture(n_components=n_gmms)
    gmm_ellipse._gmm.means_ = np.zeros((n_gmms, dimension))
    gmm_ellipse._gmm.covariances_ = 3.0 * np.eye(dimension).reshape(
        n_gmms, dimension, dimension
    )
    gmm_ellipse._gmm.precisions_cholesky = LA.pinv(gmm_ellipse._gmm.covariances_)
    gmm_ellipse._gmm.weights = np.ones(n_gmms)

    simple_ellipse = gmm_ellipse.transform_to_analytic_ellipses()

    # Test different positions
    position = np.array([4, 1])
    gamma_analytic = simple_ellipse[-1].get_gamma(position, in_obstacle_frame=False)
    gamma_vals_gmm = gmm_ellipse.get_gamma(position, index=0)

    assert np.isclose(gamma_analytic, gamma_vals_gmm), "Gamma values are not close"

    if visualize:
        plot_gammas_multigamma(gmm_ellipse, simple_ellipse)


def test_obstacle_gradient_descent(visualize=False):
    n_gmms = 2
    dimension = 2

    gmm_ellipse = GmmObstacle(n_gmms=1)
    gmm_ellipse._gmm = GaussianMixture(n_components=n_gmms)
    gmm_ellipse._gmm.means_ = np.zeros((n_gmms, dimension))
    gmm_ellipse._gmm.means_[0, :] = [1.5, 0]
    gmm_ellipse._gmm.means_[1, :] = [0, 1.5]

    gmm_ellipse._gmm.covariances_ = np.zeros((n_gmms, dimension, dimension))
    gmm_ellipse._gmm.covariances_[0, :, :] = [[1, 0], [0, 3]]
    gmm_ellipse._gmm.covariances_[1, :, :] = [[3, 0], [0, 1]]

    gmm_ellipse._gmm.precisions_cholesky = np.zeros((n_gmms, dimension, dimension))
    for ii in range(n_gmms):
        gmm_ellipse._gmm.precisions_cholesky[ii, :, :] = LA.pinv(
            gmm_ellipse._gmm.covariances_[ii, :, :]
        )
    gmm_ellipse._gmm.weights = 0.5 * np.ones(n_gmms)

    pos_intersection = gmm_ellipse.get_intersection_of_ellipses(
        indices=[0, 1], it_max=100
    )

    assert (
        LA.norm(pos_intersection - np.mean(gmm_ellipse._gmm.means_, axis=0)) < 0.4
    ), "Intersection position diverged."

    if visualize:
        n_resolution = 30
        x_lim = [-8, 8]
        y_lim = [-8, 8]

        nx = ny = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx),
            np.linspace(y_lim[0], y_lim[1], ny),
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gamma_vals_gmm_prop = np.zeros((n_gmms, positions.shape[1]))

        gradient_field = np.zeros(positions.shape)

        for ii in range(positions.shape[1]):
            for it_gmm in range(n_gmms):
                gamma_vals_gmm_prop[it_gmm, ii] = gmm_ellipse.get_gamma_proportional(
                    positions[:, ii], index=it_gmm
                )

                gradient_field[:, ii] += (-1) * gmm_ellipse.get_gamma_derivative(
                    positions[:, ii],
                    index=it_gmm,
                    powerfactor=10,
                )

        gamma_min = np.min(gamma_vals_gmm_prop, axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        levels = np.linspace(1.0, 10.0, 10)

        cs0 = ax.contourf(
            x_vals,
            y_vals,
            gamma_min.reshape(x_vals.shape),
            levels=levels,
            # cmap=cmap
        )
        ax.quiver(
            positions[0, :],
            positions[1, :],
            gradient_field[0, :],
            gradient_field[1, :],
            color="k",
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(x_lim)

        ax.grid()

        # fig.tight_layout()
        plt.colorbar(cs0, ax=ax)

        for it_max in [0, 1, 5, 10, 20, 50]:
            # for it_max in [20]:
            pos_intersection = gmm_ellipse.get_intersection_of_ellipses(
                indices=[0, 1],
                it_max=it_max,
            )

            ax.plot(pos_intersection[0], pos_intersection[1], "ko")


if (__name__) == "__main__":
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO)

    plt.close("all")
    plt.ion()
    # test_uniradius_obstacle_from_gmm(visualize=True)
    # test_obstacle_with_radius_3_from_gmm(visualize=True)
    test_obstacle_gradient_descent(visualize=True)
    pass
