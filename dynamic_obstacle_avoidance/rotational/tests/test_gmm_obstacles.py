#!/USSR/bin/python3
""" Sample the space and decide if points are collision-full or free. """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-06-08
import logging
from timeit import default_timer as timer

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
    gmm_ellipse._gmm.precisions_cholesky_ = LA.pinv(gmm_ellipse._gmm.covariances_)
    gmm_ellipse._gmm.weights = np.ones(n_gmms)

    simple_ellipse = gmm_ellipse.transform_to_analytic_ellipses()

    # Test different positions
    position = np.array([4, 1])
    gamma_analytic = simple_ellipse[-1].get_gamma(position, in_obstacle_frame=False)
    gamma_vals_gmm_prop = gmm_ellipse.get_gamma_proportional(position, index=0)
    gamma_vals_gmm = gmm_ellipse.get_gamma(position, index=0)

    assert np.isclose(gamma_analytic, gamma_vals_gmm), "Gamma values are not close"
    # assert np.isclose(gamma_analytic, gamma_vals_gmm_prop), "Gamma values are not close"

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
    gmm_ellipse._gmm.precisions_cholesky_ = LA.pinv(gmm_ellipse._gmm.covariances_)
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
    gmm_ellipse._gmm.means_[0, :] = [4.4, 0]
    gmm_ellipse._gmm.means_[1, :] = [0, 4.0]

    gmm_ellipse._gmm.covariances_ = np.zeros((n_gmms, dimension, dimension))
    gmm_ellipse._gmm.covariances_[0, :, :] = [[0.6, 0], [0, 4.2]]
    gmm_ellipse._gmm.covariances_[1, :, :] = [[3.7, 0], [0, 0.9]]

    gmm_ellipse._gmm.precisions_cholesky_ = np.zeros((n_gmms, dimension, dimension))
    for ii in range(n_gmms):
        gmm_ellipse._gmm.precisions_cholesky_[ii, :, :] = LA.pinv(
            gmm_ellipse._gmm.covariances_[ii, :, :]
        )
    gmm_ellipse._gmm.weights = 0.5 * np.ones(n_gmms)

    pos_intersection = gmm_ellipse.get_intersection_of_ellipses(
        indices=[0, 1], it_max=100
    )

    # assert (
    #     LA.norm(pos_intersection - np.mean(gmm_ellipse._gmm.means_, axis=0)) < 0.4
    # ), "Intersection position diverged."

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

        gmm_ellipse.plot_obstacle(ax=ax)

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


def test_normal_direction(visualize=False):
    n_gmms = 2
    dimension = 2

    gmm_ellipse = GmmObstacle(n_gmms=n_gmms)
    gmm_ellipse._gmm = GaussianMixture(n_components=n_gmms)
    gmm_ellipse._gmm.means_ = np.zeros((n_gmms, dimension))
    gmm_ellipse._gmm.means_[0, :] = [4.4, 0]
    gmm_ellipse._gmm.means_[1, :] = [0, 4.0]

    gmm_ellipse._gmm.covariances_ = np.zeros((n_gmms, dimension, dimension))
    gmm_ellipse._gmm.covariances_[0, :, :] = [[0.6, 0], [0, 4.2]]
    gmm_ellipse._gmm.covariances_[1, :, :] = [[3.7, 0], [0, 0.9]]

    gmm_ellipse._gmm.precisions_cholesky_ = np.zeros((n_gmms, dimension, dimension))
    for ii in range(n_gmms):
        gmm_ellipse._gmm.precisions_cholesky_[ii, :, :] = LA.pinv(
            gmm_ellipse._gmm.covariances_[ii, :, :]
        )
    gmm_ellipse._gmm.weights = 0.5 * np.ones(n_gmms)

    position = np.array([0, 0])

    normal0 = gmm_ellipse.get_normal_direction(position, index=0)
    normal1 = gmm_ellipse.get_normal_direction(position, index=1)

    assert np.dot(normal0, normal1) == 0, "Normals are not perpendicular."
    assert LA.norm(normal0) > 0, "Normal is trivial"
    assert LA.norm(normal1) > 0, "Normal is trivial"

    if visualize:
        n_resolution = 30
        x_lim = [-8, 8]
        y_lim = [-8, 8]

        nx = ny = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx),
            np.linspace(y_lim[0], y_lim[1], ny),
        )

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

        colors = ["green", "red"]

        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        normals = np.zeros((positions.shape[0], positions.shape[1], gmm_ellipse.n_gmms))
        for it_gmm in range(gmm_ellipse.n_gmms):
            for ii in range(positions.shape[1]):
                normals[:, ii, it_gmm] = gmm_ellipse.get_normal_direction(
                    position=positions[:, ii], index=it_gmm
                )

            ax.quiver(
                positions[0, :],
                positions[1, :],
                normals[0, :, it_gmm],
                normals[1, :, it_gmm],
                color=colors[it_gmm]
                # color="k",
            )

        gmm_ellipse.plot_obstacle(ax=ax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(x_lim)

        ax.grid()


def test_relative_weights(visualize=False):
    n_gmms = 2
    dimension = 2

    gmm_ellipse = GmmObstacle(n_gmms=n_gmms)
    gmm_ellipse._gmm = GaussianMixture(n_components=n_gmms)
    gmm_ellipse._gmm.means_ = np.zeros((n_gmms, dimension))
    gmm_ellipse._gmm.means_[0, :] = [3.0, 0]
    gmm_ellipse._gmm.means_[1, :] = [0, 3.0]

    gmm_ellipse._gmm.covariances_ = np.zeros((n_gmms, dimension, dimension))
    gmm_ellipse._gmm.covariances_[0, :, :] = [[1.0, 0], [0, 3.5]]
    gmm_ellipse._gmm.covariances_[1, :, :] = [[3.5, 0], [0, 1.0]]

    gmm_ellipse._gmm.precisions_cholesky_ = np.zeros((n_gmms, dimension, dimension))
    for ii in range(n_gmms):
        gmm_ellipse._gmm.precisions_cholesky_[ii, :, :] = LA.pinv(
            gmm_ellipse._gmm.covariances_[ii, :, :]
        )
    gmm_ellipse._gmm.weights_ = 0.5 * np.ones(n_gmms)
    gmm_ellipse.evaluate_hirarchy_and_reference_points()

    position = np.array([4, -4])
    gmm_ellipse.evaluate_gamma_weights(position=position)

    # position = np.array([0, 0])
    # gmm_ellipse.evaluate_gamma_weights(position=position)
    # assert np.isclose(np.sum(gmm_ellipse.relative_weights), 1), "Weights not summing up to one"
    # assert (
    #     gmm_ellipse.relative_weights[0] == gmm_ellipse.relative_weights[1]
    # ), "Weights no equal in between."

    # position = np.array([2, -6])
    # gmm_ellipse.evaluate_gamma_weights(position=position)
    # assert np.isclose(np.sum(gmm_ellipse.relative_weights), 1), "Weights not summing up to one"
    # assert (
    #     gmm_ellipse.relative_weights[0] > gmm_ellipse.relative_weights[1]
    # ), "Weights of closer ellipse should be higher."

    if visualize:
        n_resolution = 50
        x_lim = [-8, 8]
        y_lim = [-8, 8]

        nx = ny = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx),
            np.linspace(y_lim[0], y_lim[1], ny),
        )

        fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        # levels = np.linspace(0.01, 1., 20 + 1)
        levels = np.linspace(1e-5, 1.0, 20 + 1)

        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        weights = np.zeros((positions.shape[1], gmm_ellipse.n_gmms))

        print("Start loop")
        for it_gmm in range(gmm_ellipse.n_gmms):
            for ii in range(positions.shape[1]):
                gmm_ellipse.evaluate_gamma_weights(position=positions[:, ii])
                weights[ii, :] = gmm_ellipse.relative_weights

            cs0 = axs[it_gmm].contourf(
                x_vals,
                y_vals,
                weights[:, it_gmm].reshape(x_vals.shape),
                levels=levels,
                # cmap=cmap
            )
            axs[it_gmm].set_title(f"Gamma Obstacle {it_gmm}")

            gmm_ellipse.plot_obstacle(ax=axs[it_gmm], alpha_obstacle=0.0)
            delta_pos = [0.1, 0.1]
            axs[it_gmm].text(
                gmm_ellipse.center_position(it_gmm)[0] + delta_pos[0],
                gmm_ellipse.center_position(it_gmm)[1] + delta_pos[1],
                s=(f"{it_gmm}"),
                fontsize="large",
                fontweight="bold",
            )

        print(f"roots = {gmm_ellipse.gmm_index_graph.get_root_indices()}")

        for ax in axs:
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(x_lim)
            ax.set_ylim(x_lim)
            ax.grid()

        plt.colorbar(cs0, ax=axs)


def test_project_point_on_surface(visualize=False):
    n_gmms = 1
    dimension = 2

    gmm_ellipse = GmmObstacle(n_gmms=n_gmms)
    gmm_ellipse._gmm = GaussianMixture(n_components=gmm_ellipse.n_gmms)
    gmm_ellipse._gmm.means_ = np.zeros((gmm_ellipse.n_gmms, dimension))
    gmm_ellipse._gmm.means_[0, :] = [0.0, 0]

    gmm_ellipse._gmm.covariances_ = np.zeros((n_gmms, dimension, dimension))
    gmm_ellipse._gmm.covariances_[0, :, :] = [[1.0, 0], [0, 4.0]]

    gmm_ellipse._gmm.precisions_cholesky_ = np.zeros((n_gmms, dimension, dimension))
    for ii in range(n_gmms):
        gmm_ellipse._gmm.precisions_cholesky_[ii, :, :] = LA.pinv(
            gmm_ellipse._gmm.covariances_[ii, :, :]
        )
    gmm_ellipse._gmm.weights_ = np.ones(n_gmms) / n_gmms
    # gmm_ellipse.evaluate_hirarchy_and_reference_points()

    position = np.array([0, 6])
    proj_pos = gmm_ellipse.project_point_on_surface(position, index=0)
    assert np.allclose(proj_pos, [0, 4])

    position = np.array([3, 5.6])
    proj_pos = gmm_ellipse.project_point_on_surface(position, index=0)
    assert np.isclose(gmm_ellipse.get_gamma(proj_pos, index=0), 1)

    if visualize:
        n_resolution = 10
        x_lim = [-8, 8]
        y_lim = [-8, 8]

        nx = ny = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx),
            np.linspace(y_lim[0], y_lim[1], ny),
        )

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        gmm_ellipse.plot_obstacle(ax=ax, alpha_obstacle=0.2)

        # levels = np.linspace(0.01, 1., 20 + 1)
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        proj_pos = np.zeros(positions.shape)

        for ii in range(positions.shape[1]):
            proj_pos[:, ii] = gmm_ellipse.project_point_on_surface(
                positions[:, ii], index=0
            )

            ax.plot(
                [positions[0, ii], proj_pos[0, ii]],
                [positions[1, ii], proj_pos[1, ii]],
                "k--",
            )

        ax.plot(positions[0, :], positions[1, :], "ro")
        ax.plot(proj_pos[0, :], proj_pos[1, :], "go")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(x_lim)
        ax.grid()


def test_project_point_on_surface_with_offset_center(visualize=False):
    n_gmms = 1
    dimension = 2

    gmm_ellipse = GmmObstacle(n_gmms=n_gmms)
    gmm_ellipse._gmm = GaussianMixture(n_components=gmm_ellipse.n_gmms)
    gmm_ellipse._gmm.means_ = np.zeros((gmm_ellipse.n_gmms, dimension))
    gmm_ellipse._gmm.means_[0, :] = [0.0, 0]

    gmm_ellipse._gmm.covariances_ = np.zeros((n_gmms, dimension, dimension))
    gmm_ellipse._gmm.covariances_[0, :, :] = [[1.0, 0], [0, 4.0]]

    gmm_ellipse._gmm.precisions_cholesky_ = np.zeros((n_gmms, dimension, dimension))
    for ii in range(n_gmms):
        gmm_ellipse._gmm.precisions_cholesky_[ii, :, :] = LA.pinv(
            gmm_ellipse._gmm.covariances_[ii, :, :]
        )
    gmm_ellipse._gmm.weights_ = np.ones(n_gmms) / n_gmms

    offset_center = np.array([0.2, 1.5])
    # gmm_ellipse.evaluate_hirarchy_and_reference_points()

    position = np.array([0, 6])
    proj_pos = gmm_ellipse.project_point_on_surface(position, index=0)
    assert np.isclose(gmm_ellipse.get_gamma(proj_pos, index=0), 1)

    position = np.array([3, 5.6])
    proj_pos = gmm_ellipse.project_point_on_surface(position, index=0)
    assert np.isclose(gmm_ellipse.get_gamma(proj_pos, index=0), 1)

    if visualize:
        n_resolution = 10
        x_lim = [-8, 8]
        y_lim = [-8, 8]

        nx = ny = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx),
            np.linspace(y_lim[0], y_lim[1], ny),
        )

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        gmm_ellipse.plot_obstacle(ax=ax, alpha_obstacle=0.2)

        # levels = np.linspace(0.01, 1., 20 + 1)
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        proj_pos = np.zeros(positions.shape)

        for ii in range(positions.shape[1]):
            proj_pos[:, ii] = gmm_ellipse.project_point_on_surface_with_offcenter_point(
                positions[:, ii], offcenter_point=offset_center, index=0
            )

            ax.plot(
                [positions[0, ii], proj_pos[0, ii]],
                [positions[1, ii], proj_pos[1, ii]],
                "k--",
            )

        ax.plot(offset_center[0], offset_center[1], "+", color="k")

        ax.plot(positions[0, :], positions[1, :], "ro")
        ax.plot(proj_pos[0, :], proj_pos[1, :], "go")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(x_lim)
        ax.grid()


if (__name__) == "__main__":
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=logging.INFO)

    plt.close("all")
    plt.ion()

    # test_uniradius_obstacle_from_gmm(visualize=True)
    # test_obstacle_with_radius_3_from_gmm(visualize=True)
    # test_obstacle_gradient_descent(visualize=True)
    # test_normal_direction(visualize=True)
    # test_project_point_on_surface(visualize=True)
    # test_project_point_on_surface(visualize=True)
    # test_project_point_on_surface_with_offset_center(visualize=True)
    # test_relative_weights(visualize=True)

    print("Tests executed successfully.")
    pass
