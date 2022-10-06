"""
Tests (and visualizations) for KmeansMotionLearner and KMeansObstacle.

To run, in the ipython environment:
>>>
run dynamic_obstacle_avoidance/rotational/tests/test_kmeans_learner_handwritting.py

"""

import warnings

import random
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use("Agg")

from vartools.handwritting_handler import HandwrittingHandler

# from dynamic_obstacle_avoidance.rotational.kmeans_obstacle import KMeansObstacle
from dynamic_obstacle_avoidance.rotational.kmeans_motion_learner import (
    KMeansMotionLearner,
    create_kmeans_obstacle_from_learner,
)


from dynamic_obstacle_avoidance.rotational.tests.helper_functions import (
    plot_boundaries,
    plot_normals,
    plot_gamma,
    plot_reference_dynamics,
    plot_trajectories,
    plot_region_dynamics,
    plot_partial_dynamcs_of_four_clusters,
    plot_gamma_of_learner,
)


# fig_dir = "/home/lukas/Code/dynamic_obstacle_avoidance/figures/"
# fig_dir = "figures/"

# Chose figures as either png / pdf
fig_type = ".png"
# fig_type = ".pdf"


def _test_local_deviation(visualize=False, save_figure=False):
    RANDOM_SEED = 1
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    x_lim = [-5.5, 0.5]
    y_lim = [-1.5, 3.0]

    data = HandwrittingHandler(file_name="2D_Ashape.mat")
    main_learner = KMeansMotionLearner(data)

    fig, ax_kmeans = plt.subplots()
    main_learner.plot_kmeans(ax=ax_kmeans, x_lim=x_lim, y_lim=y_lim)
    ax_kmeans.axis("equal")
    # ax_kmeans.set_xlim(x_lim)
    # ax_kmeans.set_ylim(y_lim)

    if save_figure:
        fig_name = "kmeans_a_shape"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    fig, ax = plt.subplots()
    reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        fig_name = "raw_data_a_shape"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    for index in range(main_learner.kmeans.n_clusters):
        # if index == 1:
        # continue
        print(f"Doing index {index}")

        index_neighbourhood = np.array(
            [index]
            + main_learner.get_successors(index)
            + main_learner.get_predecessors(index)
        )

        x_min = (
            np.min(main_learner.kmeans.cluster_centers_[index_neighbourhood, 0])
            - main_learner.region_radius_
        )
        x_max = (
            np.max(main_learner.kmeans.cluster_centers_[index_neighbourhood, 0])
            + main_learner.region_radius_
        )

        y_min = (
            np.min(main_learner.kmeans.cluster_centers_[index_neighbourhood, 1])
            - main_learner.region_radius_
        )
        y_max = (
            np.max(main_learner.kmeans.cluster_centers_[index_neighbourhood, 1])
            + main_learner.region_radius_
        )

        n_grid = 40
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, n_grid),
            np.linspace(y_min, y_max, n_grid),
        )
        positions = np.array([xx.flatten(), yy.flatten()])
        predictions = np.zeros(positions.shape[0])

        levels = np.linspace(-pi / 2, pi / 2, 50)

        predictions = main_learner._dynamics[index].predict(positions.T)
        for pp in range(positions.shape[1]):
            is_inside = False
            for ii in index_neighbourhood:
                if main_learner.region_obstacles[ii].is_inside(
                    positions[:, pp], in_global_frame=True
                ):
                    is_inside = True
                    break

            if not is_inside:
                predictions[pp] = 0

        fig, ax = plt.subplots(figsize=(10, 8))
        # fig, ax = fig.subplots(figsize=(14, 9))
        cntr = ax.contourf(
            positions[0, :].reshape(n_grid, n_grid),
            positions[1, :].reshape(n_grid, n_grid),
            predictions.reshape(n_grid, n_grid),
            levels=levels,
            # cmap="cool",
            cmap="seismic",
            # alpha=0.7,
            extend="both",
        )
        fig.colorbar(cntr)

        main_learner.plot_boundaries(ax=ax)

        reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
        ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

        # Plot attractor
        ax.scatter(
            main_learner.data.attractor[0],
            main_learner.data.attractor[1],
            marker="*",
            s=200,
            color="black",
            zorder=10,
        )

        ax.axis("equal")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # main_learner.kmeans.
        # breakpoint()

        if save_figure:
            fig_name = f"local_rotation_around_neighbourhood_{index}"
            fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


def plot_a_shape_partial_motions(save_figure=False):
    RANDOM_SEED = 0
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    dataname = "a_shape"
    data = HandwrittingHandler(file_name="2D_Ashape.mat")

    main_learner = KMeansMotionLearner(data)

    figsize = (7, 5.0)
    x_lim, y_lim = get_min_max_from_data(data.position)

    fig, ax = plt.subplots(figsize=figsize)
    reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.axis("equal")

    if save_figure:
        fig_name = f"{dataname}_data_only"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    plot_region_dynamics(main_learner, x_lim, y_lim, ax=ax)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.axis("equal")

    plot_trajectories(ax, main_learner)

    if save_figure:
        fig_name = f"{dataname}_global_dynamics_and_trajectories"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=figsize)
    main_learner.plot_kmeans(ax=ax, x_lim=x_lim, y_lim=y_lim)
    ax.axis("equal")
    if save_figure:
        fig_name = f"{dataname}_kmeans"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=figsize)
    plot_gamma_of_learner(
        main_learner, x_lim, y_lim, hierarchy_passing_gamma=False, fig=fig, ax=ax
    )

    if save_figure:
        fig_name = f"{dataname}_gamma_values_and_trajectories"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=figsize)
    plot_gamma_of_learner(
        main_learner,
        x_lim,
        y_lim,
        hierarchy_passing_gamma=True,
        ax=ax,
        fig=fig,
    )

    if save_figure:
        fig_name = f"gamma_values_with_transition_a_shape"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


def do_individual_cluster_evaluations(save_figure=False):
    RANDOM_SEED = 0
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    dataname = "a_shape"

    # data = HandwrittingHandler(file_name="Angle.mat")
    data = HandwrittingHandler(file_name="2D_Ashape.mat")
    main_learner = KMeansMotionLearner(data)

    figsize = (6, 5.5)
    x_lim, y_lim = get_min_max_from_data(data.position)

    plot_partial_dynamcs_of_four_clusters(
        visualize=True,
        main_learner=main_learner,
        x_lim=x_lim,
        y_lim=y_lim,
        save_figure=save_figure,
        name=dataname,
    )


def plot_snake_partial_motions(save_figure=False, fig_name="", data=None):
    RANDOM_SEED = 3
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # data = HandwrittingHandler(file_name="2D_messy-snake.mat")

    # data = HandwrittingHandler(file_name="2D_Lshape.mat")
    # x_lim = [-6.5, 0.5]
    # y_lim = [-2.0, 3.0]
    # main_learner = KMeansMotionLearner(data, n_clusters=4)
    if data is None:
        data = HandwrittingHandler(file_name="2D_Sshape.mat")
        fig_name = "sshape_2d_"

    x_lim = [-6.5, 0.5]
    y_lim = [-2.2, 3.5]
    main_learner = KMeansMotionLearner(data, n_clusters=8)

    position = np.array([-3.9, 0.61])
    velocity = main_learner.predict(position)

    # Analysis of specific cluster
    # index = 5
    index = 6
    tmp_obstacle = create_kmeans_obstacle_from_learner(main_learner, index)

    fig, ax = plt.subplots()
    # plot_normals(ax, tmp_obstacle)
    plot_reference_dynamics(ax, main_learner, index)
    plot_boundaries(ax=ax, kmeans_learner=main_learner, plot_attractor=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    ax.axis("equal")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        fig_name = f"data_only_" + fig_name
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")
        breakpoint()

    main_learner.plot_boundaries(ax=ax)
    main_learner.plot_kmeans(ax=ax, x_lim=x_lim, y_lim=y_lim, centerlabel=False)

    ax.axis("equal")

    if save_figure:
        fig_name = "kmeans_handwritting_" + fig_name
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    index = 0
    tmp_obstacle = create_kmeans_obstacle_from_learner(main_learner, index)

    position = np.array([-3.04769137, -0.12654154])
    ax.plot(position[0], position[1], "ro")

    # gamma = tmp_obstacle.get_gamma(position, in_global_frame=True)

    fig, ax = plot_region_dynamics(main_learner, x_lim, y_lim)
    reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    plot_trajectories(ax, main_learner)

    if save_figure:
        fig_name = "global_dynamics_and_trajectories_" + fig_name
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


def plot_kmeans_messy_snake(save_figure=False, fig_name="", data=None):
    RANDOM_SEED = 0
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if data is None:
        data = HandwrittingHandler(file_name="2D_messy-snake.mat")
        fig_name = "2D_messy-snake"

    x_lim = [-7.5, 1.5]
    y_lim = [-2.2, 3.5]

    main_learner = KMeansMotionLearner(data, n_clusters=10)

    # index = 0
    # fig, ax = plt.subplots()
    # plot_reference_dynamics(ax, main_learner, index)
    # plot_boundaries(ax=ax, kmeans_learner=main_learner, plot_attractor=True)

    fig, ax = plt.subplots(figsize=(11, 7))
    reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    ax.axis("equal")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        fig_name = f"data_only_" + fig_name
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    main_learner.plot_boundaries(ax=ax)
    main_learner.plot_kmeans(ax=ax, x_lim=x_lim, y_lim=y_lim, centerlabel=True)

    ax.axis("equal")

    if save_figure:
        fig_name = f"kmeans_handwritting_" + fig_name
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    index = 0
    tmp_obstacle = create_kmeans_obstacle_from_learner(main_learner, index)

    position = np.array([-3.04769137, -0.12654154])
    ax.plot(position[0], position[1], "ro")

    gamma = tmp_obstacle.get_gamma(position, in_global_frame=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_region_dynamics(main_learner, x_lim, y_lim, ax=ax)
    reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    plot_trajectories(ax, main_learner)

    if save_figure:
        fig_name = "global_dynamics_and_trajectories_" + fig_name
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


def plot_snake_partial_motions():
    file_name = "2D_messy-snake.mat"
    # data = HandwrittingHandler(file_name="2D_Sshape.mat")
    data = HandwrittingHandler(file_name=file_name)

    priors, mu, sigma = get_gmm_from_matlab(data.position, data.velocity)
    return priors, mu, sigma


def get_gmm_from_matlab(position, velocity):
    est_options = {
        # "type": "diag",
        "type": 0,
        "maxK": 15.0,
        "fixed_K": matlab.double([]),
        "samplerIter": 20.0,
        "do_plots": 0,
        "sub_sample": 1,
        "estimate_l": 1.0,
        "l_sensitivity": 2.0,
        "length_scale": matlab.double([]),
    }

    pos_array = matlab.double(position.T)
    vel_array = matlab.double(velocity.T)

    priors, mu, sigma = matlab_eng.fit_gmm(pos_array, vel_array, est_options, nargout=3)

    priors = np.array(priors)
    mu = np.array(mu)
    sigma = np.array(sigma)

    return priors, mu, sigma


def get_min_max_from_data(positions, range_fraction=0.5):
    # X-Values
    x_min = positions[:, 0].min()
    x_max = positions[:, 0].max()
    x_range = x_max - x_min

    delta_x = 1
    # delta_x = x_range * range_fraction
    delta_y = 1
    # delta_y = y_range * range_fraction

    # x_min = x_min - delta_x
    # x_max = x_max + delta_x

    # Y-Values
    y_min = positions[:, 1].min()
    y_max = positions[:, 1].max()
    y_range = y_max - y_min

    # y_min = y_min - delta_y
    # y_max = y_max + delta_y

    # Returns the ranges
    return [x_min - delta_x, x_max + delta_x], [y_min - delta_y, y_max + delta_y]


def create_kmeans_obstacle_physically_consistent(
    centers, save_figure=False, data_name="", figsize=None
):
    if len(data_name):
        file_name = data_name + ".mat"
    else:
        data_name = "2D_messy-snake"
        file_name = "2D_messy-snake.mat"

    data = HandwrittingHandler(file_name=file_name)

    x_lim, y_lim = get_min_max_from_data(data.position)

    if figsize is None:
        ratio = (y_lim[1] - y_lim[0]) / (x_lim[1] - x_lim[0])
        if ratio < 0.75:
            height = 7
            figsize = (height / ratio, height)
        else:
            wdith = 5.5
            figsize = (ratio, width * ratio)

    main_learner = KMeansMotionLearner.from_centers(centers, data=data)

    fig, ax = plt.subplots(figsize=figsize)
    reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    ax.axis("equal")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        fig_name = f"{data_name}_data_only"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    main_learner.plot_boundaries(ax=ax)
    main_learner.plot_kmeans(ax=ax, x_lim=x_lim, y_lim=y_lim, centerlabel=False)

    ax.axis("equal")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        fig_name = f"{data_name}_kmeans_boundary"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=figsize)
    plot_region_dynamics(main_learner, x_lim, y_lim, ax=ax)

    reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

    plot_trajectories(ax, main_learner)

    ax.axis("equal")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:  #
        fig_name = f"{data_name}_global_dyn amics_and_trajectories"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    start_global_matlab_engine = True
    if start_global_matlab_engine and not "matlab_eng" in locals():
        # TODO: this should be included in the upcoming learning-library (!)
        warnings.warn(
            "This requires MATLAB setup. See following repository for more information: \n"
            + "https://github.com/nbfigueroa/phys-gmm"
            "Additionally install the matlab interface: \n"
            + "$ cd 'matlabroot\extern\engines\python \n'"
            + "$ python -m pip install ."
        )

        import matlab
        import matlab.engine

        matlab_eng = matlab.engine.start_matlab()

    plt.ion()
    plt.close("all")

    data_name = "2D_messy-snake"
    # plot_a_shape_partial_motions(save_figure=False)
    # plot_snake_partial_motions(save_figure=True)
    # plot_kmeans_messy_snake(save_figure=True)

    # To not have to redo the whole optimization
    if True:
        # Messy snake with Physically Consistent
        try:
            # priors, mu, sigma = plot_snake_partial_motions()
            create_kmeans_obstacle_physically_consistent(mu, save_figure=True)

        except:
            priors, mu, sigma = plot_snake_partial_motions()
            create_kmeans_obstacle_physically_consistent(mu, save_figure=True)

    print("Tests finished.")
