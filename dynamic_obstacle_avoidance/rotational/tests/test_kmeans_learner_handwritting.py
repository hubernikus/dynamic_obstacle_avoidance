"""
Tests (and visualizations) for KmeansMotionLearner and KMeansObstacle.
"""

import sys
import copy
import random
import warnings
import math
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.handwritting_handler import MotionDataHandler, HandwrittingHandler

from dynamic_obstacle_avoidance.rotational.rotational_avoidance import (
    obstacle_avoidance_rotational,
)

from dynamic_obstacle_avoidance.rotational.kmeans_obstacle import KMeansObstacle
from dynamic_obstacle_avoidance.rotational.kmeans_motion_learner import (
    KMeansMotionLearner,
    create_kmeans_obstacle_from_learner,
)

from dynamic_obstacle_avoidance.rotational.base_logger import logger
from dynamic_obstacle_avoidance.rotational.tests.helper_functions import (
    plot_region_dynamics,
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
    RANDOM_SEED = 1
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # data = HandwrittingHandler(file_name="Angle.mat")
    data = HandwrittingHandler(file_name="2D_Ashape.mat")
    main_learner = KMeansMotionLearner(data)

    _test_evaluate_partial_dynamics(
        visualize=True,
        main_learner=main_learner,
        x_lim=[-5.5, 0.5],
        y_lim=[-1.5, 2.5],
        save_figure=save_figure,
        name="a_shape",
    )

    x_lim = [-5.5, 0.5]
    y_lim = [-1.5, 2.5]

    fig, ax = plot_region_dynamics(main_learner, x_lim, y_lim)
    reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    plot_trajectories(ax, main_learner)

    if save_figure:
        fig_name = f"global_dynamics_and_trajectories_a_shape"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


def plot_snake_partial_motions(save_figure=False):
    RANDOM_SEED = 3
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # data = HandwrittingHandler(file_name="2D_messy-snake.mat")

    # data = HandwrittingHandler(file_name="2D_Lshape.mat")
    # x_lim = [-6.5, 0.5]
    # y_lim = [-2.0, 3.0]
    # main_learner = KMeansMotionLearner(data, n_clusters=4)

    data = HandwrittingHandler(file_name="2D_Sshape.mat")
    x_lim = [-6.5, 0.5]
    y_lim = [-2.0, 3.0]
    main_learner = KMeansMotionLearner(data, n_clusters=8)

    fig, ax = plt.subplots()
    main_learner.plot_kmeans(ax=ax)
    reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    main_learner.plot_boundaries(ax=ax)

    index = 0
    tmp_obstacle = create_kmeans_obstacle_from_learner(main_learner, index)

    position = np.array([-3.04769137, -0.12654154])
    ax.plot(position[0], position[1], "ro")

    gamma = tmp_obstacle.get_gamma(position, in_global_frame=True)
    # breakpoint()

    # _test_evaluate_partial_dynamics(
    #     visualize=True,
    #     main_learner=main_learner,
    #     x_lim=x_lim,
    #     y_lim=y_lim,
    #     save_figure=save_figure,
    #     name="snake",
    # )

    fig, ax = plot_region_dynamics(main_learner, x_lim, y_lim)
    reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    plot_trajectories(ax, main_learner)

    if save_figure:
        fig_name = f"global_dynamics_and_trajectories_snake"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    plt.ion()
    # plt.close("all")

    # plot_a_shape_partial_motions(save_figure=False)
    plot_snake_partial_motions(save_figure=False)

    print("Tests finished.")
