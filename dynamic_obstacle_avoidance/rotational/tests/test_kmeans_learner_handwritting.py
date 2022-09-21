 """
Tests (and visualizations) for KmeansMotionLearner and KMeansObstacle.

To run, in the ipython environment:
>>>
run dynamic_obstacle_avoidance/rotational/tests/test_kmeans_learner_handwritting.py

"""

import random
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.handwritting_handler import HandwrittingHandler

# from dynamic_obstacle_avoidance.rotational.kmeans_obstacle import KMeansObstacle
from dynamic_obstacle_avoidance.rotational.kmeans_motion_learner import (
    KMeansMotionLearner,
    create_kmeans_obstacle_from_learner,
)

# from dynamic_obstacle_avoidance.rotational.base_logger import logger
from dynamic_obstacle_avoidance.rotational.tests.helper_functions import (
    plot_region_dynamics,
)
from dynamic_obstacle_avoidance.rotational.tests.test_kmeans_learner_basic_model import (
    _test_evaluate_partial_dynamics,
)

from dynamic_obstacle_avoidance.rotational.tests.helper_functions import (
    plot_boundaries,
    plot_normals,
    plot_gamma,
    plot_reference_dynamics,
)

from dynamic_obstacle_avoidance.rotational.tests.test_kmeans_learner_basic_model import (
    _plot_gamma_of_learner,
)


# fig_dir = "/home/lukas/Code/dynamic_obstacle_avoidance/figures/"
# fig_dir = "figures/"

# Chose figures as either png / pdf
fig_type = ".png"
# fig_type = ".pdf"


def plot_trajectories(
    ax,
    main_learner,
    it_max=200,
    dt=0.1,
    convergence_margin=1e-3,
    dimension=2,
):
    # Trajectory integration
    data = main_learner.data

    for tt in range(data.start_positions.shape[1]):
        print(f"Doing trajectory {tt}")

        positions = np.zeros((dimension, it_max + 1))

        positions[:, 0] = data.start_positions[:, tt]

        for ii in range(it_max):
            velocity = main_learner.evaluate(positions[:, ii])
            positions[:, ii + 1] = velocity * dt + positions[:, ii]

            if LA.norm(positions[:, ii + 1] - positions[:, ii]) < convergence_margin:
                print(f"Trajectory {tt} has converged at it={ii}.")
                positions = positions[:, : ii + 2]
                break

        ax.plot(positions[0, :], positions[1, :], "r")
        ax.plot(positions[0, 0], positions[1, 0], "or")


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
    y_lim = [-2.0, 3.0]

    fig, ax = plot_region_dynamics(main_learner, x_lim, y_lim)
    reduced_data = main_learner.data.X[:, : main_learner.data.dimension]
    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    plot_trajectories(ax, main_learner)

    if save_figure:
        fig_name = f"global_dynamics_and_trajectories_a_shape"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    fig, ax = plt.subplots()
    main_learner.plot_kmeans(ax=ax, x_lim=x_lim, y_lim=y_lim)
    ax.axis("equal")
    if save_figure:
        fig_name = f"kmeans_a_shape"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    fig, ax = _plot_gamma_of_learner(
        main_learner, x_lim, y_lim, hierarchy_passing_gamma=False
    )

    if save_figure:
        fig_name = f"gamma_values_and_trajectories_a_shape"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")

    fig, ax = plt.subplots()
    main_learner.plot_kmeans(ax=ax, x_lim=x_lim, y_lim=y_lim)
    ax.axis("equal")

    fig, ax = _plot_gamma_of_learner(
        main_learner, x_lim, y_lim, hierarchy_passing_gamma=True
    )

    if save_figure:
        fig_name = f"gamma_values_with_transition_a_shape"
        fig.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


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
    data = HandwrittingHandler(file_name="2D_Sshape.mat")
    fig_name = "snake_2d_"


if (__name__) == "__main__":
    plt.ion()
    # plt.close("all")

    plot_a_shape_partial_motions(save_figure=True)
    # plot_snake_partial_motions(save_figure=True)
    # plot_kmeans_messy_snake(save_figure=True)

    print("Tests finished.")
