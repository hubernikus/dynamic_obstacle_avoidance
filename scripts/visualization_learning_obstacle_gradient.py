#!/USSR/bin/python3
""" Script which creates a variety of examples of local modulation of a vector
field with obstacle avoidance. 
"""
# Author: Lukas Huber
# Created: 2019-10-17

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd  # pd is not necessarily needed. only for importing data.

import copy
import json  # store/load data

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import *

from dynamic_obstacle_avoidance.obstacle_avoidance.learning_obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import *

# data_set_path = "/home/lukas/Code/ObstacleAvoidance/icub_SCA/SVM_training_toy_example/libsvm_data/datasets_train/"
data_set_path = "../data/"
use_test_data = False
if use_test_data:
    file_name_in = "datasets_test/11_icub_dataset_libsvm_10k"
else:
    file_name_in = "datasets_train/11_icub_dataset_libsvm_20k"


def create_obstacle_from_data(file_in_path, file_in_name, save_data=True):
    file_type = ".txt"
    file_name = file_in_path + file_in_name + file_type

    df = pd.read_csv(file_name, sep="[:, ]", engine="python")
    data = np.asarray(df)
    data = np.delete(data, (1, 3), axis=1)
    X = data[:, 1:].T
    y = data[:, 0].T

    Obstacles = LearningContainer()
    Obstacles.create_obstacles_from_data(data=X, label=y, plot_raw_data=True)

    return Obstacles

    # TODO: save data to file
    # if save_data:
    # file_name = file_in_path+file_in_name + "csv"
    # pickle.dump(Obstacles, file_name)


def read_obstacle_from_file(file_in_path, file_name):
    # TODO: make it work
    file_type = ".txt"
    file_name = file_in_path + file_in_name + file_type


def create_normal_vector_field(
    Obstacles, resolution=20, x_range=[0, 1], y_range=[0, 1]
):
    YY, XX = np.mgrid[
        y_range[0] : y_range[1] : resolution * 1j,
        x_range[0] : x_range[1] : resolution * 1j,
    ]

    # XX = XX.flatten()
    # YY = YY.flatten()

    col_lists = ["red", "blue"]

    if False:
        # for oo in range(len(Obstacles)):
        normal_vecs = Obstacles[oo].get_normal_direction(
            position=np.c_[XX.ravel, YY.ravel].T
        )
        ref_dirs = Obstacles[oo].get_reference_direction(
            position=np.c_[XX.ravel, YY.ravel].T, in_global_frame=True
        )
        fig, ax = plt.subplots()
        Obstacles[oo].draw_obstacle(show_contour=False, fig=fig, ax=ax)
        ax.quiver(XX, YY, normal_vecs[0, :], normal_vecs[1, :], color=col_lists[0])
        plt.axis("equal")

    res_x = res_y = resolution
    normal_vecs = np.zeros((Obstacles[0].dim, res_x, res_y))
    ref_dirs = np.zeros((Obstacles[0].dim, res_x, res_y))
    for oo in range(len(Obstacles)):
        for ix in range(res_x):
            for iy in range(res_y):
                pos = np.array([XX[ix, iy], YY[ix, iy]])
                E, E_orth = compute_decomposition_matrix(
                    x_t=pos, obs=Obstacles[oo], in_global_frame=True
                )
                normal_vecs[:, ix, iy] = E_orth[:, 0]
                ref_dirs[:, ix, iy] = E[:, 0]

        fig, ax = plt.subplots()
        Obstacles[oo].draw_obstacle(show_contour=False, fig=fig, ax=ax)
        ax.quiver(XX, YY, normal_vecs[0, :], normal_vecs[1, :], color="red")
        ax.quiver(XX, YY, ref_dirs[0, :], ref_dirs[1, :], color="blue")


def create_modulated_vector_field(
    Obstacles,
    pos_attractor=None,
    resolution=80,
    x_range=[0, 1],
    y_range=[0, 1],
    plot_type_quiver=True,
    max_vel=0.1,
):
    YY, XX = np.mgrid[
        y_range[0] : y_range[1] : resolution * 1j,
        x_range[0] : x_range[1] : resolution * 1j,
    ]

    # XX = XX.flatten()
    # YY = YY.flatten()

    res_x = res_y = resolution
    ds_initial = np.zeros((Obstacles[0].dimension, res_x, res_y))
    ds_modulated = np.zeros((Obstacles[0].dimension, res_x, res_y))

    for ix in range(res_x):
        for iy in range(res_y):
            pos = np.array([XX[ix, iy], YY[ix, iy]])

            ds_initial[:, ix, iy] = linear_ds_max_vel(
                pos, attractor=pos_attractor, max_vel=max_vel
            )

            ds_modulated[:, ix, iy] = obs_avoidance_interpolation_moving(
                pos,
                ds_initial[:, ix, iy],
                Obstacles,
                evaluate_in_global_frame=True,
                zero_vel_inside=True,
                velocicity_max=max_vel,
            )

    # fig, ax = plt.figure(figsize=(7,6))
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 6)

    if plot_type_quiver:

        ax = plt.subplot(1, 2, 1)
        for oo in range(len(Obstacles)):
            Obstacles[oo].draw_obstacle(show_contour=False, fig=fig, ax=ax)
        qv1 = ax.quiver(
            XX.flatten(),
            YY.flatten(),
            ds_initial[0, :, :].flatten(),
            ds_initial[1, :, :].flatten(),
            color="#AE204F",
        )
        plt.title("Initial DS")
        plt.axis("equal")

        ax = plt.subplot(1, 2, 2)
        for oo in range(len(Obstacles)):
            Obstacles[oo].draw_obstacle(show_contour=False, fig=fig, ax=ax)

        qv2 = ax.quiver(
            XX.flatten(),
            YY.flatten(),
            ds_modulated[0, :, :].flatten(),
            ds_modulated[1, :, :].flatten(),
            color="#0C537C",
        )
        plt.title("Modulated DS")
        plt.axis("equal")
        # plt.quiverkey(qv1, 0.9, 1.05, 1, "Initial DS", coordinates='data')
        # plt.quiverkey(qv2, 0.9, 1.15, 1, "Modulated DS", coordinates='data')

    else:
        ax = plt.subplot(1, 2, 1)
        for oo in range(len(Obstacles)):
            Obstacles[oo].draw_obstacle(show_contour=False, fig=fig, ax=ax)
        str1 = plt.streamplot(
            XX, YY, ds_initial[0, :, :], ds_initial[1, :, :], color="#A22828"
        )
        plt.title("Initial DS")
        plt.axis("equal")

        ax = plt.subplot(1, 2, 2)
        for oo in range(len(Obstacles)):
            Obstacles[oo].draw_obstacle(show_contour=False, fig=fig, ax=ax)

        str2 = plt.streamplot(
            XX, YY, ds_modulated[0, :, :], ds_modulated[1, :, :], color="#0C537C"
        )
        plt.title("Modulated DS")
        plt.axis("equal")

    plt.legend()
    # for oo, cols in  zip(range(len(Obstacles)), col_lists):

    # normal_vecs = Obstacles[oo].get_normal_direction(position=np.vstack((XX, YY)))
    # Obstacles[oo].draw_obstacle(show_contour=False, fig=fig, ax=ax)
    # ax.quiver(XX, YY, normal_vecs[0, :], normal_vecs[1, :], color=cols)


if (__name__) == "__main__":
    # Show plot
    plt.ion()
    plt.close("all")

    # del Obstacles

    if not ("Obstacles" in locals()):
        print("Imported obstacles")
        Obstacles = create_obstacle_from_data(data_set_path, file_name_in)

    for oo in range(len(Obstacles)):
        Obstacles[oo].draw_obstacle(show_contour=True)
        Obstacles[oo].draw_obstacle(show_contour=True, gamma_value=True)

    # create_normal_vector_field(Obstacles)

    attractor = np.array([0.8, 0.2])
    # create_modulated_vector_field(Obstacles, attractor, resolution=30, plot_type_quiver=True)
    create_modulated_vector_field(
        Obstacles, attractor, resolution=80, plot_type_quiver=False
    )

    # read_obstacle_from_file(file_in_path, file_in_name)
