#!/USSR/bin/python3
"""
Script which creates a variety of examples of local modulation of a vector field with obstacle avoidance. 

@author LukasHuber
@date 2019-10-17
"""

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt

import copy

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import *

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.visualization.animated_simulation import (
    run_animation,
    samplePointsAtBorder,
)

from dynamic_obstacle_avoidance.obstacle_avoidance.learning_obstacle import *

from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import *
import json  # store/load data


use_test_data = True
if use_test_data:
    data_set_path = "/home/lukas/Code/ObstacleAvoidance/icub_SCA/SVM_training_toy_example/libsvm_data/datasets_test/"
    file_name_in = "11_icub_dataset_libsvm_10k"
else:
    data_set_path = "/home/lukas/Code/ObstacleAvoidance/icub_SCA/SVM_training_toy_example/libsvm_data/datasets_train/"
    file_name_in = "11_icub_dataset_libsvm_20k"

# model_path =

from sklearn import svm

# from sklearn.cluster import DBSCAN
import pandas as pd  # pd is not necessarily needed. only for importing data.


def get_boundary_values(
    file_in_path, file_in_name, plot_raw_data=False, save_data=True
):
    file_type = ".txt"
    file_name = file_in_path + file_in_name + file_type

    df = pd.read_csv(file_name, sep="[:, ]", engine="python")
    data = np.asarray(df)
    data = np.delete(data, (1, 3), axis=1)
    X = data[:, 1:]
    y = data[:, 0]

    x_limit = [0, 1]
    y_limit = [0, 1]

    if plot_raw_data:
        plt.figure()
        class_value = 0
        plt.plot(X[y == class_value, 0], X[y == class_value, 1], ".g")
        class_value = 1
        plt.plot(X[y == class_value, 0], X[y == class_value, 1], ".b")
        plt.axis("equal")
        plt.xlim(x_limit)
        plt.ylim(y_limit)

    cassifier_svm = svm.SVC(kernel="rbf", gamma=20, C=20.0)
    model = cassifier_svm.fit(X, y)
    # model = model.fit
    xx, yy = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))

    prediction_svm = cassifier_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    prediction_svm = prediction_svm.reshape(xx.shape)
    predict_score = cassifier_svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    predict_score = predict_score.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, prediction_svm, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.show()

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, predict_score, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.show()

    margin_decision = 0.8
    is_uncertain_point = np.abs(predict_score) < margin_decision
    boundary_points = np.vstack((xx[is_uncertain_point], yy[is_uncertain_point])).T

    plt.plot(boundary_points[:, 0], boundary_points[:, 1], "k.")

    if save_data:
        file_out_path = (
            "/home/lukas/Code/ObstacleAvoidance/dynamic_obstacle_avoidance/data/"
        )
        file_out_name = file_in_name

        np.save(file_out_path + file_out_name, boundary_points)

    return boundary_points


def import_svm_from_file(file_name):
    pass


def visualize_vectorfield(sensor_data, show_plot=True):
    obs_list = get_obstacle_from_scan(sensor_data=sensor_data, input_is_polar=False)
    # import pdb; pdb.set_trace() ## DEBUG ##

    if False:
        x_limit, y_limit = [0, 1], [0, 1]
        fig, ax = plt.subplots()
        plt.plot(sensor_data[:, 0], sensor_data[:, 1], ".k")
        # plt.axis('equal')
        plt.grid()
        plt.xlim(x_limit), plt.ylim(y_limit)

    print("number of obstacles: " + str(len(obs_list)))

    edge_points = [[0, 1, 1, 0], [0, 0, 1, 1]]

    ObstacleBoundary = PolygonWithLearnedSurface(
        edge_points=edge_points, center_position=[0.5, 0.5], is_boundary=True
    )

    ObstacleBoundary.find_boundary_surfaces(obs_list)
    ObstacleBoundary.special_surfaces[0].learn_surface(epsilon=0.001, C=100, gamma=100)
    ObstacleBoundary.intersection_with_special_surfaces()
    ObstacleBoundary.draw_obstacle()

    if True:
        fig, ax = plt.subplots()
        plt.plot(ObstacleBoundary.x_obs[:, 0], ObstacleBoundary.x_obs[:, 1], "k")

        for ii in range(len(ObstacleBoundary.x_obs_special)):
            plt.plot(
                ObstacleBoundary.x_obs_special[ii][:, 0],
                ObstacleBoundary.x_obs_special[ii][:, 1],
                "k-.",
            )
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.axis("equal")

    for ii in range(len(obs_list)):
        obs_list[ii].learn_surface(epsilon=0.005, C=100, gamma=10.0)
        obs_list[ii].draw_obstacle()
        plt.plot(obs_list[ii].x_obs[:, 0], obs_list[ii].x_obs[:, 1], "k")

    n_grid_x = 50
    n_grid_y = n_grid_x
    YY, XX = np.mgrid[-0.1 : 1.1 : n_grid_x * 1j, -0.1 : 1.1 : n_grid_y * 1j]
    # YY, XX = np.mgrid[0.05:0.95:n_grid_x*1j,
    # 0.05:0.95:n_grid_y*1j]

    # Only for Development and testing
    ########## START REMOVE ##########
    # N_x = N_y = 1
    # XX = np.zeros((N_x, N_y))
    # YY = np.zeros((N_x, N_y))
    if True:
        it_start = 0
        n_samples = 0

        pos1 = [0.85, 0.45]
        pos2 = [0.855, 0.455]

        x_sample_range = [pos1[0], pos2[0]]
        y_sample_range = [pos1[1], pos2[1]]

        x_sample = np.linspace(x_sample_range[0], x_sample_range[1], n_samples)
        y_sample = np.linspace(y_sample_range[0], y_sample_range[1], n_samples)

        ii = 0
        for ii in range(n_samples):
            iy = (ii + it_start) % n_grid_y
            ix = int((ii + it_start) / n_grid_x)

            XX[ix, iy] = x_sample[ii]
            YY[ix, iy] = y_sample[ii]

    ########## STOP REMOVE ###########

    ds_init = np.zeros((2, n_grid_x, n_grid_y))
    ds_mod = np.zeros((2, n_grid_x, n_grid_y))

    pos_attractor = [0.65, 0.75]
    ax.plot(pos_attractor[0], pos_attractor[1], "k*")

    obs_list.append(ObstacleBoundary)
    # del obs_list[0]

    for ii in range(len(obs_list)):
        ax.plot(obs_list[ii].center_position[0], obs_list[ii].center_position[1], "k+")

    # TODO debugging
    # ObstacleBoundary = Polygon(edge_points=edge_points , center_position=[0.5, 0.5], is_boundary=True)
    # obs_list = [ObstacleBoundary]

    for ix in range(n_grid_x):
        for iy in range(n_grid_y):
            position = np.array([XX[ix, iy], YY[ix, iy]])

            ds_init[:, ix, iy] = linearAttractor(position, pos_attractor)
            ds_mod[:, ix, iy] = obs_avoidance_interpolation_moving(
                position, ds_init[:, ix, iy], obs_list
            )

    # ax.quiver(XX, YY, ds_init[0, :, :], ds_init[1, :, :], color='b')
    # ax.quiver(XX, YY, ds_mod[0, :, :], ds_mod[1, :, :], color='r')

    indOfnoCollision = obs_check_collision_2d(obs_list, XX, YY)

    ds1_noColl = np.squeeze(ds_mod[0, :, :]) * indOfnoCollision
    ds2_noColl = np.squeeze(ds_mod[1, :, :]) * indOfnoCollision
    # ax.quiver(XX, YY, ds1_noColl, ds2_noColl, color='r')
    ax.streamplot(XX, YY, ds1_noColl, ds2_noColl, color="b")

    # for ii in range(len(obs_list)):
    #     obs = obs_list[ii]
    #     for pp in range(obs_list[ii].surface_points.shape[0]):

    #         if ObstacleBoundary.get_gamma(obs.surface_points[pp, :], in_global_frame=True)<1.0:
    #             obs.center_position = copy.deepcopy(ObstacleBouondary.center_position)

    #             n_points = 100
    #             boundary_values = np.vstack((np.linspace(x_limit[0], x_limit[1], n_points), np.ones(n_points)*y_limit[0]))
    #             boundary_values = np.hstack((np.vstack((np.ones(n_points)*x_limit[1], np.linspace(y_limit[0], y_limit[1], n_points))), boundary_values))
    #             boundary_values = np.hstack((np.vstack((np.linspace(x_limit[1], x_limit[0], n_points),np.ones(n_points)*y_limit[1])), boundary_values))
    #             boundary_values = np.hstack((np.vstack((np.ones(n_points)*x_limit[0], np.linspace(y_limit[1], y_limit[0], n_points) )), boundary_values))

    #             boundary_values = boundary_values.T

    #             # obs.extend_with_boundary(boundary_values)
    #             break

    for ii in range(len(obs_list)):
        # for ii in [2]:
        # surface_regressed = obs_list[ii].draw_obstacle()
        # plt.plot(surface_regressed[0,:], surface_regressed[1,:], linewidth=6)
        plt.plot(
            obs_list[ii].center_position[0],
            obs_list[ii].center_position[1],
            "k+",
            markersize=12,
            markeredgewidth=4,
        )

        # obs_list[ii].get_gamma([0,0])
        # obs_list[ii].get_local_radius(0)


# Show plot
plt.ion()
# plt.close('all')

get_from_raw_data = False
if get_from_raw_data:
    boundary_values = get_boundary_values(
        data_set_path, file_name_in, plot_raw_data=True
    )
else:
    boundary_values = np.load(
        "/home/lukas/Code/ObstacleAvoidance/dynamic_obstacle_avoidance/data/"
        + file_name_in
        + ".npy"
    )
    boundary_values = boundary_values.reshape(-1, 2)

visualize_vectorfield(boundary_values)

# classify_data(data_set_path+file_name_test)
