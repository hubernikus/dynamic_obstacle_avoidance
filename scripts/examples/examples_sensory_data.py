# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.learning_obstacle import *

import json

import time


def main():
    data = json.load(open("../data/laser_lidar_recording.json"))
    # data = json.load(open("../data/laser_lidar_recording.txt"))

    ii = 1
    sensor_data = {}
    sensor_data["angle"] = (
        np.arange(len(data[ii]["ranges"])) * data[ii]["increment"]
        + data[ii]["angle_min"]
    )
    sensor_data["magnitude"] = np.array(data[ii]["ranges"])

    fig_handle, ax_handle = plt.subplots()

    ObstaclesScanned = ObstacleFromLaser()
    ObstaclesScanned.get_obstacle_from_scan(sensor_data=sensor_data)

    # plt.figure()
    data_cartesian = ObstacleFromLaser.transform_polar2cartesian(
        self=None, magnitude=sensor_data["magnitude"], angle=sensor_data["angle"]
    )
    # plt.plot(data_cartesian[0,:], data_cartesian[1,:], '.')

    plt.ion()
    # plt.figure()
    plt.plot(0, 0, "ko", markersize=20, markeredgewidth=10)

    for ii in range(len(ObstaclesScanned.obs_list)):
        # for ii in [2]:
        surface_regressed = ObstaclesScanned.obs_list[ii].draw_obstacle()
        plt.plot(surface_regressed[0, :], surface_regressed[1, :], linewidth=6)
        plt.plot(
            ObstaclesScanned.obs_list[ii].center_position[0],
            ObstaclesScanned.obs_list[ii].center_position[1],
            "k+",
            markersize=12,
            markeredgewidth=4,
        )

        ObstaclesScanned.obs_list[ii].get_gamma([0, 0])
        ObstaclesScanned.obs_list[ii].get_local_radius(0)

    x_lim = [-6.1, 13.1]
    y_lim = [-7.1, 8.1]

    n_points = 3
    points_init = np.linspace(y_lim[0], y_lim[1], n_points + 2)

    points_init = np.vstack((x_lim[0] * np.ones(n_points), points_init[1:-1]))

    # plot_streamlines_sensory(sensor_data, points_init, ax=ax_handle, attractorPos=[12.5,0],
    # dim=2, dt=0.01, convergence_margin=0.03, max_simu_step=100)

    plt.axis("equal")
    plt.grid(True)
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    plt.show()

    show_quiverPlot = True
    if show_quiverPlot:
        plt.figure()
        for ii in [2]:
            angles = np.linspace(-pi, pi, 100)
            angles_rel = ObstaclesScanned.obs_list[ii].convert_to_relative_angle(angles)
            magnitudes = ObstaclesScanned.obs_list[ii].get_local_radius(angles)

            plt.plot(angles_rel, magnitudes, "k.")

        xAttractor = [10, 4]
        N_resol = 10
        obs = [ObstaclesScanned.obs_list[2]]
        saveFigures = False

        Simulation_vectorFields(
            x_lim,
            y_lim,
            N_resol,
            obs,
            xAttractor=xAttractor,
            saveFigure=saveFigures,
            figName="linearSystem_avoidanceCube",
            noTicks=False,
            figureSize=(6.0, 5),
            automatic_reference_point=False,
        )

    # Simulation_vectorFields(x_lim, y_lim, N_resol, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_avoidanceCube', noTicks=False, figureSize=(6.,5))


def plot_streamlines_sensory(
    sensor_data,
    points_init,
    ax,
    attractorPos=[0, 0],
    dim=2,
    dt=0.01,
    max_simu_step=300,
    convergence_margin=0.03,
):

    n_points = np.array(points_init).shape[1]

    x_pos = np.zeros((dim, max_simu_step + 1, n_points))
    x_pos[:, 0, :] = points_init

    it_count = 0

    time_tot_clustering = 0
    time_tot_obstacleLearning = 0
    time_tot_modulation = 0
    time_tot_createFromScan = 0

    ObstaclesScanned = ObstacleFromLaser()

    start_time = time.time()
    for iSim in range(max_simu_step):

        print("Simulation step #{}".format(iSim))
        for j in range(n_points):
            start_time_createFromScan = time.time()
            time_clust, time_obstacleLearning = ObstaclesScanned.get_obstacle_from_scan(
                sensor_data=sensor_data
            )

            time_tot_createFromScan += time.time() - start_time_createFromScan
            time_tot_clustering += time_clust
            time_tot_obstacleLearning += time_obstacleLearning

            start_time_modulation = time.time()
            x_pos[:, iSim + 1, j] = obs_avoidance_rk4(
                dt,
                x_pos[:, iSim, j],
                obs=ObstaclesScanned.obs_list,
                x0=attractorPos,
                obs_avoidance=obs_avoidance_interpolation_moving,
            )

            time_tot_modulation += time.time() - start_time_modulation

        # Check convergence
        if (
            np.sum(
                (x_pos[:, iSim + 1, :] - np.tile(attractorPos, (n_points, 1)).T) ** 2
            )
            < convergence_margin
        ):
            x_pos = x_pos[:, : iSim + 2, :]

            print("Convergence reached after {} iterations.".format(it_count))
            break

        it_count += 1

    end_time = time.time()
    time_tot = end_time - start_time

    print("Number of points: {}".format(it_count * n_points))
    print(
        "Average time: {} ms".format(
            np.round((time_tot) / (it_count * n_points) * 1000), 5
        )
    )
    print("Modulation calculation total: {} s".format(np.round(time_tot), 4))

    percentage_times = np.round(
        np.array(
            [
                time_tot_createFromScan,
                time_tot_clustering,
                time_tot_obstacleLearning,
                time_tot_modulation,
            ]
        )
        / time_tot
        * 100,
        1,
    )

    print(
        "Percentage time [time to get obstacles from scan: {}%  --  clustering: {}% -- obstacle learning: {}% -- modulation (RK4): {}% ]".format(
            percentage_times[0],
            percentage_times[1],
            percentage_times[2],
            percentage_times[3],
        )
    )

    for j in range(n_points):
        ax.plot(x_pos[0, :, j], x_pos[1, :, j], "--", lineWidth=4)
        ax.plot(x_pos[0, 0, j], x_pos[1, 0, j], "k*", markeredgewidth=4, markersize=13)

    # return x_pos


# if __name__==("__main__"):
if True:
    main()
