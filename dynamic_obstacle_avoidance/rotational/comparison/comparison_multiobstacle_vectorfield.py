"""
Library for the Rotation (Modulation Imitation) of Linear Systems
"""
# Author: Lukas Huber
# GitHub: hubernikus
# Created: 2022-01-19

# import warnings
import os
import math
from typing import Optional

import json


import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

import matplotlib.pyplot as plt

from vartools.directional_space import get_directional_weighted_sum

# from vartools.directional_space DirectionBase
from dynamic_obstacle_avoidance.rotational.dynamics.circular_dynamics import (
    # CircularRotationDynamics,
    SimpleCircularDynamics,
)
from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
    plot_obstacle_dynamics,
)
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import Obstacle

from dynamic_obstacle_avoidance.avoidance import ModulationAvoider

from dynamic_obstacle_avoidance.rotational.nonlinear_rotation_avoider import (
    NonlinearRotationalAvoider,
)
from dynamic_obstacle_avoidance.rotational.rotation_container import RotationContainer
from dynamic_obstacle_avoidance.rotational.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)


def integrate_trajectory(
    evaluate_function,
    start_position,
    it_max,
    delta_time=0.01,
    conv_margin=1e-3,
    collision_functor=None,
):
    positions = np.zeros((start_position.shape[0], it_max))
    positions[:, 0] = start_position

    for ii in range(1, it_max):
        velocity = evaluate_function(positions[:, ii - 1])

        if LA.norm(velocity) < conv_margin:
            print(f"Reached local minima at it={ii}")
            return positions[:, : (ii - 1)]

        # Normalize to 1 such that the algorithms get comparable
        velocity = velocity / LA.norm(velocity)

        positions[:, ii] = velocity * delta_time + positions[:, ii - 1]

        # Normalize or not(?) -> for now not...
        if collision_functor is not None and collision_functor(positions[:, ii]):
            print(f"Converged at it={ii}")
            return positions[:, :ii]

    print(f"Maximum iterations of {ii} reached.")
    return positions


def create_initial_dynamics():
    return SimpleCircularDynamics()


# def create_initial_dynamics():
#     return CircularRotationDynamics(
#         radius=2.0,
#         maximum_velocity=2.0,
#         outside_approaching_factor=8.0,
#         inside_approaching_factor=4.0,
#     )


def create_six_obstacle_environment(
    distance_scaling=0.3, tail_effect=False
) -> RotationContainer:
    obstacle_environment = RotationContainer()

    r_topcirc = 0.6
    obstacle_environment.append(
        Ellipse(
            center_position=np.array([-0.9, 2.0]),
            axes_length=np.ones(2) * r_topcirc,
            distance_scaling=distance_scaling,
            tail_effect=tail_effect,
            # margin_absolut=0.3,
        )
    )

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([0.9, 2.0]),
            # axes_length=np.array([1.4, 1.4]),
            axes_length=np.ones(2) * r_topcirc,
            distance_scaling=distance_scaling,
            tail_effect=tail_effect,
            # margin_absolut=0.3,
        )
    )

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([-2.6, 0.0]),
            axes_length=np.array([0.8, 1.8]),
            distance_scaling=distance_scaling,
            tail_effect=tail_effect,
        )
    )

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([-1.4, 0.0]),
            axes_length=np.array([0.8, 1.8]),
            distance_scaling=distance_scaling,
            tail_effect=tail_effect,
        )
    )

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([0.0, -2.0]),
            axes_length=np.array([1.0, 0.5]),
            distance_scaling=distance_scaling,
            tail_effect=tail_effect,
        )
    )

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([2.0, 0.0]),
            axes_length=np.array([1.0, 0.5]),
            orientation=45.0 / 180 * math.pi,
            distance_scaling=distance_scaling,
            tail_effect=tail_effect,
        )
    )
    return obstacle_environment


def visualize_circular_dynamics_multiobstacle_nonlinear(
    visualize=True,
    n_resolution: int = 20,
    savefig: bool = False,
    traj_integrate: bool = False,
):
    obstacle_environment = create_six_obstacle_environment()
    initial_dynamics = create_initial_dynamics()

    rotation_projector = ProjectedRotationDynamics(
        attractor_position=initial_dynamics.pose.position,
        initial_dynamics=circular_ds,
        # reference_velocity=lambda x: x - center_velocity.center_position,
    )

    obstacle_avoider = NonlinearRotationalAvoider(
        initial_dynamics=initial_dynamics,
        # convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=rotation_projector,
    )

    if visualize:
        x_lim = [-4.0, 4.0]
        y_lim = [-3.0, 3.0]

        vf_color = "blue"
        figname = "nonlinear_infinite_dynamics"
        # vf_color = "black"

        figsize = (8.0, 8.0)

        from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
            plot_obstacle_dynamics,
        )
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=obstacle_environment,
            dynamics=obstacle_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            do_quiver=True,
            vectorfield_color=vf_color,
        )
        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            # noTicks=True,
            # show_ticks=False,
        )

    if traj_integrate:
        traj_positions = integrate_trajectory(
            obstacle_avoider.evaluate,
            start_position=np.array([-3, 1.5]),
            it_max=2000,
            delta_time=0.01,
        )
        ax.plot(traj_positions[0, :], traj_positions[1, :], color="black")
        traj_positions = integrate_trajectory(
            obstacle_avoider.evaluate,
            start_position=np.array([-0.5, -0.5]),
            it_max=2000,
            delta_time=0.01,
        )
        ax.plot(traj_positions[0, :], traj_positions[1, :], color="black")


def visualize_circular_dynamics_multiobstacle_modulation(
    n_resolution=20, visualize=True
):
    obstacle_environment = create_six_obstacle_environment()
    initial_dynamics = create_initial_dynamics()

    obstacle_avoider = ModulationAvoider(
        initial_dynamics=inital_dynamics,
        obstacle_environment=obstacle_environment,
    )

    if visualize:
        x_lim = [-4.0, 4.0]
        y_lim = [-3.0, 3.0]

        vf_color = "blue"
        figname = "nonlinear_infinite_dynamics"
        # vf_color = "black"

        figsize = (8.0, 8.0)

        from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
            plot_obstacle_dynamics,
        )
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=obstacle_environment,
            dynamics=obstacle_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            do_quiver=True,
            vectorfield_color=vf_color,
        )
        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            # noTicks=True,
            # show_ticks=False,
        )


def create_start_positions(
    n_grid, x_lim, y_lim, obstacle_environment, datapath, store_to_file=False
):
    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], n_grid),
        np.linspace(y_lim[0], y_lim[1], n_grid),
    )
    positions = np.array([xx.flatten(), yy.flatten()])

    # Keep only positions with no collision
    ind_outside = np.zeros(positions.shape[1], dtype=bool)
    for pp in range(positions.shape[1]):
        if obstacle_environment.get_minimum_gamma(positions[:, pp]) > 1:
            ind_outside[pp] = 1

    positions = positions[:, ind_outside]

    if store_to_file:
        filename = "initial_positions.csv"
        np.savetxt(
            os.path.join(datapath, filename),
            positions.T,
            header="x, y",
            delimiter=",",
        )


def evaluate_trajectories(
    datapath,
    avoidance_functor,
    outputfolder,
    inputfile="initial_positions.csv",
    parameter_file="comparison_parameters.json",
    store_to_file=True,
    collision_functor=None,
):
    with open(os.path.join(datapath, "..", parameter_file)) as user_file:
        simulation_parameters = json.load(user_file)

    start_positions = np.loadtxt(
        os.path.join(datapath, inputfile),
        delimiter=",",
        dtype=float,
        skiprows=1,
    )
    start_positions = start_positions.T

    output_path = os.path.join(datapath, outputfolder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for pp in range(start_positions.shape[1]):
        print(f"Starting simulation {pp+1}/{start_positions.shape[1]} ...")
        trajectory = integrate_trajectory(
            start_position=start_positions[:, pp],
            evaluate_function=avoidance_functor,
            collision_functor=collision_functor,
            **simulation_parameters,
        )

        filename = f"trajectory{pp:03d}.csv"
        np.savetxt(
            os.path.join(output_path, filename),
            trajectory.T,
            # header="x, y", # No header - as matlab does not have one
            delimiter=",",
        )


def evaluate_nonlinear_trajectories():
    obstacle_environment = create_six_obstacle_environment()
    initial_ds = create_initial_dynamics()

    rotation_projector = ProjectedRotationDynamics(
        attractor_position=initial_ds.pose.position,
        initial_dynamics=initial_ds,
        reference_velocity=lambda x: x - center_velocity.center_position,
    )

    obstacle_avoider = NonlinearRotationalAvoider(
        initial_dynamics=initial_ds,
        # convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=rotation_projector,
    )

    evaluate_trajectories(
        datapath=datapath,
        outputfolder="nonlinear_avoidance",
        avoidance_functor=obstacle_avoider.evaluate,
        collision_functor=lambda x: obstacle_environment.get_minimum_gamma(x) <= 1,
    )


def evaluate_modulated_trajectories():
    obstacle_environment = create_six_obstacle_environment()
    initial_ds = create_initial_dynamics()

    modulation_avoider = ModulationAvoider(
        initial_dynamics=initial_ds,
        obstacle_environment=obstacle_environment,
    )
    evaluate_trajectories(
        datapath=datapath,
        outputfolder="modulation_avoidance",
        avoidance_functor=modulation_avoider.evaluate,
        collision_functor=lambda x: obstacle_environment.get_minimum_gamma(x) <= 1,
    )


def evaluate_original_trajectories():
    initial_ds = create_initial_dynamics()

    evaluate_trajectories(
        datapath=datapath,
        outputfolder="original_trajectories",
        avoidance_functor=initial_ds.evaluate,
        collision_functor=lambda x: False,
    )


def visualize_trajectories(
    datapath, datafolder="modulation_avoidance", uni_color=None, ax=None
):
    obstacle_environment = create_six_obstacle_environment()

    datafolder_path = os.path.join(datapath, datafolder)
    files = os.listdir(datafolder_path)

    # color_list = ["blue", "red", "black", "orange", "magenta"]
    if ax is None:
        figsize = (8, 6.5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    # n_simu = 100
    n_simu = len(files)
    color = uni_color

    # for ii, color in enumerate(color_list):
    for ii in range(n_simu):
        if ii >= len(files):
            print(f"We haven't got that many files - we stop visualizing at {ii}")
            break
        inputfile = files[ii]  # For now just take the first one

        if uni_color is None:
            color = np.random.rand(3)

        trajectory = np.loadtxt(
            os.path.join(datafolder_path, inputfile),
            delimiter=",",
            dtype=float,
            skiprows=0,
        )

        if not len(trajectory):
            warnings.warn("Empty trajectory file.")
            continue

        trajectory = trajectory.T
        ax.plot(trajectory[0, :], trajectory[1, :], color=color)
        ax.plot(trajectory[0, 0], trajectory[1, 0], "o", color=color)
        ax.plot(trajectory[0, -1], trajectory[1, -1], ".", color=color)

    x_lim = [-4, 4]
    y_lim = [-4, 4]

    plot_obstacles(
        ax=ax,
        obstacle_container=obstacle_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        # noTicks=True,
        # show_ticks=False,
    )

    plt.ion()
    return fig, ax


def create_base_circles_to_file(datapath, parameter_file="comparison_parameters.json"):
    it_max = 1500
    obstacle_environment = create_six_obstacle_environment()
    initial_ds = create_initial_dynamics()

    with open(os.path.join(datapath, "..", parameter_file)) as user_file:
        simulation_parameters = json.load(user_file)
    simulation_parameters["it_max"] = it_max

    # Guiding_field
    # start_position = np.array([-0.966468219, 2.43618389])
    # output_folder = "guiding_field"

    # Modulation
    output_folder = "modulation_avoidance"
    start_position = np.array([0.537017, -1.99187])

    avoider = ModulationAvoider(
        initial_dynamics=initial_ds,
        obstacle_environment=obstacle_environment,
    )
    trajectory = integrate_trajectory(
        start_position=start_position,
        evaluate_function=avoider.evaluate,
        collision_functor=lambda x: obstacle_environment.get_minimum_gamma(x) <= 1,
        **simulation_parameters,
    )
    np.savetxt(
        os.path.join(datapath, output_folder + "_cycle" + ".csv"),
        trajectory.T,
        # header="x, y", # No header - as matlab does not have one
        delimiter=",",
    )

    # Nonlinear
    start_position = np.array([2.5405777, 0.4703698])
    output_folder = "nonlinear_avoidance"
    rotation_projector = ProjectedRotationDynamics(
        attractor_position=initial_ds.pose.position,
        initial_dynamics=initial_ds,
        reference_velocity=lambda x: x - center_velocity.center_position,
    )
    avoider = obstacle_avoider = NonlinearRotationalAvoider(
        initial_dynamics=initial_ds,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=rotation_projector,
    )
    trajectory = integrate_trajectory(
        start_position=start_position,
        evaluate_function=avoider.evaluate,
        collision_functor=lambda x: obstacle_environment.get_minimum_gamma(x) <= 1,
        **simulation_parameters,
    )

    np.savetxt(
        os.path.join(datapath, output_folder + "_cycle" + ".csv"),
        trajectory.T,
        # header="x, y", # No header - as matlab does not have one
        delimiter=",",
    )


def plot_trajectory_comparison(datapath, savefig=False):
    fig, ax = plt.subplots(figsize=(6, 5))

    # Create base environment
    obstacle_environment = create_six_obstacle_environment()
    initial_ds = create_initial_dynamics()

    # Plot base dynamics
    # files = os.listdir(datapath)
    # files = [ff for ff in files if ff[-9:] == "cycle.csv"]

    zorders = [0, 0, 0, -1]
    colors = ["red", "green", "blue", "gray"]
    datafolders = [
        "nonlinear_avoidance",
        "modulation_avoidance",
        "guiding_field",
        "original_trajectories",
    ]
    labels = ["ROAM", "MuMo", "VC-CAPF", "Original"]

    line_kwargs = {
        "linewidth": 2,
        # "linestyle": "dashed"
    }

    # Create legend
    for ii, color in enumerate(colors):
        ax.plot([], [], colors[ii], **line_kwargs, label=labels[ii])
    ax.legend(loc="upper left")

    for ii, algo_type in enumerate(datafolders):
        if algo_type == "original_trajectories":
            continue

        filename = algo_type + "_" + "cycle.csv"
        trajectory = np.loadtxt(
            os.path.join(datapath, filename),
            delimiter=",",
            dtype=float,
            skiprows=0,
        )

        trajectory = trajectory.T
        ax.plot(trajectory[0, :], trajectory[1, :], colors[ii], **line_kwargs)

    # Plot base circle
    angles = np.linspace(0, 2 * math.pi, 100)
    traj_base = np.vstack((np.cos(angles), np.sin(angles))) * initial_ds.R
    ax.plot(traj_base[0, :], traj_base[1, :], color="gray", **line_kwargs, zorder=-1)

    center_position = np.zeros(2)
    ax.plot(center_position[0], center_position[1], "*", color="black")

    positions = [
        [-0.4, 1.25],
        # [-0.5, -1.4],
        [-0.45, -0.44],
        # [0.45, 0.4],
        # [-3.07, 1.31],
        # [3.13, -1.35],
        [2.25, 3.94],
        # [-1.33, -3.94],
        # [-4, -4],
        [-2.2, -4],
        # [4, -2.22],
        [4, -4],
        [4, 0.41],
        [-4, 0.4],
        [0.48, -0.46],
    ]

    start_positions = np.loadtxt(
        os.path.join(datapath, "initial_positions.csv"),
        delimiter=",",
        dtype=float,
        skiprows=1,
    )
    start_positions = start_positions.T

    abs_tol = 0.3
    indexes = np.zeros(len(positions), dtype=int)
    for pp, pos in enumerate(positions):
        value_index = np.ones(start_positions.shape[1], dtype=bool)
        for dd in range(start_positions.shape[0]):
            value_index = np.logical_and(
                value_index, np.abs(start_positions[dd, :] - pos[dd]) < abs_tol
            )
        try:
            indexes[pp] = np.where(value_index)[0][0]
        except:
            breakpoint()

        filename = f"trajectory{indexes[pp]:03d}.csv"
        for aa, folder in enumerate(datafolders):
            trajectory = np.loadtxt(
                os.path.join(datapath, folder, filename),
                delimiter=",",
                dtype=float,
                skiprows=0,
            )

            trajectory = trajectory.T
            ax.plot(
                trajectory[0, :],
                trajectory[1, :],
                colors[aa],
                linewidth=2,
                linestyle=":",
                zorder=zorders[aa],
            )
            # ax.plot(trajectory[0, -1], trajectory[1, -1], ".", color=colors[aa])

        # They all have the same start position
        ax.plot(trajectory[0, 0], trajectory[1, 0], ".", color="black")

    # visualize_trajectories(
    #     datapath, datafolder="nonlinear_avoidance", uni_color="blue", ax=ax
    # )
    # visualize_trajectories(
    #     datapath, datafolder="modulation_avoidance", uni_color="red", ax=ax
    # )
    # visualize_trajectories(
    #     datapath, datafolder="guiding_field", uni_color="green", ax=ax
    # )

    x_lim = [-4, 4]
    y_lim = [-4, 4]

    plot_obstacles(
        ax=ax,
        obstacle_container=obstacle_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        # noTicks=True,
        # show_ticks=False,
    )

    plt.grid(True)

    if savefig:
        figname = "base_convergence"
        plt.savefig(
            "figures/" + "comparison_algorithms_" + figname + figtype,
            bbox_inches="tight",
        )


if (__name__) == "__main__":
    datapath = "/home/lukas/Code/dynamic_obstacle_avoidance/dynamic_obstacle_avoidance/rotational/comparison/data"
    figtype = ".pdf"
    # figtype = ".png"

    # create_start_positions(
    #     # n_grid=5,
    #     n_grid=10,
    #     x_lim=[-3.5, 3],
    #     y_lim=[-2.8, 2.8],
    #     obstacle_environment=create_six_obstacle_environment(),
    #     datapath=datapath,
    #     store_to_file=True,
    # )

    # visualize_circular_dynamics_multiobstacle_nonlinear(n_resolution=20)
    # visualize_circular_dynamics_multiobstacle_modulation(n_resolution=20)

    # evaluate_nonlinear_trajectories()
    # evaluate_modulated_trajectories()
    # evaluate_original_trajectories()

    # visualize_trajectories(datapath, datafolder="nonlinear_avoidance")
    # visualize_trajectories(datapath, datafolder="modulation_avoidance")
    # visualize_trajectories(datapath, datafolder="guiding_field")
    # visualize_trajectories(datapath, datafolder="original_trajectories")

    # create_base_circles_to_file(datapath=datapath)
    plot_trajectory_comparison(datapath=datapath, savefig=False)
