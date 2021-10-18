"""
Replication of scenarios in double-blob algorithm.
"""
import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.dynamical_systems import LinearSystem
from vartools.dynamical_systems import plot_dynamical_system_streamplot

from dynamic_obstacle_avoidance.obstacles import Sphere
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.avoidance import DynamicModulationAvoider

# from dynamic_obstacle_avoidance.obstacles import DoubleBlob

from double_blob_obstacle import DoubleBlob


def plot_vectorfield(
    ax, dynamic_avoider, x_lim, y_lim, n_grid=100, streamplotcolor="#808080"
):
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], n_grid),
        np.linspace(y_lim[0], y_lim[1], n_grid),
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities = np.zeros(positions.shape)
    for it in range(positions.shape[1]):
        if dynamic_avoider.get_gamma_product(positions[:, it]) <= 1:
            continue
        velocities[:, it] = dynamic_avoider.evaluate(positions[:, it])

    ax.streamplot(
        x_vals,
        y_vals,
        velocities[0, :].reshape(n_grid, n_grid),
        velocities[1, :].reshape(n_grid, n_grid),
        color=streamplotcolor,
        zorder=-2,
    )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


def plot_trajectories(
    ax,
    start_points,
    dynamic_avoider,
    dt_step=0.01,
    it_max=1000,
    linecolor="#008000ff",
    linewidht=4,
    markersize=12,
    plot_attractor=True,
    plot_start=True,
    convergence_margin=1e-3,
    x_lim=None,
    y_lim=None,
):
    # Done for 2D
    dim = 2

    markercolor = linecolor

    for point in start_points:
        position_list = np.zeros((dim, it_max + 1))
        position_list[:, 0] = point

        for ii in range(it_max):
            velocity = dynamic_avoider.evaluate(position_list[:, ii])

            if LA.norm(velocity) < convergence_margin:
                print(f"Converged at it={ii}")
                position_list = position_list[:, : ii + 1]
                break

            position_list[:, ii + 1] = (
                velocity * dt_step + position_list[:, ii]
            )

        plt.plot(
            position_list[0, :],
            position_list[1, :],
            color=linecolor,
            linewidth=linewidht,
        )
        if plot_start:
            plt.plot(
                position_list[0, 0],
                position_list[1, 0],
                "s",
                markersize=markersize,
                color=markercolor,
            )

    if plot_attractor:
        plt.plot(
            dynamic_avoider.initial_dynamics.attractor_position[0],
            dynamic_avoider.initial_dynamics.attractor_position[1],
            "o",
            color=markercolor,
            markersize=markersize,
        )

    if x_lim is not None:
        ax.set_xlim(x_lim)

    if y_lim is not None:
        ax.set_ylim(y_lim)


def plot_single_blob(save_figure=False):
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        DoubleBlob(
            center_position=np.array([0.0, 3.0]),
            a_value=1,
            b_value=1.1,
            tail_effect=False,
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([0.0, 0]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    x_lim = [-2.05, 2.05]
    y_lim = [-0.3, 6.3]

    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    obstacle_environment[0].plot_obstacle(
        ax=ax,
        fill_color="#ff9955",
        outline_color=None,
        plot_center_position=False,
    )

    dynamic_avoider = DynamicModulationAvoider(
        initial_dynamics=initial_dynamics, environment=obstacle_environment
    )

    start_points = [
        [0.01, 6],
        [0.1, 5],
        [-0.1, 4.6],
        [0.1, 4.2],
    ]

    plot_trajectories(
        ax=ax,
        dynamic_avoider=dynamic_avoider,
        start_points=start_points,
        it_max=1000,
    )

    plot_vectorfield(
        ax=ax,
        dynamic_avoider=dynamic_avoider,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=100,
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    if save_figure:
        figure_name = "single_blob_replica"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def plot_single_blob_left(save_figure=False):
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        DoubleBlob(
            center_position=np.array([0.0, 3.0]),
            a_value=1,
            b_value=1.1,
            tail_effect=False,
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([0.0, 0]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    x_lim = [-2.05, 6.05]
    y_lim = [-2.1, 6.3]

    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    obstacle_environment[0].plot_obstacle(
        ax=ax,
        fill_color="#ff9955",
        outline_color=None,
        plot_center_position=False,
    )

    dynamic_avoider = DynamicModulationAvoider(
        initial_dynamics=initial_dynamics, environment=obstacle_environment
    )

    start_points = [
        [4, 4],
    ]

    plot_trajectories(
        ax=ax,
        dynamic_avoider=dynamic_avoider,
        start_points=start_points,
        it_max=1000,
    )

    plot_vectorfield(
        ax=ax,
        dynamic_avoider=dynamic_avoider,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=100,
    )

    ax.tick_params(
        axis="both",
        which="major",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )

    if save_figure:
        figure_name = "single_blob_leftshift_replica"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def plot_double_old(save_figure=False):
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        DoubleBlob(
            center_position=np.array([0.0, 3.0]),
            a_value=1,
            b_value=1.1,
            tail_effect=False,
        )
    )

    obstacle_environment.append(
        DoubleBlob(
            center_position=np.array([0.0, -3.0]),
            a_value=1,
            b_value=1.1,
            tail_effect=False,
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([0.0, 0]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    x_lim = [-5.05, 5.05]
    y_lim = [-6.2, 6.2]

    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    for obs in obstacle_environment:
        obs.plot_obstacle(
            ax=ax,
            fill_color="#ff9955",
            outline_color=None,
            plot_center_position=False,
        )

    dynamic_avoider = DynamicModulationAvoider(
        initial_dynamics=initial_dynamics, environment=obstacle_environment
    )

    start_points = [
        [-4, 2],
        [-4, 4],
        [4, 4],
        [4, -2],
        [4, -4],
        [-4, -4],
    ]

    plot_trajectories(
        ax=ax,
        dynamic_avoider=dynamic_avoider,
        start_points=start_points,
        it_max=1000,
    )

    plot_vectorfield(
        ax=ax,
        dynamic_avoider=dynamic_avoider,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=100,
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    if save_figure:
        figure_name = "two_blobs_replica"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def plot_double(save_figure=False):
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        DoubleBlob(
            center_position=np.array([0.0, 3.0]),
            a_value=1,
            b_value=1.1,
            tail_effect=False,
        )
    )

    obstacle_environment.append(
        DoubleBlob(
            center_position=np.array([0.0, -3.0]),
            a_value=1,
            b_value=1.1,
            tail_effect=False,
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([0.0, 0]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    x_lim = [-5.05, 5.05]
    y_lim = [-6.2, 6.2]

    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    for obs in obstacle_environment:
        obs.plot_obstacle(
            ax=ax,
            fill_color="#ff9955",
            outline_color=None,
            plot_center_position=False,
        )

    dynamic_avoider = DynamicModulationAvoider(
        initial_dynamics=initial_dynamics, environment=obstacle_environment
    )

    start_points = [
        [-4, 2],
        [-4, 4],
        [4, 4],
        [4, -2],
        [4, -4],
        [-4, -4],
    ]

    plot_trajectories(
        ax=ax,
        dynamic_avoider=dynamic_avoider,
        start_points=start_points,
        it_max=1000,
    )

    plot_vectorfield(
        ax=ax,
        dynamic_avoider=dynamic_avoider,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=100,
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    if save_figure:
        figure_name = "double_blob"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def plot_blob_and_cirlce_old(save_figure=False):
    # This figure does not make sense, with the scale
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        DoubleBlob(
            center_position=np.array([-2.0, 2.1]),
            a_value=1,
            b_value=1.1,
            tail_effect=False,
        )
    )

    obstacle_environment.append(
        Sphere(
            center_position=np.array([-2.0, 0.8]),
            radius=0.7,
            tail_effect=False,
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([0.0, 0]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    x_lim = [-4.05, 2.05]
    y_lim = [-1.05, 4.05]

    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for obs in obstacle_environment:
        obs.plot_obstacle(
            ax=ax,
            fill_color="#ff9955",
            outline_color=None,
            plot_center_position=False,
        )

    dynamic_avoider = DynamicModulationAvoider(
        initial_dynamics=initial_dynamics, environment=obstacle_environment
    )

    start_points = [
        [1, 3],
    ]

    plot_trajectories(
        ax=ax,
        dynamic_avoider=dynamic_avoider,
        start_points=start_points,
        it_max=1000,
    )

    plot_vectorfield(
        ax=ax,
        dynamic_avoider=dynamic_avoider,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=100,
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        figure_name = "blob_and_circle_replica"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def plot_blob_and_cirlce(save_figure=False):
    # This figure does not make sense, with the scale
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        DoubleBlob(
            center_position=np.array([-2.0, 2.1]),
            a_value=1,
            b_value=1.1,
            tail_effect=False,
        )
    )

    obstacle_environment.append(
        Sphere(
            center_position=np.array([-2.0, 0.8]),
            radius=0.7,
            tail_effect=False,
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([0.0, 0]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    x_lim = [-4.05, 2.05]
    y_lim = [-1.05, 4.05]

    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for obs in obstacle_environment:
        obs.plot_obstacle(
            ax=ax,
            fill_color="#ff9955",
            outline_color=None,
            plot_center_position=False,
        )

    dynamic_avoider = DynamicModulationAvoider(
        initial_dynamics=initial_dynamics, environment=obstacle_environment
    )

    start_points = [
        [1, 3],
    ]

    plot_trajectories(
        ax=ax,
        dynamic_avoider=dynamic_avoider,
        start_points=start_points,
        it_max=1000,
    )

    plot_vectorfield(
        ax=ax,
        dynamic_avoider=dynamic_avoider,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=100,
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        figure_name = "blob_and_circle"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def plot_circular_control_lyapunov(save_figure=False):
    # This figure does not make sense, with the scale
    obstacle_environment = ObstacleContainer()

    obstacle_environment.append(
        Sphere(
            center_position=np.array([0.0, 3.0]), radius=1.5, tail_effect=False
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([0.0, 0]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    x_lim = [-6, 6]
    y_lim = [-2, 10.0]

    # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for obs in obstacle_environment:
        obs.plot_obstacle(
            ax=ax,
            fill_color="#00ff00ff",
            outline_color=None,
            plot_center_position=False,
        )

    dynamic_avoider = DynamicModulationAvoider(
        initial_dynamics=initial_dynamics, environment=obstacle_environment
    )

    start_points = [
        [-2, 3],
        [-4, 4],
        [-4, 6],
        [-2, 7],
        [0.01, 8],
        [2, 7],
        [4, 6],
        [4, 4],
        [2, 3],
    ]

    plot_trajectories(
        ax=ax,
        dynamic_avoider=dynamic_avoider,
        start_points=start_points,
        linecolor="black",
        linewidht=2,
        # startshape='o', markersize=8, markercolor='blue',
        plot_attractor=False,
        plot_start=False,
        it_max=1000,
    )

    plot_vectorfield(
        ax=ax,
        dynamic_avoider=dynamic_avoider,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=100,
    )

    start_points = np.array(start_points).T
    plt.scatter(
        start_points[0, :],
        start_points[1, :],
        s=70,
        facecolors="none",
        edgecolors="b",
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if save_figure:
        figure_name = "qp_lyapunov"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    plot_single_blob(save_figure=True)

    plot_single_blob_left(save_figure=True)

    plot_double(save_figure=True)

    plot_blob_and_cirlce(save_figure=True)

    plot_circular_control_lyapunov(save_figure=True)
