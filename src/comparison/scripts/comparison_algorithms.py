""" Visualization of convergence-summing / learned trajectory. """
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.containers import GradientContainer
from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.avoidance import (
    obs_avoidance_interpolation_moving,
)

from dynamic_obstacle_avoidance.visualization import Simulation_vectorFields

from vartools.dynamical_systems import LinearSystem
from vartools.dynamical_systems import DynamicalSystem
from vartools.dynamical_systems import plot_dynamical_system_quiver


class F1(DynamicalSystem):
    def evaluate(self, position):
        return 0.1 * LA.norm(position) * np.array([1, 1])
        # return 0.1*LA.norm(position)*np.array([1, 1])


class F2(DynamicalSystem):
    def evaluate(self, position):
        return 0.1 * LA.norm(position) * np.array([1, 1])
        # return 0.1*LA.norm(position)*np.array([1, 1])


def visualize_f1():
    f1_dynamics = F1(dimension=2)
    plot_dynamical_system_quiver(
        dynamical_system=f1_dynamics,
        x_lim=[-6.0, 6.0],
        y_lim=[-5.0, 5.0],
        axes_equal=True,
    )
    plt.title(r"Dynamics of $f_1(x) = 0.1 \| x \| [1, 1]$")


def visualize_f2():
    f2_dynamics = F2(dimension=2)
    plot_dynamical_system_quiver(
        dynamical_system=f2_dynamics,
        x_lim=[-6.0, 6.0],
        y_lim=[-5.0, 5.0],
        axes_equal=True,
    )
    plt.title(r"Dynamics of $f_2(x) = 0.1 (\| x \|  - x^T x)[1, 1]$")


def plot_comparision_lyapunov_comparison_streamplot(
    obs_list, save_figure=False, n_grid=10
):
    x_lim = [-6, 6]
    y_lim = [-2, 10]

    dim = 2

    dynamical_system = LinearSystem(
        # attractor_position=np.array([0, 0]),
        A_matrix=np.eye(dim)
    )

    # breakpoint()
    show_streamplot = False
    if show_streamplot:
        fig, ax = Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs_list,
            pos_attractor=dynamical_system.attractor_position,
            dynamical_system=dynamical_system.evaluate,
            point_grid=n_grid,
            figName="replciation_lyapunov_function",
            show_streamplot=True,
            noTicks=False,
            saveFigure=save_figure,
        )

        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")


def plot_comparision_lyapunov_comparison(
    start_position, obs_list, save_figure=False, n_grid=10
):
    fig, ax = plt.subplots(figsize=(7.5, 6))

    x_lim = [-6, 6]
    y_lim = [-2, 10]

    dim = 2

    dynamical_system = LinearSystem(
        attractor_position=np.array([0, 0]), A_matrix=-np.eye(dim)
    )

    delta_time = 0.01
    it_max = 1000
    conv_margin = 1e-2

    positions = np.zeros((dim, it_max))
    velocities = np.zeros((dim, it_max))

    obs_polygon = []

    # plot_obstacles(ax, obs_list, x_lim, y_lim)
    for obs in obs_list:
        if obs.boundary_points is None:
            obs.draw_obstacle()

        x_obs = obs.boundary_points_global_closed
        obs_polygon.append(plt.Polygon(x_obs.T, alpha=1.0, zorder=-3))
        obs_polygon[-1].set_color("#00ff00ff")

        ax.add_patch(obs_polygon[-1])

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    for pos in start_position:
        positions[:, 0] = pos
        for ii in range(it_max - 1):
            inital_ds = dynamical_system.evaluate(positions[:, ii])
            velocities[:, ii] = obs_avoidance_interpolation_moving(
                positions[:, ii], inital_ds, obs_list
            )

            if LA.norm(velocities[:, ii]) < conv_margin:
                break

            positions[:, ii + 1] = (
                positions[:, ii] + delta_time * velocities[:, ii]
            )

        ax.plot(
            positions[0, :ii],
            positions[1, :ii],
            "-",
            color="k",
            # color='#A9A9A9',
        )

    # start_pos
    start_position = np.array(start_position).T
    ax.scatter(
        start_position[0, :],
        start_position[1, :],
        s=80,
        facecolors="none",
        edgecolors="b",
    )

    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    ax.grid()
    plt.show()

    save_figure = True
    if save_figure:
        fig_name = "control_lyapunov_replication_1"
        plt.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


def plot_lyapunov_conversion_minima(start_position_list, obs_list):
    pass


if (__name__) == "__main__":
    start_position_list = [
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

    obs_list = GradientContainer()  # create empty obstacle list
    obs_list.append(
        Ellipse(
            axes_length=[1.5, 1.5],
            center_position=[0.0, 3.0],
            orientation=0,
        )
    )

    plt.close("all")
    plt.ion()
    plot_comparision_lyapunov_comparison(
        start_position_list, obs_list=obs_list, save_figure=False
    )
    # plot_lyapunov_conversion_minima(start_position_list, obs_list)

    visualize_f1()
    visualize_f2()
