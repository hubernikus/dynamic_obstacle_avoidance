#!/USSR/bin/python3
""" Example Obstacle Avoidance """
# Author: LukasHuber
# Email: lukas.huber@epfl.ch
# Created:  2021-09-23
import time
import os
import datetime
from math import pi

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation

from dynamic_obstacle_avoidance.obstacles import Polygon

# from dynamic_obstacle_avoidance.obstacles import Cuboid, Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator


class DynamicalSystemAnimation(Animator):
    dim = 2

    def setup(
        self,
        initial_dynamics,
        obstacle_environment,
        start_position=np.array([0, 0]),
        x_lim=[-1.5, 2],
        y_lim=[-0.5, 2.5],
    ):
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.obstacle_environment = obstacle_environment
        self.initial_dynamics = initial_dynamics

        self.dynamic_avoider = ModulationAvoider(
            initial_dynamics=self.initial_dynamics,
            obstacle_environment=self.obstacle_environment,
        )

        self.position_list = np.zeros((self.dim, self.it_max + 1))
        self.position_list[:, 0] = start_position

        self.fig, self.ax = plt.subplots(figsize=(10, 8))

    def update_step(self, ii):
        if not ii % 10:
            print(f"it={ii}")

        # Here come the main calculation part
        velocity = self.dynamic_avoider.evaluate(self.position_list[:, ii])

        self.position_list[:, ii + 1] = (
            velocity * self.dt_simulation + self.position_list[:, ii]
        )

        # Update obstacles
        self.obstacle_environment.do_velocity_step(delta_time=self.dt_simulation)

        self.ax.clear()

        # Drawing and adjusting of the axis
        self.ax.plot(
            self.position_list[0, : ii + 1],
            self.position_list[1, : ii + 1],
            ":",
            color="#135e08",
        )
        self.ax.plot(
            self.position_list[0, ii + 1],
            self.position_list[1, ii + 1],
            "o",
            color="#135e08",
            markersize=12,
        )
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

        plot_obstacles(
            ax=self.ax,
            obstacle_container=self.obstacle_environment,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
            showLabel=False,
        )

        self.ax.plot(
            self.initial_dynamics.attractor_position[0],
            self.initial_dynamics.attractor_position[1],
            "k*",
            markersize=8,
        )
        self.ax.grid()
        self.ax.set_aspect("equal", adjustable="box")

    def has_converged(self, ii) -> bool:
        return np.allclose(self.position_list[:, ii + 1], self.position_list[:, ii])


def simple_point_robot():
    """Simple robot avoidance."""
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Ellipse(
            axes_length=[0.6, 1.3],
            center_position=np.array([-0.2, 2.4]),
            margin_absolut=0,
            orientation=-30 * pi / 180,
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    obstacle_environment.append(
        Cuboid(
            axes_length=[0.4, 1.3],
            center_position=np.array([1.2, 0.25]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.5,
            orientation=10 * pi / 180,
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([2.0, 1.8]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    # obstacle_environment.append(
    # Cuboid(axes_length=[1.6, 0.7],
    #         center_position=np.array([0.0, 0.0]),
    #         margin_absolut=0,
    #         orientation=-0,
    #         tail_effect=False,
    #         repulsion_coeff=1.4,
    #         ))

    # obstacle_environment.append(
    # Cuboid(axes_length=[0.5, 0.5],
    #        center_position=np.array([0.0, 0.6]),
    #        # center_position=np.array([0.9, 0.25]),
    #        margin_absolut=0,
    #        orientation=0,
    #        tail_effect=False,
    #        repulsion_coeff=1.4,
    #        ))


def run_stationary_point_avoiding_dynamic_robot():
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Ellipse(
            axes_length=[0.5, 0.5],
            # center_position=np.array([-3.0, 0.2]),
            center_position=np.array([-1.0, 0.2]),
            margin_absolut=0.5,
            orientation=0,
            linear_velocity=np.array([0.5, 0.0]),
            tail_effect=False,
        )
    )

    obstacle_environment.append(
        Cuboid(
            axes_length=[0.4, 1.3],
            center_position=np.array([2.2, 0.25]),
            # center_position=np.array([0.9, 0.25]),
            margin_absolut=0.5,
            orientation=10 * pi / 180,
            tail_effect=False,
            # repulsion_coeff=1.4,
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([0.0, 0.0]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    my_animation = DynamicalSystemAnimation(
        dt_simulation=0.05,
        dt_sleep=0.01,
        file_type=".gif",
    )

    my_animation.setup(
        initial_dynamics,
        obstacle_environment,
        start_position=np.array([0.1, 2.1]),
        x_lim=[-3, 3],
        y_lim=[-2.1, 2.1],
    )

    my_animation.run(save_animation=True)


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # simple_point_robot()
    run_stationary_point_avoiding_dynamic_robot()
