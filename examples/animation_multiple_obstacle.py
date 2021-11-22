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

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance import DynamicModulationAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem

# Matplotlib extension for copy
def list_transferable_attributes(obj, except_attributes=None):
    if except_attributes is None:
        except_attributes = ("transform", "figure")

    obj_methods_list = dir(obj)

    obj_get_attr = []
    obj_set_attr = []
    obj_transf_attr = []

    for name in obj_methods_list:
        if len(name) > 4:
            prefix = name[0:4]
            if prefix == "get_":
                obj_get_attr.append(name[4:])
            elif prefix == "set_":
                obj_set_attr.append(name[4:])

    for attribute in obj_set_attr:
        if attribute in obj_get_attr and attribute not in except_attributes:
            obj_transf_attr.append(attribute)

    return obj_transf_attr


def copy_artist(original_artist, new_artist=None, attr_list=None):
    if attr_list is None:
        attr_list = list_transferable_attributes(original_artist)

    # Create artist of new_type
    if isinstance(original_artist, matplotlib.patches.Polygon):
        line2 = plt.Line2D([], [])
    elif isinstance(original_artist, matplotlib.patches.Polygon):
        new_artist = plt.Polygon([], [])
    else:
        raise Exception(f"Not ipmlemented for patch-type {type(original_artist)}")

    for i_attribute in attr_list:
        getattr(new_artist, "set_" + i_attribute)(
            getattr(original_artist, "get_" + i_attribute)()
        )


class DynamicalSystemAnimation:
    def __init__(self):
        self.animation_paused = False

        self.fig = None
        self.ax = None

    def on_click(self, event):
        if self.animation_paused:
            self.animation_paused = False
        else:
            self.animation_paused = True

    def run(
        self,
        initial_dynamics,
        obstacle_environment,
        start_position=np.array([0, 0]),
        x_lim=[-1.5, 2],
        y_lim=[-0.5, 2.5],
        it_max=1000,
        dt_step=0.03,
        dt_sleep=0.1,
        save_animation=False,
        figure_name=None,
    ):
        self.dynamic_avoider = DynamicModulationAvoider(
            initial_dynamics=initial_dynamics, environment=obstacle_environment
        )
        self.dt_step = dt_step
        self.x_lim = x_lim
        self.y_lim = y_lim

        dim = 2
        self.position_list = np.zeros((dim, it_max + 1))
        self.position_list[:, 0] = start_position

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        if save_animation:
            plt_obj_list = []

        if save_animation:
            if figure_name is None:
                now = datetime.datetime.now()
                figure_name = f"animation_{now:%Y-%m-%d_%H-%M-%S}"

                # Set filetype
                file_type = ".mp4"
                figure_name = figure_name + file_type

            ani = animation.FuncAnimation(
                self.fig,
                self.update_step,
                frames=it_max,
                interval=dt_sleep * 1000,  # Conversion [s] -> [ms]
            )

            ani.save(
                os.path.join("figures", figure_name), metadata={"artist": "Lukas Huber"}
            )

            plt.close("all")

        else:
            ii = 0
            while ii < it_max:
                ii += 1
                if ii > it_max:
                    break
                self.update_step(ii, animation_run=False)

                if self.animation_paused:
                    plt.pause(dt_sleep)
                    if not plt.fignum_exists(fig.number):
                        print("Stopped animation on closing of the figure..")
                        break
                    continue

                plt.pause(dt_sleep)
                if not plt.fignum_exists(self.fig.number):
                    print("Stopped animation on closing of the figure..")
                    break

    def update_step(self, ii, animation_run=True, print_modulo=10) -> list:
        """Returns list element."""
        if print_modulo:
            if not ii % print_modulo:
                print(f"it={ii}")

        # Here come the main calculation part
        velocity = self.dynamic_avoider.evaluate(self.position_list[:, ii - 1])
        self.position_list[:, ii] = (
            velocity * self.dt_step + self.position_list[:, ii - 1]
        )

        # Update obstacles
        self.dynamic_avoider.environment.move_obstacles_with_velocity(
            delta_time=self.dt_step
        )

        # Clear right before drawing again
        self.ax.clear()

        # Drawing and adjusting of the axis
        self.ax.plot(
            self.position_list[0, :ii], self.position_list[1, :ii], ":", color="#135e08"
        )

        self.ax.plot(
            self.position_list[0, ii],
            self.position_list[1, ii],
            "o",
            color="#135e08",
            markersize=12,
        )

        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

        plot_obstacles(
            self.ax,
            self.dynamic_avoider.environment,
            self.x_lim,
            self.y_lim,
            showLabel=False,
        )

        self.ax.plot(
            self.dynamic_avoider.initial_dynamics.attractor_position[0],
            self.dynamic_avoider.initial_dynamics.attractor_position[1],
            "k*",
            markersize=8,
        )
        self.ax.grid()
        self.ax.set_aspect("equal", adjustable="box")

        if animation_run:
            all_children = self.ax.get_children()

            for aa, artist in enumerate(all_children):
                # Only keep lines and patches
                if not (
                    isinstance(artist, matplotlib.patches.Patch)
                    or isinstance(artist, matplotlib.lines.Line2D)
                ):
                    del all_children[aa]
            return all_children
        else:
            return []


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

    initial_dynamics = LinearSystem(
        attractor_position=np.array([0.0, 0.0]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    DynamicalSystemAnimation().run(
        initial_dynamics,
        obstacle_environment,
        x_lim=[-3, 3],
        y_lim=[-2.1, 2.1],
        dt_step=0.05,
        it_max=100,
        save_animation=True,
    )


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # simple_point_robot()
    run_stationary_point_avoiding_dynamic_robot()
