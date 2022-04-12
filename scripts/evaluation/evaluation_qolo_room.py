#!/USSR/bin/python3
""" Script to evaluate the simulation in the robot room. """

__author__ = "LukasHuber"
__email__ = "lukas.huber@epfl.ch"
__date__ = "2020-08-13"

import time

import re

import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import (
    obs_avoidance_interpolation_moving,
)

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import *

# from PIL import Image
from scipy import ndimage
from scipy import misc

# qolo_img  = misc.imread("/home/crowdbot/qolo_ws/src/qolo_modulation/data/Qolo_T_CB_top_bumper.JPG")


# Environment Setup
from conference_room_setup import get_conference_room_setup

# Resolution
n_steps = int(2e1)

# Setup plot
plt.ion()
plt.close("all")


class DynamicSimulator:
    def __init__(
        self,
        input_file,
        attractor_position=None,
        x_range=[-10, 10],
        y_range=[-10, 10],
        figSize=(8, 7),
    ):

        self.fig = plt.figure()
        self.fig.set_size_inches(figSize)
        # self.ax = self.fig.add_subplot(1, )

        self.sleep_time = 2e-1

        self.dim = 2  # Two dimensional problem

        self.it_sim = 0

        self.fig_patches = []
        self.fig_contours = []
        self.fig_centers = []
        self.fig_ref_points = []
        self.fig_trajectory = None

        self.real_time = -1

        self.agent = Ellipse(
            center_position=np.zeros(2),
            orientation=0,
            axes_length=np.array([3, 2]),
            margin_absolut=0,
        )
        self.agent_trajectory = np.zeros((self.dim, 0))

        # self.obs_list = GradientContainer()

        self.obs = get_conference_room_setup(robot_margin=0.6)
        for ii in range(len(self.obs)):
            self.obs[ii].draw_obstacle()

        # Open file (! close at function destruction)
        self.simulation_file = open(input_file)

    def __del__(self):
        """Destructor of function -- close simulation file"""
        self.simulation_file.close()

    def setup_environment(self):
        """Setup environment"""
        line = re.split(",", self.simulation_file.readline()[:-1])
        if not line[0] == "time":
            raise ValueError(
                "Unkown file structure with starting value <{}>.".format(line[0])
            )
        # self.read_room_instance()

        self.plot_obstacle_environment()

    def main_loop(self):
        self.loop_count = 0

        while True:
            print("new loop")
            next_timestamp = self.read_room_instance()
            self.agent_trajectory = np.hstack(
                (self.agent_trajectory, self.agent.center_position.reshape(self.dim, 1))
            )

            self.plot_obstacle_environment()

            plt.pause(5e-1)

            if next_timestamp is None:
                return

            if not len(plt.get_fignums()):
                print()
                print("Animation ended through closing of figures.")
                break

            self.loop_count += 1

            print("Iteration {}".format(self.loop_count))

    def plot_obstacle_environment(self):
        # ''' '''
        self.fig_contours = []
        self.fig_centers = []
        self.fig_ref_points = []
        self.fig_trajectory = []

        for ii in range(len(self.obs)):
            if self.obs[ii].is_boundary:
                x_range, y_range = self.ax.get_xlim(), self.ax.get_ylim()
                outer_boundary = np.array(
                    [
                        [x_range[0], x_range[1], x_range[1], x_range[0]],
                        [y_range[0], y_range[0], y_range[1], y_range[1]],
                    ]
                ).T

                boundary_polygon = plt.Polygon(outer_boundary, alpha=0.5, zorder=-2)
                boundary_polygon.set_color(np.array([176, 124, 124]) / 255.0)
                plt.gca().add_patch(boundary_polygon)  # No track of this one

                obs_polygon = plt.Polygon(
                    self.obs[ii].boundary_points_global[:, :2], alpha=1.0, zorder=-1
                )
                # Add white center polygon
                obs_polygon.set_color(np.array([1.0, 1.0, 1.0]))

            else:
                obs_polygon = plt.Polygon(
                    self.obs[ii].boundary_points_global[:, :2],
                    animated=True,
                )

                obs_polygon.set_color(np.array([176, 124, 124]) / 255.0)
                obs_polygon.set_alpha(0.8)

            patch = plt.gca().add_patch(obs_polygon)
            self.fig_patches.append(patch)

            (boundary_line,) = plt.plot(
                self.obs[ii].boundary_points_margin_global[:, 0],
                self.obs[ii].boundary_points_margin_global[:, 1],
                "--",
                lineWidth=4,
                animated=True,
            )
            self.fig_contours.append(boundary_line)

            # Center of obstacle
            (center_position,) = self.ax.plot([], [], "k.", animated=True)
            self.fig_centers.append(center_position)

            (reference_point,) = self.ax.plot(
                [],
                [],
                "k+",
                animated=True,
                linewidth=18,
                markeredgewidth=4,
                markersize=13,
            )
            self.fig_ref_points.append(reference_point)

        plt.show()
        # plt.plot(self.attractorPos[0], self.attractorPos[1], 'k*', linewidth=7.0, markeredgewidth=4, markersize=13)

    def read_room_instance(self):
        """Get obstacle string and sample the boundary."""
        print("do another")
        line = re.split(",", self.simulation_file.readline()[:-1])
        self.obs = GradientContainer()

        while True:
            print("loop 0")
            if not line:
                return None  # No next time
            elif line[0] == "agent":
                while True:
                    print("loop 1")
                    line = re.split(",", self.simulation_file.readline()[:-1])
                    if line[0] == "position":
                        self.agent.center_position = np.array(line[1:], dtype=float)
                        continue
                    elif line[0] == "orientation":
                        self.agent.orientation = float(line[1])
                        continue
                    break

            elif line[0] == "skpping this temporarily":
                # elif line[0]=="object":
                obs_type = line[1]

                kwargs = {}
                while True:

                    print("loop 2")
                    reference_point = None
                    line = re.split(",", self.simulation_file.readline()[:-1])
                    if (
                        line[0] == "center_position"
                        or line[0] == "linear_velocity"
                        or line[0] == "axes_length"
                    ):
                        kwargs[line[0]] = np.array(line[1:], dtype=float)
                        continue

                    elif line[0] == "reference_point":
                        reference_point = np.array(line[1:], dtype=float)
                        continue

                    elif line[0] == "orientation" or line[0] == "angular_velocity":
                        kwargs[line[0]] = float(line[1])
                        continue
                    elif line[0] == "is_boundary" or line[0] == "is_static":
                        if line[1] == "True":
                            kwargs[line[0]] = True
                        else:
                            kwargs[line[0]] = False
                        continue
                    break

                if obs_type == "Polygon":
                    self.obs.append(Polygon(**kwargs))
                elif obs_type == "Cuboid":
                    self.obs.append(Cuboid(**kwargs))
                elif obs_type == "Ellipse":
                    self.obs.append(Ellipse(**kwargs))
                else:
                    print("Uknown obstacle of type <{}>.".format(obs_type))

                print(self.obs)
                print(type(self.obs[-1]))
                print("boundary", (self.obs[-1].is_boundary))
                print("args", kwargs)

                if not reference_point is None:
                    self.obs[-1].set_reference_point(
                        reference_point, in_global_frame=True
                    )

            elif line[0] == "time":
                next_time = float(line[1])
                return next_time
            else:
                # print('line', line)
                line = re.split(",", self.simulation_file.readline()[:-1])
                continue  # skipping very thing else
                # warnings.warn("Unexpected input <{}>".format(line[0]))
                # import pdb; pdb.set_trace()     ##### DEBUG #####

        # Draw obstacles
        for ii in range(len(self.obs)):
            self.obs.draw_obstacle()

        self.agent.draw_obstacle()

    def draw_obstacles(self):
        pass

    def update(self, dt):
        pass


if (__name__) == "__main__":
    path_file = "/home/lukas/Code/ObstacleAvoidance/dynamic_obstacle_avoidance/data/qolocording/"
    input_file = "qolocording_20200813_204900.txt"

    Simulator = DynamicSimulator(input_file=path_file + input_file)
    Simulator.setup_environment()
    Simulator.main_loop()

    print("Finished simulating file.")
