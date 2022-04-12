# !/USSR/bin/python3
""" Script to show lab environment on computer """

# Custom libraries
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import *
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import *
from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import *

import numpy as np
from numpy import pi

import matplotlib.pyplot as plt
import os

from PIL import Image
import glob

__author__ = "LukasHuber"
__date__ = "2020-04-03"
__email__ = "lukas.huber@epfl.ch"

plt.close("all")

target_dir = (
    "/home/lukas/Code/ObstacleAvoidance/dynamic_obstacle_avoidance/figures/sequences"
)


def get_reference_weight(distance, distance_min=0, distance_max=1e4, weight_pow=1):
    """Get a weight inverse proportinal to the distance"""
    weights_all = np.zeros(distance.shape)
    if any(distance == distance_min):
        ind0 = distance == 0
        weights_all[ind0] = 1 / np.sum(ind0)
        return weights_all

    ind_range = np.logical_and(distance > distance_min, distance < distance_max)
    if not any(ind_range):
        return weights_all

    dist_temp = distance[ind_range]
    weights = 1 / (dist_temp - distance_min) - 1 / (distance_max - distance_min)
    weights = weights**weight_pow

    # Normalize
    weights = weights / np.sum(weights)

    # Add amount of movement relative to distance
    weight_ref_displacement = 1 / (dist_temp + 1 - distance_min) - 1 / (
        distance_max + 1 - distance_min
    )

    weights_all[ind_range] = weights * weight_ref_displacement
    return weights_all


def derivative_gamma_sum(position, obs1, obs2, grad_pow=5, delta_dist=1e-6):
    # TODO: remove
    dim = obs1.dim

    derivatve = np.zeros(dim)
    for dd in range(dim):
        delta = np.zeros(dim)
        delta[dd] = delta_dist

        sum_high = obs1.get_gamma(position + delta, in_global_frame=True) ** grad_pow
        sum_high += obs2.get_gamma(position + delta, in_global_frame=True) ** grad_pow

        sum_low = obs1.get_gamma(position - delta, in_global_frame=True) ** grad_pow
        sum_low += obs2.get_gamma(position - delta, in_global_frame=True) ** grad_pow

        derivatve[dd] = (sum_high - sum_low) / (2 * delta_dist)

    return -derivatve


class ReferencePointSequence:
    def __init__(self, obs_list, x_lim, y_lim, dir_name="name"):
        self.obs_list = obs_list
        self.x_lim = x_lim
        self.y_lim = x_lim

        self.path_directory = os.path.join(target_dir, dir_name)

        if not os.path.exists(self.path_directory):
            print("Creating new directory")
            os.makedirs(self.path_directory)

        print("Initialization finished.")

    def run(self, n_figs=2, time_total=1):
        if n_figs > 10:
            plt.ioff()
        else:
            plt.ion()  #
        dt = time_total / (n_figs - 1)

        for im_it in range(n_figs):
            self.obs_list.update_reference_points()
            self.obs_list.reset_obstacles_have_moved()

            plt.figure()

            for ii in range(len(obs)):
                for jj in range(len(obs)):
                    if ii == jj:
                        continue
                    point_bref = obs.get_boundary_reference_point(ii, jj)
                    plt.plot(
                        point_bref[0],
                        point_bref[1],
                        "g",
                        marker="+",
                        markeredgewidth=4,
                        markersize=12,
                    )

            n = 0
            for obstacle in self.obs_list:
                obstacle.draw_obstacle(numPoints=50)
                plt.plot(
                    obstacle.x_obs_sf[0, :],
                    obstacle.x_obs_sf[1, :],
                    color="k",
                    linewidth=2,
                )

                plt.plot(
                    obstacle.center_position[0],
                    obstacle.center_position[1],
                    "k",
                    marker="o",
                    markeredgewidth=4,
                    markersize=6,
                )

                plt.plot(
                    obstacle.global_reference_point[0],
                    obstacle.global_reference_point[1],
                    "r",
                    marker="+",
                    markeredgewidth=3,
                    markersize=10,
                )

                plt.annotate(
                    "{}".format(n),
                    xy=obstacle.global_reference_point + 0.08,
                    textcoords="data",
                    size=16,
                    weight="bold",
                )

                n += 1

                plt.xlim([-1, 4])
                plt.ylim([-2, 3])

            # plt.tick_params(axis='both', which='major',
            # bottom=False, top=False, left=False, right=False,
            # labelbottom=False, labelleft=False)

            plt.axis("equal")
            plt.xlim(self.x_lim)
            plt.ylim(self.y_lim)

            plt.savefig(
                os.path.join(
                    self.path_directory, "shot{}.png".format(str(im_it).zfill(2))
                ),
                bbox_inches="tight",
            )

            print("Saving figure {}/{}".format(im_it + 1, n_figs))

            # Move & update obstacle
            for obstacle in self.obs_list:
                updated_pos = obstacle.update_position(dt=dt, t=ii * dt)

    def create_gif(self, file_name=None, show_gif=True, duration=800):
        """Create GIF from PNG images"""
        plt.close("all")
        images = []

        imgs = sorted(glob.glob(os.path.join(self.path_directory, "*.png")))
        for ii in imgs:
            frame = Image.open(ii)
            images.append(frame.copy())

        # First image twice to have 'break' before loop
        images[0].save(
            os.path.join(self.path_directory, "animated.gif"),
            save_all=True,
            append_images=images[0:],
            duration=duration,
            loop=0,
        )

        # if show_gif:
        # img = Image.open(os.path.join(self.path_directory, "animated.gif"))
        # img.show()


# RUN main
if (__name__) == "__main__":
    setup = 4

    if setup == 0:
        obs = GradientContainer()

        obs.append(
            Ellipse(
                center_position=[0.0, 0.0],
                axes_length=[0.8, 1.2],
                # margin_absolut=1.0,
                margin_absolut=0.5,
                orientation=30 * pi / 180,
                linear_velocity=[0.0, 0],
            )
        )

        obs.append(
            Ellipse(
                center_position=[3.5, 0.0],
                axes_length=[0.8, 1.2],
                # margin_absolut=1.0,
                margin_absolut=0.5,
                orientation=-30 * pi / 180,
                linear_velocity=[-0.3, 0],
            )
        )

        x_lim, y_lim = [-4, 8], [-6.0, 6]

        SequenceGenerator = ReferencePointSequence(
            obs_list=obs, x_lim=x_lim, y_lim=y_lim, dir_name="ellipse_horizontal"
        )

        SequenceGenerator.run(n_figs=30, time_total=30)
        SequenceGenerator.create_gif(duration=200)

    elif setup == 1:
        obs = GradientContainer()

        obs.append(
            Ellipse(
                center_position=[0.0, 0.0],
                axes_length=[0.8, 1.2],
                # margin_absolut=1.0,
                margin_absolut=0.5,
                orientation=30 * pi / 180,
                linear_velocity=[0.0, 0],
            )
        )

        obs.append(
            Ellipse(
                center_position=[3.5, 0.0],
                axes_length=[0.8, 2.2],
                # margin_absolut=1.0,
                margin_absolut=0.5,
                orientation=-30 * pi / 180,
                linear_velocity=[0.0, 0],
                angular_velocity=-30 / 180.0 * pi,
            )
        )

        x_lim = [-4, 8]
        y_lim = [-6.0, 6]

        SequenceGenerator = ReferencePointSequence(
            obs_list=obs, x_lim=x_lim, y_lim=y_lim, dir_name="ellipse_rotation"
        )

        SequenceGenerator.run(n_figs=50, time_total=10)
        SequenceGenerator.create_gif(duration=200)

    elif setup == 3:
        obs = GradientContainer()

        obs.append(
            Ellipse(
                center_position=[0.0, 0.0],
                axes_length=[1.3, 2.7],
                # margin_absolut=1.0,
                margin_absolut=0.0,
                orientation=-30 * pi / 180,
                linear_velocity=[0.0, 0],
            )
        )

        obs.append(
            Ellipse(
                center_position=[3.5, -1.0],
                axes_length=[1.3, 2.5],
                # margin_absolut=1.0,
                margin_absolut=0.0,
                # orientation=-30*pi/180,
                orientation=pi / 180 * (-30),
                linear_velocity=[0.0, 0],
                angular_velocity=0,
            )
        )

        obs.append(
            Ellipse(
                center_position=[2.0, 2.0],
                axes_length=[1.3, 2.5],
                # margin_absolut=1.0,
                margin_absolut=0.0,
                # orientation=-30*pi/180,
                orientation=pi / 180 * 30,
                linear_velocity=[0.0, 0],
                angular_velocity=1,
            )
        )

        x_lim = [-4, 8]
        y_lim = [-6.0, 6]

        SequenceGenerator = ReferencePointSequence(
            obs_list=obs, x_lim=x_lim, y_lim=y_lim, dir_name="ellipse_test"
        )

        SequenceGenerator.run(n_figs=100, time_total=10)
        SequenceGenerator.create_gif(duration=200)

    # plt.close()
    elif setup == 4:
        obs = GradientContainer()

        obs.append(
            Cuboid(
                center_position=[0.0, 0.0],
                axes_length=[8.0, 6.0],
                # margin_absolut=1.0,
                margin_absolut=0.0,
                # orientation=-30*pi/180,
                orientation=pi / 180 * (0),
                linear_velocity=[0.0, 0],
                angular_velocity=0,
                is_boundary=True,
            )
        )

        obs.append(
            Ellipse(
                center_position=[2.0, 2.0],
                axes_length=[1.3, 2.5],
                # margin_absolut=1.0,
                margin_absolut=0.0,
                # orientation=-30*pi/180,
                orientation=pi / 180 * 30,
                linear_velocity=[0.0, 0],
                angular_velocity=1,
            )
        )

        x_lim = [-5, 5]
        y_lim = [-4.0, 4]

        SequenceGenerator = ReferencePointSequence(
            obs_list=obs, x_lim=x_lim, y_lim=y_lim, dir_name="inversed_obstacle"
        )

        SequenceGenerator.run(n_figs=2, time_total=10)
        SequenceGenerator.create_gif(duration=200)
