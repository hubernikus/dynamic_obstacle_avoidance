# !/usr/bin/env python2
"""
QOLO collision free navigation using modulation-algorithm and python.
"""

__author__ = "Lukas Huber"
__date__ = "2020-08-19"
__email__ = "lukas.huber@epfl.ch"

import sys
import os
import yaml
import copy
import time
from datetime import datetime

# from PIL import Image
import glob

import numpy as np
from math import pi
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from scipy import ndimage

# from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.animation as animation

# Writer = animation.writers['ffmpeg']
# witer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

# import rospy
import rosbag

from threading import Lock

lock = Lock()

if sys.version_info < (3, 0):
    from itertools import izip as zip

    # from __future__ import izip

import sensor_msgs.point_cloud2 as pc2

# import rospkg
# rospack = rospkg.RosPack()
# rospack.list()

# Add obstacle avoidance without 'setting' up
# directory_path = rospack.get_path('qolo_modulation')
path_avoidance = os.path.join("/home/lukas/catkin_ws/src/qolo_modulation", "scripts")

if not path_avoidance in sys.path:
    sys.path.append(path_avoidance)
    # pass

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import (
    get_linear_ds,
    make_velocity_constant,
    linear_ds_max_vel,
    linearAttractor_const,
)

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import (
    get_linear_ds,
    make_velocity_constant,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import (
    GradientContainer,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.human_ellipse import (
    TrackedPedestrian,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import Obstacle
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import Ellipse
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import Polygon
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import (
    obs_avoidance_interpolation_moving,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import (
    angle_difference_directional,
    transform_polar2cartesian,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.crowd_learning_container import (
    CrowdLearningContainer,
    CrowdCircleContainer,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import (
    CircularObstacle,
)

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
    plot_obstacles,
    plt_speed_line_and_qolo,
)

# plt.close('all')

### ROBOT PARAMETERS
ROBOT_MARGIN = 0.4  # m
HUMAN_RADIUS = 0.35

RADIUS_HUMAN_PLUS_MARGIN = 0.6

num_crowd_close = 10

SAFE_ANIMATION = False
SAFE_FIG = False

DES_SPEED = 0.5
MAX_SPEED = 0.8


class QoloAnimator:
    def __init__(
        self,
        x_lim=None,
        y_lim=None,
        attractor_position=np.array([5.5, 5.5]),
        obstacle_list=None,
    ):
        # self.save_dir =  save_dir
        # os.chdir(bag_dir)
        # self.bag_dir = bag_dir
        # self.bag = rosbag.Bag(os.path.join(bag_dir, input_bag))

        self.x_lim, self.y_lim = x_lim, y_lim
        self.attractor_position = attractor_position

        # TODO: change to arguments instead of hardcoding directory
        # inputFileName = sys.argv[1]
        # print "[OK] Found bag: %s" % inputFileName

        # Initiate first time to obstacle list
        # self.obstacle_list = GradientContainer(obs_list=obstacle_list)
        if obstacle_list is None:
            self.obstacle_list = CrowdCircleContainer(robot_margin=ROBOT_MARGIN)
        else:
            self.obstacle_list = CrowdCircleContainer(
                robot_margin=ROBOT_MARGIN, obs_list=None
            )
            for obs in obstacle_list:
                # TODO: include loop in function
                self.obstacle_list.append(obs)

        self.obstacle_list.last_update_time = time.time()

        # self.obstacle_list.update_reference_points()

    def run_circular_world(
        self, rotate_image=True, attractor_position=np.array([5, 0.1])
    ):
        # attractor_position = np.array([0, 18.0])
        # attractor_position = np.array([0.0, -20.0])
        # attractor_position =

        # Setup plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        self.is_pause = False
        fig.canvas.mpl_connect(
            "button_press_event", self.on_click
        )  # Button Click Enabled

        # LocalCrowd = CrowdCircleContainer(robot_margin=ROBOT_MARGIN)

        # Add qolo-image to visualization
        self.agent = Obstacle(
            name="QOLO",
            # center_position=np.array([3, 0.5]),
            orientation=0,
        )  # Define obstaclen

        self.agent.margin = ROBOT_MARGIN
        self.agent.position = np.array([1.0, 1.0])

        if True:
            # try:
            working_directory = os.getcwd()
            img_qolo_name = "data/Qolo_T_CB_top_bumper.png"

            if working_directory[-8:] == "/scripts":
                image_path = os.path.join(working_directory, "..", img_qolo_name)
            else:
                image_path = os.path.join(working_directory, img_qolo_name)

            # arr_img = mpimg.imread(os.path.join(bag_dir, 'qolo_t_cb_top_bumper.png'))
            arr_img = mpimg.imread(image_path)
            length_x = 1.4
            length_y = (1.0) * arr_img.shape[0] / arr_img.shape[1] * length_x
            plt_qolo = True
        else:
            # except:
            print("No center qolo")
            plt_qolo = False

        if SAFE_ANIMATION:
            date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_ani_figs = "fig_temp_" + date_time + "_"

            img_list = []

        self.modulated_vel_real = np.array([0, 0])
        t_start = 0

        # t_start = 169710.300005
        # t_start = 169712.400005
        # t_start = 169712.300005

        # t_start = 244.00

        # t_start = 72851.0
        # t_start = 72850.0

        time_list = []
        it_count = 0

        self.dt_pause = 5e-2
        self.delta_t = 5e-2

        self.front_laser_msg = None
        self.rear_laser_msg = None

        t = 0
        while True:
            while self.is_pause:
                plt.pause(self.dt_pause)

            time_list.append(t)

            # print('time = {}s'.format(t))
            # if t.to_sec() < t_start:
            # continue

            # print('time', t.to_sec())
            # crowd_message = msg.crowd
            crowd_message = None

            # LocalCrowd.update_step(crowd_message, agent_position=self.agent.position, num_crowd_close=num_crowd_close)
            # self.obstacle_list.update_step(
            # crowd_message, agent_position=self.agent.position,
            # num_crowd_close=num_crowd_close, automatic_outer_boundary=False,
            # lidar_input=None,
            # lidar_input=self.laser_scan_imitation,
            # )

            # import pdb; pdb.set_trace()
            ax.cla()  # Clear axes

            show_vector_field = False
            # show_vector_field = True
            if show_vector_field:

                # Remove existing crowd obstacles
                # it = 0
                # while(it<len(LocalCrowd)):
                #     if LocalCrowd[it].is_boundary:
                #         it+=1
                #     else:
                #         del LocalCrowd[it]

                line = plt_speed_line_and_qolo(
                    # points_init=self.agent.position,
                    points_init=np.array([-5.0, 1.5]),
                    attractorPos=self.attractor_position,
                    obs=self.obstacle_list,
                    fig_and_ax_handle=(fig, ax),
                    dt=0.02,
                )

                Simulation_vectorFields(
                    self.x_lim,
                    self.y_lim,
                    obs=self.obstacle_list,
                    xAttractor=self.attractor_position,
                    saveFigure=False,
                    figName="linearSystem_boundaryCuboid",
                    noTicks=False,
                    draw_vectorField=True,
                    show_streamplot=False,
                    point_grid=30,
                    normalize_vectors=False,
                    reference_point_number=True,
                    drawVelArrow=True,
                    automatic_reference_point=True,
                    fig_and_ax_handle=(fig, ax),
                    gamma_distance=RADIUS_HUMAN_PLUS_MARGIN,
                )

                import pdb

                pdb.set_trace()
                continue

            # Add qolo-image to visualization
            if plt_qolo:
                if rotate_image:
                    rot = self.agent.orientation
                    print("rot", rot)
                    arr_img_rotated = ndimage.rotate(
                        arr_img, rot * 180.0 / pi, cval=255
                    )

                    lenght_x_rotated = (
                        np.abs(np.cos(rot)) * length_x + np.abs(np.sin(rot)) * length_y
                    )

                    lenght_y_rotated = (
                        np.abs(np.sin(rot)) * length_x + np.abs(np.cos(rot)) * length_y
                    )
                    ax.imshow(
                        arr_img_rotated,
                        extent=[
                            self.agent.position[0] - lenght_x_rotated / 2.0,
                            self.agent.position[0] + lenght_x_rotated / 2.0,
                            self.agent.position[1] - lenght_y_rotated / 2.0,
                            self.agent.position[1] + lenght_y_rotated / 2.0,
                        ],
                    )

                else:
                    ax.imshow(
                        arr_img,
                        extent=[
                            self.agent.position[0] - length_x / 2.0,
                            self.agent.position[0] + length_x / 2.0,
                            self.agent.position[1] - length_y / 2.0,
                            self.agent.position[1] + length_y / 2.0,
                        ],
                    )
            else:
                ax.plot(
                    self.agent.position[0],
                    self.agent.position[1],
                    "ko",
                    markeredgewidth=4,
                    markersize=13,
                )

            Simulation_vectorFields(
                self.x_lim,
                self.y_lim,
                obs=self.obstacle_list,
                xAttractor=attractor_position,
                saveFigure=False,
                figName="linearSystem_boundaryCuboid",
                noTicks=False,
                draw_vectorField=False,
                reference_point_number=False,
                drawVelArrow=True,
                # automatic_reference_point=True, point_grid=None,
                # automatic_reference_point=True,
                point_grid=10,
                showLabel=False,
                fig_and_ax_handle=(fig, ax),
                gamma_distance=RADIUS_HUMAN_PLUS_MARGIN,
            )

            # ds_inititial = get_linear_ds(self.agent.position, attractor=attractor_position) # initial DS
            # ds_modulated = obs_avoidance_interpolation_moving(
            #    self.agent.position, ds_inititial, obs=LocalCrowd)

            # ds_modulated = make_velocity_constant(
            #    ds_modulated, self.agent.position,
            #    attractor_position, constant_velocity=DES_SPEED,
            # )
            ds_inititial = linear_ds_max_vel(
                self.agent.position, attractor=attractor_position, vel_max=MAX_SPEED
            )

            ds_modulated = obs_avoidance_interpolation_moving(
                self.agent.position,
                ds_inititial,
                self.obstacle_list,
                tangent_eigenvalue_isometric=False,
                repulsive_obstacle=False,
            )

            # Update veloctiy
            self.agent.linear_velocity = ds_modulated
            # self.agent.angular_velocity

            # Update position
            self.agent.position = (
                self.agent.position + self.delta_t * self.agent.linear_velocity
            )
            # import pdb; pdb.set_trace()
            self.agent.orientation = np.arctan2(
                self.agent.linear_velocity[1], self.agent.linear_velocity[0]
            )

            # Update time
            t = t + self.delta_t

            # Proposed (using this script & exsimtaed crowd) velocity
            plt.quiver(
                self.agent.position[0],
                self.agent.position[1],
                ds_modulated[0],
                ds_modulated[1],
                color="b",
                alpha=0.7,
                label="Post-processing DS",
            )

            # Actual velocity command
            # plt.quiver(self.agent.position[0], self.agent.position[1],
            # self.modulated_vel_real[0], self.modulated_vel_real[1], color='r', alpha=0.7, label="Real-time DS")

            # Actual velocity
            # plt.quiver(self.agent.position[0], self.agent.position[1],
            # self.agent.linear_velocity[0], self.agent.linear_velocity[1], color='g', alpha=0.7, label="Actual Command")

            try:
                # plt.scatter(self.rear_laser_cartesian[0, :], self.rear_laser_cartesian[1, :], color='r')
                # plt.scatter(self.front_laser_cartesian[0, :], self.front_laser_cartesian[1, :], color='g')

                plt.scatter(
                    self.laser_scan_imitation[0, :],
                    self.laser_scan_imitation[1, :],
                    color="k",
                )
            except:
                pass
                # pass
                # continue

            # ax.legend(loc="lower right")
            # print("Script: ", np.round(ds_modulated, 2))
            # print("Command: ", np.round(self.modulated_vel_real, 2))
            # print("Actual: ", np.round(self.agent.linear_velocity, 2))

            annotate_time = False
            if annotate_time:
                ax.annotate(
                    "{}s".format(np.round(t, 2)),
                    xy=[
                        x_lim[1] - (x_lim[1] - x_lim[0]) * 0.2,
                        y_lim[1] - (y_lim[1] - y_lim[0]) * 0.08,
                    ],
                    textcoords="data",
                    size=16,
                    weight="bold",
                )

                # ax.annotate('{} s'.format(np.round(t.to_sec(), 2)), xy=[x_lim[1]-(x_lim[1]-x_lim[0])*0.14, y_lim[1]-(y_lim[1]-y_lim[0])*0.08], textcoords='data', size=16, weight="bold")

            if SAFE_FIG:
                fig_name = "qolo_simulation_time_{}ms_agent_number_{}".format(
                    round(t.to_sec() * 1000), num_crowd_close
                )

                import pdb

                pdb.set_trace()  ##### DEBUG #####
                plt.savefig(
                    "dynamic_obstacle_avoidance/figures/" + fig_name + ".png",
                    bbox_inches="tight",
                )

            # plt.grid('on')

            # IF only on turn (uncomment)
            # break
            # import pdb; pdb.set_trace()

            if SAFE_ANIMATION:
                fig.savefig(name_ani_figs + str(it_count).zfill(3) + ".png")
                # ax_list = ax.get_children()
                # img_list.append([])
                # for ii in range(len(ax_list)):
                # try:
                # img_list[-1].append(copy.deepcopy(ax_list[ii]))
                # except:
                # import pdb; pdb.set_trace()     ##### DEBUG #####
                # raise

            plt.pause(self.dt_pause)

            if not len(plt.get_fignums()):
                print()
                print("Animation ended by closing of figures.")
                break
            it_count += 1

            max_it = 1e6
            # max_it = 10

            if it_count > max_it:  # TODO remove
                break

        if SAFE_ANIMATION:
            # TODO: make nicer? (maybe)
            # im_ani = animation.ArtistAnimation(fig, img_list, interval=200, repeat_delay=3000, blit=True)

            # im_ani.save('animation_moving.mp4', writer=writer)
            plt.close("all")

            images = []
            imgs = sorted(glob.glob(name_ani_figs + "*"))
            for ii in imgs:
                frame = Image.open(ii)
                images.append(frame.copy())

            # First image twice to have 'break' before loop
            images[0].save(
                "animation_moving" + date_time + ".mp4",
                save_all=True,
                append_images=images[0:],
                duration=(time_list[-1] - time_list[0]) * 100,
                loop=0,
            )

            for ii in range(len(imgs)):
                os.remove(imgs[ii])

    def update_velocity(self, msg_cmd):
        self.agent.linear_velocity = msg_cmd.linear.x * np.array(
            [np.cos(self.agent.orientation), np.sin(self.agent.orientation)]
        )
        self.agent.angular_velocity = msg_cmd.angular.z / 180 * pi * (-1)

    def get_cartesian_from_laserscan(self, msg, delta_angle=0):
        """Turn laserscan into cartesian points which can be displayed."""

        ranges = np.array(msg.ranges)
        n_points = ranges.shape[0]

        angles = np.linspace(msg.angle_min, msg.angle_max, n_points)
        angles = angles + self.agent.orientation + delta_angle

        laser_cartesian = np.vstack((np.cos(angles) * ranges, np.sin(angles) * ranges))

        laser_cartesian = (
            laser_cartesian + np.tile(self.agent.position, (n_points, 1)).T
        )

        return laser_cartesian

    def run_initial_crowd(self, human_radius=HUMAN_RADIUS, robot_margin=ROBOT_MARGIN):
        # x_lim = [-5.1, 5.1]
        # y_lim = [-4.6, 4.6]

        # Setup plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)

        LocalCrowd = CrowdCircleContainer()

        # Add qolo-image to visualization
        arr_img = mpimg.imread(os.path.join(bag_dir, "qolo_t_cb_top_bumper.png"))
        length_x = 1.4
        length_y = arr_img.shape[0] / arr_img.shape[1] * length_x

        for topic, msg, t in self.bag.read_messages("/crowdbot"):
            if not round(t.to_sec(), 1) == 190.1:
                print(round(t.to_sec(), 1))
                continue

            crowd_list = msg.crowd

            human_radius = LocalCrowd.human_radius

            for ii in range(len(crowd_list)):
                pos_crowd = [crowd_list[ii].position.x, crowd_list[ii].position.z]

                # Check within plotting window
                if (
                    pos_crowd[0] - human_radius > self.x_lim[0]
                    and pos_crowd[0] + human_radius < self.x_lim[1]
                    and pos_crowd[1] - human_radius > self.y_lim[0]
                    and pos_crowd[1] + human_radius < self.y_lim[1]
                ):

                    vel_crowd = [
                        crowd_list[ii].velocity.linear.x,
                        crowd_list[ii].velocity.linear.z,
                    ]

                    LocalCrowd.append(
                        CircularObstacle(
                            center_position=pos_crowd,
                            orientation=0,
                            linear_velocity=vel_crowd,
                            angular_velocity=0,
                            radius=human_radius,
                            margin_absolut=LocalCrowd.robot_margin,
                        )
                    )

                    LocalCrowd[-1].draw_obstacle()  # plot contour

            ax.imshow(
                arr_img,
                extent=[
                    -length_x / 2.0,
                    length_x / 2.0,
                    -length_y / 2.0,
                    length_y / 2.0,
                ],
            )

            plot_obstacles(ax, LocalCrowd, x_lim, y_lim, showLabel=False)

        if SAFE_FIG:
            fig_name = "qolo_simulation_time_{}ms_allagents".format(
                round(t.to_sec() * 1000)
            )

            plt.savefig(
                "dynamic_obstacle_avoidance/figures/" + fig_name + ".png",
                bbox_inches="tight",
            )

        pass

    def run_bag(self):
        """Loop through the rosbag"""

        topics = {
            "clock": "/clock",
            "front_lidar": "/front_lidar/scan",
            "rear_lidar": "/rear_lidar/scan",
            "clients": "/connected_clients",
            "client_count": "/client_count",
        }

        self.orient_front_lidar = 0
        self.center_front_lidar = [0, 0.25]
        self.orient_rear_lidar = pi  # pointing backwards
        self.center_back_lidar = [0, -0.25]

        # Test
        for topic, msg, t in self.bag.read_messages("/crowdbot"):
            crowd_message = msg.crowd

        eval_msg_tag = 30

        # self.bag.get_message_count(topics['front_lidar'])
        it = 0
        for topic, msg, t in self.bag.read_messages(topics["front_lidar"]):
            import pdb

            pdb.set_trace()
            while it < eval_msg_tag:
                it += 1

            if len(msg.ranges) == 0:
                continue

            magnitudes = np.array(msg.ranges)
            angles = np.linspace(
                msg.angle_min + self.orient_front_lidar,
                msg.angle_max + self.orient_front_lidar,
                magnitudes.shape[0],
            )

            print("front stamp", msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
            break

        points_front = transform_polar2cartesian(
            magnitude=magnitudes, angle=angles, center_point=self.center_front_lidar
        )

        it = 0
        for topic, msg, t in self.bag.read_messages(topics["rear_lidar"]):
            while it < eval_msg_tag:
                it += 1
            if len(msg.ranges) == 0:
                continue

            # print('msg', msg)
            magnitudes = np.array(msg.ranges)
            angles = np.linspace(
                msg.angle_min + self.orient_rear_lidar,
                msg.angle_max + self.orient_rear_lidar,
                magnitudes.shape[0],
            )

            print("rear stamp", msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
            break

        points_rear = transform_polar2cartesian(
            magnitude=magnitudes, angle=angles, center_point=self.center_front_lidar
        )

        points = np.hstack((points_front, points_rear))

        plt.figure(1)
        plt.plot(points_front[0, :], points_front[1, :], "b.")
        plt.plot(points_rear[0, :], points_rear[1, :], "r.")

        plt.plot(0, 0, "ko", linewidth=18, markeredgewidth=4, markersize=13)
        plt.axis("equal")

        plt.xlim([-5.5, 5.5])
        plt.ylim([-5.5, 5.5])
        plt.ion()

        self.obs_list = CrowdLearningContainer()
        self.obs_list.update_step(lidar_data=points)

        reduced_points = transform_polar2cartesian(
            self.obs_list[-1].surface_magnitudes,
            self.obs_list[-1].surface_angles,
            center_position=self.obs_list[-1].center_position,
        )
        plt.plot(reduced_points[0, :], reduced_points[1, :], "g.")
        plt.plot(
            self.obs_list[-1].center_position[0],
            self.obs_list[-1].center_position[1],
            "k+",
            linewidth=18,
            markeredgewidth=4,
            markersize=13,
        )

        plt.figure()
        plt.plot(
            self.obs_list[-1].surface_angles, self.obs_list[-1].surface_magnitudes, "."
        )

        n_points = 100
        angles = np.linspace(-pi, pi, n_points)
        magnitudes = self.obs_list[-1].predict(angles)
        plt.plot(angles, magnitudes, "k")

        regr_cartesian = transform_polar2cartesian(
            magnitudes, angles, center_position=self.obs_list[-1].center_position
        )

        plt.figure(1)
        plt.plot(regr_cartesian[0, :], regr_cartesian[1, :], "k")

    def on_click(self, event):
        # Pause when one clicks on image
        self.is_pause ^= True

        # if self.pause:
        # self.pause_start = time.time()
        # else:
        # dT = time.time()-self.pause_start

        # if dT < 0.3: # Break simulation at double click
        # print('Animation is exited.')
        # self.ani.event_source.stop()


if (__name__) == "__main__":

    # bag_dir = '/home/lukas/catkin_ws/src/qolo_modulation/data/'
    # bag_dir = '/home/crowdbot/qolo_ws/src/qolo_modulation/data/'
    # bag_dir = os.path.join(directory_path, "data")

    # bag_file = "simulator_rds_01_2020-08-19-18-54-45.bag"
    # bag_file = "simulator_rds_01_2020-08-19-18-51-29.bag"
    # bag_file = "qolo_bag_20200901.bag"
    # bag_file = "qolo_bag_20200901_1804.bag"
    # bag_file = "qolo_bag_20200901_1857.bag"
    # bag_file = "coming_from_behind.bag"
    # bag_file = "dense_crowd.bag"
    # bag_file = "dense_crowd2.bag"
    # bag_file = "dense_crowd_20201004_1647.bag"

    # bag_file = "dense_crowd_coparallelflow_0201013.bag"
    # bag_file = "dense_crowd_counterflow_uniformcrowd_0201013.bag"
    # bag_file = "dense_crowd_counterflow_20201013.bag"

    if True:
        # bag_file = "collision_check_obstacle_avoidance.bag"
        # bag_file = "collision_check_obstacle_avoidance_.bag"
        bag_file = "collision_obstacle_avoidance_debug.bag"
        attractor_position = np.array([5.0, 5.0])
        x_lim = [-8, 7.0]
        y_lim = [-6.0, 6.0]
        # x_lim=[-2,3]; y_lim=[0, 2]
        # x_lim=[-6,-2]; y_lim=[1, 5]
        # from setup_simulator_walls import get_setup
        # obstacle_list = get_setup(robot_margin=ROBOT_MARGIN)

        x_lim = [-1, 7.0]
        y_lim = [-1.0, 6.0]
        from conference_room_setup import get_conference_room_setup

        obstacle_list = get_conference_room_setup(
            robot_margin=ROBOT_MARGIN, get_surrounding_with_puppet=False
        )

        # obstacle_list = GradientContainer()

    Animator = QoloAnimator(x_lim=x_lim, y_lim=y_lim, obstacle_list=obstacle_list)
    # robot_margin=ROBOT_MARGIN)
    # topicList = Animator.readBagTopicList()
    # print('topicList', topicList)

    Animator.run_circular_world(
        attractor_position=attractor_position,
    )
    # Reader.run_initial_crowd()

print("\n... finished script. Tune in another time!")
