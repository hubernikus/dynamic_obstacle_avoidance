#!/USSR/bin/python3
""" Script to show lab environment on computer """

import warnings
import copy
import sys
import os

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from matplotlib import ticker

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
)  #
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    plt_speed_line_and_qolo,
)

from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import Ellipse
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import Cuboid
from dynamic_obstacle_avoidance.obstacle_description.boundary_cuboid_with_gap import (
    BoundaryCuboidWithGaps,
)

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import *
from dynamic_obstacle_avoidance.obstacle_avoidance.comparison_algorithms import (
    obs_avoidance_potential_field,
    obs_avoidance_orthogonal_moving,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import (
    obs_avoidance_interpolation_moving,
    obs_check_collision_2d,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import (
    GradientContainer,
)
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import (
    linear_ds_max_vel,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.metric_evaluation import (
    MetricEvaluator,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import (
    angle_is_in_between,
    angle_difference_directional,
)

# from dynamic_obstacle_avoidance.lru_chached_property import lru_cached_property
# from dynamic_obstacle_avoidance.lru_chached_property import cached_property

rel_path = os.path.join(".", "scripts")
if not rel_path in sys.path:
    sys.path.append(rel_path)

from comparison_algorithms import ObstacleAvoidanceAgent, position_is_in_free_space

__author__ = "LukasHuber"
__date__ = "2020-01-15"
__email__ = "lukas.huber@epfl.ch"


def put_patch_behind_gap(ax, obs_list, x_min=-10, y_margin=0):
    """Add a patch for a door fully on the left end of a graph."""
    # Take the wall obstacle
    obs = obs_list[-1]

    gap_points = obs.get_global_gap_points()

    edge_points = np.zeros((2, 4))
    if y_margin:  # nonzero
        # Brown patch [cover vectorfield]
        edge_points[:, 0] = [x_min, gap_points[1, 1] + y_margin]
        edge_points[:, 1] = gap_points[:, 1]
        edge_points[:, 2] = gap_points[:, 0]
        edge_points[:, 3] = [x_min, gap_points[1, 0] - y_margin]

        door_wall_path = plt.Polygon(edge_points.T, alpha=1.0, zorder=3)
        door_wall_path.set_color(np.array([176, 124, 124]) / 255.0)
        ax.add_patch(door_wall_path)

    # White patch
    edge_points[:, 0] = [x_min, gap_points[1, 1]]
    edge_points[:, 1] = gap_points[:, 1]
    edge_points[:, 2] = gap_points[:, 0]
    edge_points[:, 3] = [x_min, gap_points[1, 0]]

    door_wall_path = plt.Polygon(edge_points.T, alpha=1.0, zorder=3)
    door_wall_path.set_color([1, 1, 1])
    ax.add_patch(door_wall_path)


def gamma_field_room_with_door(dim=2, num_resolution=20, x_min=None, fig=None, ax=None):
    x_range = [-1, 11]
    y_range = [-6, 6]

    obs_list = GradientContainer()
    obs_list.append(
        BoundaryCuboidWithGaps(
            name="RoomWithDoor",
            axes_length=[10, 10],
            center_position=[5, 0],
            gap_points_relative=np.array([[-5, -1], [-5, 1]]).T,
        )
    )

    # Let's move to the door
    attractor_position = obs_list["RoomWithDoor"].get_global_gap_center()

    if True:
        if fig is None or ax is None:
            fig_num = 1001
            fig, ax = plt.subplots(num=fig_num, figsize=(8, 6))

        Simulation_vectorFields(
            x_range,
            y_range,
            obs=obs_list,
            xAttractor=attractor_position,
            saveFigure=False,
            # obs_avoidance_func=obs_avoidance_interpolation_moving,
            # automatic_reference_point=True,
            noTicks=True,
            showLabel=False,
            show_streamplot=False,
            draw_vectorField=False,
            fig_and_ax_handle=(fig, ax),
            normalize_vectors=False,
        )

        x_vals = np.linspace(x_range[0], x_range[1], num_resolution)
        y_vals = np.linspace(y_range[0], y_range[1], num_resolution)

        pos = np.zeros((dim, num_resolution, num_resolution))
        gamma = np.zeros((num_resolution, num_resolution))

        for ix in range(num_resolution):
            for iy in range(num_resolution):
                pos[:, ix, iy] = [x_vals[ix], y_vals[iy]]

                if x_min is not None and pos[0, ix, iy] < x_min:
                    continue  #

                gamma[ix, iy] = obs_list["RoomWithDoor"].get_gamma(
                    pos[:, ix, iy], in_global_frame=True
                )

        cs = plt.contourf(
            pos[0, :, :],
            pos[1, :, :],
            gamma,
            # np.arange(0, 3.5, 0.05),
            # np.arange(1.0, 3.5, 0.25),
            # vmin=1, vmax=10,
            10 ** (np.linspace(0, 2, 11)),
            extend="max",
            locator=ticker.LogLocator(),
            alpha=0.6,
            zorder=3,
            # cmap="YlGn_r"
            # cmap="Purples_r"
            cmap="gist_gray",
        )

        cbar = fig.colorbar(cs, ticks=[1, 10, 100])
        cbar.ax.set_yticklabels(["1", "10", "100"])

        # print('gamma', np.round(gamma, 1))

        gap_points = obs_list[-1].get_global_gap_points()
        # ax.plot(gap_points[0, :], gap_points[1, :], color='white', linewidth='60', zorder=2)
        ax.plot(
            gap_points[0, :], gap_points[1, :], color="white", linewidth="2", zorder=2
        )

    # Gamma elements
    if True:
        obs = obs_list[-1]
        gap_points = obs.get_global_gap_points()

        # Hyper-Cone
        for ii in range(gap_points.shape[1]):
            plt.plot(
                [obs.center_position[0], gap_points[0, ii]],
                [obs.center_position[1], gap_points[1, ii]],
                ":",
                # color='#A9A9A9',
                # color='#0b1873',
                color="#00b3fa",
                zorder=4,
            )

        ax.plot(
            obs.center_position[0],
            obs.center_position[1],
            "k+",
            linewidth=18,
            markeredgewidth=4,
            markersize=13,
            zorder=4,
        )

        # Hyper(half)sphere
        n_points = 30
        angles = np.linspace(-pi / 2, pi / 2, n_points)
        # import pdb; pdb.set_trace()
        # circle_points = np.vstack( np.cos(angles), np.sin(angles))
        rad_gap = np.linalg.norm(obs.local_gap_center)
        ax.plot(
            rad_gap * np.cos(angles),
            rad_gap * np.sin(angles),
            ":",
            color="#22db12",
            zorder=4,
        )

    return obs_list


def vectorfield_room_with_door(
    dim=2, num_resolution=20, visualize_scene=True, obs_list=None, fig=None, ax=None
):
    x_range = [-1, 11]
    y_range = [-6, 6]

    if obs_list is None:
        obs_list = GradientContainer()
        obs_list.append(
            BoundaryCuboidWithGaps(
                name="RoomWithDoor",
                axes_length=[10, 10],
                center_position=[5, 0],
                gap_points_relative=np.array([[-5, -1], [-5, 1]]).T,
            )
        )

    # Let's move to the door
    attractor_position = obs_list["RoomWithDoor"].get_global_gap_center()

    if False:
        pos = np.array([4.6, -1.5])
        xd_init = linear_ds_max_vel(pos, attractor=attractor_position)
        xd = obs_avoidance_interpolation_moving(pos, xd_init, obs_list)

        print("xd", xd)
        import pdb

        pdb.set_trace()

    if True:
        if fig is None or ax is None:
            fig_num = 1001
            fig, ax = plt.subplots(num=fig_num, figsize=(8, 6))

        Simulation_vectorFields(
            x_range,
            y_range,
            obs=obs_list,
            xAttractor=attractor_position,
            saveFigure=False,
            # obs_avoidance_func=obs_avoidance_interpolation_moving,
            noTicks=True,
            showLabel=False,
            # automatic_reference_point=True,
            point_grid=num_resolution,
            # show_streamplot=False,
            show_streamplot=True,
            draw_vectorField=True,
            fig_and_ax_handle=(fig, ax),
            normalize_vectors=False,
        )

        # Draw gap
        plt.plot(
            obs_list[-1].center_position[0],
            obs_list[-1].center_position[1],
            "k+",
            markersize=10,
            linewidth=10,
        )
        points = obs_list["RoomWithDoor"].get_global_gap_points()
        # ax.plot(points[0, :], points[1, :], color='#46979e', linewidth='10')
        ax.plot(points[0, :], points[1, :], color="white", linewidth="2")


def test_projected_reference(
    max_it=1000,
    delta_time=0.01,
    max_num_obstacles=5,
    dim=2,
    visualize_scene=True,
    random_seed=None,
):
    x_range = [-1, 11]
    y_range = [-6, 6]

    obs_list = GradientContainer()
    obs_list.append(
        BoundaryCuboidWithGaps(
            name="RoomWithDoor",
            axes_length=[10, 10],
            center_position=[5, 0],
            gap_points_relative=np.array([[-5, -1], [-5, 1]]).T,
        )
    )

    # Let's move to the door
    attractor_position = obs_list["RoomWithDoor"].get_global_gap_center()

    if True:
        fig_num = 1001
        fig, ax = plt.subplots(num=fig_num, figsize=(8, 6))

        Simulation_vectorFields(
            x_range,
            y_range,
            obs=obs_list,
            xAttractor=attractor_position,
            saveFigure=False,
            # obs_avoidance_func=obs_avoidance_interpolation_moving,
            noTicks=False,
            # automatic_reference_point=True,
            # show_streamplot=False,
            draw_vectorField=False,
            show_streamplot=False,
            fig_and_ax_handle=(fig, ax),
            normalize_vectors=False,
        )

        # Draw gap
        plt.plot(
            obs_list[-1].center_position[0],
            obs_list[-1].center_position[1],
            "k+",
            markersize=10,
            linewidth=10,
        )
        points = obs_list["RoomWithDoor"].get_global_gap_points()
        ax.plot(points[0, :], points[1, :], color="#46979e", linewidth="10")
        ax.plot(points[0, :], points[1, :], color="white", linewidth="10")

        for jj in range(30):
            it_max = 100
            for ii in range(it_max):
                position = np.random.uniform(size=(2))
                position[0] = position[0] * (x_range[1] - x_range[0]) + x_range[0]
                position[1] = position[1] * (y_range[1] - y_range[0]) + y_range[0]

                if position_is_in_free_space(position, obs_list):
                    break

            ref_proj = obs_list["RoomWithDoor"].get_projected_reference(position)
            plt.plot(position[0], position[1], "r+")
            plt.plot(ref_proj[0], ref_proj[1], "g+")
            plt.plot([position[0], ref_proj[0]], [position[1], ref_proj[1]], "k--")


def animation_wall_with_door(
    max_it=1000,
    delta_time=0.01,
    max_num_obstacles=5,
    dim=2,
    visualize_scene=True,
    random_seed=None,
):
    x_range = [-1, 11]
    y_range = [-6, 6]

    attractor_position = np.array([10, 0])

    obs_list = GradientContainer()
    obs_list.append(
        BoundaryCuboidWithGaps(
            name="RoomWithDoor",
            axes_length=[10, 10],
            center_position=[5, 0],
            gap_points_relative=np.array([[-5, -1], [-5, 1]]).T,
            angular_velocity=0.3 * pi,
            # angular_velocity=0.0,
            orientation=0 * pi / 180,
            # orientation=0*pi/180,
            # relative_expansion_speed=np.array([-0.6, -0.6]),
            # relative_expansion_speed=np.array([0.0, 0.0]),
            expansion_speed_axes=np.array([-4.0, -4.0]),
            wall_thickness=1,
            # angular_velocity=0.1,
        )
    )

    # Let's move to the door
    attractor_position = obs_list["RoomWithDoor"].get_global_gap_center()
    start_position = np.array([9, 4])

    agents = []
    agents.append(
        ObstacleAvoidanceAgent(
            start_position=start_position,
            name="Dynamic",
            avoidance_function=obs_avoidance_interpolation_moving,
            attractor=attractor_position,
        )
    )

    if visualize_scene:
        fig_num = 1001
        fig, ax = plt.subplots(num=fig_num, figsize=(20, 12))

    for ii in range(max_it):
        for obs in obs_list:
            if obs.is_deforming:
                # obs.update_deforming_obstacle(delta_time=delta_time)
                obs.update_step(delta_time=delta_time)

        ax.cla()

        # attractor_position = obs_list['RoomWithDoor'].get_global_gap_center()
        attractor_position = obs_list["RoomWithDoor"].get_gap_outside_point(
            dist_relative=2
        )

        Simulation_vectorFields(
            x_range,
            y_range,
            obs=obs_list,
            xAttractor=attractor_position,
            saveFigure=False,
            obs_avoidance_func=obs_avoidance_interpolation_moving,
            noTicks=True,
            showLabel=False,
            # automatic_reference_point=True,
            # show_streamplot=False,
            draw_vectorField=False,
            show_streamplot=False,
            fig_and_ax_handle=(fig, ax),
            normalize_vectors=False,
            border_linestyle="-",
        )

        # plt.text(x_range[1]-1.0, y_range[1]-1.0, str(round(obs_list['RoomWithDoor'].orientation/pi*180, 2)))

        # plt.axis('equal')
        # plt.pause(0.01)

        if False:
            plt.plot(
                obs_list["RoomWithDoor"].center_position[0],
                obs_list["RoomWithDoor"].center_position[1],
                "k+",
            )

        for agent in agents:
            initial_velocity = linear_ds_max_vel(
                position=agent.position,
                attractor=attractor_position,
                vel_max=0.5,
            )

            agent.update_step(obs_list, initial_velocity=initial_velocity)
            # agent.check_collision(obs_list)
            # import pdb; pdb.set_trace()

            if visualize_scene:
                plt.plot(
                    agent.position_list[0, :],
                    agent.position_list[1, :],
                    "--",
                    label=agent.name,
                    zorder=5,
                    color="blue",
                )
                plt.plot(
                    agent.position_list[0, -1],
                    agent.position_list[1, -1],
                    "o",
                    label=agent.name,
                    zorder=5,
                    color="blue",
                )
                # print(agent.position_list)

        # put_patch_behind_gap(ax=ax, obs_list=obs_list, x_min=-0.97, y_margin=0)

        obs_list["RoomWithDoor"].get_gap_patch(ax=ax, x_lim=x_range, y_lim=y_range)

        # plt.axis('equal')
        plt.pause(0.01)

        if visualize_scene and not plt.fignum_exists(fig_num):
            print(f"Simulation ended with closing of figure")
            plt.pause(0.01)
            plt.close("all")
            break
    pass


def animation_ellipse(
    max_it=1000,
    delta_time=0.01,
    max_num_obstacles=5,
    dim=2,
    visualize_scene=True,
    random_seed=None,
):
    x_range = [-1, 11]
    y_range = [-5, 5]

    attractor_position = np.array([10, 0])

    obs_list = GradientContainer()
    obs_list.append(
        Ellipse(
            axes_length=[2, 1],
            center_position=[5, 0],
            orientation=0,
            linear_velocity=[0, 0],
            expansion_speed_axes=[1.5, 1.5],  # per axis
            tail_effect=False,
        )
    )

    start_position = np.array([0, 2])

    agents = []
    agents.append(
        ObstacleAvoidanceAgent(
            start_position=start_position,
            name="Dynamic",
            avoidance_function=obs_avoidance_interpolation_moving,
            attractor=attractor_position,
        )
    )

    if visualize_scene:
        fig_num = 1001
        fig, ax = plt.subplots(num=fig_num, figsize=(20, 12))

    for ii in range(max_it):
        for obs in obs_list:
            if obs.is_deforming:
                obs.update_deforming_obstacle(delta_time=delta_time)

        ax.cla()

        Simulation_vectorFields(
            x_range,
            y_range,
            obs=obs_list,
            xAttractor=attractor_position,
            saveFigure=False,
            obs_avoidance_func=obs_avoidance_interpolation_moving,
            noTicks=True,
            showLabel=False,
            # automatic_reference_point=True,
            # show_streamplot=False,
            draw_vectorField=False,
            show_streamplot=False,
            fig_and_ax_handle=(fig, ax),
            normalize_vectors=False,
            border_linestyle="-",
        )

        if True:
            plt.plot(
                obs_list[-1].center_position[0], obs_list[-1].center_position[1], "k+"
            )

        for agent in agents:
            initial_velocity = linear_ds_max_vel(
                position=agent.position, attractor=attractor_position, vel_max=1.0
            )
            agent.update_step(obs_list, initial_velocity=initial_velocity)
            # agent.check_collision(obs_list)

            if visualize_scene:
                plt.plot(
                    agent.position_list[0, :],
                    agent.position_list[1, :],
                    "--",
                    label=agent.name,
                    color="blue",
                )
                plt.plot(
                    agent.position_list[0, -1],
                    agent.position_list[1, -1],
                    "o",
                    label=agent.name,
                    zorder=5,
                    color="blue",
                )

        # plt.axis('equal')
        plt.pause(0.01)

        if visualize_scene and not plt.fignum_exists(fig_num):
            print(f"Simulation ended with closing of figure")
            plt.pause(0.01)
            plt.close("all")
            break


if (__name__) == "__main__":
    obs_list = None
    animation_ellipse()
    # test_projected_reference()

    # animation_wall_with_door()

    # 2 plots with vectorfield & Gamma-value of boundary-region
    if False:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        plt.subplots_adjust(wspace=-0.4)

        # points_frac = 0.1
        points_frac = 1.0

        # fig, ax2 = plt.subplots()
        obs_list = gamma_field_room_with_door(
            num_resolution=int(points_frac * 80), x_min=0, fig=fig, ax=ax2
        )
        put_patch_behind_gap(ax=ax2, obs_list=obs_list, x_min=-0.97)
        # plt.savefig('figures/' + 'boundary_with_gap_gamma' + '.png', bbox_inches='tight')

        ax2.text(
            x=1.2, y=-0.1, s=r"$\mathcal{G}$", fontsize="xx-large", color="#00b3fa"
        )

        # fig, ax1 = plt.subplots()
        line = plt_speed_line_and_qolo(
            points_init=np.array([9, 3]),
            attractorPos=obs_list[-1].get_global_gap_center(),
            obs=obs_list,
            fig_and_ax_handle=(fig, ax1),
            dt=0.02,
            line_color="#22db12",
        )

        vectorfield_room_with_door(
            num_resolution=int(points_frac * 100), fig=fig, ax=ax1, obs_list=obs_list
        )
        put_patch_behind_gap(ax=ax1, obs_list=obs_list, x_min=-0.97, y_margin=4)
        # plt.savefig('figures/' + 'boundary_with_gap' + '.png', bbox_inches='tight')

        plt.savefig(
            "figures/" + "boundary_with_gap_subplot" + ".png", bbox_inches="tight"
        )
