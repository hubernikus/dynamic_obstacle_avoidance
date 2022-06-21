"""
Setup file of conference room
"""
__author__ = "Lukas Huber"
__date__ = "2020-07-10"
__email__ = "lukas.huber@epfl.ch"

from math import pi

if (__name__) == "__main__":
    import sys
    import os

    # import os
    # os.path.dirname(os.path.abspath(__file__))

    # path_avoidance = "/home/lukas/catkin_ws/src/qolo_modulation/scripts/dynamic_obstacle_avoidance/src"
    # path_avoidance = "/home/qolo/autonomy_ws/src/qolo_modulation/scripts/dynamic_obstacle_avoidance/src"

    if not path_avoidance in sys.path:
        sys.path.append(path_avoidance)

# Custom libraries
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import (
    GradientContainer,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import Ellipse
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import (
    Polygon,
    Cuboid,
)

# Visualization
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #

# Manually set


def get_conference_room_setup(robot_margin=0.6):
    obstacle_list = GradientContainer()  # create empty obstacle list

    xAttractor = [6, 7]

    obstacle_list.append(
        Polygon(
            edge_points=[[0.0, 5.7, 5.7, 0.0], [-0.6, -0.6, 5.5, 5.5]],
            center_position=[5, 3.5],
            orientation=0.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=True,
        )
    )

    obstacle_list.append(
        Cuboid(
            name="Table Wall",
            axes_length=[1.6, 0.8],
            center_position=[0.4, 3.0],
            orientation=90.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=False,
        )
    )

    obstacle_list.append(
        Cuboid(
            name="Table Center",
            axes_length=[1.6, 1.6],
            # center_position=[3.4, 2.6],
            center_position=[2.4, 2.6],
            orientation=0.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=False,
        )
    )

    obstacle_list.append(
        Ellipse(
            name="Pupped",
            axes_length=[0.2, 0.5],
            center_position=[1.8, 5.0],
            p=[1, 1],
            orientation=30.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=False,
        )
    )

    return obstacle_list


def get_conference_room_setup_old(robot_margin=0.6):
    obstacle_list = GradientContainer()  # create empty obstacle list

    xAttractor = [6, 7]

    # obs.append(Cuboid(
    # axes_length=[8, 7],
    # center_position=[3, 3.5],
    # orientation=0./180*pi,
    # margin_absolut=robot_margin,
    # is_boundary=True,
    # ))

    # obstacle_list.append(Polygon(
    #     name="meeting_room",
    #     edge_points=[[-1, 6, 6,-1,-1.0, 0.0, 0.0,-1.0],
    #                  [0, 0, 7, 7, 5.0, 4.5, 2.5, 2.0]],
    #     center_position=[5, 3.5],
    #     orientation=0./180*pi,
    #     margin_absolut=robot_margin,
    #     is_boundary=True,
    # ))

    obstacle_list.append(
        Cuboid(
            name="table_center",
            axes_length=[1.8, 1.0],
            center_position=[4.5, 4.1],
            orientation=0.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=False,
        )
    )

    obstacle_list.append(
        Ellipse(
            name="pupped",
            axes_length=[0.2, 0.5],
            center_position=[1.8, 5.0],
            p=[1, 1],
            orientation=30.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=False,
        )
    )

    return obstacle_list


if (__name__) == "__main__":
    # Visualize vector field to test
    N_resol = 80
    saveFigures = False

    obs = get_conference_room_setup(robot_margin=0.6)

    xAttractor = [5, 6.5]
    x_lim, y_lim = [-1.1, 6.1], [-0.1, 7.1]

    # xAttractor = [0., 0]
    # x_lim, y_lim = [-3,3], [-0.1,5]

    fig_mod, ax_mod = Simulation_vectorFields(
        x_lim,
        y_lim,
        obs=obs,
        xAttractor=xAttractor,
        saveFigure=False,
        figName="linearSystem_avoidanceCube",
        draw_vectorField=False,
        automatic_reference_point=True,
        noTicks=False,
    )
