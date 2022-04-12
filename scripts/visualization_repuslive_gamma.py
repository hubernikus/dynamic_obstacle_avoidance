#!/USSR/bin/python3
""" Script which creates a variety of examples of local modulation of a vector field
with obstacle avoidance. 
"""
# Author: LukasHuber
# Email: lukas.huber@epfl.ch
# Created:  2018-02-15

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt

# from dynamic_obstacle_avoidance.dynamical_system import linearAttractor
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import (
    obs_avoidance_interpolation_moving,
)

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import *


def get_outside_enviornment_simplified(robot_margin=0.6):
    obs = GradientContainer()  # create empty obstacle list
    obs.append(
        Polygon(
            edge_points=[[-1.8, 5.4, 5.4, -1.8], [-1.8, -1.8, 0.9, 0.9]],
            center_position=[3.0, -1],
            orientation=0.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=True,
        )
    )

    obs.append(
        Cuboid(
            axes_length=[0.4, 0.4],
            center_position=[3.0, -0.5],
            orientation=90.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=False,
            repulsion_coeff=2.0,
        )
    )

    return obs


def get_outside_enviornment(robot_margin=0.6):
    obs = GradientContainer()  # create empty obstacle list
    obs.append(
        Polygon(
            edge_points=[
                [-1.8, 4.5, 4.5, 5.4, 5.4, -1.8],
                [-3.6, -3.6, -1.8, -1.8, 0.9, 0.9],
            ],
            center_position=[3.0, -1],
            orientation=0.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=True,
        )
    )

    obs.append(
        Cuboid(
            axes_length=[0.4, 0.4],
            center_position=[3.0, 0.0],
            orientation=90.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=False,
            repulsion_coeff=2.0,
        )
    )

    return obs


def get_outside_enviornment_lshape(robot_margin=0.6):

    obs = GradientContainer()  # create empty obstacle list

    obs.append(
        Polygon(
            edge_points=[
                [0.0, 4.0, 4.0, 2.0, 2.0, 0.0],
                [0.0, 0.0, 2.0, 2.0, 4.0, 4.0],
            ],
            center_position=[0.35, 0.35],
            orientation=0.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=True,
        )
    )

    obs.append(
        Ellipse(
            axes_length=[0.2, 0.3],
            center_position=[1.0, 1.4],
            orientation=20.0 / 180 * pi,
            margin_absolut=robot_margin,
        )
    )

    return obs


def get_outside_enviornment_lshape_hack(robot_margin=0.6):

    obs = GradientContainer()  # create empty obstacle list

    obs.append(
        Polygon(
            edge_points=[[0.0, 4.0, 4.0, 0.0], [0.0, 0.0, 4.0, 4.0]],
            orientation=0.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=True,
        )
    )

    obs.append(
        Cuboid(
            axes_length=np.array([3.0, 3.0]),
            center_position=[3.5, 3.5],
            orientation=0.0 / 180 * pi,
            margin_absolut=robot_margin,
        )
    )

    obs.append(
        Ellipse(
            axes_length=[0.2, 0.3],
            center_position=[1.0, 1.4],
            orientation=20.0 / 180 * pi,
            margin_absolut=robot_margin,
            repulsion_coeff=2.0,
        )
    )

    return obs


def get_attracting_square(margin_absolut=0):
    obs = GradientContainer()  # create empty obstacle list

    obs.append(
        Cuboid(
            axes_length=[0.4, 0.4],
            center_position=[0.0, 0.0],
            orientation=0.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=False,
            # repulsion_coeff=-2.0,
            repulsion_coeff=-2.0,
            tail_effect=False,
            name="center_cube",
            has_sticky_surface=False,
        )
    )

    return obs


def get_attracting_cirlce(margin_absolut=0):
    obs = GradientContainer()  # create empty obstacle list

    obs.append(
        CircularObstacle(
            radius=0.2,
            center_position=[0.0, 0.0],
            orientation=0.0 / 180 * pi,
            margin_absolut=margin_absolut,
            is_boundary=False,
            repulsion_coeff=-2.0,
            # tail_effect=False,
            has_sticky_surface=False,
            name="center_cube",
        )
    )

    return obs


def get_repulsive_cirlce(margin_absolut=0):
    obs = GradientContainer()  # create empty obstacle list

    obs.append(
        CircularObstacle(
            radius=0.2,
            center_position=[0.0, 0.0],
            orientation=0.0 / 180 * pi,
            margin_absolut=margin_absolut,
            is_boundary=False,
            repulsion_coeff=2.0,
            # tail_effect=False,
            has_sticky_surface=False,
            name="center_cube",
        )
    )
    return obs


def get_repulsive_ellipse(margin_absolut=0):
    obs = GradientContainer()  # create empty obstacle list

    obs.append(
        Ellipse(
            axes_length=[0.3, 0.4],
            center_position=[0.0, 0.0],
            orientation=0.0 / 180 * pi,
            margin_absolut=margin_absolut,
            is_boundary=False,
            repulsion_coeff=2.0,
            # tail_effect=False,
            has_sticky_surface=False,
            name="center_cube",
        )
    )
    return obs


if (__name__) == "__main__":
    num_resolution = 30
    saveFigures = True

    x_lim = [-0.3, 4.5]
    y_lim = [-2.5, 0.5]

    xAttractor = [3.5, 1.0]

    robot_margin = 0.3

    # obs = get_outside_enviornment(robot_margin)
    # obs = get_outside_enviornment_simplified(robot_margin)
    # obs = get_outside_enviornment_lshape(robot_margin)
    # obs = get_outside_enviornment_lshape_hack(robot_margin)

    if True:
        xAttractor = np.array([2.5, 0])
        x_lim, y_lim = [-2.0, 3.5], [-2.1, 2.1]
        # x_lim, y_lim = [-1.0, 1.0], [-1.1, 1.1]
        obs = get_repulsive_ellipse(robot_margin)
        fig_name = "attracting_circle"

    if False:
        xAttractor = np.array([2.5, 0])
        x_lim, y_lim = [-2.0, 3.5], [-2.1, 2.1]
        # x_lim, y_lim = [-1.0, 1.0], [-1.1, 1.1]
        obs = get_repulsive_cirlce(robot_margin)
        fig_name = "attracting_circle"

    if False:
        xAttractor = np.array([2.5, 0])
        x_lim, y_lim = [-2.0, 3.5], [-2.1, 2.1]
        # x_lim, y_lim = [-1.0, 1.0], [-1.1, 1.1]
        obs = get_attracting_cirlce(robot_margin)
        fig_name = "attracting_circle"

    if False:
        obs = get_attracting_square(robot_margin)
        fig_name = "repulsive_square"

    vectorfield = True
    if vectorfield:
        # fig_mod, ax_mod = Simulation_vectorFields(
        #     x_lim, y_lim, obs=obs, xAttractor=xAttractor, saveFigure=saveFigures,
        #     figName='linearSystem_boundaryCuboid', noTicks=False, draw_vectorField=True,
        #     show_streamplot=True,
        #     automatic_reference_point=True, point_grid=N_resol,
        #     # figureSize=(6., 4.25),
        # )

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            xAttractor=xAttractor,
            point_grid=num_resolution,
            figName=fig_name,
            show_streamplot=False,
            noTicks=False,
            vector_field_only_outside=False,
            saveFigure=saveFigures,
        )

    else:
        # Specific value
        # position = np.array([1.06, 1.07])
        position = np.array([0.0, 0.5])

        ds_init = linearAttractor(position, x0=xAttractor)
        ds_mod = obs_avoidance_interpolation_moving(position, ds_init, obs)

        print("ds_mod", ds_mod)
        # import pdb; pdb.set_trace()

# Run function
