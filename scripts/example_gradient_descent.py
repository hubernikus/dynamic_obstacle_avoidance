#!/USSR/bin/python3
"""
Tests and visualizes different dynamic boundary things. 
"""
__author__ = "LukasHuber"
__date__ = "2018-02-15"

from math import pi

import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import (
    CircularObstacle,
    Ellipse,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import (
    Cuboid,
    Polygon,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import (
    GradientContainer,
)
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
)  #
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *


def visualization_boundary_reference_point():
    LocalCrowd = GradientContainer()
    # LocalCrowd.append(
    # CircularObstacle(center_position=np.array([-0.2, 1.9]), radius=1.5, margin_absolut=0.3))
    LocalCrowd.append(
        CircularObstacle(
            center_position=np.array([2.4, -1.2]), radius=1.5, margin_absolut=0.3
        )
    )
    LocalCrowd.append(
        CircularObstacle(
            center_position=np.array([0.1, 0.07]),
            radius=5,
            is_boundary=True,
            margin_absolut=0.7,
        )
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    Simulation_vectorFields(
        x_range=[-5.5, 5.5],
        y_range=[-5.5, 5.5],
        obs=LocalCrowd,
        xAttractor=[6, 0],
        saveFigure=False,
        figName="linearSystem_boundaryCuboid",
        noTicks=False,
        draw_vectorField=False,
        reference_point_number=False,
        drawVelArrow=True,
        automatic_reference_point=True,
        point_grid=10,
        fig_and_ax_handle=(fig, ax),
    )

    for ii in range(len(LocalCrowd)):
        for jj in range(len(LocalCrowd)):
            if jj == ii:
                continue
            point = LocalCrowd.get_boundary_reference_point(ii, jj)
            ax.plot(
                point[0], point[1], "r+", linewidth=18, markeredgewidth=4, markersize=13
            )

    import pdb

    pdb.set_trace()  ##### DEBUG #####


def visualization_boundary_points_mixed_world():
    from dynamic_obstacle_avoidance.settings import DEBUG_FLAG
    from dynamic_obstacle_avoidance import settings

    if "DEBUG_FLAG" in globals() and DEBUG_FLAG:
        settings.init()

    x_lim, y_lim = [-0.5, 6.5], [-0.5, 6.0]

    xAttractor = [5.0, 5.0]
    robot_margin = 0.6

    obstacle_list = GradientContainer()  # create empty obstacle list

    obstacle_list.append(
        Polygon(
            edge_points=[[0.0, 5.7, 5.7, 0.0], [-0.6, -0.6, 5.5, 5.5]],
            # center_position=[3, 3.5],
            orientation=0.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=True,
        )
    )

    # obstacle_list.append(Cuboid(
    # axes_length=[5.7, 6.1],
    # center_position=[5.7/2, 6.1/2],
    # orientation=90./180*pi,
    # margin_absolut=robot_margin,
    # is_boundary=True
    # ))

    # import pdb; pdb.set_trace()     ##### DEBUG #####

    obstacle_list.append(
        Cuboid(
            axes_length=[1.6, 0.8],
            center_position=[0.4, 3.0],
            orientation=90.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=False,
        )
    )

    obstacle_list.append(
        Cuboid(
            axes_length=[1.6, 1.6],
            center_position=[3.4, 2.6],
            orientation=0.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=False,
        )
    )

    obstacle_list.append(
        Ellipse(
            axes_length=[0.3, 0.5],
            center_position=[1.8, 3.0],
            p=[1, 1],
            orientation=30.0 / 180 * pi,
            margin_absolut=robot_margin,
            is_boundary=False,
        )
    )

    # return obstacle_list

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    Simulation_vectorFields(
        x_range=x_lim,
        y_range=y_lim,
        obs=obstacle_list,
        xAttractor=[6, 0],
        saveFigure=False,
        figName="linearSystem_boundaryCuboid",
        noTicks=False,
        draw_vectorField=False,
        reference_point_number=False,
        drawVelArrow=True,
        automatic_reference_point=True,
        point_grid=10,
        fig_and_ax_handle=(fig, ax),
    )

    # import pdb; pdb.set_trace()     ##### DEBUG #####
    for ii in range(len(obstacle_list)):
        for jj in range(len(obstacle_list)):
            if jj == ii:
                continue
            point = obstacle_list.get_boundary_reference_point(ii, jj)
            ax.plot(
                point[0], point[1], "r+", linewidth=18, markeredgewidth=4, markersize=13
            )

    if "DEBUG_FLAG" in globals() and DEBUG_FLAG:
        # import pdb; pdb.set_trace()
        # global settings.boundary_ref_point_list

        settings.position0 = np.array(settings.position0)
        plt.plot(settings.position0[:, 0], settings.position0[:, 1], marker="x")

        settings.position1 = np.array(settings.position1)
        plt.plot(settings.position1[:, 0], settings.position1[:, 1], marker="x")

        # import pdb; pdb.set_trace()

        settings.boundary_ref_point_list = np.array(settings.boundary_ref_point_list)
        plt.figure()
        plt.plot(settings.dist_ref_points)

        plt.figure()
        plt.plot(
            settings.boundary_ref_point_list[:, 0],
            settings.boundary_ref_point_list[:, 1],
        )


def main(options=[0], N_resol=100, saveFigures=False):
    from dynamic_obstacle_avoidance.settings import DEBUG_FLAG
    from dynamic_obstacle_avoidance import settings

    if -1 in options:
        obs = GradientContainer()  # create empty obstacle list
        x_lim, y_lim = [-3, 3], [-2, 2]

        obs.append(
            Ellipse(
                axes_length=[0.8, 0.9],
                # p=[1,1],
                x0=[0, 0],
                orientation=30.0 / 180 * pi,
            )
        )

        obs.append(
            Ellipse(
                axes_length=[0.9, 0.9],
                # p=[1,1],
                x0=[2, 0],
                orientation=-65.0 / 180 * pi,
            )
        )

        xAttractor = [-2.0, 0.2]

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            xAttractor=xAttractor,
            saveFigure=saveFigures,
            figName="linearSystem_avoidanceCube",
            noTicks=False,
            draw_vectorField=True,
            automatic_reference_point=True,
            point_grid=N_resol,
            show_streamplot=False,
        )

        for ii in range(len(obs)):
            for jj in range(len(obs)):
                if ii == jj:
                    continue
                boundary_ref_point = obs.get_boundary_reference_point(ii, jj)
                plt.plot(
                    boundary_ref_point[0],
                    boundary_ref_point[1],
                    "r+",
                    linewidth=18,
                    markeredgewidth=4,
                    markersize=13,
                )

    if 0 in options:
        obs = GradientContainer()  # create empty obstacle list
        x_lim, y_lim = [-3, 3], [-2, 2]

        obs.append(
            Ellipse(
                axes_length=[0.4, 0.7],
                # p=[1,1],
                x0=[0, 0],
                orientation=30.0 / 180 * pi,
            )
        )
        obs.append(
            Ellipse(
                axes_length=[0.3, 0.8],
                # p=[1,1],
                x0=[2, 0],
                orientation=-65.0 / 180 * pi,
            )
        )

        xAttractor = [-2.0, 0.2]

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            xAttractor=xAttractor,
            saveFigure=saveFigures,
            figName="linearSystem_avoidanceCube",
            noTicks=False,
            draw_vectorField=True,
            automatic_reference_point=True,
            point_grid=N_resol,
            show_streamplot=False,
        )

        for ii in range(len(obs)):
            for jj in range(len(obs)):
                if ii == jj:
                    continue
                boundary_ref_point = obs.get_boundary_reference_point(ii, jj)
                plt.plot(
                    boundary_ref_point[0],
                    boundary_ref_point[1],
                    "r+",
                    linewidth=18,
                    markeredgewidth=4,
                    markersize=13,
                )

    if 1 in options:
        obs = GradientContainer()  # create empty obstacle list
        x_lim, y_lim = [-3, 3], [-2, 2]

        obs.append(
            Ellipse(
                axes_length=[0.4, 0.7],
                # p=[1,1],
                x0=[0, 0],
                orientation=30.0 / 180 * pi,
            )
        )

        obs.append(
            Cuboid(
                axes_length=[0.8, 1.8],
                # p=[1,1],
                x0=[2, 0],
                orientation=-65.0 / 180 * pi,
            )
        )

        xAttractor = [-2.0, 0.2]

        if "DEBUG_FLAG" in globals() and DEBUG_FLAG:
            settings.init()
            # import pdb; pdb.set_trace()

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            xAttractor=xAttractor,
            saveFigure=saveFigures,
            figName="linearSystem_avoidanceCube",
            noTicks=False,
            draw_vectorField=True,
            automatic_reference_point=True,
            point_grid=N_resol,
            show_streamplot=False,
        )

        for ii in range(len(obs)):
            for jj in range(len(obs)):
                if ii == jj:
                    continue
                boundary_ref_point = obs.get_boundary_reference_point(ii, jj)
                plt.plot(
                    boundary_ref_point[0],
                    boundary_ref_point[1],
                    "r+",
                    linewidth=18,
                    markeredgewidth=4,
                    markersize=13,
                )

        if "DEBUG_FLAG" in globals() and DEBUG_FLAG:
            # import pdb; pdb.set_trace()
            # global settings.boundary_ref_point_list

            settings.position0 = np.array(settings.position0)
            plt.plot(settings.position0[:, 0], settings.position0[:, 1], marker="x")

            settings.position1 = np.array(settings.position1)
            plt.plot(settings.position1[:, 0], settings.position1[:, 1], marker="x")

            # import pdb; pdb.set_trace()

            settings.boundary_ref_point_list = np.array(
                settings.boundary_ref_point_list
            )
            plt.figure()
            plt.plot(settings.dist_ref_points)

            plt.figure()
            plt.plot(
                settings.boundary_ref_point_list[:, 0],
                settings.boundary_ref_point_list[:, 1],
            )

            # import pdb; pdb.set_trace()

        # dist_ref_points

    if 2 in options:
        obs = GradientContainer()  # create empty obstacle list
        x_lim, y_lim = [-3, 3], [-2, 2]

        obs.append(
            Cuboid(
                axes_length=[1.3, 1.3],
                # p=[1,1],
                x0=[-1, 0],
                orientation=0.0 / 180 * pi,
            )
        )

        obs.append(
            Cuboid(
                axes_length=[1.3, 1.3],
                # p=[1,1],
                x0=[1, 0],
                orientation=0.0 / 180 * pi,
            )
        )

        xAttractor = [-2.0, 0.2]

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            xAttractor=xAttractor,
            saveFigure=saveFigures,
            figName="linearSystem_avoidanceCube",
            noTicks=False,
            draw_vectorField=True,
            automatic_reference_point=True,
            point_grid=N_resol,
            show_streamplot=False,
        )

        for ii in range(len(obs)):
            for jj in range(len(obs)):
                if ii == jj:
                    continue
                boundary_ref_point = obs.get_boundary_reference_point(ii, jj)
                plt.plot(
                    boundary_ref_point[0],
                    boundary_ref_point[1],
                    "r+",
                    linewidth=18,
                    markeredgewidth=4,
                    markersize=13,
                )

    if 3 in options:
        obs = GradientContainer()  # create empty obstacle list
        x_lim, y_lim = [-3, 3], [-2, 2]

        obs.append(
            Cuboid(axes_length=[1.3, 1.3], x0=[-1, 0.5], orientation=0.0 / 180 * pi)
        )

        obs.append(
            Cuboid(
                axes_length=[1.3, 1.3],
                # p=[1,1],
                x0=[1, -0.5],
                orientation=0.0 / 180 * pi,
            )
        )

        xAttractor = [-2.0, 0.2]

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            xAttractor=xAttractor,
            saveFigure=saveFigures,
            figName="linearSystem_avoidanceCube",
            noTicks=False,
            draw_vectorField=True,
            automatic_reference_point=True,
            point_grid=N_resol,
            show_streamplot=False,
        )

        for ii in range(len(obs)):
            for jj in range(len(obs)):
                if ii == jj:
                    continue
                boundary_ref_point = obs.get_boundary_reference_point(ii, jj)
                plt.plot(
                    boundary_ref_point[0],
                    boundary_ref_point[1],
                    "r+",
                    linewidth=18,
                    markeredgewidth=4,
                    markersize=13,
                )

    if 4 in options:
        obs = GradientContainer()  # create empty obstacle list
        x_lim, y_lim = [-3, 3], [-2, 2]

        obs.append(
            Cuboid(
                axes_length=[1.3, 1.3],
                margin_absolut=0.4,
                center_position=[-1, -0.5],
                orientation=0.0 / 180 * pi,
            )
        )

        obs.append(
            Cuboid(
                axes_length=[1.3, 1.3],
                margin_absolut=0.4,
                center_position=[1, 0.5],
                orientation=0.0 / 180 * pi,
            )
        )

        xAttractor = [-2.0, 0.2]

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            xAttractor=xAttractor,
            saveFigure=saveFigures,
            figName="linearSystem_avoidanceCube",
            noTicks=False,
            draw_vectorField=True,
            automatic_reference_point=True,
            point_grid=N_resol,
            show_streamplot=False,
        )

        for ii in range(len(obs)):
            for jj in range(len(obs)):
                if ii == jj:
                    continue
                boundary_ref_point = obs.get_boundary_reference_point(ii, jj)
                plt.plot(
                    boundary_ref_point[0],
                    boundary_ref_point[1],
                    "r+",
                    linewidth=18,
                    markeredgewidth=4,
                    markersize=13,
                )

    if 5 in options:
        obs = GradientContainer()  # create empty obstacle list
        x_lim, y_lim = [-3, 3], [-2, 2]

        obs.append(
            Cuboid(
                axes_length=[1.3, 1.3],
                # margin_absolut=0.4,
                center_position=[-1, -0.5],
                orientation=0.0 / 180 * pi,
            )
        )

        obs.append(
            Cuboid(
                axes_length=[1.3, 2.6],
                # margin_absolut=0.4,
                center_position=[1, 0.0],
                orientation=0.0 / 180 * pi,
            )
        )

        xAttractor = [-2.0, 0.2]

        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            xAttractor=xAttractor,
            saveFigure=saveFigures,
            figName="linearSystem_avoidanceCube",
            noTicks=False,
            draw_vectorField=True,
            automatic_reference_point=True,
            point_grid=N_resol,
            show_streamplot=False,
        )

        for ii in range(len(obs)):
            for jj in range(len(obs)):
                if ii == jj:
                    continue
                boundary_ref_point = obs.get_boundary_reference_point(ii, jj)
                plt.plot(
                    boundary_ref_point[0],
                    boundary_ref_point[1],
                    "r+",
                    linewidth=18,
                    markeredgewidth=4,
                    markersize=13,
                )

        plt.ion()
        plt.show()
        import pdb

        pdb.set_trace()  ##### DEBUG #####


if (__name__) == "__main__":
    print("voila")
    # visualization_boundary_reference_point()
    visualization_boundary_points_mixed_world()

    # main(options=options, N_resol=N_resol, saveFigures=saveFigures)


# Run function
