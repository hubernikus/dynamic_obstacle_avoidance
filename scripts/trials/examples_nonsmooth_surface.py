#!/USSR/bin/python3

"""
Script which creates a variety of examples of local modulation of a vector field with obstacle avoidance. 

__author__ = "LukasHuber"
__date__ = "2020-02-15"
"""

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #

# from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.visualization.animated_simulation import (
    run_animation,
    samplePointsAtBorder,
)

########################################################################
options = [0]

N_resol = 30

saveFigures = True
########################################################################


def main(options=[], N_resol=100, saveFigures=False):
    for option in options:
        obs = []  # create empty obstacle list
        if option == 0:
            x_lim = [-1.1, 8.1]
            y_lim = [-3.9, 6.3]

            xAttractor = [1, 0]

            obs.append(
                Cuboid(
                    axes_length=[8, 9.6],
                    center_position=[3, 1],
                    orientation=0.0 / 180 * pi,
                    is_boundary=True,
                )
            )

            obs.append(
                Ellipse(
                    axes_length=[1.0, 2],
                    center_position=[5, 2.1],
                    p=[1, 1],
                    orientation=150.0 / 180 * pi,
                    sf=1,
                    is_boundary=False,
                )
            )

            obs.append(
                Ellipse(
                    axes_length=[1.5, 1.5],
                    center_position=[3, -2.1],
                    p=[1, 1],
                    orientation=00.0 / 180 * pi,
                    sf=1,
                    is_boundary=False,
                )
            )

            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="linearSystem_boundaryCuboid",
                noTicks=False,
                draw_vectorField=True,
                automatic_reference_point=False,
                point_grid=N_resol,
                show_streamplot=False,
            )

            # fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=[], xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_initial', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=N_resol)

        if option == 1:
            x_lim = [-1.1, 8.1]
            y_lim = [-3.9, 6.3]

            xAttractor = [1, 0]

            obs.append(
                Cuboid(
                    axes_length=[8, 9.6],
                    center_position=[3, 1],
                    orientation=0.0 / 180 * pi,
                    absolut_margin=0.0,
                    is_boundary=True,
                )
            )

            # fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_boundaryCuboid_twoEllipses_quiver', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=N_resol)
            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="linearSystem_boundaryCuboid_twoEllipses",
                noTicks=False,
                draw_vectorField=True,
                automatic_reference_point=False,
                point_grid=N_resol,
            )

        if option == 2:
            x_lim = [-1.1, 8.1]
            y_lim = [-3.9, 6.1]

            xAttractor = [1, 0]

            obs.append(
                Ellipse(
                    axes_length=[3.5, 4.0],
                    center_position=[4, 2.0],
                    p=[1, 1],
                    orientation=-70.0 / 180 * pi,
                    sf=1,
                    is_boundary=True,
                )
            )

            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="linearSystem_boundaryEllipse",
                noTicks=False,
                draw_vectorField=True,
                automatic_reference_point=False,
                point_grid=N_resol,
            )

        if option == 3:
            x_lim = [-1.1, 8.1]
            y_lim = [-3.9, 6.1]

            xAttractor = [1, 0]

            obs.append(
                Ellipse(
                    axes_length=[3.5, 4.0],
                    center_position=[4, 2.0],
                    p=[1, 1],
                    orientation=-70.0 / 180 * pi,
                    sf=1,
                    is_boundary=True,
                )
            )

            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="linearSystem_boundaryEllipse",
                noTicks=False,
                draw_vectorField=True,
                automatic_reference_point=False,
                point_grid=N_resol,
            )

        if option == 4:
            x_lim = [-1.1, 8.1]
            y_lim = [-3.9, 6.3]

            xAttractor = [1, 0]

            obs.append(
                Cuboid(
                    axes_length=[3, 3],
                    center_position=[3, 1],
                    orientation=0.0 / 180 * pi,
                    absolut_margin=0.0,
                    is_boundary=False,
                )
            )

            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="linearSystem_boundaryCuboid_twoEllipses",
                noTicks=False,
                draw_vectorField=True,
                automatic_reference_point=False,
                point_grid=N_resol,
            )

        if option == 5:
            x_lim = [-3.1, 7.1]
            y_lim = [-3.9, 6.3]

            xAttractor = [-1, 0.2]

            edge_points = np.array([[1, 4, 2], [-1, -0.5, 4]])

            obs.append(
                Polygon(
                    edge_points=edge_points,
                    orientation=0.0 / 180 * pi,
                    absolut_margin=0.0,
                    is_boundary=False,
                )
            )

            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="linearSystem_boundaryCuboid_triangle",
                noTicks=False,
                draw_vectorField=True,
                automatic_reference_point=False,
                point_grid=N_resol,
            )

        if option == 6:
            x_lim = [-4.1, 7.1]
            y_lim = [-3.9, 6.3]

            xAttractor = [-3, 3]

            edge_points = np.array(
                [[1.3, 2.3, 2, 0, -2, -2.3, -1.3, 0], [-2, 0, 4, 1, 4, 0, -2, -2.2]]
            )

            n_points = 5
            points_init = np.vstack(
                (
                    x_lim[1] * np.ones(n_points),
                    np.linspace(y_lim[0], y_lim[1], n_points),
                )
            )

            obs.append(
                Polygon(
                    edge_points=edge_points,
                    orientation=0.0 / 180 * pi,
                    absolut_margin=0.0,
                    is_boundary=False,
                )
            )

            # fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_polygon_concave', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=N_resol, show_streamplot=False, points_init=[])
            # fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_polygon_concave', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=N_resol, show_streamplot=True, points_init=points_init)
            # fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_boundaryCuboid_twoEllipses', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=N_resol, show_streamplot=True)

        if option == 6.5:
            x_lim = [-4.7, 5.1]
            y_lim = [-4.7, 4.4]
            xAttractor = [-3.5, 1.5]

            edge_points = np.array(
                [
                    [1.3, 2.3, 2, 0, -2, -2.3, -1, -2.3, 0],
                    [-2, 0, 2, 0.25, 2, 0, -0.5, -2, -2.2],
                ]
            )

            n_points = 4
            points_init = np.vstack(
                (np.linspace(-1, 2, n_points), np.linspace(-2, 0, n_points))
            )
            # points_init = []

            obs.append(
                Polygon(
                    edge_points=edge_points,
                    orientation=0.0 / 180 * pi,
                    absolut_margin=0.0,
                    is_boundary=False,
                )
            )

            # run_animation(points_init, obs, x_range=x_lim, y_range=y_lim, dt=0.0001, N_simuMax=600, convergenceMargin=0.3, sleepPeriod=0.01)

            # fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_boundaryCuboid_twoEllipses', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=N_resol, show_streamplot=False)
            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="linearSystem_boundaryPolygon_starshape",
                noTicks=False,
                draw_vectorField=True,
                automatic_reference_point=False,
                point_grid=N_resol,
                show_streamplot=False,
                points_init=[],
            )

        if option == 7:
            x_lim = [-2.7, 3.1]
            y_lim = [-2.7, 2.4]

            xAttractor = [-1.5, 1.2]

            edge_points = np.array(
                [
                    [1.3, 2.3, 2, 0, -2, -2.3, -1, -2.3, 0],
                    [-2, 0, 2, 0.25, 2, 0, -0.5, -2, -2.2],
                ]
            )

            n_points = 4
            points_init = np.vstack(
                (np.linspace(-1, 2, n_points), np.linspace(-2, 0, n_points))
            )
            # points_init = []

            obs.append(
                Polygon(
                    edge_points=edge_points,
                    orientation=0.0 / 180 * pi,
                    absolut_margin=0.0,
                    is_boundary=True,
                )
            )

            # run_animation(points_init, obs, x_range=x_lim, y_range=y_lim, dt=0.0001, N_simuMax=600, convergenceMargin=0.3, sleepPeriod=0.01)

            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="linearSystem_boundaryPolygon_starshape",
                noTicks=False,
                draw_vectorField=True,
                automatic_reference_point=False,
                point_grid=N_resol,
                show_streamplot=False,
                points_init=[],
            )
            # fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_boundaryPolygon_starshape', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=N_resol, show_streamplot=True, points_init=points_init)

            # fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_boundaryPolygon_starshape', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=N_resol, show_streamplot=True, points_init=[])

        if option == 8:
            x_lim = [-4.1, 4.1]
            y_lim = [-4.1, 4.1]

            xAttractor = [-3.0, 3.0]

            obs = []
            obs.append(StarshapedFlower(orientation=00.0 / 180 * pi))

            points_init = []

            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="flower_shape_normalObstacle",
                noTicks=False,
                draw_vectorField=True,
                automatic_reference_point=False,
                point_grid=N_resol,
                show_streamplot=True,
                points_init=points_init,
            )
            obs = []
            obs.append(StarshapedFlower(orientation=30.0 / 180 * pi))
            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="flower_shape_normalObstacle_twisted",
                noTicks=False,
                draw_vectorField=True,
                automatic_reference_point=False,
                point_grid=N_resol,
                show_streamplot=True,
                points_init=points_init,
            )

        if option == 9:
            x_lim = [-4.1, 4.1]
            y_lim = [-4.1, 4.1]

            xAttractor = [-3.0, 0.2]

            points_init = []

            obs = []
            obs.append(
                StarshapedFlower(
                    radius_magnitude=1,
                    radius_mean=3,
                    orientation=30.0 / 180 * pi,
                    is_boundary=True,
                    number_of_edges=5,
                )
            )
            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="flower_shape_boundary",
                noTicks=False,
                draw_vectorField=True,
                automatic_reference_point=False,
                point_grid=N_resol,
                show_streamplot=True,
                points_init=points_init,
            )

        if option == 10:
            x_lim = [-4.1, 25.1]
            y_lim = [-4.1, 12.1]

            xAttractor = [-3.0, 0.2]

            points_init = []

            obs = []

            obs.append(
                Cuboid(
                    axes_length=[7, 5.6],
                    center_position=[12, 2],
                    orientation=20.0 / 180 * pi,
                    absolut_margin=0.0,
                    is_boundary=False,
                )
            )

            edge_points = np.array([[1, 4, 2], [-1, -0.5, 4]])
            obs.append(
                Polygon(
                    edge_points=edge_points,
                    orientation=-90.0 / 180 * pi,
                    absolut_margin=0.0,
                    is_boundary=False,
                )
            )
            obs[-1].move_obstacle_to_referencePoint(
                position=np.array([7, 8]), in_global_frame=True
            )

            edge_points = np.array(
                [[1.3, 2.3, 2, 0, -2, -2.3, -1.3, 0], [-2, 0, 4, 1, 4, 0, -2, -2.2]]
            )
            obs.append(
                Polygon(
                    edge_points=edge_points,
                    orientation=0.0 / 180 * pi,
                    absolut_margin=0.0,
                    is_boundary=False,
                )
            )
            obs[-1].move_obstacle_to_referencePoint(
                position=np.array([2, -1]), in_global_frame=True
            )

            edge_points = np.array([[1, 4, 2], [-1, -0.5, 4]])
            obs.append(
                Polygon(
                    edge_points=edge_points,
                    orientation=180.0 / 180 * pi,
                    absolut_margin=0.0,
                    is_boundary=False,
                )
            )
            obs[-1].move_obstacle_to_referencePoint(
                position=np.array([19, 8]), in_global_frame=True
            )
            obs[-1].set_reference_point(np.array([19, 6.5]), in_global_frame=True)

            obs.append(
                Ellipse(
                    axes_length=[2, 1.5],
                    center_position=[19, 5],
                    p=[1, 1],
                    orientation=-70.0 / 180 * pi,
                    sf=1,
                    is_boundary=False,
                )
            )
            obs[-1].set_reference_point(np.array([19, 6.5]), in_global_frame=True)

            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="noonsmooth_several_obstacles",
                noTicks=True,
                draw_vectorField=True,
                automatic_reference_point=False,
                point_grid=N_resol,
                show_streamplot=True,
                points_init=points_init,
                figureSize=(25.0, 10),
                showLabel=False,
            )


if __name__ == ("__main__"):
    if False:
        # if len(sys.argv) > 1 and sys.argv[1] != '-i':
        N_resol = int(sys.argv[1])

        if len(sys.argv) > 2:
            options = [float(sys.argv[2])]

            if len(sys.argv) > 3:
                saveFigures = bool(sys.argv[3])

    main(options=options, N_resol=N_resol, saveFigures=saveFigures)

    # input("\nPress enter to continue...")

# Run function
