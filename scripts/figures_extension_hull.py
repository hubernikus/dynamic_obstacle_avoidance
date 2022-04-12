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
options = [1]

N_resol = 100

saveFigures = True
########################################################################


def main(options=[], N_resol=100, saveFigures=False):
    for option in options:
        obs = []  # create empty obstacle list
        if option == 0:
            x_lim = [-2.1, 3.1]
            y_lim = [-2.1, 2.1]

            xAttractor = [3, 0]

            obs.append(
                Ellipse(
                    axes_length=[0.8, 1.3],
                    center_position=[0, 0],
                    p=[1, 1],
                    orientation=150.0 / 180 * pi,
                    # orientation=0./180*pi,
                    is_boundary=False,
                )
            )

            obs[0].set_reference_point(np.array([0.5, 0.2]), in_global_frame=True)
            ref = obs[0].get_reference_point(in_global_frame=True)
            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="dynamic_extension_x{}_y{}".format(int(ref[0]), int(ref[1])),
                noTicks=True,
                showLabel=False,
                figureSize=(6, 5),
                reference_point_number=False,
                draw_vectorField=True,
                automatic_reference_point=False,
                point_grid=N_resol,
            )

            pos = obs[0].set_reference_point(np.array([1, 0.2]), in_global_frame=True)
            ref = obs[0].get_reference_point(in_global_frame=True)
            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="dynamic_extension_x{}_y{}".format(int(ref[0]), int(ref[1])),
                noTicks=True,
                showLabel=False,
                figureSize=(6, 5),
                reference_point_number=False,
                draw_vectorField=True,
                automatic_reference_point=False,
                point_grid=N_resol,
            )

            pos = obs[0].set_reference_point(np.array([2.5, 0.2]), in_global_frame=True)
            ref = obs[0].get_reference_point(in_global_frame=True)
            fig_mod, ax_mod = Simulation_vectorFields(
                x_lim,
                y_lim,
                obs=obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="dynamic_extension_x{}_y{}".format(int(ref[0]), int(ref[1])),
                noTicks=True,
                showLabel=False,
                figureSize=(6, 5),
                reference_point_number=False,
                automatic_reference_point=False,
                point_grid=N_resol,
                show_streamplot=True,
            )

    if option == 1:
        x_lim = [-2.1, 3.1]
        y_lim = [-1.1, 3.1]

        xAttractor = [3, 1.5]

        obs.append(
            Cuboid(
                axes_length=[1.6, 0.3],
                center_position=[0, 0],
                orientation=0.0 / 180 * pi,
                # orientation=0./180*pi,
                margin_absolut=0.3,
                is_boundary=False,
            )
        )

        obs[0].set_reference_point(np.array([0.5, 0.2]), in_global_frame=True)
        ref = obs[0].get_reference_point(in_global_frame=True)
        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            xAttractor=xAttractor,
            saveFigure=saveFigures,
            figName="dynamic_extension_cuboid_x{}_y{}".format(int(ref[0]), int(ref[1])),
            noTicks=True,
            showLabel=False,
            figureSize=(6, 5),
            reference_point_number=False,
            draw_vectorField=True,
            automatic_reference_point=False,
            point_grid=N_resol,
        )

        pos = obs[0].set_reference_point(np.array([0.6, 1.0]), in_global_frame=True)
        ref = obs[0].get_reference_point(in_global_frame=True)
        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            xAttractor=xAttractor,
            saveFigure=saveFigures,
            figName="dynamic_extension_cuboid_x{}_y{}".format(int(ref[0]), int(ref[1])),
            noTicks=True,
            showLabel=False,
            figureSize=(6, 5),
            reference_point_number=False,
            draw_vectorField=True,
            automatic_reference_point=False,
            point_grid=N_resol,
        )

        pos = obs[0].set_reference_point(np.array([2.5, 1.8]), in_global_frame=True)
        ref = obs[0].get_reference_point(in_global_frame=True)
        fig_mod, ax_mod = Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            xAttractor=xAttractor,
            saveFigure=saveFigures,
            figName="dynamic_extension_cuboid_x{}_y{}".format(int(ref[0]), int(ref[1])),
            noTicks=True,
            showLabel=False,
            figureSize=(6, 5),
            reference_point_number=False,
            automatic_reference_point=False,
            point_grid=N_resol,
            show_streamplot=True,
        )

        # fig_mod, ax_mod = Simulation_vectorFields(x_lim, y_lim,  obs=[], xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_initial', noTicks=False, draw_vectorField=True,  automatic_reference_point=False, point_grid=N_resol)


if __name__ == ("__main__"):
    if len(sys.argv) > 1 and sys.argv[1] != "-i":
        N_resol = int(sys.argv[1])
        if len(sys.argv) > 2:
            options = [float(sys.argv[2])]
            if len(sys.argv) > 3:
                saveFigures = bool(sys.argv[3])

    main(options=options, N_resol=N_resol, saveFigures=saveFigures)

    # input("\nPress enter to continue...")

# Run function
