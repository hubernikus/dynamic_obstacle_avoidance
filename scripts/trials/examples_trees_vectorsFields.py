# /usr/bin/python3
"""
Script which creates a variety of examples of local modulation of a vector field with obstacle avoidance.

@author LukasHuber
@date 2018-02-15
"""

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.visualization.vectorField_visualization_nonlinear import *  #

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import *

########################################################################
# Chose the option you want to run as a number in the option list (integer from -2 to 10)
options = [-1]

n_resolution = 10
saveFigures = False

########################################################################


def main(options=[0], n_resolution=100, saveFigures=False):
    # print('options', options)

    for option in options:
        obs = []  # create empty obstacle list

        if option == -2:
            xlim = [-1.0, 7]
            ylim = [-5, 5]

            obs.append(
                Ellipse(
                    axes_length=[0.2, 0.2],
                    center_position=[0.5, 0.1],
                    orientation=0 / 180 * pi,
                )
            )

            obs.append(
                Ellipse(
                    axes_length=[2.0, 2.0],
                    center_position=[3.0, -0.1],
                    orientation=0 / 180 * pi,
                )
            )

            xAttractor = [0, 0]

            # obs[0].center_dyn = x0

            dt = 0.01
            x = [1, 1]

            obs_avoidance_rungeKutta(dt, x, obs)

            VectorFields_nonlinear(
                xlim,
                ylim,
                n_resolution,
                obs,
                xAttractor=xAttractor,
                saveFigure=False,
                figName="ellipse_centerMiddle",
                noTicks=True,
            )

        if option == -1:
            # Two ellipses placed at x1=0 with dynamic center diplaced and center line in gray
            a = [0.4, 1]
            p = [1, 1]
            x0 = [1.5, 0]
            th_r = 0 / 180 * pi
            sf = 1

            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

            xlim = [-0.5, 4]
            ylim = [-2, 2]

            xAttractor = [0, 0]

            # obs[0].center_dyn = x0

            dt = 0.01
            x = [1, 1]

            obs_avoidance_rungeKutta(dt, x, obs)

            VectorFields_nonlinear(
                xlim,
                ylim,
                n_resolution,
                obs,
                xAttractor=xAttractor,
                saveFigure=False,
                figName="ellipse_centerMiddle",
                noTicks=True,
            )

        # elif option==-1:
        #     theta = 0*pi/180
        #     n = 0

        #     pos = np.zeros((2))
        #     pos[0]= obs[n].a[0]*np.cos(theta)
        #     pos[1] = np.copysign(obs[n].a[1], theta)*(1 - np.cos(theta)**(2*obs[n].p[0]))**(1./(2.*obs[n].p[1]))
        #     pos = obs[n].rotMatrix @ pos + obs[n].x0

        #     xd = obs_avoidance_nonlinear_radial(pos, nonlinear_stable_DS, obs, attractor=xAttractor)

        if option == 0:
            xlim = [-0.8, 7]
            ylim = [-3.3, 3.3]

            xAttractor = [0, 0]

            obs = []
            # Ellipse 2
            a = [0.4, 2.2]
            p = [1, 1]
            x0 = [6, 0]
            th_r = 0 / 180 * pi
            sf = 1
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))
            # obs[n].center_dyn = np.array([2,1.4])

            VectorFields_nonlinear(
                xlim,
                ylim,
                n_resolution,
                obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="nonlinearSystem",
                noTicks=True,
                obs_avoidance_func=obs_avoidance_nonlinear_hirarchy,
                dynamicalSystem=nonlinear_stable_DS,
                nonlinear=True,
            )

        if option == 1:

            xlim = [-0.8, 5]
            ylim = [-2.5, 2.5]

            xAttractor = [0, 0]

            N_it = 4
            for ii in range(N_it):
                obs = []
                a = [0.5, 2]
                p = [1, 1]
                x0 = [2.2, 0.1]
                th_r = 30 / 180 * pi
                sf = 1

                # if ii>0:
                if True:
                    # Ellipse 2
                    a = [a[jj] * (ii + 0.01) / (N_it - 1) for jj in range(len(a))]
                    obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

                VectorFields_nonlinear(
                    xlim,
                    ylim,
                    n_resolution,
                    obs,
                    xAttractor=xAttractor,
                    saveFigure=saveFigures,
                    figName="nonlinearGrowing" + str(ii),
                    noTicks=True,
                    obs_avoidance_func=obs_avoidance_nonlinear_hirarchy,
                    dynamicalSystem=nonlinear_wavy_DS,
                    nonlinear=True,
                )

        if option == 2:
            xlim = [-1.2, 11.5]
            ylim = [-6, 6]

            xAttractor = [0, 0]

            obs = []
            # Ellipse 2
            a = [0.9, 5.0]
            p = [1, 1]
            x0 = [4, 0]
            th_r = -30 / 180 * pi
            sf = 1
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))
            # obs[n].center_dyn = np.array([2,1.4])

            # VectorFields_nonlinear(xlim, ylim, n_resolution, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_modulation', noTicks=True, obs_avoidance_func=obs_avoidance_interpolation_moving, dynamicalSystem=nonlinear_stable_DS, nonlinear=False)

            VectorFields_nonlinear(
                xlim,
                ylim,
                n_resolution,
                obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="nonlinear_displacement",
                noTicks=True,
                obs_avoidance_func=obs_avoidance_nonlinear_hirarchy,
                dynamicalSystem=nonlinear_stable_DS,
                nonlinear=True,
            )

            # obs = []
            # VectorFields_nonlinear(xlim, ylim, n_resolution, obs=[], xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_noObs', noTicks=True, obs_avoidance_func=obs_avoidance_interpolation_moving, dynamicalSystem=nonlinear_stable_DS, nonlinear=False)

        if option == 3:
            xlim = [0.3, 13]
            ylim = [-6, 6]

            xAttractor = [0, 0]

            obs = []

            a = [1.0, 1.0]
            p = [1, 1]
            x0 = [5.5, 0]
            th_r = 20 / 180 * pi
            sf = 1
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

            a = [2.0, 0.8]
            p = [1, 1]
            x0 = [2, 1]
            th_r = 50 / 180 * pi
            sf = 1
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

            a = [2.5, 1.5]
            p = [1, 1]
            x0 = [8, 4]
            th_r = 30 / 180 * pi
            sf = 1
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

            a = [0.4, 2.2]
            p = [1, 1]
            x0 = [10, -3]
            th_r = 80 / 180 * pi
            sf = 1
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

            a = [0.9, 1.1]
            p = [1, 1]
            x0 = [3, -4]
            th_r = 80 / 180 * pi
            sf = 1
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

            VectorFields_nonlinear(
                xlim,
                ylim,
                n_resolution,
                obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="nonlinear_multipleEllipses",
                noTicks=False,
                obs_avoidance_func=obs_avoidance_nonlinear_hirarchy,
                dynamicalSystem=nonlinear_stable_DS,
                nonlinear=True,
            )

            # obs = []
            # VectorFields_nonlinear(xlim, ylim, n_resolution, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_multipleEllipses_initial', noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_hirarchy, dynamicalSystem=nonlinear_stable_DS, nonlinear=True)

        if option == 4:
            xlim = [-0.7, 12]
            ylim = [-6, 6]

            # xlim = [3,8]
            # ylim = [-1,4]
            xAttractor = [0, 0]

            obs = []

            a = [0.80, 3.0]
            p = [1, 1]
            x0 = [5.5, -1]
            th_r = 40 / 180 * pi
            sf = 1
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

            a = [1.0, 3.0]
            p = [1, 1]
            x0 = [5.0, 2]
            th_r = -50 / 180 * pi
            sf = 1
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

            # VectorFields_nonlinear(xlim, ylim, n_resolution, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_intersectingEllipses', noTicks=False, obs_avoidance_func=obs_avoidance_nonlinear_hirarchy, dynamicalSystem=nonlinear_stable_DS, nonlinear=True)

            VectorFields_nonlinear(
                xlim,
                ylim,
                n_resolution,
                obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="nonlinear_intersectingEllipses",
                noTicks=False,
                obs_avoidance_func=obs_avoidance_nonlinear_hirarchy,
                dynamicalSystem=linearAttractor_const,
                nonlinear=True,
            )

            # obs = []
            # VectorFields_nonlinear(xlim, ylim, n_resolution, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_intersectingEllipses_initial', noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_hirarchy, dynamicalSystem=nonlinear_stable_DS, nonlinear=True)

            # pltLines(obs[0].x0, np.array([0,0]) )

        if option == 5:
            xlim = [-3, 10]
            ylim = [-6, 6]

            xAttractor = [0, 0]

            obs = []
            # Ellipse 2
            a = [2.0, 3.5]
            p = [1, 1]
            x0 = [5.3, -0.4]
            th_r = -30 / 180 * pi
            sf = 1
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

            VectorFields_nonlinear(
                xlim,
                ylim,
                n_resolution,
                obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="nonlinear_convergence",
                noTicks=True,
                obs_avoidance_func=obs_avoidance_nonlinear_hirarchy,
                dynamicalSystem=nonlinear_wavy_DS,
                nonlinear=True,
            )
        if option == 6:
            xlim = [-1.0, 11]
            ylim = [-5.5, 5.5]

            xAttractor = [0, 0]

            obs = []

            a = [2.0, 2.0]
            p = [1, 1]
            x0 = [4.5, 1]
            th_r = 20 / 180 * pi
            sf = 1
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

            axes = [3, 0.8]
            roundness = [1, 1]
            center = [7, 3]
            orientation = 30 / 180 * pi
            margin = 1
            obs.append(
                Ellipse(a=axes, p=roundness, x0=center, th_r=orientation, sf=margin)
            )

            a = [0.4, 1.8]
            p = [1, 1]
            x0 = [4, -3.5]
            th_r = 80 / 180 * pi
            sf = 1
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

            a = [0.8, 2.2]
            p = [1, 1]
            x0 = [4, -2]
            th_r = -40 / 180 * pi
            sf = 1
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

            a = [0.4, 2.2]
            p = [1, 1]
            x0 = [10, 0]
            th_r = 80 / 180 * pi
            sf = 1
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

            a = [0.4, 2.2]
            p = [1, 1]
            x0 = [10, -2]
            th_r = 20 / 180 * pi
            sf = 1
            hirarchy = 0
            obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf, hirarchy=hirarchy))

            VectorFields_nonlinear(
                xlim,
                ylim,
                n_resolution,
                obs,
                xAttractor=xAttractor,
                saveFigure=saveFigures,
                figName="linear_convergence",
                noTicks=True,
                obs_avoidance_func=obs_avoidance_nonlinear_hirarchy,
                dynamicalSystem=linearAttractor,
                nonlinear=True,
                displacement_visualisation=False,
            )
            # Simulation_vectorFields(xlim, ylim, n_resolution, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linear_convergence', noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_hirarchy, dynamicalSystem=linearAttractor, nonlinear=True)

    if option == 7:
        # xRange = [-5.2,11.5]
        # yRange = [-8,8]

        xRange = [0.45, 3.88]
        yRange = [0.86, 3.22]

        print("new range")

        # N=20
        # x_init = samplePointsAtBorder(N, xRange, yRange)

        N = 10
        x_init = np.vstack(
            (xRange[1] * np.ones(N), np.linspace(yRange[0], yRange[1], N))
        )

        xAttractor = [0, 0]

        obs = []
        # Ellipse 2
        a = [0.9, 5]
        p = [1, 1]
        x0 = [4, 2]
        th_r = -30 / 180 * pi
        sf = 1
        obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

        a = [0.9, 5.0]
        p = [1, 1]
        x0 = [4, -2.0]
        th_r = 30 / 180 * pi
        sf = 1
        obs.append(Ellipse(a=a, p=p, x0=x0, th_r=th_r, sf=sf))

        # VectorFields_nonlinear(xRange, yRange, n_resolution, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='nonlinear_convergence', noTicks=True, obs_avoidance_func=obs_avoidance_nonlinear_hirarchy, dynamicalSystem=nonlinear_wavy_DS, nonlinear=True)
        VectorFields_nonlinear(
            xRange,
            yRange,
            n_resolution,
            obs,
            xAttractor=xAttractor,
            saveFigure=saveFigures,
            figName="nonlinear_",
            noTicks=True,
            obs_avoidance_func=obs_avoidance_nonlinear_hirarchy,
            dynamicalSystem=linearAttractor,
            nonlinear=True,
        )

    if option == 8:
        xlim = [-1.0, 11]
        ylim = [-5.5, 5.5]

        xAttractor = [0, 0]

        obs = []

        obs.append(Ellipse(a=[2.6, 1.5], p=[2, 2], x0=[7, 0], th_r=0.0 / 180 * pi))

        obs.append(Ellipse(a=[3, 0.8], p=[2, 2], x0=[5, 2.8], th_r=0 / 180 * pi))
        obs.append(Ellipse(a=[3, 0.8], p=[2, 2], x0=[2.7, 0], th_r=90 / 180 * pi))
        obs.append(Ellipse(a=[3, 0.8], p=[2, 2], x0=[5, -2.8], th_r=180 / 180 * pi))

        VectorFields_nonlinear(
            xlim,
            ylim,
            n_resolution,
            obs,
            xAttractor=xAttractor,
            saveFigure=saveFigures,
            figName="linear_convergence",
            noTicks=True,
            obs_avoidance_func=obs_avoidance_nonlinear_hirarchy,
            dynamicalSystem=linearAttractor,
            nonlinear=True,
            displacement_visualisation=False,
        )

    if option == 9:
        xlim = [-4.0, 11]
        ylim = [-5.5, 5.5]

        xAttractor = [-3, -1]

        obs = []

        r_1 = 0.1
        r_2 = 1.0
        delta_d = 0.1
        d = r_1 + r_2 + delta_d

        obs.append(Ellipse(a=[r_1, r_1], p=[1, 1], x0=[0, 0], th_r=0 / 180 * pi))
        obs.append(Ellipse(a=[r_2, r_2], p=[1, 1], x0=[d, 0], th_r=0.0 / 180 * pi))

        VectorFields_nonlinear(
            xlim,
            ylim,
            n_resolution,
            obs,
            xAttractor=xAttractor,
            saveFigure=saveFigures,
            figName="linear_convergence",
            noTicks=True,
            obs_avoidance_func=obs_avoidance_nonlinear_hirarchy,
            dynamicalSystem=linearAttractor,
            nonlinear=True,
            displacement_visualisation=False,
        )


# main(options=options, n_resolution=n_resolution, saveFigures=saveFigures)

if __name__ == ("__main__"):
    if len(sys.argv) > 1:
        n_resolution = int(sys.argv[1])

    if len(sys.argv) > 2:
        options = [float(sys.argv[2])]

    if len(sys.argv) > 3:
        saveFigures = bool(sys.argv[3])

    main(options=options, n_resolution=n_resolution, saveFigures=saveFigures)

    # input("\nPress enter to continue...")


# Add for running in command line mode
# input("\nPress enter to continue...")
