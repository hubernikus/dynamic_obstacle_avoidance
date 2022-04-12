"""
Obstacle Avoidance Algorithm script with vecotr field

@author LukasHuber
@date 2018-02-15
"""

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import time

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system import *
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import *
from dynamic_obstacle_avoidance.obstacle_avoidance.nonlinear_modulation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import *

# from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import obs_common_section_hirarchy
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_dynamic_center_3d import *


def pltLines(pos0, pos1, xlim=[-100, 100], ylim=[-100, 100]):
    if pos1[0] - pos0[0]:  # m < infty
        m = (pos1[1] - pos0[1]) / (pos1[0] - pos0[0])

        ylim = [0, 0]
        ylim[0] = pos0[1] + m * (xlim[0] - pos0[0])
        ylim[1] = pos0[1] + m * (xlim[1] - pos0[0])
    else:
        xlim = [pos1[0], pos1[0]]

    plt.plot(xlim, ylim, "--", color=[0.3, 0.3, 0.3], linewidth=2)


def plot_streamlines(
    points_init,
    ax,
    obs=[],
    attractorPos=[0, 0],
    dim=2,
    dt=0.01,
    max_simu_step=300,
    convergence_margin=0.03,
):

    n_points = np.array(points_init).shape[1]

    x_pos = np.zeros((dim, max_simu_step + 1, n_points))
    x_pos[:, 0, :] = points_init

    it_count = 0
    for iSim in range(max_simu_step):
        for j in range(n_points):
            x_pos[:, iSim + 1, j] = obs_avoidance_rk4(
                dt,
                x_pos[:, iSim, j],
                obs,
                x0=attractorPos,
                obs_avoidance=obs_avoidance_interpolation_moving,
            )

        # Check convergence
        if (
            np.sum(
                (x_pos[:, iSim + 1, :] - np.tile(attractorPos, (n_points, 1)).T) ** 2
            )
            < convergence_margin
        ):
            x_pos = x_pos[:, : iSim + 2, :]

            print("Convergence reached after {} iterations.".format(it_count))
            break

        it_count += 1
    for j in range(n_points):
        ax.plot(x_pos[0, :, j], x_pos[1, :, j], "--", lineWidth=4)
        ax.plot(
            x_pos[0, 0, j],
            x_pos[1, 0, j],
            "k*",
            markeredgewidth=4,
            markersize=13,
        )

    # return x_pos


def VectorFields_nonlinear(
    x_range=[0, 10],
    y_range=[0, 10],
    point_grid=10,
    obs=[],
    sysDyn_init=False,
    xAttractor=np.array(([0, 0])),
    saveFigure=False,
    figName="default",
    noTicks=True,
    showLabel=True,
    figureSize=(7.0, 6),
    obs_avoidance_func=obs_avoidance_nonlinear_hirarchy,
    attractingRegion=False,
    drawVelArrow=False,
    colorCode=False,
    streamColor=[0.05, 0.05, 0.7],
    obstacleColor=[],
    plotObstacle=True,
    plotStream=True,
    figHandle=[],
    alphaVal=1,
    dynamicalSystem=linearAttractor,
    nonlinear=True,
    hirarchy=True,
    displacement_visualisation=True,
):
    # Numerical hull of ellipsoid
    for n in range(len(obs)):
        obs[n].draw_obstacle(numPoints=50)  # 50 points resolution

    # Adjust dynamic center
    if nonlinear:
        intersection_obs = obs_common_section_hirarchy(obs)
    else:
        intersection_obs = obs_common_section(obs)

    # Create meshrgrid of points
    if type(point_grid) == int:
        N_x = N_y = point_grid
        YY, XX = np.mgrid[
            y_range[0] : y_range[1] : N_y * 1j,
            x_range[0] : x_range[1] : N_x * 1j,
        ]

    else:
        N_x = N_y = 1
        XX, YY = np.array([[point_grid[0]]]), np.array([[point_grid[1]]])

    if attractingRegion:  # Forced to attracting Region

        def obs_avoidance_temp(x, xd, obs):
            # return obs_avoidance_func(x, xd, obs, attractor=xAttractor)
            return obs_avoidance_func(x, xd, obs, xAttractor)

        obs_avoidance = obs_avoidance_temp
    else:
        obs_avoidance = obs_avoidance_func

    # Initialize array
    xd_init = np.zeros((2, N_x, N_y))
    max_hirarchy = 0

    if hirarchy:
        hirarchy_array = np.zeros(len(obs))
        for oo in range(len(obs)):
            hirarchy_array[oo] = obs[oo].hirarchy
            if obs[oo].hirarchy > max_hirarchy:
                max_hirarchy = obs[oo].hirarchy

        xd_mod = np.zeros((2, N_x, N_y, max_hirarchy + 1))
        m_x = np.zeros((2, N_x, N_y, max_hirarchy + 2))
    else:
        xd_mod = np.zeros((2, N_x, N_y))

    if True:  # DEBUGGING PARAGRAPH
        # N_x = N_y = 1
        # XX = np.zeros((N_x, N_y))
        # YY = np.zeros((N_x, N_y))

        it_start = 0
        n_samples = 5

        pos1 = [0.655, 0.2633]
        pos2 = [1.013, 0.2333]

        x_sample_range = [pos1[0], pos2[0]]
        y_sample_range = [pos1[1], pos2[1]]

        x_sample = np.linspace(x_sample_range[0], x_sample_range[1], n_samples)
        y_sample = np.linspace(y_sample_range[0], y_sample_range[1], n_samples)

        ii = 0
        for ii in range(n_samples):
            ix = (ii + it_start) % N_x
            iy = int((ii + it_start) / N_y)
            XX[ix, iy] = x_sample[ii]
            YY[ix, iy] = y_sample[ii]

    if nonlinear:
        if hirarchy:
            for ix in range(N_x):
                for iy in range(N_y):

                    (
                        xd_mod[:, ix, iy, :],
                        m_x[:, ix, iy, :],
                    ) = obs_avoidance_nonlinear_hirarchy(
                        np.array([XX[ix, iy], YY[ix, iy]]),
                        dynamicalSystem,
                        obs,
                        attractor=xAttractor,
                    )
        else:  # nonlinear, no hirarchy
            for ix in range(N_x):
                for iy in range(N_y):
                    xd_mod[:, ix, iy] = obs_avoidance_func(
                        np.array([XX[ix, iy], YY[ix, iy]]),
                        dynamicalSystem,
                        obs,
                        attractor=xAttractor,
                    )

    else:  # linear
        for ix in range(N_x):
            for iy in range(N_y):
                pos = np.array([XX[ix, iy], YY[ix, iy]])
                xd_init[:, ix, iy] = dynamicalSystem(pos, x0=xAttractor)  # initial DS

                xd_mod[:, ix, iy] = obs_avoidance(
                    pos, xd_init[:, ix, iy], obs
                )  # modulataed DS with IFD

    if sysDyn_init:
        fig_init, ax_init = plt.subplots(figsize=(5, 2.5))
        res_init = ax_init.streamplot(
            XX, YY, xd_init[0, :, :], xd_init[1, :, :], color=[(0.3, 0.3, 0.3)]
        )

        ax_init.plot(xAttractor[0], xAttractor[1], "k*")
        plt.gca().set_aspect("equal", adjustable="box")

        plt.xlim(x_range)
        plt.ylim(y_range)

        if saveFigure:
            plt.savefig("fig/" + figName + "initial" + ".eps", bbox_inches="tight")

    if hirarchy:
        dx_noColl = np.zeros((2, N_x, N_y, max_hirarchy + 1))
    else:
        collisions = obs_check_collision_2d(obs, XX, YY)

        dx1_noColl = np.squeeze(xd_mod[0, :, :]) * collisions
        dx2_noColl = np.squeeze(xd_mod[1, :, :]) * collisions

    ind_hirarchy = np.zeros(max_hirarchy + 1, dtype=bool)
    if displacement_visualisation:
        # if True:
        ind_hirarchy = np.ones(max_hirarchy + 1, dtype=bool)
    else:
        ind_hirarchy[-1] = True

    for hh in np.arange(max_hirarchy + 1)[ind_hirarchy]:
        if len(figHandle):
            fig_ifd, ax_ifd = figHandle[0], figHandle[1]
        elif hirarchy:
            XX = m_x[0, :, :, hh + 1]
            YY = m_x[1, :, :, hh + 1]
            # collisions = obs_check_collision_2d([obs[jj] for jj in np.arange(len(obs))[hirarchy_array<=hh]], XX, YY)

            collisions = np.ones(np.squeeze(xd_mod[0, :, :, hh]).shape)
            # TODO EXPLORE- why are collisions not plotted

            dx_noColl[0, :, :, hh] = np.squeeze(xd_mod[0, :, :, hh]) * collisions
            dx_noColl[1, :, :, hh] = np.squeeze(xd_mod[1, :, :, hh]) * collisions

            dx1_noColl = dx_noColl[0, :, :, hh]
            dx2_noColl = dx_noColl[1, :, :, hh]

            fig_ifd, ax_ifd = plt.subplots(figsize=figureSize)
            fig_ifd.canvas.set_window_title(
                "Figure - Modulation at level {}".format(hh)
            )

        else:
            fig_ifd, ax_ifd = plt.subplots(figsize=figureSize)

        if plotStream:
            if colorCode:
                velMag = (
                    np.linalg.norm(np.dstack((dx1_noColl, dx2_noColl)), axis=2)
                    / 6
                    * 100
                )

                strm = res_ifd = ax_ifd.streamplot(
                    XX,
                    YY,
                    dx1_noColl,
                    dx2_noColl,
                    color=velMag,
                    cmaph="winter",
                    norm=matplotlib.colors.Normalize(vmin=0, vmax=10.0),
                )
                # strm = res_ifd = ax_ifd.quiver(XX, YY,dx1_noColl, dx2_noColl, color=velMag, cmap='winter', norm=matplotlib.colors.Normalize(vmin=0, vmax=10.) )
            else:
                # Normalize
                normVel = np.sqrt(dx1_noColl**2 + dx2_noColl**2)
                nonZeroInd = normVel.astype(bool)

                # dx1_noColl, dx2_noColl = dx1_noColl/normVel, dx2_noColl/normVel
                dx1_noColl[nonZeroInd] = dx1_noColl[nonZeroInd] / normVel[nonZeroInd]
                dx2_noColl[nonZeroInd] = dx2_noColl[nonZeroInd] / normVel[nonZeroInd]

                # gi res_ifd = ax_ifd.streamplot(XX, YY,dx1_noColl, dx2_noColl, color=streamColor)
                res_ifd = ax_ifd.quiver(
                    XX, YY, dx1_noColl, dx2_noColl, color=streamColor, zorder=0
                )

                ax_ifd.plot(
                    xAttractor[0],
                    xAttractor[1],
                    "k*",
                    linewidth=18.0,
                    markersize=18,
                )

        plt.gca().set_aspect("equal", adjustable="box")

        ax_ifd.set_xlim(x_range)
        ax_ifd.set_ylim(y_range)

        if noTicks:
            plt.tick_params(
                axis="both",
                which="major",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )

        if showLabel:
            plt.xlabel(r"$\xi_1$", fontsize=16)
            plt.ylabel(r"$\xi_2$", fontsize=16)

        plt.tick_params(axis="both", which="major", labelsize=14)
        plt.tick_params(axis="both", which="minor", labelsize=12)

        # Draw obstacles
        if plotObstacle:
            obs_polygon = []
            obs_polygon_sf = []

            for n in range(len(obs)):
                print("obstacle {} of {}".format(n, len(obs)))

                if obs[n].hirarchy > hh:  # don't include higher hirarchies
                    obs_polygon.append(0)
                    continue
                x_obs_sf = obs[n].x_obs_sf  # todo include in obs_draw_ellipsoid

                plt.plot(
                    [x_obs_sf[i][0] for i in range(len(x_obs_sf))],
                    [x_obs_sf[i][1] for i in range(len(x_obs_sf))],
                    "k--",
                )

                obs_polygon.append(plt.Polygon(obs[n].x_obs, alpha=0.8, zorder=1))

                if obs[n].is_boundary:
                    boundary_polygon = plt.Polygon(
                        np.vstack(
                            (
                                np.array(
                                    [
                                        [
                                            x_range[0],
                                            x_range[1],
                                            x_range[1],
                                            x_range[0],
                                        ],
                                        [
                                            y_range[0],
                                            y_range[0],
                                            y_range[1],
                                            y_range[1],
                                        ],
                                    ]
                                ).T
                            )
                        ),
                        alpha=0.5,
                        zorder=-2,
                    )
                    boundary_polygon.set_color(np.array([176, 124, 124]) / 255)

                    obs_polygon[n].set_color("white")
                    obs_polygon[n].set_alpha(1)
                    obs_polygon[n].set_zorder(-1)
                else:
                    if len(obstacleColor) == len(obs):
                        obs_polygon[n].set_color(obstacleColor[n])
                    else:
                        obs_polygon[n].set_color(np.array([176, 124, 124]) / 255.0)
                    # obs_polygon[n].set_alpha(0.5)

                obs_polygon_sf.append(
                    plt.Polygon(obs[n].x_obs_sf, zorder=-2, alpha=0.2)
                )
                obs_polygon_sf[-1].set_color([1, 1, 1])

                plt.gca().add_patch(obs_polygon[-1])
                plt.gca().add_patch(obs_polygon_sf[-1])

                if obs[n].is_boundary:
                    plt.gca().add_patch(boundary_polygon)

                ax_ifd.plot(obs[n].center_position[0], obs[n].center_position[1], "k.")
                if hasattr(obs[n], "reference_point"):  # automatic adaptation of center
                    reference_point = obs[n].get_reference_point(in_global_frame=True)
                    ax_ifd.plot(
                        reference_point[0],
                        reference_point[1],
                        "k+",
                        linewidth=18,
                        markeredgewidth=4,
                        markersize=13,
                    )
                    # ax_ifd.annotate('{}'.format(obs[n].hirarchy), xy=np.array(obs[n].reference_point)+0.08, textcoords='data', size=16, weight="bold")  #
                    ax_ifd.annotate(
                        "{}".format(obs[n].hirarchy),
                        xy=reference_point + 0.08,
                        textcoords="data",
                        size=16,
                        weight="bold",
                    )  #

                if drawVelArrow and np.linalg.norm(obs[n].xd) > 0:
                    col = [0.5, 0, 0.9]
                    fac = 5  # scaling factor of velocity
                    ax_ifd.arrow(
                        obs[n].center_position[0],
                        obs[n].center_position[1],
                        obs[n].xd[0] / fac,
                        obs[n].xd[1] / fac,
                        head_width=0.3,
                        head_length=0.3,
                        linewidth=10,
                        fc=col,
                        ec=col,
                        alpha=1,
                    )

    # plt.figure()

    if displacement_visualisation:
        for ix in range(N_x):
            for iy in range(N_y):
                plt.plot(m_x[0, ix, iy, :], m_x[1, ix, iy, :], "r")
                plt.plot(m_x[0, ix, iy, -1], m_x[1, ix, iy, -1], "bo")
                plt.plot(m_x[0, ix, iy, 0], m_x[1, ix, iy, 0], "go")
                plt.plot(m_x[0, ix, iy, 1:-1], m_x[1, ix, iy, 1:-1], "k.")

    plt.ion()
    plt.show()

    if saveFigure:
        plt.savefig("fig/" + figName + ".eps", bbox_inches="tight")

    return fig_ifd, ax_ifd
