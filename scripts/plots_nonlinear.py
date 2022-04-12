#!/USSR/bin/python3

"""
Script to show lab environment on computer

"""
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

plt.close("all")
plt.ion()

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import *

__author__ = "LukasHuber"
__date__ = "2020-01-15"
__email__ = "lukas.huber@epfl.ch"


def main(n_resol=90, *args, **kwargs):
    # x_lim = [0.9, 3.1]
    # y_lim = [0.9, 3.1]

    x_lim = [-4.4, 4.4]
    y_lim = [-4.1, 4.1]

    x_lim = [-5.4, 5.4]
    y_lim = [-5.1, 5.1]

    pos_attractor = [-1.5, 1.0]

    obs = ObstacleContainer()

    edge_points = np.array(
        (
            [4.0, 1.0, 1.0, 0.0, 0.0, -4.0, -4.0, -2.5, -2.5, 0.0, 0.0, 4.0],
            [0.0, 0.0, 1.0, 1.0, 1.5, 1.5, -2.0, -2.0, -3.6, -3.6, -3.0, -3.0],
        )
    )

    # edge_points = np.array([[ 4, 4, 3, 1, -1, -3, -4, -4],
    # [-4, 4, 4, 2, 2, 4,  4, -4]])

    case_list = {"lab": 0, "one_square": 1, "one_ellipse": 2}

    # case = "lab"
    # case = "one_square"
    case = "one_ellipse"

    cases = [3]

    if 3 in cases:
        obs.append(
            Ellipse(
                center_position=[0.0, 0.0],
                axes_length=[2.8, 4.2],
                # margin_absolut=1.0,
                margin_absolut=0.0,
                orientation=40 * pi / 180,
            )
        )

        obs.append(
            Ellipse(
                center_position=[0.0, 0.0],
                axes_length=[2.8, 4.2],
                # margin_absolut=1.0,
                margin_absolut=0.0,
                orientation=40 * pi / 180,
                is_boundary=True,
            )
        )

    pos = np.array([-0.150, -2.7])
    normal0 = obs[0].get_normal_direction(pos, in_global_frame=True)
    gamma0 = obs[0].get_gamma(pos, in_global_frame=True)

    n_resolution = 100
    x_grid = np.linspace(x_lim[0], x_lim[1], n_resolution)
    y_grid = np.linspace(y_lim[0], y_lim[1], n_resolution)

    n_obs = len(obs)
    Gamma_vals = np.zeros((n_resolution, n_resolution, n_obs))
    normals = np.zeros((obs[0].dim, n_resolution, n_resolution, n_obs))
    positions = np.zeros((obs[0].dim, n_resolution, n_resolution))

    for it_obs in range(n_obs):
        for ix in range(n_resolution):
            for iy in range(n_resolution):
                pos = np.array([x_grid[ix], y_grid[iy]])

                positions[:, ix, iy] = pos

                Gamma_vals[ix, iy, it_obs] = obs[it_obs].get_gamma(
                    pos, in_global_frame=True
                )
                # normals[:, ix, iy, it_obs] = obs[it_obs].get_normal_direction(pos, in_global_frame=True)

    # Gamma_vals[22, 3] = 100
    Gamma_vals = np.flip(Gamma_vals, axis=1)
    Gamma_vals = np.swapaxes(Gamma_vals, 0, 1)

    obs_polygon_sf = []
    obs_polygon = []
    x_range, y_range = x_lim, y_lim

    obs_color = np.array([176, 124, 124]) / 255

    fig = plt.figure(figsize=(12, 5))
    for ii in range(len(obs)):
        plt.subplot(1, 2, (ii + 1))
        Gamma_obs = Gamma_vals[:, :, ii]

        max_val = 3
        if max_val:
            Gamma_obs[Gamma_obs > max_val] = max_val

        masked_array = np.ma.masked_where(Gamma_obs < 0.99, Gamma_obs)

        cmap = matplotlib.cm.winter
        cmap.set_bad(color=obs_color)

        dx2 = (x_grid[1] - x_grid[0]) / 2.0
        dy2 = (y_grid[1] - y_grid[0]) / 2.0

        value_min = 1
        value_max = np.max(Gamma_obs)

        im = plt.imshow(
            masked_array,
            cmap=cmap,
            vmin=value_min,
            vmax=value_max,
            extent=[x_lim[0] - dx2, x_lim[1] + dx2, y_lim[0] - dy2, y_lim[1] + dy2],
        )

        n = ii
        obs[n].draw_obstacle(numPoints=50)
        x_obs = obs[n].boundary_points_global_closed
        x_obs_sf = obs[n].boundary_points_margin_global_closed
        plt.plot(x_obs_sf[0, :], x_obs_sf[1, :], "k", linewidth=4)

        if obs[n].is_boundary:
            outer_boundary = np.array(
                [
                    [x_range[0], x_range[1], x_range[1], x_range[0]],
                    [y_range[0], y_range[0], y_range[1], y_range[1]],
                ]
            ).T

            boundary_polygon = plt.Polygon(outer_boundary, alpha=0.8, zorder=-1)
            boundary_polygon.set_color(np.array([176, 124, 124]) / 255.0)
            plt.gca().add_patch(boundary_polygon)

            obs_polygon.append(plt.Polygon(x_obs.T, alpha=1.0, zorder=-2))
            obs_polygon[n].set_color(np.array([1.0, 1.0, 1.0]))

        else:
            obs_polygon.append(plt.Polygon(x_obs.T, alpha=0.8, zorder=2))
            obs_polygon[n].set_color(obs_color)

        obs_polygon_sf.append(plt.Polygon(x_obs_sf.T, zorder=1, alpha=0.2))
        obs_polygon_sf[n].set_color([1, 1, 1])

        plt.gca().add_patch(obs_polygon_sf[n])
        plt.gca().add_patch(obs_polygon[n])

        reference_point = obs[n].get_reference_point(in_global_frame=True)
        plt.plot(
            reference_point[0],
            reference_point[1],
            "k+",
            linewidth=18,
            markeredgewidth=4,
            markersize=13,
        )

    fig.subplots_adjust(
        bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02
    )

    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8

    plt.yticks([])

    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)

    # cbar = ax.cax.colorbar(im)
    # cbar = grid.cbar_axes[0].colorbar(im)

    # cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    savee_fig = True
    if savee_fig:
        fig_name = "../figures/gamma_visualization_ellipse"
        fig.savefig(fig_name + ".png", bbox_inches="tight")
        fig.savefig(fig_name + ".pdf", bbox_inches="tight")


if (__name__) == "__main__":
    main()
