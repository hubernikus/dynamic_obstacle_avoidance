s  #!/USSR/bin/python3
""" Script to show lab environment on computer """

# Author: Lukas Huber
# Created: 2020-01-15
# Email: lukas.huber@epfl.ch

import numpy as np
from numpy import linalg as LA
from numpy import pi

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
    plot_obstacles,
)

from vartools.dynamical_systems import LinearSystem


plt.close("all")
plt.ion()


def main_simple_ellipse(n_resolution=10, save_figure=False):
    x_lim = [-5, 5]
    y_lim = [-5, 5]

    ellipse_0 = Ellipse(
        center_position=[0, 0], orientation=45 * np.pi / 180, axes_length=[4, 8]
    )

    ellipse_inv = Ellipse(
        center_position=ellipse_0.center_position,
        orientation=ellipse_0.orientation,
        axes_length=ellipse_0.axes_length,
        is_boundary=True,
    )

    nx = n_resolution
    ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    gamma_0 = np.zeros(positions.shape[1])
    gamma_inv = np.zeros(positions.shape[1])

    for ii in range(positions.shape[1]):
        gamma_0[ii] = ellipse_0.get_gamma(positions[:, ii], in_global_frame=True)
        gamma_inv[ii] = ellipse_inv.get_gamma(positions[:, ii], in_global_frame=True)

    # fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    fig = plt.figure(figsize=(8, 3.5))
    grid = ImageGrid(
        fig,
        111,  # as in plt.subplot(111)
        nrows_ncols=(1, 2),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="7%",
        cbar_pad=0.15,
    )

    axs = grid

    levels = np.linspace(1, 4, 7)

    axs[0].contourf(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        gamma_0.reshape(nx, ny),
        levels=levels,
        extend="max",
        zorder=-10,
        cmap="winter",
    )

    plot_obstacles(
        obstacle_container=[ellipse_0],
        ax=axs[0],
        x_lim=x_lim,
        y_lim=y_lim,
        noTicks=True,
        # draw_reference=,
    )

    cols = axs[1].contourf(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        gamma_inv.reshape(nx, ny),
        levels=levels,
        extend="max",
        # interpolation='none',
        zorder=0,
        cmap="winter",
    )

    plot_obstacles(
        obstacle_container=[ellipse_inv],
        ax=axs[1],
        x_lim=x_lim,
        y_lim=y_lim,
        noTicks=True,
        # draw_reference=,
    )

    # Colorbar
    cbar = axs[0].cax.colorbar(cols)
    axs[1].cax.toggle_label(True)
    cbar.ax.set_title(r"$\Gamma(\xi) $")

    if save_figure:
        figure_name = "gamma_visualization_ellipse"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight", dpi=600)


def main_multicuboid_gamma(n_resolution=10, save_figure=False):
    x_lim = [-3, 3]
    y_lim = [-0.5, 3.5]

    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Cuboid(
            center_position=[0, 2.5],
            orientation=-30 * np.pi / 180,
            axes_length=np.array([1, 5]) / 2,
            margin_absolut=0.2,
        )
    )
    obstacle_environment[-1].set_reference_point(
        np.array([0, 1.0]), in_obstacle_frame=True
    )

    obstacle_environment.append(
        Cuboid(
            center_position=[1.24, 0.25],
            orientation=0 * np.pi / 180,
            axes_length=np.array([1, 3]) / 2,
            margin_absolut=0.2,
        )
    )

    obstacle_environment[-1].set_reference_point(
        np.array([0, -0.7]), in_obstacle_frame=True
    )

    initial_dynamics = LinearSystem(attractor_position=np.array([2.5, 0.5]))

    nx = n_resolution
    ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    gamma = np.zeros(positions.shape[1])

    for ii in range(positions.shape[1]):
        gamma[ii] = obstacle_environment.get_multiobstacle_gamma(positions[:, ii])

    fig = plt.figure(figsize=(8, 4.0))
    grid = ImageGrid(
        fig,
        111,  # as in plt.subplot(111)
        nrows_ncols=(1, 1),
        axes_pad=0.1,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="4%",
        cbar_pad=0.10,
    )

    axs = grid

    # levels = np.linspace(1, 6.0, 41)
    levels = np.linspace(1, 4.0, 31)

    cols = axs[0].contourf(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        gamma.reshape(nx, ny),
        levels=levels,
        extend="max",
        zorder=-10,
        cmap="hot",
        alpha=0.8,
    )

    Simulation_vectorFields(
        x_lim,
        y_lim,
        point_grid=n_resolution,
        obs=obstacle_environment,
        pos_attractor=initial_dynamics.attractor_position,
        dynamical_system=initial_dynamics.evaluate,
        noTicks=True,
        automatic_reference_point=False,
        show_streamplot=True,
        draw_vectorField=True,
        normalize_vectors=False,
        showLabel=False,
        streamColor="black",
        fig_and_ax_handle=(fig, axs[0]),
    )

    cbar = axs[0].cax.colorbar(cols)
    axs[0].cax.toggle_label(True)
    cbar.ax.set_title(r"$\Gamma^d(\xi) $")

    if save_figure:
        figure_name = "gamma_danger_field_for_multiobstacle_and_vector_field"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight", dpi=600)


def main(n_resol=20, *args, **kwargs):
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

    cases = [4]

    if -1 in cases:
        x_lim = [-3.1, 2.0]
        y_lim = [-2.2, 2.0]

        robot_radius = 0.48
        exponential_weight = 1

        edge_points = np.array(
            (
                [3.8, 3.8, -0.5, -0.5, 0.2, 0.2, 3.8],
                [0.0, 2.0, 2.0, -0.8, -0.8, -2.0, -2.0],
            )
        )

        obs.append(
            Polygon(
                edge_points=edge_points,
                is_boundary=True,
                margin_absolut=robot_radius,
                sigma=exponential_weight,
                name="world_lab",
            )
        )

        # Displacement
        obs[-1].center_position += np.array([-2.4, 0])
        obs[-1].orientation += 0.5 / 180 * np.pi

        # Human
        obs.append(
            Ellipse(
                center_position=[-1.0, 0.8],
                axes_length=[0.35, 0.15],
                orientation=30 / 180.0 * np.pi,
                margin_absolut=robot_radius,
                sigma=exponential_weight,
                name="coworker",
            )
        )
        obs[-1].is_static = False

        # Table
        obs.append(
            Cuboid(
                center_position=[0.3, -1.70],
                axes_length=[1.0, 1.0],
                margin_absolut=robot_radius,
                sigma=exponential_weight,
                name="KIA",
            )
        )
        obs[-1].is_static = True

        # Table
        obs.append(
            Cuboid(
                center_position=[-0.65, -1.35],
                axes_length=[0.8, 1.8],
                margin_absolut=robot_radius,
                sigma=exponential_weight,
                name="table",
            )
        )
        obs[-1].is_static = True

        # Table
        obs.append(
            Cuboid(
                center_position=[1.18, 0.86],
                axes_length=[0.8, 1.8],
                margin_absolut=robot_radius,
                sigma=exponential_weight,
                name="table_computer",
            )
        )
        obs[-1].is_static = True

        pos_attractor = [0.3, -0.75]
        pos_attractor = [-1.75, -0.97]

        # pos_attractor = obs['coworker'].get_reference_point(in_global_frame=True)
        n_resol = 3

    if 0 in cases:
        robot_margin = 0.4  # radius
        # edge_points = np.array((
        # [100.0,-100.0,-100.0, 100.0],
        # [100.0, 100.0, -100.0,-100.0]))

        # frame_id = "world_lab"
        # obs.append( Polygon(edge_points=edge_points, margin_absolut=0.5))
        x_lim = [-4.1, 4.1]
        y_lim = [-4, 2]
        obs.append(
            Polygon(
                edge_points=edge_points,
                is_boundary=True,
                margin_absolut=robot_margin,
                center_position=np.array([-1.0, -1]),
                name="lab",
            )
        )

        # Displacement

        # Table
        obs.append(
            Cuboid(
                center_position=[-2.5, 0.0],
                axes_length=[0.8, 0.8],
                margin_absolut=0.4,
                name="table",
            )
        )

        # Tool-Trolley
        obs.append(
            Cuboid(
                center_position=[2.1, -1.5],
                axes_length=[0.2, 0.4],
                margin_absolut=0.4,
                name="trolley",
                orientation=90 / 180.0 * np.pi,
            )
        )

        # Human
        obs.append(
            Ellipse(
                center_position=[-0.5, -2.0],
                orientation=-30 / 180.0 * pi,
                axes_length=[0.5, 0.3],
                margin_absolut=0.4,
                name="human",
            )
        )

        plt.grid("true")

    if 1.1 in cases:
        robot_margin = 0.4  # radius

        obs.append(
            Polygon(
                edge_points=edge_points,
                is_boundary=True,
                margin_absolut=robot_margin,
                center_position=np.array([-1.0, -1]),
                name="lab",
            )
        )

    if 2 in cases:
        obs.append(
            Cuboid(
                center_position=[0.0, 0.0],
                axes_length=[2.0, 2.0],
                # margin_absolut=0.0,
                margin_absolut=1.0,
                orientation=0 * pi / 180,
            )
        )

    if 2.1 in cases:
        obs.append(
            Cuboid(
                center_position=[0.0, 0.0],
                axes_length=[2.0, 2.0],
                # margin_absolut=0.0,
                margin_absolut=1.0,
                orientation=0 * pi / 180,
            )
        )

        obs[-1].set_reference_point(np.array([3, 0]), in_global_frame=False)

    if 3 in cases:
        obs.append(
            Ellipse(
                center_position=[2.0, 0.0],
                axes_length=[0.8, 1.2],
                # margin_absolut=1.0,
                margin_absolut=0.5,
                orientation=-30 * pi / 180,
            )
        )

        obs.append(
            Ellipse(
                center_position=[0.0, 0.0],
                axes_length=[0.8, 1.2],
                # margin_absolut=1.0,
                margin_absolut=0.5,
                orientation=30 * pi / 180,
            )
        )

        # x_lim = [-2, 4]
        # y_lim = [-3., 3]

        x_lim = [-0.5, 2.2]
        y_lim = [-1.5, 1]

    if 4 in cases:
        obs.append(
            Ellipse(
                center_position=[0.0, 0.0],
                axes_length=[0.8, 1.2],
                # margin_absolut=1.0,
                margin_absolut=0.5,
                orientation=30 * pi / 180,
                linear_velocity=np.array([0.0, 0]),
            )
        )

        obs.append(
            Ellipse(
                center_position=[3.5, 0.0],
                axes_length=[0.8, 2.2],
                # margin_absolut=1.0,
                margin_absolut=0.5,
                orientation=-30 * (1 - 3.8) * pi / 180,
                linear_velocity=[0.0, 0],
                angular_velocity=30 / 180.0 * pi,
            )
        )

        x_lim = [-2, 4]
        y_lim = [-3.0, 3]

    if 5 in cases:
        obs.append(
            Ellipse(
                center_position=[4.0, -3.0],
                axes_length=[0.8, 1.2],
                # margin_absolut=1.0,
                margin_absolut=0.5,
                orientation=30 * pi / 180,
            )
        )

        obs.append(
            Cuboid(
                center_position=[0.0, 0.0],
                axes_length=[2.0, 2.0],
                # margin_absolut=0.0,
                margin_absolut=1.0,
                orientation=0 * pi / 180,
            )
        )

    n_resolution = n_resol

    # pos = np.array([1.0, 0])
    # pos = np.array([-.0, -3.0])
    pos = np.array([-0.150, -2.7])
    normal0 = obs[0].get_normal_direction(pos, in_global_frame=True)
    gamma0 = obs[0].get_gamma(pos, in_global_frame=True)

    x_grid = np.linspace(x_lim[0], x_lim[1], n_resolution)
    y_grid = np.linspace(y_lim[0], y_lim[1], n_resolution)

    n_obs = len(obs)
    Gamma_vals = np.zeros((n_resolution, n_resolution, n_obs))
    normals = np.zeros((obs[0].dim, n_resolution, n_resolution, n_obs))
    positions = np.zeros((obs[0].dim, n_resolution, n_resolution))

    local_rad = np.zeros((n_resolution, n_resolution, n_obs))

    for it_obs in range(n_obs):
        for ix in range(n_resolution):
            for iy in range(n_resolution):
                pos = np.array([x_grid[ix], y_grid[iy]])

                positions[:, ix, iy] = pos

                Gamma_vals[ix, iy, it_obs] = obs[it_obs].get_gamma(
                    pos, gamma_type="linear", in_global_frame=True
                )

                pos = obs[it_obs].transform_global2relative(pos)
                norm_pos = np.linalg.norm(pos)
                if norm_pos:
                    pos = pos / norm_pos

                local_rad[ix, iy, it_obs] = np.linalg.norm(
                    obs[it_obs].get_local_radius_point(pos)
                )

                # normals[:, ix, iy, it_obs] = obs[it_obs].get_normal_direction(pos, in_global_frame=True)

    # import pdb; pdb.set_trace()
    # merge_type = None
    merge_type = "sum"

    if merge_type == "prod":
        # Product
        Gamma_prod = np.ones((n_resolution, n_resolution))
        for oo in range(n_obs):
            # Gamma_prod = Gamma_prod*np.max((Gamma_vals[:, :, oo]-1, np.zeros(Gamma_vals[:, :, oo].shape)), axis=0 )
            Gamma_prod = Gamma_prod * Gamma_vals[:, :, oo]
        Gamma_vals = Gamma_prod

    elif merge_type == "sum":
        sum_power = 3
        Gamma_mege = np.zeros((n_resolution, n_resolution))
        for oo in range(n_obs):
            # if not oo==0: continue
            Gamma_mege = Gamma_mege + Gamma_vals[:, :, oo] ** sum_power
        Gamma_vals = Gamma_mege

    elif merge_type == "local_sum":
        sum_power = 3
        Gamma_mege = np.zeros((n_resolution, n_resolution))
        for oo in range(n_obs):
            Gamma_mege = (
                Gamma_mege + (Gamma_vals[:, :, oo] * local_rad[:, :, oo]) ** sum_power
            )
            # if oo==0:
            # Gamma_mege = Gamma_mege+(local_rad[:, :, oo])**sum_power
        Gamma_vals = Gamma_mege

    elif merge_type == "weighted":
        sum_power = 1
        Gamma_mege = np.zeros((n_resolution, n_resolution))
        for oo in range(n_obs):
            Gamma_mege = Gamma_mege + Gamma_vals[:, :, oo] ** sum_power
        Gamma_vals = Gamma_mege

    else:
        it_obs = 0
        Gamma_vals = Gamma_vals[:, :, it_obs]

    (ix_min, iy_min) = np.unravel_index(
        np.argmin(Gamma_vals, axis=None), (n_resolution, n_resolution)
    )
    pos_min = positions[:, ix_min, iy_min]

    do_quiver = False
    if do_quiver:
        gamma_derivative = np.zeros((obs[0].dim, n_resolution, n_resolution))

        for ix in range(n_resolution):
            for iy in range(n_resolution):
                if (
                    obs[0].get_gamma(positions[:, ix, iy], in_global_frame=True) <= 1
                    and obs[1].get_gamma(positions[:, ix, iy], in_global_frame=True)
                    <= 1
                ):
                    gamma_derivative[:, ix, iy] = derivative_gamma_sum(
                        positions[:, ix, iy], obs[0], obs[1]
                    )
                    norm = np.linalg.norm(gamma_derivative[:, ix, iy])
                    if norm:  # nonzero
                        gamma_derivative[:, ix, iy] = gamma_derivative[:, ix, iy] / norm

    print("Merge type {}".format(merge_type))

    max_val = None
    if not max_val is None:
        Gamma_vals[Gamma_vals > max_val] = max_val

    fig = plt.figure(figsize=(10, 8))
    cs = plt.contourf(
        positions[0, :, :],
        positions[1, :, :],
        Gamma_vals,
        np.arange(0, 3.5, 0.05),
        extend="max",
        alpha=0.6,
        zorder=-3,
    )

    cbar = fig.colorbar(cs)
    for obstacle in obs:
        obstacle.draw_obstacle(numPoints=50)
        plt.plot(
            obstacle.x_obs_sf[0, :],
            obstacle.x_obs_sf[1, :],
            ":",
            color="k",
            linewidth=3,
        )

        plt.plot(
            obstacle.center_position[0],
            obstacle.center_position[1],
            "k+",
            linewidth=18,
            markeredgewidth=4,
            markersize=13,
        )

    plt.plot(
        pos_min[0],
        pos_min[1],
        color="g",
        marker="D",
        markersize=13,
        label="Gridsearch Minimum",
    )
    plt.axis("equal")
    plt.legend()

    if do_quiver:
        # plt.quiver(positions[0, :, :], positions[1, :],
        # gamma_derivative[0, :, :], gamma_derivative[1, :, :], color='k')

        plt.streamplot(
            positions[0, :, 0],
            positions[1, 0, :],
            gamma_derivative[0, :, :].T,
            gamma_derivative[1, :, :].T,
            color="k",
            zorder=-2,
        )

    plt.xlim(x_lim)
    plt.ylim(y_lim)

    plt.show()


if (__name__) == "__main__":
    # main(n_resol=100)

    # main_simple_ellipse(n_resolution=100, save_figure=True)
    main_multicuboid_gamma(n_resolution=100, save_figure=True)
