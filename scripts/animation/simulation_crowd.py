#!/USSR/bin/python3

"""
Script to show lab environment on computer

@author LukasHuber
@date 2020-01-15
@conact lukas.huber@epfl.ch
"""

# Command to automatically reload libraries -- in ipython before executing
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import *


def main(n_resol=90, *args, **kwargs):
    # x_lim = [0.9, 3.1]
    # y_lim = [0.9, 3.1]

    x_lim = [-4.4, 4.4]
    y_lim = [-4.1, 4.1]

    x_lim = [-5.4, 5.4]
    y_lim = [-5.1, 5.1]

    pos_attractor = [-1.5, 1.0]

    # obs = ObstacleContainer()
    obs = GradientContainer()

    edge_points = np.array(
        (
            [4.0, 1.0, 1.0, 0.0, 0.0, -4.0, -4.0, -2.5, -2.5, 0.0, 0.0, 4.0],
            [0.0, 0.0, 1.0, 1.0, 1.5, 1.5, -2.0, -2.0, -3.6, -3.6, -3.0, -3.0],
        )
    )

    # edge_points = np.array([[ 4, 4, 3, 1, -1, -3, -4, -4],
    # [-4, 4, 4, 2, 2, 4,  4, -4]])

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
        obs[-1].orientation += 0.5 / 180 * pi

        # Human
        obs.append(
            Ellipse(
                center_position=[-1.0, 0.8],
                axes_length=[0.35, 0.15],
                orientation=30 / 180.0 * pi,
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
        Simulation_vectorFields(
            x_lim,
            y_lim,
            n_resol,
            obs,
            xAttractor=pos_attractor,
            saveFigure=True,
            figName="lab_environment_closely_sparse",
            noTicks=False,
            automatic_reference_point=True,
            draw_vectorField=True,
        )

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
                orientation=90 / 180.0 * pi,
            )
        )
        # obs.append( Cuboid(center_position=[-1.0, -0.1], axes_length=[0.2, 0.4], margin_absolut=0.4, name="trolley", orientation=10/180.*pi))

        # Human
        # obs.append( Ellipse(center_position=[-1.0, -0.5], orientation=-30/180.*pi, axes_length=[0.5, 0.3], margin_absolut=0.4, name="human"))
        obs.append(
            Ellipse(
                center_position=[-0.5, -2.0],
                orientation=-30 / 180.0 * pi,
                axes_length=[0.5, 0.3],
                margin_absolut=0.4,
                name="human",
            )
        )

        Simulation_vectorFields(
            x_lim,
            y_lim,
            n_resol,
            obs,
            xAttractor=pos_attractor,
            saveFigure=True,
            figName="lab_environment_closely_sparse",
            noTicks=False,
            automatic_reference_point=True,
            draw_vectorField=True,
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

        Simulation_vectorFields(
            x_lim,
            y_lim,
            n_resol,
            obs,
            xAttractor=pos_attractor,
            saveFigure=True,
            figName="lab_environment_empty",
            noTicks=False,
            automatic_reference_point=True,
            draw_vectorField=True,
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

        # obs[-1].set_reference_point(np.array([3, 0]), in_global_frame=False)

        Simulation_vectorFields(
            x_lim,
            y_lim,
            n_resol,
            obs,
            xAttractor=pos_attractor,
            saveFigure=True,
            figName="one_square",
            noTicks=False,
            automatic_reference_point=True,
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

        Simulation_vectorFields(
            x_lim,
            y_lim,
            n_resol,
            obs,
            xAttractor=pos_attractor,
            saveFigure=True,
            figName="one_square_boundary_ref",
            noTicks=False,
            automatic_reference_point=True,
        )

        # obs[-1].set_reference_point(np.array([1.4, 1.4]), in_global_frame=False)
        # obs.append( Cuboid(center_position=[-1.0, -1.2], axes_length=[1.0, 1.0], margin_absolut=1.0, orientation=120*pi/180))
        # obs.append( Cuboid(center_position=[0.0, -0.9], axes_length=[1.0, 1.0], margin_absolut=1.0, orientation=0))
        # obs[-1].set_reference_point(np.array([1.4, 1.4]), in_global_frame=False)
    if 3 in cases:
        obs.append(
            Ellipse(
                center_position=[0.0, -1.0],
                axes_length=[0.8, 1.2],
                # margin_absolut=1.0,
                margin_absolut=0.5,
                orientation=30 * pi / 180,
            )
        )

        # obs[-1].set_reference_point(np.array([0, 2]), in_global_frame=True)

        Simulation_vectorFields(
            x_lim,
            y_lim,
            n_resol,
            obs,
            xAttractor=pos_attractor,
            saveFigure=True,
            figName="lab_environment",
            noTicks=False,
            automatic_reference_point=True,
            draw_vectorField=True,
        )

    if 4 in cases:
        obs.append(
            Ellipse(
                center_position=[0.0, -1.0],
                axes_length=[0.8, 1.2],
                margin_absolut=0.5,
                orientation=30 * pi / 180,
            )
        )

        # obs.append( Ellipse(
        # center_position=[0.0, -2.0],
        # axes_length=[0.4, 1.0],
        # margin_absolut=0.5,
        # orientation=-30*pi/180))

        # obs.append( Ellipse(
        # center_position=[0.0, 1.0],
        # axes_length=[0.4, 1.0],
        # margin_absolut=1.0,
        # orientation=-30*pi/180))

        n_resol = 10
        obs[-1].set_reference_point(np.array([2.9, 2]), in_global_frame=True)
        Simulation_vectorFields(
            x_lim,
            y_lim,
            n_resol,
            obs,
            xAttractor=pos_attractor,
            saveFigure=True,
            figName="ellipse_reference_point",
            noTicks=False,
            automatic_reference_point=True,
            draw_vectorField=True,
        )

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

        # obs[-1].set_reference_point(np.array([2.9, 2]), in_global_frame=True)

        Simulation_vectorFields(
            x_lim,
            y_lim,
            n_resol,
            obs,
            xAttractor=pos_attractor,
            saveFigure=True,
            figName="two_obstacles",
            noTicks=False,
            automatic_reference_point=True,
            draw_vectorField=True,
        )

    draw_gamma(obs, y_lim, x_lim, n_resolution=100)
    return


def draw_gamma(obs, y_lim, x_lim, n_resolution=100):
    plt.figure()
    plt.grid(True)

    # coloring_gamma = True
    # if coloring_gamma:
    # Gamma_coloring =

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

    merge_gammas = False
    if merge_gammas:
        Gamma_prod = np.ones((n_resolution, n_resolution))
        for oo in range(n_obs):
            Gamma_prod = Gamma_prod * np.max(
                (Gamma_vals[:, :, oo] - 1, np.zeros(Gamma_vals[:, :, oo].shape)), axis=0
            )
        # Gamma_prod = Gamma_prod*Gamma_vals[:, :, oo]

        # Gamma_prod = (Gamma_prod+1)**(1.0/n_obs)
        # Gamma_prod = (Gamma_prod + 1)**(1.0/n_obs)

        # Gamma_vals = Gamma_vals[:, :, 0]
        Gamma_vals = Gamma_prod
    else:
        it_obs = 0
        Gamma_vals = Gamma_vals[:, :, it_obs]

    max_val = 3
    if max_val:
        Gamma_vals[Gamma_vals > max_val] = max_val
    # masked_array = np.ma.masked_where(Gamma_vals<0.99, Gamma_vals)
    masked_array = np.ma.masked_where(Gamma_vals < 0.99, Gamma_vals)
    # masked_array = np.ma.masked_where(Gamma_vals>1.01, Gamma_vals)

    cmap = matplotlib.cm.winter
    cmap.set_bad(color="white")

    # plt.figure()
    # im = plt.imshow(Gamma_vals, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]])
    dx2 = (x_grid[1] - x_grid[0]) / 2.0
    dy2 = (y_grid[1] - y_grid[0]) / 2.0

    im = plt.imshow(
        masked_array,
        cmap=cmap,
        extent=[x_lim[0] - dx2, x_lim[1] + dx2, y_lim[0] - dy2, y_lim[1] + dy2],
    )
    ax = plt.gca()
    cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    if False:
        for it_obs in [0]:
            plt.quiver(
                positions[0, :, :],
                positions[1, :, :],
                normals[0, :, :, it_obs],
                normals[1, :, :, it_obs],
            )


if str(__name__) == ("__main__"):
    main()
