"""Three (3) dimensional representation of obstacle avoidance. """
# Author: Lukas Huber
# Date: 2021-06-25
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import os
import warnings

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# ! Miavy needed for 3D plotting
# Turn on 3D plotting (!)
# %gui qt

# import mayavi.mlab

from vartools.dynamical_systems import LinearSystem

plt.ion()  # Show plot without stopping code


def plot_obstacles(ObstacleContainer, ax=None):
    """ """
    if ax is None:
        # breakpoint()
        fig = plt.figure()
        ax = fig.gca(projection="3d")
    # else:
    # fig, ax = fig_and_ax_handle

    for obs in ObstacleContainer:
        data_points = obs.draw_obstacle()
        ax.plot_surface(
            data_points[0],
            data_points[1],
            data_points[2],
            rstride=4,
            cstride=4,
            color=np.array([176, 124, 124]) / 255.0,
        )


def plot_obstacles_and_vector_levelz_3d(
    ObstacleContainer,
    InitialDynamics,
    func_obstacle_avoidance,
    x_lim=None,
    y_lim=None,
    z_lim=None,
    z_value=None,
    n_grid=10,
    n_grid_z=4,
    fig_and_ax_handle=None,
    show_axes_label=False,
):
    dimension = 3  # 3D-visualization

    x_values = np.linspace(x_lim[0], x_lim[1], n_grid)
    y_values = np.linspace(y_lim[0], y_lim[1], n_grid)

    if z_value is None:
        z_values = np.linspace(z_lim[0], z_lim[1], n_grid_z)
    else:
        # Single value for observtion
        z_values = np.array([z_value])
        n_grid_z = z_values.shape[0]

    xx, yy, zz = np.meshgrid(x_values, y_values, z_values)

    ########## DEBUGGING ONLY ##########
    # TODO: DEBUGGING Only for Development and testing
    n_samples = 0
    if n_samples:  # nonzero
        n_x = n_y = n_grid
        it_start = 0

        # pos1 = [0.5935, 0.1560, 0]
        # pos2 = [1.1134, 0.1530, 0]
        pos1 = [0.8580, 0.1661, 0]
        pos2 = [1.0005, 0.1537, 0]

        x_sample_range = [pos1[0], pos2[0]]
        y_sample_range = [pos1[1], pos2[1]]

        x_sample = np.linspace(x_sample_range[0], x_sample_range[1], n_samples)
        y_sample = np.linspace(y_sample_range[0], y_sample_range[1], n_samples)

        ii = 0
        iz = 0
        for ii in range(n_samples):
            iy = (ii + it_start) % n_y
            ix = int((ii + it_start) / n_x)

            xx[ix, iy, iz] = x_sample[ii]
            yy[ix, iy, iz] = y_sample[ii]
    ########## STOP REMOVE ###########

    # color_map = 'cividis'
    color_map = "cool"
    # color_map = 'hsv'

    positions = np.zeros((dimension, n_grid, n_grid))
    linear_velocities = np.zeros((dimension, n_grid, n_grid, n_grid_z))
    modulated_velocities = np.zeros((dimension, n_grid, n_grid, n_grid_z))

    for ix in range(n_grid):
        for iy in range(n_grid):
            for iz in range(n_grid_z):
                pos = np.array([xx[ix, iy, iz], yy[ix, iy, iz], zz[ix, iy, iz]])
                if ObstacleContainer.check_collision(pos):
                    continue

                linear_velocities[:, ix, iy, iz] = InitialDynamics.evaluate(pos)
                modulated_velocities[:, ix, iy, iz] = func_obstacle_avoidance(
                    pos, linear_velocities[:, ix, iy, iz], ObstacleContainer
                )

    uvw = modulated_velocities
    norm = np.linalg.norm(uvw, axis=0)
    max_norm = np.max(norm)
    if max_norm == 0:
        breakpoint()
        warnings.warn("Fully zero velocity surrounding. Ploting does not seem useful.")
        max_norm = 1  # set nonzero for future divides...
    mask = norm == 0
    min_norm = 0.3  # you want every arrow to be longer than this fraction of max_norm
    # rescale vs for illustrative purposes, so small vectors become visible
    # and zero vectors become nonzero so colors of the arrow shaft and head correspond.
    # Later these are made transparent
    uvw = uvw + min_norm * np.tile(mask[np.newaxis], (3, 1, 1, 1)) / max_norm
    # recalculate norms so you don't divide by zero
    norm = np.linalg.norm(uvw, axis=0)
    uvw = min_norm * uvw / norm + (1 - min_norm) * uvw / max_norm
    # uu, vv, ww = uvw
    repeated_mask = np.concatenate((mask.ravel(), np.repeat(mask.ravel(), 2)))

    cc = zz
    col_range = cc.ptp()
    if not col_range:
        # Avoid zero divide
        col_range = 1
    # Flatten & normalize
    cc = (cc.ravel() - cc.min()) / col_range
    # Adjust for missing quivers
    # c = c[np.nonzero((u.ravel() != 0) * (v.ravel() != 0) * (w.ravel() != 0))]
    # Repeat for each body line and two head lines
    cc = np.concatenate((cc, np.repeat(cc, 2)))
    repeated_mask = np.concatenate((mask.ravel(), np.repeat(mask.ravel(), 2)))

    # Colormap
    cc = getattr(plt.cm, color_map)(cc)
    # set zero values transparent, you made them nonzero not to mess up the tip colors
    cc[repeated_mask, 3] = 0.0

    if fig_and_ax_handle is None:
        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes(projection="3d")
    else:
        fig, ax = fig_and_ax_handle
    plot_obstacles(ObstacleContainer=ObstacleContainer, ax=ax)

    q = ax.quiver(
        xx,
        yy,
        zz,
        uvw[0, :, :],
        uvw[1, :, :],
        uvw[2, :, :],
        # modulated_velocities[0, :, :, :],
        # modulated_velocities[1, :, :, :],
        # modulated_velocities[2, :, :, :],
        cmap=color_map,
        length=0.3,
        normalize=True,
    )

    q.set_array(np.linspace(0, max_norm, 10))
    # fig.colorbar(q)
    q.set_edgecolor(cc)
    q.set_facecolor(cc)

    if hasattr(InitialDynamics, "attractor_position"):
        ax.plot(
            [InitialDynamics.attractor_position[0]],
            [InitialDynamics.attractor_position[1]],
            [InitialDynamics.attractor_position[2]],
            "k*",
        )

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if z_lim is not None:
        ax.set_zlim(z_lim)

    if show_axes_label:
        ax.set_xlabel(r"$\xi_1$", fontsize=16)
        ax.set_ylabel(r"$\xi_2$", fontsize=16)
        ax.set_zlabel(r"$\xi_3$", fontsize=16)

    plt.ion()
    plt.show()


def plot_obstacles_and_trajectory_3d(
    ObstacleContainer,
    InitialDynamics,
    func_obstacle_avoidance,
    start_positions=None,
    n_points=2,
    delta_time=0.01,
    n_max_it=10000,
    convergence_margin=1e-4,
    zero_vel_margin=1e-5,
    x_lim=None,
    y_lim=None,
    z_lim=None,
    fig_and_ax_handle=None,
    show_axes_label=True,
):
    dimension = 3  # 3D-visualization

    if fig_and_ax_handle is None:
        fig = plt.figure(figsize=(8, 6))
        ax = plt.axes(projection="3d")
    else:
        fig, ax = fig_and_ax_handle

    plot_obstacles(ObstacleContainer=ObstacleContainer, ax=ax)

    if start_positions is None:
        start_positions = np.random.rand(dimension, n_points)

        start_positions[0, :] = start_positions[0, :] * (x_lim[1] - x_lim[0]) + x_lim[0]
        start_positions[1, :] = start_positions[1, :] * (y_lim[1] - y_lim[0]) + y_lim[0]
        start_positions[2, :] = start_positions[2, :] * (z_lim[1] - z_lim[0]) + z_lim[0]
    else:
        n_points = start_positions.shape[1]

    trajectory_points = [np.zeros((dimension, n_max_it)) for ii in range(n_points)]
    for ii in range(n_points):
        trajectory_points[ii][:, 0] = start_positions[:, ii]

    active_trajectories = np.ones((n_points), dtype=bool)
    velocity_old = None
    for it_step in range(n_max_it - 1):
        for ii in np.arange(n_points)[active_trajectories]:
            initial_velocity = InitialDynamics.evaluate(
                trajectory_points[ii][:, it_step]
            )
            modulated_velocity = func_obstacle_avoidance(
                trajectory_points[ii][:, it_step],
                initial_velocity,
                ObstacleContainer,
            )
            trajectory_points[ii][:, it_step + 1] = (
                trajectory_points[ii][:, it_step] + modulated_velocity * delta_time
            )

            # Check convergence
            if (
                np.linalg.norm(
                    trajectory_points[ii][:, it_step + 1]
                    - InitialDynamics.attractor_position
                )
                < convergence_margin
            ):
                print(f"Trajectory {ii} has converged at step {it_step}.")
                active_trajectories[ii] = False

            if np.linalg.norm(modulated_velocity) < zero_vel_margin:
                print(
                    f"Trajectory {ii} ist stuck at step {it_step} and "
                    + f"position {trajectory_points[ii][:, it_step+1]}."
                )
                active_trajectories[ii] = False

            if velocity_old is not None:
                if np.linalg.norm(modulated_velocity - velocity_old) > 3.5e-1:
                    print(f"Watch out at it={it_step}")
                    print(
                        f"vel1={velocity_old} \n"
                        + f"vel2={modulated_velocity} \n"
                        + f"pos1={trajectory_points[ii][:, it_step-1]} \n"
                        + f"pos2={trajectory_points[ii][:, it_step]} \n\n"
                    )
                    ax.plot(
                        [trajectory_points[ii][0, it_step]],
                        [trajectory_points[ii][1, it_step]],
                        [trajectory_points[ii][2, it_step]],
                        "k.",
                    )

                    # Currently only interested in first cuttie
                    # break
            velocity_old = modulated_velocity

        if not any(active_trajectories):
            print("All trajectories have converged. Stopping the simulation.")
            break

    for ii in np.arange(n_points):
        ax.plot(
            [trajectory_points[ii][0, 0]],
            [trajectory_points[ii][1, 0]],
            [trajectory_points[ii][2, 0]],
            "k.",
        )

        ax.plot(
            trajectory_points[ii][0, :],
            trajectory_points[ii][1, :],
            trajectory_points[ii][2, :],
        )

    if hasattr(InitialDynamics, "attractor_position"):
        ax.plot(
            [InitialDynamics.attractor_position[0]],
            [InitialDynamics.attractor_position[1]],
            [InitialDynamics.attractor_position[2]],
            "k*",
        )

    # ax.set_aspect('equal')
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if z_lim is not None:
        ax.set_zlim(z_lim)

    if show_axes_label:
        ax.set_xlabel(r"$\xi_1$", fontsize=16)
        ax.set_ylabel(r"$\xi_2$", fontsize=16)
        ax.set_zlabel(r"$\xi_3$", fontsize=16)

    plt.ion()
    plt.show()
