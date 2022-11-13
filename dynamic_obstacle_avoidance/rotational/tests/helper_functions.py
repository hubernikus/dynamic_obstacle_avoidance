import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.rotational import kmeans_motion_learner as kml

fig_type = ".png"
# fig_type = ".pdf"


def get_grid_points(mean_x, delta_x, mean_y, delta_y, n_points):
    """Returns grid based on input x and y values."""
    x_min = mean_x - delta_x
    x_max = mean_x + delta_x

    y_min = mean_y - delta_y
    y_max = mean_y + delta_y

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_points),
        np.linspace(y_min, y_max, n_points),
    )

    return np.array([xx.flatten(), yy.flatten()])


def plot_region_dynamics(main_learner, x_lim, y_lim, n_grid=20, ax=None):
    """ Plot the dynamics withing the obstacle."""
    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], n_grid),
        np.linspace(y_lim[0], y_lim[1], n_grid),
    )
    positions = np.array([xx.flatten(), yy.flatten()])

    velocities = np.zeros_like(positions)

    for pp in range(positions.shape[1]):
        is_inside = False
        for obs in main_learner.region_obstacles:
            if obs.is_inside(positions[:, pp], in_global_frame=True):
                is_inside = True
                break

        if not is_inside:
            continue

        velocities[:, pp] = main_learner.predict(positions[:, pp])

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))
    else:
        fig = None
    main_learner.plot_boundaries(ax=ax, plot_attractor=True)
    ax.quiver(
        positions[0, :],
        positions[1, :],
        velocities[0, :],
        velocities[1, :],
        # color="red",
        scale=50,
    )
    ax.axis("equal")

    return fig, ax


def plot_global_dynamics(main_learner, x_lim, y_lim, n_grid=20, ax=None):
    """ Plot the dynamics withing the obstacle."""
    print("entering")
    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], n_grid),
        np.linspace(y_lim[0], y_lim[1], n_grid),
    )
    positions = np.array([xx.flatten(), yy.flatten()])
    velocities = np.zeros_like(positions)

    for pp in range(positions.shape[1]):
        velocities[:, pp] = main_learner.predict(positions[:, pp])

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 9))
    else:
        fig = None
    main_learner.plot_boundaries(ax=ax, plot_attractor=True)
    ax.quiver(
        positions[0, :],
        positions[1, :],
        velocities[0, :],
        velocities[1, :],
        # color="red",
        scale=50,
    )
    ax.axis("equal")

    return fig, ax



def plot_gamma(ax, obstacle, x_lim=None, y_lim=None, n_grid=100):
    if x_lim is None:
        x_lim = obstacle.center_position[0] + np.array([-1, 1]) * obstacle.radius

    if y_lim is None:
        y_lim = obstacle.center_position[1] + np.array([-1, 1]) * obstacle.radius

    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], n_grid),
        np.linspace(y_lim[0], y_lim[1], n_grid),
    )
    positions = np.array([xx.flatten(), yy.flatten()])
    gamma = np.zeros(positions.shape[1])

    for pp in range(positions.shape[1]):
        if not obstacle.is_inside(positions[:, pp], in_global_frame=True):
            continue

        gamma[:, pp] = obstacle.get_gamma(positions[:, pp], in_global_frame=True)


def plot_normals(ax, obstacle, x_lim=None, y_lim=None, n_grid=10):
    if x_lim is None:
        x_lim = obstacle.center_position[0] + np.array([-1, 1]) * obstacle.radius

    if y_lim is None:
        y_lim = obstacle.center_position[1] + np.array([-1, 1]) * obstacle.radius

    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], n_grid),
        np.linspace(y_lim[0], y_lim[1], n_grid),
    )
    positions = np.array([xx.flatten(), yy.flatten()])

    normals = np.zeros_like(positions)

    for pp in range(positions.shape[1]):
        if not obstacle.is_inside(positions[:, pp], in_global_frame=True):
            continue

        normals[:, pp] = obstacle.get_normal_direction(
            positions[:, pp], in_global_frame=True
        )

    ax.quiver(positions[0, :], positions[1, :], normals[0, :], normals[1, :], scale=50)
    # breakpoint()


def plot_boundaries(kmeans_learner, ax, plot_attractor=False) -> None:

    for ii in range(kmeans_learner.kmeans.n_clusters):
        tmp_obstacle = kml.create_kmeans_obstacle_from_learner(kmeans_learner, ii)

        positions = tmp_obstacle.evaluate_surface_points()
        ax.plot(
            positions[0, :],
            positions[1, :],
            color="black",
            linewidth=3.5,
            zorder=5,
        )

    centroids = kmeans_learner.kmeans.cluster_centers_
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="black",
        zorder=5,
    )

    if plot_attractor:
        ax.scatter(
            kmeans_learner.data.attractor[0],
            kmeans_learner.data.attractor[1],
            marker="*",
            s=200,
            color="black",
            zorder=5,
        )


def plot_reference_dynamics(
    ax, kmeans_learner, index, x_lim=None, y_lim=None, n_grid=10
) -> None:
    obstacle = kmeans_learner.region_obstacles[index]

    if x_lim is None:
        x_lim = obstacle.center_position[0] + np.array([-1, 1]) * obstacle.radius

    if y_lim is None:
        y_lim = obstacle.center_position[1] + np.array([-1, 1]) * obstacle.radius

    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], n_grid),
        np.linspace(y_lim[0], y_lim[1], n_grid),
    )

    positions = np.array([xx.flatten(), yy.flatten()])
    normals = np.zeros_like(positions)

    for pp in range(positions.shape[1]):
        if not obstacle.is_inside(positions[:, pp], in_global_frame=True):
            continue

        normals[:, pp] = kmeans_learner._dynamics[index].evaluate_convergence_velocity(
            positions[:, pp]
        )

    ax.quiver(positions[0, :], positions[1, :], normals[0, :], normals[1, :], scale=50)


def plot_kmeans(
    kmeans_learner,
    mesh_distance: float = 0.01,
    limit_to_radius=True,
    ax=None,
    x_lim=None,
    y_lim=None,
    centerlabel=True,
):
    reduced_data = kmeans_learner.data.X[:, : kmeans_learner.data.dimension]

    if x_lim is None:
        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    else:
        x_min, x_max = x_lim
    if y_lim is None:
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    else:
        y_min, y_max = y_lim

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_distance),
        np.arange(y_min, y_max, mesh_distance),
    )

    n_points = xx.shape[0] * xx.shape[1]
    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans_learner.kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    if limit_to_radius:
        value_far = -1
        for label in kmeans_learner.get_feature_labels():

            xx_flat = xx.flatten()
            yy_flat = yy.flatten()

            ind_level = Z == label

            ind = np.arange(xx_flat.shape[0])[ind_level]

            pos = np.array([xx_flat[ind], yy_flat[ind]]).T

            dist = LA.norm(
                pos
                - np.tile(
                    kmeans_learner.kmeans.cluster_centers_[label, :],
                    (np.sum(ind_level), 1),
                ),
                axis=1,
            )
            ind = ind[dist > kmeans_learner.region_radius_]

            Z[ind] = value_far

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    if ax is None:
        _, ax = plt.subplots()

    # ax.clf()
    ax.imshow(
        Z,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        cmap=plt.cm.Paired,
        aspect="auto",
        origin="lower",
    )

    ax.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans_learner.kmeans.cluster_centers_
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="w",
        zorder=10,
    )

    if centerlabel:
        for ii in range(kmeans_learner.get_number_of_features()):
            d_txt = 0.15
            level = kmeans_learner._graph.nodes[ii]["level"]
            ax.text(
                kmeans_learner.kmeans.cluster_centers_[ii, 0] + d_txt,
                kmeans_learner.kmeans.cluster_centers_[ii, 1] + d_txt,
                f"# {ii} @ {level}",
                fontsize=15,
                color="black",
                # color="white",
                zorder=10,
            )

    # Plot attractor
    ax.scatter(
        kmeans_learner.data.attractor[0],
        kmeans_learner.data.attractor[1],
        marker="*",
        s=200,
        color="white",
        zorder=10,
    )


def plot_trajectories(
    ax,
    main_learner,
    it_max=200,
    dt=0.1,
    convergence_margin=1e-3,
    dimension=2,
):
    # Trajectory integration
    data = main_learner.data

    for tt in range(data.start_positions.shape[1]):
        print(f"Doing trajectory {tt}")

        positions = np.zeros((dimension, it_max + 1))

        positions[:, 0] = data.start_positions[:, tt]

        for ii in range(it_max):
            velocity = main_learner.evaluate(positions[:, ii])
            positions[:, ii + 1] = velocity * dt + positions[:, ii]

            if LA.norm(positions[:, ii + 1] - positions[:, ii]) < convergence_margin:
                print(f"Trajectory {tt} has converged at it={ii}.")
                positions = positions[:, : ii + 2]
                break

        ax.plot(positions[0, :], positions[1, :], "r")
        ax.plot(positions[0, 0], positions[1, 0], "or")


def plot_gamma_of_learner(
    main_learner, x_lim, y_lim, hierarchy_passing_gamma=True, fig=None, ax=None
):
    """A local helper function to plot the gamma fields."""
    # from dynamic_obstacle_avoidance.rotational.kmeans_obstacle import KMeansObstacle

    if ax is None:
        if fig is None:
            raise ValueError("Need figure AND axes.")

        fig, ax = plt.subplots()

    levels = np.linspace(1, 21, 51)  # For gamma visualization

    for ii in range(main_learner.kmeans.n_clusters):
        if hierarchy_passing_gamma:
            region_obstacle = kml.create_kmeans_obstacle_from_learner(main_learner, ii)

        else:
            region_obstacle = kml.KMeansObstacle(
                radius=main_learner.region_radius_,
                kmeans=main_learner.kmeans,
                index=ii,
            )

        positions = region_obstacle.evaluate_surface_points()
        ax.plot(positions[0, :], positions[1, :], color="black", linewidth=3.5)

        ff = 1.2
        n_grid = 60
        positions = get_grid_points(
            main_learner.kmeans.cluster_centers_[ii, 0],
            main_learner.region_radius_ * ff,
            main_learner.kmeans.cluster_centers_[ii, 1],
            main_learner.region_radius_ * ff,
            n_points=n_grid,
        )

        gammas = np.zeros(positions.shape[1])
        for jj in range(positions.shape[1]):

            if (
                LA.norm(positions[:, jj] - region_obstacle.center_position)
                > region_obstacle.radius
            ):
                # For nicer visualization, only internally
                continue

            gammas[jj] = region_obstacle.get_gamma(
                positions[:, jj], in_global_frame=True
            )

        cntr = ax.contourf(
            positions[0, :].reshape(n_grid, n_grid),
            positions[1, :].reshape(n_grid, n_grid),
            gammas.reshape(n_grid, n_grid),
            levels=levels,
            # cmap="Blues_r",
            # cmap="magma",
            cmap="pink",
            # alpha=0.7,
        )

    cbar = fig.colorbar(cntr)

    ax.axis("equal")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    return fig, ax


def plot_partial_dynamcs_of_four_clusters(
    visualize=False,
    save_figure=False,
    main_learner=None,
    x_lim=None,
    y_lim=None,
    name="",
):
    # Generate very simple dataset
    # if main_learner is None:
    #     main_learner = create_four_point_datahandler()

    if x_lim is None:
        x_lim = [-2.1, 3.1]
    if y_lim is None:
        y_lim = [-1.1, 3.1]

    if visualize:
        fig_init, axs_init = plt.subplots(2, 2, figsize=(14, 9))
        fig_mod, axs_mod = plt.subplots(2, 2, figsize=(14, 9))

        for ii in range(main_learner.n_clusters):
            ax_ini = axs_init[ii % 2, ii // 2]
            ax_mod = axs_mod[ii % 2, ii // 2]

            main_learner.plot_boundaries(ax=ax_ini, plot_attractor=True)
            main_learner.plot_boundaries(ax=ax_mod, plot_attractor=True)

            ff = 1.05
            n_grid = 10
            positions = get_grid_points(
                main_learner.kmeans.cluster_centers_[ii, 0],
                main_learner.region_radius_ * ff,
                main_learner.kmeans.cluster_centers_[ii, 1],
                main_learner.region_radius_ * ff,
                n_points=n_grid,
            )
            initial_velocities = np.zeros_like(positions)
            modulated_velocities = np.zeros_like(positions)

            for jj in range(positions.shape[1]):
                if not main_learner.region_obstacles[ii].is_inside(
                    positions[:, jj], in_global_frame=True
                ):
                    continue

                initial_velocities[:, jj] = main_learner._dynamics[ii].evaluate(
                    positions[:, jj]
                )

                modulated_velocities[:, jj] = obstacle_avoidance_rotational(
                    positions[:, jj],
                    initial_velocities[:, jj],
                    [main_learner.region_obstacles[ii]],
                    convergence_velocity=main_learner._dynamics[
                        ii
                    ].evaluate_convergence_velocity(positions[:, jj]),
                    sticky_surface=False,
                )

            ax_ini.quiver(
                positions[0, :],
                positions[1, :],
                initial_velocities[0, :],
                initial_velocities[1, :],
                # color="blue",
                scale=15,
            )
            ax_ini.axis("equal")
            ax_mod.quiver(
                positions[0, :],
                positions[1, :],
                modulated_velocities[0, :],
                modulated_velocities[1, :],
                # color="red",
                scale=15,
            )
            ax_mod.axis("equal")

        if save_figure:
            fig_name = "initial_local_velocities" + "_" + name
            fig_init.savefig("figures/" + fig_name + fig_type, bbox_inches="tight")

            fig_name = "modulated_local_velocities" + "_" + name
            fig_mod.savefig("figures/" + fig_name + fig_type, bbox_inches="tight")
