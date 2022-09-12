import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.rotational import kmeans_motion_learner as kml


def plot_region_dynamics(main_learner, x_lim, y_lim, n_grid=20, ax=None):
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
            zorder=20,
        )

    centroids = kmeans_learner.kmeans.cluster_centers_
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=169,
        linewidths=3,
        color="black",
        zorder=10,
    )

    if plot_attractor:
        ax.scatter(
            kmeans_learner.data.attractor[0],
            kmeans_learner.data.attractor[1],
            marker="*",
            s=200,
            color="black",
            zorder=10,
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
                f"{ii} @ {level}",
                fontsize=15,
                # color="black",
                color="white",
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
