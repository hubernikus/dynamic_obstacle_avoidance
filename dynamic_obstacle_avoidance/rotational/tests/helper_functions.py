import numpy as np

import matplotlib.pyplot as plt


def plot_region_dynamics(main_learner, x_lim, y_lim, n_grid=20):
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

    fig, ax = plt.subplots(figsize=(12, 9))
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
