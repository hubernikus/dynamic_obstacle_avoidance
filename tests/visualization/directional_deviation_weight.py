# Visual of 'weighting' function to help with debugging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

plt.ion()


def directional_deviation_weight(save_figure=False):
    from dynamic_obstacle_avoidance.avoidance.rotation import (
        _get_directional_deviation_weight,
    )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    n_grid = 20
    dist0 = np.linspace(1e-6, 1 - 1e-6, n_grid)
    weight = np.linspace(1e-6, 1 - 1e-6, n_grid)
    gamma = 1.0 / weight

    val = np.zeros((n_grid, n_grid))
    for ii, d_weight in enumerate(dist0):
        for jj, ww in enumerate(weight):
            val[ii, jj] = _get_directional_deviation_weight(d_weight, ww)

    # val = weight ** (1.0/(pow_factor*dist0))

    weight_mesh, dist0_mesh = np.meshgrid(weight, dist0)
    surf = ax.plot_surface(
        dist0_mesh, weight_mesh, val, cmap=cm.YlGnBu, linewidth=0.2, edgecolors="k"
    )
    # antialiased=False)
    import matplotlib as mpl

    mpl.rc("font", family="Times New Roman")
    ax.set_xlabel(r"Relative Rotation $\tilde d (\xi)$")
    ax.set_ylabel(r"Weight $1/\Gamma(\xi)$")
    ax.set_zlabel(r"Rotational Weights $w_r(\Gamma, \tilde{d})$")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    if save_figure:
        figure_name = "rotational_weight_with_power_" + str(int(pow_factor))
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    directional_deviation_weight(save_figure=False)
