#!/USSR/bin/python3.9
""" Test the directional space.
Creates graph of the rotational/directional weight."""
# Author: Lukas Huber
# Email: lukas.huber@epfl.ch
# Created: 2021-05-18
# License: BSD (c) 2021

import unittest
import numpy as np


def weight_invgamma(inv_gamma, pow_fac):
    return weight**pow_fac


def weight_dist(dist, inv_gamma, pow_fac):
    return 1.0 / (1 - dist / (1 - inv_gamma))


class TestRotational(unittest.TestCase):
    def rotation_weight(self, save_figure=True):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        import numpy as np

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        n_grid = 20
        dist0 = np.linspace(1e-6, 1 - 1e-6, n_grid)
        weight = np.linspace(1e-6, 1 - 1e-6, n_grid)
        weight, dist0 = np.meshgrid(weight, dist0)
        gamma = 1.0 / weight

        # Make data.
        pow_factor = 3.0
        val = weight ** (1.0 / (pow_factor * dist0))
        surf = ax.plot_surface(
            dist0, weight, val, cmap=cm.YlGnBu, linewidth=0.2, edgecolors="k"
        )
        # antialiased=False)
        # breakpoint()
        import matplotlib as mpl

        mpl.rc("font", family="Times New Roman")
        ax.set_xlabel(r"Relative Rotation $\tilde d (\xi)$")
        ax.set_ylabel(r"Weight $1/\Gamma(\xi)$")
        ax.set_zlabel(r"Rotational Weights $w_r(\Gamma, \tilde{d})$")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])
        print("Done")

        if save_figure:
            figure_name = "rotational_weight_with_power_" + str(int(pow_factor))
            plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

        plt.ion()
        plt.show()
        # plt.close('all')


if __name__ == "__main__":
    visualize = True
    if visualize:
        Tester = TestRotational()
        Tester.rotation_weight(save_figure=True)
