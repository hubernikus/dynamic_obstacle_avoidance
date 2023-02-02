""" Test a specific function

This file is to evaluate (and justify) the choice of a distance modulation.
"""

import numpy as np

import matplotlib.pyplot as plt


def main():
    plt.ion()

    n_grid = 100

    abs_err = 1e-6
    gamma_values = np.linspace(1 + abs_err, 1000, n_grid)
    inv_gammas = np.linspace(1 - abs_err, 0 + abs_err, n_grid)

    pp_values = np.linspace(0 + abs_err, 1 - abs_err, n_grid)

    func_values = np.zeros((n_grid, n_grid))

    for gg, inv_gamma in enumerate(inv_gammas):
        for pp, pp_val in enumerate(pp_values):

            # func_values[gg, pp] = (1 - inv_gamma) ** (pp_val)

            func_values[gg, pp] = (pp_val) ** (1 - inv_gamma)

    # syntax for 3-D plotting
    ax = plt.axes(projection="3d")
    ax.plot_surface(
        np.outer(gamma_values, np.ones(n_grid)),
        # np.outer(1.0 / inv_gammas, np.ones(n_grid)),
        np.outer(pp_values, np.ones(n_grid)).T,
        func_values
        # , cmap ='viridis', edgecolor =''
    )
    ax.set_xlabel("Inv-Gamma")
    ax.set_ylabel("pp")
    plt.show()
    breakpoint()


if (__name__) == "__main__":
    plt.close("all")
    main()
