#!/USSR/bin/python3
""" Test different weighting functions for 'repulsion-displacement'. """
# NOTE: this is currently discontinuied.

import numpy as np
import matplotlib.pyplot as plt


# Define variables
r0 = 0.1
r1 = 1.0

delta_d = 0.01
d = r0 + r1 + delta_d

d_m_max_0 = r0
d_m_max_1 = r1


def get_gamma(x, ind_obs):
    if ind_obs == 0:
        return x - r0 + 1
        # return x/r0

    if ind_obs == 1:
        return x + d - r1 + 1
        # return (x+d)/r1


def weight_abs(x, ind_obs, pow_gamma=1):
    gamma = get_gamma(x, ind_obs)

    return (gamma**pow_gamma - 1) / gamma


def weight_slider(x, ind_obs, weight_min=1, weight_pow=2):
    gammas = np.zeros(2)

    for ii in range(gammas.shape[0]):
        gammas[ii] = get_gamma(x, ii)
    # weights[1] = weight_abs(x, 1)-weight_min
    if any(gammas == 1):
        if gammas[ind_obs] == 1:
            return 1
        else:
            return 0

    gammas = gammas - weight_min
    gammas = (1 / gammas) ** weight_pow
    # weights = [1.0/(x-r1+1), 1.0/(x+d-r1+1)]

    return gammas[ind_obs] / np.sum(gammas)


if (__name__) == "__main__":
    n_points = 1000

    x_range = [r0, 5]
    x_vals = np.linspace(x_range[0], x_range[1], n_points)
    dx = (x_range[1] - x_range[0]) / n_points

    # y_vals = np.zeros(x_range)
    delta_m = np.zeros(n_points)
    dx_delta_m = np.zeros(n_points - 1)

    w_0 = np.zeros(n_points)
    w_1 = np.zeros(n_points)
    alpha = np.zeros(n_points)

    weight_pow = 1

    for ii in range(n_points):
        w_0[ii] = weight_abs(x_vals[ii], 0)
        w_1[ii] = weight_abs(x_vals[ii], 1)

        alpha[ii] = weight_slider(x_vals[ii], 0)

        delta_m[ii] = d_m_max_0 * w_0[ii] ** weight_pow * alpha[ii] + d_m_max_1 * w_1[
            ii
        ] ** weight_pow * (1 - alpha[ii])

    dx_delta_m = (delta_m[1:] - delta_m[:-1]) / dx

    # Create plot
    plt.ion()
    plt.close("all")

    plt.figure()
    plt.plot(x_vals, delta_m, label="delta m")
    plt.plot(x_vals, -delta_m + x_vals, label="m(x)")
    plt.plot(x_vals, w_0, label="Weight 0")
    plt.plot(x_vals, w_1, label="Weight 1")
    plt.plot(x_vals, alpha, label="Alpha 0")
    # plt.plot(x_vals, 1-alpha, label="Alpha 1")
    plt.plot(x_vals[:-1] + dx / 2, dx_delta_m, label="d delta_m/dx")
    plt.plot(x_vals[:-1] + dx / 2, 1 - dx_delta_m, label="dm/dx")

    # plt.ylim([-.01, 2])

    plt.legend()
    plt.grid(True)
    plt.show()
