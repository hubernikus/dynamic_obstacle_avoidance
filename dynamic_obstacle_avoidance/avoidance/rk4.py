""" Runge Kutta 4 algorithm for general obstacle avoidance"""
import numpy as np

from dynamic_obstacle_avoidance.avoidance import (
    obs_avoidance_interpolation_moving,
)
from dynamic_obstacle_avoidance.avoidance import (
    obs_avoidance_nonlinear_hirarchy,
)


def obs_avoidance_rk4(dt, x, obs, obs_avoidance, ds, x0=False):
    """Fourth order integration of obstacle avoidance differential equation
    Paramters
    ---------
    dt: time step [s]
    x: position
    obs: obstacle list
    obs_avoidance: Obstacle Avoidance algorithm
    ds: initial dynamics

    Returns
    -------
    Runge-Kutta step of the obstacle avoidance
    """
    # NOTE: The movement of the obstacle is considered as small, hence
    # position and movement changed are not considered. This will be fixed in future iterations.
    # TODO: More General Implementation (find library)

    if type(x0) == bool:
        x0 = np.zeros(np.array(x).shape[0])

    # k1
    xd = ds(x, x0)
    # xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x, xd, obs)
    k1 = dt * xd

    # k2
    xd = ds(x + 0.5 * k1, x0)
    # xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x + 0.5 * k1, xd, obs)
    k2 = dt * xd

    # k3
    xd = ds(x + 0.5 * k2, x0)
    # xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x + 0.5 * k2, xd, obs)

    k3 = dt * xd

    # k4
    xd = ds(x + k3, x0)
    # xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x + k3, xd, obs)
    k4 = dt * xd

    # x final
    # Maybe: directional sum? Can this be done?
    x = x + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)  # + O(dt^5)

    return x


def obs_avoidance_rungeKutta(
    dt, x, obs, obs_avoidance, ds_init, center_position=False, order=4
):
    # Fourth order integration of obstacle avoidance differential equation
    # NOTE: The movement of the obstacle is considered as small, hence position and movement changed are not considered. This will be fixed in future iterations.
    dim = np.array(x).shape[0]
    if type(center_position) == bool:
        # TODO --- no default value
        center_position = np.zeros(dim)

    if order == 1:
        step_fraction = np.array[1]
        rk_fac = np.array[1]
    elif order == 4:
        step_fraction = np.array([0, 0.5, 0.5, 1.0])
        rk_fac = np.array([1, 2, 2, 1]) / 6
    else:
        print("WARNING: implement rk with order {}".format(order))
        step_fraction = np.array([1])
        rk_fac = np.array([1])

    k = np.zeros((dim, len(step_fraction) + 1))

    for ii in range(len(step_fraction)):
        # TODO remove after debugging
        xd, m_x = obs_avoidance(x + k[:, ii] * step_fraction[ii], ds_init, obs)

        xd = xd[:, -1]
        # xd = velConst_attr(x, xd, center_position)
        k[:, ii + 1] = dt * xd

    x = x + np.sum(np.tile(rk_fac, (dim, 1)) * k[:, 1:], axis=1)  # + O(dt^5)

    return x
