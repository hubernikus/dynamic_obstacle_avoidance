#!/USSR/bin/python3
"""
Dynamic Simulation - Obstacle Avoidance Algorithm
"""
# Author LukasHuber
# Created 2018-05-24

import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA
import copy
import warnings

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *

plt.ion()
plt.close("all")


def transform_to_directionSpace(reference_direction, directions, normalize=True):
    ind_nonzero = weights > 0

    reference_direction = np.copy(reference_direction)
    directions = directions[:, ind_nonzero]
    weights = weights[ind_nonzero]

    n_directions = weights.shape[0]
    if n_directions <= 1:
        return directions[:, 0]

    dim = np.array(reference_direction).shape[0]
    if dim > 2:
        warnings.warn("Implement for higher dimensions.")

    if normalize:
        norm_refDir = LA.norm(reference_direction)
        if norm_refDir:  # nonzero
            reference_direction /= norm_refDir

        norm_dir = LA.norm(directions, axis=0)
        ind_nonzero = norm_dir > 0
        if ind_nonzero.shape[0]:
            directions[:, ind_nonzero] = directions[:, ind_nonzero] / np.tile(
                norm_dir[ind_nonzero], (dim, 1)
            )

    OrthogonalBasisMatrix = get_orthogonal_basis(reference_direction)

    directions_referenceSpace = np.zeros(np.shape(directions))
    for ii in range(np.array(directions).shape[1]):
        directions_referenceSpace[:, ii] = OrthogonalBasisMatrix.T.dot(
            directions[:, ii]
        )

    directions_directionSpace = directions_referenceSpace[1:, :]
    return directions_directionSpace


def windup_smoothening(angle_windup, angle):
    # Make it a oneliner? lambda-function
    # Correct the integration error
    num_windups = np.round((angle_windup - angle) / (2 * pi))

    angle_windup = 2 * pi * num_windups + angle
    return angle_windup


def ds_init(x, attractor=np.array([0, 0]), max_vel=0.5, slow_down_region=0.5):
    vel = attractor - x

    dist = np.linalg.norm(vel)
    if dist < slow_down_region:
        max_vel = max_vel * dist / slow_down_region

    norm_vel = np.linalg.norm(vel)
    if norm_vel > max_vel:
        vel = vel / norm_vel * max_vel

    return vel


if (__name__) == "__main__":
    r_boundary = 2
    # orientation_object=0
    case = 1

    obstacles = ObstacleContainer()

    if True:
        # Couple of pedestrian ellipses
        # obstacles.append(Ellipse(axes_length=[0.3, 0.8]))
        obstacles.append(Ellipse(axes_length=[2.0, 0.8]))
        obstacles[-1].orientation = 30.0 / 180 * pi
        obstacles[-1].center_position = [-1, 0.2]

        # obstacles.append(copy.deepcopy(obstacles[0]))
        # obstacles[-1].orientation = 30./180*pi
        # obstacles[-1].center_position = [0, 2]

        # obstacles.append(copy.deepcopy(obstacles[0]))
        # obstacles[-1].orientation = 40./180*pi
        # obstacles[-1].center_position = [1, -1]

        for ii in range(len(obstacles)):
            obstacles[ii].tail_effect = True

        obstacles.reset_clusters()
        # obstacles._index_families = np.arange(1)
        # obstacles._unique_families = np.arange(1)

    n_grid = 40

    xx_vals = np.linspace(-5, 5, n_grid)
    yy_vals = np.linspace(-4, 4, n_grid)

    dim = 2

    pos_init = np.zeros((dim, n_grid, n_grid))
    pos_surface = np.zeros((dim, n_grid, n_grid))
    gamma = np.zeros((n_grid, n_grid))

    normal = np.zeros((dim, n_grid, n_grid))

    oo = 0

    plt.close("all")
    # plt.figure()
    fig, ax = plt.subplots()

    for ix in range(n_grid):
        for iy in range(n_grid):

            position = np.array([xx_vals[ix], yy_vals[iy]])

            print("pos init", position)

            mag, ang = transform_cartesian2polar(
                position, obstacles[oo].global_reference_point
            )

            local_radius = obstacles[oo].get_radius_of_angle(ang, in_global_frame=True)

            pos_init[:, ix, iy] = position
            pos_surface[:, ix, iy] = (
                np.array([np.cos(ang), np.sin(ang)]) * local_radius
                + obstacles[oo].center_position
            )
            gamma[ix, iy] = obstacles[oo].get_gamma(position, in_global_frame=True)

            gamma[ix, iy] = np.min([gamma[ix, iy], 2])

            normal[:, ix, iy] = obstacles[oo].get_normal_direction(
                position, in_global_frame=True
            )

    plt.plot(pos_surface[0, :, :], pos_surface[1, :, :], "g.")
    plt.quiver(
        pos_surface[0, :, :],
        pos_surface[1, :, :],
        normal[0, :, :],
        normal[1, :, :],
        color="g",
    )

    # plt.plot(pos_init[0, :, :], pos_init[1, :, :], 'k.')

    for oo in range(len(obstacles)):
        obstacles[oo].draw_obstacle(numPoints=100)
        plt.plot(obstacles[oo].x_obs[:, 0], obstacles[oo].x_obs[:, 1], "k")
        plt.axis("equal")
        plt.plot(
            obstacles[oo].global_reference_point[0],
            obstacles[oo].global_reference_point[1],
            "k+",
        )

    if False:
        cs = ax.contourf(
            pos_init[0, :, :], pos_init[1, :, :], gamma, cmap=plt.cm.coolwarm, alpha=0.8
        )
        cbar = fig.colorbar(cs)
    plt.grid()
    plt.show()
