#!/usr/bin/python3

"""
Dynamic Simulation - Obstacle Avoidance Algorithm

@author LukasHuber
@date 2018-05-24

"""

import sys

import numpy as np
from numpy import pi

import datetime

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import (
    ObstacleContainer,
)

# from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import Polygon
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *
from dynamic_obstacle_avoidance.visualization.animated_multibody import (
    run_animation_multibody,
)

print(" ----- Script <<dynamic simulation>> started. ----- ")
#############################################################
# Choose a simulation between 0 and 12
simulationNumber = 1

saveFigures = True
#############################################################


def main(simulationNumber=0, saveFigures=False):
    if simulationNumber == 1:

        rad_obs_ring = 20
        x_range = y_range = [-rad_obs_ring - 2.0, rad_obs_ring + 2.0]

        n_obs = 20

        center_ring = np.array([0, 0])

        delta_phi = 2 * pi / n_obs * np.arange(n_obs)
        # N = n_obs

        xx = np.random.rand(n_obs) * (x_range[1] - x_range[0]) + x_range[0]
        yy = np.random.rand(n_obs) * (y_range[1] - y_range[0]) + y_range[0]
        x_init = np.vstack((xx, yy))
        # x_init = np.vstack(( np.cos(delta_phi), np.sin(delta_phi) ))*(rad_obs_ring+1) \
        # + np.tile(center_ring, (n_obs, 1)).T

        # x_init[:, np.random.permutation(n_obs)]

        obs = []

        # pos_obs = np.vstack(( np.cos(delta_phi), np.sin(delta_phi) ))*rad_obs_ring \
        # + np.tile(center_ring, (n_obs, 1)).T
        pos_obs = x_init
        #
        dphi = 5.0 / 180 * pi
        delta_phi = delta_phi + dphi  # slight initial rotation
        attr_list = (
            -np.vstack((np.cos(delta_phi), np.sin(delta_phi))) * rad_obs_ring
            + np.tile(center_ring, (n_obs, 1)).T
        )

        rad_robot = 1.0
        for oo in range(n_obs):
            obs.append(
                Ellipse(
                    axes_length=[rad_robot, rad_robot],
                    center_position=pos_obs[:, oo],
                    orientation=0,
                    margin_absolut=rad_robot,
                )
            )
            # orientation=-25*pi/180, margin_absolut=0.0 ))

        obs[0].draw_obstacle()

        run_animation_multibody(
            x_init,
            obs,
            x_range=x_range,
            y_range=y_range,
            dt=0.020,
            N_simuMax=800,
            convergenceMargin=0.3,
            sleepPeriod=0.001,
            attractorPos=attr_list,
            velocity_max=3.0,
            saveFigure=saveFigures,
            animationName="multibody_",
        )


if (__name__) == "__main__":
    if len(sys.argv) > 1 and not (sys.argv[1]) == "-i":
        # if len(sys.argv)>=2 and
        # del sys.argv[1]
        simulationNumber = sys.argv[1]

        if len(sys.argv) > 2:
            saveFigures = sys.argv[2]

    try:
        main(simulationNumber=simulationNumber, saveFigures=saveFigures)
    except:
        raise
