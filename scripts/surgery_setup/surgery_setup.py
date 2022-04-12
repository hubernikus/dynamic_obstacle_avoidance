#### !/usr/bin/python3
"""
Dynamic Simulation - Obstacle Avoidance Algorithm
"""

__author__ = "LukasHuber"
__date__ = "2020-07-21"

import numpy as np
from numpy import pi
import time

from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import (
    ObstacleContainer,
)

# from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import GradientContainer
# from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_ import Ellipse
from dynamic_obstacle_avoidance.obstacle_avoidance.dynamic_boundaries_polygon import (
    DynamicBoundariesPolygon,
)
from dynamic_obstacle_avoidance.visualization.visualization_3d_level import (
    Visualization3dLevel,
)

print(" ----- Script <<surgery_setup>> started. ----- ")

#############################################################
plt.ion()
# Choose a simulation between 0 and 12
simulationNumber = 0

saveFigures = False

#############################################################


def main(simulationNumber=0, saveFigures=False):
    if simulationNumber == 0:
        N = 10
        x_init = np.vstack((np.ones(N) * 20, np.linspace(-10, 10, num=N)))

        ### Create obstacle ###
        obs = []
        a1 = 0.05
        a2 = 0.20
        d_a = (a2 - a1) / 2

        l = 0.30
        points = np.array(
            [
                [-a1, -a1, 0],
                [a1, -a1, 0],
                [a1, a1, 0],
                [-a1, a1, 0],
                [-a2, -a2, l],
                [a2, -a2, l],
                [a2, a2, l],
                [-a2, a2, l],
            ]
        ).T

        indeces_of_tiles = np.array(
            [
                # [0,1,2,3], # Bottom Part
                # [4,5,6,7], # Lid
                [0, 1, 4, 5],
                [1, 2, 5, 6],
                [2, 3, 6, 7],
                [3, 0, 7, 5],
            ]
        )

        obs = ObstacleContainer([DynamicBoundariesPolygon(is_surgery_setup=True)])

        x_range = [-0.2, 0.2]
        y_range = [-0.2, 0.2]
        z_range = [-a2, a2 * 2]

        attractorPos = np.array([0, 0, 0])
        eanimationName = "surgery_simulation.mp4"

        inflation_parameter = np.ones(4) * 0.00
        # inflation_parameter = [0, 0.02, 0.02, 0.02]
        # inflation_parameter = [0.02, 0, 0, 0]

        # pos = np.array([0.06, 0.05, 0.01])
        # gamma = obs[0].get_gamma(pos, in_global_frame=True)
        # import pdb; pdb.set_trace()

        static_simulation = True
        if static_simulation:
            obs[0].inflation_parameter = inflation_parameter

            visualizer = Visualization3dLevel(
                obs=obs, x_range=x_range, y_range=y_range, z_range=z_range
            )

            # obs[0].draw_obstacle(z_val=0.1)
            visualizer.vectorfield2d(save_figure=True, z_value=0.01)
            import pdb

            pdb.set_trace()
        elif False:
            x_init = np.array([[-0.1, 0.1, 0.3]])
            visualizer = Visualization3dLevel(
                obs=obs, x_range=x_range, y_range=y_range, z_range=0
            )
            visualizer.animate2d(x_init, attractorPos)

    print("\n\n---- Script finished ---- \n\n")
    # import pdb; pdb.set_trace() ### DEBUG ###


if (__name__) == "__main__":

    # if len(sys.argv) > 1:
    # simulationNumber = int(sys.argv[1])

    # if len(sys.argv) > 2:
    # saveFigures = bool(int(sys.argv[2]))

    main(simulationNumber=simulationNumber, saveFigures=saveFigures)


print(" ----- Script <<surgery_setup>> finished 2. ----- ")
