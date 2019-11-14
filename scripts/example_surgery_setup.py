#!/usr/bin/python3

'''
Dynamic Simulation - Obstacle Avoidance Algorithm

@author LukasHuber
@date 2018-05-24
'''

import sys

import numpy as np
from numpy import pi

import time

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *
from dynamic_obstacle_avoidance.visualization.animated_simulation_3d import samplePointsAtBorder
from dynamic_obstacle_avoidance.visualization.animated_simulation_3d import run_animation

print(' ----- Script <<dynamic simulation>> started. ----- ')
#############################################################
# Choose a simulation between 0 and 12
simulationNumber = 0

saveFigures = False
#############################################################

def main(simulationNumber=0, saveFigures=False):
    if simulationNumber==0:
        N = 10
        x_init = np.vstack((np.ones(N)*20,
                            np.linspace(-10,10,num=N) ))

        ### Create obstacle 
        obs = []
        a1 = 0.05
        a2 = 0.20
        d_a = (a2-a1)/2
        l = 0.30
        points = np.array([[0, 0, 0],
                           [0, a1, 0,],
                           [0, a1, a1],
                           [0, 0, a1],
                           [l, -d_a, -d_a],
                           [l, a2-d_a, -d_a],
                           [l, a2-d_a, a2-d_a],
                           [0, -d_a, a2-d_a]).T

        ind_tiles = np.array([[0,1,2,3],
                              [4,5,6,7],
                              [0,1,4,5],
                              [1,2,5,6],
                              [2,3,6,7],
                              [3,0,7,5]])
                          
        obs.append(DynamicBoundariesPolygon(edge_points=points, ind_tiles=ind_tiles, th_r=0))

        x_range = [-l ,l*2]
        y_range = [-a2,a2*2]
        z_Range = [-a2, a2*2]

        attractorPos = [0,0]

        animationName = 'surgery_simulation.mp4'
                          
        run_animation_3d(x_init, obs, x_range=x_range, y_range=y_range, dt=0.05, N_simuMax=1040, convergenceMargin=0.3, sleepPeriod=0.01,attractorPos=attractorPos, animationName=animationName, saveFigure=saveFigures)

    print('\n\n---- Script finished ---- \n\n')    


if __name__ == "__main__":
    if len(sys.argv) > 1:
        simulationNumber = sys.argv[1]

    if len(sys.argv) > 2:
        saveFigures = sys.argv[2]

    main(simulationNumber=simulationNumber, saveFigures=saveFigures)
