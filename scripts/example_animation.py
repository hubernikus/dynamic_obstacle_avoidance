#!/usr/bin/python3


## General classes
import numpy as np
from numpy import pi
import copy

from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import FloatSlider, IntSlider
import ipywidgets as widgetsb

# Unocmment in case of error
import sys

# sys.path.append('/home/jovyan/src/')
# sys.path.append('/home/lukas/Code/ObstacleAvoidance/dynamic_obstacle_avoidance_python_linear/src/')

from dynamic_obstacle_avoidance.dynamical_system import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import Obstacle
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
)
from dynamic_obstacle_avoidance.visualization.animated_simulation import *
from dynamic_obstacle_avoidance.visualization.widget_function_vectorfield import *

x_range, y_range = [-16, 16], [-2, 18]
x_init = samplePointsAtBorder(number_of_points=10, x_range=x_range, y_range=y_range)

x_init = np.zeros((2, 1))
x_init[:, 0] = [8, 1]

obs = []
x0 = [3, 4]
a = [3, 3]
p = [1, 1]
th_r = 0 / 180 * pi
vel = [0, 0]

obs.append(Obstacle(a=a, p=p, x0=x0, th_r=th_r, sf=1))


# %matplotlib notebook
ani = run_animation(
    x_init,
    obs=obs,
    x_range=x_range,
    y_range=y_range,
    dt=0.005,
    N_simuMax=1000,
    convergenceMargin=0.3,
    sleepPeriod=0.001,
    RK4_int=True,
    hide_ticks=False,
    return_animationObject=True,
)
plt.ion()
ani.show()

input("Get input \n ")
