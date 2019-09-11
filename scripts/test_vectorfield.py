########################################################################
# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
import matplotlib.pyplot as plt

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *

########################################################################
# Chose the option you want to run as a number in the option list (integer from -2 to 10)
options = [-3]

N_resol = 10
saveFigures=False


x_lim = [-0.8,10.1]
y_lim = [-4,4]

# x_lim = [-10,100]
# y_lim = [-40,40]

xAttractor = [0,0]

obs = [] # create empty obstacle list
        
a=[0.5, 1.5]
p=[1,1]
x0=[5.5, 0.8]
th_r=-40/180*pi
sf=1
vel = [0, 0]
# obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=vel))

a=[0.5, 1.5]
p=[1,1]
x0=[5.5, -0.8]
th_r=40/180*pi
sf=1
vel = [0, 0]
obs.append(Obstacle(a=a, p=p, x0=x0,th_r=th_r, sf=sf, xd=vel))

obs[0].set_reference_point([0.3, 3], in_global_frame=False)

# obs[1].set_reference_point(obs[0].get_reference_point(in_global_frame=True), in_global_frame=True)

Simulation_vectorFields(x_lim, y_lim, N_resol, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_avoidanceCircle', noTicks=False)

n_points = 6
points_init = np.vstack((np.ones(n_points)*x_lim[1],
                         np.linspace(y_lim[0], y_lim[1], n_points)))

points_init = points_init[:, 1:-1]
# Simulation_vectorFields(x_lim, y_lim, N_resol, obs, xAttractor=xAttractor, saveFigure=saveFigures, figName='linearSystem_avoidanceCircle', noTicks=False, draw_vectorField=False, points_init=points_init)


# dynamicalSystem=linearAttractor
# pos
# xd_init = dynamicalSystem(pos, x0=xAttractor)
# xd_mod[:,ix,iy] = obs_avoidance(pos, xd_init[:,ix,iy], obs)

