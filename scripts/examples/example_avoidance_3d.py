#### !/usr/bin/python3
'''
Dynamic Simulation - Obstacle Avoidance Algorithm
'''

__author__ = "LukasHuber"
__date__ = "2020-07-21"


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

plt.close('all')

import numpy as np

import time

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import get_linear_ds, make_velocity_constant, linear_ds_max_vel, linearAttractor_const
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import obs_avoidance_interpolation_moving
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import GradientContainer
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import Ellipse
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import Obstacle
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *

# from dynamic_obstacle_avoidance.visualization.visualization_3d_level import Visualization3dLevel

MAX_SPEED = 0.3
class ObstacleAvoidance3d():
    def __init__(self):
        self.local_attractor = np.array([2, 1.5, 0])
    
        self.obstacle_list = GradientContainer()
        self.obstacle_list.append(Ellipse(center_position=np.array([0, 0, 0]),
                                          axes_length=np.array([2, 1, 1]),
                                          name="ellipse_obstacle")
        )

        # ROS INITALIZATION AND SETUP
        
        self.agent = Obstacle(name="QOLO",
                              center_position=np.array([-2, -1.4, 0.3]),
        ) # Define obstacle

        self.dt = 0.1

    def run(self, loop_max=100):

        loop_count = 0

        position_list = np.zeros((self.agent.dim, loop_max+1))
        while True:
            # PUT ROS LOOP HERE
            linear_ds = linear_ds_max_vel(self.agent.position,
                                          attractor=self.local_attractor, vel_max=MAX_SPEED)

            modulated_ds = obs_avoidance_interpolation_moving(
                self.agent.position, linear_ds, self.obstacle_list, 
                tangent_eigenvalue_isometric=False, repulsive_obstacle=False)

            self.agent.position = self.agent.position + modulated_ds*self.dt

            position_list[:, loop_count] = self.agent.position

            loop_count += 1
   
            if loop_count > loop_max:
                print("Maximum iteration reached.")
                break
            
        self.visualize_modulation(trajectory=position_list)


    def visualize_modulation(self, trajectory=None):
        
        fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
        ax = fig.add_subplot(111, projection='3d')
        
        if not trajectory is None:
            ax.plot(trajectory[0, :], trajectory[1, :], trajectory[2, :])
        
        # Plot:
        for obstacle in self.obstacle_list:
            surface_points = obstacle.draw_obstacle(numPoints=100)
            
            ax.plot_surface(surface_points[0],
                            surface_points[1],
                            surface_points[2],
                            rstride=4, cstride=4, color=[0.3, 0.3, 0.3], alpha=0.3)

        for axis in 'xyz':
            # getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
            getattr(ax, 'set_{}lim'.format(axis))((-3, 3))
    


if __name__=="__main__":
    plt.close('all')
    plt.ion()
    RobotController = ObstacleAvoidance3d()
    RobotController.run()
        
