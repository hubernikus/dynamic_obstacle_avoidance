#!/USSR/bin/python3

''' Script to show lab environment on computer '''

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import Ellipse
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import *
from dynamic_obstacle_avoidance.obstacle_avoidance.comparison_algorithms import obs_avoidance_potential_field, obs_avoidance_orthogonal_moving
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import obs_avoidance_interpolation_moving
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import GradientContainer

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

__author__ = "LukasHuber"
__date__ = "2020-01-15"
__email__ = "lukas.huber@epfl.ch"

plt.close('all')
plt.ion()

class DynamicEllipse(Ellipse):
    
    # Movement is in the brownian motion style
    max_velocity = 1.0
    max_angular_velocity = 1.0
    
    def __init__(self, x_range, y_range):
        self.x_range = x_range
        self.y_range = y_range
        
        # Random creation
        axis_min = [0.2, 0.8]
        axis_max = [1.2, 1.5]

        freq_oscilation = None

        position = np.random.rand(2)
        position[0] = position[0]*(x_range[1] - x_range[0]) + x_range[0]
        position[1] = position[0]*(y_range[1] - y_range[0]) + y_range[0]

        vel_dir = np.random.uniform(0, 2*np.pi)
        velocity = np.random.uniform(high=self.max_velocity) * np.array([np.cos(vel_dir), np.sin(vel_dir)])

        angular_velocity = np.random.uniform(low=-self.max_angular_velocity, high=-self.max_angular_velocity)

        orientation = 0
        axes_length = [1, 2]
        
        super().__init__(
            center_position=position,
            orientation=orientation,
            axes_length=axes_length
        )
        
    def update_step(self, time_step=0.1, acceleration_rand=0.1, expansion_acc_rand=0.1):
        ''' Update velocity & expansion.'''
        # TODO: update velocity
        self.position = self.position + time_step * self.velocity
        self.orientation = self.orientation + time_step * self.angular_velocity
        pass
    
    def get_surface_velocity(self):
        pass
    
    
def compare_algorithms_random(max_it=1000, delta_time=0.1, max_num_agents=5):
    ''' Compare the algorithms with a random environment setup. '''
    x_range = [-15, 15]
    y_range = [-10, 10]

    # func = compare_algorithms_interpolation
    # xd = obs_avoidance_interpolation_moving()
    attractor_position = np.array([9.5, 0])

    obs_list = GradientContainer()
    num_agents = np.random.randint(low=0, high=max_num_agents)
    num_agents = 1
    for oo in range(num_agents):
        obs_list.append(DynamicEllipse(x_range=x_range,
                                       y_range=y_range))

    import pdb; pdb.set_trace()
    
    fig_num = 1001
    fig, ax = plt.subplots(num=fig_num)
    
    for ii in range(max_it):
        for obs in obs_list:
            # obs.update()
            obs.draw_obstacle()

        Simulation_vectorFields(
            x_range, y_range,  obs=obs_list,
            xAttractor=attractor_position,
            saveFigure=False,
            obs_avoidance_func=obs_avoidance_interpolation_moving,
            # noTicks=False, automatic_reference_point=True,
            # show_streamplot=False,
            draw_vectorField=False, show_streamplot=False,
            fig_and_ax_handle=(fig, ax),
        )

        if not plt.fignum_exists(fig_num):
            print(f'Simulation ended with closing of figure')
            plt.pause(0.01)
            plt.close('all')
            break

        plt.pause(0.1)
        
    
def compare_algorithms_plot():
    # create empty obstacle list
    obs = GradientContainer() 

    obs.append(
        Ellipse(
            center_position=[3.5, 0.4],
            orientation=30./180.*pi,
            axes_length=[1.2, 2.0]
        )
    )
    
    x_lim = [-0.1, 11]
    y_lim = [-4.5, 4.5]
    
    xAttractor = np.array([8., -0.1])
    # x_lim, y_lim = [-0, 7.1], [-0.1, 7.1]

    num_point_grid = 100

    if False:
        Simulation_vectorFields(
            x_lim, y_lim,  obs=obs,
            xAttractor=xAttractor, saveFigure=False,
            figName='compare_algorithms_interpolation',
            obs_avoidance_func=obs_avoidance_interpolation_moving,
            # noTicks=False, automatic_reference_point=True,
            # show_streamplot=False,
            point_grid=num_point_grid
        )

    vel = np.array([1.0, 0])
    pos = np.array([0.0, 1.0])
    vel_mod = obs_avoidance_interpolation_moving(pos, vel, obs=obs)
    print(vel_mod)

    vel_mod = obs_avoidance_orthogonal_moving(pos, vel, obs=obs)
    print(vel_mod)

    if True:
        Simulation_vectorFields(
        x_lim, y_lim,  obs=obs,
            xAttractor=xAttractor, saveFigure=False,
            figName='compare_algorithms_potential_field',
            obs_avoidance_func=obs_avoidance_potential_field,
            point_grid=num_point_grid,
            # show_streamplot=False,
            noTicks=False,
            # draw_vectorField=True,  automatic_reference_point=True, point_grid=N_resol
        )

    if False:
        Simulation_vectorFields(
            x_lim, y_lim,  obs=obs,
            xAttractor=xAttractor, saveFigure=False,
            figName='compare_algorithms_orthogonal',
            obs_avoidance_func=obs_avoidance_orthogonal_moving,
            # obs_avoidance_func=obs_avoidance_interpolation_moving,
            # draw_vectorField=True,
            # automatic_reference_point=True,
            show_streamplot=True,
            point_grid=num_point_grid,
        )


if (__name__)=="__main__":
    # compare_algorithms_plot()
    compare_algorithms_random()
