#!/USSR/bin/python3
"""
Test script for obstacle avoidance algorithm
Test normal formation
"""
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer, GradientContainer
from dynamic_obstacle_avoidance.avoidance import DynamicModulationAvoider

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem

import unittest


class TestPolygon(unittest.TestCase):
    def test_single_polygon(self, visualize=False):
        margin_absolut = 0.1
                
        obs = Cuboid(center_position=np.array([0.2, 2.4]),
               axes_length=[0.4, 2.4],
               margin_absolut=margin_absolut,
               orientation=-30*pi/180,
               tail_effect=False,
               )

        obstacle_environment = ObstacleContainer()
        obstacle_environment.append(obs)

        point0 = obs.center_position + np.array([1, 0])
        point1 = obs.center_position + np.array([2, 0])
        
        gamma0 = obs.get_gamma(point0, in_global_frame=True)
        gamma1 = obs.get_gamma(point1, in_global_frame=True)

        self.assertTrue(gamma0 < gamma1)
        
        radius0 = obs.get_local_radius(point0, in_global_frame=True)
        radius1 = obs.get_local_radius(point1, in_global_frame=True)

        self.assertTrue(np.isclose(radius0, radius1))

        if visualize:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            x_lim = [-3, 3]
            y_lim = [-0.5, 3.5]
            
            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

            n_resolution = 40
            dim = 2
            x_vals = np.linspace(x_lim[0], x_lim[1], n_resolution)
            y_vals = np.linspace(y_lim[0], y_lim[1], n_resolution)
    
            gamma_values = np.zeros((n_resolution, n_resolution))
            positions = np.zeros((dim, n_resolution, n_resolution))

            for ix in range(n_resolution):
                for iy in range(n_resolution):
                    positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]
            
                    gamma_values[ix, iy] = obs.get_gamma(
                        positions[:, ix, iy], in_global_frame=True)

            cs = ax.contourf(positions[0, :, :], positions[1, :, :],  gamma_values, 
                             np.arange(1.0, 5.0, 0.1),
                             # cmap=plt.get_cmap('autumn'),
                             cmap=plt.get_cmap('hot'),
                             extend='max', alpha=0.6, zorder=-3)
            
            cbar = fig.colorbar(cs)

        
    def test_polygon_multigamma(self, visualize=False, save_figure=False):
        margin_absolut = 0.1
        obstacle_environment = ObstacleContainer()
        obstacle_environment.append(
            Cuboid(center_position=np.array([0.2, 2.4]),
                   axes_length=[0.4, 2.4],
                   margin_absolut=margin_absolut,
                   orientation=-30*pi/180,
                   tail_effect=False,
                   ))

        obstacle_environment.append(
            Cuboid(center_position=np.array([1.2, 0.25]),
                   axes_length=[0.4, 1.45],
                   margin_absolut=margin_absolut,
                   orientation=0*pi/180,
                   tail_effect=False,
                   ))

        # attractor_position = np.array([2, 2])
        attractor_position = np.array([2, 0.7])
        initial_dynamics = LinearSystem(attractor_position=attractor_position) 

        dynamic_avoider = DynamicModulationAvoider(
            initial_dynamics=initial_dynamics, environment=obstacle_environment)
        
        # point0 = np.array([-0.950, 1.225])
        point0 = np.array([-1.92307692,  0.21794872])
        gamma0 = dynamic_avoider.get_gamma_product(point0)

        if visualize:
            fig, ax = plt.subplots(figsize=(10, 5))
            x_lim = [-3, 3]
            y_lim = [-0.5, 3.5]
            
            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

            n_resolution = 100
            dim = 2
            x_vals = np.linspace(x_lim[0], x_lim[1], n_resolution)
            y_vals = np.linspace(y_lim[0], y_lim[1], n_resolution)
    
            gamma_values = np.zeros((n_resolution, n_resolution))
            positions = np.zeros((dim, n_resolution, n_resolution))

            for ix in range(n_resolution):
                for iy in range(n_resolution):
                    positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]
            
                    gamma_values[ix, iy] = dynamic_avoider.get_gamma_product(
                        positions[:, ix, iy])

            cs = ax.contourf(positions[0, :, :], positions[1, :, :],  gamma_values, 
                             np.arange(1.0, 5.0, 0.1),
                             # cmap=plt.get_cmap('autumn'),
                             cmap=plt.get_cmap('hot'),
                             extend='max', alpha=0.6, zorder=-3)
            
            cbar = fig.colorbar(cs)

            if save_figure:
                figName = "gamma_danger_field_for_multiobstacle"
                plt.savefig('figures/' + figName + '.png', bbox_inches='tight')
                


if (__name__)=="__main__":
    run_all = False
    if run_all:
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
        
    else:
        plt.close('all')
        plt.ion()
        plt.show()
        
        my_tester = TestPolygon()
        
        # my_tester.test_single_polygon(visualize=True)
        my_tester.test_polygon_multigamma(visualize=True, save_figure=True)

