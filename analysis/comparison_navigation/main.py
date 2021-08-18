"""
Replicating the paper of
'Safety of Dynamical Systems With MultipleNon-Convex Unsafe Sets UsingControl Barrier Functions'
"""
from dataclasses import dataclass

from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
from matplotlib import cm

from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.obstacles import Obstacle, Ellipse
# from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.containers import BaseContainer


def get_beta(obstacle, position):
    """ Beta value based on 'gamma' such that beta>0 in free space."""
    return obstacle.get_gamma(position, in_global_frame=True) - 1


def get_starset_deforming_factor(obstacle, beta, position=None, rel_obs_position=None):
    """ Get starshape deforming factor 'nu'."""
    if rel_obs_position is None:
        if position is None:
            raise Exception("Wrong input arguments. 'position' or 'rel_obs_position' needed.")
        rel_obs_position = position - self[ii].position
        
    min_dist = obstacle.get_minimal_distance()
    return min_dist*(1+beta)/LA.norm(rel_obs_position)


class NavigationContainer(BaseContainer):
    """
    p: obstacle-center
    q: ray-center
    rho: minimum radius
    """
    @property
    def attractor_position(self):
        return self._attractor_position
    
    @attractor_position.setter
    def attractor_position(self, value):
        self._attractor_position = value
        
    def get_relative_attractor_position(self, position):
        return position - self.attractor_position
    
    def get_beta_values(self, position):
        beta_values = np.zeros(self.n_obstacles)
        for oo, obstacle in enumerate(self._obstacle_list):
            beta_values[oo] = get_beta(obstacle, position)
        return beta_values
        
    def get_analytic_switches(self, position, beta_values, lambda_constant=1):
        """ Analytic switches of obstacle-avoidance function. """
        ind_zero = np.isclose(beta_values, 0)
        if np.sum(ind_zero) == 0:
            beta_bar = beta_values / np.tile(np.prod(beta_values), (beta_values.shape[0]))
        elif np.sum(ind_zero) == 1:
            beta_bar = np.zeros(beta_values.shape)
            beta_bar[ind_zero] = 1
        else:
            raise Exception("Two zero-value beta's detected. This indicates an invalid \n"
                            + "environment of intersecting obstacles.")
        
        # Or gamma_d
        rel_dist_attractor = LA.norm(self.get_relative_attractor_position(position))

        scaled_dist = beta_bar*rel_dist_attractor
        switch_value = scaled_dist / (scaled_dist + lambda_constant*beta_values)
        
        return switch_value
    
    def transform_to_sphereworld(self, position):
        """ h(x) -  """
        # rel_pos_obstacle = np.zeros((self.dimension, self.n_obstacles))
        position_starshape = np.zeros(position.shape)
        beta_values = self.get_beta_values(position)
        analytic_switches = self.get_analytic_switches(position, beta_values=beta_values)
        
        for ii in range(self.n_obstacles):
            rel_obs_position = position - self[ii].position
            
            mu = get_starset_deforming_factor(
                self[ii], beta=beta_values[ii], rel_obs_position=rel_obs_position)

            position_starshape += analytic_switches[ii]*(mu*rel_obs_position + self[ii].position)
        return position_starshape


def plot_star_and_sphere_world():
    """ Sphere world & sphere-world."""
    x_lim = [-5, 5]
    y_lim = [-5, 5]

    obstacle_container = NavigationContainer()
    obstacle_container.attractor_position = np.array([0, 5])
    obstacle_container.append(
        Ellipse(
            center_position=np.array([2, 0]), 
            axes_length=np.array([2, 1]),
            )
        )

    obstacle_container.append(
        Ellipse(
            center_position=np.array([-2, 2]), 
            axes_length=np.array([2, 1]),
            orientation=30./180*pi,
            )
        )

    obstacle_container.append(
        Ellipse(
            center_position=np.array([-1, -2.5]), 
            axes_length=np.array([2, 1]),
            orientation=-50./180*pi,
            )
        )

    n_resolution = 50
    # velocities = np.zeros(positions.shape)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    
    for it_obs in range(obstacle_container.n_obstacles):
        obstacle_container[it_obs].draw_obstacle(n_grid=n_resolution)
        boundary_points = obstacle_container[it_obs].boundary_points_global
        ax.plot(boundary_points[0, :], boundary_points[1, :], 'k-')
        ax.plot(obstacle_container[it_obs].center_position[0],
                obstacle_container[it_obs].center_position[1], 'k+')

        boundary_displaced = np.zeros(boundary_points.shape)
        for ii in range(boundary_points.shape[1]):
            boundary_displaced[:, ii] = obstacle_container.transform_to_sphereworld(
                boundary_points[:, ii])
                
            # collisions[ii] = obstacle_container.check_collision(positions[:, ii])
            
        ax.plot(boundary_displaced[0, :], boundary_displaced[1, :], 'b')

        ax.plot(obstacle_container[it_obs].center_position[0],
                obstacle_container[it_obs].center_position[1], 'b+')
            
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.plot(obstacle_container.attractor_position[0],
            obstacle_container.attractor_position[1], 'k*')
    
    ax.set_aspect('equal', adjustable='box')
    
    plt.ion()
    plt.show()

    # cbar = fig.colorbar(cs, ticks=np.linspace(-10, 0, 11))
    
    # for ii in range(n_resolution):
        # obstacle_container.check_collision(positions[:, ii])
        # n_resolution =
        # pass


if (__name__) == "__main__":
    plt.close('all')
    plot_star_and_sphere_world()
    pass
