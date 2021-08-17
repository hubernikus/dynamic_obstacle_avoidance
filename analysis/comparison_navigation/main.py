"""
Replicating the paper of
'Safety of Dynamical Systems With MultipleNon-Convex Unsafe Sets UsingControl Barrier Functions'
"""
from dataclasses import dataclass

import numpy as np
from numpy import linalg as LA

from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.obstacles import Obstacle
# from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.containers import BaseContainer

def get_beta(obstacle, position):
    """ Beta value based on 'gamma' such that beta>0 in free space."""
    return obstacle.get_gamma(position, in_global_frame=True) - 1

def get_starset_deforming_factor(obstacle, beta, position=None, rel_obs_position=None):
    """ Get starshape deforming factor according to Rimon."""
    if rel_obs_position is None:
        rel_obs_position = position - obstacle.position
        
    min_dist = obstacle.get_minimal_distance()
    return min_dist*(1+beta)/LA.norm(rel_obs_position)


class NavigationContainer(BaseContainer):
    def get_relative_attractor_position(self, position):
        return position - self.attractor_position
    
    def get_beta_values(self, position):
        beta_values = np.zeros(self.n_obstacles)
        for oo, obstacle in enumerate(self.obstacle_list):
            beta_values[oo] = get_beta(obstacle, position)
        return beta_values
        
    def get_analytic_switches(self, position, beta_values, lambda_constant=1):
        """ Analytic switches of obstacle-avoidance function. """
        beta_bar = beta_values / np.tile(np.prod(beta_values), (beta_values.shape[0]))

        # Or gamma_d
        rel_dist_attractor = LA.norm(self.get_relative_attractor_position(position))

        scaled_dist = beta_bar*rel_dist_attractor
        switch_value = scaled_dist / (scaled_dist + lambda_constant*beta_values)

        return switch_value
                
    def transform_to_star_world(self, position):
        """ h(x). """
        weights = self.get_obstacle_weights(position)


        
        
    
