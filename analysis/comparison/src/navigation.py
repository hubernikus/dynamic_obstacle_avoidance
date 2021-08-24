"""
Navigation Function & Sphere-Starshape Trafo according to Rimon & Koditscheck
"""
# Author: Lukas Huber
# License: BSD (c) 2021

from math import pi

import numpy as np
from numpy import linalg as LA

from vartools.math import get_numerical_gradient

from dynamic_obstacle_avoidance.containers import BaseContainer


def get_rotation_matrix(rotation):
    """ Returns 2D rotation matrix from rotation input."""
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    return np.array([[ cos_r, sin_r],
                     [-sin_r, cos_r]])


def get_beta(obstacle, position):
    """ Beta value based on 'gamma' such that beta>0 in free space."""
    # return obstacle.get_gamma(position, in_global_frame=True) - 1
    if obstacle.is_boundary:
        return obstacle.radius**2 - LA.norm(position - obstacle.position)**2
    else:
        return LA.norm(position - obstacle.position)**2 - obstacle.radius**2
    

def get_starset_deforming_factor(obstacle, beta, position=None, rel_obs_position=None):
    """ Get starshape deforming factor 'nu'."""
    if rel_obs_position is None:
        if position is None:
            raise Exception("Wrong input arguments. 'position' or 'rel_obs_position' needed.")
        rel_obs_position = position - self[ii].position
        
    min_dist = obstacle.get_minimal_distance()
    return min_dist*(1+beta)/LA.norm(rel_obs_position)


class SphereToStarTransformer(BaseContainer):
    """
    The transformation based on
    'THE CONSTRUCTION OF ANALYTIC DIFFEOMORPHISMS FOR EXACT ROBOT NAVIGATION ON STAR WORLD'
    by ELON RIMON AND DANIEL E. KODITSCHEK in 1991

    p: obstacle-center
    q: ray-center
    rho: minimum radius

    NOTE: only applicable when space is GLOBALLY known(!)
    """
    def __init__(self, lambda_constant=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lambda_constant = lambda_constant
        # For easy trial&error
        self.default_kappa_factor = 2

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
        
    def get_analytic_switches(self, rel_pos_attractor, beta_values,
                              goal_norm_ord=1):
        """ Return analytic switches of obstacle-avoidance function.
        
        Parameters
        ----------
        goal_norm_ord:
        """
        # Or gamma_d /
        if not np.isinf(goal_norm_ord):
            goal_norm_ord = goal_norm_ord*2
        rel_dist_attractor = LA.norm(rel_pos_attractor, ord=goal_norm_ord)
        
        if not rel_dist_attractor:
            # Position at attractor
            return np.zeros(beta_values.shape)
        
        ind_zero = np.isclose(beta_values, 0)
        if np.sum(ind_zero) == 0:
            beta_bar = beta_values / np.tile(np.prod(beta_values), (beta_values.shape[0]))
        elif np.sum(ind_zero) == 1:
            beta_bar = np.zeros(beta_values.shape)
            beta_bar[ind_zero] = 1
        else:
            raise Exception("Two zero-value beta's detected. This indicates an invalid \n"
                            + "environment of intersecting obstacles.")

        scaled_dist = beta_bar*rel_dist_attractor
        switch_value = scaled_dist / (scaled_dist + self.lambda_constant*beta_values)
        return switch_value

    def evaluate_dynamics(self, position):
        gradient = get_numerical_gradient(
            position=position,
            function=self.evaluate_navigation_function)
        return (-1)*gradient
    
    def transform_to_sphereworld(self, position):
        """ h(x) -  """
        # rel_pos_obstacle = np.zeros((self.dimension, self.n_obstacles))
        rel_pos_attractor = self.get_relative_attractor_position(position)
        position_starshape = np.zeros(position.shape)
        beta_values = self.get_beta_values(position)
        analytic_switches = self.get_analytic_switches(
            rel_pos_attractor=rel_pos_attractor, beta_values=beta_values)
        
        for ii in range(self.n_obstacles):
            rel_obs_position = position - self[ii].position
            
            mu = get_starset_deforming_factor(
                self[ii], beta=beta_values[ii], rel_obs_position=rel_obs_position)

            position_starshape += analytic_switches[ii]*(mu*rel_obs_position + self[ii].position)
        return position_starshape


class NavigationContainer(SphereToStarTransformer):
    # TODO: Cohesion not coupling (!)
    def get_minimum_epsilon(self, variable=None):
        pass
    
    def get_minimum_N(self, variable=None):
        
        return 0.5*1/epsilon
        pass

    def rho0(self):
        pass
    
    def get_epsilon_factor(self):
        """
        Return the epsilon factor for the navigation-fuction.
        
        rho: radius
        q: (center) position
        """
        if True:
            # Simple value
            return 1e-3
            
        # Token evaluation
        rho0 = self.boundary.radius
        
        beta_i_max = np.zeros(self.n_obstacles)
        for ii in range(self.n_obstacles):
            beta_i_max[ii] = ((rho0 + LA.norm(self.center_position))**2
                              - self[ii].radius**2)

            if self[ii].is_boundary:
                beta_i_max[ii] = rho0**2

        norm_beta_i_j_grad_min = np.zeros((
            self.dimension, self.n_obstacles, self.n_obstacles))
        for ii in range(self.n_obstacles):
            for jj in range(self.n_obstacles):
                if ii == jj:
                    norm_beta_i_j_grad_min[:, ii, jj] = 2*self[ii].radius
                    continue
                
                norm_beta_i_j_grad_min[:, ii, jj] = 2*(np.sqrt(epsilon))
        
        gradient_of_beta_i = np.zeros(0)
        hessian_of_beta_i = np.zeros(0)

        epsilon0_prime = LA.norm(q_d - q_i)**2 - rho_i**2
        # epsilon0_pprime = 
        # epsilon0 = 
        
        epsilon1 = rho0**2 - LA.norm(q_d)**2

        epsilon2_prime = 1.0/2*rho_i**2
        epsilon2_pprime = (1.0/4.0 * min(sqrt(beta_i_bar * LA.norm(grad_beta_i)))
                           / whatever)
        epsilon2 = min(epsilon2_prime, epsilon2_pprime)

        epsilon = 1.0/2 * min(epsilon0, epsilon1, epsilon2)
        return epslion
        pass

    def get_N_constant(self, epsilon=None):
        if epsilon is None:
            epsilon = self.get_epsilon_factor()

        # N = 1.0/2.0 * 1/epsilon*np.sqrt(max_gamma_d) + np.sum(max_norm_beta_grad)
        return 1e5

    def get_kappa_factor(self):
        """ Return kappa-factor which 'pushes' the local-attractor
        to the boundary. """
        N_constant = self.get_N_constant()
        kappa = np.ceil(N_constant) + 1
        return kappa
    
    def evaluate_navigation_function(
        self, position, beta_prod=None, kappa_factor=None):
        if kappa_factor is None:
            # kappa_factor = self.get_kappa_factor()
            kappa_factor = self.default_kappa_factor
            
        if beta_prod is None:
            beta_values = self.get_beta_values(position)
            beta_prod = np.prod(beta_values)

        rel_pos_attractor = self.get_relative_attractor_position(position)
        rel_dist_attractor = LA.norm(rel_pos_attractor)

        phi = (rel_dist_attractor**2
               / (rel_dist_attractor**kappa_factor + beta_prod)**(1.0/kappa_factor))
        return phi
    
    
