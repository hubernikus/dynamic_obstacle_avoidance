"""
Replicating the paper of
'Safety of Dynamical Systems With MultipleNon-Convex Unsafe Sets UsingControl Barrier Functions'
"""
# Author: Lukas Huber
# License: BSD (c) 2021
from dataclasses import dataclass

from math import pi

import numpy as np
from numpy import linalg as LA


import matplotlib.pyplot as plt
from matplotlib import cm

from vartools.states import ObjectPose
from vartools.math import get_numerical_gradient

from dynamic_obstacle_avoidance.obstacles import Obstacle, Ellipse, Sphere
# from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.containers import BaseContainer

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import Simulation_vectorFields



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



class NavigationContainer(BaseContainer):
    """
    The transformation based on
    'THE CONSTRUCTION OF ANALYTIC DIFFEOMORPHISMS FOR EXACT ROBOT NAVIGATION ON STAR WORLD'
    by ELON RIMON AND DANIEL E. KODITSCHEK in 1991

    p: obstacle-center
    q: ray-center
    rho: minimum radius

    NOTE: only applicable when space is GLOBALLY known(!)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                              lambda_constant=1, goal_norm_ord=1):
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
        switch_value = scaled_dist / (scaled_dist + lambda_constant*beta_values)
        return switch_value

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


def plot_sphere_world_and_nav_function():
    dimension = 2
    x_lim = [-5.5, 5.5]
    y_lim = [-5.5, 5.5]
    
    obstacle_container = NavigationContainer()
    
    obstacle_container.attractor_position = np.array([4.9, 0])

    rot_matr = get_rotation_matrix(rotation=45./180*pi)
    obstacle_container.attractor_position = rot_matr @ np.array([4.9, 0])
    obstacle_container.append(
        Sphere(
            radius=5,
            center_position=np.array([0, 0]),
            is_boundary=True,
            )
        )

    positions_minispheres = [
        [2, 2], [2, -2], [-2, -2], [-2, 2]]

    for pos in positions_minispheres:
        obstacle_container.append(
            Sphere(radius=0.75,
                   center_position=np.array(pos),
                   ))

    obstacle_container.append(
        Sphere(radius=0.5,
               center_position=np.array([0, 0]),
               ))
               
    # obstacle_container.append(
        # Sphere(radius=0.2,
               # center_position=np.array([3, 3]),
               # ))

    plot_obstacles = False
    if plot_obstacles:
        Simulation_vectorFields(
            x_range=x_lim, y_range=y_lim,
            obs=obstacle_container,
            draw_vectorField=False,
            automatic_reference_point=False,
            )
    ####################################
    # FOR DEBUGGING
    obstacle_container.default_kappa_factor = 5
    ####################################
            
    fig, ax = plt.subplots(figsize=(7.5, 6))

    n_grid_surf = 100
    x_vals, y_vals = np.meshgrid(np.linspace(x_lim[0], x_lim[1], n_grid_surf),
                                 np.linspace(y_lim[0], y_lim[1], n_grid_surf))
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    navigation_values = np.zeros(positions.shape[1])
    collisions = obstacle_container.check_collision_array(positions)
    
    for ii in range(positions.shape[1]):
        if collisions[ii]:
            continue
        
        navigation_values[ii] = obstacle_container.evaluate_navigation_function(
            positions[:, ii])

    n_grid = n_grid_surf
    # cs = ax.contourf(positions[0, :].reshape(n_grid, n_grid),
                    # positions[1, :].reshape(n_grid, n_grid),
                    # navigation_values.reshape(n_grid, n_grid),
                    # np.linspace(1e-6, 10.0, 41),
                    # cmap=cm.YlGnBu,
                    # linewidth=0.2, edgecolors='k'
                    # )

    cs = ax.contour(positions[0, :].reshape(n_grid, n_grid),
                    positions[1, :].reshape(n_grid, n_grid),
                    navigation_values.reshape(n_grid, n_grid),
                    levels=41,
                    # levels=np.linspace(1e-6, 5.0, 41),
                    # cmap=cm.YlGnBu,
                    linewidth=0.2, edgecolors='k'
                    )

    n_grid_quiver = 40
    x_vals, y_vals = np.meshgrid(np.linspace(x_lim[0], x_lim[1], n_grid_quiver),
                                 np.linspace(y_lim[0], y_lim[1], n_grid_quiver))
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities = np.zeros(positions.shape)
    collisions = obstacle_container.check_collision_array(positions)
    
    for ii in range(positions.shape[1]):
        if collisions[ii]:
            continue
        velocities[:, ii] = obstacle_container.evaluate_dynamics(positions[:, ii])

        norm_vel = LA.norm(velocities[:, ii])
        if norm_vel:
            velocities[:, ii] = velocities[:, ii] / norm_vel

    show_quiver = True
    if show_quiver:
        ax.quiver(positions[0, :], positions[1, :],
                  velocities[0, :], velocities[1, :], color="black")
    else:
        n_grid = n_grid_quiver
        ax.streamplot(x_vals, y_vals,
                      velocities[0, :].reshape(n_grid, n_grid),
                      velocities[1, :].reshape(n_grid, n_grid), color="blue")

    n_obs_resolution = 50
    for it_obs in range(obstacle_container.n_obstacles):
        obstacle_container[it_obs].draw_obstacle(n_grid=n_obs_resolution)
        boundary_points = obstacle_container[it_obs].boundary_points_global
        ax.plot(boundary_points[0, :], boundary_points[1, :], 'k-')
        ax.plot(obstacle_container[it_obs].center_position[0],
                obstacle_container[it_obs].center_position[1], 'k+')
    
    cbar = fig.colorbar(cs,
                        # ticks=np.linspace(-10, 0, 11)
                        )

    plot_trajcetory = True
    if plot_trajcetory:
        n_traj = 10000
        delta_time = 0.01
        positions = np.zeros((dimension, n_traj))
        start_position = np.array([-3.23, 2.55])
        start_position = np.array([-0.76, 0.93])
        start_position = np.array([-0.59, 1.45])
        
        positions[:, 0] = start_position
        for ii in range(positions.shape[1]-1):
            vel = obstacle_container.evaluate_dynamics(positions[:, ii])
            positions[:, ii+1] = delta_time*vel + positions[:, ii]

            if LA.norm(vel) < 1e-2:
                positions = positions[:, :ii+1]
                print(f"Zero veloctiy - stop loop at it={ii}")
                break

        plt.plot(positions[0, :], positions[1, :], 'r-')
        plt.plot(positions[0, 0], positions[1, 0], 'ro')
            
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    

if (__name__) == "__main__":
    plt.close('all')
    # plot_star_and_sphere_world()
    plot_sphere_world_and_nav_function()
    print('Done')
    pass
