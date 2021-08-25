"""
Replicating the paper of
'Safety of Dynamical Systems With MultipleNon-Convex Unsafe Sets UsingControl Barrier Functions'
"""
from math import pi

import numpy as np
from numpy import linalg as LA

from dynamic_obstacle_avoidance.obstacles import Obstacle


class DoubleBlob(Obstacle):
    """ Double blob obstacle. """
    def __init__(self, a_value, b_value, *args, **kwargs):
        self.aa = a_value
        self.bb = b_value
        super().__init__(*args, **kwargs)

        # Only defined for dimension==2
        self.dimension = 2

    def barrier_function(self, position):
        """ Barrier funciton in local-frame."""
        x1, x2 = position[0], position[1]
        return ((x1-self.aa)**2 + x2**2)*((x1+self.aa)**2 + x2**2) - self.bb**4

    def get_local_radius(self, position, in_global_frame=False,
                         it_max=100, convergence_margin=1e-4):
        # Numerical evaluation based on barrier_function
        if in_global_frame:
            position = self.transform_global2relative(position)

        # Find numerically position where barrier_funciton==0
        h_barrier = self.barrier_function(position)

        if abs(h_barrier) < convergence_margin:
            return LA.norm(position)
        elif h_barrier > 0:
            pos_out = position
            pos_in = np.zeros(position.shape)
        else:   # h_barrier < 0:
            pos_in = position
            for ii in range(it_max):
                pos_new = pos_in*2.0
                h_barrier = self.barrier_function(pos_new)
                
                if abs(h_barrier) < convergence_margin:
                    return LA.norm(pos_new)
                
                elif h_barrier > 0:
                    pos_out = pos_new
                    break
                
                else:    # h_barrier < 0:
                    pos_in = pos_new
                    delta_dist = delta_dist*2.0

        for ii in range(it_max):
            pos_new = 0.5*(pos_out + pos_in)
            h_barrier = self.barrier_function(pos_new)
            
            if abs(h_barrier) < convergence_margin:
                return LA.norm(pos_new)
            
            elif h_barrier > 0:
                pos_out = pos_new
                
            else:
                pos_in = pos_new

        return LA.norm(pos_new)
        
    def get_gamma(self, position, in_global_frame=False, it_max=100):
        if in_global_frame:
            position = self.transform_global2relative(position)
        radius = self.get_local_radius(position)
        return LA.norm(position) / radius

    def get_normal_direction(self, position):
        pass

    def draw_obstacle(self, n_grid=30):
        angles = np.linspace(0, 2*pi, n_grid)
        dirs = np.vstack((np.cos(angles), np.sin(angles)))
        local_rad = [self.get_local_radius(dirs[:, dd]) for dd in range(dirs.shape[1])]

        self.boundary_points_local = dirs * np.tile(local_rad, (self.dimension, 1))
        
        if self.margin_absolut: # Nonzero
            raise NotImplementedError("Margin not implemented.")
        else:
            self.boundary_points_margin_local = self.boundary_points_local

