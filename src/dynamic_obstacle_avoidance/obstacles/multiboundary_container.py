"""
Container for Obstacle to treat the intersction (and exiting) between different walls
"""
# Author: Lukas Huber 
# Mail hubernikus@gmail.com
# License: BSD (c) 2021

import warnings
import numpy as np

from dynamic_obstacle_avoidance.obstacles import BaseContainer


class MultiBoundaryContainer(BaseContainer):
    """ Container to treat multiple boundaries / walls. """
    def get_boundary_list(self):
        """ Returns obstacle list containing all boundary-elements."""
        # TODO MAYBE: store boundaries in separate list (?)
        return [self[ii] for ii in range(self.n_obstacles) if self[ii].is_boundary]

    def get_boundary_ind(self):
        """ Returns indeces of the current container which are equivalent to obstacles."""
        return np.array([self[ii].is_boundary for ii in range(self.n_obstacles)])

    def check_collision(self, position):
        """ Returns collision with environment (type Bool)
        Note that obstacles are mutually additive, i.e. no collision with any obstacle
        while the boundaries are mutually subractive, i.e. collision free with at least one boundary
        """
        gamma_list_boundary = []
        
        for oo in range(self.n_obstacles):
            gamma = self[oo].get_gamma(position, in_global_frame=True)

            if self[oo].is_boundary:
                gamma_list_boundary.append(gamma)
                
            elif gamma < 1:
                # Collided with an obstacle
                return True
            
        # No collision with any obstacle so far
        return all(np.array(gamma_list_boundary) <= 1)

    def check_collision_array(self, positions):
        """ Return array of checked collisions of type bool. """
        collision_array = np.zeros(positions.shape[1], dtype=bool)
        for it in range(positions.shape[1]):
            collision_array[it] = self.check_collision(positions[:, it])
        return collision_array
        
    def update_relative_reference_point(self, position, gamma_margin_close_wall=1e-6):
        """ Get the local reference point as described in active-learning.
        !!! Current assumption: all obstacles are wall. """

        ind_boundary = self.get_boundary_ind()
        gamma_list = np.zeros(self.n_obstacles)
        for ii in range(self.n_obstacles):
            gamma_list[ii] = self[ii].get_gamma(position, in_global_frame=True)
        
        ind_inside = np.logical_and(gamma_list > 1, ind_boundary)
        ind_close = np.logical_and(gamma_list > gamma_margin_close_wall, ind_boundary)
        
        num_close = np.sum(ind_close)
            
        for ii, ii_self in zip(range(np.sum(ind_inside)), np.arange(self.n_obstacles)[ind_inside]):
            # Displacement_weight for each obstacle
            # TODO: make sure following function is for obstacles other than ellipses (!)
            boundary_point = self[ii_self].get_intersection_with_surface(
                direction=(position - self[ii_self].center_position), in_global_frame=True)

            weights = np.zeros(num_close)
            for jj, jj_self in zip(range(num_close), np.arange(self.n_obstacles)[ind_close]):
                if ii_self == jj_self:
                    continue
                gamma_boundary_point = self[jj_self].get_gamma(boundary_point, in_global_frame=True)
                if gamma_boundary_point < 1:
                    # Only obstacles are considered which intersect at the (projected) boundar point
                    continue

                dist_boundary_point = np.linalg.norm(boundary_point-self[ii_self].center_position)
                dist_point = np.linalg.norm(position-self[ii_self].center_position)

                gamma_center_point = self[jj_self].get_gamma(
                    self[ii_self].center_position, in_global_frame=True)

                # Weight for the distance to the surface
                weight_1 = (dist_point)/(dist_boundary_point-dist_point)
                
                # Weight for importance of the corresponding boundary
                weight_2 = gamma_boundary_point - 1

                weights[jj] = 1-1 / (1 + weight_1*weight_2)

            rel_reference_weight = np.max(weights)
            # breakpoint()

            if rel_reference_weight > 1:
                # TODO: remove aftr debugging..
                breakpoint()
                raise ValueError("Weight greater than 1...")

            self[ii_self].global_relative_reference_point = (
                rel_reference_weight*position +
                (1 - rel_reference_weight)*self[ii_self].global_reference_point)
            
        for ii_self in np.arange(self.n_obstacles)[~ind_inside]:
            self[ii_self].reset_relative_reference()

    def test(self):
        pass
