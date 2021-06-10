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
    def update_relative_reference_point(self, position, gamma_margin_close_wall=1e-6):
        """ Get the local reference point as described in active-learning.
        !!! Current assumption: all obstacles are wall. """

        gamma_list = np.zeros(self.n_obstacles)
        for ii in range(self.n_obstacles):
            gamma_list[ii] = self[ii].get_gamma(position, in_global_frame=True)

        ind_inside = gamma_list > 1
        ind_close = gamma_list > gamma_margin_close_wall
            
        num_close_obs = np.sum(ind_inside)
        
        for ii, ii_self in zip(range(num_close_obs), np.arange(self.n_obstacles)[ind_close]):
            # Displacement_weight for each obstacle
            # TODO: make sure following function is for obstacles other than ellipses (!)
            boundary_point = self[ii_self].get_intersection_with_surface(
                direction=(position - self[ii_self].center_position), in_global_frame=True)

            weights = np.zeros(num_close_obs)
            
            for jj, jj_self in zip(range(num_close_obs), np.arange(self.n_obstacles)[ind_close]):
                if ii_self == jj_self:
                    continue

                gamma_boundary_point = self[jj_self].get_gamma(
                    boundary_point, in_global_frame=True)

                if gamma_boundary_point < 1:
                    # Only obstacles are considered which intersect at the (projected) boundar point
                    continue

                gamma_center_point = self[jj_self].get_gamma(
                    self[ii_self].center_point, in_global_frame=True)

                weight_1 = ((gamma_reference_point - gamma_list[ii_self])
                            / (gamma_reference_point - gamma_boundary_point))
                weight_2 = gamma_boundary_point - 1

                weights[jj] = 1 / (1 + weight_1*weight_2)

            rel_reference_weight = np.max(weights)

            if rel_reference_weight > 1:
                # TODO: remove aftr debugging..
                breakpoint()
                raise ValueError("Weight greater than 1...")

            self[ii_self].global_relative_reference_point = (
                rel_reference_weight*position + (1 - rel_reference_weight)*self[ii_self].global_reference_point)
            
        for ii_self in np.arange(self.n_obstacles)[~ind_close]:
            self[ii_self].reset_relative_reference()

    def test(self):
        pass
