"""
Replicating the paper of
'Safety of Dynamical Systems With MultipleNon-Convex Unsafe Sets UsingControl Barrier Functions'
"""
from math import pi

import numpy as np
from numpy import linalg as LA

from dynamic_obstacle_avoidance.obstacles import Obstacle


class DoubleBlob(Obstacle):
    """Double blob obstacle."""

    def __init__(self, a_value, b_value, *args, **kwargs):
        self.aa = a_value
        self.bb = b_value
        # breakpoint()
        super().__init__(*args, **kwargs)

        # Only defined for dimension==2
        self.dimension = 2

    def get_minimal_distance(self):
        if self.boundary_points_local is None:
            self.draw_obstacle(n_grid=50)
        return np.min(LA.norm(self.boundary_points_local, axis=0))

    def barrier_function(self, position):
        """Barrier funciton in local-frame."""
        x1, x2 = position[0], position[1]
        return ((x1 - self.aa) ** 2 + x2**2) * (
            (x1 + self.aa) ** 2 + x2**2
        ) - self.bb**4

    def get_local_radius_not_working(
        self,
        position,
        in_global_frame=False,
        it_max=100,
        convergence_margin=1e-4,
    ):
        # (!) Currently not working
        # A trial to find an analyitical description of the boundary
        if in_global_frame:
            position = self.transform_global2relative(position)

        # Find numerically position where barrier_funciton==0
        h_barrier = self.barrier_function(position)

        rad_local = LA.norm(position) - h_barrier
        return rad_local

    def get_local_radius(
        self,
        position,
        in_global_frame=False,
        it_max=100,
        atol=1e-4,
        null_value=0,
    ) -> float:
        """Return numerical evaluation of the local radius based on barrier_function."""
        if in_global_frame:
            position = self.transform_global2relative(position)

        if not LA.norm(position):
            # At origin -> return smallest axes (semi-random)
            return min(self.aa, self.bb)

        # Find numerically position where barrier_funciton==0
        h_barrier = self.barrier_function(position)
        if np.isclose(h_barrier, null_value, atol=1e-4):
            return LA.norm(position)
        elif h_barrier > 0:
            pos_out = position
            pos_in = np.zeros(position.shape)
        else:  # h_barrier < 0:
            pos_in = position
            for ii in range(it_max):
                pos_new = pos_in * 2.0
                h_barrier = self.barrier_function(pos_new)

                if np.isclose(h_barrier, null_value, atol=1e-4):
                    return LA.norm(pos_new)

                elif h_barrier > 0:
                    pos_out = pos_new
                    break

                else:  # h_barrier < 0:
                    pos_in = pos_new

        for ii in range(it_max):
            pos_new = 0.5 * (pos_out + pos_in)
            h_barrier = self.barrier_function(pos_new)

            if np.isclose(h_barrier, null_value, atol=1e-4):
                return LA.norm(pos_new)

            elif h_barrier > 0:
                pos_out = pos_new

            else:
                pos_in = pos_new
        return LA.norm(pos_new)

    def get_gamma(
        self,
        position,
        in_global_frame=False,
        it_max=100,
        gamma_distance=None,
        gamma_type=None,
    ):
        if gamma_distance is not None:
            # Legacy error -> remove this keyword argument...
            pass

        if gamma_type is not None:
            # Not implemented yet. Just default type for now.
            pass

        if in_global_frame:
            position = self.transform_global2relative(position)
        radius = self.get_local_radius(position)
        return LA.norm(position) / radius

    def get_normal_direction(
        self, position, delta_angle=1e-2, in_global_frame=False, normalize=True
    ):
        """Calculate the normal direction."""
        # Numerical calculation  of the normal direction based on local-radius
        if in_global_frame:
            position = self.transform_global2relative(position)

        angle = np.arctan2(position[1], position[0])

        angles = [angle - delta_angle * 0.5, angle + delta_angle * 0.5]
        surf_pos = np.zeros((self.dimension, len(angles)))

        for aa, ang in enumerate(angles):
            pos = np.array([np.cos(ang), np.sin(ang)])
            radius = self.get_local_radius(pos)

            surf_pos[:, aa] = pos / LA.norm(pos) * radius

        tangent = surf_pos[:, 1] - surf_pos[:, 0]
        normal = np.array([-tangent[1], tangent[0]])

        if normalize:
            normal = normal / LA.norm(normal)

        if in_global_frame:
            normal = self.transform_relative2global_dir(normal)

        return normal

    def draw_obstacle(self, n_grid=50):
        angles = np.linspace(0, 2 * pi, n_grid)
        dirs = np.vstack((np.cos(angles), np.sin(angles)))
        local_rad = [self.get_local_radius(dirs[:, dd]) for dd in range(dirs.shape[1])]

        self.boundary_points_local = dirs * np.tile(local_rad, (self.dimension, 1))

        if self.margin_absolut:  # Nonzero
            raise NotImplementedError("Margin not implemented.")
        else:
            self.boundary_points_margin_local = self.boundary_points_local
