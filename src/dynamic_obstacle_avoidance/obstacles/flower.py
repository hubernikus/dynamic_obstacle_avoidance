""" """
# Author: Lukas Huber
# Email: lukas.huber@epfl.ch

import time
import warnings
import sys
import copy

import numpy as np

from vartools.angle_math import angle_modulo
from vartools.angle_math import *

from dynamic_obstacle_avoidance.utils import *
from dynamic_obstacle_avoidance.avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.avoidance.obs_dynamic_center_3d import *

from dynamic_obstacle_avoidance.obstacles import Obstacle


class StarshapedFlower(Obstacle):
    def __init__(
        self,
        radius_magnitude=1,
        radius_mean=2,
        number_of_edges=4,
        time_now=None,
        property_functions={},
        *args,
        **kwargs
    ):
        if sys.version_info > (3, 0):
            super().__init__(*args, **kwargs)
        else:
            super(StarshapedFlower, self).__init__(*args, **kwargs)

        self.property_dict = {
            "radius_magnitude": radius_magnitude,
            "radius_mean": radius_mean,
            "number_of_edges": number_of_edges,
        }
        # Object Specific Paramters
        # self.radius_magnitude=radius_magnitude
        # self.radius_mean=radius_mean
        # self.number_of_edges=number_of_edges

        self.is_convex = False  # What for?

        if self.is_deforming:
            self.property_functions = property_functions

            if time_now is None:
                self.time_now = time.time()
            else:
                self.time_now = time_now

            # Duplicate list
            self.update_deforming_obstacle(self.time_now)

    @property
    def radius_magnitude(self):
        return self.property_dict["radius_magnitude"]

    @radius_magnitude.setter
    def radius_magnitude(self, value):
        self.property_dict["radius_magnitude"] = value

    @property
    def radius_mean(self):
        return self.property_dict["radius_mean"]

    @radius_mean.setter
    def radius_mean(self, value):
        self.property_dict["radius_mean"] = value

    @property
    def number_of_edges(self):
        return self.property_dict["number_of_edges"]

    @number_of_edges.setter
    def number_of_edges(self, value):
        self.property_dict["number_of_edges"] = value

    def get_radius_of_angle(self, angle, in_global_frame=False):
        if in_global_frame:
            angle -= self.orientation
        return self.radius_mean + self.radius_magnitude * np.cos(
            (angle) * self.number_of_edges
        )

    def get_radiusDerivative_of_angle(self, angle, in_global_frame=False):
        if in_global_frame:
            angle -= self.orientation
        return (
            -self.radius_magnitude
            * self.number_of_edges
            * np.sin((angle) * self.number_of_edges)
        )

    def draw_obstacle(self, include_margin=False, n_curve_points=100, numPoints=None):
        # warnings.warn("Remove numPoints from function argument.")

        angular_coordinates = np.linspace(0, 2 * pi, n_curve_points)
        radius_angle = self.get_radius_of_angle(angular_coordinates)

        if self.dim == 2:
            direction = np.vstack(
                (np.cos(angular_coordinates), np.sin(angular_coordinates))
            )

        x_obs = radius_angle * direction
        x_obs_sf = radius_angle * self.sf * direction

        if self.orientation:  # nonzero
            for jj in range(x_obs.shape[1]):
                x_obs[:, jj] = self.rotMatrix.dot(x_obs[:, jj]) + np.array(
                    [self.center_position]
                )
            for jj in range(x_obs_sf.shape[1]):
                x_obs_sf[:, jj] = self.rotMatrix.dot(x_obs_sf[:, jj]) + np.array(
                    [self.center_position]
                )

        self.boundary_points_local = x_obs
        self.boundary_points_margin_local = x_obs_sf

    def get_local_radius_point(self, position, in_global_frame=False):
        """Get radius from local radius point."""
        if in_global_frame:
            position = self.transform_global2relative(position)

        direction = np.arctan2(position[1], position[0])
        radius = self.get_radius_of_angle(direction)

        surface_point = 1.0 * radius / np.linalg.norm(position) * position

        if in_global_frame:
            surface_point = self.transform_relative2global(surface_point)

        return surface_point

    def get_deformation_velocity(self, position, in_global_frame=False):
        """Get numerical evaluation of velocity"""
        if in_global_frame:
            position = self.transform_global2relative(position)

        dt = self.time_now - self.time_old

        if not dt:  # == 0
            warnings.warn("No time passed to evaluate defomration velocity.")
            return np.zeros(2)

        radius_point = self.get_local_radius_point(position)

        # Evaluate for last point!
        self.property_dict_temp = self.property_dict
        self.property_dict = self.property_dict_old
        radius_point_old = self.get_local_radius_point(position)

        # Reset to previous
        self.property_dict = self.property_dict_temp

        # Return numerical differentiation
        vel_surface = (radius_point - radius_point_old) / dt

        if in_global_frame:
            vel_surface = self.transform_relative2global_dir(vel_surface)

        return vel_surface

    def update_deforming_obstacle(self, time_now):
        """Update values."""
        self.time_old = self.time_now
        self.time_now = time_now

        self.property_dict_old = copy.deepcopy(self.property_dict)

        for key in self.property_functions.keys():
            self.property_dict[key] = self.property_functions[key](time_now)

    def get_gamma(
        self,
        position,
        in_global_frame=False,
        norm_order=2,
        gamma_distance=None,
    ):
        """Calculate gamma-distance function."""
        if not type(position) == np.ndarray:
            position = np.array(position)

        if in_global_frame:
            position = self.transform_global2relative(position)

        mag_position = np.linalg.norm(position)
        if mag_position == 0:
            if self.is_boundary:
                return sys.float_info.max
            else:
                return 0

        direction = np.arctan2(position[1], position[0])

        Gamma = mag_position / self.get_radius_of_angle(direction)

        # TODO extend rule to include points with Gamma < 1 for both cases
        if self.is_boundary:
            Gamma = 1 / Gamma

        if gamma_distance is not None:
            warnings.warn("Implement linear gamma type.")

        return Gamma

    def get_normal_direction(self, position, in_global_frame=False, normalize=True):
        if in_global_frame:
            position = self.transform_global2relative(position)

        mag_position = np.linalg.norm(position)
        if not mag_position:
            return np.ones(self.dim) / self.dim  # Just return one direction

        direction = np.arctan2(position[1], position[0])
        derivative_radius_of_angle = self.get_radiusDerivative_of_angle(direction)

        radius = self.get_radius_of_angle(direction)

        normal_vector = np.array(
            (
                [
                    derivative_radius_of_angle * np.sin(direction)
                    + radius * np.cos(direction),
                    -derivative_radius_of_angle * np.cos(direction)
                    + radius * np.sin(direction),
                ]
            )
        )
        normal_vector = (-1) * normal_vector

        if normalize:
            mag_vector = np.linalg.norm(normal_vector)
            if mag_vector:  # Nonzero
                normal_vector = normal_vector / mag_vector

        if False:
            # TODO: remove after DEBUG
            import matplotlib.pyplot as plt  # Remove after debugging

            # self.draw_reference_hull(normal_vector, position)
            pos_abs = self.transform_relative2global(position)
            norm_abs = self.transform_relative2global_dir(normal_vector)
            plt.quiver(pos_abs[0], pos_abs[1], norm_abs[0], norm_abs[1], color="g")
            ref_abs = self.get_reference_direction(position)
            ref_abs = self.transform_relative2global_dir(ref_abs)
            plt.quiver(pos_abs[0], pos_abs[1], ref_abs[0], ref_abs[1], color="k")

            plt.ion()
            plt.show()

        if in_global_frame:
            normal_vector = self.transform_relative2global_dir(normal_vector)

        return normal_vector
