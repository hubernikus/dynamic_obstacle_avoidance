""" """
# Author: Lukas Huber
# Email: lukas.huber@epfl.ch

import time
import warnings
import sys
import copy
from typing import Optional

import numpy as np

from vartools.angle_math import angle_modulo
from vartools.angle_math import *

from vartools.dynamical_systems import LinearSystem


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
        **kwargs,
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

    def get_characteristic_length(self):
        return self.radius_mean + self.radius_magnitude

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
        x_obs_sf = (radius_angle + self.margin_absolut) * direction

        if False:
            if self.orientation:  # nonzero
                for jj in range(x_obs.shape[1]):
                    # x_obs[:, jj] = self.rotMatrix.dot(x_obs[:, jj]) + np.array(
                    #     [self.center_position]
                    # )
                    x_obs[:, jj] = self.pose.transform_position_from_relative(
                        x_obs[:, jj]
                    )

                    for jj in range(x_obs_sf.shape[1]):
                        x_obs_sf[:, jj] = self.pose.transform_position_from_relative(
                            x_obs_sf[:, jj]
                        )
                        # x_obs_sf[:, jj] = self.rotMatrix.dot(x_obs_sf[:, jj]) + np.array(
                        #     [self.center_position]
                        # )

        self.boundary_points_local = x_obs
        self.boundary_points_margin_local = x_obs_sf

    def get_local_radius(self, position, in_global_frame: bool = False) -> float:
        if in_global_frame:
            position = self.transform_global2relative(position)

        direction = np.arctan2(position[1], position[0])
        return self.get_radius_of_angle(direction)

    def get_local_radius_point(
        self, position, in_global_frame: bool = False
    ) -> np.ndarray:
        """Get radius from local radius point."""
        if in_global_frame:
            position = self.transform_global2relative(position)

        direction = np.arctan2(position[1], position[0])
        radius = self.get_radius_of_angle(direction)

        if pos_norm := np.linalg.norm(position):
            surface_point = radius / pos_norm * position
        else:
            surface_point = np.zeros_like(position)
            surface_point[0] = radius

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
        in_obstacle_frame: Optional[bool] = None,
        norm_order=2,
        gamma_distance=None,
    ):
        """Calculate gamma-distance function."""
        if not type(position) == np.ndarray:
            position = np.array(position)

        if in_obstacle_frame is not None:
            in_global_frame = not (in_obstacle_frame)

        if in_global_frame:
            position = self.pose.transform_position_to_relative(position)

        if not (mag_position := np.linalg.norm(position)):
            if self.is_boundary:
                return sys.float_info.max
            else:
                return 0

        mag_position *= self.distance_scaling

        radius = self.get_radius_of_angle(np.arctan2(position[1], position[0]))

        # TODO extend rule to include points with Gamma < 1 for both cases
        if self.is_boundary:
            # Gamma = 1 / Gamma
            return radius / mag_position
        elif mag_position < radius:
            return mag_position / radius
        else:
            return mag_position - radius + 1

        # if gamma_distance is not None:
        #     warnings.warn("Implement linear gamma type.")

        # return Gamma

    def get_normal_direction(self, position, in_global_frame=False):
        if in_global_frame:
            position = self.pose.transform_position_to_relative(position)

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
        normal_vector = normal_vector / LA.norm(normal_vector)

        if in_global_frame:
            normal_vector = self.pose.transform_direction_from_relative(normal_vector)

        return normal_vector


def test_starshape_flower(visualize=False):
    flower_obstacle = StarshapedFlower(
        center_position=np.array([1, -2]),
        radius_magnitude=0.5,
        radius_mean=1.5,
        number_of_edges=3,
        # axes_length=np.array([4, 2]),
        orientation=30 / 90.0 * math.pi,
    )

    initial_dynamics = LinearSystem(attractor_position=np.array([4.0, 0]))

    rotation_container = RotationContainer()
    rotation_container.set_convergence_directions(converging_dynamics=initial_dynamics)
    rotation_container.append(flower_obstacle)

    if visualize:
        import matplotlib.pyplot as plt

        plt.ion()

        x_lim = [-6.5, 6.5]
        y_lim = [-5.0, 5.0]

        from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacles(
            obstacle_container=rotation_container,
            ax=ax,
            alpha_obstacle=1.0,
        )

        # fig, _ = Simulation_vectorFields(
        #     x_lim=x_lim,
        #     y_lim=y_lim,
        #     n_resolution=110,
        #     obstacle_list=rotation_container,
        #     noTicks=True,
        #     showLabel=False,
        #     draw_vectorField=True,
        #     dynamical_system=initial_dynamics.evaluate,
        #     obs_avoidance_func=obstacle_avoidance_rotational,
        #     automatic_reference_point=False,
        #     pos_attractor=initial_dynamics.attractor_position,
        #     # Quiver or stream-plot
        #     # show_streamplot=False,
        #     show_streamplot=True,
        # )


def test_gamma_value(visualize=False):
    center = np.array([2.2, 0.0])
    obstacle = StarshapedFlower(
        center_position=center,
        radius_magnitude=0.2,
        number_of_edges=5,
        radius_mean=0.75,
        orientation=33 / 180 * pi,
        distance_scaling=1,
        # tail_effect=False,
        # is_boundary=True,
    )

    if visualize:
        x_lim = [-1, 5]
        y_lim = [-3, 3]
        n_grid = 100

        nx = ny = n_grid
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx),
            np.linspace(y_lim[0], y_lim[1], ny),
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gammas = np.zeros(positions.shape[1])

        for pp in range(positions.shape[1]):
            gammas[pp] = obstacle.get_gamma(positions[:, pp], in_global_frame=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        plot_obstacles(obstacle_container=[obstacle], ax=ax, x_lim=x_lim, y_lim=y_lim)

        ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas.reshape(nx, ny),
            levels=np.linspace(1, 10.0, 19),
            extend="both",
            zorder=-2,
            cmap="Blues",
        )

    # Test gamma a bit away from the center
    gamma_value = obstacle.get_gamma(center + 0.1, in_global_frame=True)
    assert 0 < gamma_value < 1

    # Test at the center
    gamma_value = obstacle.get_gamma(center, in_global_frame=True)
    assert np.isclose(0, gamma_value)


def test_radius_computation():
    center = np.array([2.2, 0.0])
    obstacle = StarshapedFlower(
        center_position=center,
        radius_magnitude=0.2,
        number_of_edges=5,
        radius_mean=0.75,
        orientation=33 / 180 * pi,
        distance_scaling=1,
    )

    # Test gamma a bit away from the center
    radius = obstacle.get_gamma(center, in_global_frame=True)
    assert not np.isnan(radius)


if (__name__) == "__main__":
    # test_starshape_flower(visualize=True)
    test_gamma_value(visualize=True)
    # test_radius_computation()

    print("Tests done.")
