#!/USSR/bin/python3
""" Define the Polygon with Dynamic Boundaries as used in LASA. """

import sys
import time
import copy

from math import pi
import numpy as np

import matplotlib.pyplot as plt  # TODO: remove after debugging

from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import (
    get_directional_weighted_sum,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import (
    Polygon,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import (
    get_inverse_proprtional_weight,
)


class DynamicBoundariesPolygon(Polygon):
    """
    Dynamic Boundary for Application in 3D with surface polygons

    Pyramid shape (without top edge)
    """

    def __init__(
        self,
        indeces_of_flexibleTiles=None,
        inflation_parameter=None,
        is_surgery_setup=False,
        *args,
        **kwargs
    ):

        if is_surgery_setup:
            # With of bottom (a1) and top (a2) square respectively
            # Bottom and top width
            self.width = np.array([3e-2, 17e-2])
            # self.width = np.array([17e-2, 17e-2])

            # box height
            self.height = 15e-2

            edge_points = np.array(
                [
                    [-self.width[0] / 2.0, -self.width[0] / 2.0, 0],
                    [self.width[0] / 2.0, -self.width[0] / 2.0, 0],
                    [self.width[0] / 2.0, self.width[0] / 2.0, 0],
                    [-self.width[0] / 2.0, self.width[0] / 2.0, 0],
                    [-self.width[1] / 2.0, -self.width[1] / 2.0, self.height],
                    [self.width[1] / 2.0, -self.width[1] / 2.0, self.height],
                    [self.width[1] / 2.0, self.width[1] / 2.0, self.height],
                    [-self.width[1] / 2.0, self.width[1] / 2.0, self.height],
                ]
            ).T

            indeces_of_tiles = np.array(
                [[0, 1, 4, 5], [1, 2, 5, 6], [2, 3, 6, 7], [3, 0, 7, 5]]
            )

            kwargs["edge_points"] = edge_points
            kwargs["indeces_of_tiles"] = indeces_of_tiles
            kwargs["is_boundary"] = True

            # import pdb; pdb.set_trace()     ##### DEBUG #####
            # Define range of 'cube'
            self.x_min = np.min(edge_points[0, :])
            self.x_max = np.max(edge_points[0, :])
            self.y_min = np.min(edge_points[1, :])
            self.y_max = np.max(edge_points[1, :])
            self.z_min = np.min(edge_points[2, :])
            self.z_max = np.max(edge_points[2, :])

        # define boundary functions
        center_position = np.array([0, 0, kwargs["edge_points"][2, -1] / 2.0])
        if sys.version_info > (3, 0):
            super().__init__(center_position=center_position, *args, **kwargs)
        else:
            super(DynamicBoundariesPolygon, self).__init__(
                center_position=center_position, *args, **kwargs
            )

        if indeces_of_flexibleTiles is None:
            self.indices_of_flexibleTiles = self.ind_tiles
        else:
            self.indeces_of_flexibleTiles = indeces_of_flexibleTiles

        self.is_boundary = True

        self.num_planes = 4

        self._inflation_parameter = np.zeros((self.num_planes, 2))
        self.inflation_parameter_old = np.zeros((self.num_planes, 2))
        if not inflation_parameter is None:
            self.set_inflation_parameter(inflation_parameter)

        self.time = time.time()
        # self.update(self.inflation_parameter)

        # self.dirs_evaluation = np.array([[0,1,0],
        # [1,0,0],
        # [0,1,0]
        # [1,0,0]]).T
        # SETUP --- Plane indices
        #
        # y  .---2---.
        # ^  |       |
        # |  3       1
        # |  |       |
        # |  .---0---.
        # .-------> x
        #
        #

    def set_inflation_parameter(self, value, plane_index=None):
        """Set inflation parameter"""

        ya = np.array([1e-2, 3e-2])
        if plane_index is None:
            self._inflation_parameter = (
                np.tile(ya, (self.num_planes, 1)) * np.tile(value, (2, 1)).T
            )
            self._inflation_percentage = value
        else:
            self._inflation_parameter[plane_index, :] = value

        # import pdb; pdb.set_trace()

    @property
    def inflation_parameter(self):
        return self._inflation_parameter

    @property
    def inflation_percentage(self):
        return self._inflation_percentage

    def _project_local_to_planeframe(self, position, plane_index):
        """Project from 3D into the frame parallel to a plane."""
        position_projected = np.array([0, 0, position[2]])

        if plane_index == 0:
            position_projected[0], position_projected[1] = (
                position[0],
                -position[1],
            )
        elif plane_index == 1:
            position_projected[0], position_projected[1] = (
                position[1],
                position[0],
            )
        elif plane_index == 2:
            position_projected[0], position_projected[1] = (
                -position[0],
                position[1],
            )
        elif plane_index == 3:
            position_projected[0], position_projected[1] = (
                -position[1],
                -position[0],
            )
        else:
            raise ValueError("Unknown plane index {}".format(plane_index))

        return position_projected

    def _project_planeframe_to_local(self, position_projected, plane_index):
        """Project from 2D + depth value into 3D vector based on plane."""

        position = np.array([0, 0, position_projected[2]])

        if plane_index == 0:
            position[0], position[1] = (
                position_projected[0],
                -position_projected[1],
            )
        elif plane_index == 1:
            position[0], position[1] = (
                position_projected[1],
                position_projected[0],
            )
        elif plane_index == 2:
            position[0], position[1] = (
                -position_projected[0],
                position_projected[1],
            )
        elif plane_index == 3:
            position[0], position[1] = (
                -position_projected[1],
                -position_projected[0],
            )
        else:
            raise ValueError("Unknown plane index {}".format(plane_index))

        return position

    def update_pos(t, dt, xlim=None, ylim=None, inflation_parameter=None, z_value=0):

        if inflation_parameter is None:
            freq = 2 * pi / 5
            inflation_parameter = np.sin(t * freq) * np.ones(self.num_plane)

        self.update(inflation_parameter, time_new=t, dt=dt)
        self.draw_obstacle(numPoints=50, z_val=z_value)

    def update(self, inflation_parameter, time_new=0, dt=None):
        self.inflation_parameter_old = self.inflation_parameter
        self.set_inflation_parameter(inflation_parameter)

        if dt is None:
            self.time_step = time_new - self.time
        else:
            self.time_step = dt
        self.time = time_new

    def draw_obstacle(
        self,
        num_points=50,
        z_val=None,
        inflation_parameters=None,
        in_global_frame=False,
    ):
        """Draw obstacle on the level z_val"""

        # print('z val', z_val)
        if z_val is None:
            raise NotImplementedError("Implement _drawing in 3D")

        if not in_global_frame:
            z_val = z_val + self.center_position[2]

        # Check z value
        if z_val > self.z_max or z_val < self.z_min:
            raise ValueError("Value of 'z' out of bound (z={})".format(z_val))

        self.boundary_points_local = np.zeros((self.dim, num_points))

        # Assume symmetric setup
        xy_max = self.get_flat_wall_value(z_val)

        # Iterator of the boundary point-list
        it_xobs = 0

        wz = (self.width[1] - self.width[0]) / self.height * z_val + self.width[0]

        # For each wall
        for it_plane in range(self.num_planes):
            if inflation_parameters is None:
                inflation_parameter = self.inflation_parameter[it_plane]
            else:
                inflation_parameter = inflation_parameter[it_plane]

            # Make sure there is exactly num_points in the array
            if it_plane < self.num_planes - 1:
                n_points = int(num_points / self.num_planes)
            else:
                n_points = num_points - it_xobs

            if it_plane == 0:
                x_vals = np.linspace(-wz / 2, wz / 2, n_points)
                y_vals = np.zeros(n_points)
            elif it_plane == 1:
                x_vals = np.zeros(n_points)
                y_vals = np.linspace(-wz / 2, wz / 2, n_points)
            elif it_plane == 2:
                x_vals = np.linspace(wz / 2, -wz / 2, n_points)  # reversed
                y_vals = np.zeros(n_points)
            elif it_plane == 3:
                x_vals = np.zeros(n_points)
                y_vals = np.linspace(wz / 2, -wz / 2, n_points)  # reversed

            pos = np.array([0, 0, z_val])

            for it_xy in range(n_points):
                pos[:2] = [x_vals[it_xy], y_vals[it_xy]]

                self.boundary_points_local[:, it_xobs] = self.get_surface_position(
                    pos,
                    inflation_parameter=inflation_parameter,
                    plane_index=it_plane,
                    in_global_frame=True,
                )

                it_xobs += 1

    def get_reference_direction(self, position, in_global_frame=False, normalize=True):
        """Return reference direction at position with respect to center line. (x=0, y=0)"""

        if in_global_frame:
            position = self.transform_global2relative(position)

        # No reference direction at center
        if not np.linalg.norm(position):
            return

        reference_direction = -(position - self.reference_point)
        reference_direction[2] = 0

        if in_global_frame:
            reference_direction = self.transform_global2relative_dir(
                reference_direction
            )

        return reference_direction

    def get_local_width(self, z_val=None, position=None, in_global_frame=False):
        """Returns the width of the pyramid at input height z."""
        if z_val is None:
            z_val = position[2]

        if not in_global_frame:
            z_val = z_val + self.center_position[2]

        return (self.width[1] - self.width[0]) / self.height * z_val + self.width[0]

    def get_surface_position(
        self,
        position,
        plane_index=None,
        inflation_parameter=None,
        in_global_frame=False,
        position_is_projected_on_plane=False,
    ):
        """Get the y value for a given position x&z.
        The position input can is 3D [x, y, z]."""

        position = np.copy(position)

        if not position_is_projected_on_plane:
            if in_global_frame:
                position = self.transform_global2relative(position)

            if plane_index is None:
                plane_index = self.get_closest_plane(position)

            position = self._project_local_to_planeframe(position, plane_index)

        if inflation_parameter is None:
            inflation_parameter = self.inflation_parameter[plane_index, :]

        x_val = position[0]
        z_val = position[2]

        # Evaluate z_val in global frame
        z_val = z_val + self.center_position[2]
        if z_val < self.z_min or z_val > self.z_max:
            raise ValueError(
                "Z value outside of defined region. (z={})".format(round(z_val, 4))
            )

        wz = self.get_local_width(z_val, in_global_frame=True)
        if abs(x_val) >= wz / 2.0:  # Project on 'edge'
            x_val = np.copysign(wz / 2.0, x_val)
            # Edge is never moving
            y_val = wz / 2.0
            position[0] = x_val

        else:
            # Are the nex lines useful?!?
            # inflation_parameter[1, 0] = inflation_parameter[0, 0] + inflation_parameter[2, 0]
            # inflation_parameter[3, 0] = inflation_parameter[1, 0]

            # Evaluation Parameteres
            za = np.array([3e-2, 12e-2])

            # dya = np.tile(ya, (n_planes, 1)) * np.tile(elevation_param, (2,1)).T
            ya = (
                inflation_parameter
                - ((self.width[1] - self.width[0]) / self.height * za.T + self.width[0])
                / 4
            )

            # Coefficients of 4th order polynom
            ca = np.array(
                [
                    [0, 0, 0, 0, 1],  # geometrical BC
                    [za[0] ** 4, za[0] ** 3, za[0] ** 2, za[0], 1],
                    [za[1] ** 4, za[1] ** 3, za[1] ** 2, za[1], 1],
                    [
                        self.height**4,
                        self.height**3,
                        self.height**2,
                        self.height,
                        1,
                    ],
                ]
            )

            ca = np.linalg.pinv(ca).dot(
                np.array([-self.width[0] / 4, ya[0], ya[1], -self.width[1] / 4])
            )

            yz = wz / 4.0

            # 4th order polynom evaluation
            a = np.array([z_val**4, z_val**3, z_val**2, z_val, 1]).dot(ca)
            b = (a + yz) / (wz / 2) ** 2  # -yz = -b*(wz/2)**2+a

            y_val = -b * (x_val**2).T + a - yz

            # Project y_val to >0
            y_val = y_val * (-1)

        position[1] = y_val

        if not position_is_projected_on_plane:
            position = self._project_planeframe_to_local(
                position, plane_index=plane_index
            )

        if in_global_frame:
            position = self.transform_relative2global(position)

        return position

    def get_flat_wall_value(self, z_value):
        """Get value in case of inflation value=0 or the edge points respectively.
        Assumption of pyramid shape (without edge). It takes the height as an input."""
        z_max = self.edge_points[2, -1]

        xy_at_min_z = self.edge_points[0, 1]
        xy_at_max_z = self.edge_points[0, -2]
        flat_wall = (xy_at_max_z - xy_at_min_z) / (2 * z_max) * (
            z_value + z_max
        ) + xy_at_min_z

        return flat_wall

    def get_point_of_plane(self, *args, **kwargs):
        # TODO: depreciated name; remove
        return self.get_surface_position(*args, **kwargs)

    def get_velocities(self, position, in_global_frame=False):
        # TODO: walls individually for 'smoother performance'
        position_wall = self.line_search_surface_point(position)
        position_new = self.get_point_of_plane(
            position_wall, inflation_parameter=self.inflation_parameter_old
        )
        position_old = self.get_point_of_plane(
            position_wall, inflation_parameter=self.inflation_parameter
        )

        linear_velocity = (position_old - position_new) / self.time_step

        direction_new = self.get_normal_direction(
            position_wall, inflation_parameter=self.inflation_parameter_old
        )
        direction_old = self.get_normal_direction(
            position_wall, inflation_parameter=self.inflation_parameter_old
        )

        angular_velocity = (direction_new - direction_old) / self.time_step

        import pdb

        pdb.set_trace()  ## DEBUG ##
        return linear_velocity, angular_velocity

    def get_edge_of_plane(self, plane_index, z_value, clockwise_plane_edge):
        # get_value_high -- decides about direction of value
        max_value = self.get_flat_wall_value(z_value)

        pos_2d = (
            np.array([max_value, -max_value])
            if clockwise_plane_edge
            else np.array([-max_value, -max_value])
        )

        rot_matrix = np.array([[0, -1], [1, 0]])

        edge_point = np.zeros(self.dim)
        edge_point[:2] = np.linalg.matrix_power(rot_matrix, plane_index).dot(pos_2d)
        edge_point[2] = z_value

        # print('edge_point', edge_point)
        return edge_point

    def _get_normal_direction_numerical_to_plane(
        self, position, plane_index, delta_dir_rel=1e-4
    ):
        """Numerical evaluation of normal for one plane individually.
        This is a specific case of the surface being a function of x&z (in the projected frame).

        Defined in local frame only.
        """
        step_size = delta_dir_rel * np.min(self.width)
        position_proj = self._project_local_to_planeframe(position, plane_index)

        normal_proj = np.ones(position_proj.shape)

        # Project onto corner
        wz = self.get_local_width(position_proj[2], in_global_frame=False)
        wz_margin = wz / 2.0 - step_size
        if abs(position_proj[0]) > wz_margin:
            position_proj[0] = np.copysign(wz_margin, position_proj[0])

        for it_dim in [0, 2]:
            pos_proj_high = np.copy(position_proj)
            pos_proj_high[it_dim] = pos_proj_high[it_dim] + step_size

            pos_proj_low = np.copy(position_proj)
            pos_proj_low[it_dim] = pos_proj_low[it_dim] - step_size

            pos_high = self.get_surface_position(
                position=pos_proj_high,
                plane_index=plane_index,
                position_is_projected_on_plane=True,
            )
            dist_high = pos_high[1]

            pos_low = self.get_surface_position(
                position=pos_proj_low,
                plane_index=plane_index,
                position_is_projected_on_plane=True,
            )
            dist_low = pos_low[1]

            # y = x*m + q  => m = dy/dx
            # tangent = [1 m].T
            normal_proj[it_dim] = (-1) * (dist_high - dist_low) / (2 * step_size)
        normal = self._project_planeframe_to_local(normal_proj, plane_index)

        DEBUG_FLAG = False
        if DEBUG_FLAG:
            plane_index = 0
            n_grid = 10
            x_vals = np.linspace(-wz / 2.0, wz / 2.0, n_grid)

            surf_pos = np.zeros((self.dim, n_grid))
            pos = np.copy(pos_proj_low)

            for ii in range(n_grid):
                pos[0] = x_vals[ii]
                surf_pos[:, ii] = self.get_surface_position(
                    position=pos,
                    plane_index=plane_index,
                    position_is_projected_on_plane=True,
                )
            plt.figure()
            plt.xlim([-0.2, 0.2])
            plt.ylim([-0.2, 0.2])

            plt.plot(surf_pos[0, :], surf_pos[1, :], marker="x")
            # import pdb; pdb.set_trace()     ##### DEBUG #####

        return normal

    def get_normal_direction(self, position, in_global_frame=False, weights=None):
        """Get the normal direction as the weighted sum of the different planes."""
        if in_global_frame:
            position = self.transform_global2relative(position)

        if weights is None:
            weights = self.get_plane_weights(position)
        ind_nonzero = weights > 0

        normal_directions = np.zeros((self.dim, self.n_planes))

        for ii in np.arange(self.n_planes)[ind_nonzero]:
            normal_directions[:, ii] = self._get_normal_direction_numerical_to_plane(
                position, plane_index=ii
            )

        mean_normal = get_directional_weighted_sum(
            null_direction=-self.get_reference_direction(position),
            directions=normal_directions[:, ind_nonzero],
            weights=weights[ind_nonzero],
        )

        if in_global_frame:
            position = self.transform_relative2global(position)

        return mean_normal

    def _get_tangent_2d_numerical(self, position, plane_index, normalize=True):
        """Get tangent direction by rotation normal 90 degrees."""
        normal = self._get_normal_direction_numerical_to_plane(position, plane_index)

        # normal  = normal[:2]
        tangent = np.array([-normal[1], normal[0]])

        mag_tangent = np.linalg.norm(tangent)
        if mag_tangent:
            tangent = tangent / mag_tangent
        else:
            raise ValueError(
                "Tangent vector has length zero (position={} and plane={}).".format(
                    position, plane_index
                )
            )

        return tangent

    # def _get_angle_to_edge_point(position, plane_index):
    # normal, tangent = _get_tangent2d_direction_plane_numerical(position, plane_index)

    def get_plane_weights(self, position, plane_index=None, in_global_frame=False):
        """Get the weight of the planes with respect to a distance measuer (plane angle)."""

        if in_global_frame:
            position = self.transform_global2relative(position)

        if plane_index is None:
            plane_index = self.get_closest_plane(position)

        # Project the position outside of the boundary
        gamma = self.get_gamma(position=position, plane_index=plane_index)
        if (self.is_boundary and gamma > 1) or (not self.is_boundary and gamma < 1):
            position_projected = np.copy(position)
            position_projected[:2] = position[:2] * (gamma * gamma)
        else:
            position_projected = np.copy(position)

        width_xz = self.get_local_width(position[2], in_global_frame=False)

        edge_points = np.zeros((self.dim, 2))

        edge_points[:, 0] = self._project_planeframe_to_local(
            position_projected=np.array([-width_xz / 2, width_xz / 2, position[2]]),
            plane_index=plane_index,
        )

        edge_points[:, 1] = self._project_planeframe_to_local(
            position_projected=np.array([width_xz / 2, width_xz / 2, position[2]]),
            plane_index=plane_index,
        )

        angles = np.zeros(self.n_planes)

        # Get angle of plane anti-clockwise
        it_plane = (plane_index - 1) % self.n_planes
        tangent = self._get_tangent_2d_numerical(
            position=edge_points[:, 0], plane_index=it_plane
        )

        # Flip to point inwards (to plane)
        tangent = (-1) * tangent

        angles[it_plane] = self.get_angle_2d(
            edge2position_vector=position_projected[:2] - edge_points[:2, 0],
            tangent_vector=tangent,
            clockwise_plane_edge=True,
        )

        # Get angle of main-plane
        point_on_plane = self.line_search_surface_point(
            position=position, plane_index=plane_index
        )

        # For the symmetric plane just get the closer plane
        it_close_edge = np.argmin(
            np.linalg.norm(edge_points - np.tile(point_on_plane, (2, 1)).T, axis=0)
        )

        # Pseudo-tangent (Planes are concave seen from outside)
        tangent = point_on_plane[:2] - edge_points[:2, it_close_edge]
        norm_tang = np.linalg.norm(tangent)

        if norm_tang > 1e-3:  # nonzero
            tangent = tangent / norm_tang
        else:
            tangent = self._get_tangent_2d_numerical(
                position=position, plane_index=plane_index
            )

            if it_close_edge == 1:
                tangent = (-1) * tangent

        if it_close_edge == 1:
            clockwise_plane_edge = True
        else:
            clockwise_plane_edge = False

        angles[plane_index] = self.get_angle_2d(
            edge2position_vector=position_projected[:2]
            - edge_points[:2, it_close_edge],
            tangent_vector=tangent,
            clockwise_plane_edge=clockwise_plane_edge,
        )

        # Get angle of plane clockwise
        it_plane = (plane_index + 1) % self.n_planes
        tangent = self._get_tangent_2d_numerical(
            edge_points[:, 1], plane_index=it_plane
        )

        angles[it_plane] = self.get_angle_2d(
            edge2position_vector=position_projected[:2] - edge_points[:2, 1],
            tangent_vector=tangent,
            clockwise_plane_edge=False,
        )

        weights = get_inverse_proprtional_weight(
            angles, distance_min=0, distance_max=pi
        )

        # import pdb; pdb.set_trace()     ##### DEBUG #####
        return weights

    def get_angle_2d(self, edge2position_vector, tangent_vector, clockwise_plane_edge):
        """Get the angle between two vectors.
        Normalization of vectors not necessary"""
        sign_angle = np.cross(tangent_vector, edge2position_vector)

        if not clockwise_plane_edge:
            sign_angle = (-1) * sign_angle

        # import pdb; pdb.set_trace()     ##### DEBUG #####
        arccos_value = edge2position_vector.T.dot(tangent_vector) / (
            np.linalg.norm(edge2position_vector) * np.linalg.norm(tangent_vector)
        )
        angle = np.arccos(min(max(arccos_value, -1), 1))  # avoid numerical errors

        if sign_angle < 0:
            angle = 2 * pi - angle
        return angle

    def get_gamma(
        self,
        position,
        plane_index=None,
        in_global_frame=False,
        gamma_power=1,
        position_is_projected_on_plane=False,
    ):
        """Get (dsitance) gamma value with respect to center.
        Only consider x&y i.e. distance to the center-axis (z=0)."""

        if in_global_frame:
            position = self.transform_global2relative(position)

        # Avoid zero divison
        if not np.linalg.norm(position[:2]):
            return sys.float_info.max

        if plane_index is None:
            # Automatically finds plane
            position_surface = self.line_search_surface_point(position)
        else:
            position_surface = self.line_search_surface_point(
                position,
                plane_index,
                position_is_projected_on_plane=position_is_projected_on_plane,
            )

        rad_local = np.linalg.norm(position_surface[:2])  # local frame
        dist_position = np.linalg.norm(position[:2])

        if self.is_boundary:
            Gamma = np.abs(rad_local / dist_position) ** gamma_power
        else:
            Gamma = np.abs(dist_position / rad_local) ** gamma_power

        return Gamma

    def line_search_surface_point(
        self,
        position,
        plane_index=None,
        position_is_projected_on_plane=False,
        in_global_frame=False,
        max_it=20,
        convergence_margin=1e-4,
    ):
        """Returns the intersection position with the surface in direction of the INPUT position."""
        # print('position global', position)
        # Find the surface point depending on the elevation
        if in_global_frame:
            position = self.transform_global2relative(position)
            # direction = self.transform_global2relative_dir(direction)

        if plane_index is None:
            plane_index = self.get_closest_plane(position)

        position = np.copy(position)

        ### DEBUG ###
        DEBUG_FLAG = False
        if DEBUG_FLAG:
            debug_list = [np.copy(position)]

        # Check if it's feasible search direction
        mag_xy_pos = np.linalg.norm(position[:2])
        if not mag_xy_pos:
            return position  # at center

        point_on_plane = self.get_surface_position(
            position,
            plane_index=plane_index,
            position_is_projected_on_plane=position_is_projected_on_plane,
        )

        # Choose if outside / inside search direction
        sign_dir = 1 if mag_xy_pos < np.linalg.norm(point_on_plane[:2]) else -1

        for ii in range(max_it):
            distance_plane = np.linalg.norm(position - point_on_plane)

            # import pdb; pdb.set_trace()     ##### DEBUG #####
            # Check convergence
            if distance_plane < convergence_margin:
                break

            distance_plane = distance_plane * sign_dir

            # Move to new position
            new_dist = distance_plane + mag_xy_pos
            position[:2] = new_dist / mag_xy_pos * position[:2]
            mag_xy_pos = new_dist

            if DEBUG_FLAG:
                plt.plot(0, 0, "kx")
                xy_val = self.get_local_width(position[2], in_global_frame=False)

                plt.plot(
                    [
                        -xy_val / 2,
                        xy_val / 2,
                        xy_val / 2,
                        -xy_val / 2,
                        -xy_val / 2,
                    ],
                    [
                        -xy_val / 2,
                        -xy_val / 2,
                        xy_val / 2,
                        xy_val / 2,
                        -xy_val / 2,
                    ],
                )
                plt.axis("equal")
                debug_list.append(np.copy(position))
                debug_list_ = np.array(debug_list)
                plt.plot(debug_list_[0, 0], debug_list_[0, 1], marker="o")
                plt.plot(debug_list_[:, 0], debug_list_[:, 1], marker="x")
                import pdb

                pdb.set_trace()
                # plt.close('all')

            point_on_plane = self.get_surface_position(
                position,
                plane_index=plane_index,
                position_is_projected_on_plane=position_is_projected_on_plane,
            )

        # plt.plot(debug_list_[:, 0], debug_list_[:, 1], marker="x")

        return position

    def get_closest_plane(self, position, in_global_frame=False):
        """Get the index of the closes plane.
        First one is at low y value (iterating in positive rotation around z)

        Assumption of squared symmetry along x&y and Walls aligned with axes."""
        # y  .---2---.
        # ^  |       |
        # |  3       1
        # |  |       |
        # |  .---0---.
        # .-------> x
        if in_global_frame:
            position = self.transform_global2relative(position)

        if position[0] > position[1]:
            return 0 if position[0] < (-position[1]) else 1
        else:
            return 2 if position[0] > (-position[1]) else 3

    def flexible_boundary_local_velocity(self, position, in_global_frame=False):
        if in_global_frame:
            position = self.transform_global2relative(position)
