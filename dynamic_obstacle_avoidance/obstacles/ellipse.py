""" Ellipse Obstacle for Obstacle Avoidance and Visualization Purposes. """
# Author Lukas Huber
# Email lukas.huber@epfl.ch
# License BSD

import sys
import copy
import warnings
import time
from math import sin, cos, pi, ceil

import numpy as np
from shapely.geometry.point import Point
from shapely import affinity

import matplotlib.pyplot as plt  # TODO: remove for production

from vartools.angle_math import *
from vartools.angle_math import angle_modulo, angle_difference_directional_2pi
from vartools.directional_space import get_directional_weighted_sum

from dynamic_obstacle_avoidance.utils import *
from dynamic_obstacle_avoidance.obstacles import Obstacle


class Ellipse(Obstacle):
    """Ellipse type obstacle

    Attributes / Properties
    -----------------------
    Geometry specifi attributes are
    axes_length:
    curvature: float / array (list)
    """

    def __init__(
        self,
        axes_length=None,
        curvature=None,
        p=None,
        # margin_absolut=0,
        is_deforming=False,
        expansion_speed_axes=None,
        hull_with_respect_to_reference=False,
        *args,
        **kwargs,
    ):

        if expansion_speed_axes is None:
            is_deforming = is_deforming
        else:
            is_deforming = True

        super().__init__(*args, is_deforming=is_deforming, **kwargs)

        self.expansion_speed_axes = expansion_speed_axes

        if not axes_length is None:
            self.axes_length = np.array(axes_length)
        else:
            warnings.warn("No axis length given!")
            self.axes_length = np.ones((self.dim))

        if not curvature is None:
            self.curvature = curvature

        elif not p is None:
            warnings.warn(f"Argument <<{p}>> is depreciated.")
            self.curvature = p  # TODO: depreciated, remove

        else:
            self.curvature = np.ones((self.dim))

        self.hull_with_respect_to_reference = hull_with_respect_to_reference

        self.is_convex = True

        # No go zone assuming a uniform margin around the obstacle
        self.edge_margin_points = np.zeros((self.dim, 0))

        # Extended in case that the reference point is outside the obstacle
        self.edge_reference_points = self.edge_margin_points

        # Extended in case that the reference point is outside the obstacle
        # 1st pair of points corresponds to 'extension' of reference point
        # 2nd pair of points are the tangent points on the ellipse
        self.edge_reference_points = self.edge_margin_points

        self.ind_edge_ref = 0
        self.ind_edge_tang = 1

        self.create_shapely()

    @property
    def axes_length(self):
        return self._axes_length

    @axes_length.setter
    def axes_length(self, value):
        self._axes_length = value

    @property
    def expansion_speed_axes(self):
        return self._expansion_speed_axes

    @expansion_speed_axes.setter
    def expansion_speed_axes(self, value):
        if value is None:
            self._expansion_speed_axes = value
            return

        value = np.array(value)

        if not value.shape == (self.dim,):
            raise TypeError(
                "Wrong Expansion Type should be and array of length {}".format(self.dim)
            )
        self._expansion_speed_axes = value

    @property
    def p(self):  # TODO: remove
        breakpoint()
        warnings.warn("'p' is depreciated use 'curvature' instead")
        return self._curvature

    @p.setter
    def p(self, value):  # TODO: remove
        warnings.warn("'p' is depreciated use 'curvature' instead")
        self.curvature = value

    @property
    def curvature(self):
        return self._curvature

    @curvature.setter
    def curvature(self, value):
        if isinstance(value, (list)):  # TODO remove only allow one value...
            self._curvature = np.array(value)
        else:
            self._curvature = value

    @property
    def margin_absolut(self):
        return self._margin_absolut

    @margin_absolut.setter
    def margin_absolut(self, value):
        self._margin_absolut = value

    @property
    def axes_with_margin(self):
        if self.is_boundary:
            return self.axes_length - self.margin_absolut
        else:
            return self.axes_length + self.margin_absolut

    def get_minimal_distance(self):
        """Minimal distance or minimal radius."""
        return np.min(self.axes_length)

    def get_maximal_distance(self):
        """Minimal distance or maximal radius."""
        return np.sqrt(np.sum(self.axes_length * 2))

    def get_characteristic_length(self):
        """Get a characeteric (or maximal) length of the obstacle.
        For an ellipse obstacle,the longest axes."""
        return np.prod(self.axes_length + self.margin_absolut) ** (1 / self.dimension)

    def get_reference_length(self):
        """Get a characeteric (or maximal) length of the obstacle.
        For an ellipse obstacle,the longest axes."""
        return np.linalg.norm(self.axes_length) + self.margin_absolut

    def calculate_normalVectorAndDistance(self):
        normal_vector = np.zeros((self.dim, self.n_planes))
        normalDistance2center = np.zeros(self.n_planes)

        if self.hull_with_respect_to_reference:
            position = self.reference_point
        else:
            position = np.zeros(self.dim)

        for ii in range(self.n_planes):
            normal_vector[:, ii] = (
                self.edge_reference_points[:, ii, 0]
                - self.edge_reference_points[:, ii - 1, 1]
            )

            normal_vector[:, ii] = np.array(
                [
                    normal_vector[1, ii],
                    -normal_vector[0, ii],
                ]
            )

        for ii in range(self.n_planes):
            normalDistance2center[ii] = normal_vector[:, ii].T.dot(
                self.edge_reference_points[:, ii, 1]
            )

        normal_vector = normal_vector / np.tile(
            np.linalg.norm(normal_vector, axis=0), (self.dim, 1)
        )
        return normal_vector, normalDistance2center

    def get_distance_to_hullEdge(self, position, hull_edge=None, in_global_frame=False):
        if in_global_frame:
            position = self.pose.transform_position_to_relative(position)

        if hull_edge is None:
            hull_edge = self.hull_edge
            normal_vector = self.normal_vector

        else:
            normal_vector, dist = self.calculate_normalVectorAndDistance(hull_edge)

        hull_edge = hull_edge.reshape(self.dim, -1)
        n_planes = hull_edge.shape[1]
        if len(hull_edge.shape) < 2:
            vec_position2edge = np.tile(position - hull_edge, (n_planes, 1)).T
        else:
            vec_position2edge = np.tile(position, (n_planes, 1)).T - hull_edge

        distance2plane = np.sum((normal_vector * vec_position2edge), axis=0)

        if False:
            vec_position2edge = np.tile(position, (n_planes, 1)).T - self.tangent_points
            distance2plane = np.sum((self.normal_vector * vec_position2edge), axis=0)

        return distance2plane

    def position_is_in_direction_of_ellipse(self, position, in_global_frame=False):
        if in_global_frame:
            position = self.transform_global2relative(position)

        mag_position = np.linalg.norm(position, axis=0)
        position_dir = position / mag_position

        angle_tangents = np.zeros(self.edge_reference_points.shape[2])

        for ii in range(self.edge_reference_points.shape[2]):
            angle_tangents[ii] = np.arctan2(
                self.edge_reference_points[1, self.ind_edge_tang, ii],
                self.edge_reference_points[0, self.ind_edge_tang, ii],
            )

        angle_position = np.arctan2(position[1], position[0])

        angle_tang = angle_difference_directional_2pi(
            angle_tangents[1], angle_tangents[0]
        )
        angle_pos_tang0 = angle_difference_directional_2pi(
            angle_position, angle_tangents[0]
        )
        angle_tang1_pos = angle_difference_directional_2pi(
            angle_tangents[1], angle_position
        )

        margin_subtraction = 1e-12

        return (
            abs(angle_tang - (angle_tang1_pos + angle_pos_tang0)) < margin_subtraction
        )

    def get_gamma(
        self,
        position,
        in_global_frame=False,
        with_reference_point_expansion=True,
        gamma_type=None,
        gamma_distance=None,
        inverted=None,
        relative_gamma=True,
    ):
        """Returns gamma value of an ellipse shaped obstacle at position

        Parameters
        ----------
        position: array like position of size (dimension,)
        in_global_frame: If position input is in global frame, transform to local frame
        gamma_type: Different types of the distance measure-evaluation
        inverted: Enforce normal / inverted evaluation (if None use the object / boundary default)
        relative_gamma: Bool value to indicate if it is allowed to use the relative_value only

        Return
        ------
        Gamma: distance value gamma of float
        """
        if not gamma_type is None:
            # TODO: remove this before release (...)
            warnings.warn("Gammatype is not yet implemented for ellipse obstacle.")

        if relative_gamma and self.has_relative_gamma:
            # The relative gamma is assumed for to include notion of obstacle vs. boundary
            Gamma = self.get_relative_gamma_at_position(
                position, in_global_frame=in_global_frame
            )
            if Gamma is None:
                raise NotImplementedError(
                    "Relative Gamma evalaution not implemented "
                    "outside of evalaution point."
                )

            return Gamma

        if in_global_frame:
            position = self.pose.transform_position_to_relative(position)

        if gamma_type is not None:
            if (
                not gamma_type == "proportional"
                or gamma_distance is not None
                or self.gamma_distance is not None
            ):
                warnings.warn("Implement linear gamma type.")
                # raise Exception("Now it's enough")

        Gamma = np.sum(
            (np.abs(position) / self.axes_with_margin) ** (2 * self.curvature)
        ) ** (1.0 / (2 * np.mean(self.curvature)))

        if self.is_boundary:
            Gamma = 1.0 / Gamma
        return Gamma

    def get_normal_ellipse(self, position):
        """Return normal to ellipse surface"""
        # return (2*self.curvature/self.axes_length*(position/self.axes_length)**(2*self.curvature-1))
        return (
            2
            * self.curvature
            / self.axes_with_margin
            * (position / self.axes_with_margin) ** (2 * self.curvature - 1)
        )

    def get_angle2referencePatch(self, position, max_angle=pi, in_global_frame=False):
        """
        Returns an angle in [0, pi]
        """
        if in_global_frame:
            position = self.transform_global2relative(position)

        n_planes = self.edge_reference_points.shape[1]

        vec_position2edge = (
            np.tile(position, (n_planes, 1)).T - self.edge_reference_points[:, :, 0]
        )
        normalDistance2plane = np.sum((self.normal_vector * vec_position2edge), axis=0)

        angle2refencePatch = np.ones(n_planes) * max_angle

        for ii in np.arange(n_planes)[normalDistance2plane > 0]:
            # vec_position2edge[:, ii] /= np.linalg.norm(vec_position2edge[:, ii])

            # cos_position2edge = vec_position2edge[:, ii].T.dot(self.tangent_vector[:,ii])
            # angle2refencePatch[ii] = np.arccos(cos_position2edge)

            edge_points_temp = np.vstack(
                (
                    self.edge_reference_points[:, ii, 0],
                    self.edge_reference_points[:, ii - 1, 1],
                )
            ).T
            # Calculate angle to agent-position
            ind_sort = np.argsort(
                np.linalg.norm(np.tile(position, (2, 1)).T - edge_points_temp, axis=0)
            )
            tangent_line = (
                edge_points_temp[:, ind_sort[1]] - edge_points_temp[:, ind_sort[0]]
            )

            position_line = position - edge_points_temp[:, ind_sort[0]]
            angle2refencePatch[ii] = self.get_angle2dir(position_line, tangent_line)

        return angle2refencePatch

    def get_normal_direction(self, position, in_global_frame=False):
        """Return normal direction."""
        if in_global_frame:
            position = self.transform_global2relative(position)

        if self.hull_with_respect_to_reference:
            position = position - self.reference_point
            raise NotImplementedError(
                "Everything needs to be with respect to reference."
            )

        if self.reference_point_is_inside or self.position_is_in_direction_of_ellipse(
            position
        ):
            normal_vector = self.get_normal_ellipse(position)
        else:
            if False:
                # if self.margin_absolut:
                angle_ref = np.arctan2(
                    self.edge_reference_points[1, self.ind_edge_ref, 0],
                    self.edge_reference_points[0, self.ind_edge_ref, 0],
                )

                angle_position = np.arctan2(position[1], position[0])

                if angle_difference_directional(angle_ref, angle_position) > 0:
                    pass

                raise NotImplementedError()

            else:
                angle2referencePlane = self.get_angle2referencePatch(position)
                weights = self.get_angle_weight(angle2referencePlane)

                normal_vector = get_directional_weighted_sum(
                    null_direction=position,
                    directions=self.normal_vector,
                    weights=weights,
                    normalize=False,
                    normalize_reference=True,
                )

        if in_global_frame:
            normal_vector = self.transform_relative2global_dir(normal_vector)

        mag_norm = LA.norm(normal_vector)
        if mag_norm:
            normal_vector = normal_vector / mag_norm

        return normal_vector

    def get_gamma_ellipse(
        self, position, in_global_frame=False, axes=None, curvature=None
    ):
        if in_global_frame:
            position = self.transform_global2relative(position)
        # import pdb; pdb.set_trace()

        if len(position.shape) > 1:
            n_points = position.shape[1]
        else:
            n_points = -1

        if axes is None:
            axes = self.axes_with_margin

        if curvature is None:
            curvature = self.curvature

        if isinstance(curvature, (list, np.ndarray)):  # TODO: remove after testing
            warnings.warn("Wrong curvature dimension.")
            curvature = curvature[0]

        if n_points > 0:
            # Assumption -- zero-norm check already performed
            # norm_position = np.linalg.norm(position, axis=0)
            # np.sum( (position / np.tile(self.axes_with_margin, (n_points,1)).T ) ** (2*np.tile(self.curvature, (n_points,1)).T), axis=0)

            # rad_local = np.sqrt(1.0/np.sum((position/np.tile(self.axes_with_margin, (n_points, 1)).T)**np.tile(self.curvature, (n_points, 1)).T, axis=0))
            # return np.linalg.norm(position, axis=0)/(rad_local*np.linalg.norm(position, axis=0))
            # return 1.0/rad_local
            return np.sqrt(
                np.sum(
                    (position / np.tile(axes, (n_points, 1)).T)
                    ** np.tile(2 * curvature, (n_points, self.dim)).T,
                    axis=0,
                )
            )

        else:
            norm_position = np.linalg.norm(position)
            if norm_position == 0:
                return 0
            # rad_local = np.sqrt(1.0/np.sum(position/self.axes_with_margin**self.curvature) )
            # return 1.0/rad_local
            return np.sqrt(np.sum((position / axes) ** (2 * curvature)))

    def get_surface_derivative_angle(self, angle_space, in_global_frame=False):
        if self.dim == 2:
            if in_global_frame:
                angle_space = angle_space - self.orientation
                # import pdb; pdb.set_trace() ## DEBUG ##

            direction = np.array([np.cos(angle_space), np.sin(angle_space)])
            direction_perp = np.array([-direction[1], direction[0]])

        else:
            raise NotImplementedError("TODO for d>2")

        # TODO: should this be more general definition
        local_radius = self.get_intersection_with_surface(
            direction=direction, only_positive_direction=True
        )

        local_radius = np.linalg.norm(local_radius, 2)
        derivative = direction_perp * local_radius - 0.5 * direction * (
            local_radius**3
        ) * self.get_radius_derivative_direction(angle_space)

        if in_global_frame:
            derivative = self.transform_relative2global_dir(derivative)
        return derivative

    def get_radius_derivative_direction(self, angle_space):
        axes = self.axes_with_margin
        if self.dim == 2:
            return (
                2
                * np.cos(angle_space)
                * np.sin(angle_space)
                * (-1.0 / (axes[0] * axes[0]) + 1.0 / (axes[1] * axes[1]))
            )
        else:
            raise NotImplementedError("Implement for d>2")

    def get_local_radius_point(self, direction, in_global_frame=False):
        return self.get_intersection_with_surface(
            direction=direction,
            in_global_frame=in_global_frame,
            only_positive_direction=True,
        )

    def get_deformation_velocity(self, position, in_global_frame=True, delta_time=0.01):
        if in_global_frame:
            position = self.transform_global2relative(position)

        if not np.linalg.norm(position):  # zero vector
            return np.zeros(position.shape)

        axes_backup = copy.deepcopy(self.axes_length)

        # Numerical evaluation for simplicity
        point0 = self.get_local_radius_point(position)
        self.axes_length = self.axes_length + self.expansion_speed_axes * delta_time
        point1 = self.get_local_radius_point(position)

        # Reset axes
        self.axes_length = axes_backup

        surface_vel = (point1 - point0) / delta_time
        if in_global_frame:
            surface_vel = self.transform_relative2global_dir(surface_vel)
        return surface_vel

    def get_local_normal_direction(self, direction, in_global_frame=False):
        if in_global_frame:
            position = self.center_position + direction
        else:
            position = direction

        norm = self.get_normal_ellipse(position)
        if in_global_frame:
            norm = self.transform_relative2global(norm)
        return norm

    def get_intersection_with_surface(
        self,
        edge_point=None,
        direction=None,
        axes=None,
        center_ellipse=None,
        only_positive_direction=True,
        in_global_frame=False,
    ):
        """Intersection of (x_1/a_1)^2 +( x_2/a_2)^2 = 1 & x_2=m*x_1+c

        edge_point / c : Starting point of line
        direction / m : direction of line

        axes / a1 & a2: Axes of ellipse
        center_ellipse: Center of ellipse"""
        if in_global_frame:
            direction = self.transform_global2relative_dir(direction)

        if axes is None:
            axes = self.axes_with_margin

        if edge_point is None:
            # edge_point = np.zeros(self.dim)
            mag_x = 1.0 / np.sqrt(np.sum((direction / axes) ** 2))

            intersections = mag_x * direction

            if not only_positive_direction:
                intersections = np.tile(intersections, (2, 1)).T
                intersections[:, 1] = -intersections[:, 1]

            if in_global_frame:
                intersections = self.transform_relative2global(intersections)

            return intersections

        elif in_global_frame:
            edge_point = self.transform_global2relative(edge_point)

        # Dimension
        if self.dim > 2:
            raise NotImplementedError("Not yet implemented for D>2")

        if not center_ellipse is None:
            edge_point = edge_point - center_ellipse

        # x_1^2 * a_2^2 + (m*x_1+c)^2*a_1^2 = a_1^2*a_2^2
        # x_1^2 (a_2^2 + m^2*a_1^2) + x_1*(a_1^2*m*c*2) + c^2*a_1^2 - a_1^2*a_2^2
        if direction[0] == 0:
            if only_positive_direction:
                intersections = np.zeros(self.dim)
                intersections[0] = edge_point[0]
                intersections[1] = np.copysign(axes[1], direction[1])

            else:
                intersections = np.zeros((2, self.dim))
                intersections[0, 0] = intersections[0, 1] = edge_point[0]

                intersections[1, 0] = axes[1]
                intersections[1, 1] = -axes[1]

        else:
            m = direction[1] / direction[0]
            c = edge_point[1] - m * edge_point[0]

            A = (axes[0] * m) ** 2 + axes[1] ** 2
            B = 2 * axes[0] ** 2 * m * c
            C = (axes[0] * c) ** 2 - (axes[0] * axes[1]) ** 2

            D = B * B - 4 * A * C

            if D < 0:
                # import pdb; pdb.set_trace() ## DEBUG ##
                warnings.warn("No intersection found.")
                return

            sqrtD = np.sqrt(D)

            if only_positive_direction:
                intersections = np.zeros(self.dim)
                intersections[0] = (-B + sqrtD) / (2 * A)

                if (intersections[0] - edge_point[0]) / direction[0] < 0:
                    intersections[0] = (-B - sqrtD) / (2 * A)
                intersections[1] = intersections[0] * m + c

            else:
                intersections = np.zeros((self.dim, 2))
                intersections[0, :] = np.array([(-B + sqrtD), (-B - sqrt(D))]) / (2 * A)
                intersections[1, :] = intersections[0, :] * m + c
                # return intersections

        if not center_ellipse is None:
            intersections = (
                intersections + np.tile(center_ellipse, (intersections.shape[1], 1)).T
            )

        if in_global_frame:
            intersections = self.transform_relative2global(intersections)

        return intersections

    def get_local_radius_ellipse(
        self, position, in_global_frame=False, relative_center=None
    ):
        if in_global_frame:
            position = self.transform_global2relative(position)
        return self._get_local_radius_ellipse(position, relative_center)

    def _get_local_radius_ellipse(self, position, relative_center=None):
        """Get radius of ellipse in direction of position from the reference point"""
        # TODO: extend for actual relative center
        if relative_center is None:
            relative_center = np.zeros(self.dim)

        direction = position - relative_center

        intersection = self.get_intersection_with_surface(
            relative_center, direction, only_positive_direction=True
        )

        if relative_center is None:
            dist = np.linalg.norm(intersection)
        else:
            dist = np.linalg.norm(intersection - relative_center)
        return dist

    def get_local_radius_with_outside_reference(self, position: np.array) -> float:
        """Returns the local radius for case of reference point in the local frame
        everything happens in the local frame for a single position."""
        if self.position_is_in_direction_of_ellipse(position):
            radius = self._get_local_radius_ellipse(position)
        else:
            angle_position = np.arctan2(position[1], position[0])

            dist_intersect = -1
            for ii, sign in zip(range(self.n_planes), [1, -1]):
                angle_ref = np.arctan2(
                    self.edge_reference_points[1, self.ind_edge_ref, ii],
                    self.edge_reference_points[0, self.ind_edge_ref, ii],
                )

                if sign * angle_difference_directional(angle_ref, angle_position) >= 0:
                    surface_dir = (
                        self.edge_reference_points[:, self.ind_edge_ref, ii]
                        - self.edge_reference_points[:, self.ind_edge_tang, 1 - ii]
                    )

                    dist_intersect, dist_tangent = np.linalg.lstsq(
                        np.vstack((position[:], -surface_dir)).T,
                        self.edge_reference_points[:, self.ind_edge_ref, ii],
                        rcond=-1,
                    )[0]

                    dist_intersect = dist_intersect * np.linalg.norm(position[:])

            if dist_intersect < 0:  #
                if not margin_absolut:
                    raise ValueError("Negative value not possible.")

                intersections = self.get_intersection_with_surface(
                    edge_point=np.zeros(self.dim),
                    direction=position[:],
                    axes=np.ones(self.dim) * margin_absolut,
                )

                # self.get_intersectionWithEllipse()

                distances = np.linalg.norm(intersections, axis=0)
                dist_intersect = np.max(distances)
            radius = dist_intersect
        return radius

    def _get_local_radius(self, position, relative_center=None):
        # TODO: test for margin / reference point
        # TODO: improve speed
        if relative_center is None:
            relative_center = np.zeros(self.dim)

        # TODO: remove and make actual reference point
        if relative_center is None:
            relative_center = np.zeros(self.dimension)

        margin_absolut = self.margin_absolut
        n_points = position.shape[1]
        radius = np.zeros(position.shape[1])

        # Original Gamma
        if self.dim == 2:
            if self.reference_point_is_inside:
                for pp in range(n_points):
                    radius[pp] = self._get_local_radius_ellipse(position[:, pp])
            else:
                for pp in range(n_points):
                    radius[pp] = self.get_local_radius_with_outside_reference(
                        position[:, pp]
                    )
        return radius

    def create_shapely(self):
        """Create object (shape) based on the shapely library."""
        if self.dim != 2:
            raise NotImplementedError("Shapely object only existing for 2D")

        # Point is set at zero, and only moved when needed
        circ = Point(np.zeros(self.dimension)).buffer(1)

        axes = self.axes_length
        shapely_outside = affinity.scale(circ, axes[0], axes[1])
        self.shapely.set(in_global_frame=False, margin=False, value=shapely_outside)

        axes = self.axes_with_margin
        shapely_outside = affinity.scale(circ, axes[0], axes[1])
        self.shapely.set(in_global_frame=False, margin=True, value=shapely_outside)

    def draw_obstacle(
        self,
        n_grid=50,
        update_core_boundary_points=True,
        point_density=2 * pi / 50,
        numPoints=None,
    ):
        """Creates points for obstacle and obstacle margin.
        n_grid is used for 3D drawing"""
        if numPoints is not None:
            warnings.warn("'numPoints' depreciated - use 'n_grid' instead.")
        p = self.curvature
        a = self.axes_length

        if update_core_boundary_points:
            if self.dim == 2:
                theta = np.linspace(-pi, pi, num=n_grid)
                # resolution = n_grid # Resolution of drawing #points
                boundary_points = np.zeros((self.dim, n_grid))
                boundary_points[0, :] = a[0] * np.cos(theta)
                boundary_points[1, :] = np.copysign(a[1], theta) * (
                    1 - np.cos(theta) ** (2 * p[0])
                ) ** (1.0 / (2.0 * p[1]))
                self.boundary_points = boundary_points

            elif self.dim == 3:
                n_grid = [n_grid, ceil(n_grid / 2)]
                theta, phi = np.meshgrid(
                    np.linspace(-pi, pi, num=n_grid[0]),
                    np.linspace(-pi / 2, pi / 2, num=n_grid[1]),
                )  #
                n_grid = n_grid[0] * n_grid[1]
                # resolution = n_grid # Resolution of drawing #points
                theta = theta.T
                phi = phi.T

                boundary_points = np.zeros((self.dim, n_grid))
                boundary_points[0, :] = (a[0] * np.cos(phi) * np.cos(theta)).reshape(
                    (1, -1)
                )
                boundary_points[1, :] = (
                    a[1]
                    * np.copysign(1, theta)
                    * np.cos(phi)
                    * (1 - np.cos(theta) ** (2 * p[0])) ** (1.0 / (2.0 * p[1]))
                ).reshape((1, -1))
                boundary_points[2, :] = (
                    a[2]
                    * np.copysign(1, phi)
                    * (
                        1
                        - (
                            np.copysign(1, theta)
                            * np.cos(phi)
                            * (1 - 0 ** (2 * p[2]) - np.cos(theta) ** (2 * p[0]))
                            ** (1 / (2 ** p[1]))
                        )
                        ** (2 * p[1])
                        - (np.cos(phi) * np.cos(theta)) ** (2 * p[0])
                    )
                    ** (1 / (2 * p[2]))
                ).reshape((1, -1))
                self.boundary_points_local = boundary_points

            else:
                raise NotImplementedError("Not yet implemented for dimension >3")

        if self.dim == 2:
            boundary_points_margin = np.zeros((self.dim, 0))
            if not self.reference_point_is_inside:
                angle_tangents = np.zeros(2)
                for ii in range(angle_tangents.shape[0]):
                    angle_tangents[ii] = np.arctan2(
                        self.edge_reference_points[1, self.ind_edge_tang, ii],
                        self.edge_reference_points[0, self.ind_edge_tang, ii],
                    )
                    # angle_tangents[ii] = np.arctan2(self.tangent_points[1, ii], self.tangent_points[0, ii])
                if angle_tangents[0] < angle_tangents[1]:
                    theta = np.arange(
                        angle_tangents[0], angle_tangents[1], point_density
                    )
                else:
                    theta = np.arange(
                        angle_tangents[0],
                        angle_tangents[1] + 2 * pi,
                        point_density,
                    )
                    theta = angle_modulo(theta)

            elif not self.margin_absolut:
                # No boundary and reference point inside
                self.boundary_points_margin_local = self.boundary_points
                return

            a = self.axes_with_margin

            # Margin points
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            power = 2 * self.curvature[0]
            # try:
            factor = 1.0 / (
                (cos_theta / a[0]) ** power + (sin_theta / a[1]) ** power
            ) ** (1.0 / power)
            # except:
            # breakpoint()

            if self.reference_point_is_inside:
                boundary_points_margin = np.hstack(
                    (
                        boundary_points_margin,
                        factor * np.vstack((cos_theta, sin_theta)),
                    )
                )

            else:
                boundary_points_margin = np.hstack(
                    (
                        boundary_points_margin,
                        np.reshape(
                            self.edge_reference_points[:, self.ind_edge_ref, 1],
                            (self.dim, 1),
                        ),
                        factor * np.vstack((cos_theta, sin_theta)),
                        np.reshape(
                            self.edge_reference_points[:, self.ind_edge_tang, 1],
                            (self.dim, 1),
                        ),
                    )
                )

            self.boundary_points_margin_local = boundary_points_margin

        elif self.dim == 3:
            axes_length = self.axes_length

            # Set of all spherical angles:
            # n_u = int(np.floor(np.sqrt(n_grid)))
            # n_v = int(np.ceil(n_grid/n_u))
            n_u = n_v = n_grid

            u = np.linspace(0, 2 * np.pi, n_u)
            v = np.linspace(0, np.pi, n_v)

            boundary_points_local = []

            # Cartesian coordinates that correspond to the spherical angles:
            # (this is the equation of an ellipsoid):
            # self.boundary_points
            boundary_points_local.append(
                axes_length[0] * np.outer(np.cos(u), np.sin(v))
            )
            boundary_points_local.append(
                axes_length[1] * np.outer(np.sin(u), np.sin(v))
            )
            boundary_points_local.append(
                axes_length[2] * np.outer(np.ones_like(u), np.cos(v))
            )

            boundary_points_local = np.array(boundary_points_local).reshape(
                self.dim, n_grid * n_grid
            )
            boundary_points_global = self.transform_relative2global(
                boundary_points_local
            )

            return boundary_points_global.reshape(self.dim, n_grid, n_grid)

        else:
            raise ValueError(
                "Drawing of obstacle not implemented in high-dimensional space."
            )

        return self.transform_relative2global(self._boundary_points_margin)

    def get_radius_of_angle(self, angle, in_global_frame=False):
        """Extend the hull of non-boundary, convex obstacles such that the reference point lies in
        inside the boundary again.
        """
        if in_global_frame:
            position = transform_polar2cartesian(
                magnitude=10, angle=angle - self.orientation
            )
        else:
            position = transform_polar2cartesian(magnitude=1, angle=angle)

        gamma = self.get_gamma(position, gamma_type="proportional")
        return np.linalg.norm(position) / gamma

    def extend_hull_around_reference(
        self, edge_reference_dist=0.3, relative_hull_margin=0.1
    ):
        """Extend the hull of non-boundary, convex obstacles such that the reference point
        lies in inside the boundary again."""
        self.reference_point_is_inside = True  # Default assumption

        dist_max = self.get_maximal_distance() * relative_hull_margin
        mag_ref_point = np.linalg.norm(self.reference_point)

        if mag_ref_point:
            reference_point_temp = self.reference_point * (1 + dist_max / mag_ref_point)
            # gamma = self.get_gamma(reference_point_temp, np.zeros(self.dim))
            gamma = self.get_gamma(reference_point_temp)

        if mag_ref_point and gamma > 1:
            tt, tang_points = get_tangents2ellipse(
                edge_point=reference_point_temp, axes=self.axes_with_margin
            )

            # tang_points[:, 0], tang_points[:, 1] = tang_points[:, 1], tang_points[:, 0]
            tang_points = np.flip(tang_points, axis=1)
            # = np.flip(tang_points, axis=1)

            if np.cross(tang_points[:, 0], tang_points[:, 1]) > 0:  # TODO: remove
                tang_points = np.flip(tang_points, axis=1)
                warnings.warn("Had to flip. Reverse tangent order [1]<-->[0]! ")

            self.edge_reference_points = np.zeros((self.dim, 2, 2))
            self.edge_reference_points[:, self.ind_edge_ref, :] = np.tile(
                reference_point_temp, (2, 1)
            ).T

            self.edge_reference_points[:, self.ind_edge_tang, :] = tang_points

            self.reference_point_is_inside = False
            self.n_planes = 2
            (
                self.normal_vector,
                self.normalDistance2center,
            ) = self.calculate_normalVectorAndDistance()

            self.tangent_vector = np.zeros(self.normal_vector.shape)
            for ii in range(self.normal_vector.shape[1]):
                self.tangent_vector[:, ii] = [
                    -self.normal_vector[1, ii],
                    self.normal_vector[0, ii],
                ]
        else:
            self.n_planes = 0

    def update_deforming_obstacle(self, delta_time):
        """Update step."""
        self.axes_length = self.axes_length + self.expansion_speed_axes * delta_time

        self.draw_obstacle()


class Sphere(Ellipse):
    """Ellipse obstacle with equal axes"""

    def __init__(self, radius=None, axes_length=None, *args, **kwargs):
        if not radius is None:
            axes_length = np.array([radius, radius])
        elif not radius is None:
            if radius.shape[0] == 1:
                axes_length = np.array([axes_length, axes_length])
        else:
            raise RuntimeError("No radius input.")

        if sys.version_info > (3, 0):
            super().__init__(axes_length=axes_length, *args, **kwargs)
        else:
            super(Ellipse, self).__init__(axes_length=axes_length, *args, **kwargs)
            # super(Ellipse, self).__init__(*args, **kwargs) # works for python < 3.0?!

        if self.is_deforming:
            self.radius_old = copy.deepcopy(self.radius)

    # Axes length is equal to radius for an circular obtject
    @property
    def radius(self):
        return self.axes_length[0]

    @radius.setter
    def radius(self, value):
        self.axes_length = np.ones(self.dim) * value

    @property
    def radius_with_margin(self):
        return self.axes_with_margin[0]

    @property
    def inflation_speed_radial(self):
        # positive when expanding, negative when shrinking
        return self._inflation_speed_radial

    @inflation_speed_radial.setter
    def inflation_speed_radial(self, value):
        self._inflation_speed_radial = value

    def get_characteristic_length(self):
        return self.radius + self.margin_absolut

    def _get_local_radius(self, *args, **kwargs):
        return self.radius_with_margin

    def get_deformation_velocity(self, position, in_global_frame=False):
        """Get relative velocity of a boundary point.
        This is zero if the deformation would be pulling."""

        if in_global_frame:
            raise NotImplementedError()

        norm_pos = np.linalg.norm(position)
        if norm_pos:  # nonzero
            position = position / norm_pos

        if self.is_boundary:
            # Only consider when increasing
            deformation_vel = min(self.inflation_speed_radial, 0) * position
        else:
            # Only consider when incrreasing
            deformation_vel = max(self.inflation_speed_radial, 0) * position
        return deformation_vel

    def update_deforming_obstacle(
        self, radius_new, position, orientation, time_current=None
    ):
        """Update an obstacle which can also deform"""
        if time_current is None:
            time_current = time.time()

        dt = time_current - self.timestamp

        self.inflation_speed_radial = (self.radius - self.radius_old) / dt
        self.radius_old = copy.deepcopy(self.radius)
        self.radius = radius_new

        self.update_position_and_orientation(
            position=position,
            orientation=orientation,
            time_current=time_current,
        )


class CircularObstacle(Sphere):
    # Depreciated remove
    pass
