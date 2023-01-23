"""
Polygon Obstacle for Avoidance Calculations
"""
# Author: Lukas Huber
# Date: created: 2020-02-28
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import sys
import warnings
import copy
import time

from math import pi

import numpy as np
from numpy import linalg as LA

import shapely
from shapely import affinity
from shapely.geometry import Point, MultiPoint
from shapely.geometry.polygon import LinearRing

from vartools.directional_space import (
    get_angle_space,
    get_angle_space_of_array,
)
from vartools.directional_space import get_directional_weighted_sum
from vartools.angle_math import (
    angle_is_in_between,
    angle_difference_directional,
)
from vartools.angle_math import *

from dynamic_obstacle_avoidance.utils import get_tangents2ellipse

from ._base import Obstacle, GammaType


def is_one_point(point1, point2, margin=1e-9):
    """Check if it the two points coincide [1-norm]"""
    return np.allclose(point1, point2, rtol=1e-9)


class Polygon(Obstacle):
    """Class to define Star Shaped Polygons

    Many calculations focus on 2D-problem.
    Generalization and extension to higher dimensions is possible, but not complete (yet).

    This class defines obstacles to modulate the DS around it
    At current stage the function focuses on Ellipsoids,
    but can be extended to more general obstacles.

    Attributes
    ----------
    edge_points:

    """

    def __init__(
        self,
        edge_points: np.ndarray,
        absolute_edge_position: bool = True,
        indeces_of_tiles: np.ndarray = None,
        ind_open: int = None,
        # reference_point=None,
        margin_absolut: float = 0,
        center_position: np.ndarray = None,
        *args,
        **kwargs,
    ):
        """
        Arguments
        ---------
        absolute_edge_position: bool to define if edge_points is in the local or absolute
            (global) frame
        """
        if center_position is None:
            center_position = np.sum(edge_points, axis=1) / edge_points.shape[1]
        else:
            center_position = np.array(center_position)

        kwargs["center_position"] = center_position

        self.edge_points = np.array(edge_points)
        if absolute_edge_position:
            self.edge_points = (
                self.edge_points
                - np.tile(center_position, (self.edge_points.shape[1], 1)).T
            )

        self.dim = center_position.shape[0]
        if self.dim == 2:
            # n_planes not really needed anymore...
            self.n_planes_edge = self.edge_points.shape[1]
            self.n_planes = self.edge_points.shape[1]  # with reference
            self.ind_edge_ref = None

        elif self.dim == 3:
            self.ind_tiles = indeces_of_tiles
            self.n_planes = indeces_of_tiles.shape[0]
            self.n_planes_edge = self.n_planes

            # TODO: How hard would it be to find flexible tiles?
        else:
            raise NotImplementedError(
                "Not yet implemented for dimensions higher than 3"
            )

        super().__init__(*args, **kwargs)

        # No go zone assuming a uniform margin around the obstacle
        self.edge_margin_points = self.edge_points

        # Extended in case that the reference point is outside the obstacle
        self.edge_reference_points = self.edge_margin_points

        # Create shapely object
        self.create_shapely()

        # Margin setting after first shapely initialization
        self.margin_absolut = margin_absolut

    @property
    def hull_edge(self):
        # TODO: remove // change
        return self.edge_points

    @property
    def edge_points_absolut(self):
        return self.transform_relative2global(self.edge_points)

    @property
    def hull_points(self):
        return self.edge_reference_points

    @property
    def margin_absolut(self):
        return self._margin_absolut

    @margin_absolut.setter
    def margin_absolut(self, value):
        self._margin_absolut = value

        if not self.is_reference_point_inside() and not self.is_boundary:
            self.extend_hull_around_reference()
        else:
            self.create_shapely()

    def get_characteristic_length(self):
        dist_edges = np.linalg.norm(
            self.edge_points
            - np.tile(self.center_position, (self.edge_points.shape[1], 1)).T,
            axis=0,
        )
        return np.mean(dist_edges)

    def get_maximal_distance(self):
        dist_edges = np.linalg.norm(
            self.edge_points
            - np.tile(self.center_position, (self.edge_points.shape[1], 1)).T,
            axis=0,
        )
        return np.max(dist_edges)

    def get_minimal_distance(self):
        dist_edges = np.linalg.norm(
            self.edge_points
            - np.tile(self.center_position, (self.edge_points.shape[1], 1)).T,
            axis=0,
        )
        return np.max(dist_edges)

    def create_shapely(self):
        """Creates (or updates) internal shapely in global frame."""
        if self.dimension > 2:
            raise NotImplementedError(
                f"Shapely shape not defined for d={self.dimension}"
            )

        # TODO: if it's deforming it needs to be adapted
        shapely_ = self.shapely.get(in_global_frame=False, margin=False)
        if shapely_ is None:
            shapely_ = LinearRing(self.edge_points.T)
            self.shapely.set(in_global_frame=False, margin=False, value=shapely_)

        if self.margin_absolut:
            if self.is_boundary:
                # self._shapely = poly_line.buffer(self.margin_absolut).exterior
                shapely_ = shapely_.buffer(self.margin_absolut).interiors
            else:
                shapely_ = shapely_.buffer(self.margin_absolut).exterior

        else:
            self.shapely.set(in_global_frame=False, margin=True, value=shapely_)

        if self.orientation:
            # Do orientation first just to ensure that it is around `self.center_position`
            shapely_ = affinity.rotate(
                shapely_, self.orientation * 180 / pi, origin=Point(0, 0)
            )

        shapely_ = affinity.translate(
            shapely_, self.center_position[0], self.center_position[1]
        )

        self.shapely.set(in_global_frame=True, margin=True, value=shapely_)

    def draw_obstacle(
        self,
        include_margin=False,
        n_curve_points=5,
        numPoints=None,
        add_circular_margin=False,
        point_density=(2 * pi / 50),
    ):
        # Compute only locally
        num_edges = self.edge_points.shape[1]

        self._boundary_points = self.edge_points

        if self.margin_absolut:
            self._boundary_points_margin = np.zeros((self.dim, 0))
            for ii in range(self.n_planes):
                if not is_one_point(
                    self.edge_reference_points[:, ii, 0],
                    self.edge_reference_points[:, ii, 1],
                ):
                    # If the reference point is outside the obstacle,
                    # an additional <<edge_reference_point has been created
                    if self.reference_point_is_inside or ii < self.ind_edge_ref:
                        it_edge = ii
                    else:
                        it_edge = ii - (self.n_planes - self.n_planes_edge)

                    v1 = (
                        self.edge_reference_points[:, ii, 0]
                        - self.edge_points[:, it_edge]
                    )
                    v1 = v1 / np.linalg.norm(v1)

                    v2 = (
                        self.edge_reference_points[:, ii, 1]
                        - self.edge_points[:, it_edge]
                    )
                    v2 = v2 / np.linalg.norm(v2)

                    angle1 = np.copysign(np.arccos(v1[0]), v1[1])
                    angle2 = np.copysign(np.arccos(v2[0]), v2[1])

                    direction = -1 if self.is_boundary else 1
                    angles = np.arange(
                        angle1,
                        angle1 + angle_difference_directional(angle2, angle1),
                        point_density * direction,
                    )

                    self._boundary_points_margin = np.hstack(
                        (
                            self._boundary_points_margin,
                            self.margin_absolut
                            * np.vstack((np.cos(angles), np.sin(angles)))
                            + np.tile(
                                self.edge_points[:, it_edge],
                                (angles.shape[0], 1),
                            ).T,
                        )
                    )

                self._boundary_points_margin = np.hstack(
                    (
                        self._boundary_points_margin,
                        self.edge_reference_points[:, ii, 1].reshape(self.dim, 1),
                    )
                )

        else:
            # self._boundary_points_with_margin = self._boundary_points
            self._boundary_points_margin = self._boundary_points
        # return

        if add_circular_margin:
            # TODO CHECK & ACTIVATE!!
            angles = np.linspace(0, 2 * pi, num_edges * n_curve_points + 1)
            obs_margin_cirlce = self.margin_absolut * np.vstack(
                (np.cos(angles), np.sin(angles))
            )

            self._boundary_points_with_margin = np.zeros((self.dim, 0))
            for ii in range(num_edges):
                self._boundary_points_with_margin = np.hstack(
                    (
                        self._boundary_points_with_margin,
                        np.tile(self.edge_points[:, ii], (n_curve_points + 1, 1)).T
                        + obs_margin_cirlce[
                            :,
                            ii * n_curve_points : (ii + 1) * n_curve_points + 1,
                        ],
                    )
                )
            self._boundary_points_with_margin = np.hstack(
                (
                    self._boundary_points_with_margin,
                    self._boundary_points_with_margin[:, 0].reshape(2, 1),
                )
            )

        else:
            dir_boundary_points = self._boundary_points / np.linalg.norm(
                self._boundary_points, axis=0
            )
            if self.is_boundary:
                self._boundary_points_with_margin = (
                    self._boundary_points - dir_boundary_points * self.margin_absolut
                )

            else:
                self._boundary_points_with_margin = (
                    self._boundary_points + dir_boundary_points * self.margin_absolut
                )

    def get_reference_length(self):
        """Get a length which corresponds to the largest distance from the
        center of the obstacle."""
        return np.min(np.linalg.norm(self.edge_points, axis=0)) + self.margin_absolut

    def adapt_normal_to_arc_extension(
        self, position, normal_vector, in_global_frame=False
    ):
        """
        Smooth extensions are created to account for the unidirectional margin around obstacles.
        This is considered here.
        """
        if self.dim > 2:
            raise NotImplementedError("Higher dimensions lack functionality.")

        if in_global_frame:
            position = self.transform_global2relative(position)

        angle_position = np.arctan2(position[1], position[0])

        for ii in range(self.edge_reference_points.shape[1]):

            if not is_one_point(
                self.edge_reference_points[:, ii, 0],
                self.edge_reference_points[:, ii, 1],
            ):
                # Round surface extension
                angle_low = np.arctan2(
                    self.edge_reference_points[1, ii, 0],
                    self.edge_reference_points[0, ii, 0],
                )
                angle_high = np.arctan2(
                    self.edge_reference_points[1, ii, 1],
                    self.edge_reference_points[0, ii, 1],
                )

                if angle_is_in_between(angle_position, angle_low, angle_high):
                    if self.reference_point_is_inside or ii < self.ind_edge_ref:
                        it_edge = ii
                    else:
                        # Reference point is outside, i.e. additional point is added
                        it_edge = ii - (self.n_planes - self.n_planes_edge)

                    # Direction
                    arc_normal = position - self.edge_points[:, it_edge]
                    mag_normal = np.linalg.norm(arc_normal)
                    if mag_normal:  # Nonzero
                        arc_normal = arc_normal / mag_normal
                    else:
                        arc_normal = self.get_reference_direction(position)

                    # Weight
                    local_radius = self.get_local_radius(
                        position, in_global_frame=False
                    )
                    delta_radius = max(np.linalg.norm(position) - local_radius, 0)
                    max_delta = 1.1 * self.margin_absolut  # > 1x
                    weight = max((1 - delta_radius) / max_delta, 0)

                    normal_vector = get_directional_weighted_sum(
                        null_direction=position,
                        directions=np.vstack((normal_vector, arc_normal)).T,
                        weights=np.array([(1 - weight), weight]),
                        normalize=False,
                        normalize_reference=True,
                    )
                    break

        if in_global_frame:
            normal_vector = self.transform_global2relative(normal_vector)

        return normal_vector

    def get_normal_direction(
        self,
        position,
        in_global_frame=False,
        normal_calulation_type="distance",
    ):
        if self.dim > 2:
            raise NotImplementedError("Under construction for d>2.")

        if in_global_frame:
            position = self.transform_global2relative(position)

        if self.margin_absolut:
            raise NotImplementedError("Not implemented for nonzero margin.")

        mag_position = LA.norm(position)
        if mag_position == 0:  # aligned with center, treat sepearately
            return np.ones(self.dim) / self.dim

        if self.is_boundary:
            # Child and Current Class have to call Polygon
            Gamma = Polygon.get_gamma(self, position)

            if Gamma < 0:
                return -self.get_reference_direction(position)

            temp_position = Gamma * Gamma * position
        else:
            temp_position = np.copy(position)

        temp_edge_points = self.shapely.get_local_edge_points()

        # Make sure that the edge-points are 'closed',
        # i.e. last point equal to first point
        if not np.allclose(temp_edge_points[:, 0], temp_edge_points[:, -1]):
            temp_edge_points = np.hstack((temp_edge_points, temp_edge_points[:, 0]))

        tangents, normals = self.get_tangents_and_normals_of_edge(temp_edge_points)
        distances2plane = self.get_normal_distance_to_surface_pannels(
            temp_position, temp_edge_points, normals
        )

        ind_outside = distances2plane > 0

        if not np.sum(ind_outside):  # zero value
            return self.get_outwards_reference_direction(position)

        # Pi is the maximum angle
        angle2hull = np.ones(ind_outside.shape) * pi

        for ii in np.arange(ind_outside.shape[0])[ind_outside]:
            # Get closest point
            corner_points = np.vstack(
                (temp_edge_points[:, ii + 1], temp_edge_points[:, ii])
            ).T

            # Calculate angle to agent-position
            ind_sort = np.argsort(
                LA.norm(np.tile(temp_position, (2, 1)).T - corner_points, axis=0)
            )

            tangent_line = corner_points[:, ind_sort[1]] - corner_points[:, ind_sort[0]]
            position_line = temp_position - corner_points[:, ind_sort[0]]
            angle2hull[ii] = self.get_angle2dir(position_line, tangent_line)

        # Weight onli dependent on the angle
        weights = self.get_angle_weight(angle2hull)

        normal_vector = get_directional_weighted_sum(
            null_direction=self.get_outwards_reference_direction(position),
            directions=normals,
            weights=weights,
        )

        if self.margin_absolut:  # Nonzero
            normal_vector = self.adapt_normal_to_arc_extension(position, normal_vector)

        # Invert to ensure pointing away from surface
        normal_vector = normal_vector / LA.norm(normal_vector)

        if in_global_frame:
            normal_vector = self.transform_relative2global_dir(normal_vector)

        return normal_vector

    def get_tangents_and_normals_of_edge(self, edge_points: np.ndarray):
        """Returns normal and tangent vector of tiles.
                -> could be an 'abstractmethod'

        Paramters
        ---------
        local_position: position in local_frame
        edge_points: (2D)-array of edge points, with the first equal to the last one

        Returns
        -------
        Normals: Tangent vectors (to the surfaces)
        Tangents: Normal vectors (to the surfaces)

        """
        if self.dim > 2:
            raise NotImplementedError("Higher dimensions lack functionality.")

        if self.margin_absolut:
            raise NotImplementedError(
                "Not implemented for dynamic polygon with absolut margin."
            )

        # Get tangents and normalize
        tangents = edge_points[:, 1:] - edge_points[:, :-1]
        tangents = tangents / np.tile(LA.norm(tangents, axis=0), (tangents.shape[0], 1))

        if np.cross(edge_points[:, 0], edge_points[:, 1]) > 0:
            normals = np.vstack((tangents[1, :], (-1) * tangents[0, :]))
        else:
            normals = np.vstack(((-1) * tangents[1, :], tangents[0, :]))

        return tangents, normals

    def get_normal_distance_to_surface_pannels(
        self, local_position: np.ndarray, edge_points: np.ndarray, normals: np.ndarray
    ):
        """
        Get the distance to all surfaces panels

        Paramters
        ---------
        local_position: position in local_frame
        edge_points: (2D)-array of edge points, with the first equal to the last one
        normals: Normal vectors (with one dimension less than edge_points)

        Returns
        -------
        (a tuple containing)
        Distances: distance to each of the normal directions
        """
        if self.dim > 2:
            raise NotImplementedError("Higher dimensions lack functionality.")

        if self.margin_absolut:
            raise NotImplementedError(
                "Not implemented for dynamic polygon with absolut margin."
            )

        distances = np.zeros(normals.shape[1])
        for ii in range(normals.shape[1]):
            # MAYBE: transform to array function
            dist_edge = normals[:, ii].dot(edge_points[:, ii])
            dist_pos = normals[:, ii].dot(local_position)
            distances[ii] = max(dist_pos - dist_edge, 0)
        return distances

    def extend_hull_around_reference(
        self,
        in_global_frame=False,
    ):
        """Extend hull around reference using shapely."""
        if self.is_boundary:
            raise NotImplementedError("Not defined for boundary")

        reference_point_temp = self.get_reference_point_with_margin()

        if (
            self.get_gamma(
                reference_point_temp,
                with_reference_point_expansion=False,
            )
            < 1
        ):
            # No displacement needed, since point within margins
            return

        self.edge_reference_points = copy.deepcopy(self.edge_margin_points)

        shapely_ = self.shapely.get(
            in_global_frame=False, margin=False, reference_extended=False
        )

        points = np.array(shapely_.xy)
        points = np.hstack((points, reference_point_temp.reshape(-1, 1)))

        new_polygon = MultiPoint(points.T).convex_hull

        if self.margin_absolut:
            if self.is_boundary:
                # self._shapely = poly_line.buffer(self.margin_absolut).exterior
                new_polygon = new_polygon.buffer(self.margin_absolut).interiors
            else:
                new_polygon = new_polygon.buffer(self.margin_absolut).exterior

        self.shapely.set(
            in_global_frame=False,
            margin=True,
            reference_extended=True,
            value=new_polygon,
        )

    def get_local_radius(
        self,
        position: np.ndarray,
        in_global_frame: bool = False,
        with_reference_point_expansion: bool = True,
    ) -> float:
        """Get local / radius or the surface intersection point by using shapely."""
        if in_global_frame:
            position = self.transform_global2relative(position)

        local_radius_point = self.get_local_radius_point(
            position, with_reference_point_expansion=with_reference_point_expansion
        )
        try:
            return LA.norm(local_radius_point)
        except:
            breakpoint()

    def get_local_radius_point(
        self,
        position: np.ndarray,
        in_global_frame: bool = False,
        with_reference_point_expansion: bool = True,
    ) -> np.ndarray:
        """Returns the point on the surface in direction of the position."""
        if in_global_frame:
            position = self.transform_global2relative(position)

        line_shapely = shapely.geometry.LineString([[0, 0], position])

        my_shapely = None

        if with_reference_point_expansion:
            my_shapely = self.shapely.get(
                in_global_frame=False, margin=True, reference_extended=True
            )

        if my_shapely is None:
            my_shapely = self.shapely.get(
                in_global_frame=False, margin=True, reference_extended=False
            )

        if my_shapely is None:
            raise Exception("No fitting shape for radius point found.")

        intersection = line_shapely.intersection(my_shapely)

        # If position is inside, the intersection point is equal to the
        # position-point, in that case redo the calulation with an extended line
        # to obtain the actual radius-point
        if intersection.is_empty:
            # if np.allclose(intersection[-1], position):
            # Point is assumed to be inside
            point_dist = LA.norm(position)
            if not point_dist:
                # Return nonzero value to avoid 0-division conflicts
                return self.get_minimal_distance()

            # Make sure position is outside the boundary (random mutiple factor)
            position = position / point_dist * self.get_maximal_distance() * 5.0

            line_shapely = shapely.geometry.LineString([[0, 0], position])
            intersection = my_shapely.intersection(line_shapely)

        return np.array([intersection.x, intersection.y])

    def get_gamma(
        self,
        position,
        in_global_frame=False,
        gamma_type=GammaType.EUCLEDIAN,
        gamma_distance=None,
        with_reference_point_expansion=True,
    ):
        # TODO: gamma, radius, hull edge
        # should be implemented in parent class & can be removed here...
        # gamma_distance is not used -> should it be removed (?!)
        if in_global_frame:
            position = self.transform_global2relative(position)

        local_radius = self.get_local_radius(
            position, with_reference_point_expansion=with_reference_point_expansion
        )
        dist_center = LA.norm(position)

        if self.is_boundary:
            position = self.mirror_local_position_on_boundary(
                position, local_radius=local_radius, pos_norm=dist_center
            )

        # Choose proporitional
        if gamma_type == GammaType.EUCLEDIAN:
            if dist_center < local_radius:
                # Return proportional inside to have -> [0, 1]
                gamma = dist_center / local_radius
            else:
                gamma = (dist_center - local_radius) + 1

        else:
            raise NotImplementedError("Implement othr gamma-types if desire.")

        if self.is_boundary:
            return 1 / gamma

        return gamma
