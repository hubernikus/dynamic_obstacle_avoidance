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
        **kwargs
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

        if sys.version_info > (3, 0):  # TODO: remove in future
            super().__init__(*args, **kwargs)
        else:
            super(Polygon, self).__init__(*args, **kwargs)

        if ind_open is None:
            # TODO: implement in a useful manner to have doors etc. // or use ind_tiles
            ind_open = []

        (
            self.normal_vector,
            self.normalDistance2center,
        ) = self.calculate_normalVectorAndDistance(self.edge_points)

        # No go zone assuming a uniform margin around the obstacle
        self.edge_margin_points = self.edge_points

        # Extended in case that the reference point is outside the obstacle
        self.edge_reference_points = self.edge_margin_points

        # self.hull_points = self.edge_points
        self.margin_absolut = margin_absolut

        # Create shapely object
        # TODO update shapely position (!?)
        edge = self.edge_points

        edge = np.vstack((edge.T, edge[:, 0]))
        self._shapely = shapely.geometry.Polygon(edge).buffer(self.margin_absolut)

    @property
    def hull_edge(self):
        # TODO: remove // change
        return self.edge_points

    # @hull_edge.setter
    # def hull_edge(self, value):
    # self.edge_points = value

    @property
    def hull_points(self):
        return self.edge_reference_points

    @property
    def margin_absolut(self):
        return self._margin_absolut

    @margin_absolut.setter
    def margin_absolut(self, value):
        self._margin_absolut = value
        self.update_margin()

    def update_margin(self):
        if self._margin_absolut > 0:
            if self.dim == 2:
                norm_factor = -1 if self.is_boundary else 1

                self.edge_margin_points = np.zeros((self.dim, self.n_planes, 2))
                tangent1 = self.edge_points[:, 0] - self.edge_points[:, -1]
                tangent1 = tangent1 / np.linalg.norm(tangent1)

                for ii in range(self.n_planes):
                    tangent0 = tangent1
                    tangent1 = (
                        self.edge_points[:, (ii + 1) % self.n_planes]
                        - self.edge_points[:, ii]
                    )
                    tangent1 = tangent1 / np.linalg.norm(tangent1)

                    if (not self.is_boundary and np.cross(tangent0, tangent1) > 0) or (
                        self.is_boundary and np.cross(tangent0, tangent1) < 0
                    ):
                        self.edge_margin_points[:, ii, 0] = (
                            self.edge_points[:, ii]
                            + self.normal_vector[:, ii - 1]
                            * self._margin_absolut
                            * norm_factor
                        )
                        self.edge_margin_points[:, ii, 1] = (
                            self.edge_points[:, ii]
                            + self.normal_vector[:, ii]
                            * self._margin_absolut
                            * norm_factor
                        )
                    else:
                        angle_corner = np.arccos(tangent0.T.dot(-tangent1))

                        dir_mean = (
                            self.normal_vector[:, ii - 1] + self.normal_vector[:, ii]
                        )
                        dir_mean = dir_mean / np.linalg.norm(dir_mean) * norm_factor

                        margin_hat = self.margin_absolut / np.sin(angle_corner / 2.0)

                        self.edge_margin_points[:, ii, 0] = self.edge_margin_points[
                            :, ii, 1
                        ] = (self.edge_points[:, ii] + dir_mean * margin_hat)

            else:
                raise NotImplementedError("Not defined for dim>2.")
        else:
            # self.edge_margin_points = self.edge_points
            self.edge_margin_points = np.dstack((self.edge_points, self.edge_points))

        self.extend_hull_around_reference()

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

    def calculate_normalVectorAndDistance(self, edge_points=None):
        """Calculate Normal Distance and Distance to Edge points."""
        # TODO: is distance to surface still needed
        if isinstance(edge_points, type(None)):
            edge_points = self.edge_points

        # normal_vector = np.zeros(edge_points.shape)
        # normalDistance2center = np.zeros(edge_points.shape[1])
        normal_vector = np.zeros((edge_points.shape[0], self.n_planes))
        normalDistance2center = np.zeros(self.n_planes)

        if self.dim == 2:
            if len(edge_points.shape) == 2:
                for ii in range(self.n_planes):
                    normal_vector[:, ii] = (
                        edge_points[:, (ii + 1) % normal_vector.shape[1]]
                        - edge_points[:, ii]
                    )
            elif len(edge_points.shape) == 3:  #
                # TODO: does this make sense
                for ii in range(self.n_planes):
                    normal_vector[:, ii] = (
                        edge_points[:, (ii + 1) % normal_vector.shape[1], 0]
                        - edge_points[:, ii, 1]
                    )
            else:
                raise ValueError("")

            # From tangent to normal
            normal_vector = np.vstack((normal_vector[1, :], (-1) * normal_vector[0, :]))

        elif self.dim == 3:
            for ii in range(self.n_planes):
                tangent_0 = (
                    self.edge_points[:, self.ind_tiles[ii, 1]]
                    - self.edge_points[:, self.ind_tiles[ii, 0]]
                )

                tangent_1 = (
                    self.edge_points[:, self.ind_tiles[ii, 2]]
                    - self.edge_points[:, self.ind_tiles[ii, 1]]
                )

                normal_vector[:, ii] = np.cross(tangent_0, tangent_1)

                norm_mag = np.linalg.norm(normal_vector[:, ii])
                if norm_mag:  # nonzero
                    normal_vector[:, ii] = normal_vector[:, ii] / norm_mag
        else:
            raise ValueError("Implement for d>3.")

        if len(edge_points.shape) == 2:
            for ii in range(self.n_planes):
                normalDistance2center[ii] = normal_vector[:, ii].T.dot(
                    edge_points[:, ii]
                )
        else:
            for ii in range(self.n_planes):
                normalDistance2center[ii] = normal_vector[:, ii].T.dot(
                    edge_points[:, ii, 0]
                )

            if normalDistance2center[ii] < 0:
                normal_vector[:, ii] = (-1) * normal_vector[:, ii]
                normalDistance2center[ii] = (-1) * normalDistance2center[ii]
        # Normalize
        normal_vector = normal_vector / np.tile(
            np.linalg.norm(normal_vector, axis=0), (self.dim, 1)
        )

        return normal_vector, normalDistance2center

    def draw_obstacle(
        self,
        include_margin=False,
        n_curve_points=5,
        numPoints=None,
        add_circular_margin=False,
        point_density=2 * pi / 50,
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

        # for jj in range(x_obs_sf.shape[1]): # TODO replace for loop with numpy-math
        # x_obs_sf[:, jj] = self.rotMatrix.dot(x_obs_sf[:, jj]) + np.array([self.center_position])

        # TODO rename more intuitively
        # self.x_obs = self._boundary_points.T # Surface points
        # self.x_obs_sf = x_obs_sf.T # Margin points

    def get_reference_length(self):
        """Get a length which corresponds to the largest distance from the center of the obstacle."""
        return np.min(np.linalg.norm(self.edge_points, axis=0)) + self.margin_absolut

    def get_distances_and_normal_to_surfacePannels(
        self, position, edge_points=None, in_global_frame=False
    ):
        """
        Get the distance to all surfaces panels
        """

        if self.dim > 2:
            raise NotImplementedError("Higher dimensions lack functionality.")

        if edge_points is None:
            edge_points = self.edge_points

            if in_global_frame:
                position = self.transform_global2relative(position)

        distances = np.zeros(edge_points.shape[1])
        normal_vectors = np.zeros((self.dim, edge_points.shape[1]))
        for ii in range(edge_points.shape[1]):
            tangent_vector = edge_points[:, ii] - edge_points[:, (ii - 1)]
            normal_vectors[:, ii] = np.array([tangent_vector[1], -tangent_vector[0]])

            normal_vectors[:, ii] = normal_vectors[:, ii] / np.linalg.norm(
                normal_vectors[:, ii], axis=0
            )

            # dist_refPoint =  normal_vectors[:, ii].dot(self.reference_point)
            dist_edge = normal_vectors[:, ii].dot(edge_points[:, ii])
            dist_pos = normal_vectors[:, ii].dot(position)

            # distances[ii] = np.max(dist_edge - dist_refPoint, 0)
            distances[ii] = max(dist_pos - dist_edge, 0)

        return distances, normal_vectors

    def get_local_radius_point(self, direction, in_global_frame=False):
        """Get local radius points from relative direction."""
        if in_global_frame:
            # position = direction + self.center_position
            position = self.transform_global2relative_dir(direction)
        else:
            position = direction

        rad = self.get_distance_to_hullEdge(position=position)

        norm_pos = np.linalg.norm(position)
        if norm_pos:  # nonzero
            position = position / norm_pos

        surface_position = position * rad

        if in_global_frame:
            surface_position = self.transform_relative2global(surface_position)
        return surface_position

    def get_distance_to_hullEdge(self, position, in_global_frame=False):
        """Distance along the center-direction to the hull for a convex obstacle towards"""
        # TODO: change to reference-direction? What would this imply?

        if in_global_frame:
            position = self.transform_global2relative(position)

        multiple_positions = len(position.shape) > 1
        if multiple_positions:
            n_points = position.shape[1]
        else:
            n_points = 1
            position = position.reshape((self.dim, n_points))

        mag_position = np.linalg.norm(position, axis=0)

        zero_mag = mag_position == 0
        if np.sum(zero_mag):
            return -1

        dist2hull = np.ones(n_points) * (-1)

        if self.dim == 2:
            # TODO -- speed up!!!
            position_dir = position / mag_position
            if not self.margin_absolut:
                for jj in np.arange(n_points)[~zero_mag]:
                    # angle_to_reference = get_angle_space(
                    # null_direction=position_dir[:, jj], directions=self.edge_points)
                    angle_to_reference = get_angle_space_of_array(
                        null_direction_abs=position_dir[:, jj],
                        directions=self.edge_points,
                    )

                    magnitude_angles = np.linalg.norm(angle_to_reference, axis=0)

                    # angle_to_reference = np.arccos(ref)
                    ind_low = np.argmin(np.abs(magnitude_angles))

                    if magnitude_angles[ind_low] == 0:
                        dist2hull[jj] = np.linalg.norm(self.edge_points[:, ind_low])

                    else:
                        if angle_to_reference[0, ind_low] < 0:
                            ind_high = (ind_low + 1) % self.n_planes
                        else:
                            ind_high = ind_low
                            ind_low = (ind_high - 1) % self.n_planes

                        surface_dir = (
                            self.edge_points[:, ind_high] - self.edge_points[:, ind_low]
                        )

                        dist2hull[jj], dist_tangent = np.linalg.lstsq(
                            np.vstack((position_dir[:, jj], -surface_dir)).T,
                            self.edge_points[:, ind_low],
                            rcond=-1,
                        )[0]

                    # Gamma[jj] = mag_position[jj]/dist2hull

            else:
                # Obstacle has aboslut-margin
                for jj in np.arange(n_points)[~zero_mag]:
                    # directions=self.edge_reference_points[:, :, 0])
                    angle_to_reference = get_angle_space_of_array(
                        null_direction_abs=position_dir[:, jj],
                        directions=self.edge_reference_points[:, :, 0],
                    )

                    magnitude_angles = np.linalg.norm(angle_to_reference, axis=0)

                    ind_low = np.argmin(np.abs(magnitude_angles))
                    if angle_to_reference[0, ind_low] <= 0:
                        ind_high = (ind_low + 1) % self.n_planes
                    else:
                        ind_high = ind_low
                        ind_low = (ind_high - 1) % self.n_planes

                    if is_one_point(
                        self.edge_reference_points[:, ind_low, 0],
                        self.edge_reference_points[:, ind_low, 1],
                    ):  # one point
                        surface_dir = (
                            self.edge_reference_points[:, ind_high, 0]
                            - self.edge_reference_points[:, ind_low, 0]
                        )
                        edge_point = self.edge_reference_points[:, ind_low, 0]

                    else:
                        angle_hull_double_low = get_angle_space(
                            null_direction=position_dir[:, jj],
                            direction=self.edge_reference_points[:, ind_low, 1],
                        )

                        if angle_hull_double_low > 0:
                            # Solve quadratic equation to get intersection with (partial-) circle
                            vv = position_dir[:, jj]  # normalized
                            if (
                                self.reference_point_is_inside
                                or ind_low < self.ind_edge_ref
                            ):
                                ind_edge = ind_low
                            else:
                                ind_edge = ind_low - (
                                    self.n_planes - self.n_planes_edge
                                )
                            pp = self.edge_points[:, ind_edge]

                            AA = vv[0] * vv[0] + vv[1] * vv[1]
                            BB = -2 * (vv[0] * pp[0] + vv[1] * pp[1])
                            CC = (
                                pp[0] * pp[0]
                                + pp[1] * pp[1]
                                - self.margin_absolut * self.margin_absolut
                            )
                            D_sqrt = np.sqrt(BB * BB - 4 * AA * CC)

                            ref_magnitude = np.array(
                                [(-BB + D_sqrt), (-BB - D_sqrt)]
                            ) / (2 * AA)
                            if self.is_boundary:
                                dist2hull[jj] = np.min(ref_magnitude)
                            else:
                                dist2hull[jj] = np.max(ref_magnitude)

                            # Gamma[jj] = mag_position[jj]/dist2hull
                            continue

                        else:
                            surface_dir = (
                                self.edge_reference_points[:, ind_high, 0]
                                - self.edge_reference_points[:, ind_low, 1]
                            )
                            edge_point = self.edge_reference_points[:, ind_low, 1]

                    # Get distance to hull for both ifs
                    dist2hull[jj], dist_tangent = np.linalg.lstsq(
                        np.vstack((position_dir[:, jj], -surface_dir)).T,
                        edge_point,
                        rcond=-1,
                    )[0]

                    # Gamma[jj] = mag_position[jj]/dist2hull
                    # print('distances', dist2hull)

        if not multiple_positions:
            dist2hull = dist2hull[0]

        return dist2hull

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
                    local_radius = self.get_distance_to_hullEdge(position)
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

    def get_gamma_old(
        self,
        position,
        in_global_frame=False,
        norm_order=2,
        include_special_surface=True,
        gamma_type=GammaType.RELATIVE,
    ):
        """
        Get distance-measure from surface of the obstacle.
        INPUT: position: list or array of position
        OUTPUT
        RAISE ERROR:Function is partially defined for only the 2D case
        """
        # TOOD: redo different gamma types
        if in_global_frame:
            position = self.transform_global2relative(position)

        if gamma_type is GammaType.EUCLEDIAN:
            # Proportionally reduce gamma distance with respect to paramter
            dist2hulledge = self.get_distance_to_hullEdge(position)

            if dist2hulledge > 0:
                mag_position = np.linalg.norm(position)

                # Divide by laragest-axes factor to avoid weird behavior with elongated ellipses
                if self.is_boundary:
                    mag_position = dist2hulledge * dist2hulledge / mag_position

                Gamma = (mag_position - dist2hulledge) + 1

            else:
                Gamma = 0

        elif gamma_type == GammaType.RELATIVE:
            # TODO: extend rule to include points with Gamma < 1 for both cases
            # dist2hull = np.ones(self.edge_points.shape[1])*(-1)
            dist2hulledge = self.get_distance_to_hullEdge(position)

            if dist2hulledge:
                mag_position = np.linalg.norm(position)

                # Divide by laragest-axes factor to avoid weird behavior with elongated ellipses
                Gamma = mag_position / dist2hulledge

            else:
                Gamma = 0

            if self.is_boundary:
                pow_boundary_gamma = 2
                Gamma = self.get_boundaryGamma(Gamma) ** pow_boundary_gamma

        elif gamma_type == GammaType.OTHER:
            distances2plane = self.get_distance_to_hullEdge(position)

            delta_Gamma = np.min(distances2plane) - self.margin_absolut
            ind_outside = distances2plane > 0
            delta_Gamma = (
                np.linalg.norm(distances2plane[ind_outside], ord=norm_order)
                - self.margin_absolut
            )

            normalization_factor = np.max(self.normalDistance2center)
            # Gamma = 1 + delta_Gamma / np.max(self.axes_length)
            Gamma = 1 + delta_Gamma / normalization_factor

        else:
            raise TypeError("Unknown gmma_type {}".format(gamma_type))

        if not self.is_boundary:
            # TODO: implement better... in base class maybe(?!) / general
            warnings.warn("Unit-Gamma type for Polygon implemented.")
            dist_center = LA.norm(position)
            local_radius = dist_center / Gamma
            Gamma = dist_center - local_radius + 1

        return Gamma

    def get_normal_direction(
        self,
        position,
        in_global_frame=False,
        normalize=True,
        normal_calulation_type="distance",
    ):
        # breakpoint()
        if in_global_frame:
            position = self.transform_global2relative(position)

        mag_position = np.linalg.norm(position)
        if mag_position == 0:  # aligned with center, treat sepearately
            if self.is_boundary:
                return np.ones(self.dim) / self.dim
            else:
                return np.ones(self.dim) / self.dim

        if self.is_boundary:
            # Child and Current Class have to call Polygon
            Gamma = Polygon.get_gamma(self, position)

            if Gamma < 0:
                return -self.get_reference_direction(position)

            temp_position = Gamma * Gamma * position
        else:
            temp_position = np.copy(position)

        if self.reference_point_is_inside and self.margin_absolut == 0:
            temp_edge_points = np.copy(self.edge_points)
        else:
            temp_edge_points = self.edge_reference_points.reshape(2, -1)
            if sys.version_info > (3, 0):  # TODO: remove in future
                index_unique = np.unique(temp_edge_points, axis=1, return_index=True)[1]
            else:
                index_unique = []
                for jj in range(temp_edge_points.shape[1]):
                    is_unique = True
                    for kk in index_unique:
                        if is_one_point(
                            temp_edge_points[:, jj], temp_edge_points[:, kk]
                        ):
                            is_unique = False
                            break
                    if is_unique:
                        index_unique.append(jj)
                index_unique = np.array(index_unique)

            temp_edge_points = temp_edge_points[:, np.sort(index_unique)]

        (
            distances2plane,
            normal_vectors,
        ) = self.get_distances_and_normal_to_surfacePannels(
            temp_position, temp_edge_points
        )

        ind_outside = distances2plane > 0

        if not np.sum(ind_outside):  # zero value
            return self.get_reference_direction(position)

        distance2plane = ind_outside * np.abs(distances2plane)

        # pi is the maximum angle.
        angle2hull = np.ones(ind_outside.shape) * pi

        if self.dim > 2:
            raise NotImplementedError("Under construction for d>2.")

        for ii in np.arange(temp_edge_points.shape[1])[ind_outside]:
            # Calculate distance to agent-position
            # dir_tangent = (temp_edge_points[:, (ii+1)%temp_edge_points.shape[1]] - temp_edge_points[:, ii])
            dir_tangent = temp_edge_points[:, ii] - temp_edge_points[:, ii - 1]
            position2edge = temp_position - temp_edge_points[:, ii]

            if dir_tangent.T.dot(position2edge) < 0:
                distance2plane[ii] = np.linalg.norm(position2edge)
            else:
                # dir_tangent = -(temp_edge_points[:, (ii+1)%temp_edge_points.shape[1]] - temp_edge_points[:, ii])
                dir_tangent = -(temp_edge_points[:, ii] - temp_edge_points[:, ii - 1])
                position2edge = (
                    temp_position
                    - temp_edge_points[:, (ii + 1) % temp_edge_points.shape[1]]
                )
                if dir_tangent.T.dot(position2edge) < 0:
                    distance2plane[ii] = np.linalg.norm(position2edge)

            # Get closest point
            edge_points_temp = np.vstack(
                (temp_edge_points[:, ii - 1], temp_edge_points[:, ii])
            ).T

            # Calculate angle to agent-position
            ind_sort = np.argsort(
                np.linalg.norm(
                    np.tile(temp_position, (2, 1)).T - edge_points_temp, axis=0
                )
            )

            tangent_line = (
                edge_points_temp[:, ind_sort[1]] - edge_points_temp[:, ind_sort[0]]
            )
            position_line = temp_position - edge_points_temp[:, ind_sort[0]]

            angle2hull[ii] = self.get_angle2dir(position_line, tangent_line)

        distance_weights = 1
        angle_weights = self.get_angle_weight(angle2hull)

        weights = distance_weights * angle_weights  # TODO: multiplication needed?
        weights = weights / np.sum(weights)

        normal_vector = get_directional_weighted_sum(
            null_direction=position,
            directions=normal_vectors,
            weights=weights,
            normalize=False,
            normalize_reference=True,
        )

        if self.margin_absolut:  # Nonzero
            normal_vector = self.adapt_normal_to_arc_extension(position, normal_vector)

        # Make normal vector point away from obstacle
        normal_vector = (-1) * normal_vector

        if normalize:
            normal_vector = normal_vector / np.linalg.norm(normal_vector)

        if in_global_frame:
            normal_vector = self.transform_relative2global_dir(normal_vector)

        # In order to be pointing outside (!)
        return normal_vector

    def extend_hull_around_reference(
        self, edge_reference_dist=0.3, relative_hull_margin=0.1
    ):
        """
        Extend the hull of non-boundary, convex obstacles such that the reference point lies in
        inside the boundary again.
        """

        dist_max = self.get_maximal_distance() * relative_hull_margin
        mag_ref_point = np.linalg.norm(self.reference_point)

        # Reset number of planes & outside-edge
        self.n_planes = self.n_planes_edge
        self.edge_reference_points = copy.deepcopy(self.edge_margin_points)

        if mag_ref_point:
            reference_point_temp = self.reference_point * (1 + dist_max / mag_ref_point)

        if not self.reference_point_is_inside:  # Reset boundaryies
            (
                self.normal_vector,
                self.normalDistance2center,
            ) = self.calculate_normalVectorAndDistance(self.edge_points)

        if (
            (not self.is_boundary)
            and mag_ref_point
            and self.get_gamma(reference_point_temp) > 1
        ):
            # TODO add margin / here or somewhere else
            if not self.dim == 2:
                raise NotImplementedError("Not defined for d>2")

            if self.margin_absolut:
                for pp in range(self.n_planes):
                    t0 = (
                        self.edge_points[:, pp]
                        - self.edge_points[:, (pp - 1) % self.n_planes]
                    )
                    t1 = (
                        self.edge_points[:, (pp + 1) % self.n_planes]
                        - self.edge_points[:, pp]
                    )

                    n0 = np.array([t0[1], -t0[0]]) / np.linalg.norm(t0)
                    n1 = np.array([t1[1], -t1[0]]) / np.linalg.norm(t1)

                    outer_edge_point = self.edge_points[:, pp] + self.margin_absolut * (
                        n0 + n1
                    )

                    vec_ref = reference_point_temp - outer_edge_point
                    ref_mag = np.linalg.norm(vec_ref)

                    if np.cross(vec_ref, t0) > 0:
                        if np.cross(vec_ref, t1) > 0:
                            self.edge_reference_points = copy.deepcopy(
                                self.edge_margin_points
                            )

                            self.edge_reference_points[
                                :, pp, 0
                            ] = self.edge_reference_points[
                                :, pp, 1
                            ] = reference_point_temp

                            self.ind_edge_ref = self.n_planes  # Set large

                            qq = (pp + 1) % self.n_planes
                            it_edge = (pp + 1) % self.n_planes_edge

                        else:
                            self.edge_reference_points = np.hstack(
                                (
                                    np.reshape(
                                        self.edge_margin_points[:, :pp, :],
                                        (self.dim, -1, 2),
                                    ),
                                    np.reshape(
                                        np.tile(reference_point_temp, (2, 1)).T,
                                        (self.dim, 1, 2),
                                    ),
                                    np.reshape(
                                        self.edge_margin_points[:, pp:, :],
                                        (self.dim, -1, 2),
                                    ),
                                )
                            )
                            self.n_planes += 1
                            self.ind_edge_ref = pp

                            qq = (pp + 1) % self.n_planes
                            it_edge = pp

                        tt, tang_points = get_tangents2ellipse(
                            edge_point=self.edge_reference_points[:, pp, 0],
                            axes=[self.margin_absolut] * 2,
                            center_point=self.edge_points[:, it_edge],
                        )

                        # TODO: remove following check
                        self.edge_reference_points[:, qq, 0] = (
                            tang_points[:, 0]
                            if np.cross(tt[:, 0], tt[:, 1]) > 0
                            else tang_points[:, 1]
                        )

                        qq = (pp - 1) % self.n_planes
                        it_edge = (pp - 1) % self.n_planes_edge
                        tt, tang_points = get_tangents2ellipse(
                            edge_point=self.edge_reference_points[:, pp, 0],
                            axes=[self.margin_absolut] * 2,
                            center_point=self.edge_points[:, it_edge],
                        )
                        self.edge_reference_points[:, qq, 1] = (
                            tang_points[:, 0]
                            if np.cross(tt[:, 0], tt[:, 1]) < 0
                            else tang_points[:, 1]
                        )
                        break

                    elif np.cross(vec_ref, t1) < 0:
                        vec_ref = reference_point_temp - self.edge_points[:, pp]
                        if np.cross(vec_ref, t0) > 0 and np.cross(vec_ref, t1) > 0:
                            self.edge_reference_points = np.hstack(
                                (
                                    np.reshape(
                                        self.edge_margin_points[:, : (pp + 1), :],
                                        (self.dim, -1, 2),
                                    ),
                                    np.reshape(
                                        np.tile(reference_point_temp, (2, 1)).T,
                                        (self.dim, 1, 2),
                                    ),
                                    np.reshape(
                                        self.edge_margin_points[:, pp:, :],
                                        (self.dim, -1, 2),
                                    ),
                                )
                            )

                            self.n_planes += 2
                            self.ind_edge_ref = pp + 1

                            tt, tang_points = get_tangents2ellipse(
                                edge_point=self.edge_reference_points[:, pp + 1, 0],
                                axes=[self.margin_absolut] * 2,
                                center_point=self.edge_points[:, pp],
                            )

                            if np.cross(tt[:, 0], tt[:, 1]) > 0:
                                # TODO -- test and remove one
                                self.edge_reference_points[:, (pp), 1] = tang_points[
                                    :, 1
                                ]
                                self.edge_reference_points[
                                    :, (pp + 2) % self.n_planes, 0
                                ] = tang_points[:, 0]
                            else:
                                self.edge_reference_points[:, (pp), 1] = tang_points[
                                    :, 0
                                ]
                                self.edge_reference_points[
                                    :, (pp + 2) % self.n_planes, 0
                                ] = tang_points[:, 1]
                            break

                    # Adapted normal
                    (
                        self.normal_vector,
                        self.normalDistance2center,
                    ) = self.calculate_normalVectorAndDistance(self.hull_points)

            else:
                for pp in range(self.n_planes):
                    n0 = (
                        self.edge_points[:, pp]
                        - self.edge_points[:, (pp - 1) % self.n_planes]
                    )
                    n1 = (
                        self.edge_points[:, (pp + 1) % self.n_planes]
                        - self.edge_points[:, pp]
                    )

                    vec_ref = reference_point_temp - self.edge_points[:, pp]

                    if np.cross(vec_ref, n0) > 0:
                        if np.cross(vec_ref, n1) > 0:
                            self.edge_reference_points = copy.deepcopy(self.edge_points)
                            self.edge_reference_points[:, pp] = reference_point_temp
                        else:
                            self.edge_reference_points = np.hstack(
                                (
                                    self.edge_reference_points[:, :pp],
                                    np.reshape(self.reference_point, (self.dim, 1)),
                                    self.edge_reference_points[:, pp:],
                                )
                            )
                        break

                (
                    self.normal_vector,
                    self.normalDistance2center,
                ) = self.calculate_normalVectorAndDistance(self.edge_reference_points)

            self.reference_point_is_inside = False
            self.n_planes = self.edge_reference_points.shape[1]

        else:
            self.edge_reference_points = self.edge_margin_points  # include margin

            if not self.reference_point_is_inside:
                (
                    self.normal_vector,
                    self.normalDistance2center,
                ) = self.calculate_normalVectorAndDistance(self.edge_points)

            self.reference_point_is_inside = True
            self.n_planes = self.n_planes_edge
            self.ind_edge_ref = None

    def get_local_radius(self, position, in_global_frame=False):
        """Get local / radius or the surface intersection point by using shapely."""
        if in_global_frame:
            position = self.transform_global2relative(position)

        shapely_line = shapely.geometry.LineString([[0, 0], position])
        # try
        intersection = self._shapely.intersection(shapely_line).coords
        # except:
        # breakpoint()

        # If position is inside, the intersection point is equal to the position-point,
        # in that case redo the calulation with an extended line to obtain the actual
        # radius-point
        if np.allclose(intersection[-1], position):
            # Point is assumed to be inside
            point_dist = LA.norm(position)
            if not point_dist:
                # Return nonzero value to avoid 0-division conflicts
                return self.get_minimal_distance()

            # Make sure position is outside the boundary (random mutiple factor)
            position = position / point_dist * self.get_maximal_distance() * 5.0

            shapely_line = shapely.geometry.LineString([[0, 0], position])
            intersection = self._shapely.intersection(shapely_line).coords

        return LA.norm(intersection[-1])

    def get_gamma(
        self,
        position,
        in_global_frame=False,
        gamma_type=GammaType.EUCLEDIAN,
        gamma_distance=None,
    ):
        # gamma_distance is not used -> should it be removed (?!)
        if in_global_frame:
            position = self.transform_global2relative(position)

        dist_center = LA.norm(position)
        local_radius = self.get_local_radius(position)

        # Choose proporitional
        if gamma_type == GammaType.EUCLEDIAN:
            if dist_center < local_radius:
                # Return proportional inside to have -> [0, 1]
                gamma = dist_center / local_radius
            else:
                gamma = (dist_center - local_radius) + 1

        else:
            raise NotImplementedError("Implement othr gamma-types if desire.")
        return gamma
