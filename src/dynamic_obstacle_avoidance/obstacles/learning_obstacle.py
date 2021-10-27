""" 
Learn obstacles with different methods. 
"""
# Author: Lukas Huber
# Created:  2020-08-19
# Email: lukas.huber@epfl.ch

import sys
import warnings
from math import sin, cos, pi, ceil
import time

import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt  # REMOVE AFTER DEVELOPMENT

from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import *

import sklearn
from sklearn.cluster import DBSCAN

debug_viz = False


def get_obstacle_from_scan(
    sensor_data,
    center_dist_scaling=1.5,
    clustering_minimal_distance=0.2,
    clustering_min_samples=5,
    input_is_polar=True,
    regression_obstacle=True,
    center_position_offset=1.1,
    cutoff_distance=6,
    print_timing=False,
):

    if input_is_polar:
        magnitude = sensor_data["magnitude"]
        angle = sensor_data["angle"]
        points_cartesian = transform_polar2cartesian(magnitude, angle)
    else:
        points_cartesian = sensor_data

    if input_is_polar:
        ind_close = np.linalg.norm(points_cartesian, axis=0) < cutoff_distance
        magnitude = magnitude[ind_close]
        angle = angle[ind_close]
        points_cartesian = points_cartesian[:, ind_close]

    # if debug_viz:
    # plt.plot(points_cartesian[0, (~ind_close)], points_cartesian[1,(~ind_close)], 'r.')

    start_time_clustering = time.time()
    db_cluster = DBSCAN(eps=0.2, min_samples=5, metric="euclidean").fit(
        points_cartesian
    )
    time_clustering = time.time() - start_time_clustering

    cluster_labels = db_cluster.labels_

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    time_obstacleLearning = 0
    obs_list = []

    for ii in range(n_clusters):
        ind_ii = cluster_labels == ii

        if input_is_polar:
            mean_angle = np.mean(angle[ind_ii] + pi) - pi
            # mean_mag = np.mean(magnitude[ind_ii])
            # mean_mag = mean_mag*center_position_offset
            ref_mag = np.max(magnitude[ind_ii])

            center_position = transform_polar2cartesian(
                magnitude=ref_mag, angle=mean_angle
            )
        else:
            center_position = np.mean(points_cartesian[ind_ii, :], axis=0)
        center_position = center_position.reshape(2)

        # if debug_viz:
        # plt.plot(points_cartesian[0,ind_ii], points_cartesian[1, ind_ii], '.' )

        start_time_obstacleLearning = time.time()

        if regression_obstacle:
            obs_list.append(
                RegressionObstacle(
                    center_position=center_position,
                    surface_points=points_cartesian[ind_ii, :],
                )
            )
        else:
            obs_list.append(
                GaussianEllipseObstacle(
                    center_position=center_position,
                    surface_points=points_cartesian[:, ind_ii],
                )
            )
            time_obstacleLearning += time.time() - start_time_obstacleLearning

    if print_timing:
        print("Time for clustering:" + str(time_clustering))
        print("Time for obstacle learning" + str(time_obstacleLearning))

    return obs_list


class ObstacleFromLaser:
    # Kept for
    def __init__(self, *args, **kwargs):
        warnings.warn("Depreciated use function <<get_obstacle_from_scan>> instead.")

    def get_obstacle_from_scan(self, *args, **kwargs):
        self.obs_list = get_obstacle_from_scan(*args, **kwargs)


class ObstacleFromData(Obstacle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def __init__(self, surface_points,radial_coordinates=True, *args, **kwargs):
    # center_position = self.transform_polar2cartesian(mean_mag, mean_angle)
    # self.construct_obstacle(*args, **kwargs)

    # def construct_obstacle_from_data(self, surface_points,  points_per_component=15,  *args, **kwargs):
    # warnings.warn("Implement obstacle constructor.")

    def transform_cartesian2polar(self, *args, **argv):
        return transform_cartesian2polar(*args, **argv)

    def transform_polar2cartesian(self, *args, **argv):
        return transform_polar2cartesian(*args, **argv)


class RegressionObstacle(ObstacleFromData):
    def __init__(
        self,
        center_position,
        surface_points=None,
        polar_surface_representation=True,
        minimal_radius=0.01,
        normalize=False,
        learn_surface=False,
        is_boundary=False,
        *args,
        **kwargs
    ):

        # super().__init__(*args, **kwargs)

        self.dim = center_position.shape[0]
        self.set_center_pos(center_position)

        self.polar_surface_representation = polar_surface_representation

        self.minimal_radius = minimal_radius

        origin_direction = (-1) * self.center_position
        self.origin_angle = np.arctan2(origin_direction[1], origin_direction[0])
        # print('origin angle {} deg'.format(self.origin_angle*180/pi) )

        self.surface_points = surface_points
        # self.learn_surface(surface_points=surface_points)

        if learn_surface:
            self.learn_surface(epsilon=0.02, C=1, kernel_curvature=100)

        # TODO remove/rename these
        self.orientation = 0
        self.rotMatrix = np.eye(self.dim)
        # self.w = 0
        # self.th_r = 0
        self.angular_velocity = 0  #
        self.linear_velocity = np.array([0, 0])
        # self.linear_velocity = 0
        # self.sigma = 1

        self.is_boundary = is_boundary

        # Initial magnitudes & points
        self.surface_angles = None
        self.surface_magnitudes = None

    def set_surface_points(
        self,
        surface_points=None,
        angles=None,
        magnitudes=None,
        in_global_frame=True,
    ):
        """Set surface angles&magnitude as either points OR angles&magnitudes."""
        if not surface_points is None:
            if in_global_frame:
                surface_points = self.transform_global2relative(surface_points)
            self.surface_magnitudes = np.linalg.norm(surface_points, axis=0)
            self.surface_angles = np.arctan2(surface_points[1, :], surface_points[0, :])

        elif not surface_points is None:
            if in_global_frame:
                raise ValueError("Angles can not be inputet in global frame")
            self.surface_angles = angles
            self.surface_magnitudes = magnitudes

        else:
            raise ValueError("Not properly defined error.")

    def reduce_angle_resolution(self, angular_resolution=1000):
        """Down-sample points (check if not doubling at 2*pi)"""
        int_angles = np.round(
            self.surface_angles / (2 * pi) * angular_resolution
            + angular_resolution / 2.0,
            0,
        )
        # rounded_angles = rounded_angles*(2*pi)/angular_resolution

        angles = np.linspace(-pi, pi, angular_resolution)
        magnitudes = np.zeros(angular_resolution)

        index_nonzero = np.zeros(angles.shape, dtype=bool)
        # TODO: check that values exist everywhere (otherwise delete)
        for aa in range(angular_resoluion):
            ind_in_range = aa == ind_angles[aa]
            magnitudes[aa] = np.min(self.surface_magnitudes[ind_in_range])

            if not np.sum(ind_in_range):
                index_nonzero[aa] = 0

        # Remove points of list where
        self.surface_angles = angles[index_nonzero]
        self.surface_magnitudes = surface_magnitudes[index_nonzero]

    def get_point_on_surface(self, position, in_global_frame=True):
        """Get surface point value based on position."""
        if in_global_frame:
            position = self.transform_global2relative(position)

        angle = np.arctan2(surface_points[1], position[0])
        mag = self.predict(angle)

        position = mag * np.array([np.cos(angle), np.sin(angle)])

        if in_global_frame:
            position = self.transform_relative2global(position)

    def predict_singe(
        self,
    ):
        # TODO
        raise NotImplementedError("TODO")

    def predict(self, angle, convert_to_relative=True):
        """Get prediction value based on angle OR position"""

        shape_angle = angle.shape
        # TODO: include shape
        angle = np.reshape(angle, (-1, 1))

        if convert_to_relative:
            angle_relative = self.convert_to_relative_angle(angle)

        dist_origin2surf = self.surface_regression.predict(angle_relative)

        return dist_origin2surf.reshape(shape_angle)

    def set_center_pos(self, center_position, reset_reference=True):
        """Set center position and reset reference point."""
        self.center_position = center_position
        self.reference_point = np.zeros(self.dim)

    def convert_to_relative_angle(self, angle):
        """???"""
        # TODO -- use relative angle
        # OR use circular regression

        # return (angle + 3*pi - self.origin_angle) % (2*pi) - pi
        return angle

    def get_normalization_parameters(self, input_data, regr_data):
        self.mean_input = np.mean(input_data, axis=1)
        # input_data =
        self.variance_input = np.mean(input_data, axis=1)

        return input_data, regr_data

    def normalize_data(self, data):
        n_points = self.data.shape[1]
        data = data - np.tile(self.mean_input, (n_points, 1)).T
        return data / np.tile(self.variance_input, (n_points, 1)).T

    def regularize_data(self, data):
        n_points = self.data.shape[1]
        data = data * np.tile(self.variance_input, (n_points, 1)).T
        return data + np.tile(self.mean_input, (n_points, 1)).T

    def extend_with_boundary(self, boundary_values, angle_margin=0.03):
        mag_surf, ang_surf = self.transform_cartesian2polar(
            self.surface_points, center_position=self.center_position
        )

        mag_boundary, ang_boundary = self.transform_cartesian2polar(
            boundary_values, center_position=self.center_position
        )

        n_surf = mag_surf.shape[0]
        n_boundary = mag_boundary.shape[0]

        ang_dist_abs = np.abs(
            np.tile(ang_surf, (n_boundary, 1)).T - np.tile(ang_boundary, (n_surf, 1))
        )

        ind_close = np.logical_or(
            np.sum(ang_dist_abs < angle_margin, axis=0),
            np.sum(
                np.logical_and(
                    (2 * pi - angle_margin) < ang_dist_abs,
                    ang_dist_abs < (2 * pi + angle_margin),
                ),
                axis=0,
            ),
        )

        self.surface_points = np.vstack(
            (self.surface_points, boundary_values[~ind_close, :])
        )

        if False:
            plt.figure()
            plt.plot(self.surface_points[:, 0], self.surface_points[:, 1], ".")
            import pdb

            pdb.set_trace()  ## DEBUG ##

    def learn_surface(
        self,
        regression_type="svr",
        epsilon=1.0,
        C=5,
        kernel="rbf",
        gamma=0.2,
        surface_points=None,
    ):

        if isinstance(surface_points, (list, np.ndarray)):
            surface_points = np.squeeze(surface_points)
            magnitude, angle = self.transform_cartesian2polar(
                surface_points, center_position=self.center_position
            )

        if regression_type == "svr":
            self.kernel_curvature = gamma
            self.surface_regression = sklearn.svm.SVR(
                kernel="rbf", C=C, gamma=gamma, epsilon=epsilon
            )

            self.surface_regression.set_params(gamma="scale")
            self.surface_regression.fit(
                self.surface_angles.reshape(-1, 1),
                self.surface_magnitudes.reshape(-1),
            )

            # TODO get regression function
            # def surface_regression_predict(value):
            # return self.surface_regression.predict(np.reshape(value,(-1,1)))
            # self.grad_surface = grad(surface_regression_predict)
            print("n_points", self.surface_angles.shape[0])
            print("center_position", self.center_position)
            print(
                "n support vectors_",
                self.surface_regression.support_vectors_.shape[0],
            )

        else:
            raise TypeError("Unkown regression type.")

    def get_normal_direction(
        self,
        position,
        relative_derivative_increment=1e-3,
        in_global_frame=False,
    ):
        position_abs = np.copy(position)

        if in_global_frame:
            position = self.transform_global2relative(position)

        # import pdb; pdb.set_trace() ## DEBUG ##
        magnitude, angle = transform_cartesian2polar(position)
        # angle_relative = self.convert_to_relative_angle(angle)

        derivative_increment = self.kernel_curvature * relative_derivative_increment

        # only 2D
        if self.polar_surface_representation:
            # Numerical derivative
            regr_val1 = self.predict(angle - derivative_increment / 2.0)
            regr_val2 = self.predict(angle + derivative_increment / 2.0)

            mean_radius = (regr_val2 + regr_val1) / 2.0
            # gradient = (regr_val2 - regr_val1) / (derivative_increment*mean_radius)
            # gradient = (regr_val2 - regr_val1) / (derivative_increment)

            # normal_direction = np.array([1, gradient[0]])
            # normal_direction = normal_direction/LA.norm(normal_direction)
            # normal_direction = np.array([1, 0]) # Debugging purpose

            derivative_angle = (regr_val2 - regr_val1) / (derivative_increment)

            # x = cos(phi), y = sin(phi)
            gradient = np.array([1, (-1) * 1 / mean_radius * derivative_angle])
            normal_direction = gradient

            # TODO check if derivative with respect to angle is correct
            if (not isinstance(angle, (float, int))) and angle.shape[0] > 1:
                raise TypeError("Not implemented for arrays of angles.")

            # Rotation
            rotationMatrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ]
            )

            # rotationMatrix = np.eye(self.dim)
            normal_direction = rotationMatrix.dot(normal_direction)

        else:
            warnings.warn("Non-polar not implemented")

        if debug_viz:
            pos_abs = self.transform_relative2global(position)
            plt.quiver(
                pos_abs[0],
                pos_abs[1],
                normal_direction[0],
                normal_direction[1],
                color="k",
            )

        return normal_direction

    def get_local_radius(
        self, angle, convert_to_relative=True, check_minimal_radius=True
    ):
        # angle = self.convert_to_relative_angle(angle)
        radius = self.predict(angle, convert_to_relative=convert_to_relative)

        if check_minimal_radius:
            radius = np.max(
                np.vstack((radius, np.ones(radius.shape) * self.minimal_radius)),
                axis=0,
            )
        return radius

    def get_gamma(
        self,
        position,
        in_global_frame=False,
        gamma_scaling=1.0,
        gamma_type="proportional",
    ):
        if in_global_frame:
            position = self.transform_global2relative(position)

        magnitude, angle = self.transform_cartesian2polar(position)

        dist_origin2surf = self.predict(angle)

        if self.polar_surface_representation:
            dist_origin2position = LA.norm(position)
        else:
            dist_origin2position = position[1]

        if gamma_type == "proportional":
            if dist_origin2surf == 0:
                return 0
            Gamma = dist_origin2position / dist_origin2surf
            if self.is_boundary:
                if Gamma == 0:
                    # warnings.warn('print that')
                    # print('position', position)
                    # raise Exception()
                    return sys.float_info.max
                Gamma = 1 / Gamma
            return Gamma
        elif gamma_type == "linear":
            if self.is_boundary:
                raise TypeError("Proportional gamma not defined for boundaries.")
            return 1 + (dist_origin2position - dist_origin2surf) / gamma_scaling
        else:
            raise TypeError("Gamma type {} is not implemented.".format(gamma_type))

    def draw_obstacle(self, numPoints=100):
        angles = np.linspace(-pi, pi, numPoints)
        magnitude = self.get_local_radius(angles)

        self.x_obs = self.transform_polar2cartesian(
            magnitude=magnitude,
            angle=angles,
            center_position=self.center_position,
        )
        self.x_obs = self.x_obs.T
        self.x_obs_sf = self.x_obs

        return self.x_obs.T


class GaussianEllipseObstacle(ObstacleFromData):
    """Fit GMM on radial values"""

    def __init__(
        self,
        center_position,
        surface_points,
        points_per_component=15,
        radial_coordinates=False,
        fit_init=True,
    ):

        self.dim = center_position.shape[0]

        self.center_position = center_position
        self.reference_point = np.zeros(self.dim)

        self.n_gaussians = int(np.floor(surface_points / points_per_component))

        gmm_model = sklearn.mixture.GaussianMixture(
            n_components=self.n_gaussians, covariance_type="full"
        )

        gmm_model.fit(surface_points)

        self.obs_list = []

        for ii in range(self.n_gaussians):
            self.obs_list[ii].set

        self.n_obstacles = len(self.obs_list)

    def get_normal_direction(self, position):
        normal_directions = np.zeros((self.dim, self.n_obstacles))
        for ii in range(self.n_obstacles):
            normal_directions[:, ii] = self.obs_list[ii].get_normal_direction(position)

        weights = self.get_weights(position)

        reference_dir = self.get_reference_direction(position)

        normal_vector = get_directional_weighted_sum(
            reference_direction=reference_dir,
            direcptions=normal_directions,
            weights=weights,
        )

        # plt.quiver(position[])

        return normal_vector

    def get_weights(self, position):
        gammas = np.zeros(self.n_obstacles)

        for ii in range(self.n_obstacles):
            gammas[ii] = self.obs_list[ii].get_gamma(position)

        return compute_weights(gammas, distMeas_lowerLimit=1)

    def get_gamma(self):
        gammas = np.zeros(self.n_obstacles)

        for ii in range(self.n_obstacles):
            gammas[ii] = self.obs_list[ii].get_gamma(position)

        weights = compute_weights(gammas, distMeas_lowerLimit=1)

        return np.sum(gammas * weights)


class PolygonWithLearnedSurface(Polygon):
    """Currently only for boundary polygons with no points outside the free space."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_special_surfaces_angles = 1

        self.special_surfaces = []
        self.special_surfaces_angles = np.array((self.n_special_surfaces_angles, 0))
        self.x_obs_special = []

    def get_normal_direction(
        self,
        position,
        in_global_frame=False,
        normalize=True,
        normal_calulation_type="distance",
    ):
        if in_global_frame:
            position = self.transform_global2relative(position)

        normal_vector_poly = super().get_normal_direction(
            position,
            in_global_frame=False,
            normalize=True,
            normal_calulation_type="distance",
        )

        if len(self.special_surfaces):  # >0
            pseudo_normal_surfaces = self.get_pseudoNormals_specialSurfaces(position)

            directions = np.vstack((normal_vector_poly, pseudo_normal_surfaces.T)).T

            gamma_poly = super().get_gamma(position)
            gamma_surfaces = self.get_gammas_specialSurface(position)

            weights = self.get_distance_weight(
                np.hstack((gamma_poly, gamma_surfaces)), distance_min=1
            )
            normal_vector = get_directional_weighted_sum(
                reference_direction=position,
                directions=directions,
                weights=weights,
            )

        if in_global_frame:
            normal_vector = self.transform_relative2global_dir(normal_vector)

        if False:  # DEBUGGING
            pos_abs = self.transform_relative2global(position)
            # pos_abs_temp = self.transform_relative2global(temp_position)
            norm_abs = self.transform_relative2global_dir(normal_vector)

            plt.quiver(
                pos_abs[0],
                pos_abs[1],
                norm_abs[0],
                norm_abs[1],
                color="g",
                label="normal",
            )

            norm_poly_abs = self.transform_relative2global_dir(normal_vector_poly)
            # plt.quiver(pos_abs[0], pos_abs[1], norm_poly_abs[0], norm_poly_abs[1], color='b', label='polygon')

            # plt.quiver(pos_abs_temp[0], pos_abs_temp[1], norm_abs[0], norm_abs[1], color='m')
            ref_abs = position
            if LA.norm(ref_abs):
                ref_abs = position / LA.norm(ref_abs)
            ref_abs = self.transform_relative2global_dir(ref_abs)
            # plt.quiver(pos_abs[0], pos_abs[1], ref_abs[0], ref_abs[1], color='k', label='reference')

            for ii in range(pseudo_normal_surfaces.shape[1]):
                pseudo_norm_trans = self.transform_relative2global_dir(
                    pseudo_normal_surfaces[:, ii]
                )
                # plt.quiver(pos_abs[0], pos_abs[1], pseudo_norm_trans[0],  pseudo_norm_trans[1], color='r', label='surfaces')

            # plt.legend()
            plt.ion()
            plt.show()
        return normal_vector

    def get_gamma(self, position, in_global_frame=False):
        if in_global_frame:
            position = self.transform_global2relative(position)
        angle2surfaces = self.get_angles_to_specialSurfaces(position)

        Gammas_special = self.get_gammas_specialSurface(position, in_global_frame=False)
        Gammas_special = np.array(Gammas_special)

        Gamma = super().get_gamma(position)

        if Gammas_special.shape[0]:  # ~=0
            Gamma = np.min(np.hstack((Gammas_special[Gammas_special >= 0], Gamma)))
        return Gamma

    def get_angles_to_specialSurfaces(self, position, in_global_frame=False):
        if in_global_frame:
            position = self.transform_global2relative(position)

        angle_to_surface = np.zeros(len(self.special_surfaces))
        for ii in range(len(self.special_surfaces)):
            mag, angle2pos = transform_cartesian2polar(position)
            delta_phi_low = angle_difference_directional(
                self.special_surfaces_angles[0, ii], angle2pos
            )
            delta_phi_high = angle_difference_directional(
                angle2pos, self.special_surfaces_angles[1, ii]
            )

            if delta_phi_low <= 0 and delta_phi_high <= 0:
                angle_to_surface[ii] = 0
            else:
                angle_to_surface[ii] = np.min(np.abs([delta_phi_low, delta_phi_high]))
        return angle_to_surface

    def get_gammas_specialSurface(
        self, position, in_global_frame=False, angle_limit=pi / 8
    ):
        if in_global_frame:
            position = self.transform_global2relative(position)

        # Return minimal gamma of all special surfaces
        Gammas = []
        angle2surfaces = self.get_angles_to_specialSurfaces(position)

        for ii in range(len(self.special_surfaces)):
            if angle2surfaces[ii] == 0:
                Gammas.append(self.special_surfaces[ii].get_gamma(position))
            elif angle2surfaces[ii] < angle_limit:
                Gamma = self.special_surfaces[ii].get_gamma(position)
                Gamma = np.squeeze(Gamma)  # TODO get better output of special_surface
                Gamma = Gamma * angle_limit / (angle_limit - angle2surfaces[ii])
                Gammas.append(Gamma)
            else:
                # Gamma == -1 means that the special surface is not present in this direction
                Gammas.append(-1)

            # if self.is_boundary:
            # Gammas[-1] = 1/Gammas[-1]
        return Gammas

    def get_pseudoNormals_specialSurfaces(self, position, angle_limit=pi / 8):
        # get direction to all surfaces

        angle2surfaces = self.get_angles_to_specialSurfaces(position)

        weights = (angle_limit - angle2surfaces) / angle_limit
        weights = np.max(np.vstack((weights, np.zeros(weights.shape))), axis=0)
        reference_direction = -self.get_reference_direction(
            position, in_global_frame=False
        )

        pseudo_normals = np.zeros((self.dim, len(self.special_surfaces)))
        for ii in range(len(self.special_surfaces)):
            if weights[ii] > 0:
                normal_direction = self.special_surfaces[ii].get_normal_direction(
                    position
                )
                pseudo_normals[:, ii] = (
                    weights[ii] * normal_direction
                    + (1 - weights[ii]) * reference_direction
                )
            else:
                pseudo_normals[:, ii] = reference_direction

        return pseudo_normals

    def find_boundary_surfaces(self, obs_list):
        # Finds boundary surfaces and removes them from obs_list
        # and attaches them to self.special_surfaces
        # print('start')
        for ii in range(len(obs_list)):
            # print('start first lopop')
            intersection_found = False
            for pp in range(obs_list[ii].surface_points.shape[0]):
                # print('start second lopop')
                # print('and abit moroo')
                # print(pp, 'gamma', self.get_gamma(obs_list[ii].surface_points[pp, :], in_global_frame=True))
                if (
                    super().get_gamma(
                        obs_list[ii].surface_points[pp, :],
                        in_global_frame=True,
                    )
                    < 1.5
                ):
                    self.special_surfaces.append(obs_list[ii])
                    self.special_surfaces[-1].center_position = self.center_position
                    self.special_surfaces[-1].is_boundary = True
                    # self.intersection_with_special_surfaces(obs_list[ii])
                    del obs_list[ii]

                    intersection_found = True
                    print("found intersection with boundary")

                    break

            if intersection_found:
                break

    def intersection_with_special_surfaces(self):
        # Assumption: the new boundary does not cover/cross the self.center_position
        # This is for fix boundaries        if self.dim>2:
        if self.dim > 2:
            raise TypeError("Too high dimension")

        self.special_surfaces_angles = np.zeros((self.dim, len(self.special_surfaces)))

        it_surf = 0
        for surface_reference in self.special_surfaces:
            mean_surf_point = np.mean(surface_reference.surface_points.T, axis=1)

            mag, surf_mean_angle = transform_cartesian2polar(
                mean_surf_point, self.center_position
            )

            d_angle_list = [-0.1, 0.1]

            angles_intersection = np.zeros(2)
            points_intersection = np.zeros((self.dim, 2))
            x_obs_special = []

            for ii in range(len(d_angle_list)):
                d_ang = d_angle_list[ii]

                it_count = 0
                Gamma_old = 2

                # def draw_obstacle(self, numPoints=100):
                #     angles = np.linspace(-pi, pi, numPoints)
                #     magnitude = self.get_local_radius(angles)

                #     self.x_obs = self.transform_polar2cartesian(magnitude=magnitude, angle=angles, center_position=self.center_position)
                #     self.x_obs = self.x_obs.T
                #     self.x_obs_sf = self.x_obs

                #     return self.x_obs.T

                while abs(it_count * d_ang) <= pi:
                    angle = surf_mean_angle + it_count * d_ang
                    radius_surface = surface_reference.get_local_radius(angle)

                    position = transform_polar2cartesian(
                        angle=angle,
                        magnitude=radius_surface,
                        center_position=surface_reference.center_position,
                    )

                    Gamma_position = super().get_gamma(position, in_global_frame=True)

                    x_obs_special.append(position)

                    if (Gamma_position - 1) * (Gamma_old - 1) < 0:
                        # Going from in to outside of the boundary

                        angles_intersection[ii] = angle_modulo(angle)
                        points_intersection[:, ii] = position
                        break

                    Gamma_old = Gamma_position  # reverse list
                    it_count += 1

                if len(x_obs_special) == 0:
                    break

                if ii == 0:
                    x_obs_special.reverse()  # reverse list on first iteration
                    del x_obs_special[-1]  # delete duplicated middle element

            if False:
                x_obs_special = np.array(x_obs_special)
                print("shape", x_obs_special.shape)
                plt.figure()
                plt.plot(x_obs_special[:, 0], x_obs_special[:, 1], "-.")

            if angles_intersection[0] == angles_intersection[1]:
                print("No surface intersection found")
                self.special_surfaces_angles
                continue

            self.special_surfaces_angles[:, it_surf] = angles_intersection

            x_obs_special = np.array(x_obs_special)
            # x_obs_special = self.transform_relative2global(x_obs_special.T)
            self.x_obs_special.append(x_obs_special)

            it_surf += 1

    def draw_obstacle(self, *args, **kwargs):
        return super().draw_obstacle(*args, **kwargs), self.x_obs_special
