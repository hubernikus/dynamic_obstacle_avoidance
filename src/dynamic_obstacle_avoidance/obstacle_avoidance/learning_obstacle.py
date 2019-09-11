# coding: utf-8
from math import sin, cos, pi, ceil
import warnings, sys

import numpy as np
import numpy.linalg as LA

from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *

from autograd import grad, grad # both needed??

import sklearn
from sklearn.cluster import DBSCAN

import time

# from sklearn import mixture
# TODO higher dimensions
# Store 'learning' within frames -- HOW TO?

debug_viz = True

class ObstacleFromLaser():
    def __init__(self, cutoff_distance=6):
        self.cutoff_distance = cutoff_distance
        self.obs_list = []

        
    def transform_polar2cartesian(self, magnitude, angle, center_point=[0,0]):
        magnitude = np.reshape(magnitude, (-1))
        angle = np.reshape(angle, (-1))
        
        # points = [r, phi]
        points = (magnitude * np.vstack((np.cos(angle), np.sin(angle)))
                  + np.tile(center_point, (magnitude.shape[0],1)).T )
        return points
    
        
    def get_obstacle_from_scan(self, sensor_data, center_dist_scaling=1.5,
                               clustering_minimal_distance=10, clustering_min_samples=0.5,
                               input_is_cratesian=True, regression_obstacle=True,
                               center_point_offset=1.1):
                
        magnitude = sensor_data["magnitude"]
        angle = sensor_data["angle"]
        points_cartesian = self.transform_polar2cartesian(magnitude, angle)

        ind_close = (LA.norm(points_cartesian, axis=0) < self.cutoff_distance)

        if debug_viz:
            plt.plot(points_cartesian[0, (~ind_close)], points_cartesian[1,(~ind_close)], 'r.')

        
        
        magnitude = magnitude[ind_close]
        angle = angle[ind_close]
        points_cartesian = points_cartesian[:,ind_close]

        start_time_clustering = time.time()                       
        db_cluster = DBSCAN(eps=1.0, min_samples=5, metric="euclidean").fit(points_cartesian.T)
        time_clustering = time.time() - start_time_clustering
        
        cluster_labels = db_cluster.labels_

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        # print('n clusters', n_clusters)

        time_obstacleLearning = 0

        for ii in range(n_clusters):
            ind_ii = (cluster_labels==ii)

            mean_angle = np.mean(angle[ind_ii]+pi)-pi
            # mean_mag = np.mean(magnitude[ind_ii])
            # mean_mag = mean_mag*center_point_offset
            ref_mag = np.max(magnitude[ind_ii])
            
            center_point = self.transform_polar2cartesian(magnitude=ref_mag, angle=mean_angle)
            center_point = center_point.reshape(2)

            if debug_viz:
                plt.plot(points_cartesian[0,ind_ii], points_cartesian[1, ind_ii], '.' )

            start_time_obstacleLearning = time.time()
            
            if regression_obstacle:
                self.obs_list.append(
                    RegressionObstacle(center_position=center_point,
                                       surface_points=points_cartesian[:, ind_ii]))

                print('n_points', np.sum(ind_ii))
                print('center_point', center_point)
                print("n support vectors_", self.obs_list[ii].surface_regression.support_vectors_.shape[0])
            else:
                self.obs_list.append(
                    GaussianEllipseObstacle(center_position=center_point,
                                            surface_points=points_cartesian[:, ind_ii]))
            time_obstacleLearning += (time.time()-start_time_obstacleLearning)

        return time_clustering, time_obstacleLearning


class ObstacleFromData(Obstacle):
    # def __init__(self, surface_points,radial_coordinates=True, *args, **kwargs):
        # center_position = self.transform_polar2cartesian(mean_mag, mean_angle)
        # self.construct_obstacle(*args, **kwargs)
        
    # def construct_obstacle_from_data(self, surface_points,  points_per_component=15,  *args, **kwargs):
        # warnings.warn("Implement obstacle constructor.")

    def transform_cartesian2polar(self, points, center_position):
        # if type(center_position)==type(None):
            # center_position = np.zeros(self.dim)
            
        points = np.reshape(points, (self.dim,-1))
        points = points - np.tile(center_position, (points.shape[1], 1)).T
        
        magnitude = LA.norm(points, axis=0)
        angle = np.arctan2(points[1,:], points[0,:])
        
        # output: [r, phi]
        return magnitude, angle

    
    def transform_polar2cartesian(self, magnitude, angle, center_position):
        # points = [r, phi]
        # only 2D
        # if type(center_position)==type(None):
            # center_position = np.zeros(self.dim)
            
        magnitude = np.reshape(magnitude, (1,-1))
        angle = np.reshape(angle, (1,-1))

        points_cartesian = (magnitude * np.vstack((np.cos(angle), np.sin(angle))) + np.tile(center_position, (magnitude.shape[0],1)).T )
                  
        return points_cartesian

    
class RegressionObstacle(ObstacleFromData):
    def __init__(self, center_position,
                 surface_points,
                 polar_surface_representation=True,
                 minimal_radius=0.01,
                 normalize=False):
        
        self.dim = center_position.shape[0]
        self.set_center_pos(center_position)
        
        self.polar_surface_representation = polar_surface_representation

        self.minimal_radius = minimal_radius

        origin_direction = (-1)*self.center_position
        self.origin_angle = np.arctan2(origin_direction[1], origin_direction[0])
        # print('origin angle {} deg'.format(self.origin_angle*180/pi) )

        
        self.learn_surface(surface_points=surface_points)

        # TODO remove/rename these 
        self.th_r = 0
        self.rotMatrix = np.eye(self.dim)
        self.w = 0
        self.xd = 0
        self.sigma = 1

        
    def set_center_pos(self, center_position, reset_reference=True):
        self.center_position = center_position
        self.reference_point = np.zeros(self.dim)

    def convert_to_relative_angle(self, angle):
        angle = np.reshape(angle, (-1, 1))
        return (angle + 3*pi - self.origin_angle) % (2*pi) - pi
        # return angle

    def get_normalization_parameters(self, input_data, regr_data):
        self.mean_input = np.mean(input_data, axis=1)
        # input_data = 
        self.variance_input = np.mean(input_data, axis=1)

        return input_data, regr_data

    def normalize_data(self, data):
        n_points = self.data.shape[1]
        data = data-np.tile(self.mean_input, (n_points, 1)).T
        return data / np.tile(self.variance_input, (n_points, 1)).T

    def regularize_data(self, data):
        n_points = self.data.shape[1]
        data = data * np.tile(self.variance_input, (n_points, 1)).T 
        return data+np.tile(self.mean_input, (n_points, 1)).T
    
    def learn_surface(self, surface_points, regression_type="svr", epsilon=1.0, C=5, kernel='rbf', kernel_curvature=0.2):

        magnitude, angle = self.transform_cartesian2polar(surface_points,
                                                          center_position=self.center_position)
        angle = self.convert_to_relative_angle(angle)
        
        if regression_type=="svr":
            self.kernel_curvature = kernel_curvature
            self.surface_regression = sklearn.svm.SVR(kernel="rbf", C=C, gamma=kernel_curvature, epsilon=epsilon)
            
            self.surface_regression.set_params(gamma="scale")
            self.surface_regression.fit(angle.reshape(-1,1), magnitude.reshape(-1))

            # if debug_viz:
                # plt.figure()
                # plt.plot(angle, magnitude, '.')
            # TODO get regression function
            # def surface_regression_predict(value):
                # return self.surface_regression.predict(np.reshape(value,(-1,1)))
            # self.grad_surface = grad(surface_regression_predict)

        if debug_viz and False:
            plt.figure()
            plt.plot(angle, magnitude, '.')
            ang_reg = np.linspace(-pi, pi, 100)
            ang_reg = np.reshape(ang_reg, (-1, 1))
            mag_reg = self.surface_regression.predict(ang_reg)
            plt.plot(ang_reg, mag_reg, 'r')
            plt.ylim(0, plt.gca().get_ylim()[1])

    
    def get_normal_direction(self, position, relative_derivative_increment=1e-3, in_global_frame=False):
        position_abs = np.copy(position)
        
        if in_global_frame:
            position = self.transform_global2relative(position)
        magnitude, angle = self.transform_cartesian2polar(position, in_global_frame)
        angle_relative = self.convert_to_relative_angle(angle)

        derivative_increment = self.kernel_curvature*relative_derivative_increment
        
        # only 2D
        if self.polar_surface_representation:
            # Numerical derivative
            regr_val1 = self.surface_regression.predict(angle_relative-derivative_increment/2.0)
            regr_val2 = self.surface_regression.predict(angle_relative+derivative_increment/2.0)

            mean_radius = (regr_val2 + regr_val1) / 2.0
            # gradient = (regr_val2 - regr_val1) / (derivative_increment*mean_radius)
            # gradient = (regr_val2 - regr_val1) / (derivative_increment)
            
            # normal_direction = np.array([1, gradient[0]])
            # normal_direction = normal_direction/LA.norm(normal_direction)
            # normal_direction = np.array([1, 0]) # Debugging purpose

            derivative_angle = (regr_val2 - regr_val1) / (derivative_increment)

            # x = cos(phi), y = sin(phi)
            gradient = np.array([1,(-1)*1/mean_radius*derivative_angle[0]])
            normal_direction =  gradient
            
            # TODO check if derivative with respect to angle is correct
            if angle.shape[0]>1:
                warnings.warn("Not implemented for several angles.")
                
            # Rotation
            rotationMatrix = np.array([[np.cos(angle[0]), -np.sin(angle[0])],
                                       [np.sin(angle[0]), np.cos(angle[0])]])

            # rotationMatrix = np.eye(self.dim)
            normal_direction = rotationMatrix.dot(normal_direction)
        else:
            warnings.warn("Non-polar not implemented")

        if debug_viz:
            pos_abs = self.transform_relative2global(position)
            plt.quiver(pos_abs[0], pos_abs[1], normal_direction[0], normal_direction[1], color='k')
            
        return normal_direction

        
    def get_local_radius(self, angle, convert_to_relative=True, check_minimal_radius=True):
        angle = self.convert_to_relative_angle(angle)
        radius = self.surface_regression.predict(angle)
        
        if check_minimal_radius:
            radius = np.max(np.vstack((radius, np.ones(radius.shape)*self.minimal_radius)), axis=0)
        return radius

    
    def get_gamma(self, position, in_global_frame=False, gamma_scaling=1.0):
        if in_global_frame:
            position = self.transform_global2relative(position)
        
        magnitude, angle = self.transform_cartesian2polar(position, center_position=self.center_position)
        angle = self.convert_to_relative_angle(angle)
            
        dist_origin2surf = self.surface_regression.predict(angle)

        if self.polar_surface_representation:
            dist_origin2position = LA.norm(position)
        else:
            dist_origin2position = position[1]

        return 1+(dist_origin2position-dist_origin2surf)/gamma_scaling

    
    def draw_obstacle(self, numPoints=100):
        angles = np.linspace(-pi, pi, numPoints)
        magnitude = self.get_local_radius(angles)

        self.x_obs = self.transform_polar2cartesian(magnitude=magnitude, angle=angles, center_position=self.center_position)
        self.x_obs = self.x_obs.T
        self.x_obs_sf = self.x_obs
        # import pdb; pdb.set_trace() ## DEBUG ##
        
        return self.x_obs.T



class GaussianEllipseObstacle(ObstacleFromData):
    def __init__(self, center_position, 
                 surface_points,
                 points_per_component=15, radial_coordinates=False):

        self.dim = center_position.shape[0]
        
        self.center_position = center_position
        self.reference_point = np.zeros(self.dim)

        
        self.n_gaussians = int(np.floor(surface_points/points_per_component))

        gmm_model = sklearn.mixture.GaussianMixture(n_components=self.n_gaussians,
                                                    covariance_type="full")
        
        gmm_model.fit(surface_points)
        
        self.obs_list = []
        
        for ii in range(self.n_gaussians):
            self.obs_list[ii].set

        self.n_obstacles = len(self.obs_list)

        
    def get_normal_direction(self, position):
        normal_directions = np.zeros((self.dim, self.n_obstacles))
        for ii in range(self.n_obstacles):
            normal_directions[:,ii] = self.obs_list[ii].get_normal_direction(position)
            
        weights = self.get_weights(position)

        reference_dir = self.get_reference_direction(position)
        
        normal_vector =  get_directional_weighted_sum(
            reference_direction=reference_dir, directions=normal_directions, weights=weights)

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
        
        return np.sum(gammas*weights)
