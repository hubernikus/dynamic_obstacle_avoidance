#!/USSR/bin/python3
'''
@date 2019-10-15
@author Lukas Huber 
@email lukas.huber@epfl.ch
'''

import time
import numpy as np
from math import sin, cos, pi, ceil
import warnings, sys

import numpy.linalg as LA

from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import *

from dynamic_obstacle_avoidance.obstacle_avoidance.state import *
from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import angle_modulo
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_dynamic_center_3d import *

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *

import matplotlib.pyplot as plt
# import quaternion

visualize_debug = False

class Ellipse(Obstacle):
    # self.ellipse_type = dynamic_obstacle_avoidance.obstacle_avoidance.obstacle.Ellipse
    def __init__(self, axes_length=None, a=None, p=[1,1], *args, **kwargs):
        if sys.version_info>(3,0):
            super().__init__(*args, **kwargs)
        else:
            super(Ellipse, self).__init__(*args, **kwargs) # works for python < 3.0?!
        
        self.axes_length = axes_length
        self.axes_with_safety =  self.axes_length*np.array(self.sf)+np.array(self.delta_margin)
        
        self.p = np.array(p) # TODO: rename & make p one value (NOT array)
        self.is_convex = True

    @property
    def a(self): # TODO: remove
        return self._axes_length

    @a.setter # TODO:remove
    def a(self, value):
        self._axes_length = value

    @property
    def axes(self): # TODO: remove
        return self._axes_length

    @axes.setter # TODO:remove
    def axes(self, value):
        self._axes_length = value

    @property
    def axes_length(self):
        return self._axes_length

    @axes_length.setter
    def axes_length(self, value):
        self._axes_length = value
    
    def get_minimal_distance(self):
        return np.min(self.a)

    def get_maximal_distance(self):
        # Eucledian
        return np.sqrt(np.sum(self.a*2))

    def get_normal_direction(self, position, in_global_frame=False, normalize=True):
        if in_global_frame:
            position = self.transform_global2relative(position)

        if not self.reference_point_is_inside:
            ind_intersect = np.zeros(self.normalDistance2center.shape, dtype=bool)

            distances2plane = self.get_distance_to_hullEdge(position)

            ind_outside = (distances2plane > 0)

            if np.sum(ind_outside)>0:
                for ii in np.arange(ind_outside.shape[0])[ind_outside]:

                    reference_line = {"point_start":[0,0],
                                      "point_end":position}
                    # TODO - don't use reference point, but little 'offset' to avoid singularity
                    tangent_line = {"point_start":self.hull_edge,
                                    "point_end":self.tangent_points[:, ii]}

                    ind_intersect[ii], dist = self.are_lines_intersecting(reference_line, tangent_line)

                    if ind_intersect[ii]:
                        break

                if np.sum(ind_intersect): # nonzero
                    angle2referencePlane = self.get_angle2referencePatch(position)
                    weights = self.get_angle_weight(angle2referencePlane)

                    normal_vector = get_directional_weighted_sum(reference_direction=position, directions=self.normal_vector, weights=weights, normalize=False, normalize_reference=True)
                    
                    # return normal_vector

        if self.reference_point_is_inside or np.sum(ind_intersect)==0:
            # normal_vector = (2*self.p/self.margin_axes*(position/self.margin_axes)**(2*self.p-1))
            normal_vector = (2*self.p/self.axes_length*(position/self.axes_length)**(2*self.p-1))

        if normalize:
            # TODO: can it be removed?
            normal_vector = normal_vector/LA.norm(normal_vector)

        if in_global_frame:
            normal_vector = self.transform_relative2global_dir(normal_vector) 
        return normal_vector


    def get_gamma(self, position, in_global_frame=False, gamma_type='proportional'):
        # TODO: use cython to speed up
        
        # Get distance value from surface
        if in_global_frame:
            position = self.transform_global2relative(position)

        multiple_positions = (len(position.shape)>1)
        
        if multiple_positions:
            n_points = position.shape[1]
        else:
            n_points = 1
            position = position.reshape((self.dim, n_points))

        Gamma = np.zeros(n_points)
        intersecting_ind = np.ones(n_points, dtype=bool)
        
        # Original Gamma
        if gamma_type=='proportional':
            
            if not self.reference_point_is_inside:
            # TODO: expand for multiple position input

                for pp in range(n_points): # speed up
                    for ii in np.arange(self.tangent_points.shape[1]):
                        reference_line = {"point_start":[0,0], "point_end":position[:, pp]}

                        # TODO: - don't use reference point, but little 'offset' to avoid singularity
                        tangent_line = {"point_start":self.hull_edge, "point_end":self.tangent_points[:, ii]}
                        ind_intersect, dist_intersect = self.are_lines_intersecting(reference_line, tangent_line)
                        if ind_intersect:
                            # return LA.norm(position)/dist_intersect
                            Gamma[pp] = LA.norm(position[:, pp])/dist_intersect
                            intersecting_ind[pp] = False

            if np.sum(intersecting_ind):
                n_points = np.sum(intersecting_ind)
                Gamma[intersecting_ind] = np.sum( (position[:, intersecting_ind] / np.tile(self.axes_with_safety, (n_points,1)).T ) ** (2*np.tile(self.p, (n_points,1)).T), axis=0)
        else:
            raise NotImplementedError()

        if self.is_boundary:
            Gamma = self.get_boundaryGamma(Gamma,Gamma_ref=self.Gamma_ref)

        if not multiple_positions:
            return Gamma[0] # 1x1-array to value
        
        return Gamma

    
    def draw_ellipsoid(self, *args, **kwargs):
        # TODO remove
        raise NotImplementedError("<<draw_ellipsoid>> has been renamed <<draw_obstacle>>")

    def extend_hull_around_reference(self, edge_reference_dist=0.3):
        # TODO add margin

        if self.get_gamma(self.reference_point)<1:
            self.reference_point_is_inside = True
            return
        else:
            self.reference_point_is_inside = False

        vec_cent2ref = np.array(self.get_reference_point(in_global_frame=False))
        dist_cent2ref = LA.norm(vec_cent2ref)

        self.hull_edge = vec_cent2ref*(1 + edge_reference_dist*np.min(self.axes_length)/dist_cent2ref)
        # self.hull_edge =  np.array(self.get_reference_point(in_global_frame=False))
        self.tangent_vector, self.tangent_points = get_tangents2ellipse(self.hull_edge, self.axes)
        self.normal_vector = np.zeros((self.dim, 2))
        self.normalDistance2center = np.zeros(2)

        # # Intersection of (x_1/a_1)^2 +( x_2/a_2)^2 = 1 & x_2=m*x_1+c
        # # Solve for determinant D=0 (tangent with only one intersection point)
        # A_ =  self.hull_edge[0]**2 - self.axes[0]**2
        # B_ = -2*self.hull_edge[0]*self.hull_edge[1]
        # C_ = self.hull_edge[1]**2 - self.axes[1]**2
        # D_ = B_**2 - 4*A_*C_

        # m = [0, 0]

        # m[1] = (-B_ - np.sqrt(D_)) / (2*A_)
        # m[0] = (-B_ + np.sqrt(D_)) / (2*A_)

        # self.tangent_points = np.zeros((self.dim, 2))
        # self.tangent_vector = np.zeros((self.dim, 2))

        # for ii in range(2):
        #     c = self.hull_edge[1] - m[ii]*self.hull_edge[0]

        #     A = (self.axes[0]*m[ii])**2 + self.axes[1]**2
        #     B = 2*self.axes[0]**2*m[ii]*c
        #     # D != 0 so C not interesting

        #     self.tangent_points[0, ii] = -B/(2*A)
        #     self.tangent_points[1, ii] = m[ii]*self.tangent_points[0, ii] + c

        #     self.tangent_vector[:,ii] = self.tangent_points[:, ii]-self.hull_edge
        #     self.tangent_vector[:,ii] /= LA.norm(self.tangent_vector[:,ii])

        for ii in range(2):
            self.normal_vector[:, ii] = np.array([self.tangent_vector[1,ii],
                                                  -self.tangent_vector[0,ii]])
            # Check direction
            self.normalDistance2center[ii] = self.normal_vector[:, ii].T.dot(self.hull_edge)

            if (self.normalDistance2center[ii] < 0):
                self.normal_vector[:, ii] = self.normal_vector[:, ii]*(-1)
                self.normalDistance2center[ii] *= -1

        if False:
            # plt.plot()
            for ii in range(2):
                norm_abs = transform_relative2global_dir(self.normal_vector[:,ii])
                plt.quiver(0,0, norm_abs, norm_abs, color='y', label='Normal')

    
    
    def draw_obstacle(self, numPoints=20, a_temp=None, draw_sfObs=False):
        if not self.reference_point_is_inside:
            angle_tangents = np.zeros(2)
            for ii in range(angle_tangents.shape[0]):
                angle_tangents[ii] = np.arctan2(self.tangent_points[1, ii], self.tangent_points[0, ii])
            theta = np.zeros(numPoints)

            if angle_tangents[0]>angle_tangents[1]:
                theta[1:-1] = np.linspace(angle_tangents[0], angle_tangents[1], numPoints-2)
            else:
                theta[1:-1] = np.linspace(angle_tangents[1], angle_tangents[0]+2*pi, numPoints-2)
                theta = angle_modulo(theta)
        else:
            if self.d == 2:
                theta = np.linspace(-pi,pi, num=numPoints)
                resolution = numPoints # Resolution of drawing #points
            else:
                numPoints = [numPoints, ceil(numPoints/2)]
                theta, phi = np.meshgrid(np.linspace(-pi,pi, num=numPoints[0]),np.linspace(-pi/2,pi/2,num=numPoints[1]) ) #
                numPoints = numPoints[0]*numPoints[1]
                resolution = numPoints # Resolution of drawing #points
                theta = theta.T
                phi = phi.T

        if a_temp is None:
            a = self.axes_length
        else:
            a = a_temp
            warnings.warn("depreciated will be removed")

        p = self.p[:]

        self._boundary_points = np.zeros((self.d, numPoints))
        
        if self.d == 2:
            self._boundary_points[0,:] = a[0]*np.cos(theta)
            self._boundary_points[1,:] = np.copysign(a[1], theta)*(1 - np.cos(theta)**(2*p[0]))**(1./(2.*p[1]))
        else:
            self._boundary_points[0,:] = (a[0]*np.cos(phi)*np.cos(theta)).reshape((1,-1))
            self._boundary_points[1,:] = (a[1]*np.copysign(1, theta)*np.cos(phi)*(1 - np.cos(theta)**(2*p[0]))**(1./(2.*p[1]))).reshape((1,-1))
            self._boundary_points[2,:] = (a[2]*np.copysign(1,phi)*(1 - (np.copysign(1,theta)*np.cos(phi)*(1 - 0 ** (2*p[2]) - np.cos(theta)**(2*p[0]))**(1/(2**p[1])))**(2*p[1]) - (np.cos(phi)*np.cos(theta)) ** (2*p[0])) ** (1/(2*p[2])) ).reshape((1,-1))

        if not self.reference_point_is_inside:
            self._boundary_points[:,0] = self._boundary_points[:,-1] = self.hull_edge

        # TODO: - more appropriate margin definition
        self._boundary_points_margin = self.sf*self._boundary_points
        
            # x_obs_sf = R.dot(self.sf*self._boundary_points) + np.tile(np.array([self.center_position]).T,(1,numPoints))
        # else:
            # x_obs_sf = R.dot() + np.tile(self.center_position, (numPoints,1)).T
            
        # if sum(a_temp) == 0:
            # self.x_obs = x_obs.T.tolist()
            # self.x_obs_sf = x_obs_sf.T.tolist()
            # self.x_obs = x_obs.T
            # self.x_obs_sf = x_obs_sf.T
        # else:
        
        return self.transform_relative2global(self._boundary_points_margin)

    def get_radius_of_angle(self, angle, in_global_frame=False):
        if in_global_frame:
            position =  transform_polar2cartesian(magnitude=10, angle=angle-self.orientation)
        else:
            position = transform_polar2cartesian(magnitude=1, angle=angle)
        
        gamma = self.get_gamma(position, gamma_type='proportional')
        return LA.norm(position)/gamma
