 #!/usr/bin/python3

'''
@date 2019-10-15
@author Lukas Huber 
@mail lukas.huber@epfl.ch
'''

import time
import numpy as np
from math import sin, cos, pi, ceil
import warnings, sys

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *


class Polygon(Obstacle):
    '''
    Star Shaped Polygons
    '''
    def __init__(self,  edge_points, ind_tiles=None,
                 *args, **kwargs):
        # This class defines obstacles to modulate the DS around it
        # At current stage the function focuses on Ellipsoids, but can be extended to more general obstacles
        self.edge_points = np.array(edge_points)

        if type(center_position)==type(None):
            center_position = np.sum(self.edge_points, axis=1)/self.edge_points.shape[1]
        else:
            center_position = np.array(center_position) # new name for future version

        self.edge_points = self.edge_points-np.tile(self.center_position, (self.edge_points.shape[1], 1)).T

        if self.dim==2:
            self.n_planes = self.edge_points.shape[1]
            
        if self.dim==3:
            self.ind_tiles = self.ind_tiles
            self.n_planes = self.ind_tiles.shape[0]
        
        super().__init__(*args, center_position=center_position, **kwargs)


    def calculate_normalVectorAndDistance(self, edge_points=None):
        if isinstance(edge_points, type(None)):
            edge_points = self.edge_points

        normal_vector = np.zeros(edge_points.shape)
        normalDistance2center = np.zeros(edge_points.shape[1])
        
        if self.dim==2:
            for ii in range(self.n_planes):
                normal_vector[:, ii] = (edge_points[:,(ii+1)%normal_vector.shape[1]]
                                         - edge_points[:,ii])
                normal_vector[:, ii] = np.array([normal_vector[1, ii],
                                                      -normal_vector[0, ii],])
        elif self.dim==3:
            for ii in range(self.n_planes):
                tangent_0 = self.edge_points[:, self.ind_tiles[ii,1]] \
                            - self.edge_points[:, self.ind_tiles[ii,0]]
                
                tangent_1 = self.edge_points[:, self.ind_tiles[ii,2]] \
                            - self.edge_points[:, self.ind_tiles[ii,1]]

                normal_vector[:, ii] = np.cross(tangent_0, tangent_1)

                norm_mag = LA.norm(normal_vector[:, ii])
                if norm_mag: # nonzero
                    normal_vector[:, ii]= normal_vector[:, ii]/norm_mag
                
            else:
                raise ValueError("Implement for d>3.")

        
        for ii in range(self.n_planes):
            normalDistance2center[ii] = normal_vector[:, ii].T @ edge_points[:, ii]

            if normalDistance2center[ii] < 0:
                normal_vector[:, ii] = (-1) * normal_vector[:, ii]
                normalDistance2center[ii] = (-1)*normalDistance2center[ii]

        # Normalize
        normal_vector /= np.tile(LA.norm(normal_vector, axis=0), (self.dim,1))

        return normal_vector, normalDistance2center


    def draw_obstacle(self, include_margin=False, n_curve_points=5, numPoints=None):
        num_edges = self.edge_points.shape[1]

        if not type(numPoints)==type(None):
            n_curve_points = int(np.ceil(numPoints/(num_edges+1)) )
            # warnings.warn("Remove numPoints from function argument.")

        self.bounday_points = np.hstack((self.edge_points,
                                         np.reshape(self.edge_points[:,0],(self.dim,1))))

        for pp in range(self.edge_points.shape[1]):
            self.bounday_points[:,pp] = self.rotMatrix @ self.bounday_points[:, pp] + np.array([self.center_position])

        self.bounday_points[:, -1]  = self.bounday_points[:, 0]

        angles = np.linspace(0, 2*pi, num_edges*n_curve_points+1)

        obs_margin_cirlce = self.absolut_margin* np.vstack((np.cos(angles), np.sin(angles)))

        x_obs_sf = np.zeros((self.dim, 0))
        for ii in range(num_edges):
            x_obs_sf = np.hstack((x_obs_sf, np.tile(self.edge_points[:, ii], (n_curve_points+1, 1) ).T + obs_margin_cirlce[:, ii*n_curve_points:(ii+1)*n_curve_points+1] ))
        x_obs_sf  = np.hstack((x_obs_sf, x_obs_sf[:,0].reshape(2,1)))

        for jj in range(x_obs_sf.shape[1]): # TODO replace for loop with numpy-math
            x_obs_sf[:, jj] = self.rotMatrix @ x_obs_sf[:, jj] + np.array([self.center_position])

        # TODO rename more intuitively
        self.x_obs = self.bounday_points.T # Surface points
        self.x_obs_sf = x_obs_sf.T # Margin points


    def get_gamma(self, position, in_global_frame=False, norm_order=2, include_special_surface=True, gamma_type="proportional"):
        if in_global_frame:
            position = self.transform_global2relative(position)

        if isinstance(position, list):
            position = np.array(position)
        
        if gamma_type=="proportional":
        # TODO: extend rule to include points with Gamma < 1 for both cases
            dist2hull = np.ones(self.edge_points.shape[1])*(-1)
            is_intersectingSurfaceTile = np.zeros(self.edge_points.shape[1], dtype=bool)

            mag_position = LA.norm(position)
            if mag_position==0: # aligned with center, treat sepearately
                if self.is_boundary:
                    return sys.float_info.max
                else:
                    return 0 #
                
            reference_dir = position / mag_position

            if self.dim==2:
                for ii in range(self.edge_points.shape[1]):
                    # Use self.are_lines_intersecting() function
                    surface_dir = (self.edge_points[:, (ii+1)%self.edge_points.shape[1]] - self.edge_points[:, ii])

                    matrix_ref_tang = np.vstack((reference_dir, -surface_dir)).T
                    if LA.matrix_rank(matrix_ref_tang)>1:
                        dist2hull[ii], dist_tangent = LA.lstsq(np.vstack((reference_dir, -surface_dir)).T, self.edge_points[:, ii], rcond=None)[0]
                    else:
                        dist2hull[ii] = -1
                        dist_tangent = -1
                    dist_tangent = np.round(dist_tangent, 10)
                    is_intersectingSurfaceTile[ii] = (dist_tangent>=0) & (dist_tangent<=1)

                Gamma = mag_position/np.min(dist2hull[((dist2hull>0) & is_intersectingSurfaceTile)])
                
            elif self.dim==3:
                # for ii in range(self.n_planes):
                    # n_corners = self.ind_tiles[ii].shape[0]
                    # for jj in range(n_corners):
                        # edge_dir = self.edge_points[:, self.ind_tiles[ii, (jj+1)%n_corners]] \
                                   # - self.edge_points[:, self.ind_tiles[ii,jj]] 
                        # edge_dir = edge_dir / LA.norm(edge_dir)
                        # point2corner = position - self.edge_points[:, self.ind_tiles[ii,jj]]
                        # projection = edge_dir.T.dot(point2corner)

                        # perpendicular_line = point2corner - projection
                        # dist_edge = 
                        
                Gamma = 1
                
            else:
                raise ValueError("Not defined for d=={}".format(self.dim))
                
            if self.is_boundary:
                Gamma = 1/Gamma

        elif gamma_type=="norm2":
            distances2plane = self.get_distance_to_hullEdge(position)

            delta_Gamma = np.min(distances2plane) - self.absolut_margin
            ind_outside = (distances2plane > 0)
            delta_Gamma = (LA.norm(distances2plane[ind_outside], ord=norm_order)-self.absolut_margin)
            normalization_factor = np.max(self.normalDistance2center)
            # Gamma = 1 + delta_Gamma / np.max(self.axes_length)
            Gamma = 1 + delta_Gamma / normalization_factor
        else:
            raise TypeError("Unknown gmma_type {}".format(gamma_type))

        if Gamma<0: # DEBUGGING
            import pdb; pdb.set_trace() ## DEBUG ##
            
        return Gamma

    
    def get_normal_direction(self, position, in_global_frame=False, normalize=True, normal_calulation_type="distance"):
        if in_global_frame:
            position = self.transform_global2relative(position)
        
        mag_position = LA.norm(position)
        if mag_position==0: # aligned with center, treat sepearately
            if self.is_boundary:
                return np.ones(self.dim)/self.dim
            else:
                return np.ones(self.dim)/self.dim #

        if self.is_boundary:
            # Child and Current Class have to call Polygon
            Gamma = Polygon.get_gamma(self, position)
                
            if Gamma<0:
                return -self.get_reference_direction(position)
            
            temp_position = Gamma*Gamma*position
        else:
            temp_position = np.copy(position)

        temp_edge_points = np.copy(self.edge_points)
        temp_position = np.copy(temp_position)

        distances2plane = self.get_distance_to_hullEdge(temp_position)

        ind_outside = (distances2plane > 0)

        if not np.sum(ind_outside) : # zero value
            # TODO solver in a more proper way
            return self.get_reference_direction(position)

        distance2plane = ind_outside*np.abs(distances2plane)
        angle2hull = np.ones(ind_outside.shape)*pi

        for ii in np.arange(temp_edge_points.shape[1])[ind_outside]:
            # TODO - More complex for dimensiosn>2

            # Calculate distance to agent-position
            dir_tangent = (temp_edge_points[:, (ii+1)%temp_edge_points.shape[1]] - temp_edge_points[:, ii])
            position2edge = temp_position - temp_edge_points[:, ii]
            if dir_tangent.T.dot(position2edge) < 0:
                distance2plane[ii] = LA.norm(position2edge)
            else:
                dir_tangent = -(temp_edge_points[:, (ii+1)%temp_edge_points.shape[1]] - temp_edge_points[:, ii])
                position2edge = temp_position - temp_edge_points[:, (ii+1)%temp_edge_points.shape[1]]
                if dir_tangent.T.dot(position2edge) < 0:
                    distance2plane[ii] = LA.norm(position2edge)

            # TODO - don't use reference point, but little 'offset' to avoid singularity
            # Get closest point
            edge_points_temp = np.vstack((temp_edge_points[:,ii],
                                     temp_edge_points[:,(ii+1)%temp_edge_points.shape[1]])).T

            # Calculate angle to agent-position
            ind_sort = np.argsort(LA.norm(np.tile(temp_position,(2,1)).T-edge_points_temp, axis=0))

            tangent_line = edge_points_temp[:,ind_sort[1]] - edge_points_temp[:, ind_sort[0]]
            position_line = temp_position - edge_points_temp[:, ind_sort[0]]

            angle2hull[ii] = self.get_angle2dir(position_line, tangent_line)

        distance_weights = 1
        angle_weights = self.get_angle_weight(angle2hull)

        weights = distance_weights*angle_weights
        weights = weights/np.sum(weights)

        normal_vector = get_directional_weighted_sum(reference_direction=position, directions=self.normal_vector, weights=weights, normalize=False, obs=self, position=position, normalize_reference=True)
        
        if self.is_boundary and False:
            positionVec = position/mag_position # needed?
            normal_partParallel2positionVec = normal_vector.T.dot(positionVec)*positionVec
            normal_partPerpendicular2positionVec = normal_vector - normal_partParallel2positionVec

            # Mirror along position vector [back to original frame]
            normal_vector = normal_partPerpendicular2positionVec - normal_partParallel2positionVec

        if normalize:
            normal_vector = normal_vector/LA.norm(normal_vector)

        if False:# 
            # TODO: remove DEBUGGING
            # self.draw_reference_hull(normal_vector, position)
            pos_abs = self.transform_relative2global(position)
            pos_abs_temp = self.transform_relative2global(temp_position)
            norm_abs = self.transform_relative2global_dir(normal_vector)
            plt.quiver(pos_abs[0], pos_abs[1], norm_abs[0], norm_abs[1], color='g')
            plt.quiver(pos_abs_temp[0], pos_abs_temp[1], norm_abs[0], norm_abs[1], color='m')
            ref_abs = self.get_reference_direction(position)
            ref_abs = self.transform_relative2global_dir(ref_abs)
            plt.quiver(pos_abs[0], pos_abs[1], ref_abs[0], ref_abs[1], color='k')

            plt.ion()
            plt.show()

        return normal_vector
    

class Cuboid(Polygon):
    def __init__(self,  axes_length=[1,1], *args, **kwargs):
        '''
        This class defines obstacles to modulate the DS around it
        At current stage the function focuses on Ellipsoids, 
        but can be extended to more general obstacles
        '''
        self.axes_length = np.array(axes_length)
        
        dim = axes_length.shape[0] # Dimension of space

        if not dim==2:
            raise ValueError("Cuboid not yet defined for dimensions= {}".format(dim))

        edge_points = np.zeros((self.dim, 4))
        edge_points[:,0] = self.axes_length/2.0*np.array([1,1])
        edge_points[:,1] = self.axes_length/2.0*np.array([-1,1])
        edge_points[:,2] = self.axes_length/2.0*np.array([-1,-1])
        edge_points[:,3] = self.axes_length/2.0*np.array([1,-1])
        
        super().__init__(*args, edge_points=edge_points, **kwargs)
        


class DynamicBoundariesPolygon(Polygon):
    '''
    Dynamic Boundary for Application in 3D with surface polygons
    '''
    def __init__(self, *args, **kwargs):
        # define boundary functions
        super().__init__(*args, **kwargs)

    # @property
    # def 

    # @ 
