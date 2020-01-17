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

# import quaternion # numpy-quaternion 
# import dynamic_obstacle_avoidance

from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import *

from dynamic_obstacle_avoidance.obstacle_avoidance.state import *
from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import angle_modulo
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_dynamic_center_3d import *

import matplotlib.pyplot as plt

# import quaternion

visualize_debug = False


class Obstacle(State):
    """ 
    (Virtual) base class of obstacles 
    """

    # TODO -- enforce certain functions
    def __repr__(self):
        return "Obstacle of Type: {}".format(type(self))

    def __init__(self, orientation=0, sf=1, delta_margin=0, sigma=1,  center_position=[0,0], tail_effect=True, always_moving=True, 
                 x0=None, th_r=None, dimension=None,
                 # position_3d=None, orientation_3d=None, # 3D compatibility
                 linear_velocity=None, angular_velocity=None, xd=None, w=None,
                 func_w=None, func_xd=None,  x_start=0, x_end=0, timeVariant=False,
                 Gamma_ref=0, is_boundary=False, hirarchy=0, ind_parent=-1):
                 # , *args, **kwargs): # maybe random arguments
        # This class defines obstacles to modulate the DS around it
        # At current stage the function focuses on Ellipsoids, but can be extended to more general obstacles

        self.sf = sf # TODO - rename
        self.delta_margin = delta_margin
        self.sigma = sigma
        self.tail_effect = tail_effect # Modulation if moving away behind obstacle

        # if not position_3d is None:
            # self.position_3d = position_3d
        # if not orientation_3d is None:
            # self.orientatation_3d = position_3d

        # Obstacle attitude
        if type(x0) != type(None):
            center_position = x0 # TODO remove and rename
        self.position = center_position
        self.center_position = self.position
        self.x0 = center_position

        self.dim = len(self.center_position) # Dimension of space
        self.d = len(self.center_position) # Dimension of space # TODO remove
        
        if type(th_r)!= type(None):
            orientation = th_r
        self.orientation = orientation

        self.rotMatrix = []
        self.compute_R() # Compute Rotation Matrix

        self.resolution = 0 #Resolution of drawing

        self._boundary_points = [] # Numerical drawing of obstacle boundarywq
        self._boundary_points_margin = [] # Obstacle boundary plus margin!

        self.timeVariant = timeVariant
        if self.timeVariant:
            self.func_xd = 0
            self.func_w = 0
        else:
            self.always_moving = always_moving
        
        if angular_velocity is None:
            if w is None:
                if self.dim==2:
                    angular_velocity = 0
                elif self.dim==3:
                    angular_velocity = np.zeros(self.dim)
                else:
                    raise ValueError("Define angular velocity for higher dimensions.")
            else:
                angular_velocity = w
        self.angular_velocity = angular_velocity
        self.w = self.angular_velocity # TOOD - remove

        if linear_velocity is None:
            if xd is None:
                linear_velocity=np.zeros(self.dim)
            else:
                linear_velocity = xd
        self.linear_velocity = linear_velocity
        self.xd = self.linear_velocity
        
        # Special case of moving obstacle (Create subclass)
        if sum(np.abs(self.linear_velocity)) or np.sum(self.angular_velocity) \
           or self.timeVariant:
            # Dynamic simulation - assign varibales:
            self.x_start = x_start
            self.x_end = x_end
            self.always_moving = False
        else:
            self.x_start = 0
            self.x_end = 0

        self.update_timestamp()

        # Trees of stars // move to 'properties'
        self.hirarchy = hirarchy
        self.ind_parent = ind_parent
        self.ind_children = []

        # Relative Reference point // Dyanmic center
        # self.reference_point = self.center_position # TODO remove and rename
        self.reference_point = np.zeros(self.dim) # TODO remove and rename
        # self.center_dyn = self.reference_point # TODO remove and rename

        self.reference_point_is_inside = True

        self.Gamma_ref = Gamma_ref
        self.is_boundary = is_boundary

        self.is_convex = False # Needed?

        self.properties = {} # TODO: use kwargs
        

    # def update_reference(self, new_ref):
        # TODo write function

    
    def transform_global2relative(self, position): 
        if not position.shape[0]==self.dim:
            raise ValueError("Wrong position dimensions")
            
        if len(position.shape)==1:
            return self.rotMatrix.T.dot(position - np.array(self.center_position))
        elif len(position.shape)==2:
            n_points = position.shape[1]
            return self.rotMatrix.T.dot(position - np.tile(self.center_position, (n_points,1)).T)
        else:
            raise ValueError("Unexpected position-shape")

    def transform_relative2global(self, position):
        if not isinstance(position, (list, np.ndarray)):
            raise TypeError()

        if isinstance(position, (list)):
            position = np.array(position)
            
        if not position.shape[0]==self.dim:
            raise TypeError()

        if len(position.shape)==1:
            return self.rotMatrix.dot(position) + self.center_position
        elif len(position.shape)==2:
            # TODO - make it a oneliner without for loop to speed up
            # for ii in range(position.shape[1]):
                # position[:, ii] = self.rotMatrix.dot(position[:, ii]) + self.center_position
            # return position
            n_points = position.shape[1]
            return self.rotMatrix.dot(position) + np.tile(self.center_position, (n_points,1)).T
        # return (self.rotMatrix.dot(position))  + np.array(self.center_position)
        else:
            raise ValueError("Unexpected position-shape")
        
    def transform_relative2global_dir(self, direction): # TODO - inherit
        return self.rotMatrix.dot(direction)

    def transform_global2relative_dir(self, direction): # TODO - inherit
        return self.rotMatrix.T.dot(direction)

    @property
    def center_dyn(self):# TODO: depreciated -- delete
        return self.reference_point
    
    @property
    def global_reference_point(self):
        # Rename kernel-point?
        return self.transform_relative2global(self._reference_point)
    
    # @global_reference_point.setter
    # def global_reference_point(self, value):
        # self._reference_point = self.transform_global2relative(value)
        
    # @property
    # def kernel_point(self):
        # Rename kernel-point?
        # return self._reference_point
    
    # @kernel_point.setter
    # def kernel_point(self, value):
        # self._reference_point = value
        
    @property
    def reference_point(self):
        # Rename kernel-point?
        return self._reference_point

    @reference_point.setter
    def reference_point(self, value):
        self._reference_point = value
    # @reference_point.setter
    # def reference_point(self, value):
        # self._reference_point = value

    @property
    def orientation(self):
        return self._orientation
    
    @orientation.setter
    def orientation(self, value):
        if isinstance(value, list) and self.dim==3:
            self._orientation = np.array(value) # TODO: change to quaternion
        else:
            self._orientation = value
        self.compute_R()

    @property
    def position(self):
        return self.center_position

    @position.setter
    def position(self, value):
        self.center_position = value
    
    @property
    def center_position(self):
        return self._center_position
    
    @center_position.setter
    def center_position(self, value):
        if isinstance(value, list):
            self._center_position = np.array(value) 
        else:
            self._center_position = value

    @property
    def th_r(self): # TODO: will be removed since outdated
        return self.orientation # getter

    @th_r.setter
    def th_r(self, value): # TODO: will be removed since outdated
        self.orientation = value # setter

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        # if timestamp is None:
        self._timestamp = value

    def update_timestamp(self):
        self._timestamp = time.time()

    @property
    def linear_velocity(self):
        return self._linear_velocity

    @linear_velocity.setter
    def linear_velocity(self, value):
        if len(value)>2:
            import pdb; pdb.set_trace()
        self._linear_velocity = value

    @property
    def boundary_points_local(self):
        return self._boundary_points

    @property
    def x_obs(self):
        return self.boundary_points_global

    @property
    def boundary_points_global_closed(self):
        boundary = self.boundary_points_global
        return np.hstack((boundary, boundary[:, 0:1]))

    @property
    def boundary_points_global(self):
        return self.transform_relative2global(self._boundary_points)

    @property
    def boundary_points_margin_local(self):
        return self._boundary_points_margin

    @property
    def x_obs_sf(self):
        return self.boundary_points_margin_global

    @property
    def boundary_points_margin_global(self):
        return self.transform_relative2global(self._boundary_points_margin)

    @property
    def boundary_points_margin_global_closed(self):
        boundary = self.boundary_points_margin_global
        return np.hstack((boundary, boundary[:, 0:1]))


    # @boundary_points.setter
    # def boundary_points
        
    def compute_R(self):
        # TODO - replace with quaternions
        # Find solution for higher dimensions
        orientation = self._orientation

        # Compute the rotation matrix in 2D and 3D
        if orientation is None:
            self.rotMatrix = np.eye(self.dim)
            return

        # rotating the query point into the obstacle frame of reference
        if self.dim==2:
            self.rotMatrix = np.array([[cos(orientation), -sin(orientation)],
                                       [sin(orientation),  cos(orientation)]])
        elif self.dim==3:
            R_x = np.array([[1, 0, 0,],
                        [0, np.cos(orientation[0]), np.sin(orientation[0])],
                        [0, -np.sin(orientation[0]), np.cos(orientation[0])] ])

            R_y = np.array([[np.cos(orientation[1]), 0, -np.sin(orientation[1])],
                        [0, 1, 0],
                        [np.sin(orientation[1]), 0, np.cos(orientation[1])] ])

            R_z = np.array([[np.cos(orientation[2]), np.sin(orientation[2]), 0],
                        [-np.sin(orientation[2]), np.cos(orientation[2]), 0],
                        [ 0, 0, 1] ])

            self.rotMatrix= R_x.dot(R_y).dot(R_z)
        else:
            warnings.warn('rotation not yet defined in dimensions d > 3 !')
            self.rotMatrix = np.eye(self.dim)

    


    def set_reference_point(self, position, in_global_frame=False): # Inherit
        # TODO --- Check if correct
        if in_global_frame:
            position = self.transform_global2relative(position)
            
        self.reference_point = position
        
        self.extend_hull_around_reference()

    def move_obstacle_to_referencePoint(self, position, in_global_frame=True):
        if not in_global_frame:
            position = self.transform_relative2global(position)

        self.center_position = position
        # self.reference_point = position
        # self.center_dyn = self.reference_point

    def move_center(self, position):
        self.center_position = position


    def update_position_and_orientation(self, position, orientation, k_position=0.1, k_linear_velocity=0.1, k_orientation=0.1, k_angular_velocity=0.1):

        # TODO implement Kalman filter
        time_current = time.time()
        dt = time_current-self.timestamp
        
        if isinstance(position, list):
            position = np.array(position)

        if self.dim==2:
            # 2D navigation, but 3D sensor input

            new_linear_velocity = (position-self.position)/dt
            
            # Periodicity of oscillation
            delta_orientation = orientation-self.orientation
            if np.abs(delta_orientation)>pi:
                if np.abs(delta_orientation+2*pi)<pi:
                    orientation += 2*pi
                elif np.abs(delta_orientation-2*pi)<pi:
                    orientation -= 2*pi
                else:
                    raise ValueError('Unexpected angle difference')

            new_angular_velocity = (orientation-self.orientation)/dt
            
            self.linear_velocity = new_linear_velocity 
            self.angular_velocity = new_angular_velocity
            # self.linear_velocity = k_linear_velocity*new_linear_velocity + (1-k_linear_velocity)*self.linear_velocity
            # self.angular_velocity = k_angular_velocity*new_angular_velocity + (1-k_angular_velocity)*self.angular_velocity
            
            # self.center_position = (position)
            self.center_position = (k_position*(position) \
                                    + (1-k_position)*(self.linear_velocity*dt + self.center_position))
            self.orientation = (k_orientation*(orientation) + (1-k_orientation)*(self.angular_velocity*dt + self.orientation) ) # TODO: UPDATE ORIENTATION ROTATIONAL
            self.orientation = angle_modulo(self.orientation)
            # import pdb; pdb.set_trace()
            #TODO add filter

        self.timestamp = time_current

    def are_lines_intersecting(self, direction_line, passive_line):
        # TODO only return intersection point or None
        # solve equation line1['point_start'] + a*line1['direction'] = line2['point_end'] + b*line2['direction']
        connection_direction = np.array(direction_line['point_end']) - np.array(direction_line['point_start'])
        connection_passive = np.array(passive_line['point_end']) - np.array(passive_line['point_start'])
        connection_matrix = np.vstack((connection_direction, -connection_passive)).T
        
        # if True:
            # plt.plot()
            # plt.plot(self.boundary_points_local[:, 0], self.boundary_points_local[:, 1], 'k--')
            # import pdb; pdb.set_trace() ## DEBUG ##
        
        if LA.det(connection_matrix): # nonzero value
            direction_factors = (LA.inv(connection_matrix).dot(
                                 np.array(passive_line['point_start'])
                                  - np.array(direction_line['point_start']) ))

            # Smooth because it's a tangent
            if direction_factors[0]>=0:
                if direction_factors[1]>=0 and LA.norm(direction_factors[1]*connection_passive) <= LA.norm(connection_passive):

                    return True, LA.norm(direction_factors[0]*connection_direction)
 
        if False: # show plot
            dir_start = self.transform_relative2global(direction_line['point_start'])
            dir_end = self.transform_relative2global(direction_line['point_end'])

            pas_start = self.transform_relative2global(passive_line['point_start'])
            pas_end = self.transform_relative2global(passive_line['point_end'])

            plt.ion()
            plt.plot([dir_start[0], dir_end[0]], [dir_start[1], dir_end[1]], 'g--')
            plt.plot([pas_start[0], pas_end[0]], [pas_start[1], pas_end[1]], 'r--')
            plt.show()
            print('done intersections')

        return False, -1


    def get_obstacle_radius(self, position, in_global_frame=False, Gamma=None): # Inherit
        if in_global_frame:
            position = self.transform_global2relative(position)

        if not Gamma==None:
            Gamma = self.get_gamma(position)
        dist_to_center = LA.norm(position)

        return dist_to_center/Gamma

    def get_reference_point(self, in_global_frame=False): # Inherit
        if in_global_frame:
            return self.transform_relative2global(self.reference_point)
        else:
            return self.reference_point

    def get_boundaryGamma(self, Gamma, Gamma_ref=0):
        if len(Gamma)>1:
            ind_small_gamma = (Gamma <= Gamma_ref)
            Gamma[ind_small_gamma] = sys.float_info.max
            Gamma[~ind_small_gamma] = (1-Gamma_ref)/(Gamma[~ind_small_gamma]-Gamma_ref)
            return Gamma
        else:
            if Gamma <= Gamma_ref:
                return sys.float_info.max
            else:
                return (1-Gamma_ref)/(Gamma-Gamma_ref)

    def get_distance_to_hullEdge(self, position, hull_edge=None):
        if type(hull_edge)==type(None):
            hull_edge = self.hull_edge
            normal_vector = self.normal_vector
        else:
            normal_vector, dist = self.calculate_normalVectorAndDistance(hull_edge)


        hull_edge = hull_edge.reshape(self.dim, -1)
        n_planes = hull_edge.shape[1]
        if len(hull_edge.shape)<2:
            vec_position2edge = np.tile(position-hull_edge, (n_planes, 1)).T
        else:
            vec_position2edge = np.tile(position, (n_planes, 1)).T - hull_edge
                
        distance2plane = np.sum((normal_vector * vec_position2edge), axis=0)

        if False:
            vec_position2edge = np.tile(position, (n_planes, 1)).T - self.tangent_points
            distance2plane = np.sum((self.normal_vector * vec_position2edge), axis=0)

        return distance2plane


    def get_angle2dir(self, position_dir, tangent_dir, needs_normalization=True):
        if needs_normalization:
            if len(position_dir.shape) > 1:
                position_dir /= np.tile(LA.norm(position_dir,axis=0), (self.dim, 1))
                tangent_dir /= np.tile(LA.norm(tangent_dir, axis=0), (self.dim, 1))
                angle_arccos = np.sum(position_dir * tangent_dir, axis=0)
            else:
                position_dir /= LA.norm(position_dir)
                tangent_dir /= LA.norm(tangent_dir)
                angle_arccos = np.sum(position_dir * tangent_dir)
        return np.arccos(angle_arccos)


    def get_angle2referencePatch(self, position, max_angle=pi):
        # angle between 0 and pi
        n_planes = self.normal_vector.shape[1]

        vec_position2edge = np.tile(position-self.hull_edge, (n_planes, 1)).T
        distance2plane = np.sum((self.normal_vector*vec_position2edge), axis=0)

        angle2refencePatch = np.ones(n_planes)*max_angle


        for ii in range(n_planes):
            if distance2plane[ii]<0:
                continue

            vec_position2edge[:, ii] /= LA.norm(vec_position2edge[:, ii])

            cos_position2edge = vec_position2edge[:, ii].T.dot(self.tangent_vector[:,ii])
            angle2refencePatch[ii] = np.arccos(cos_position2edge)

        if False:
            cos_tangs = np.sum(self.tangent_vector[:,0].T.dot(self.tangent_vector[:,1]))
            print('angle', np.arccos(cos_tangs))

        return angle2refencePatch


    def get_angle_weight(self, angles, max_angle=pi, min_angle=0, check_range=False, weight_pow=1):
        # n_angless = np.array(angles).shape[0]
        if check_range:
            ind_low = angles <= min_angle
            if np.sum(ind_low):
                return ind_low/np.sum(ind_low)

            angles = np.min(np.vstack((angles, np.ones(n_angles)*max_angle)) )

        zero_ind = angles<=min_angle
        if np.sum(zero_ind):
            return zero_ind/np.sum(zero_ind)

        nonzero_ind = angles<max_angle
        if not np.sum(nonzero_ind):
            warnings.warn("No angle has an influence")
            # print('Angles', angles)
            return np.zeros(angles.shape)
        
        elif np.sum(nonzero_ind)==1:
            return nonzero_ind*1.0
        
        # [min, max] -> [0, 1] weights
        weights = (angles[nonzero_ind]-min_angle)/(max_angle-min_angle)
        
        # [min, max] -> [infty, 1]
        weights = 1/weights

        # [min, max] -> [infty, 0]
        weights = (weights - 1)**weight_pow

        weight_norm = np.sum(weights)
        
        if weight_norm:
            weights =  weights/weight_norm

        weights_all = np.zeros(angles.shape)
        weights_all[nonzero_ind] = weights 
        return weights_all

    
    def get_distance_weight(self, distance, power=1, distance_min=0):
        ind_positiveDistance = (distance>0)

        distance = distance - distance_min
        weights = np.zeros(distance.shape)
        weights[ind_positiveDistance] = (1./distance[ind_positiveDistance])**power
        weights[ind_positiveDistance] /= np.sum(weights[ind_positiveDistance])
        # weights[~ind_positiveDistance] = 0

        return weights

    
    def draw_reference_hull(self, normal_vector, position):
        pos_abs = self.transform_relative2global(position)
        norm_abs = self.transform_relative2global_dir(normal_vector)

        plt.quiver(pos_abs[0], pos_abs[1], norm_abs[0], norm_abs[1], color='k', label="Normal")

        ref_dir = self.transform_relative2global_dir(self.get_reference_direction(position, in_global_frame=False, normalize=True))

        plt.quiver(pos_abs[0], pos_abs[1], ref_dir[0], ref_dir[1], color='g', label="Reference")

        ref_abs = self.transform_relative2global(self.hull_edge)

        for ii in range(2):
            tang_abs = self.transform_relative2global(self.tangent_points[:, ii])
            plt.plot([tang_abs[0], ref_abs[0]], [tang_abs[1], ref_abs[1]], 'k--')

        # plt.ion()
        # plt.show()

    # @property
    # def global_reference_(self, position):
        # self.get_reference_direction(position, in_global_frame=True)

        
    def get_reference_direction(self, position, in_global_frame=False, normalize=True):
        # Inherit
        if in_global_frame:
            position = self.transform_global2relative(position)

        if hasattr(self, 'reference_point') or hasattr(self,'center_dyn'):  # automatic adaptation of center
            reference_direction = - (position-self.reference_point)
        else:
            reference_direction = - position

        if normalize:
            ref_norm = LA.norm(reference_direction)
            if ref_norm>0: 
                return reference_direction/ref_norm

        if in_global_frame:
            reference_direction = self.transform_global2relative_dir(reference_direction)

        return reference_direction

    def update_pos(self, t, dt, x_lim=[], y_lim=[]):
        # Inherit
        # TODO - implement function dependend movement (yield), nonlinear integration
        # First order Euler integration

        if self.always_moving or self.x_end > t :
            if self.always_moving or self.x_start<t:
                # Check if xd and w are functions
                if self.timeVariant:
                    # TODO - implement RK4 for movement

                    self.xd = self.func_xd(t)
                    self.w = self.func_w(t)

                self.center_position = [self.center_position[i] + dt*self.xd[i] for i in range(self.d)] # update position

                if len(x_lim):
                    self.center_position[0] = np.min([np.max([self.center_position[0], x_lim[0]]), x_lim[1]])
                if len(y_lim):
                    self.center_position[1] = np.min([np.max([self.center_position[1], y_lim[0]]), y_lim[1]])

                if self.w: # if new rotation speed

                    if self.d <= 2:
                        self.th_r = self.th_r + dt*self.w  #update orientation/attitude
                    else:
                        self.th_r = [self.th_r[i]+dt*self.w[i] for i in range(self.d)]  #update orientation/attitude
                    self.compute_R() # Update rotation matrix

                self.draw_obstacle()

    def get_scaled_boundary_points(self, scale, safety_margin=True, redraw_obstacle=False):
        # Draws at 1:scale
        if safety_margin:
            scaled_boundary_points = scale*self._boundary_points_margin
        else:
            scaled_boundary_points = scale*self._boundary_points
            
        return self.transform_relative2global(scaled_boundary_points)
        
    def obs_check_collision(self, ):
        print('TODO: check class')
        raise NotImplementedError()
