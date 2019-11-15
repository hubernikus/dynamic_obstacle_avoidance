#!/USSR/bin/python3
'''
@date 2019-10-15
@author Lukas Huber 
@mail lukas.huber@epfl.ch
'''

import time
import numpy as np
from math import sin, cos, pi, ceil
import warnings, sys

import numpy.linalg as LA

from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import *

from dynamic_obstacle_avoidance.obstacle_avoidance.state import *
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_dynamic_center_3d import *

import matplotlib.pyplot as plt

visualize_debug = False

class Obstacle(State):
    """ General class of obstacles """
    # TODO create obstacle container/list which can fast iterate through obstacles and perform other calculations
    def __init__(self, orientation=0, sf=1, delta_margin=0, sigma=1,  center_position=[0,0], tail_effect=True, always_moving=True, 
                 x0=None, th_r=None,
                 linear_velocity=[0,0], angular_velocity=0, xd=[0,0], w=0,
                 func_w=None, func_xd=None,  x_start=0, x_end=0, timeVariant=False,
                 Gamma_ref=0, is_boundary=False, hirarchy=0, ind_parent=-1):
                 # , *args, **kwargs): # maybe random arguments
         # This class defines obstacles to modulate the DS around it
        # At current stage the function focuses on Ellipsoids, but can be extended to more general obstacles

        self.sf = sf # TODO - rename
        self.delta_margin = delta_margin
        self.sigma = sigma
        self.tail_effect = tail_effect # Modulation if moving away behind obstacle

        # Obstacle attitude
        if type(x0) != type(None):
            center_position = x0 # TODO remove and rename
        self.position = center_position
        self.center_position = self.position
        self.x0 = center_position

        if type(th_r)!= type(None):
            _orientation = th_r
        self._orientation = orientation

        self.dim = len(self.center_position) #Dimension of space
        self.d = len(self.center_position) #Dimension of space # TODO remove

        self.rotMatrix = []
        self.compute_R() # Compute Rotation Matrix

        self.resolution = 0 #Resolution of drawing

        self.x_obs = [] # Numerical drawing of obstacle boundarywq
        self.x_obs_sf = [] # Obstacle boundary plus margin!

        self.timeVariant = timeVariant
        if self.timeVariant:
            self.func_xd = 0
            self.func_w = 0
        else:
            self.always_moving = always_moving

        if sum(np.abs(xd)) or w or self.timeVariant:
            # Dynamic simulation - assign varibales:
            self.x_start = x_start
            self.x_end = x_end
            self.always_moving = False
        else:
            self.x_start = 0
            self.x_end = 0

        if not isinstance(w, type(None)):
            angular_velocity = w
        self.angular_velocity = angular_velocity
        self.w = self.angular_velocity # TOOD - remove

        self.linear_velocity = linear_velocity
        self.xd = xd # TOOD - remove

        # Trees of stars
        self.hirarchy = hirarchy
        self.ind_parent = ind_parent

        # Relative Reference point // Dyanmic center
        # self.reference_point = self.center_position # TODO remove and rename
        self.reference_point = np.zeros(self.dim) # TODO remove and rename
        self.center_dyn = self.reference_point # TODO remove and rename

        self.reference_point_is_inside = True

        self.Gamma_ref = Gamma_ref
        self.is_boundary = is_boundary

        self.is_convex = False # Needed?

        self.properties = {} # TODO: use kwargs

    # def update_reference(self, new_ref):
        # TODo write function

    def transform_global2relative(self, position): # Inherit
        if len(position.shape)==1:
            return self.rotMatrix.T.dot(position - np.array(self.center_position))
        elif len(position.shape)==2:
            n_points = self.position.shape[1]
            return self.rotMatrix.T.dot(position.T - np.tile(self.center_position, (n_points,1)).T )
        else:
            warning.warn("Unexpected position-shape")

    def transform_relative2global(self, other): # TODO - inherit
        if not isinstance(other, (list, np.ndarray)):
            raise TypeError()
        if isinstance(other, (list)):
            other = np.array(other)
        if not other.shape[0]==self.dim:
            raise TypeError()

        if len(other.shape)==1:
            return self.rotMatrix.dot(other) + self.center_position
        elif len(other.shape)==2:
            # TODO - make it a oneliner without for loop to speed up
            for ii in range(other.shape[1]):
                other[:, ii] = self.rotMatrix.dot(other[:, ii]) + self.center_position
            return other
        # return (self.rotMatrix.dot(position))  + np.array(self.center_position)

    @property
    def orientation(self):
        return self._orientation
    
    @orientation.setter
    def orientation(self, value):
        self._orientation = value
        self.compute_R()

    @property
    def th_r(self):
        # TODO: remove since redundant
        return self.orientation # getter

    @th_r.setter
    def th_r(self, value):
        # TODO: remove since redundant
        self.orientation = value # setter
        
    def compute_R(self):
        # TODO - replace with quaternions
        # Find solution for higher dimensions

        # Compute the rotation matrix in 2D and 3D
        if self.orientation == 0:
            self.rotMatrix = np.eye(self.dim)
            return

        # rotating the query point into the obstacle frame of reference
        if self.dim==2:
            self.rotMatrix = np.array([[cos(self.orientation), -sin(self.orientation)],
                                       [sin(self.orientation),  cos(self.orientation)]])
        elif self.dim==3:
            R_x = np.array([[1, 0, 0,],
                        [0, np.cos(self.orientation[0]), np.sin(self.orientation[0])],
                        [0, -np.sin(self.orientation[0]), np.cos(self.orientation[0])] ])

            R_y = np.array([[np.cos(self.orientation[1]), 0, -np.sin(self.orientation[1])],
                        [0, 1, 0],
                        [np.sin(self.orientation[1]), 0, np.cos(self.orientation[1])] ])

            R_z = np.array([[np.cos(self.orientation[2]), np.sin(self.orientation[2]), 0],
                        [-np.sin(self.orientation[2]), np.cos(self.orientation[2]), 0],
                        [ 0, 0, 1] ])

            self.rotMatrix= R_x.dot(R_y).dot(R_z)
        else:
            warnings.warn('rotation not yet defined in dimensions d > 3 !')
            self.rotMatrix = np.eye(self.dim)


    def transform_relative2global_dir(self, direction): # TODO - inherit
        return self.rotMatrix.dot(direction)

    def transform_global2relative_dir(self, direction): # TODO - inherit
        return self.rotMatrix.T.dot(direction)

    @property
    def global_reference_point(self):
        return self.transform_relative2global(self.reference_point)
        
    def extend_hull_around_reference(self, edge_reference_dist=0.3):
        # TODO add margin

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


    def set_reference_point(self, position, in_global_frame=False): # Inherit
        # TODO --- Check if correct
        if in_global_frame:
            position = self.transform_global2relative(position)
            
        self.reference_point = position
        self.center_dyn = self.reference_point

        if self.get_gamma(self.reference_point)>1:
            self.extend_hull_around_reference()
            self.reference_point_is_inside = False
        else:
            self.reference_point_is_inside = True


    def move_obstacle_to_referencePoint(self, position, in_global_frame=True):
        if not in_global_frame:
            position = self.transform_relative2global(position)

        self.center_position = position
        # self.reference_point = position
        # self.center_dyn = self.reference_point


    def update_position_and_orientation(self, position, orientation,
                                        k_position=0.1, k_linear_velocity=0.1,
                                        k_orientation=0.1, k_angular_velocity=0.1):
        # TODO implement Kalman filter
        time_current = time.time()
        dt = self.timestamp-time_current

        new_linear_velocity = (self.position-position)/dt
        new_angular_velocity = (self.orientation-orientation)/dt

        self.linear_velocity = k_linear_velocity*new_angular_velocity + (1-k_angular_velocity)*self.angular_velocity
        self.angular_velocity = k_angular_velocity*new_angular_velocity + (1-k_angular_velocity)*self.angular_velocity

        self.position = (k_position*(position)
                         + (1-k_position)*(self.linear_velocity*dt + self.position) )
        self.orienation = (k_orientation*(orientation)
                           + (1-k_orienation)*(self.angular_velocity*dt + self.position) )

        self.timestamp = time_current

    def are_lines_intersecting(self, direction_line, passive_line):
        # solve equation line1['point_start'] + a*line1['direction'] = line2['point_end'] + b*line2['direction']
        connection_direction = np.array(direction_line['point_end']) - np.array(direction_line['point_start'])
        connection_passive = np.array(passive_line['point_end']) - np.array(passive_line['point_start'])
        connection_matrix = np.vstack((connection_direction, -connection_passive)).T

        if LA.det(connection_matrix): # nonzero value
            direction_factors = (LA.inv(connection_matrix) @
                                 (np.array(passive_line['point_start'])
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
        if len(Gamma.shape)>1:
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

            cos_position2edge = vec_position2edge[:, ii].T @ self.tangent_vector[:,ii]
            angle2refencePatch[ii] = np.arccos(cos_position2edge)

        if False:
            cos_tangs = np.sum(self.tangent_vector[:,0].T @ self.tangent_vector[:,1])
            print('angle', np.arccos(cos_tangs))

        return angle2refencePatch


    def get_angle_weight(self, angles, max_angle=pi, min_angle=0, check_range=False, weight_pow=1):
        # n_angless = np.array(angles).shape[0]

        if check_range:
            ind_low = angles <= min_angle
            if np.sum(ind_low):
                return ind_low/np.sum(ind_low)

            angles = np.min(np.vstack((angles, np.ones(n_angles)*max_angle)) )

        # [min, max] -> [0, 1] weights
        weights = (angles-min_angle)/(max_angle-min_angle)

        # [min, max] -> [infty, 1]
        weights = 1/weights

        # [min, max] -> [infty, 0]
        weights = (weights - 1)**weight_pow

        weight_norm = np.sum(weights)


        if weight_norm:
            return weights/weight_norm
        return weights

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

                

    def obs_check_collision(self, ):
        print('TODO: check class')



class Ellipse(Obstacle):
    def __init__(self, *args,
                 axes_length=None, a=None, p=[1,1],
                 **kwargs):
        # import pdb; pdb.set_trace() ## DEBUG ##
        super().__init__(*args, **kwargs)

        # TODO: remove redundant Leave at the moment for backwards compatibility
        if type(axes_length) == type(None):
            self.a = a
            self.axes = np.array(a)
            self.axes_length = np.array(a)
        else:
            self.axes_length = axes_length
            self.axes = np.array(axes_length)
            self.a = axes_length
            
        self.margin_axes =  self.axes*np.array(self.sf)+np.array(self.delta_margin)

        # TODO: rename & make p one value (NOT array)
        self.p = np.array(p) # TODO - rename
        
        self.is_convex = True

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

                    try:
                        normal_vector = get_directional_weighted_sum(reference_direction=position, directions=self.normal_vector, weights=weights, normalize=False, obs=self, position=position, normalize_reference=True)

                    except:
                        # pass
                        import pdb; pdb.set_trace() ## DEBUG ##
                    # return normal_vector

        if self.reference_point_is_inside or np.sum(ind_intersect)==0:
            # normal_vector = (2*self.p/self.margin_axes*(position/self.margin_axes)**(2*self.p-1))
            normal_vector = (2*self.p/self.axes_length*(position/self.axes_length)**(2*self.p-1))

        if normalize:
            normal_vector = normal_vector/LA.norm(normal_vector)

        if in_global_frame:
            normal_vector = self.transform_relative2global_dir(normal_vector) 

        return normal_vector


    def get_gamma(self, position, in_global_frame=False, gamma_type='proportional'):
        # if not type(position)==np.ndarray:
            # position = np.array(position)

        if in_global_frame:
            position = self.transform_global2relative(position)

        if not self.reference_point_is_inside:
            for ii in np.arange(self.tangent_points.shape[1]):
                reference_line = {"point_start":[0,0], "point_end":position}

                # TODO - don't use reference point, but little 'offset' to avoid singularity
                tangent_line = {"point_start":self.hull_edge,
                                "point_end":self.tangent_points[:, ii]}

                ind_intersect, dist_intersect = self.are_lines_intersecting(reference_line, tangent_line)
                if ind_intersect:
                    return LA.norm(position)/dist_intersect

        # Original Gamma
        if gamma_type=='proportional':
            Gamma = np.sum((position / self.margin_axes)**(2*self.p[0]))  # distance
            Gamma = Gamma**(1./(2*self.p[0]))
        else:
            raise NotImplementedError()
        
        if len(Gamma.shape)>1:
            Gamma = (position / np.tile(self.axes_length, (N_points,1)).T )
            Gamma = np.sum( (1/self.sf * Gamma) ** (2*np.tile(self.p, (N_points,1)).T), axis=0)

        if self.is_boundary:
            Gamma = self.get_boundaryGamma(Gamma,Gamma_ref=self.Gamma_ref)
        return Gamma

    
    def draw_ellipsoid(self, *args, **kwargs):
        # TODO remove
        raise NotImplementedError("<<draw_ellipsoid>> has been renamed <<draw_obstacle>>")
    
    
    def draw_obstacle(self, numPoints=20, a_temp = [0,0], draw_sfObs=False):
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

        if sum(a_temp) == 0:
            a = self.a
        else:
            a = a_temp

        p = self.p[:]
        R = np.array(self.rotMatrix)

        x_obs = np.zeros((self.d, numPoints))

        if self.d == 2:
            x_obs[0,:] = a[0]*np.cos(theta)
            x_obs[1,:] = np.copysign(a[1], theta)*(1 - np.cos(theta)**(2*p[0]))**(1./(2.*p[1]))
        else:
            x_obs[0,:] = (a[0]*np.cos(phi)*np.cos(theta)).reshape((1,-1))
            x_obs[1,:] = (a[1]*np.copysign(1, theta)*np.cos(phi)*(1 - np.cos(theta)**(2*p[0]))**(1./(2.*p[1]))).reshape((1,-1))
            x_obs[2,:] = (a[2]*np.copysign(1,phi)*(1 - (np.copysign(1,theta)*np.cos(phi)*(1 - 0 ** (2*p[2]) - np.cos(theta)**(2*p[0]))**(1/(2**p[1])))**(2*p[1]) - (np.cos(phi)*np.cos(theta)) ** (2*p[0])) ** (1/(2*p[2])) ).reshape((1,-1))


        if not self.reference_point_is_inside:
            x_obs[:,0] = x_obs[:,-1] = self.hull_edge

        x_obs_sf = np.zeros((self.d,numPoints))
        if not hasattr(self, 'sf'):
            self.sf = 1

        if type(self.sf) == int or type(self.sf) == float:
            x_obs_sf = R @ (self.sf*x_obs) + np.tile(np.array([self.center_position]).T,(1,numPoints))
        else:
            x_obs_sf = R @ (x_obs*np.tile(self.sf,(1,numPoints))) + np.tile(self.center_position, (numPoints,1)).T

        x_obs = R @ x_obs + np.tile(np.array([self.center_position]).T,(1,numPoints))

        if sum(a_temp) == 0:
            # self.x_obs = x_obs.T.tolist()
            # self.x_obs_sf = x_obs_sf.T.tolist()
            self.x_obs = x_obs.T
            self.x_obs_sf = x_obs_sf.T
        else:
             return x_obs_sf

    def get_radius_of_angle(self, angle, in_global_frame=False):
        if in_global_frame:
            position =  transform_polar2cartesian(magnitude=10, angle=angle-self.orientation)
        else:
            position = transform_polar2cartesian(magnitude=1, angle=angle)
        
        gamma = self.get_gamma(position, gamma_type='proportional')
        return LA.norm(position)/gamma

    
class StarshapedFlower(Obstacle):
    def __init__(self,  radius_magnitude=1, radius_mean=2, number_of_edges=4,
                 *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        # Object Specific Paramters
        self.radius_magnitude=radius_magnitude
        self.radius_mean=radius_mean

        self.number_of_edges=number_of_edges

    def get_radius_of_angle(self, angle, in_global_frame=False):
        if in_global_frame:
            angle -= self.orientation
        return self.radius_mean+self.radius_magnitude*np.cos((angle)*self.number_of_edges)


    def get_radiusDerivative_of_angle(self, angle, in_global_frame=False):
        if in_global_frame:
            angle -= self.orientation
        return -self.radius_magnitude*self.number_of_edges*np.sin((angle)*self.number_of_edges)


    def draw_obstacle(self, include_margin=False, n_curve_points=100, numPoints=None):
        # warnings.warn("Remove numPoints from function argument.")

        angular_coordinates = np.linspace(0,2*pi, n_curve_points)
        radius_angle = self.get_radius_of_angle(angular_coordinates)

        if self.dim==2:
            direction = np.vstack(( np.cos(angular_coordinates), np.sin(angular_coordinates) ))

        self.x_obs = (radius_angle * direction)
        self.x_obs_sf = (radius_angle * self.sf * direction)

        if self.orientation: # nonzero
            for jj in range(self.x_obs.shape[1]):
                self.x_obs[:, jj] = self.rotMatrix @ self.x_obs[:, jj] + np.array([self.center_position])
            for jj in range(self.x_obs_sf.shape[1]):
                self.x_obs_sf[:,jj] = self.rotMatrix @ self.x_obs_sf[:, jj] + np.array([self.center_position])

        self.x_obs = self.x_obs.T
        self.x_obs_sf = self.x_obs_sf.T


    def get_gamma(self, position, in_global_frame=False, norm_order=2):
        if not type(position)==np.ndarray:
            position = np.array(position)

        # Rename
        if in_global_frame:
            position = self.transform_global2relative(position)

        mag_position = LA.norm(position)
        if mag_position==0:
            if self.is_boundary:
                return sys.float_info.max
            else:
                return 0

        direction = np.arctan2(position[1], position[0])

        Gamma = mag_position/self.get_radius_of_angle(direction)

        # TODO extend rule to include points with Gamma < 1 for both cases
        if self.is_boundary:
            Gamma = 1/Gamma

        return Gamma


    def get_normal_direction(self, position, in_global_frame=False, normalize=True):
        if in_global_frame:
            position = self.transform_global2relative(position)

        mag_position = LA.norm(position)
        if not mag_position:
            return np.ones(self.dim)/self.dim # just return one direction

        direction = np.arctan2(position[1], position[0])
        derivative_radius_of_angle = self.get_radiusDerivative_of_angle(direction)

        radius = self.get_radius_of_angle(direction)

        normal_vector = np.array(([
            derivative_radius_of_angle*np.sin(direction) + radius*np.cos(direction),
            - derivative_radius_of_angle*np.cos(direction) + radius*np.sin(direction)]))

        if normalize:
            mag_vector = LA.norm(normal_vector)
            if mag_vector: #nonzero
                normal_vector = normal_vector/mag_vector
                
        if False:
            # self.draw_reference_hull(normal_vector, position)
            pos_abs = self.transform_relative2global(position)
            norm_abs = self.transform_relative2global_dir(normal_vector)
            plt.quiver(pos_abs[0], pos_abs[1], norm_abs[0], norm_abs[1], color='g')
            ref_abs = self.get_reference_direction(position)
            ref_abs = self.transform_relative2global_dir(ref_abs)
            plt.quiver(pos_abs[0], pos_abs[1], ref_abs[0], ref_abs[1], color='k')

            plt.ion()
            plt.show()

        if in_global_frame:
            normal_vector = self.transform_relative2global_dir(normal_vector)
            
        return normal_vector
