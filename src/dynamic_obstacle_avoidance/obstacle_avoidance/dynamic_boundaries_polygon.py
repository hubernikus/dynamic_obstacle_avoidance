#!/USSR/bin/python3

from math import pi
import numpy as np
import copy
# import itertools
import sys
import matplotlib.pyplot as plt # TODO: remove after debugging

from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import get_directional_weighted_sum
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import Polygon


class DynamicBoundariesPolygon(Polygon):
    '''
    Dynamic Boundary for Application in 3D with surface polygons

    Pyramid shape (without top edge)
    '''
    def __init__(self, indeces_of_flexibleTiles=None, inflation_parameter=None, is_surgery_setup=False, 
                 *args, **kwargs):

        if is_surgery_setup:
            # With of bottom (a1) and top (a2) square respectively
            a1, a2 = 0.01, 0.16
            d_a = (a2-a1)/2
            
            # Height z
            l = 0.16 

            edge_points = np.array([[-a1, -a1, 0],
                                    [a1, -a1, 0],
                                    [a1, a1, 0],
                                    [-a1, a1, 0],
                                    [-a2, -a2, l],
                                    [a2, -a2, l],
                                    [a2, a2, l],
                                    [-a2, a2, l]]).T

            indeces_of_tiles = np.array([
                # [0,1,2,3], # Bottom Part
                # [4,5,6,7], # Lid
                [0,1,4,5],
                [1,2,5,6],
                [2,3,6,7],
                [3,0,7,5]])

            kwargs['edge_points'] = edge_points
            kwargs['indeces_of_tiles'] = indeces_of_tiles
            kwargs['is_boundary'] = True
            inflation_parameter = [0.03, 0.03, 0.03, 0.03]


            
        # define boundary functions
        center_position = np.array([0, 0, kwargs['edge_points'][2, -1]/2.0]) 
        super(DynamicBoundariesPolygon, self).__init__(center_position=center_position, *args, **kwargs)

        # Define range of 'cube'
        self.x_min = np.min(self.edge_points[0, :])
        self.x_max = np.max(self.edge_points[0, :])
        self.y_min = np.min(self.edge_points[1, :])
        self.y_max = np.max(self.edge_points[1, :])
        self.z_min = np.min(self.edge_points[2, :])
        self.z_max = np.max(self.edge_points[2, :])

        if indeces_of_flexibleTiles is None:
            self.indices_of_flexibleTiles = self.ind_tiles
        else:
            self.indeces_of_flexibleTiles = indeces_of_flexibleTiles

        self.is_boundary = True

        self.num_planes = 4

        if inflation_parameter is None:
            self.inflation_parameter = np.zeros(self.num_planes)
            self.time = 0
        else:
            self.inflation_parameter = inflation_parameter

        self.time = 0
        self.update(self.inflation_parameter)


        


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
        
    # @property    
    # def reference_point(self):
        # TODO change to property
        # if hasattr(self, 'reference_point'): #
            # return reference_point
        # return


    def update_pos(t, dt, xlim=None, ylim=None, inflation_parameter=None, z_value=0):

        if inflation_parameter is None:
            freq = 2*pi/5
            inflation_parameter = np.sin(t*freq)*np.ones(self.num_plane)
        
        self.update(inflation_parameter, time_new=t, dt=dt)
        self.draw_obstacle(numPoints=50, z_val=z_value) 

        
    def update(self, inflation_parameter, time_new=0, dt=None):
        self.inflation_parameter_old = self.inflation_parameter
        self.inflation_parameter = inflation_parameter

        if dt is None:
            self.time_step = (time_new-self.time)
        else:
            self.time_step = dt
        self.time = time_new
        

    def draw_obstacle(self, numPoints=20, z_val=None, inflation_parameter=None):
        # Specific to square
        if z_val is None:
            raise NotImplementedError("Implement _drawing in 3D")

        # Check z value
        if z_val>self.z_max or z_val<self.z_min:
            print("z_value out of bound")
            import pdb; pdb.set_trace()
            return
        
        self.boundary_points_local = np.zeros((self.d, numPoints))

        # Assume symmetric setup
        xy_max = self.get_flat_wall_value(z_val)

        # Iterator of the boundary point-list
        it_xobs = 0

        # For each wall 
        for it_plane in range(self.num_planes):
            if inflation_parameter is None:
                inflation_parameter = self.inflation_parameter[it_plane]
                
            if it_plane < self.num_planes-1:
                num_plane_points = int(numPoints/self.num_planes)
            else:
                num_plane_points = numPoints - int(numPoints/self.num_planes)*(self.num_planes-1)

            tangent_surf = self.edge_points[:2, (it_plane+1)%self.num_planes] \
                           - self.edge_points[:2, it_plane]

            # tangent_surf = tangent_surf / np.linalg.norm(tangent_surf)
            
            pos_xy = np.zeros((self.dim, num_plane_points))
            for ii in range(self.dim-1):
                if tangent_surf[ii]:
                    max_val = np.copysign(xy_max, tangent_surf[ii])
                    pos_xy[ii, :] = np.linspace(-max_val, max_val, num_plane_points)
                    
            pos_xy[2,:] = np.ones(num_plane_points)*z_val
            # import pdb; pdb.set_trace()
            # a=0
            
            for ii in range(num_plane_points):
                # pos_xy = (xy_max-xy_min)/num_plane_points*ii + xy_min
                self.boundary_points_local[:, it_xobs] = self.get_point_of_plane(
                    position=pos_xy[:, ii], plane_index=it_plane,
                    inflation_parameter=inflation_parameter)
                                
                it_xobs += 1
                # No rotation in absolute frame, since only relative 2D analysis
                
            if False:
                if not it_plane:
                    plt.figure()
                plt.plot(pos_xy[0, :], pos_xy[1, :], 'b.')
                plt.plot(self.boundary_points_local[0, it_xobs-num_plane_points:it_xobs],
                         self.boundary_points_local[1, it_xobs-num_plane_points:it_xobs], 'gx')
        # import pdb; pdb.set_trace();
        # a=0

    def get_reference_direction(self, position, in_global_frame=False):
        if in_global_frame:
            position = self.transform_global2relative(position)

        reference_direction = - (position-self.reference_point)
        reference_direction[2] = 0

        if in_global_frame:
            reference_direction = self.transform_global2relative_dir(reference_direction)
        return reference_direction
    

    def project_position(self, position, plane_index, in_global_frame=False):
        # TODO projection for more general direction vector
        # Specific to LASA2019-setup
        if in_global_frame:
            position = self.transform_global2relative(position)

        tangent_dir = self.edge_points[:, (plane_index+1)%self.num_planes] \
                      - self.edge_points[:, plane_index]

        projected_position = np.zeros(self.dim-1)

        for ii in range(self.dim-1):
            if tangent_dir[ii]:
                projected_position[0] = np.copysign(position[ii], tangent_dir[ii])
                break
        projected_position[1] = position[2]

        if self.num_planes>self.n_planes:
            raise ValueError("Unknown plane index")
        return projected_position

    def unproject_position(self, elevation_plane, position_init, plane_index, in_global_frame=False):
        # TODO projection for more general direction vector
        if plane_index>self.n_planes:
            raise ValueError("Unknown plane index")
        
        position = np.copy(position_init)
        
        # Specific to LASA2019-setup
        tangent_dir = self.edge_points[:, (plane_index+1)%self.num_planes] \
                      - self.edge_points[:, plane_index]

        normal_dir = np.array([tangent_dir[1] , -tangent_dir[0]])

        for ii in range(self.dim-1):
            if normal_dir[ii]: # nonzero
                position[ii] = np.copysign(elevation_plane, normal_dir[ii])
                break
        
        if in_global_frame:
            position = self.transform_global2relative(position)
        
        return position

    
    def get_flat_wall_value(self, z_value):
        # Assumption of pyramid shape (without edge)
        z_max = self.edge_points[2,-1]

        xy_at_min_z = self.edge_points[0, 1]
        xy_at_max_z = self.edge_points[0,-2]
        flat_wall = (xy_at_max_z - xy_at_min_z) /(2*z_max)*(z_value+z_max) + xy_at_min_z
        # if np.isnan(flat_wall): # TODO: remove
            # import pdb; pdb.set_trace() ## DEBUG ##
        return flat_wall
    

    def get_point_of_plane(self, position, plane_index, inflation_parameter=None):
        if inflation_parameter is None:
            inflation_parameter = self.inflation_parameter[plane_index]
        
        # Point which is in along x resp y direction on the same z level to the agent
        position_projected = self.project_position(position, plane_index=plane_index)
        
        # Specific to LASA2019-setup
        # Twice squared function of the form
        # All planes behave the same way
        z_max = self.edge_points[2,-1]
        height_value = position_projected[1]
        max_elevation_z = inflation_parameter
        
        # x = a*z^2 + b*z + c ==== b=0 due to symmetry
        c = max_elevation_z
        a = -(c/(z_max*z_max))
        max_elevation_x = a*height_value*height_value + c

        flat_wall = self.get_flat_wall_value(position[2])
        
        # elev = a*x^2 + b*x + c === b=0 due to symmetry
        c = max_elevation_z
        a = -(c/(flat_wall*flat_wall))
        elevation_from_wall = a*position_projected[0]*position_projected[0] + c

        # Convert to global frame
        elevation_plane = flat_wall-elevation_from_wall

        return self.unproject_position(elevation_plane, position, plane_index)
    

    def get_velocities(self, position, in_global_frame=False):
        # TODO: walls individually for 'smoother performance'
        position_wall = self.line_search_surface_point(position)
        position_new  = self.get_point_of_plane(position_wall, inflation_parameter=self.inflation_parameter_old)
        position_old  = self.get_point_of_plane(position_wall, inflation_parameter=self.inflation_parameter)

        linear_velocity = (position_old-position_new)/self.time_step
        
        direction_new = self.get_normal_direction(position_wall, inflation_parameter=self.inflation_parameter_old)
        direction_old = self.get_normal_direction(position_wall, inflation_parameter=self.inflation_parameter_old)

        angular_velocity = (direction_new-direction_old)/self.time_step

        import pdb; pdb.set_trace() ## DEBUG ##
        return linear_velocity, angular_velocity
    
    def get_edge_of_plane(self, plane_index, z_value, clockwise_plane_edge):
        # get_value_high -- decides about direction of value
        max_value = self.get_flat_wall_value(z_value)

        pos_2d = np.array([max_value, -max_value]) \
                 if clockwise_plane_edge \
                 else np.array([-max_value, -max_value])

        rot_matrix = np.array([[0, -1], [1, 0]])
        
        edge_point = np.zeros(self.dim)
        edge_point[:2] = np.linalg.matrix_power(rot_matrix, plane_index).dot(pos_2d)
        edge_point[2] = z_value

        # print('edge_point', edge_point)
        return edge_point
    

    def get_normal_on_plane(self, position, plane_index, delta_dist=0.0010):
        tangents_plane = np.zeros((self.dim, self.dim-1))

        axes = [1,2] if plane_index%2 else [0,2]
        
        for ii, ax in zip(range(len(axes)), axes):
            delta_vec = np.zeros(self.dim)
            delta_vec[ax] = delta_dist
            
            pos_high = self.get_point_of_plane(position+delta_vec, plane_index)
            pos_low = self.get_point_of_plane(position-delta_vec, plane_index)
            
            tangents_plane[:, ii] = (pos_high-pos_low)/(2*delta_dist)
            
        if plane_index<=1:
            normal_plane = np.cross(tangents_plane[:, 0], tangents_plane[:, 1])
        else:
            normal_plane = np.cross(tangents_plane[:, 1], tangents_plane[:, 0])
            

        mag_norm = np.linalg.norm(normal_plane)
        if mag_norm:
            normal_plane = normal_plane / mag_norm
        
        return normal_plane

    
    def get_tangent2D_and_normal_of_plane(self, position, plane_index, clockwise_plane_edge):
        # TODO: check the direction of the normal
        # normal_vector_2d = self.get_normal_on_plane_2d( position, plane_index)
        normal_vector = self.get_normal_on_plane(position, plane_index)

        tangent_2d = np.array([-normal_vector[1], normal_vector[0]])
        # if (~clockwise_plane_edge) ^ self.is_boundary:
        if (clockwise_plane_edge):
            tangent_2d = (-1)*tangent_2d

        return tangent_2d, normal_vector
        
    def get_normal_direction(self, position, surface_point=None, in_global_frame=False, normalize=False):
        position_surface = self.line_search_surface_point(position) # Automatically find plane
        plane_index = self.get_closest_plane(position)

        indeces_planes = np.array([(plane_index-1)%self.n_planes, plane_index, (plane_index+1)%self.n_planes])
        # find closest edge
        points_edges = np.zeros((self.dim, indeces_planes.shape[0]))
        tangents_edges = np.zeros((self.dim-1, points_edges.shape[1])) # 2d
        normals_planes = np.zeros(points_edges.shape)

        for ii, clockwise in zip([0,2], [True, False]):
            points_edges[:, ii] = self.get_edge_of_plane(plane_index=indeces_planes[ii], z_value=position[2], clockwise_plane_edge=clockwise)
            
            tangents_edges[:, ii], normals_planes[:, ii] = self.get_tangent2D_and_normal_of_plane(points_edges[:, ii], plane_index=indeces_planes[ii], clockwise_plane_edge=clockwise)
        
        dist_to_edges = np.linalg.norm(points_edges[:, [0,2]]-np.tile(position, (2,1)).T, axis=0)
        
        if dist_to_edges[0] < dist_to_edges[1]:
            middle_is_clockwise=False
            points_edges[:, 1] = points_edges[:, 0]
        else:
            middle_is_clockwise=True
            points_edges[:, 1] = points_edges[:, 2]
        
        # Seudo-tangent due to curvature
        point_intersect = self.line_search_surface_point(position, indeces_planes[1])
        tangents_edges[:, 1] = (point_intersect - points_edges[:, 1])[:2]

        # Evaluate that one in the middle, not at edge
        normals_planes[:, 1] = self.get_normal_on_plane(position, indeces_planes[1])

        position_temp = np.copy(position)
        if self.is_boundary:
            if np.linalg.norm(position_temp-self.reference_point)==0:
                return np.ones(self.dim)/(1.0*np.sum(self.dim)) # Return nontrivial vector
            
            Gamma = self.get_gamma(position_temp)
            position_temp = Gamma*Gamma*position_temp

        angle_to_tangent = np.zeros(points_edges.shape[1])
        for ii, clockwise in zip(range(points_edges.shape[1]), [True, middle_is_clockwise, False]):
            angle_to_tangent[ii] = self.get_angle_2d((position_temp-points_edges[:, ii])[:2], tangents_edges[:, ii], clockwise_plane_edge=clockwise)

            if np.isnan(angle_to_tangent[ii]):
                import pdb; pdb.set_trace() ## DEBUG ##
        
        angle_weights = self.get_angle_weight(angle_to_tangent)

        # import pdb; pdb.set_trace() ## DEBUG ##
        normal_vector = get_directional_weighted_sum(null_direction=position, directions=normals_planes, weights=angle_weights, total_weight=min(1/Gamma, 1), normalize_reference=True)

        if normalize and False:
            mag_normal = np.linalg.norm(normal_vector)
            if mag_normal:
                normal_vector = normal_vector / mag_normal
        

        if False:
        # if True:
            print('position temp', position_temp)
            print('angle_to_tangent', angle_to_tangent)
            print('angle_weights', angle_weights)

            ii=1
            print('tang_surf_point', tangents_edges[:,1]/np.linalg.norm(tangents_edges[:,1]))
            tang = self.get_tangent2D_and_normal_of_plane(points_edges[:, ii], plane_index=indeces_planes[ii], clockwise_plane_edge=True)[0]
            tang = tang/np.linalg.norm(tang)
            print('tang_edge_tangs', )
            
            # fig, ax = plt.subplots()
            ax = plt.gca()
            plt.plot([0], [0], 'b.')
            plt.axis('equal')

            # dist = 10
            # plt.plot(position[0], position[1], 'kx')
            # dist = np.linalg.norm(position_temp[:2])*1.2
            # plt.plot(position_temp[0], position_temp[1], 'kx')
            # plt.plot([0, -dist], [0, -dist], 'k--')
            # for ii in range(2):
                # tangents_edges[:, ii] = tangents_edges[:, ii]/np.linalg.norm(tangents_edges[:, ii])
                # point_tang = points_edges[:2, ii] - tangents_edges[:2, ii]*dist
                # plt.plot([points_edges[0, ii], point_tang[0]], [points_edges[1, ii], point_tang[1]], 'r--')
                
            # ii=1
            # ax.quiver(points_edges[0,ii], points_edges[1,ii], tang[0], tang[1], color='m')
            
            # for ii, col in zip(range(3), ['g', 'b', 'r']):
                # ax.quiver(points_edges[0,ii], points_edges[1,ii], tangents_edges[0,ii], tangents_edges[1,ii], color=col)
                # plt.plot([position_temp[0], points_edges[0,ii]], [position_temp[1], points_edges[1,ii]], 'k' )
            
            for ii, col in zip([0,2], ['g', 'r']):
                ax.quiver(points_edges[0,ii], points_edges[1,ii], normals_planes[0,ii], normals_planes[1,ii], color=col)
            ii=1
            ax.quiver(position[0], position[1], normals_planes[0,ii], normals_planes[1,ii], color='b')
            ax.quiver(position[0], position[1], normal_vector[0], normal_vector[1], color='k')
            
            plt.ion()
            plt.show()
            import pdb; pdb.set_trace() ## DEBUG ##
            
        return normal_vector

    
    def get_angle_2d(self, edge2position_vector, tangent_vector, clockwise_plane_edge):
        # Only for 2d
        if clockwise_plane_edge:
            sign_angle = np.cross(tangent_vector, edge2position_vector)
        else:
            sign_angle = np.cross(edge2position_vector, tangent_vector)

        arccos_value = edge2position_vector.T.dot(tangent_vector) / (np.linalg.norm(edge2position_vector)*np.linalg.norm(tangent_vector))
        angle = np.arccos(min(max(arccos_value, -1), 1)) # avoid numerical errors
        
        if sign_angle<0:
            angle = 2*pi - angle
        return angle

        
    def get_gamma(self, position, plane_index=None, in_global_frame=False, gamma_power=2):
        if in_global_frame:
            position = self.transform_global2relative(position)

        if np.linalg.norm((position-self.reference_point)[:2])==0:
            return sys.float_info.max

        if plane_index is None:
            position_surface = self.line_search_surface_point(position) # Automatically find plane
        else:
            position_surface = self.line_search_surface_point(position, plane_index)

        rad_local = np.linalg.norm(position_surface) # local frame
        dist_position = np.linalg.norm(position)

        import pdb; pdb.set_trace()

        if self.is_boundary:
            if dist_position:
                Gamma = (rad_local/dist_position)**gamma_power
            else:
                Gamma = sys.float_info.max
        else:
            Gamma = (dist_position/rad_local)**gamma_power
        return Gamma
    
    
    def line_search_surface_point(self, position, plane_index=None, in_global_frame=False, max_it=20):
        # TODO -- faster convergence through local evaluation + Pythagoras 
        if in_global_frame:
            position = self.transform_global2relative(position)
            direction = self.transform_global2relative_dir(direction)

        if plane_index is None:
            plane_index = self.get_closest_plane(position)
                    
        position_inside = copy.deepcopy(position)

        direction = -self.get_reference_direction(position)
        
        max_dimension = np.argmax(np.abs(direction))
        
        position_on_plane = self.get_point_of_plane(position_inside, plane_index=plane_index)
        if np.linalg.norm(position_on_plane) > np.linalg.norm(position_inside):
            value_flat_wall = self.get_flat_wall_value(position[2])
        
            position_outside = direction/np.abs(direction[max_dimension])*value_flat_wall
            
        else: # Turn around
            position_outside = position_inside
            position_inside = np.zeros(self.dim)
            max_dimension = np.argmax(np.abs(direction))

        import pdb; pdb.set_trace()

        for ii in range(max_it):
            position_middle = 0.5*(position_inside+position_outside)
            position_on_plane = self.get_point_of_plane(position_middle, plane_index=plane_index)
            
            if np.abs(position_on_plane[max_dimension]) < np.abs(position_middle[max_dimension]):
                position_outside = position_middle
            else:
                position_inside = position_middle

        position_middle = (position_inside+position_outside)*0.5
        if in_global_frame:
            position_middle = self.transform_relative2global(position_middle)
        # plt.plot(position_middle[0], position_middle[1], 'k.')

        return position_middle

    
    def get_closest_plane(self, position, in_global_frame=False):
        ''' Get the index of the closes plane. 
        First one is at low y value (iterating in positive rotation around z)

        Assumption of squared symmetry along x&y and Walls aligned with axes. '''
        
        if in_global_frame:
            position = self.transform_global2relative(position)
            
        if position[0]>position[1]:
            return 0 if position[0]<(-position[1]) else 1
        else:
            return 2 if position[0]>(-position[1]) else 3
    
    
    def flexible_boundary_local_velocity(self, position, in_global_frame=False):
        if in_global_frame:
            position = self.transform_global2relative(position)
        
    # @property
    # def 
 
    # @ 
