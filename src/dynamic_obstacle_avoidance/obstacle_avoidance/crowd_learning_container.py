#!/USSR/bin/python3
'''
Container encapsulates all obstacles.
Gradient container finds the dynamic reference point through gradient descent.
'''

__author__ = "LukasHuber"
__date__ =  "2020-06-30"
__email__ =  "lukas.huber@epfl.ch"

import warnings
import sys
import copy
import numpy as np
from math import sqrt 

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import BaseContainer
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import GradientContainer

from dynamic_obstacle_avoidance.obstacle_avoidance.learning_obstacle import RegressionObstacle
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import CircularObstacle

from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import transform_polar2cartesian, transform_cartesian2polar


def findCircle(pos_array) :
    '''Function to find the circle on 
    which the given three points lie '''
    x1, y1 = pos_array[0, 0], pos_array[1, 0]
    x2, y2 = pos_array[0, 1], pos_array[1, 1]
    x3, y3 = pos_array[0, 2], pos_array[1, 2]
    
    x12 = x1 - x2 
    x13 = x1 - x3 

    y12 = y1 - y2 
    y13 = y1 - y3 

    y31 = y3 - y1 
    y21 = y2 - y1

    x31 = x3 - x1 
    x21 = x2 - x1 

    # x1^2 - x3^2 
    sx13 = pow(x1, 2) - pow(x3, 2)

    # y1^2 - y3^2 
    sy13 = pow(y1, 2) - pow(y3, 2) 

    sx21 = pow(x2, 2) - pow(x1, 2) 
    sy21 = pow(y2, 2) - pow(y1, 2) 

    f = (((sx13) * (x12) + (sy13) *
            (x12) + (sx21) * (x13) +
            (sy21) * (x13)) // (2 *
            ((y31) * (x12) - (y21) * (x13))))

    g = (((sx13) * (y12) + (sy13) * (y12) +
            (sx21) * (y13) + (sy21) * (y13)) //
            (2 * ((x31) * (y12) - (x21) * (y13))))

    c = (-pow(x1, 2) - pow(y1, 2) -
            2 * g * x1 - 2 * f * y1)

    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0 
    # where centre is (h = -g, k = -f) and 
    # radius r as r^2 = h^2 + k^2 - c 
    h = -g
    k = -f 
    sqr_of_r = h * h + k * k - c

    # r is the radius 
    r = round(sqrt(sqr_of_r), 5) 

    print("Centre = (", h, ", ", k, ")")
    print("Radius = ", r)

    # Create Obstacle
    # obs = CircularObstacle(center_position=np.array([h, k]), radius=r)
    # return obs
    return np.array([h, k]), r



class CrowdCircleContainer(GradientContainer):
    def __init__(self, obs_list=None, robot_margin=0):
        if sys.version_info>(3,0):
            super().__init__(obs_list)
        else: # Python 2
            super(GradientContainer, self).__init__(obs_list) # works for python < 3.0?!
        
        # self.num_gmm = None

        self.robot_margin=0.3
        self._dim = 2

        self.non_active_obstacles = None

    # @property
    # def dim(self):
        # return self._dim

    # @dim.setter
    # def dim(self, value):
        # self._dim = value

    def update_step(self, crowd_list, human_radius=0.3, num_crowd_close=10, dist_far=5, max_center_displacement=2):
        pos_crowd = np.zeros((self._dim, len(crowd_list) ))
        vel_crowd = np.zeros((self._dim, len(crowd_list) ))

        for ii in range(len(crowd_list)):
            pos_crowd[:, ii] = [crowd_list[ii].position.x, crowd_list[ii].position.z]
            vel_crowd[:, ii] = [crowd_list[ii].velocity.linear.x,
                                crowd_list[ii].velocity.linear.z] 
            # Rotation is neglected due to circular representation

        magnitudes = np.linalg.norm(pos_crowd, axis=0)

        # Neglect far away obstacles (to speed up calculation)
        ind_close = (magnitudes<dist_far)
        pos_crowd = pos_crowd[:, ind_close]
        vel_crowd = vel_crowd[:, ind_close]
        magnitudes = magnitudes[ind_close]

        # Sort values
        ind_sorted = np.argsort(magnitudes)
        pos_crowd = pos_crowd[:, ind_sorted]
        vel_crowd = vel_crowd[:, ind_sorted]
        magnitudes = magnitudes[ind_sorted]

        # Remove existing crowd obstacles
        it = 0
        while(it<len(self)):
            if self[it].is_boundary:
                it+=1
            else:
                del self[it]

        for ii in range(num_crowd_close):
            self.append(CircularObstacle(
                center_position=pos_crowd[:, ii], orientation=0,
                linear_velocity=vel_crowd[:, ii], angular_velocity=0, 
                radius=human_radius, margin_absolut=self.robot_margin)) # TODO: add robot margin

        # Only consider 'far' obstacles for the wall repulsion
        pos_crowd = pos_crowd[:, num_crowd_close:]
        magnitudes = magnitudes[num_crowd_close:]

        # Caluclate repulsion force of center
        exp_repulsion = 1
        fac = np.exp(-exp_repulsion*magnitudes) / np.exp(self.robot_margin) # [1, 0]
        
        center_wall = np.sum(np.tile(fac/magnitudes*(-1), (self.dim, 1)) * pos_crowd, axis=1)
        mag_center = np.linalg.norm(center_wall)
        if mag_center>max_center_displacement:
            center_wall = center_wall/mag_center*max_center_displacement

        radius_wall = np.linalg.norm(pos_crowd[:, 0] - center_wall)
        
        # print('pos', self[-1].center_position)
        # print('pos crowd', pos_crowd)

        # 3 points define the circle
        # surface_points = np.zeros((self._dim, 3))
        # self.boundary_creator_obstacles = []

        # for ii in range(surface_points.shape[1]):
            # surface_points[:, ii] = pos_crowd[:, ind_sorted[ii+num_crowd_close]]

            # self.boundary_creator_obstacles.append(CircularObstacle(
                # center_position=surface_points[:, ii], orientation=0,
                # radius=human_radius, margin_absolut=self.robot_margin)
            # )
        # center_wall, radius_wall = findCircle(surface_points)

                # Check that the circle is not artificially large

        if radius_wall < human_radius:
            raise NotImplementedError("Collision with robot")
            
        # if radius_wall > 10: # Very large
            # pass
            # raise NotImplementedError("Implement maximum radius")

        if self.contains_wall_obstacle:
            self[index_wall].update_deforming_obstacle(
                position=center_wall, orientation=0, radius=radius_wall-human_radius)
        else:
            # Create new/first wall obstacle
            self.append(CircularObstacle(
                center_position=center_wall, orientation=0, radius=radius_wall-human_radius,
                margin_absolut=self.robot_margin,
                is_boundary=True, is_deforming=True))
        # Return



class CrowdLearningContainer(BaseContainer):
    def __init__(self, obs_list=None, robot_margin=0):
        if sys.version_info>(3,0):
            super().__init__(obs_list)
        else: # Python 2
            super(BaseContainer, self).__init__(obs_list) # works for python < 3.0?!
        
        self.num_gmm = None

        self.robot_margin=robot_margin
        self.dim = 2

        self.append(RegressionObstacle(
            center_position=np.zeros(self.dim),
            is_boundary=True))

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value


    def update_step(self,  lidar_data=None, laser_data=None, cutoff_distance=5, max_displacement=2.0, angular_resolution=1000, exp_repulsion=3):
        ''' Input: lidar or obstacle data in 'obstacle frame of reference'''
        
        # Remove z-information / make_2d
        points = lidar_data[:2]

        magnitudes = np.linalg.norm(points, axis=0)
        
        # Only keep close points
        ind_close = np.logical_and(magnitudes>0, magnitudes<cutoff_distance+self.robot_margin)
        points = points[:, ind_close]
        magnitudes = magnitudes[ind_close]
        # angles_all  = np.arctan2(points[1, :], points[0, :])

        # TODO: shift to center by 'margin'

        # Shift center to in 'free space' (negative exponential)
        fac = np.exp(-exp_repulsion*magnitudes) / np.exp(self.robot_margin) # [1, 0]

        center_wall = np.sum(np.tile(fac/magnitudes*(-1), (self.dim, 1)) * points, axis=1)
        
        mag_center_wall = np.linalg.norm(center_wall)
        if mag_center_wall > max_displacement:
            center_wall = center_wall/mag_center_wall*max_displacement

        # Create wall obstacle
        self[self.index_wall].center_position = center_wall
        # Make wall orientation point towards robot (since non-continuous regression

        self[self.index_wall].orientation = np.arctan2(-center_wall[1], -center_wall[0])
        self[self.index_wall].set_surface_points(points, in_global_frame=True)
        # self[self.index_wall].reduce_angle_resolution()
        self[self.index_wall].learn_surface()
                    
    def learn_wall(self):
        pass


    def evaluate_wall(self,):
        # TODO: when evaluating old obstacles; include the motion of the robot
        pass
        
        
    def evaluation(self):
        pass



    

class LearningContainer(BaseContainer):
    def __init__(self, obs_list=None):
        # self.a = 0
        if sys.version_info>(3,0):
            super().__init__(obs_list)
        else:
            super(BaseContainer, self).__init__(obs_list) # works for python < 3.0?!

        # self.temp = 0
            
    def create_obstacles_from_data(self, data, label, cluster_eps=0.1, cluster_min_samles=10, label_free=0, label_obstacle=1, plot_raw_data=False):
        # TODO: numpy import instead?
        
        data_obs = data[:, label==label_obstacle]
        data_free = data[:, label==label_free]
        
        
        if plot_raw_data:
            # 2D
            plt.figure(figsize=(6, 6))
            plt.plot(data_free[0, :], data_free[1, :], '.', color='#57B5E5', label='No Collision')
            plt.plot(data_obs[0, :], data_obs[1, :], '.', color='#833939', label='Collision')
            plt.axis('equal')
            plt.title("Raw Data")
            plt.legend()

            plt.xlim([np.min(data[0, :]), np.max(data[0, :])])
            plt.ylim([np.min(data[1, :]), np.max(data[1, :])])

        # TODO: try OPTICS?  & compare
        clusters = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samles).fit(data_obs.T)

        cluster_labels, obs_index = np.unique(clusters.labels_, return_index=True)
        # TODO: can obs_index be used?

        n_obstacles = np.sum(cluster_labels>=0)
        
        obs_points = [] #
        
        for oo in range(n_obstacles):
            ind_clusters = (clusters.labels_==oo)
            obs_points.append(data_obs[:, ind_clusters])

            mean_position = np.mean(obs_points[-1], axis=1)
            # TODO: make sure mean_position is within obstacle...
            
            self._obstacle_list.append(LearningObstacle(center_position=mean_position))

            
            data_non_obs_temp = np.hstack((data_obs[:, ~ind_clusters], data_free))
            self._obstacle_list[oo].learn_obstacles_from_data(data_obs=obs_points[oo], data_free=data_non_obs_temp)

    def load_obstacles_from_file(self, file_name):
        pass
