#!/USSR/bin/python3
'''
Container encapsulates all obstacles.
Gradient container finds the dynamic reference point through gradient descent.
'''

__author__ = "LukasHuber"
__date__ =  "2020-06-30"
__email__ =  "lukas.huber@epfl.ch"

import warnings, sys
import numpy as np
import copy

# from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import *
# from dynamic_obstacle_avoidance.obstacle_avoidance.obs_dynamic_center_3d import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import ObstacleContainer
# from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import *
# from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import get_reference_weight

# from dynamic_obstacle_avoidance.settings import DEBUG_FLAG
# from dynamic_obstacle_avoidance import settings


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
