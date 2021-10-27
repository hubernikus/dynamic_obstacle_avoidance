# Author Lukas Huber
# Mail lukas.huber@epfl.ch
# Created 2021-06-22
# License: BSD (c) 2021

import time
import numpy as np
import copy
from math import pi
import warnings, sys

import matplotlib.pyplot as plt

from vartools.angle_math import *

from dynamic_obstacle_avoidance.utils import *
from dynamic_obstacle_avoidance.avoidance.obs_common_section import (
    Intersection_matrix,
)
from dynamic_obstacle_avoidance.avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.avoidance.obs_dynamic_center_3d import *

from dynamic_obstacle_avoidance.containers import BaseContainer


class LearningContainer(BaseContainer):
    def __init__(self, obs_list=None):
        if sys.version_info > (3, 0):
            super().__init__(obs_list)
        else:  # Python 2
            super(BaseContainer, self).__init__(obs_list)  # works for python < 3.0?!

    def create_obstacles_from_data(
        self,
        data,
        label,
        cluster_eps=0.1,
        cluster_min_samles=10,
        label_free=0,
        label_obstacle=1,
        plot_raw_data=False,
    ):
        # TODO: numpy import instead?

        data_obs = data[:, label == label_obstacle]
        data_free = data[:, label == label_free]

        if plot_raw_data:
            # 2D
            plt.figure(figsize=(6, 6))
            plt.plot(
                data_free[0, :],
                data_free[1, :],
                ".",
                color="#57B5E5",
                label="No Collision",
            )
            plt.plot(
                data_obs[0, :],
                data_obs[1, :],
                ".",
                color="#833939",
                label="Collision",
            )
            plt.axis("equal")
            plt.title("Raw Data")
            plt.legend()

            plt.xlim([np.min(data[0, :]), np.max(data[0, :])])
            plt.ylim([np.min(data[1, :]), np.max(data[1, :])])

        # TODO: try OPTICS?  & compare
        clusters = DBSCAN(eps=cluster_eps, min_samples=cluster_min_samles).fit(
            data_obs.T
        )

        cluster_labels, obs_index = np.unique(clusters.labels_, return_index=True)
        # TODO: can obs_index be used?

        n_obstacles = np.sum(cluster_labels >= 0)

        obs_points = []  #

        for oo in range(n_obstacles):
            ind_clusters = clusters.labels_ == oo
            obs_points.append(data_obs[:, ind_clusters])

            mean_position = np.mean(obs_points[-1], axis=1)
            # TODO: make sure mean_position is within obstacle...

            self._obstacle_list.append(LearningObstacle(center_position=mean_position))

            data_non_obs_temp = np.hstack((data_obs[:, ~ind_clusters], data_free))
            self._obstacle_list[oo].learn_obstacles_from_data(
                data_obs=obs_points[oo], data_free=data_non_obs_temp
            )

    def load_obstacles_from_file(self, file_name):
        pass
