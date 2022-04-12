#!/USSR/bin/python3
"""
@date 2019-10-15
@author Lukas Huber 
@email lukas.huber@epfl.ch
"""

import time
import numpy as np
from math import sin, cos, pi, ceil
import warnings, sys

import numpy.linalg as LA

from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import *

from dynamic_obstacle_avoidance.obstacle_avoidance.state import *
from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import (
    angle_modulo,
    angle_difference_directional_2pi,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_dynamic_center_3d import *

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import Obstacle

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap

# import quaternion

from sklearn import svm

visualize_debug = False


class LearningObstacle(Obstacle):
    """Obstacle is learned through any function.
    Note: compared to other obstacles, all the description are in the 'global' frame.
    """

    # self.ellipse_type = dynamic_obstacle_avoidance.obstacle_avoidance.obstacle.Ellipse
    def __init__(self, *args, **kwargs):

        if sys.version_info > (3, 0):
            super().__init__(*args, **kwargs)
        else:
            super(Ellipse, self).__init__(*args, **kwargs)  # works for python < 3.0?!

        self._cassifier_obstacle = None

        self._max_dist = None

        self._outer_dist_fac = 2  # Descent of added gamma

    def learn_obstacles_from_data(
        self,
        data_obs,
        data_free,
        gamma_svm=10,
        C_svm=20.0,
        only_close_points=True,
    ):
        # TODO: make gmmma dependent on obstacle 'size' or outer_dist_fac to ensure continuous decent.
        dist = np.linalg.norm(
            data_obs - np.tile(self.global_reference_point, (data_obs.shape[1], 1)).T,
            axis=0,
        )
        self._max_dist = np.max(dist)

        self._outer_ref_dist = self._max_dist * self._outer_dist_fac

        if only_close_points:  # For faster calculation
            free_dist = np.linalg.norm(
                data_free
                - np.tile(self.global_reference_point, (data_free.shape[1], 1)).T,
                axis=0,
            )
            ind_close = free_dist < self._outer_ref_dist
            data_free = data_free[:, ind_close]

        # self.gamma_svm = gamma_svm
        self.gamma_svm = (self._outer_dist_fac - self._max_dist) * 2
        print("gamma svm", self.gamma_svm)

        data = np.hstack((data_free, data_obs))
        label = np.hstack((np.zeros(data_free.shape[1]), np.ones(data_obs.shape[1])))
        self._classifier = svm.SVC(kernel="rbf", gamma=self.gamma_svm, C=C_svm).fit(
            data.T, label
        )

        print("Number of support vectors / data points")
        print(
            "Free space: ({} / {}) --- Obstacle ({} / {})".format(
                self._classifier.n_support_[0],
                data_free.shape[1],
                self._classifier.n_support_[1],
                data_obs.shape[1],
            )
        )

    def draw_obstacle(self, fig=None, ax=None, show_contour=True, gamma_value=False):
        xx, yy = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))

        if ax is None or fig is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(7, 6)

        if gamma_value:
            predict_score = self.get_gamma(np.c_[xx.ravel(), yy.ravel()].T)
            predict_score = (
                predict_score - 1
            )  # Subtract 1 to have differentiation boundary at 1
            plt.title("$\Gamma$-Score")
        else:
            predict_score = self._classifier.decision_function(
                np.c_[xx.ravel(), yy.ravel()]
            )
            plt.title("SVM Score")
        predict_score = predict_score.reshape(xx.shape)
        # import pdb; pdb.set_trace() ## DEBUG ##

        levels = np.array([0])

        cs0 = ax.contour(
            xx,
            yy,
            predict_score,
            levels,
            origin="lower",
            colors="k",
            linewidths=2,
        )
        if show_contour:

            cs = ax.contourf(
                xx,
                yy,
                predict_score,
                np.arange(-16, 16, 2),
                cmap=plt.cm.coolwarm,
                extend="both",
                alpha=0.8,
            )

            cbar = fig.colorbar(cs)
            cbar.add_lines(cs0)

        else:
            cmap = colors.ListedColormap(["#000000", "#A86161"])
            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:, -1] = np.linspace(0.05, 0.7, cmap.N)
            my_cmap = ListedColormap(my_cmap)
            # bounds=[0,5,10]
            bounds = [-1, 0, 1]
            norm = colors.BoundaryNorm(bounds, my_cmap.N)
            # alphas = np.ones((2,2))
            # alphas = np.array([0., 0., 1., 1.])
            alphas = 0.5

            cs = ax.contourf(
                xx, yy, predict_score, origin="lower", cmap=my_cmap, norm=norm
            )

            reference_point = self.get_reference_point(in_global_frame=True)
            ax.plot(
                reference_point[0],
                reference_point[1],
                "k+",
                linewidth=18,
                markeredgewidth=4,
                markersize=13,
            )

    # def load_obstacles_from_file(self, file_name):
    # raise NotImplementedError()

    def get_gamma(self, position, in_global_frame=True):
        """Gamma value is learned for each obstacle individually"""
        if not in_global_frame:
            position = self.transform_relative2global(position)

        pos_shape = position.shape
        position = position.reshape(self.dim, -1)

        dist = np.linalg.norm(
            position - np.tile(self.global_reference_point, (position.shape[1], 1)).T,
            axis=0,
        )
        ind_noninf = self._outer_ref_dist > dist

        score = np.zeros(position.shape[1])

        if np.sum(ind_noninf):  # At least one element
            score[ind_noninf] = self._classifier.decision_function(
                np.c_[position[0, ind_noninf], position[1, ind_noninf]]
            )

        dist = np.clip(dist, self._max_dist, self._outer_ref_dist)
        distance_score = (self._outer_ref_dist - self._max_dist) / (
            self._outer_ref_dist - dist[ind_noninf]
        )

        max_float = sys.float_info.max
        max_float = 1e12
        gamma = np.zeros(dist.shape)
        gamma[ind_noninf] = (-score[ind_noninf] + 1) * distance_score
        gamma[~ind_noninf] = max_float

        if len(pos_shape) == 1:
            gamma = gamma[0]
        return gamma

    def get_normal_direction(
        self, position, in_global_frame=True, normalize=True, delta_dist=1.0e-5
    ):
        """Numerical differentiation to of Gamma to get normal direction."""
        if not in_global_frame:
            position = self.transform_relative2global(position)

        pos_shape = position.shape
        positions = position.reshape(self.dim, -1)

        delta_dist = self.gamma_svm * delta_dist

        normals = np.zeros((positions.shape))

        for dd in range(self.dimension):
            pos_low, pos_high = np.copy(positions), np.copy(positions)
            pos_high[dd, :] = pos_high[dd, :] + delta_dist
            pos_low[dd, :] = pos_low[dd, :] - delta_dist

            normals[dd, :] = (
                (self.get_gamma(pos_high) - self.get_gamma(pos_low)) / 2 * delta_dist
            )

        if normalize:
            mag_normals = np.linalg.norm(normals, axis=0)
            nonzero_ind = mag_normals > 0

            if any(nonzero_ind):
                normals[:, nonzero_ind] = (
                    normals[:, nonzero_ind] / mag_normals[nonzero_ind]
                )

        if not in_global_frame:
            normals = self.transform_relative2global_dir(normals)

        if len(pos_shape) == 1:  # Same input as ouput format
            normals = normals[:, 0]

        # return (-1)*normals # due to gradent definition
        return normals
