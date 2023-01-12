#!/USSR/bin/python3
"""
Container encapsulates all obstacles.
Gradient container finds the dynamic reference point through gradient descent.
"""

__author__ = "LukasHuber"
__date__ = "2020-06-30"
__email__ = "lukas.huber@epfl.ch"

import warnings
import sys
import copy
import numpy as np
from math import sqrt

from dynamic_obstacle_avoidance.obstacles import CircularObstacle

from dynamic_obstacle_avoidance.containers import BaseContainer, GradientContainer

from vartools.angle_math import (
    transform_polar2cartesian,
    transform_cartesian2polar,
)


def findCircle(pos_array):
    # NOT USED... REMOVE! // REPLACE
    """Function to find the circle on
    which the given three points lie"""
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

    f = ((sx13) * (x12) + (sy13) * (x12) + (sx21) * (x13) + (sy21) * (x13)) // (
        2 * ((y31) * (x12) - (y21) * (x13))
    )

    g = ((sx13) * (y12) + (sy13) * (y12) + (sx21) * (y13) + (sy21) * (y13)) // (
        2 * ((x31) * (y12) - (x21) * (y13))
    )

    c = -pow(x1, 2) - pow(y1, 2) - 2 * g * x1 - 2 * f * y1

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
        if sys.version_info > (3, 0):
            super().__init__(obs_list)
        else:  # Python 2
            super(CrowdCircleContainer, self).__init__(
                obs_list
            )  # works for python < 3.0?!

        # self.num_gmm = None

        self.robot_margin = robot_margin
        self._dim = 2

        self.non_active_obstacles = None

    # @property
    # def dim(self):
    # return self._dim

    # @dim.setter
    # def dim(self, value):
    # self._dim = value

    def update_step(
        self,
        crowd_list,
        human_radius=0.35,
        num_crowd_close=10,
        dist_far=10,
        max_center_displacement=2,
        agent_position=None,
        automatic_outer_boundary=True,
        lidar_input=None,
        is_simulation=True,
        FLAG_DETECTOR_STATIC=False,
    ):
        """Update the obstacle list based on the crowd-input."""

        # Remove existing crowd obstacles
        it = 0
        while it < len(self):
            if hasattr(self[it], "is_human") and self[it].is_human:
                del self[it]
            else:
                it += 1

        # import pdb; pdb.set_trace()

        # Check if there are obstacles in crowd
        if len(crowd_list) == 0 and automatic_outer_boundary:
            self.delete_boundary()
            return

        pos_crowd = np.zeros((self._dim, len(crowd_list)))
        vel_crowd = np.zeros((self._dim, len(crowd_list)))

        if is_simulation:
            for ii in range(len(crowd_list)):
                pos_crowd[:, ii] = [
                    crowd_list[ii].position.z,
                    -crowd_list[ii].position.x,
                ]
                vel_crowd[:, ii] = [
                    crowd_list[ii].velocity.linear.z,
                    -crowd_list[ii].velocity.linear.x,
                ]
                # Check for simulation-jumping
                vel_max = 5
                for ind_it in range(2):
                    if abs(vel_crowd[ind_it, ii]) > vel_max:
                        vel_crowd[ind_it, ii] = 0
                        warnings.warn(
                            "Simulation-jump detected. Agent velocity set to zero"
                        )

        else:
            # Different message types on real robot
            # if FLAG_DETECTOR_STATIC:
            # import pdb; pdb.set_trace()
            # else:
            for ii in range(len(crowd_list)):
                pos_crowd[:, ii] = [
                    crowd_list[ii].pose.pose.position.x,
                    crowd_list[ii].pose.pose.position.y,
                ]
                # Don't get velocity from tracker at current stage
                # vel_crowd[:, ii] = [crowd_list[ii].twist.twist.linear.x, crowd_list[ii].twist.twist.linear.y]

        # Rotation is neglected due to circular representation
        # Relative distance to the agent of each obstacle
        if agent_position is None:
            magnitudes = np.linalg.norm(pos_crowd, axis=0)
        else:
            magnitudes = np.linalg.norm(
                pos_crowd - np.tile(agent_position, (pos_crowd.shape[1], 1)).T,
                axis=0,
            )

        # Neglect far away obstacles (to speed up calculation)
        ind_close = magnitudes < dist_far
        if np.sum(ind_close) < num_crowd_close:
            if automatic_outer_boundary:
                # Remove the boundary, very simple environment close obstacles
                self.delete_boundary()

            num_crowd_close = np.sum(ind_close)

            # No close obstacle
            if num_crowd_close == 0:
                return

        pos_crowd = pos_crowd[:, ind_close]
        vel_crowd = vel_crowd[:, ind_close]
        magnitudes = magnitudes[ind_close]

        # Sort values
        ind_sorted = np.argsort(magnitudes)
        pos_crowd = pos_crowd[:, ind_sorted]
        vel_crowd = vel_crowd[:, ind_sorted]
        magnitudes = magnitudes[ind_sorted]

        for ii in range(num_crowd_close):
            human_obs = CircularObstacle(
                center_position=pos_crowd[:, ii],
                orientation=0,
                linear_velocity=vel_crowd[:, ii],
                angular_velocity=0,
                tail_effect=False,
                radius=human_radius,
                margin_absolut=self.robot_margin,
            )

            # TODO include in CircularObstacle / crowd-obstacle
            human_obs.is_human = True

            human_obs.sigma = 7  # exponential weight for veloctiy reduction
            human_obs.reactivity = 3  # veloctiy reduction
            human_obs.repulsion_coeff = 1.5

            self.append(human_obs)  # TODO: add robot margin

        if (
            num_crowd_close == np.sum(ind_close) or not automatic_outer_boundary
        ) and lidar_input is None:
            # No 'artificial wall' is created
            return

        if lidar_input is None:
            print("self. margin", self.robot_margin)
            # Only consider 'far' obstacles for the wall repulsion
            pos_crowd = pos_crowd[:, num_crowd_close:]
            magnitudes = magnitudes[num_crowd_close:]
        else:
            pos_crowd = lidar_input
            magnitudes = np.linalg.norm(
                pos_crowd - np.tile(agent_position, (pos_crowd.shape[1], 1)).T,
                axis=0,
            )

        # Caluclate repulsion force of center
        exp_repulsion = 1.5
        fac = np.exp(-exp_repulsion * magnitudes) / np.exp(self.robot_margin)  # [1, 0]

        rel_center_wall = np.sum(
            np.tile(fac / magnitudes * (-1), (self.dim, 1))
            * (pos_crowd - np.tile(agent_position, (pos_crowd.shape[1], 1)).T),
            axis=1,
        )

        mag_center = np.linalg.norm(rel_center_wall)
        if mag_center > max_center_displacement:
            rel_center_wall = rel_center_wall / mag_center * max_center_displacement
        center_wall = agent_position + rel_center_wall

        if lidar_input is None:
            radius_wall = np.linalg.norm(pos_crowd[:, 0] - center_wall)

            if radius_wall < human_radius:
                raise NotImplementedError("Collision with robot")
            radius_wall = radius_wall - human_radius
        else:
            radius_wall = np.min(
                np.linalg.norm(
                    pos_crowd - np.tile(center_wall, (pos_crowd.shape[1], 1)).T,
                    axis=0,
                )
            )

        if self.contains_wall_obstacle:
            self[self.index_wall].update_deforming_obstacle(
                position=center_wall, orientation=0, radius_new=radius_wall
            )
        else:
            # Create new/first wall obstacle
            self.append(
                CircularObstacle(
                    center_position=center_wall,
                    orientation=0,
                    radius=radius_wall,
                    margin_absolut=self.robot_margin,
                    is_boundary=True,
                    is_deforming=True,
                    tail_effect=False,
                )
            )
        # Return


class CrowdLearningContainer(BaseContainer):
    def __init__(self, obs_list=None, robot_margin=0):
        if sys.version_info > (3, 0):
            super().__init__(obs_list)
        else:  # Python 2
            super(BaseContainer, self).__init__(obs_list)  # works for python < 3.0?!

        self.num_gmm = None

        self.robot_margin = robot_margin
        self.dim = 2

        self.append(
            RegressionObstacle(center_position=np.zeros(self.dim), is_boundary=True)
        )

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        self._dim = value

    def udpate_step_real(
        self,
        lidar_data=None,
        laser_data=None,
        cutoff_distance=5,
        max_displacement=2.0,
        angular_resolution=1000,
        exp_repulsion=3,
    ):

        import pdb

        pdb.set_trace()

    def update_step(
        self,
        lidar_data=None,
        laser_data=None,
        cutoff_distance=5,
        max_displacement=2.0,
        angular_resolution=1000,
        exp_repulsion=3,
    ):

        """Input: lidar or obstacle data in 'obstacle frame of reference"""

        # Remove z-information / make_2d
        points = lidar_data[:2]

        magnitudes = np.linalg.norm(points, axis=0)

        # Only keep close points
        ind_close = np.logical_and(
            magnitudes > 0, magnitudes < cutoff_distance + self.robot_margin
        )
        points = points[:, ind_close]
        magnitudes = magnitudes[ind_close]
        # angles_all  = np.arctan2(points[1, :], points[0, :])

        # TODO: shift to center by 'margin'

        # Shift center to in 'free space' (negative exponential)
        fac = np.exp(-exp_repulsion * magnitudes) / np.exp(self.robot_margin)  # [1, 0]

        center_wall = np.sum(
            np.tile(fac / magnitudes * (-1), (self.dim, 1)) * points, axis=1
        )

        mag_center_wall = np.linalg.norm(center_wall)
        if mag_center_wall > max_displacement:
            center_wall = center_wall / mag_center_wall * max_displacement

        # Create wall obstacle
        self[self.index_wall].center_position = center_wall
        # Make wall orientation point towards robot (since non-continuous regression

        self[self.index_wall].orientation = np.arctan2(-center_wall[1], -center_wall[0])
        self[self.index_wall].set_surface_points(points, in_global_frame=True)
        # self[self.index_wall].reduce_angle_resolution()
        self[self.index_wall].learn_surface()

    def learn_wall(self):
        pass

    def evaluate_wall(
        self,
    ):
        # TODO: when evaluating old obstacles; include the motion of the robot
        pass

    def evaluation(self):
        pass


class LearningContainer(BaseContainer):
    def __init__(self, obs_list=None):
        # self.a = 0
        if sys.version_info > (3, 0):
            super().__init__(obs_list)
        else:
            super(BaseContainer, self).__init__(obs_list)  # works for python < 3.0?!

        # self.temp = 0

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
