"""
@author lukashuber
@date 2019-02-03
"""

import numpy as np
from numpy import linalg as LA

from math import ceil, sin, cos, sqrt
import matplotlib.pyplot as plt  # for debugging

import warnings

# from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import Ellipse
# from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import Polygon, Cuboid

# TODO: include research from graph theory & faster computation


class DistanceMatrix:
    """Symmetric matrix storage. Only stores one half of the values, as a 1D"""

    def __init__(self, n_obs):
        self._dim = n_obs
        # self._value_list = [None for ii in range(int((n_obs-1)*n_obs/2))]
        self._value_list = (-1) * np.ones(int((n_obs - 1) * n_obs / 2))

    @property
    def num_obstacles(self):
        return self._dim

    def __repr__(self):
        matr = np.zeros((self._dim, self._dim))
        for ii in range(self._dim):
            for jj in range(ii + 1, self._dim):
                matr[ii, jj] = matr[jj, ii] = self[ii, jj]

        return str(matr)

    def __str__(self):
        return self.__repr__()

    def __setitem__(self, key, value):
        if len(key) == 2:
            ind = self.get_index(key[0], key[1])
            self._value_list[ind] = value
        else:
            raise ValueError("Not two indexes given.")

    def __getitem__(self, key):
        if len(key) == 2:
            ind = self.get_index(key[0], key[1])
            return self._value_list[ind]
        else:
            raise ValueError("Not two indexes given.")

    def get_matrix(self):
        """Get matrix as numpy-array."""
        matr = np.zeros((self._dim, self._dim))
        for ix in range(self._dim):
            for iy in range(ix + 1, self._dim):
                matr[ix, iy] = matr[iy, ix] = self[ix, iy]
        return matr

    def get_index(self, row, col):
        """Returns the corresponding list index [ind] from matrix index [row, col]"""
        if row > np.abs(self._dim):
            raise RuntimeError("Fist object index out of bound.")
            # row = 0

        if col > np.abs(self._dim):
            raise RuntimeError("Second object index out of bound.")
            # col = 1

        if row < 0:
            row = self._dim + 1 - row

        if col < 0:
            row = self._dim + 1 - col

        if row == col:
            raise RuntimeError("Self collision observation meaningless.")
            # row, col = 1, 0

        # Symmetric matrix - reverse  indices
        if col > row:
            col, row = row, col

        return int(
            int((row - col - 1) + col * ((self._dim - 1) + self._dim - (col)) * 0.5)
        )


class IntersectionMatrix(DistanceMatrix):
    """
    Matrix uses less space this way this is useful with many obstacles! e.g. dense crowds
    Symmetric matrix with zero as diagonal values

    Stores one common point of intersecting obstacles
    """

    # TODO: use scipy sparse matrices to replace this partially!!!

    def __init__(self, n_obs, dim=2):
        self._value_list = [None for ii in range(int((n_obs - 1) * n_obs / 2))]
        self._dim = n_obs

    def __repr__(self):
        return str(self.get_bool_matrix())

    def set(self, row, col, value):
        self[row, col] = value

    def get(self, row, col):
        return self[row, col]

    def is_intersecting(self, row, col):
        return bool(not (self[row, col] is None))

    def get_intersection_matrix(self):
        # Maybe not necessary function
        space_dim = 2
        # matr = np.zeros((self._dim, self._dim), dtype=bool)
        # for col in range(self._dim):
        # for row in range(col+1, self._dim):
        # matr[col, row] = matr[row, col] = not (self[row, col] is None)

        matr = np.zeros((space_dim, self._dim, self._dim))
        for col in range(self._dim):
            for row in range(col + 1, self._dim):
                if col == row:
                    continue
                val = self.get(row, col)
                if not val is None and not isinstance(self.get(row, col), bool):
                    matr[:, col, row] = matr[:, row, col] = self[row, col]
        return matr

    def get_bool_triangle_matrix(self):
        raise NotImplementedError(
            "Function was removed. Use 'get_bool_matrix' instead."
        )

        # intersection_exists_matrix = np.zeros((self._dim+1,self._dim+1), dtype=bool)
        # for col in range(self._dim+1):
        # for row in range(col+1, self._dim+1):

        # intersection_exists_matrix = np.zeros((self._dim, self._dim), dtype=bool)
        # for col in range(self._dim):
        # for row in range(col+1, self._dim):
        # if isinstance(self.get(row,col), bool) and (self.get(row,col) == False):
        # continue
        # intersection_exists_matrix[row, col] = True

        # return intersection_exists_matrix

    def get_bool_matrix(self):
        bool_matrix = np.zeros((self._dim, self._dim), dtype=bool)
        for ii in range(self._dim):
            for jj in range(ii + 1, self._dim):
                bool_matrix[ii, jj] = bool_matrix[jj, ii] = self.is_intersecting(ii, jj)

        return bool_matrix


class Intersection_matrix(IntersectionMatrix):
    pass


def obs_common_section(obs):
    # OBS_COMMON_SECTION finds common section of two ore more obstacles
    # at the moment only solution in two d is implemented

    # TODO: REMOVE ' depreciated
    warnings.warn("This function depreciated and will be removed")

    N_obs = len(obs)
    # No intersection region
    if N_obs <= 1:
        return []

    # Intersction surface
    intersection_obs = []
    intersection_sf = []
    intersection_sf_temp = []
    it_intersect = -1

    # Ext for more dimensions
    dim = d = len(obs[0].center_position)

    N_points = 30  # Choose number of points each iteration
    Gamma_steps = 5  # Increases computational cost

    rotMat = np.zeros((dim, dim, N_obs))

    for it_obs in range(N_obs):
        rotMat[:, :, it_obs] = np.array((obs[it_obs].rotMatrix))
        obs[it_obs].draw_obstacle()
        obs[it_obs].cent_dyn = np.copy(obs[it_obs].center_position)  # set default value

    for it_obs1 in range(N_obs):
        intersection_with_obs1 = False
        # Check if current obstacle 'it_obs1' has already an intersection with another
        # obstacle
        memberFound = False
        for ii in range(len(intersection_obs)):
            if it_obs1 in intersection_obs[ii]:
                memberFound = True
                continue

        for it_obs2 in range(it_obs1 + 1, N_obs):
            # Check if obstacle has already an intersection with another
            # obstacle
            memberFound = False
            for ii in range(len(intersection_obs)):
                if it_obs2 in intersection_obs[ii]:
                    memberFound = True
                    continue

            if memberFound:
                continue

            if intersection_with_obs1:  # Modify intersecition part
                obsCloseBy = False

                if True:
                    N_inter = intersection_sf[it_intersect].shape[
                        1
                    ]  # Number of intersection points

                    ## R = compute_R(d,obs[it_obs2].th_r)
                    Gamma_temp = (
                        rotMat[:, :, it_obs2].T.dot(
                            intersection_sf[it_intersect]
                            - np.tile(obs[it_obs2].center_position, (N_inter, 1)).T
                        )
                        / np.tile(obs[it_obs2].a, (N_inter, 1)).T
                    ) ** (2 * np.tile(obs[it_obs2].p, (N_inter, 1)).T)
                    Gamma = np.sum(1 / obs[it_obs2].sf * Gamma_temp, axis=0)

                    ind = Gamma < 1

                    if sum(ind):
                        intersection_sf[it_intersect] = intersection_sf[it_intersect][
                            :, ind
                        ]
                        intersection_obs[it_intersect] = intersection_obs[
                            it_intersect
                        ] + [it_obs2]
            else:
                if True:

                    # get all points of obs2 in obs1
                    Gamma = obs[it_obs1].get_gamma(
                        obs[it_obs2].x_obs_sf, in_global_frame=True
                    )
                    intersection_sf_temp = np.array(obs[it_obs2].x_obs_sf)[:, Gamma < 1]

                    # Get all poinst of obs1 in obs2
                    Gamma = obs[it_obs2].get_gamma(
                        obs[it_obs1].x_obs_sf, in_global_frame=True
                    )
                    intersection_sf_temp = np.hstack(
                        (
                            intersection_sf_temp,
                            np.array(obs[it_obs1].x_obs_sf)[:, Gamma < 1],
                        )
                    )

                    if intersection_sf_temp.shape[1] > 0:
                        it_intersect = it_intersect + 1
                        intersection_with_obs1 = True
                        intersection_sf.append(intersection_sf_temp)
                        intersection_obs.append([it_obs1, it_obs2])

                        # Increase resolution by sampling points within
                        for kk in range(2):
                            if kk == 0:
                                it_obs1_ = it_obs1
                                it_obs2_ = it_obs2

                            elif kk == 1:  # Do it both ways
                                it_obs1_ = it_obs2
                                it_obs2_ = it_obs1

                            for ii in range(1, Gamma_steps):
                                N_points_interior = ceil(N_points / Gamma_steps * ii)

                                x_obs_sf_interior = obs[
                                    it_obs1_
                                ].get_scaled_boundary_points(1.0 * ii / Gamma_steps)
                                # resolution = x_obs_sf_interior.shape[1] # number of points

                                # Get Gamma value
                                # Gamma = np.sum( (1/obs[it_obs2_].sf *  rotMat[:,:,it_obs2_].T.dot(x_obs_sf_interior-np.tile(obs[it_obs2_].center_position,(resolution,1)).T ) / np.tile(obs[it_obs2_].a, (resolution,1)).T ) ** (2*np.tile(obs[it_obs2_].p, (resolution,1)).T), axis=0)
                                Gamma = obs[it_obs2_].get_gamma(
                                    x_obs_sf_interior, in_global_frame=True
                                )
                                intersection_sf[it_intersect] = np.hstack(
                                    (
                                        intersection_sf[it_intersect],
                                        x_obs_sf_interior[:, Gamma < 1],
                                    )
                                )

                            # Check center point
                            # if 1 > sum( (1/obs[it_obs2_].sf*rotMat[:,:,it_obs2_].T.dot( np.array(obs[it_obs1_].center_position) - np.array(obs[it_obs2_].center_position) )/ np.array(obs[it_obs2_].a) ) ** (2*np.array(obs[it_obs2_].p))):
                            if 1 > obs[it_obs2_].get_gamma(
                                obs[it_obs1_].center_position,
                                in_global_frame=True,
                            ):
                                intersection_sf[it_intersect] = np.hstack(
                                    [
                                        intersection_sf[it_intersect],
                                        np.tile(
                                            obs[it_obs1_].center_position,
                                            (1, 1),
                                        ).T,
                                    ]
                                )

    # if intersection_with_obs1 continue
    if len(intersection_sf) == 0:
        return []

    # plt.plot(intersection_sf[0][0,:], intersection_sf[0][1,:], 'r.')

    for ii in range(len(intersection_obs)):
        intersection_sf[ii] = np.unique(intersection_sf[ii], axis=1)

        # Get numerical mean
        x_center_dyn = np.mean(intersection_sf[ii], axis=1)

        for it_obs in intersection_obs[ii]:
            obs[it_obs].global_reference_point = x_center_dyn

        # TODO - replace atan2 for speed
    #     [~, ind] = sort( atan2(intersec_sf_cent(2,:), intersec_sf_cent(1,:)))

    #     intersection_sf = intersection_sf(:, ind)
    #     intersection_sf = [intersection_sf, intersection_sf(:,1)]

    #     intersection_obs = [1:size(obs,2)]
    return intersection_obs


def obs_common_section_hirarchy(*args, **kwargs):
    # TODO: depreciated -- remove
    raise Exception("Outdated: use 'get_intersections_obstacles' instead")
    # return get_intersections_obstacles(*args, representation_type="hirarchy", **kwargs)


def get_intersections_obstacles(
    obs,
    hirarchy=True,
    get_intersection_matrix=False,
    N_points=30,
    Gamma_steps=5,
    representation_type="single_point",
):
    """
    OBS_COMMON_SECTION finds common section of two ore more obstacles

    Currently implemented solution is for the 2-dimensional case
    """
    # Depreciated? Remove?
    warnings.warn("Depreciated --- Remove!!!")

    raise Exception("Outdated: use 'get_intersections_obstacles' instead")

    num_obstacles = len(obs)

    # No intersection region
    if num_obstacles <= 1:
        return np.zeros((num_obstacles, num_obstacles))

    # Intersction surface
    intersection_obs = []
    intersection_sf = []
    intersection_sf_temp = []
    it_intersect = -1

    # Exit for more dimensions
    dim = len(obs[0].center_position)
    d = dim  # TODO remove!

    # Find Boundaries
    # ind_wall = obs.ind_wall
    # ind_wall = -1
    # for o in range(num_obstacles):
    # if obs[o].is_boundary:
    # ind_wall = o
    # break

    # Choose number of points each iteration
    if isinstance(obs, list):
        Intersections = Intersection_matrix(num_obstacles)
        warnings.warn("We advice to use the <<Obstacle Container>> to store obstacles.")
    else:
        Intersections = obs.get_distance() == 0
        # raise Warning()

    R_max = np.zeros((num_obstacles))  # Maximum radius for ellipsoid

    for it_obs in range(num_obstacles):
        obs[it_obs].draw_obstacle()
        R_max[it_obs] = obs[it_obs].get_reference_length()

    for it_obs1 in range(num_obstacles):
        for it_obs2 in range(it_obs1 + 1, num_obstacles):

            if obs.get_distance(it_obs1, it_obs2) or R_max[it_obs1] + R_max[
                it_obs2
            ] < LA.norm(
                np.array(obs[it_obs1].center_position)
                - np.array(obs[it_obs2].center_position)
            ):
                continue  # NO intersection possible, to far away

            # get all points of obs2 in obs1
            # N_points = len(obs[it_obs1].x_obs_sf)

            Gamma = obs[it_obs1].get_gamma(obs[it_obs2].x_obs_sf, in_global_frame=True)
            intersection_points = np.array(obs[it_obs2].x_obs_sf)[:, Gamma < 1]

            Gamma = obs[it_obs2].get_gamma(obs[it_obs1].x_obs_sf, in_global_frame=True)
            intersection_points = np.hstack(
                (intersection_points, obs[it_obs1].x_obs_sf[:, Gamma < 1])
            )
            # if intersection_sf_temp.shape[1] > 0:
            if intersection_points.shape[1] > 0:
                # Increase resolution by sampling points within obstacle, too
                # obstaacles of 2 in 1
                for kk in range(2):
                    if kk == 0:
                        it_obs1_ = it_obs1
                        it_obs2_ = it_obs2
                    elif kk == 1:  # Turn around obstacles
                        it_obs1_ = it_obs2
                        it_obs2_ = it_obs1

                    if obs[it_obs1_].is_boundary:
                        continue

                    for ii in range(1, Gamma_steps):
                        x_obs_sf_interior = obs[it_obs1_].get_scaled_boundary_points(
                            1.0 * ii / Gamma_steps
                        )

                        # Get Gamma value
                        Gamma = obs[it_obs2_].get_gamma(
                            x_obs_sf_interior, in_global_frame=True
                        )
                        intersection_points = np.hstack(
                            (
                                intersection_points,
                                x_obs_sf_interior[:, Gamma < 1],
                            )
                        )
                    # Check center point
                    if 1 > obs[it_obs2_].get_gamma(
                        obs[it_obs1_].center_position, in_global_frame=True
                    ):
                        intersection_points = np.hstack(
                            [
                                intersection_points,
                                np.tile(obs[it_obs1_].center_position, (1, 1)).T,
                            ]
                        )

                # Get mean
                # intersection_points = np.unique(intersection_points, axis=1)
                Intersections.set(
                    it_obs1, it_obs2, np.mean(intersection_points, axis=1)
                )

    intersection_cluster_list = get_intersection_cluster(
        Intersections, obs, representation_type
    )
    if get_intersection_matrix:
        return intersection_cluster_list, Intersections
    else:
        return intersection_cluster_list


def get_intersection_cluster(Intersections, obs, representation_type="single_point"):
    """Get the clusters number of the intersections.
    It automatically assign the reference points for intersecting clusters."""
    # Get variables
    num_obstacles = Intersections.num_obstacles
    dim = obs[0].center_position.shape[0]

    R_max = np.zeros(num_obstacles)
    for it_obs in range(num_obstacles):
        R_max[it_obs] = obs[it_obs].get_reference_length()

    # Iterate over all obstacles with an intersection
    intersection_matrix = Intersections.get_bool_matrix()

    # All obstacles, which have at least one intersection
    intersecting_obstacles = np.arange(num_obstacles)[
        np.sum(intersection_matrix, axis=0) > 0
    ]

    if not obs.index_wall is None:
        # TODO solve more cleanly...
        intersecting_obstacles = np.delete(
            intersecting_obstacles,
            np.nonzero(intersecting_obstacles == obs.index_wall),
            axis=0,
        )

    intersection_cluster_list = []

    while intersecting_obstacles.shape[0]:
        intersection_matrix_reduced = intersection_matrix[intersecting_obstacles, :][
            :, intersecting_obstacles
        ]
        intersection_cluster = np.zeros(intersecting_obstacles.shape[0], dtype=bool)

        intersection_cluster[0] = True

        # By default new obstacles in cluster
        new_obstacles = True

        # Iteratively search through clusters. Similar to google page ranking
        while new_obstacles:
            intersection_cluster_old = intersection_cluster
            intersection_cluster = (
                intersection_matrix_reduced.dot(intersection_cluster)
                + intersection_cluster
            )
            intersection_cluster = intersection_cluster.astype(bool)

            # Bool operation. Equals to one if not equal
            new_obstacles = np.any(intersection_cluster ^ intersection_cluster_old)

        intersection_cluster_list.append(
            intersecting_obstacles[intersection_cluster].tolist()
        )

        if not obs.index_wall is None:  # Add wall connection
            if np.sum(
                intersection_matrix[intersection_cluster_list[-1], :][:, obs.index_wall]
            ):  # nonzero
                intersection_cluster_list[-1].append(obs.index_wall)

        # Only keep non-intersecting obstacles
        intersecting_obstacles = intersecting_obstacles[intersection_cluster == 0]

    if representation_type == "single_point":
        get_single_reference_point(
            obs=obs,
            obstacle_weight=R_max,
            intersection_clusters=intersection_cluster_list,
            Intersections=Intersections,
        )

    elif representation_type == "hirarchy":
        get_obstacle_tree(
            obs=obs,
            obstacle_weight=R_max,
            intersection_clusters=intersection_cluster_list,
            Intersections=Intersections,
            dim=dim,
        )

    else:
        raise NotImplementedError("Method '{}' not defined".format(representation_type))

    return intersection_cluster_list


def get_single_reference_point(
    obs,
    intersection_clusters: list,
    Intersections: IntersectionMatrix,
    obstacle_weight: np.ndarray = None,
):
    """Sets the references-points of all obstacles in the clusteres, if they are not walls.
    The reference points are set by assigining obs.
    local reference points from intersection clusters.
    """

    if obstacle_weight is None:
        obstacle_weight = np.ones(len(obs)) / len(obs)

    # Get index wall once, since it might be costly
    index_wall = obs.index_wall

    # Iterate over the list of intersection clusters and find one common
    # reference point for each of the clusters
    for ii in range(len(intersection_clusters)):
        # Intersection points of obstacles with wall
        wall_intersection_points = []

        if index_wall in intersection_clusters[ii]:
            # The cluster is touching wall, hence the reference point has to be placed at the wall.
            for jj in intersection_clusters[ii]:
                if jj == index_wall:
                    continue

                # Check if obstacle jj is intersecting with the wall
                if Intersections.is_intersecting(jj, index_wall):
                    wall_intersection_points.append(Intersections.get(jj, index_wall))

            wall_intersection_points = np.array(wall_intersection_points).T
            for jj in intersection_clusters[ii]:
                if jj == index_wall:
                    continue

                if wall_intersection_points.shape[1] == 1:
                    obs[jj].set_reference_point(
                        wall_intersection_points[:, 0], in_global_frame=True
                    )
                else:
                    # In case of severall interesections with wall; take the closest.
                    dist_intersection = np.linalg.norm(
                        wall_intersection_points
                        - np.tile(
                            obs[jj].position,
                            (wall_intersection_points.shape[1], 1),
                        ).T,
                        axis=0,
                    )
                    obs[jj].set_reference_point(
                        wall_intersection_points[:, np.argmin(dist_intersection)],
                        in_global_frame=True,
                    )

        else:
            # No intersection with wall
            geometric_center = np.zeros(obs.dim)
            total_weight = 0

            # TODO: take intersection position mean
            for jj in intersection_clusters[ii]:
                geometric_center += obstacle_weight[jj] * np.array(
                    obs[jj].center_position
                )
                total_weight += obstacle_weight[jj]
            geometric_center /= total_weight

            for jj in intersection_clusters[ii]:
                obs[jj].set_reference_point(geometric_center, in_global_frame=True)
            # Take first position of intersection # TODO make more general


def get_obstacle_tree(obs, obstacle_weight, intersection_clusters, Intersection, dim):
    # All (close) relatives of one object
    intersection_relatives = intersection_matrix
    # Choose center

    geometric_center = np.zeros(dim)
    for ii in range(len(intersection_clusters)):  # list of cluster-lists
        total_weight = 0

        # Find center obstacle
        for jj in intersection_clusters[ii]:
            geometric_center += obstacle_weight[jj] * np.array(obs[jj].center_position)
            total_weight += obstacle_weight[jj]
        geometric_center /= total_weight

        center_distance = [
            LA.norm(geometric_center - np.array(obs[kk].center_position))
            for kk in range(len(intersection_clusters[ii]))
        ]

        # Center obstacle // root_index
        root_index = np.arange(num_obstacles)[intersection_clusters[ii]][
            np.argmin(center_distance)
        ]

        obs[root_index].hirarchy = 0
        obs[root_index].reference_point = obs[root_index].center_position

        obstacle_tree = [root_index]

        # For all elements in one cluster
        while len(obstacle_tree):
            # Iterate over all children
            for jj in np.arange(num_obstacles)[
                intersection_relatives[:, obstacle_tree[0]]
            ]:
                if jj != obstacle_tree[0]:
                    obs[jj].hirarchy = obs[obstacle_tree[0]].hirarchy + 1
                    obs[jj].ind_parent = obstacle_tree[0]  # TODO use pointer...

                    obs[jj].reference_point = Intersections.get(jj, obstacle_tree[0])
                    # intersection_relatives[jj, obstacle_tree[0]] = False
                    intersection_relatives[obstacle_tree[0], jj] = False
                    obstacle_tree.append(jj)

            del obstacle_tree[0]
