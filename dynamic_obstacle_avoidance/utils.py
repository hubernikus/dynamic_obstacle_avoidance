"""
bVarios tools and uitls for advanced obstacle-avoidance-library
"""
# Author Lukas Huber
# Date 2018-02-15

import warnings
from typing import Optional

import numpy as np
import numpy.linalg as LA
import numpy.typing as npt

from vartools.angle_math import *
from vartools.linalg import get_orthogonal_basis


def avoid_moving_obstacle_with_maximum(
    velocity: np.ndarray, max_velocity: float
) -> np.ndarray:
    # TODO: test this library & get normalized-norm
    # Implement the three cases from the paper

    velocity = velocity

    return safe_velocity


def get_relative_obstacle_velocity(
    position: np.ndarray,
    obstacle_list,
    E_orth: np.ndarray,
    weights: list,
    ind_obstacles: Optional[int] = None,
    gamma_list: Optional[list] = None,
    cut_off_gamma: float = 1e4,
    velocity_only_in_positive_normal_direction: bool = True,
    normal_weight_factor: float = 1.3,
) -> np.ndarray:
    """Get the relative obstacle velocity

    Parameters
    ----------
    E_orth: array which contains orthogonal matrix with repsect to the normal
    direction at <position>
    array of (dimension, dimensions, n_obstacles
    obstacle_list: list or <obstacle-conainter> with obstacles
    ind_obstacles: Inidicates which obstaces will be considered (array-like of int)
    gamma_list: Precalculated gamma-values (list of float) -
                It is adviced to use 'proportional' gamma values, rather
                than relative ones

    Return
    ------
    relative_velocity: array-like of float
    """
    n_obstacles = len(obstacle_list)

    if gamma_list is None:
        gamma_list = np.zeros(n_obstacles)
        for n in range(n_obstacles):
            gamma_list[n] = obs[n].get_gamma(position, in_global_frame=True)

    if ind_obstacles is None:
        ind_obstacles = gamma_list < cut_off_gamma
        gamma_list = gamma_list[ind_obstacles]

    obs = obstacle_list
    ind_obs = ind_obstacles
    dim = position.shape[0]

    xd_obs = np.zeros((dim))
    for ii, it_obs in zip(range(np.sum(ind_obs)), np.arange(n_obstacles)[ind_obs]):
        if dim == 2:
            if obs[it_obs].angular_velocity is None:
                xd_w = np.zeros(dim)
            else:
                xd_w = np.cross(
                    np.hstack(([0, 0], obs[it_obs].angular_velocity)),
                    np.hstack((position - np.array(obs[it_obs].center_position), 0)),
                )
                xd_w = xd_w[0:2]
        elif dim == 3:
            if obs[it_obs].angular_velocity is not None and LA.norm(
                obs[it_obs].angular_velocity
            ):
                xd_w = np.cross(
                    obs[it_obs].angular_velocity,
                    position - obs[it_obs].center_position,
                )
            else:
                xd_w = np.zeros(dim)
        else:
            if obs[it_obs].angular_velocity is not None and LA.norm(
                obs[it_obs].angular_velocity
            ):
                warnings.warn("Angular velocity is not defined for={}".format(d))
            xd_w = np.zeros(dim)

        weight_angular = np.exp(-1.0 * (np.max([gamma_list[ii], 1]) - 1))

        linear_velocity = obs[it_obs].linear_velocity

        if velocity_only_in_positive_normal_direction:
            lin_vel_local = (E_orth[:, :, ii]).T.dot(obs[it_obs].linear_velocity)
            if lin_vel_local[0] < 0 and not obs[it_obs].is_boundary:
                # Obstacle is moving towards the agent
                linear_velocity = np.zeros(lin_vel_local.shape[0])
            else:
                # For safety in close region, we multiply the velocity
                lin_vel_local[0] = normal_weight_factor * lin_vel_local[0]
                linear_velocity = E_orth[:, 0, ii].dot(lin_vel_local[0])

            weight_linear = np.exp(-1 / 1 * (np.max([gamma_list[ii], 1]) - 1))
        xd_obs_n = weight_linear * linear_velocity + weight_angular * xd_w

        # The Exponential term is very helpful as it help to avoid
        # the crazy rotation of the robot due to the rotation of the object
        if obs[it_obs].is_deforming:
            weight_deform = np.exp(-1 / 1 * (np.max([gamma_list[ii], 1]) - 1))
            vel_deformation = obs[it_obs].get_deformation_velocity(pos_relative[:, ii])

            if velocity_only_in_positive_normal_direction:
                vel_deformation_local = E_orth[:, :, ii].T.dot(vel_deformation)
                if (vel_deformation_local[0] > 0 and not obs[it_obs].is_boundary) or (
                    vel_deformation_local[0] < 0 and obs[it_obs].is_boundary
                ):
                    vel_deformation = np.zeros(vel_deformation.shape[0])

                else:
                    vel_deformation = E_orth[:, 0, ii].dot(vel_deformation_local[0])

            xd_obs_n += weight_deform * vel_deformation
        xd_obs = xd_obs + xd_obs_n * weights[ii]
    return xd_obs


def get_weight_from_gamma(*args, **kwargs):
    raise Exception("Renamed to 'get_weight_from_inv_of_gamma'")


def get_weight_from_inv_of_gamma(
    gamma_array: np.ndarray, power_value: float = 1.0
) -> np.ndarray:
    """Returns weight-array based on input of gamma_array."""
    weight = 1.0 / np.abs(gamma_array) ** power_value
    return weight


def get_weight_gamma(gamma_array: np.ndarray, power_value: float = 1.0) -> np.ndarray:
    """Returns weight-array based on input of gamma_array."""
    weight = np.abs(gamma_array) ** power_value
    weight = weight / np.sum(weight)
    return weight


def getGammmaValue_ellipsoid(ob, x_t, relativeDistance=True):
    if relativeDistance:
        return np.sum(
            (x_t / np.tile(ob.a, (x_t.shape[1], 1)).T)
            ** (2 * np.tile(ob.p, (x_t.shape[1], 1)).T),
            axis=0,
        )
    else:
        return np.sum(
            (x_t / np.tile(ob.a, (x_t.shape[1], 1)).T)
            ** (2 * np.tile(ob.p, (x_t.shape[1], 1)).T),
            axis=0,
        )


def get_radius_ellipsoid(x_t, a=[], ob=[]):
    # Derivation from  x^2/a^2 + y^2/b^2 = 1
    if not np.array(a).shape[0]:
        a = ob.a

    if x_t[0]:  # nonzero value
        rat_x1_x2 = x_t[1] / x_t[0]
        x_1_val = np.sqrt(1.0 / (1.0 / a[0] ** 2 + 1.0 * rat_x1_x2**2 / a[1] ** 2))
        return x_1_val * np.sqrt(1 + rat_x1_x2**2)
    else:
        return a[1]


def get_radius(vec_point2ref, vec_cent2ref=[], a=[], obs=[]):
    dim = 2  # TODO higher dimensions

    if not np.array(vec_cent2ref).shape[0]:
        vec_cent2ref = np.array(obs.reference_point) - np.array(obs.center_position)

    if not np.array(a).shape[0]:
        a = obs.axes_length

    if obs.th_r:
        vec_cent2ref = np.array(obs.rotMatrix).T.dot(vec_cent2ref)
        vec_point2ref = np.array(obs.rotMatrix).T.dot(vec_point2ref)

    if not LA.norm(vec_cent2ref):  # center = ref
        return get_radius_ellipsoid(vec_point2ref, a)

    dir_surf_cone = np.zeros((dim, 2))
    rad_surf_cone = np.zeros((2))

    if np.cross(vec_point2ref, vec_cent2ref) > 0:
        # 2D vectors pointing in opposite direction
        dir_surf_cone[:, 0] = vec_cent2ref
        rad_surf_cone[0] = np.abs(
            get_radius_ellipsoid(dir_surf_cone[:, 0], a) - LA.norm(vec_cent2ref)
        )

        dir_surf_cone[:, 1] = -1 * np.array(vec_cent2ref)
        rad_surf_cone[1] = get_radius_ellipsoid(dir_surf_cone[:, 1], a) + LA.norm(
            vec_cent2ref
        )

    else:
        dir_surf_cone[:, 0] = -1 * np.array(vec_cent2ref)

        rad_surf_cone[0] = get_radius_ellipsoid(dir_surf_cone[:, 0], a) + LA.norm(
            vec_cent2ref
        )

        dir_surf_cone[:, 1] = vec_cent2ref
        rad_surf_cone[1] = np.abs(
            get_radius_ellipsoid(dir_surf_cone[:, 1], a) - LA.norm(vec_cent2ref)
        )

    # color_set = ['g', 'r']
    # for i in range(2):
    # plt.plot([obs.center_dyn[0], obs.center_dyn[0]+dir_surf_cone[0,i]], [obs.center_dyn[1], obs.center_dyn[1]+dir_surf_cone[1,i]], color_set[i])
    # plt.show()

    ang_tot = np.pi / 2
    for ii in range(12):  # n_iter
        rotMat = np.array(
            [
                [np.cos(ang_tot), np.sin(ang_tot)],
                [-np.sin(ang_tot), np.cos(ang_tot)],
            ]
        )

        # vec_ref2dir = rotMat @ dir_surf_cone[:, 0]
        # vec_ref2dir /= LA.norm(vec_ref2dir) # nonzero value expected
        # rad_ref2 = get_radius_ellipsoid(vec_ref2dir, a)
        # vec_ref2surf = rad_ref2*vec_ref2dir - vec_cent2ref

        vec_cent2dir = rotMat.dot(dir_surf_cone[:, 0])
        vec_cent2dir /= LA.norm(vec_cent2dir)  # nonzero value expected
        rad_ref2 = get_radius_ellipsoid(vec_cent2dir, a)
        vec_ref2surf = rad_ref2 * vec_cent2dir - vec_cent2ref

        crossProd = np.cross(vec_ref2surf, vec_point2ref)
        if crossProd < 0:
            # dir_surf_cone[:, 0] = vec_ref2dir
            dir_surf_cone[:, 0] = vec_cent2dir
            rad_surf_cone[0] = LA.norm(vec_ref2surf)
        elif crossProd == 0:  # how likely is this lucky guess?
            return LA.norm(vec_ref2surf)
        else:
            # dir_surf_cone[:, 1] = vec_ref2dir
            dir_surf_cone[:, 1] = vec_cent2dir
            rad_surf_cone[1] = LA.norm(vec_ref2surf)

        ang_tot /= 2.0

        # vec_transp = np.array(obs.rotMatrix).dot(vec_ref2surf)
        # plt.plot([obs.center_dyn[0], obs.center_dyn[0]+vec_transp[0]],[obs.center_dyn[1], obs.center_dyn[1]+vec_transp[1]], 'b')
        # plt.show()
        # import pdb; pdb.set_trace() ## DEBUG ##

    return np.mean(rad_surf_cone)


def findRadius(ob, direction, a=[], repetition=6, steps=10):
    # NOT SURE IF USEFULL -- NORMALLY x = Gamma*Rad
    # TODO check
    if not len(a):
        a = [np.min(ob.a), np.max(ob.a)]

    # repetition
    for ii in range(repetition):
        if a[0] == a[1]:
            return a[0]

        magnitudeDir = np.linspace(a[0], a[1], num=steps)
        Gamma = getGammmaValue_ellipsoid(
            ob,
            np.tile(direction, (steps, 1)).T
            * np.tile(magnitudeDir, (np.array(ob.x0).shape[0], 1)),
        )

        if np.sum(Gamma == 1):
            return magnitudeDir[np.where(Gamma == 1)]
        posBoundary = np.where(Gamma < 1)[0][-1]

        a[0] = magnitudeDir[posBoundary]
        posBoundary += 1
        while Gamma[posBoundary] <= 1:
            posBoundary += 1

        a[1] = magnitudeDir[posBoundary]

    return (a[0] + a[1]) / 2.0


def findBoundaryPoint(ob, direction):
    # Numerical search -- TODO analytic
    dirNorm = LA.norm(direction, 2)
    if dirNorm:
        direction = direction / dirNorm
    else:
        print("No feasible direction is given")
        return ob.x0

    a = [np.min(x0.a), np.max(x0.a)]

    return (a[0] + a[1]) / 2.0 * direction + x0


def compute_eigenvalueMatrix(Gamma, rho=1, dim=2, radialContuinity=True):
    if radialContuinity:
        Gamma = np.max([Gamma, 1])

    delta_lambda = 1.0 / np.abs(Gamma) ** (1 / rho)
    lambda_referenceDir = 1 - delta_lambda
    lambda_tangentDir = 1 + delta_lambda

    return np.diag(
        np.hstack((lambda_referenceDir, np.ones(dim - 1) * lambda_tangentDir))
    )


def compute_weights(
    distMeas: npt.ArrayLike,
    distMeas_lowerLimit: float = 1,
    weightPow: float = 1,
) -> np.ndarray:
    """Compute weights based on a distance measure (with no upper limit)"""
    distMeas = np.array(distMeas)
    n_points = distMeas.shape[0]

    critical_points = distMeas <= distMeas_lowerLimit

    if np.sum(critical_points):  # at least one
        if np.sum(critical_points) == 1:
            w = critical_points * 1.0
            return w
        else:
            # TODO: continuous weighting function
            warnings.warn("Implement continuity of weighting function.")
            w = critical_points * 1.0 / np.sum(critical_points)
            return w

    distMeas = distMeas - distMeas_lowerLimit
    w = (1 / distMeas) ** weightPow
    if np.sum(w) == 0:
        return w

    w = w / np.sum(w)  # Normalization

    return w


def compute_R(d, th_r):
    warnings.warn("This function will be removed. Don't use it")
    if th_r == 0:
        rotMatrix = np.eye(d)
    # rotating the query point into the obstacle frame of reference
    if d == 2:
        rotMatrix = np.array(
            [[np.cos(th_r), -np.sin(th_r)], [np.sin(th_r), np.cos(th_r)]]
        )
    elif d == 3:
        R_x = np.array(
            [
                [
                    1,
                    0,
                    0,
                ],
                [0, np.cos(th_r[0]), np.sin(th_r[0])],
                [0, -np.sin(th_r[0]), np.cos(th_r[0])],
            ]
        )

        R_y = np.array(
            [
                [np.cos(th_r[1]), 0, -np.sin(th_r[1])],
                [0, 1, 0],
                [np.sin(th_r[1]), 0, np.cos(th_r[1])],
            ]
        )

        R_z = np.array(
            [
                [np.cos(th_r[2]), np.sin(th_r[2]), 0],
                [-np.sin(th_r[2]), np.cos(th_r[2]), 0],
                [0, 0, 1],
            ]
        )

        rotMatrix = R_x.dot(R_y).dot(R_z)
    else:
        warnings.warn("rotation not yet defined in dimensions d > 3 !")
        rotMatrix = np.eye(d)

    return rotMatrix


def obs_check_collision_2d(obs_list, XX, YY):
    """Check if points (as a list in *args) are colliding with any of the obstacles.
    Function is implemented for 2D only."""
    d = 2

    dim_points = XX.shape
    if len(dim_points) == 1:
        N_points = dim_points[0]
    else:
        N_points = dim_points[0] * dim_points[1]

    # No obstacles
    if not len(obs_list):
        return np.ones((dim_points))

    points = np.array(([np.reshape(XX, (N_points,)), np.reshape(YY, (N_points,))]))
    # At the moment only implemented for 2D
    collision = np.zeros(dim_points)

    N_points = points.shape[1]

    noColl = np.ones((1, N_points), dtype=bool)

    for it_obs in range(len(obs_list)):
        Gamma = np.zeros(N_points)
        for ii in range(N_points):
            Gamma[ii] = obs_list[it_obs].get_gamma(points[:, ii], in_global_frame=True)

        noColl = noColl * Gamma > 1

    return np.reshape(noColl, dim_points)


def obs_check_collision(obs_list, dim, *args):
    """Check if points (as a list in *args) are colliding with any of the
    obstacles."""
    # No obstacles
    if len(obs_list) == 0:
        return np.ones(args[0].shape)

    dim = obs_list[0].dim

    if len(*args) == dim:
        points = np.array([])
        for ii in range(dim):
            input_shape = args[0].shape
            points = np.vstack((points, np.arary(args[ii]).flatten()))

    N_points = points.shape[1]

    # At the moment only implemented for 2D
    collision = np.zeros((N_points))

    for ii in range(N_points):
        pass
    import pdb

    pdb.set_trace()
    return noColl


def obs_check_collision_ellipse(obs_list, dim, points):
    warnings.warn("Depreciated --- delete this function ")
    # TODO: delete / depreciated
    for it_obs in range(len(obs_list)):
        # \Gamma = \sum_{i=1}^d (xt_i/a_i)^(2p_i) = 1
        R = compute_R(dim, obs_list[it_obs].th_r)

        Gamma = sum(
            (
                1
                / obs_list[it_obs].sf
                * R.T.dot(
                    points - np.tile(np.array([obs_list[it_obs].x0]).T, (1, N_points))
                )
                / np.tile(np.array([obs_list[it_obs].a]).T, (1, N_points))
            )
            ** (np.tile(2 * np.array([obs_list[it_obs].p]).T, (1, N_points)))
        )

        noColl = noColl * Gamma > 1
    return noColl


def get_tangents2ellipse(edge_point, axes, center_point=None, dim=2):
    """
    Get 2D tangent vector of ellipse with axes <<axes>> and center <<center_point>>
    with respect to a point <<edge_point>>

    Function returns the tangents and the points of contact

    """
    if not dim == 2:
        # TODO cut ellipse along direction & apply 2D-problem
        raise TypeError("Not implemented for higher dimension")

    if not center_point is None:
        edge_point = edge_point - center_point

    # Intersection of (x_1/a_1)^2 +( x_2/a_2)^2 = 1 & x_2=m*x_1+c
    # Solve for determinant D=0 (tangent with only one intersection point)
    A_ = edge_point[0] ** 2 - axes[0] ** 2
    B_ = -2 * edge_point[0] * edge_point[1]
    C_ = edge_point[1] ** 2 - axes[1] ** 2
    D_ = B_**2 - 4 * A_ * C_

    if D_ < 0:
        # print(edge_point)
        # print(axes)
        raise RuntimeError("Invalid value for D_<0 (D_={})".format(D_))

    m = np.zeros(2)

    m[1] = (-B_ - np.sqrt(D_)) / (2 * A_)
    m[0] = (-B_ + np.sqrt(D_)) / (2 * A_)

    tangent_points = np.zeros((dim, 2))
    tangent_vectors = np.zeros((dim, 2))
    tangent_angles = np.zeros(2)

    for ii in range(2):
        c = edge_point[1] - m[ii] * edge_point[0]

        A = (axes[0] * m[ii]) ** 2 + axes[1] ** 2
        B = 2 * axes[0] ** 2 * m[ii] * c
        # D != 0 to be tangent, so C not interesting.

        tangent_points[0, ii] = -B / (2 * A)
        tangent_points[1, ii] = m[ii] * tangent_points[0, ii] + c

        tangent_vectors[:, ii] = tangent_points[:, ii] - edge_point
        tangent_vectors[:, ii] /= LA.norm(tangent_vectors[:, ii])

        # normal_vectors[:, ii] = np.array([tangent_vectors[1,ii], -tangent_vectors[0,ii]])

        # Check direction
        # normalDistance2center[ii] = normal_vectors[:, ii].T.dot(edge_point)

        # if (normalDistance2center[ii] < 0):
        # normal_vectors[:, ii] = normal_vectors[:, ii]*(-1)
        # normalDistance2center[ii] *= -1
        tangent_angles[ii] = np.arctan2(tangent_points[1, ii], tangent_points[0, ii])

    if angle_difference_directional(tangent_angles[1], tangent_angles[0]) < 0:
        tangent_points = np.flip(tangent_points, axis=1)
        tangent_vectors = np.flip(tangent_vectors, axis=1)

    if not center_point is None:
        tangent_points = (
            tangent_points + np.tile(center_point, (tangent_points.shape[1], 1)).T
        )

    return tangent_vectors, tangent_points


def get_reference_weight(
    distance,
    obs_reference_size=None,
    distance_min=0,
    distance_max=3,
    weight_pow=1,
):
    """Get a weight inverse proportinal to the distance"""

    # TODO: based on inverse prop weight calculation
    # weights = get_inverse_proprtional_weight(distance, distance_min, distance_max, weight_pow)
    weights_all = np.zeros(distance.shape)

    # if False:
    if any(np.logical_and(distance <= distance_min, distance > 0)):
        ind0 = distance == 0
        weights_all[ind0] = 1 / np.sum(ind0)
        return weights_all

    # print('distance_max', distance_max)
    ind_range = np.logical_and(distance > distance_min, distance < distance_max)
    if not any(ind_range):
        return weights_all

    dist_temp = distance[ind_range]
    weights = 1 / (dist_temp - distance_min) - 1 / (distance_max - distance_min)
    weights = weights**weight_pow

    # Normalize
    weights = weights / np.sum(weights)

    # Add amount of movement relative to distance
    if not obs_reference_size is None:
        distance_max = distance_max * obs_reference_size[ind_range]

    weight_ref_displacement = 1 / (dist_temp + 1 - distance_min) - 1 / (
        distance_max + 1 - distance_min
    )

    weights_all[ind_range] = weights * weight_ref_displacement

    return weights_all


def get_inverse_proprtional_weight(
    distance, distance_min=0, distance_max=3, weight_pow=1
):
    """Get a weight inverse proportinal to the distance"""
    weights = np.zeros(distance.shape)

    if any(np.logical_and(distance <= distance_min, distance > 0)):
        ind0 = distance == 0
        weights[ind0] = 1 / np.sum(ind0)
        return weights

    ind_range = np.logical_and(distance > distance_min, distance < distance_max)
    if not any(ind_range):
        return weights

    dist_temp = distance[ind_range]
    weights[ind_range] = 1 / (dist_temp - distance_min) - 1 / (
        distance_max - distance_min
    )
    weights[ind_range] = weights[ind_range] ** weight_pow

    # Normalize
    weights = weights / np.sum(weights)
    return weights
