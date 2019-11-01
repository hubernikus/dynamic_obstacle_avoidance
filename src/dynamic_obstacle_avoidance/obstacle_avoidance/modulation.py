'''
Obstacle Avoidance Library with different options

@author Lukas Huber
@date 2018-02-15

'''
import numpy as np
import numpy.linalg as LA
from numpy import pi

from math import cos, sin

import warnings

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *

import matplotlib.pyplot as plt

def get_orthogonal_basis(vector, normalize=False):
    if not type(vector) == np.ndarray:
        vector = np.array(vector)

    if normalize:
        v_norm = LA.norm(vector)
        if v_norm:
            vector = vector / v_norm

    dim = vector.shape[0]

    Basis_Matrix = np.zeros((dim, dim))

    if dim == 2:
        Basis_Matrix[:, 0] = vector
        Basis_Matrix[:, 1] = np.array([Basis_Matrix[1, 0],
                                       -Basis_Matrix[0, 0]])

    if dim > 2:
        warnings.warn("Implement higher dimensionality than d={}".format(dim))

    return Basis_Matrix


def get_directional_weighted_sum(reference_direction, directions, weights, normalize=True, normalize_reference=True, obs=None, position=[]):
    # TODO remove obs and position
    # Move to different file
    ind_nonzero = (weights>0)

    reference_direction = np.copy(reference_direction)
    directions = directions[:, ind_nonzero]
    weights = weights[ind_nonzero]

    # TODO remove obs from arguments after debugging
    n_directions = weights.shape[0]
    if n_directions<=1:
        return directions[:, 0]

    dim = np.array(reference_direction).shape[0]

    # Create copy to avoid changing initial values
    if False and not isinstance(obs, type(None)):
        ref_abs = obs.transform_relative2global_dir(reference_direction)

        position = obs.transform_relative2global(reference_direction)
        plt.quiver(position[0], position[1], ref_abs[0], ref_abs[1], color='g', label='Reference')

        dir_abs = np.zeros((dim, n_directions))
        for ii in range(n_directions):
            dir_abs[:, ii] = obs.transform_relative2global_dir(directions[:,ii])
            plt.quiver(position[0], position[1], dir_abs[0,ii], dir_abs[1,ii], color='b', label='Normal')

    if normalize_reference:
        norm_refDir = LA.norm(reference_direction)
        if norm_refDir: # nonzero
            reference_direction /= norm_refDir

     # TODO - higher dimensions
    if normalize:
        norm_dir = LA.norm(directions, axis=0)
        ind_nonzero = (norm_dir>0)
        directions[:,ind_nonzero] = directions[:,ind_nonzero]/np.tile(norm_dir[ind_nonzero], (dim, 1))

    OrthogonalBasisMatrix = get_orthogonal_basis(reference_direction)

    directions_referenceSpace = np.zeros(np.shape(directions))
    for ii in range(np.array(directions).shape[1]):
        directions_referenceSpace[:,ii] = OrthogonalBasisMatrix.T.dot( directions[:,ii])

    directions_directionSpace = directions_referenceSpace[1:, :]

    norm_dirSpace = LA.norm(directions_directionSpace, axis=0)
    ind_nonzero = (norm_dirSpace > 0)

    directions_directionSpace[:,ind_nonzero] = (directions_directionSpace[:, ind_nonzero] /  np.tile(norm_dirSpace[ind_nonzero], (dim-1, 1)))

    # Do not check cosinus, since normalization happened
    # TODO check why low, and remove

    cos_directions = directions_referenceSpace[0,:]
    if np.sum(cos_directions > 1) or np.sum(cos_directions < -1):
        cos_directions = np.min(np.vstack((cos_directions, np.ones(n_directions))), axis=0)
        cos_directions = np.max(np.vstack((cos_directions, -np.ones(n_directions))), axis=0)
        warnings.warn("Cosinus value out of bound.")

    directions_directionSpace *= np.tile(np.arccos(cos_directions), (dim-1, 1))
    direction_dirSpace_weightedSum = np.sum(directions_directionSpace*
                                            np.tile(weights, (dim-1, 1)), axis=1)

    norm_directionSpace_weightedSum = LA.norm(direction_dirSpace_weightedSum)
    if norm_directionSpace_weightedSum:
        direction_weightedSum = (OrthogonalBasisMatrix.dot(
                                  np.hstack((np.cos(norm_directionSpace_weightedSum),
                                              np.sin(norm_directionSpace_weightedSum) / norm_directionSpace_weightedSum * direction_dirSpace_weightedSum)) ))
    else:
        direction_weightedSum = OrthogonalBasisMatrix[:,0]

    if False and not isinstance(obs, type(None)):
        wei_abs = obs.transform_relative2global_dir(direction_weightedSum)
        plt.quiver(position[0], position[1], wei_abs[0], wei_abs[1], color='r', label='mean')
        plt.legend()
        
    return direction_weightedSum


def compute_modulation_matrix(x_t, obs, R, matrix_singularity_margin=pi/2.0*1.05):
    # The function evaluates the gamma function and all necessary components needed to construct the modulation function, to ensure safe avoidance of the obstacles.
    # Beware that this function is constructed for ellipsoid only, but the algorithm is applicable to star shapes.
    # 
    # Input
    # x_t [dim]: The position of the robot in the obstacle reference frame
    # obs [obstacle class]: Description of the obstacle with parameters
    # R [dim x dim]: Rotation matrix 
    #
    # Output
    # E [dim x dim]: Basis matrix with rows the reference and tangent to the obstacles surface
    # D [dim x dim]: Eigenvalue matrix which is responsible for the modulation
    # Gamma [dim]: Distance function to the obstacle surface (in direction of the reference vector)
    # E_orth [dim x dim]: Orthogonal basis matrix with rows the normal and tangent

    dim = obs.dim
    
    if hasattr(obs, 'rho'):
        rho = np.array(obs.rho)
    else:
        rho = 1

    Gamma = obs.get_gamma(x_t, in_global_frame=False) # function for ellipsoids

    normal_vector = obs.get_normal_direction(x_t, in_global_frame=False)
    reference_direction = obs.get_reference_direction(x_t, in_global_frame=False)

    # Check if there was correct placement of reference point
    Gamma_referencePoint = obs.get_gamma(obs.reference_point)

    if not obs.is_boundary and Gamma_referencePoint >= 1:
        # Check what this does and COMMENT!!!!
        
        # surface_position = obs.get_obstace_radius* x_t/LA.norm(x_t)
        # direction_surface2reference = obs.get_reference_point()-surface_position

        # Per default negative
        referenceNormal_angle = np.arccos(reference_direction.T @ normal_vector)
        
        # if referenceNormal_angle < (matrix_singularity_margin):
            # x_global = obs.transform_relative2global(x_t)
            # plt.quiver(x_global[0],x_global[1], normal_vector[0], normal_vector[1], 'r')
            # plt.quiver(x_global[0],x_global[1], normal_vector[0], normal_vector[1], color='r')
            # plt.quiver(x_global[0],x_global[1], reference_direction[0], reference_direction[1], color='b')
            
            # referenceNormal_angle = np.min([0, referenceNormal_angle-pi/2.0])
    
            # Gamma_convexHull = 1*referenceNormal_angle/(matrix_singularity_margin-pi/2.0)
            # Gamma = np.max([Gamma, Gamma_convexHull])
            
            # reference_orthogonal = (normal_vector -
                                    # reference_direction * reference_direction.T @ normal_vector)
            # normal_vector = (reference_direction*np.sin(matrix_singularity_margin)
                             # + reference_orthogonal*np.cos(matrix_singularity_margin))

            # plt.quiver(x_global[0],x_global[1], normal_vector[0], normal_vector[1], color='g')
            # plt.ion()

    E_orth = np.zeros((dim, dim))
    
    # Create orthogonal basis matrix        
    E_orth[:, 0] = normal_vector# Basis matrix

    for ii in range(1,dim):

        if dim ==2:
            E_orth[0, 1] = E_orth[1, 0]
            E_orth[1, 1] = - E_orth[0, 0]
        else:
            warnings.warn('Implement higher dimensions for E')
            
        # TODO higher dimensions
        # E[:dim-(ii), ii] = normal_vector[:dim-(ii)]*normal_vector[dim-(ii)]
        # E[dim-(ii), ii] = -np.dot(normal_vector[:dim-(ii)], normal_vector[:dim-(ii)])
        # E_orth[:, ii] = E_orth[:, ii]/LA.norm(E_orth[:, ii])

    E = np.copy((E_orth))
    E[:, 0] = -reference_direction
    
    eigenvalue_reference, eigenvalue_tangent = calculate_eigenvalues(Gamma,
                                                                     is_boundary=obs.is_boundary)
    D = np.diag(np.hstack((eigenvalue_reference, np.ones(dim-1)*eigenvalue_tangent)))
    
    return E, D, Gamma, E_orth


def calculate_eigenvalues(Gamma, rho=1, is_boundary=False):
    if Gamma<=1:# point inside the obstacle
        delta_eigenvalue = 1 
    else:
        delta_eigenvalue = 1./abs(Gamma)**(1/rho)

    eigenvalue_reference = 1 - delta_eigenvalue
    # eigenvalue_reference = 0.1
    
    if is_boundary:
        # eigenvalue_tangent = 1
        eigenvalue_tangent = 1 + delta_eigenvalue            
    else:
        eigenvalue_tangent = 1 + delta_eigenvalue            

    # print('eig vals r={}, tang={}'.format(eigenvalue_reference, eigenvalue_tangent))
    # import pdb; pdb.set_trace() ## DEBUG ##
    
    return eigenvalue_reference, eigenvalue_tangent


# def get_gamma(x_t, a, p=1, mode="proportional", Gamma_min=1, dist_ref=0):
#     # TODO --- Integrate in obstacle
#     dim = np.array(x_t).shape[0]
#     # TODO MAYBE use reference point as reference
#     if mode=="proportional":
#         return np.sum((x_t/a)**(2*p)) # distance function for ellipsoids
#     elif mode=="distance":
#         if not dist_ref:
#             dist_ref = np.max(a) # other choices possible
#         return (LA.norm(x_t)-get_radius(x_t, vec_cent2ref=np.zeros(dim), a=a))/dist_ref+Gamma_min
#     elif mode=="spherical":
#         rad_directional = get_radius(x_t, vec_cent2ref=np.zeros(dim), a=a)
        
#         Gamma_limit = 2 # Multiple of  max(a) (>1)
#         rad_range = np.max(a)*Gamma_limit - rad_directional
#         relative_dist = (LA.norm(x_t)-rad_directional)/rad_range

#         if relative_dist < 0:
#             warnings.warn("Inside the obstacle.")
        
#         Gamma = np.sin((relative_dist-0.5)*pi) + Gamm_min
#         return Gamma


def getGammmaValue_ellipsoid(ob, x_t, relativeDistance=True):
    if relativeDistance:
        return np.sum( (x_t/np.tile(ob.a, (x_t.shape[1],1)).T) **(2*np.tile(ob.p, (x_t.shape[1],1) ).T ), axis=0)
    else:
        return np.sum( (x_t/np.tile(ob.a, (x_t.shape[1],1)).T) **(2*np.tile(ob.p, (x_t.shape[1],1) ).T ), axis=0)

def get_radius_ellipsoid(x_t, a=[], ob=[]):
    # Derivation from  x^2/a^2 + y^2/b^2 = 1
    
    if not np.array(a).shape[0]:
        a = ob.a

    if x_t[0]: # nonzero value
        rat_x1_x2 = x_t[1]/x_t[0]
        x_1_val = np.sqrt(1./(1./a[0]**2+1.*rat_x1_x2**2/a[1]**2))
        return x_1_val*np.sqrt(1+rat_x1_x2**2)
    else:
        return a[1]

def get_radius(vec_point2ref, vec_cent2ref=[], a=[], obs=[]):
    dim = 2 # TODO higher dimensions

    if not np.array(vec_cent2ref).shape[0]:
        vec_cent2ref = np.array(obs.reference_point) - np.array(obs.center_position)
        
    if not np.array(a).shape[0]:
        a = obs.axes_length

    if obs.th_r:
        vec_cent2ref = np.array(obs.rotMatrix).T @ vec_cent2ref
        vec_point2ref = np.array(obs.rotMatrix).T @ vec_point2ref
        
        
    if not LA.norm(vec_cent2ref): # center = ref
        return get_radius_ellipsoid(vec_point2ref, a)
    
    dir_surf_cone = np.zeros((dim, 2))
    rad_surf_cone = np.zeros((2))

    if np.cross(vec_point2ref, vec_cent2ref) > 0:
        # 2D vectors pointing in opposite direction
        dir_surf_cone[:, 0] = vec_cent2ref
        rad_surf_cone[0] = np.abs(get_radius_ellipsoid(dir_surf_cone[:, 0], a)-LA.norm(vec_cent2ref))
        
        dir_surf_cone[:, 1] = -1*np.array(vec_cent2ref)
        rad_surf_cone[1] = (get_radius_ellipsoid(dir_surf_cone[:, 1], a)+LA.norm(vec_cent2ref))
 
    else:
        dir_surf_cone[:, 0] = -1*np.array(vec_cent2ref)
        # import pdb; pdb.set_trace() ## DEBUG ##
        
        rad_surf_cone[0] = (get_radius_ellipsoid(dir_surf_cone[:, 0], a)+LA.norm(vec_cent2ref))
        
        dir_surf_cone[:, 1] = vec_cent2ref
        rad_surf_cone[1] = np.abs(get_radius_ellipsoid(dir_surf_cone[:, 1], a)-LA.norm(vec_cent2ref))

    # color_set = ['g', 'r']
    # for i in range(2):
        # plt.plot([obs.center_dyn[0], obs.center_dyn[0]+dir_surf_cone[0,i]], [obs.center_dyn[1], obs.center_dyn[1]+dir_surf_cone[1,i]], color_set[i])
    # plt.show()

    ang_tot = pi/2
    for ii in range(12): # n_iter
        rotMat = np.array([[np.cos(ang_tot), np.sin(ang_tot)],
                           [-np.sin(ang_tot), np.cos(ang_tot)]])

        # vec_ref2dir = rotMat @ dir_surf_cone[:, 0]
        # vec_ref2dir /= LA.norm(vec_ref2dir) # nonzero value expected
        # rad_ref2 = get_radius_ellipsoid(vec_ref2dir, a)
        # vec_ref2surf = rad_ref2*vec_ref2dir - vec_cent2ref

        vec_cent2dir = rotMat @ dir_surf_cone[:, 0]
        vec_cent2dir /= LA.norm(vec_cent2dir) # nonzero value expected
        rad_ref2 = get_radius_ellipsoid(vec_cent2dir, a)
        vec_ref2surf = rad_ref2*vec_cent2dir - vec_cent2ref

        crossProd = np.cross(vec_ref2surf, vec_point2ref)
        if crossProd < 0:
            # dir_surf_cone[:, 0] = vec_ref2dir
            dir_surf_cone[:, 0] = vec_cent2dir
            rad_surf_cone[0] = LA.norm(vec_ref2surf)
        elif crossProd==0: # how likely is this lucky guess? 
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

# def get_radius(vec_ref2point, vec_cent2ref=[], a=[], obs=[]):
#     dim = 2 # TODO higher dimensions

#     if not np.array(vec_cent2ref).shape[0]:
#         vec_cent2ref = np.array(obs.center_dyn) - np.array(obs.x0)
        
#     if not np.array(a).shape[0]:
#         a = obs.a
        
#     if not LA.norm(vec_cent2ref): # center = ref
#         return get_radius_ellipsoid(vec_ref2point, a)
    
#     dir_surf_cone = np.zeros((dim, 2))
#     rad_surf_cone = np.zeros((2))

#     if np.cross(vec_ref2point, vec_cent2ref) > 0:
#         dir_surf_cone[:, 0] = vec_cent2ref
#         rad_surf_cone[0] = get_radius_ellipsoid(dir_surf_cone[:, 0], a)-LA.norm(vec_cent2ref)
        
#         dir_surf_cone[:, 1] = -1*np.array(vec_cent2ref)
#         rad_surf_cone[1] = get_radius_ellipsoid(dir_surf_cone[:, 1], a)+LA.norm(vec_cent2ref)
 
#     else:
#         dir_surf_cone[:, 0] = -1*np.array(vec_cent2ref)
#         rad_surf_cone[0] = get_radius_ellipsoid(dir_surf_cone[:, 0], a)+LA.norm(vec_cent2ref)
        
#         dir_surf_cone[:, 1] = vec_cent2ref
#         rad_surf_cone[1] = get_radius_ellipsoid(dir_surf_cone[:, 1], a)-LA.norm(vec_cent2ref)
    
#     ang_tot = pi/2
#     for ii in range(12): # n_iter
#         rotMat = np.array([[np.cos(ang_tot), np.sin(ang_tot)],
#                            [-np.sin(ang_tot), np.cos(ang_tot)]])

#         vec_ref2dir = rotMat @ dir_surf_cone[:, 0]
        
#         vec_ref2dir /= LA.norm(vec_ref2dir) # nonzero value expected
        
#         rad_ref2 = get_radius_ellipsoid(vec_ref2dir, a)
#         vec_ref2surf = rad_ref2*vec_ref2dir - vec_cent2ref

#         if np.cross(vec_ref2surf, vec_ref2point)==0: # how likely is this lucky guess? 
#             return LA.norm(vec_ref2surf)
#         elif np.cross(vec_ref2surf, vec_ref2point) < 0:
#             dir_surf_cone[:, 0] = vec_ref2dir
#             rad_surf_cone[0] = LA.norm(vec_ref2surf)
#         else:
#             dir_surf_cone[:, 1] = vec_ref2dir
#             rad_surf_cone[1] = LA.norm(vec_ref2surf)

#         ang_tot /= 2.0
#     return np.mean(rad_surf_cone)


def findRadius(ob, direction, a = [], repetition = 6, steps = 10):
    # NOT SURE IF USEFULL -- NORMALLY x = Gamma*Rad
    # TODO check
    if not len(a):
        a = [np.min(ob.a), np.max(ob.a)]
        # a = obs.a
        
    # repetition
    for ii in range(repetition):
        if a[0] == a[1]:
            return a[0]
        
        magnitudeDir = np.linspace(a[0], a[1], num=steps)
        Gamma = getGammmaValue_ellipsoid(ob, np.tile(direction, (steps,1)).T*np.tile(magnitudeDir, (np.array(ob.x0).shape[0],1)) )

        if np.sum(Gamma==1):
            return magnitudeDir[np.where(Gamma==1)]
        posBoundary = np.where(Gamma<1)[0][-1]

        a[0] = magnitudeDir[posBoundary]
        posBoundary +=1
        while Gamma[posBoundary]<=1:
            posBoundary+=1

        a[1] = magnitudeDir[posBoundary]
        
    return (a[0]+a[1])/2.0


def findBoundaryPoint(ob, direction):
    # Numerical search -- TODO analytic
    dirNorm = LA.norm(direction,2)
    if dirNorm:
        direction = direction/dirNorm
    else:
        print('No feasible direction is given')
        return ob.x0

    a = [np.min(x0.a), np.max(x0.a)]
    
    return (a[0]+a[1])/2.0*direction + x0


def compute_eigenvalueMatrix(Gamma, rho=1, dim=2, radialContuinity=True):
    if radialContuinity:
        Gamma = np.max([Gamma, 1])
        
    delta_lambda = 1./np.abs(Gamma)**(1/rho)
    lambda_referenceDir = 1-delta_lambda
    lambda_tangentDir = 1+delta_lambda

    return np.diag(np.hstack((lambda_referenceDir, np.ones(dim-1)*lambda_tangentDir)) )


def compute_weights(distMeas, N=0, distMeas_lowerLimit=1, weightType='inverseGamma', weightPow=2):
    # UNTITLED5 Summary of this function goes here
    #   Detailed explanation goes here

    distMeas = np.array(distMeas)
    n_points = distMeas.shape[0]
    
    critical_points = distMeas <= distMeas_lowerLimit
    
    if np.sum(critical_points): # at least one
        if np.sum(critical_points)==1:
            w = critical_points*1.0
            return w
        else:
            # TODO: continuous weighting function
            warnings.warn('Implement continuity of weighting function.')
            w = critical_points*1./np.sum(critical_points)
            return w
        
    if weightType == 'inverseGamma':
        distMeas = distMeas - distMeas_lowerLimit
        w = 1/distMeas**weightPow
        w = w/np.sum(w) # Normalization

    else:
        warnings.warn("Unkown weighting method.")

    return w


def compute_R(d, th_r):
    if th_r == 0:
        rotMatrix = np.eye(d)
    # rotating the query point into the obstacle frame of reference
    if d==2:
        rotMatrix = np.array([[np.cos(th_r), -np.sin(th_r)],
                              [np.sin(th_r),  np.cos(th_r)]])
    elif d==3:
        R_x = np.array([[1, 0, 0,],
                        [0, np.cos(th_r[0]), np.sin(th_r[0])],
                        [0, -np.sin(th_r[0]), np.cos(th_r[0])] ])

        R_y = np.array([[np.cos(th_r[1]), 0, -np.sin(th_r[1])],
                        [0, 1, 0],
                        [np.sin(th_r[1]), 0, np.cos(th_r[1])] ])

        R_z = np.array([[np.cos(th_r[2]), np.sin(th_r[2]), 0],
                        [-np.sin(th_r[2]), np.cos(th_r[2]), 0],
                        [ 0, 0, 1] ])

        rotMatrix = R_x.dot(R_y).dot(R_z)
    else:
        warnings.warn('rotation not yet defined in dimensions d > 3 !')
        rotMatrix = np.eye(d)

    return rotMatrix


def obs_check_collision_2d(obs_list, XX, YY):
    d = 2

    dim_points = XX.shape
    if len(dim_points)==1:
        N_points = dim_points[0]
    else:
        N_points = dim_points[0]*dim_points[1]

    # No obstacles
    if not len(obs_list):
        return np.ones((dim_points))
        
    points = np.array(([np.reshape(XX,(N_points,)) , np.reshape(YY, (N_points,)) ] ))
    # At the moment only implemented for 2D
    collision = np.zeros( dim_points )

    N_points = points.shape[1]

    noColl = np.ones((1,N_points), dtype=bool)

    for it_obs in range(len(obs_list)):
        # on the surface, we have: \Gamma = \sum_{i=1}^d (xt_i/a_i)^(2p_i) == 1
        R = compute_R(d,obs_list[it_obs].th_r)

        # Gamma = np.sum( ( 1/obs_list[it_obs].sf * R.T @ (points - np.tile(np.array([obs_list[it_obs].x0]).T,(1,N_points) ) ) / np.tile(np.array([obs_list[it_obs].a]).T, (1, N_points)) )**(np.tile(2*np.array([obs_list[it_obs].p]).T, (1,N_points)) ), axis=0 )

        Gamma = np.zeros(N_points)
        for ii in range(N_points):
            Gamma[ii] = obs_list[it_obs].get_gamma(points[:,ii], in_global_frame=True)
            
        noColl = (noColl* Gamma>1)

    return np.reshape(noColl, dim_points)


def obs_check_collision(points, obs_list=[]):
    # No obstacles
    if len(obs_list) == 0:
        return

    dim = points.shape[0]
    N_points = points.shape[1]

    # At the moment only implemented for 2D
    collision = np.zeros((N_points))

    noColl = np.ones((1,N_points))

    for it_obs in range(len(obs_list)):
        # \Gamma = \sum_{i=1}^d (xt_i/a_i)^(2p_i) = 1
        R = compute_R(dim,obs_list[it_obs].th_r)

        Gamma = sum( ( 1/obs_list[it_obs].sf * R.T @ (points - np.tile(np.array([obs_list[it_obs].x0]).T,(1,N_points) ) ) / np.tile(np.array([obs_list[it_obs].a]).T, (1, N_points)) )**(np.tile(2*np.array([obs_list[it_obs].p]).T, (1,N_points)) ) )

        noColl = (noColl* Gamma>1)

    return noColl


# def linearAttractor(x, x0=False, k_factor=1.0):
#     x = np.array(x)
    
#     if type(x0)==bool:
#         x0 = np.zeros(x.shape)
        
#     return (x0-x)*k_factor
    
