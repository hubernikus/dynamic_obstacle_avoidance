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
from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import *

import matplotlib.pyplot as plt


def compute_modulation_matrix(x_t, obs, R, matrix_singularity_margin=pi/2.0*1.05):
    '''
     The function evaluates the gamma function and all necessary components needed to construct the modulation function, to ensure safe avoidance of the obstacles.
    Beware that this function is constructed for ellipsoid only, but the algorithm is applicable to star shapes.
    
    Input
    x_t [dim]: The position of the robot in the obstacle reference frame
    obs [obstacle class]: Description of the obstacle with parameters
    R [dim x dim]: Rotation matrix 
    
    Output
    E [dim x dim]: Basis matrix with rows the reference and tangent to the obstacles surface
    D [dim x dim]: Eigenvalue matrix which is responsible for the modulation
    Gamma [dim]: Distance function to the obstacle surface (in direction of the reference vector)
    E_orth [dim x dim]: Orthogonal basis matrix with rows the normal and tangent
    '''

    dim = obs.dim
    
    if hasattr(obs, 'rho'):
        rho = np.array(obs.rho)
    else:
        rho = 1

    Gamma = obs.get_gamma(x_t, in_global_frame=False) # function for ellipsoids
    
    normal_vector = obs.get_normal_direction(x_t, in_global_frame=False)
    reference_direction = obs.get_reference_direction(x_t, in_global_frame=False)

    # Check if there was correct placement of reference point
    Gamma_referencePoint = obs.get_gamma(obs.reference_point, in_global_frame=False)

    # if not obs.is_boundary and Gamma_referencePoint >= 1:
        # Check what this does and COMMENT!!!!
        # import pdb; pdb.set_trace() ## DEBUG ##
        # Per default negative
        # referenceNormal_angle = np.arccos(reference_direction.T.dot(normal_vector))
        
        
        # surface_position = obs.get_obstace_radius* x_t/LA.norm(x_t)
        # direction_surface2reference = obs.get_reference_point()-surface_position

        
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

    E_orth = get_orthogonal_basis(normal_vector, normalize=True)
    E = np.copy((E_orth))
    E[:, 0] = -reference_direction
    norm_refDir = np.linalg.norm(E[:, 0])
    if norm_refDir:
        E[:, 0] = E[:, 0]/norm_refDir
        
    eigenvalue_reference, eigenvalue_tangent = calculate_eigenvalues(Gamma, is_boundary=obs.is_boundary)
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
        vec_cent2ref = np.array(obs.rotMatrix).T.dot(vec_cent2ref)
        vec_point2ref = np.array(obs.rotMatrix).T.dot(vec_point2ref)
        
        
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

        vec_cent2dir = rotMat.dot(dir_surf_cone[:, 0])
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

#         vec_ref2dir = rotMat.dot(dir_surf_cone[:, 0])
        
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
    # Detailed explanation goes here
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
        w = (1/distMeas)**weightPow
        if np.sum(w)==0:
            return w
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


def obs_check_collision(obs_list, dim, *args):
    # No obstacles
    if len(obs_list)==0:
        return np.ones(args[0].shape)
    dim = obs_list[0].dim

    if len(*args)==dim:
        points = np.array([])
        for ii in range(dim):
            input_shape = args[0].shape
            points = np.vstack((points, np.arary(args[ii]).flatten()))
            
    N_points = points.shape[1]

    # At the moment only implemented for 2D
    collision = np.zeros((N_points))

    for ii in range(N_points):
        pass
    return noColl

def obs_check_collision_ellipse(obs_list, dim, points):
    
    for it_obs in range(len(obs_list)):
        # \Gamma = \sum_{i=1}^d (xt_i/a_i)^(2p_i) = 1
        R = compute_R(dim,obs_list[it_obs].th_r)

        Gamma = sum( ( 1/obs_list[it_obs].sf * R.T.dot(points - np.tile(np.array([obs_list[it_obs].x0]).T,(1,N_points) ) ) / np.tile(np.array([obs_list[it_obs].a]).T, (1, N_points)) )**(np.tile(2*np.array([obs_list[it_obs].p]).T, (1, N_points)) ) )

        noColl = (noColl* Gamma>1)

    return noColl


def get_tangents2ellipse(edge_point, axes, dim=2):
    # TODO cut ellipse along direction
    if not dim==2:
        # TODO cut ellipse along direction
        raise TypeError("Not implemented for higher dimension")
    
    # Intersection of (x_1/a_1)^2 +( x_2/a_2)^2 = 1 & x_2=m*x_1+c
    # Solve for determinant D=0 (tangent with only one intersection point)
    A_ =  edge_point[0]**2 - axes[0]**2
    B_ = -2*edge_point[0]*edge_point[1]
    C_ = edge_point[1]**2 - axes[1]**2
    D_ = B_**2 - 4*A_*C_

    m = np.zeros(2)

    m[1] = (-B_ - np.sqrt(D_)) / (2*A_)
    m[0] = (-B_ + np.sqrt(D_)) / (2*A_)

    tangent_points = np.zeros((dim, 2))
    # normal_vectors = np.zeros((dim, 2))
    tangent_vectors = np.zeros((dim, 2))
    # normalDistance2center = np.zeros(2)
    tangent_angles = np.zeros(2)

    for ii in range(2):
        c = edge_point[1] - m[ii]*edge_point[0]

        A = (axes[0]*m[ii])**2 + axes[1]**2
        B = 2*axes[0]**2*m[ii]*c
        # D != 0 so C not interesting

        tangent_points[0, ii] = -B/(2*A)
        tangent_points[1, ii] = m[ii]*tangent_points[0, ii] + c

        tangent_vectors[:,ii] = tangent_points[:, ii]-edge_point
        tangent_vectors[:,ii] /= LA.norm(tangent_vectors[:,ii])

        # normal_vectors[:, ii] = np.array([tangent_vectors[1,ii], -tangent_vectors[0,ii]])
                                              
        # Check direction
        # normalDistance2center[ii] = normal_vectors[:, ii].T.dot(edge_point)

        # if (normalDistance2center[ii] < 0):
            # normal_vectors[:, ii] = normal_vectors[:, ii]*(-1)
            # normalDistance2center[ii] *= -1
        tangent_angles[ii] = np.arctan2(tangent_points[1,ii], tangent_points[0,ii])

    if angle_difference_directional(tangent_angles[1], tangent_angles[0]) < 0:
        tangent_points = np.flip(tangent_points, axis=1)
        tangent_vectors = np.flip(tangent_vectors, axis=1)
        
    return tangent_vectors, tangent_points


def get_intersectionWithEllipse(edge_point, direction, axes, only_positive_direction=False):
    # Intersection of (x_1/a_1)^2 +( x_2/a_2)^2 = 1 & x_2=m*x_1+c
    # Dimension
    dim = 2
    
    if direction[0]==0:
        m = 0
    else:
        m = direction[1]/direction[0]
    c = edge_point[1] - m*edge_point[0]

    A = (axes[0]*m)**2 + axes[1]**2
    B = 2*axes[0]**2*m*c
    C = (axes[0]*c)**2 + (axes[0]*axes[1])**2
    
    D = B*B - 4*A*C

    if D<0:
        return None
    
    sqrtD = np.sqrt(D)

    if only_positive_direction:
        intersection = np.zeros(dim)
        intersection[0] = (-B+sqrtD)/(2*A)
        intersection[1] = intersection[1,:]*m + c

        dist = (intersection - edge_point)/direction
        
        if any (dist<0).to_list():
            # TODO: check if really the positive direction and remove 'if'
            import pdb; pdb.set_trace() ## DEBUG ##
        return intersection
    else:
        intersections = np.zeros((dim,2))
        intersections[0,:] = np.array([(-B+sqrtD), (-B-sqrt(D))])/(2*A)
        intersections[1,:] = intersections[0,:]*m + c

        return intersections

def cut_planeWithEllispoid(reference_position, axes, plane):
    # TODO
    raise NotImplementedError()


def cut_lineWithEllipse(line_points, axes):
    raise NotImplementedError()

    
