#!/usr/bin/python3
'''
Angle math for python
Helper function for directional & angle evaluations

@author Lukas Huber
@date 2019-11-15
'''

import numpy as np
import warnings
from math import pi


def angle_is_in_between(angle_test, angle_low, angle_high, margin=1e-9):
    ''' Verify if angle_test is in between angle_low & angle_high
    Values are between [0, 2pi]. An absolute margin seems appropriate '''
    delta_low = angle_difference_directional_2pi(angle_test, angle_low)
    delta_high = angle_difference_directional_2pi(angle_high, angle_test)

    delta_tot = angle_difference_directional_2pi(angle_high, angle_low)

    return (np.abs((delta_high+delta_low)-delta_tot) < margin)


def angle_modulo(angle):
    ''' Get angle in [-pi, pi[  '''
    return ((angle+pi) % (2*pi)) - pi


def angle_difference_directional_2pi(angle1, angle2):
    angle_diff = (angle1-angle2)
    while angle_diff > 2*pi:
        angle_diff -= 2*pi
    while angle_diff < 0:
        angle_diff += 2*pi
    return angle_diff

    
def angle_difference_directional(angle1, angle2):
    '''
    Difference between two angles ]-pi, pi]
    Note: angle1-angle2 (non-commutative)
    '''
    angle_diff = (angle1-angle2)
    while angle_diff > pi:
        angle_diff = angle_diff-2*pi
    while angle_diff <= -pi:
        angle_diff = angle_diff+2*pi
    return angle_diff

def angle_difference(angle1, angle2):
    return angle_difference_directional(angle1, angle2)

def angle_difference_abs(angle1, angle2):
    '''
    Difference between two angles [0,pi[
    angle1-angle2 = angle2-angle1(commutative)
    '''
    angle_diff = np.abs(angle2-angle1)
    while angle_diff >= pi:
        angle_diff = 2*pi-angle_diff
    return angle_diff

def transform_polar2cartesian(magnitude, angle, center_position=[0,0], center_point=None):
    # Only 2D input

    if not isinstance(center_point, type(None)):
        # TODO remove center_position or center_position
        center_position = center_point
    magnitude = np.reshape(magnitude, (-1))
    angle = np.reshape(angle, (-1))
    
    # points = [r, phi]
    points = (magnitude * np.vstack((np.cos(angle), np.sin(angle)))
              + np.tile(center_position, (magnitude.shape[0],1)).T )
    return np.squeeze(points)


def transform_cartesian2polar(points, center_position=None, second_axis_is_dim=True):
    '''
    Two dimensional transformation of cartesian to polar coordinates
    Based on center_position (default value center_position=np.zeros(dim))
    '''
    # TODO -- check dim and etc
    # Don't just squeeze, maybe...
    
    # if type(center_position)==type(None):
        # center_position = np.zeros(self.dim)

    points = np.squeeze(points)
    if second_axis_is_dim:
        points = points.T
    dim = points.shape[0]

    if isinstance(center_position, type(None)):
        center_position = np.zeros(dim)
    else:
        center_position = np.squeeze(center_position)
    
    if len(points.shape)==1:
        points = points - center_position
                
        angle = np.arctan2(points[1], points[0])        
    else:
        points = points - np.tile(center_position, (points.shape[1], 1)).T
        angle = np.arctan2(points[1,:], points[0,:])
        
    magnitude = np.linalg.norm(points, axis=0)

    # output: [r, phi]
    return magnitude, angle


def get_orthogonal_basis(vector, normalize=True):
    '''
    Orthonormal basis for a vector
    '''
    if isinstance(vector, list):
        vector = np.array(vector)
    elif not isinstance(vector, np.ndarray):
        raise TypeError("Wrong input type vector")

    if normalize:
        v_norm = np.linalg.norm(vector)
        if v_norm:
            vector = vector / v_norm
        else:
            raise ValueError("Orthogonal basis Matrix not defined for 0-direction vector.")

    dim = vector.shape[0]
    basis_matrix = np.zeros((dim, dim))

    if dim == 2:
        basis_matrix[:, 0] = vector
        basis_matrix[:, 1] = np.array([basis_matrix[1, 0],
                                       -basis_matrix[0, 0]])
    elif dim == 3:
        basis_matrix[:, 0] = vector
        basis_matrix[:, 1] = np.array([-vector[1], vector[0], 0])
        
        norm_vec2 = np.linalg.norm(basis_matrix[:, 1])
        if norm_vec2:
            basis_matrix[:, 1] = basis_matrix[:, 1] / norm_vec2
        else:
            basis_matrix[:, 1] = [1, 0, 0]
            
        basis_matrix[:, 2] = np.cross(basis_matrix[:, 0], basis_matrix[:, 1])
        
        norm_vec = np.linalg.norm(basis_matrix[:, 2])
        if norm_vec:
            basis_matrix[:, 2] = basis_matrix[:, 2] / norm_vec
        
    elif dim > 3: # TODO: general basis for d>3
        basis_matrix[:, 0] = vector
        for ii in range(1,dim):
            # TODO: higher dimensions
            if vector[ii]: # nonzero
                basis_matrix[:ii, ii] = vector[:ii]
                basis_matrix[ii, ii] = (-np.sum(vector[:ii]**2)/vector[ii])
                basis_matrix[:ii+1, ii] = basis_matrix[:ii+1, ii]/np.linalg.norm(basis_matrix[:ii+1, ii])
            else:
                basis_matrix[ii, ii] = 1
            # basis_matrix[dim-(ii), ii] = -np.dot(vector[:dim-(ii)], vector[:dim-(ii)])
            # basis_matrix[:, ii] = basis_matrix[:, ii]/LA.norm(basis_matrix[:, ii])

        # import pdb; pdb.set_trace() ## DEBUG ##
        # raise ValueError("Not implemented for d>3")
        # warnings.warn("Implement higher dimensionality than d={}".format(dim))
    return basis_matrix


def get_angle_space(reference_direction, directions, normalize=True):
    """
    Get angle space transformation
    """
    dim = np.array(reference_direction).shape[0]
    
    if len(directions.shape)==1:
        num_dirs = None
        directions = directions.reshape(dim, 1)
    else:
        num_dirs = directions.shape[1]
        
    reference_direction = np.copy(reference_direction)
    directions = np.copy(directions)

    if normalize:
        norm_refDir = np.linalg.norm(reference_direction)
        if norm_refDir==0: # nonzero
            raise ValueError("Zero norm direction as input")
        reference_direction /= norm_refDir
        
        norm_dir = np.linalg.norm(directions, axis=0)
        ind_nonzero = (norm_dir>0)
        directions[:, ind_nonzero] = directions[:, ind_nonzero]/np.tile(norm_dir[ind_nonzero], (dim, 1))

    OrthogonalBasisMatrix = get_orthogonal_basis(reference_direction)

    directions_referenceSpace = np.zeros(np.shape(directions))
    for ii in range(np.array(directions).shape[1]):
        directions_referenceSpace[:,ii] = OrthogonalBasisMatrix.T.dot( directions[:,ii])

    directions_referenceSpace = np.zeros(np.shape(directions))
    for ii in range(np.array(directions).shape[1]):
        directions_referenceSpace[:,ii] = OrthogonalBasisMatrix.T.dot( directions[:,ii])

    directions_directionSpace = directions_referenceSpace[1:, :]

    norm_dirSpace = np.linalg.norm(directions_directionSpace, axis=0)
    ind_nonzero = (norm_dirSpace > 0)

    directions_directionSpace[:,ind_nonzero] = (directions_directionSpace[:, ind_nonzero] /  np.tile(norm_dirSpace[ind_nonzero], (dim-1, 1)))

    cos_directions = directions_referenceSpace[0,:]
    if np.sum(cos_directions > 1) or np.sum(cos_directions < -1):
        cos_directions = np.min(np.vstack((cos_directions, np.ones(num_dirs))), axis=0)
        cos_directions = np.max(np.vstack((cos_directions, -np.ones(num_dirs))), axis=0)
        warnings.warn("Cosinus value out of bound.")

    directions_directionSpace *= np.tile(np.arccos(cos_directions), (dim-1, 1))

    directions_directionSpace *= (-1) # in 2D for convention

    if num_dirs is None:
        directions_directionSpace = np.squeeze(directions_directionSpace)
    return directions_directionSpace


def get_directional_weighted_sum(reference_direction, directions, weights, total_weight=1, normalize=True, normalize_reference=True):
    '''
    Weighted directional mean for inputs vector ]-pi, pi[ with respect to the reference_direction

    # INPUT
    reference_direction: basis direction for the angle-frame
    directions: the directions which the weighted sum is taken from
    weights: used for weighted sum
    total_weight: [<=1] 
    normalize: 

    # OUTPUT 
    
    '''
    # TODO remove obs and position
    ind_nonzero = (weights>0) # non-negative

    reference_direction = np.copy(reference_direction)
    directions = directions[:, ind_nonzero] 
    weights = weights[ind_nonzero]

    if total_weight<1:
        weights = weights/np.sum(weights) * total_weight

    n_directions = weights.shape[0]
    if (n_directions==1) and total_weight>=1:
        return directions[:, 0]

    dim = np.array(reference_direction).shape[0]

    if normalize_reference:
        norm_refDir = np.linalg.norm(reference_direction)
        if norm_refDir==0: # nonzero
            raise ValueError("Zero norm direction as input")
        reference_direction /= norm_refDir

     # TODO - higher dimensions
    if normalize:
        norm_dir = np.linalg.norm(directions, axis=0)
        ind_nonzero = (norm_dir>0)
        directions[:, ind_nonzero] = directions[:, ind_nonzero]/np.tile(norm_dir[ind_nonzero], (dim, 1))

    OrthogonalBasisMatrix = get_orthogonal_basis(reference_direction)

    directions_referenceSpace = np.zeros(np.shape(directions))
    for ii in range(np.array(directions).shape[1]):
        directions_referenceSpace[:,ii] = OrthogonalBasisMatrix.T.dot( directions[:,ii])

    directions_directionSpace = directions_referenceSpace[1:, :]

    norm_dirSpace = np.linalg.norm(directions_directionSpace, axis=0)
    ind_nonzero = (norm_dirSpace > 0)

    directions_directionSpace[:,ind_nonzero] = (directions_directionSpace[:, ind_nonzero] /  np.tile(norm_dirSpace[ind_nonzero], (dim-1, 1)))

    cos_directions = directions_referenceSpace[0,:]
    if np.sum(cos_directions > 1) or np.sum(cos_directions < -1):
        # Numerical error correction
        cos_directions = np.min(np.vstack((cos_directions, np.ones(n_directions))), axis=0)
        cos_directions = np.max(np.vstack((cos_directions, -np.ones(n_directions))), axis=0)
        # warnings.warn("Cosinus value out of bound.") 

    directions_directionSpace *= np.tile(np.arccos(cos_directions), (dim-1, 1))

    direction_dirSpace_weightedSum = np.sum(directions_directionSpace* np.tile(weights, (dim-1, 1)), axis=1)

    norm_directionSpace_weightedSum = np.linalg.norm(direction_dirSpace_weightedSum)

    if norm_directionSpace_weightedSum:
        direction_weightedSum = (OrthogonalBasisMatrix.dot(
                                  np.hstack((np.cos(norm_directionSpace_weightedSum),
                                              np.sin(norm_directionSpace_weightedSum) / norm_directionSpace_weightedSum * direction_dirSpace_weightedSum)) ))
    else:
        direction_weightedSum = OrthogonalBasisMatrix[:,0]

    return direction_weightedSum


def periodic_weighted_sum(angles, weights, reference_angle=None):
    '''Weighted Average of angles (1D)'''
    # TODO: unify with directional_weighted_sum() // see above
    # Extend to dimenions d>2
    if isinstance(angles, list): 
        angles = np.array(angles)
    if isinstance(weights, list): 
        weights = np.array(weights)

    
    if reference_angle is None:
        if len(angles)>2:
            raise NotImplementedError("No mean defined for periodic function with more than two angles.")
        reference_angle = angle_difference_directional(angles[0], angles[1])/2.0 + angles[1]
        reference_angle = angle_modulo(reference_angle)

    angles = angle_modulo(angles-reference_angle)
    
    mean_angle = angles.T.dot(weights)
    mean_angle = angle_modulo(mean_angle + reference_angle)

    return mean_angle
