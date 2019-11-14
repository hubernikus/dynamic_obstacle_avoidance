#!/usr/bin/python3

'''
Angle math for python

@author Lukas Huber
@date 2019-11-15
'''

import numpy as np
from math import pi

def angle_modulo(angle):
    '''
    Get angle in [-pi, pi[ 
    '''
    return ((angle+pi) % (2*pi)) - pi

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

def angle_difference_abs(angle1, angle2):
    '''
    Difference between two angles [0,pi[
    (commutative)
    '''
    angle_diff = np.babs(angle2-angle1)
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

def get_directional_weighted_sum(reference_direction, directions, weights, normalize=True, normalize_reference=True, obs=None, position=[]):
    '''
    Weighted directional mean for inputs vector ]-pi, pi[ with respect to the reference_direction
    '''
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
