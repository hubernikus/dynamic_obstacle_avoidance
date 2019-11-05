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
