#!/usr/bin/python3
'''
Test quaternion

@author Lukas Huber
@date 2019-11-15
'''

# TODO: explore further the connection of 'angle'-space & trafo
import quaternion

import numpy as np
import matplotlib.pyplot as plt

from math import pi
from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import *

vec0 = [1, 0, 0]

quat0 = quaternion.from_euler_angles(0*pi/180, 0, 0)
# quat1 = quaternion.from_euler_angles(-30*pi/180, 0, 0)
# quat2 = quaternion.from_euler_angles(30*pi/180, 0, 0)
quat1 = quaternion.from_euler_angles(-90*pi/180, 0, 0)
quat2 = quaternion.from_euler_angles(90*pi/180, 0, 0)
quat3 = quaternion.from_euler_angles(0, -30*pi/180, 0)
quat4 = quaternion.from_euler_angles(0, 30*pi/180, 0)
# quat3 = quaternion.from_euler_angles(-30*pi/180, 0, 0)
# quat4 = quaternion.from_euler_angles(30*pi/180, 0, 0)

vec_final1 = quaternion.rotate_vectors(quat1, vec0)
vec_final2 = quaternion.rotate_vectors(quat2, vec0)

quat0_array = quaternion.as_float_array(quat0)
quat1_array = quaternion.as_float_array(quat1)
quat2_array = quaternion.as_float_array(quat2)
quat3_array = quaternion.as_float_array(quat3)
quat4_array = quaternion.as_float_array(quat4)

quat_matr = np.vstack((quat1_array, quat2_array, quat3_array, quat4_array)).T
weights = np.ones(quat_matr.shape[1])/quat_matr.shape[1]

quat_mean = get_directional_weighted_sum(reference_direction=quat0_array, directions=quat_matr, weights=weights)

quat_mean = quaternion.from_float_array(quat_mean)
vec_mean =  quaternion.rotate_vectors(quat_mean, vec0)
print('vec_mean 4', vec_mean)

two_vector_slerp = False
if two_vector_slerp:
    # Two vector 
    quat_matr = np.vstack((quat1_array, quat2_array)).T
    # weights = np.array([1./2, 1./2])

    plt.figure()
    n_arrow = 10
    for ii in range(n_arrow+1):
        weights = np.array([ii/n_arrow, (n_arrow-ii)/n_arrow])
        print('weights', weights )

        quat_mean = get_directional_weighted_sum(reference_direction=quat0_array,
                                                 directions=quat_matr,
                                                 weights=weights)

        quat_mean = quaternion.from_float_array(quat_mean)
        vec_mean =  quaternion.rotate_vectors(quat_mean, vec0)
        print('vec_mean', vec_mean)

        plt.plot([0, vec_mean[0]], [0,vec_mean[1]] )

    plt.ion()
    plt.show()
    plt.axis('equal')

