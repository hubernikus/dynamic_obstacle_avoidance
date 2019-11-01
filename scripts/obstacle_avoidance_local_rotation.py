#!/usr/bin/python3

'''
Dynamic Simulation - Obstacle Avoidance Algorithm

@author LukasHuber
@date 2018-05-24

'''

import matplotlib.pyplot as plt
plt.ion()
plt.close('all')


import numpy as np

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *

def ds_init(x, attractor=np.array([0,0]), max_vel=0.5, slow_down_region=0.5):
    vel = attractor-x

    dist = np.linalg.norm(vel)
    if dist < slow_down_region:
        max_vel = max_vel*dist/slow_down_region
        
    norm_vel = np.linalg.norm(vel)
    if norm_vel>max_vel:
        vel = vel/norm_vel*max_vel
        
    return vel

def get_local_weight(r, r_ref, r_boundary=1):
    delta_r = r-r_ref

    if delta_r>r_boundary:
        return 0
    else:
        return (r_boundary-delta_r)/r_boundary

obstacle = StarshapedFlower(orientation=40/180*pi,
                            radius_magnitude=1, radius_mean=2, number_of_edges=4)
obstacle.draw_obstacle()

dim = 2
n_steps = 1000

n_points = 20
x_range = [-2,4]
y_range = [-7, 7]
x_inits = np.vstack(([x_range[1]*np.ones(n_points),
                      np.linspace(y_range[0], y_range[1], n_points)]))
# x_inits = 4*np.array([[np.cos(rot)],
                      # [np.sin(rot)]])
dt = 0.03

attractor = np.array([-6.0, -1])


for it_point in range(x_inits.shape[1]):
    positions = np.zeros((dim, n_steps))
    velocities = np.zeros((dim, n_steps))
    vel_init = np.zeros((dim, n_steps))
    angle2tangent = np.zeros(n_steps)

    x_init = x_inits[:, it_point]

    ii = 0
    positions[:, ii] = x_init

    vortex_direction = 0

    first_entrance=True
    free_sight = False

    for ii in range(n_steps-1):
        vel_init[:, ii] = ds_init(positions[:, ii], attractor=attractor)

        # print('position', positions[:, ii])
        # print('vel init', vel_init[:, ii])
        # print('attractor', attractor)
        
        if free_sight:
            velocities[:, ii] = vel_init[:, ii]
            positions[:, ii+1] = positions[:, ii] + velocities[:, ii]*dt
        
        check_step = 0.1
        n_checks = np.linalg.norm(positions[:,ii]-attractor)/check_step
        check_points = np.vstack((np.linspace(positions[0,ii], attractor[0], n_checks),
                                  np.linspace(positions[1,ii], attractor[1], n_checks)))

        # free_sight = True
        for jj in range(check_points.shape[1]):
            if obstacle.get_gamma(check_points[:,jj], in_global_frame=True)<1:
                free_sight = False
                break

        if free_sight:
            velocities[:, ii] = vel_init[:, ii]
            positions[:, ii+1] = positions[:, ii] + velocities[:, ii]*dt
            continue
        
        normal = obstacle.get_normal_direction(positions[:, ii], in_global_frame=True)
        tangent = np.array([-normal[1], normal[0]])

        mag, ang = transform_cartesian2polar(positions[:, ii], obstacle.reference_point)

        dist2center = np.linalg.norm(positions[:, ii]-obstacle.reference_point)
        local_radius = obstacle.get_radius_of_angle(ang, in_global_frame=True)

        weight = get_local_weight(dist2center, local_radius)
        # print('dist2center', dist2center)
        # print('dist2center', local_radius)
        # print('weight', weight)
        
        if weight > 0:
            if first_entrance:
                first_entrance = False
                obstacle_dir = obstacle.reference_point - attractor
                point_dir = positions[:, ii] - attractor

                choose_vortex = True
                if choose_vortex:
                    if np.cross(obstacle_dir, point_dir) > 0:
                        vortex_direction = 1
                    else:
                        vortex_direction = -1

            tangent = tangent*vortex_direction
            angle_tangent = np.arctan2(tangent[1], tangent[0])
            # angle_tang2vel = np.arctan2(vel_init[1, ii], vel_init[0, ii])
            angle_vel_init = np.arctan2(vel_init[1, ii], vel_init[0, ii])

            if first_entrance:
                # angle2tangent[ii] = angle_tang2vel
                angle2tangent[ii] = angle_difference_directional(angle_tangent, angle_vel_init)
            else:
                angle_tang2vel = np.copysign(
                    np.arccos(np.dot(vel_init[:, ii], tangent)/
                              (np.linalg.norm(vel_init[:, ii])*np.linalg.norm(tangent))),
                    np.cross(vel_init[:, ii], tangent))
                angle2tangent[ii] = angle_difference_directional(angle_tangent, angle_vel_init)
                # delta_angle2tangent = angle_difference_directional(angle2tangent[ii-1], angle_tang2vel)
                # angle2tangent[ii] = delta_angle2tangent+angle2tangent[ii-1]

            angle_desired = angle2tangent[ii]*weight + angle_vel_init
            velocities[:, ii] = np.linalg.norm(vel_init[:, ii]) \
                                  * np.array([np.cos(angle_desired), np.sin(angle_desired)])

            if False:
                plt.close('all')
                plt.figure()
                plt.plot(obstacle.x_obs[:,0], obstacle.x_obs[:,1])
                plt.plot(positions[0, :ii], positions[1, :ii])
                plt.axis('equal')
                plt.show()
        else:
            velocities[:, ii] = vel_init[:, ii]
        positions[:, ii+1] = positions[:, ii] + velocities[:, ii]*dt

        if np.linalg.norm(positions[:,ii+1] - positions[:, ii]) < 1e-4:
            positions = np.delete(positions, np.arange(ii, positions.shape[1]), axis=1)
            print('converged')
            break

        # if np.linalg.norm(positions[:,ii+1] - attractor) < 1e-1 \
           # or np.linalg.norm(positions[:,ii+1] - obstacle.reference_point) < 1e-1:
        # print('converged')            


    if it_point==0:
        plt.figure()
        plt.plot(obstacle.x_obs[:,0], obstacle.x_obs[:,1], 'k')
        plt.plot(obstacle.reference_point[0], obstacle.reference_point[1], 'k+')
        plt.axis('equal')
        plt.plot(attractor[0], attractor[1], 'k*')
        plt.show()
        
    # plt.plot(positions[0, :], positions[1, :], '.')
    plt.plot(positions[0, :], positions[1, :])
plt.show()    
 
