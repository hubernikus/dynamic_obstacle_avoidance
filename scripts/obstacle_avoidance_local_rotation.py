#!/USSR/bin/python3

'''
Dynamic Simulation - Obstacle Avoidance Algorithm

@author LukasHuber
@date 2018-05-24

'''

import matplotlib.pyplot as plt
plt.ion()
plt.close('all')


import numpy as np
import warnings

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *


def windup_smoothening(angle_windup, angle):
    # Make it a oneliner? lambda-function
    # Correct the integration error
    num_windups = np.round((angle_windup - angle)/(2*pi))
    # print('num windups', num_windups)
    angle_windup = 2*pi*num_windups + angle
    return angle_windup
    
    
def ds_init(x, attractor=np.array([0,0]), max_vel=0.5, slow_down_region=0.5):
    vel = attractor-x

    dist = np.linalg.norm(vel)
    if dist < slow_down_region:
        max_vel = max_vel*dist/slow_down_region
        
    norm_vel = np.linalg.norm(vel)
    if norm_vel>max_vel:
        vel = vel/norm_vel*max_vel
        
    return vel

def get_local_weight(r, r_ref, r_boundary=2, power_weight=2):
    delta_r = r-r_ref

    if delta_r>r_boundary:
        return 0
    elif delta_r < 0:
        return ((r_boundary-delta_r)/r_boundary)**(1/power_weight)
    else:
        return ((r_boundary-delta_r)/r_boundary)**(1/power_weight)

def obs_avoidance_rk4(dt, x, obs):
    # Fourth order integration of obstacle avoidance differential equation
    # NOTE: The movement of the obstacle is considered as small, hence position and movement changed are not considered. This will be fixed in future iterations.

    if type(x0)==bool:
        x0 = np.zeros(np.array(x).shape[0])

    # k1
    xd = ds(x, x0)
    xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x, xd, obs)
    k1 = dt*xd

    # k2
    xd = ds(x+0.5*k1, x0)
    xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x+0.5*k1, xd, obs)
    k2 = dt*xd

    # k3
    xd = ds(x+0.5*k2, x0)
    xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x+0.5*k2, xd, obs)
    
    k3 = dt*xd

    # k4
    xd = ds(x+k3, x0)
    xd = velConst_attr(x, xd, x0)
    xd = obs_avoidance(x+k3, xd, obs)
    k4 = dt*xd

    # x final
    x = x + 1./6*(k1+2*k2+2*k3+k4) # + O(dt^5)

    return x

orientation_object=55

obstacle = StarshapedFlower(orientation=orientation_object/180*pi,
                            radius_magnitude=1, radius_mean=2, number_of_edges=4)
obstacle.draw_obstacle()

dim = 2
n_steps = 1000

n_points = 20
x_range = [-2,4]
y_range = [-5, -2]
x_inits = np.vstack(([x_range[1]*np.ones(n_points),
                      np.linspace(y_range[0], y_range[1], n_points)]))
# x_inits = np.array([[4],
                    # [-2]])
dt = 0.03

attractor = np.array([-6.0, -1])

fig, ax = plt.subplots()

for it_point in range(x_inits.shape[1]):
    positions = np.zeros((dim, n_steps))
    velocities = np.zeros((dim, n_steps))
    vel_init = np.zeros((dim, n_steps))

    angle_tangents_list = np.zeros(n_steps)
    angle_tangent2init = np.zeros(n_steps)
    angle_velInit = np.zeros(n_steps)
    angle_velInit_windup = np.zeros(n_steps)
    
    tangents = np.zeros((dim, n_steps))

    x_init = x_inits[:, it_point]

    ii = 0
    positions[:, ii] = x_init

    vortex_direction = 1
    choose_vortex = False

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
        try:
            check_points = np.vstack((np.linspace(positions[0,ii], attractor[0], n_checks),np.linspace(positions[1,ii], attractor[1], n_checks)))
        except:
            import pdb; pdb.set_trace() ## DEBUG ##
            

        if ii>=1 and angle_tangent2init[ii-1]*vortex_direction >= 0:
            free_sight = True
            
            for jj in range(check_points.shape[1]):
                if obstacle.get_gamma(check_points[:,jj], in_global_frame=True)<1:
                    free_sight = False
                    break

            if free_sight:
                velocities[:, ii] = vel_init[:, ii]
                positions[:, ii+1] = positions[:, ii] + velocities[:, ii]*dt
                continue
        
        normal = obstacle.get_normal_direction(positions[:, ii], in_global_frame=True)
        tangents[:, ii] = np.array([-normal[1], normal[0]])

        mag, ang = transform_cartesian2polar(positions[:, ii], obstacle.reference_point)

        dist2center = np.linalg.norm(positions[:, ii]-obstacle.reference_point)
        local_radius = obstacle.get_radius_of_angle(ang, in_global_frame=True)

        weight = get_local_weight(dist2center, local_radius)
        # print('dist2center', dist2center)
        # print('dist2center', local_radius)
        # print('weight', weight)
        
        if weight > 0:
            if first_entrance:
                obstacle_dir = obstacle.reference_point - attractor
                point_dir = positions[:, ii] - attractor

                if choose_vortex:
                    if np.cross(obstacle_dir, point_dir) > 0:
                        vortex_direction = 1
                    else:
                        vortex_direction = -1

            tangents[:, ii] = tangents[:, ii]*vortex_direction
            angle_tangents_list[ii] = np.arctan2(tangents[1, ii], tangents[0, ii])
            
            angle_velInit[ii] = np.arctan2(vel_init[1, ii], vel_init[0, ii])
            
            angle_tangent2init_ref = angle_difference_directional(angle_tangents_list[ii], angle_velInit[ii])
            angle_tangent2init_ref = angle_modulo(angle_tangent2init_ref)
            # angle_tangent2init_ref= delta_angle
            # angle_tangent2inpit_ref= angle_modulo(angle_velInit[ii]+delta_angle)
            # angle_tangent2init_ref = angle_tangents_list[ii] # TODO remove
            # angle_tangent2init_ref = angle_velInit[ii]
            # angle_tangents2init_ref = np.arctan2(tangents[1, ii], tangents[0, ii])

            if False:
                plt.quiver(positions[0, ii], positions[1, ii], np.cos(angle_tangent2init_ref+angle_velInit[ii]), np.sin(angle_tangent2init_ref+angle_velInit[ii]), color='r', scale=15)
                
                plt.quiver(positions[0, ii], positions[1, ii],
                           np.cos(angle_tangents_list[ii]),
                           np.sin(angle_tangents_list[ii]), color='g', scale=15)
                # plt.quiver(positions[0, ii], positions[1, ii], tangents[0, ii], tangents[1,ii], color='k')
                # import pdb; pdb.set_trace() ## DEBUG ##

            if first_entrance:
                angle_tangent2init[ii] = angle_tangent2init_ref
                angle_velInit_windup[ii] = angle_velInit[ii]
                first_entrance = False
            else:
                delta_angle_tangent = angle_difference_directional(angle_tangents_list[ii], angle_tangents_list[ii-1])
                delta_velInit = angle_difference_directional(angle_velInit[ii], angle_velInit[ii-1])
                
                angle_tangent2init[ii] =  angle_tangent2init[ii-1] + delta_angle_tangent - delta_velInit # continuous integration
                # import pdb; pdb.set_trace() ## DEBUG ##

                angle_tangent2init[ii] = windup_smoothening(angle_tangent2init[ii], angle_tangent2init_ref)
                # angle_tangent2init[ii] = windup_smoothening(0, angle_tangent2init_ref) # TODO remove
                
                # angle_velInit_windup[ii] = angle_velInit[ii] + delta_velInit
                # angle_velInit_windup[ii] = windup_smoothening(angle_velInit_windup[ii], angle_velInit[ii])
                
                # angle_velInit_list[ii] = np.arctan2(vel_init[1, ii], vel_init[0, ii])
                # angle_tangents2Init[ii] = windup_smoothening(angle_tangents2Init[ii], angle_tangent)
            angle_desired = angle_tangent2init[ii]*weight + angle_velInit[ii]
            
            velocities[:, ii] = np.linalg.norm(vel_init[:, ii])  \
                                  * np.array([np.cos(angle_desired), np.sin(angle_desired)])

            if False:
            # if  np.linalg.norm(positions[:, ii]-np.array([1.34, -0.62]))<0.1:
                plt.plot(obstacle.x_obs[:,0], obstacle.x_obs[:,1])
                plt.plot(positions[0, :ii], positions[1, :ii])
                plt.axis('equal')
                plt.show()
                import pdb; pdb.set_trace() ## DEBUG ##
                
            if False:
                plt.close('all')
                plt.figure()
                import pdb; pdb.set_trace() ## DEBUG ##
        else:
            velocities[:, ii] = vel_init[:, ii]
        positions[:, ii+1] = positions[:, ii] + velocities[:, ii]*dt

        if False:
        # if True:
            plt.quiver(positions[0, ii], positions[1, ii],
                       np.cos(angle_tangent2init[ii]+angle_velInit[ii]),
                       np.sin(angle_tangent2init[ii]+angle_velInit[ii]), color='b')
            # plt.quiver(positions[0, ii], positions[1, ii], np.cos(angle_velInit[ii]), np.sin(angle_velInit[ii]), color='r')
            # plt.quiver(positions[0, ii], positions[1, ii], velocities[0, ii], velocities[1, ii], color='g')
            plt.quiver(positions[0, ii], positions[1, ii], tangents[0, ii], tangents[1,ii], color='k')
            # import pdb; pdb.set_trace() ## DEBUG ##
            
            
        if False and np.linalg.norm(positions[:, ii+1]-np.array([0,-1]))<0.2:
            plt.figure()
            plt.plot(obstacle.x_obs[:,0], obstacle.x_obs[:,1], 'k')
            plt.plot(obstacle.reference_point[0], obstacle.reference_point[1], 'k+')
            plt.axis('equal')
            plt.plot(attractor[0], attractor[1], 'k*')
            plt.plot(positions[0, :ii+2], positions[1, :ii+2], '.')
            import pdb; pdb.set_trace() ## DEBUG ##

        if np.linalg.norm(positions[:,ii+1] - positions[:, ii]) < 1e-4:
            positions = np.delete(positions, np.arange(ii, positions.shape[1]), axis=1)
            print('converged')
            break

        # if np.linalg.norm(positions[:,ii+1] - attractor) < 1e-1 \
           # or np.linalg.norm(positions[:,ii+1] - obstacle.reference_point) < 1e-1:
        # print('converged')            

    if False and it_point==0:
        plt.figure()
        plt.plot(angle_tangent2init, 'b')
        plt.figure()

    if it_point==0:
        # plt.figure()
        plt.plot(obstacle.x_obs[:,0], obstacle.x_obs[:,1], 'k')
        plt.plot(obstacle.reference_point[0], obstacle.reference_point[1], 'k+')
        plt.axis('equal')
        plt.plot(attractor[0], attractor[1], 'k*')
        plt.show()
        
    # plt.plot(positions[0, :], positions[1, :], '.')
    plt.plot(positions[0, :], positions[1, :])

    
plt.show()    

save_figure = True
if save_figure:
    fig.savefig('../figures/circeling_starshapedFlower_npoints{}_orientation{}_ylim{}to{}.pdf'.format(n_points, orientation_object, y_range[0], y_range[1]))
    
