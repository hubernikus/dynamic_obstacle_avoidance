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
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *

def transform_to_directionSpace(reference_direction, directions, normalize=True):
    ind_nonzero = (weights>0)

    reference_direction = np.copy(reference_direction)
    directions = directions[:, ind_nonzero]
    weights = weights[ind_nonzero]

    n_directions = weights.shape[0]
    if n_directions<=1:
        return directions[:, 0]

    dim = np.array(reference_direction).shape[0]
    if dim>2:
        warnings.warn("Implement for higher dimensions.")
        
    if normalize:
        norm_refDir = LA.norm(reference_direction)
        if norm_refDir: # nonzero
            reference_direction /= norm_refDir

        norm_dir = LA.norm(directions, axis=0)
        ind_nonzero = (norm_dir>0)
        if ind_nonzero.shape[0]:
            directions[:,ind_nonzero] = directions[:,ind_nonzero]/np.tile(norm_dir[ind_nonzero], (dim, 1))

    OrthogonalBasisMatrix = get_orthogonal_basis(reference_direction)

    directions_referenceSpace = np.zeros(np.shape(directions))
    for ii in range(np.array(directions).shape[1]):
        directions_referenceSpace[:,ii] = OrthogonalBasisMatrix.T.dot( directions[:,ii])

    directions_directionSpace = directions_referenceSpace[1:, :]
    return directions_directionSpace


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


def get_local_weight(r, r_ref, r_boundary=2, power_weight=1, delta_weight=0.4):
    delta_r = r-r_ref

    if delta_r>r_boundary:
        return 0
    elif delta_r < 0:
        return ((r_boundary-delta_r)/r_boundary)**(1/power_weight)
    else:
        return ((r_boundary-delta_r)/r_boundary)**(1/power_weight)
    
def get_weighted_angular_mean(angle_weights, angle_tangents2init, init_weight=0):
    angle_init = 0
    if not sum(angle_weights):
        return angle_init
    angle_weights = np.hstack((angle_weights, init_weight))
    angle_tangents2init = np.hstack((angle_tangents2init, angle_init))

    ind_exceedMax = angle_weights>1
    if np.sum(ind_exceedMax):
        return np.sum(angle_weights[ind_exceedMax]*angle_tangents2init[ind_exceedMax])/np.sum(ind_exceedMax)
    
    if np.sum(angle_weights)<1:
        angle_weights[-1] = 1 - np.sum(angle_weights[:-1])
        
    one_ind = (angle_weights==1)
    if np.sum(one_ind):
        return np.ones(np.sum(one_ind))/np.sum(one_ind)*angle_tangent2init[one_ind]

    angle_weights = angle_weights/np.sum(angle_weights)
    angle_weights = 1/(1-angle_weights) - 1
    angle_weights = angle_weights/np.sum(angle_weights)
    
    # print('angle weights', angle_weights)
    # print('angles', angle_tangents2init)
    angle_desired = np.sum(angle_weights*angle_tangents2init)
    
    return angle_desired

    
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

# orientation_object=0

obstacles = []
obstacles.append(StarshapedFlower(orientation=-2/180*pi,
                                  center_position=[0.4,3], radius_magnitude=1, radius_mean=2, number_of_edges=4))
obstacles.append(StarshapedFlower(orientation=1/180*pi, radius_magnitude=1, radius_mean=2, number_of_edges=4))

for oo in range(len(obstacles)):
    obstacles[oo].draw_obstacle()

dim = 2
n_steps = 1000

n_points = 10
x_range = [-2,4]
y_range = [1.2, 2]
x_inits = np.vstack(([x_range[1]*np.ones(n_points),
                      np.linspace(y_range[0], y_range[1], n_points)]))
# x_inits = np.array([[4],[1.4]])

dt = 0.1

attractor = np.array([-6.0, -1])

fig, ax = plt.subplots()

for it_point in range(x_inits.shape[1]):
    positions = np.zeros((dim, n_steps))
    velocities = np.zeros((dim, n_steps))
    vel_init = np.zeros((dim, n_steps))

    # Only last one is important, not n_steps list needed (for speed)
    angle_tangents_list = np.zeros((n_steps, len(obstacles) ))
    angle_tangent2init = np.zeros((n_steps, len(obstacles) ))
    weights= np.zeros( len(obstacles) )

    angle_velInit = np.zeros(n_steps)
    
    # tangents = np.zeros((dim, n_steps, len(obstacles) ))

    x_init = x_inits[:, it_point]

    ii = 0
    positions[:, ii] = x_init

    set_vortex_direction = False
    for obstacle in obstacles:
        obstacle.properties["vortex_direction"] = 1
        obstacle.properties["outside_influence_region"]=True
        obstacle.properties["passed_obstacle"]=False

        # obstacle.properties["vortex_direction"] = 1
        
        
    for ii in range(n_steps-1):
        vel_init[:, ii] = ds_init(positions[:, ii], attractor=attractor)
        
        if all([obstacles[oo].properties["passed_obstacle"] for oo in range(len(obstacles))]):
            velocities[:, ii] = vel_init[:, ii]
            positions[:, ii+1] = positions[:, ii] + velocities[:, ii]*dt
        
        check_step = 0.1
        n_checks = np.linalg.norm(positions[:,ii]-attractor)/check_step

        try:
            check_points = np.vstack((np.linspace(positions[0,ii], attractor[0], n_checks),np.linspace(positions[1,ii], attractor[1], n_checks)))
        except:
            import pdb; pdb.set_trace() ## DEBUG ##
            

        for oo in range(len(obstacles)):
            if ii>=1 and angle_tangent2init[ii-1, oo]*obstacles[oo].properties["vortex_direction"]>=0:
                # obstacle.properties["passed_obstacle"] = True
                for jj in range(check_points.shape[1]):
                    
                    if obstacles[oo].get_gamma(check_points[:,jj], in_global_frame=True)<1:
                        obstacle.properties["passed_obstacle"] = False
                        break
        
        if all([obstacles[oo].properties["passed_obstacle"] for oo in range(len(obstacles))]):
            velocities[:, ii] = vel_init[:, ii]
            positions[:, ii+1] = positions[:, ii] + velocities[:, ii]*dt
            continue

        
        angle_velInit[ii] = np.arctan2(vel_init[1, ii], vel_init[0, ii])
        if ii==0:
            delta_velInit = 0
        delta_velInit = angle_difference_directional(angle_velInit[ii], angle_velInit[ii-1])

        for oo in range(len(obstacles)):
            if obstacle.properties["passed_obstacle"]:
                continue
                
            normal = obstacles[oo].get_normal_direction(positions[:, ii], in_global_frame=True)
            tangent = np.array([-normal[1], normal[0]])

            mag, ang = transform_cartesian2polar(positions[:, ii], obstacles[oo].global_reference_point)

            dist2center = np.linalg.norm(positions[:, ii]-obstacles[oo].global_reference_point)
            local_radius = obstacles[oo].get_radius_of_angle(ang, in_global_frame=True)

            weights[oo] = get_local_weight(dist2center, local_radius)
        
        # Local weights calculation
        # for oo in range(len(obstacles)):
            if weights[oo] > 0:
                if obstacles[oo].properties["outside_influence_region"]:
                    obstacle_dir = obstacles[oo].global_reference_point - attractor
                    point_dir = positions[:, ii] - attractor

                    if set_vortex_direction:
                        if np.cross(obstacle_dir, point_dir) > 0:
                            obstacles[oo].properties["vortex_direction"] = 1
                        else:
                            obstacles[oo].properties["vortex_direction"] = -1
                
                tangent = tangent*obstacles[oo].properties["vortex_direction"]
                angle_tangents_list[ii, oo] = np.arctan2(tangent[1], tangent[0])
                
                angle_tangent2init_ref = angle_difference_directional(angle_tangents_list[ii, oo], angle_velInit[ii])
                angle_tangent2init_ref = angle_modulo(angle_tangent2init_ref)
                

                if False:
                    plt.quiver(positions[0, ii], positions[1, ii], np.cos(angle_tangent2init_ref+angle_velInit[ii]), np.sin(angle_tangent2init_ref+angle_velInit[ii]), color='r', scale=15)

                    plt.quiver(positions[0, ii], positions[1, ii],
                               np.cos(angle_tangents_list[ii, oo]),
                               np.sin(angle_tangents_list[ii, oo]), color='g', scale=15)

                if obstacles[oo].properties["outside_influence_region"]:
                    # print('angang',  angle_tangent2init[ii, oo])
                    obstacles[oo].properties["outside_influence_region"] = False

                    ind_nonzero = np.array(angle_tangent2init[ii-1,:], dtype=bool)
                    if not np.sum(ind_nonzero):
                        ind_nonzero = np.array(angle_tangent2init[ii,:], dtype=bool)
                    angle_tangent2init[ii, oo] = angle_tangent2init_ref

                    
                    if np.sum(ind_nonzero):
                        # TODO - find direct neighbor ind
                        # TODO - more strategic
                        first_ind = np.arange(ind_nonzero.shape[0])[ind_nonzero][0] 

                        referenceDirection_knownObstacle = obstacles[first_ind].get_reference_direction(positions[:,ii], in_global_frame=True)
                        referenceDirection_newObstacle = obstacles[oo].get_reference_direction(positions[:,ii], in_global_frame=True)

                        
                        if np.cross(referenceDirection_newObstacle, referenceDirection_knownObstacle) >= 0:
                            while(angle_tangent2init[ii, oo] > angle_tangent2init[ii, first_ind]):
                                angle_tangent2init[ii, oo] -= 2*pi
                        else:
                            while(angle_tangent2init[ii, oo] < angle_tangent2init[ii, first_ind]):
                                angle_tangent2init[ii, oo] += 2*pi
                    # else:
                    if False:
                        dir_ref = obstacles[oo].get_reference_direction(positions[:,ii], in_global_frame=True)
                        angle_ref = np.arctan2(dir_ref[1], dir_ref[0])
                        
                        angle_ref2init = angle_difference_directional(angle_ref, angle_velInit[ii])
                        angle_ref2init = angle_modulo(angle_ref2init)
                        
                        if obstacles[oo].properties["vortex_direction"] ==1:
                            while (angle_tangent2init[ii,oo] < angle_ref2init):
                                angle_tangent2init[ii,oo]+=2*pi
                        else:
                            while(angle_tangent2init[ii,oo] > angle_ref2init):                                                            angle_tangent2init[ii,oo]-=2*pi
                else:
                    delta_angle_tangent = angle_difference_directional(angle_tangents_list[ii, oo], angle_tangents_list[ii-1, oo])
                    
                    # Continuous Integration
                    angle_tangent2init[ii, oo] =  angle_tangent2init[ii-1, oo] + delta_angle_tangent - delta_velInit 

                    angle_tangent2init[ii, oo] = windup_smoothening(angle_tangent2init[ii, oo], angle_tangent2init_ref)
            else: # weight <=0
                obstacles[oo].properties["outside_influence_region"] = True
        
        if np.sum(weights):
            # Angle weight
            delt_angle_desired = get_weighted_angular_mean(angle_weights=weights, angle_tangents2init=angle_tangent2init[ii, :])
            angle_desired = delt_angle_desired + angle_velInit[ii]

            # print('w', weights)
            # angle_desired = weights[oo]*angle_tangent2init[ii,oo] + angle_velInit[ii]

            velocities[:, ii] = np.linalg.norm(vel_init[:, ii])  \
                                * np.array([np.cos(angle_desired), np.sin(angle_desired)])
        else:
            velocities[:, ii] = vel_init[:, ii]
        positions[:, ii+1] = positions[:, ii] + velocities[:, ii]*dt

        if False:
        # if True:
            for oo in range(len(obstacles)):
                plt.quiver(positions[0, ii], positions[1, ii],
                           np.cos(angle_tangent2init[ii, oo]+angle_velInit[ii]),
                           np.sin(angle_tangent2init[ii, oo]+angle_velInit[ii]), color='b')
            plt.quiver(positions[0, ii], positions[1, ii], np.cos(angle_velInit[ii]), np.sin(angle_velInit[ii]), color='r')
            plt.quiver(positions[0, ii], positions[1, ii], velocities[0, ii], velocities[1, ii], color='g')
            print('angle', angle_tangent2init[ii, :])
            
        if np.linalg.norm(positions[:,ii+1] - positions[:, ii]) < 1e-4:
            positions = np.delete(positions, np.arange(ii, positions.shape[1]), axis=1)
            print('Converged towards attractor at iteration #{}'.format(ii))
            break

        # if np.linalg.norm(positions[:,ii+1] - attractor) < 1e-1 \
           # or np.linalg.norm(positions[:,ii+1] - obstacles[oo].global_reference_point) < 1e-1:
        # print('converged')            

    if False and it_point==0:
        plt.figure()
        plt.plot(angle_tangent2init, 'b')
        plt.figure()
        

    if it_point==0:
        for oo in range(len(obstacles)):
            # plt.figure()
            plt.plot(obstacles[oo].x_obs[:,0], obstacles[oo].x_obs[:,1], 'k')
            plt.plot(obstacles[oo].global_reference_point[0], obstacles[oo].global_reference_point[1], 'k+')
            plt.axis('equal')
            plt.plot(attractor[0], attractor[1], 'k*')
            plt.show()
        
    # plt.plot(positions[0, :], positions[1, :], '.')
    plt.plot(positions[0, :], positions[1, :])

    print('Finished simulation without convergence.')
plt.show()    

save_figure = False
if save_figure:
    fig.savefig('../figures/circeling_starshapedFlower_npoints{}_orientation{}_ylim{}to{}.pdf'.format(n_points, orientation_object, y_range[0], y_range[1]))
    
