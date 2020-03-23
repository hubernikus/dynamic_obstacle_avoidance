#!/USSR/bin/python3

'''
Reference Point Search

@author LukasHuber
@date 2020-02-28
@conact Lukas.huber@epfl.ch
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
from numpy import pi

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import Obstacle
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import Ellipse

# plt.close('all')
n_points = 50

#def eval_eucledian_dsitance(phi_1, phi_2):
angle_position = np.zeros((2, n_points, n_points))
distance = np.zeros((n_points, n_points))
dist_proj = np.zeros((n_points, n_points))

case = [1, 2]

# phi_2 = np.linspace(-pi, pi, n_points)
phi_1 = np.linspace(0, 2*pi, n_points)
phi_2 = np.linspace(-pi, pi, n_points)

# phi_1 = np.linspace(pi/2.0, 2*pi-pi/2.0, n_points)
# phi_2 = np.linspace(-pi/2.0, pi/2.0, n_points)

directions_1 = np.vstack((np.cos(phi_1), np.sin(phi_1)))
directions_2 = np.vstack((np.cos(phi_2), np.sin(phi_2)))

obs1 = Ellipse(axes_length=[0.5, 1.0], orientation=30./180*pi, center_position=[2.0, 1.0])
obs2 = Ellipse(axes_length=[0.2, 2.0], orientation=-30./180*pi, center_position=[-1.4, -0.4])


if 1 in case:
    # obs1 = Ellipse(axes_length=[5.0, 0.8], orientation=50./180*pi, center_position=[0.0, 0])
    # obs2 = Ellipse(axes_length=[0.4, 1], center_position=[3, 0])
    
    center_dir = obs2.center_position - obs1.center_position
    norm_center_dir = np.linalg.norm(center_dir, 2)
    if norm_center_dir:
        center_dir = center_dir / norm_center_dir

    obs_orientation = Obstacle(orientation=np.arctan2(center_dir[1], center_dir[0]))

    dim = obs1.dim

    surface_points_1 = np.zeros((dim, n_points))
    surface_points_2 = np.zeros((dim, n_points))

    surf_dir1 = np.zeros((dim, n_points))
    surf_dir2 = np.zeros((dim, n_points))

    dist_norm = np.zeros((n_points, n_points))
    d_dphi = np.zeros(( (dim-1)*2, n_points, n_points ))
        
    for ii in range(n_points):
        surface_points_1[:, ii] = obs1.get_intersection_with_surface(direction=directions_1[:, ii], only_positive_direction=True, in_global_frame=True)

        surf_dir1[:, ii] = obs1.get_surface_derivative_angle(phi_1[ii], in_global_frame=True)
        
    for jj in range(n_points):
        surface_points_2[:, jj] = obs2.get_intersection_with_surface(direction=directions_2[:, jj], only_positive_direction=True, in_global_frame=True)
        
        surf_dir2[:, jj] =  obs2.get_surface_derivative_angle(phi_2[jj], in_global_frame=True)

    for ii in range(n_points):
        for jj in range(n_points):
            angle_position[:, ii, jj] = [phi_1[ii], phi_2[jj]]

            dist_dir = (surface_points_2[:, jj]-surface_points_1[:, ii])
            distance[ii, jj] = np.linalg.norm(dist_dir, axis=0)

            # TODO: check sign
            d_dphi[0, ii, jj] = 1.0/distance[ii, jj]*(dist_dir).T.dot(surf_dir1[:, ii])
            d_dphi[1, ii, jj] = -1.0/distance[ii, jj]*(dist_dir).T.dot(surf_dir2[:, jj])
            
            dist_proj[ii, jj] = dist_dir.T.dot(center_dir)

            # distance[ii, jj] = dist_proj[ii, jj]*distance[ii, jj]
            # distance[ii, jj] = dist_proj[ii, jj]

            # distance[ii, jj] = np.copysign(distance[ii, jj], dist_dir.dot(center_dir))
            # val =  np.copysign(distance[ii, jj], dist_dir.dot(center_dir))

    (ii_min, jj_min) = np.unravel_index(np.argmin(distance, axis=None), (n_points, n_points))
    

    # plt.figure()
    # plt.plot()
    # plt.title("Surface Derivative")
    
    # plt.figure()
    # plt.quiver(phi_1.reshape(n_points, 1), np.zeros((n_points, 1)), surf_dir1[0, :].reshape(n_points, 1), surf_dir1[1, :].reshape(n_points, 1))
    # plt.xlabel('$\phi_1$')
    # plt.title('Surface Direction')
    
    if False:
        print('sur dir \n', np.round(surf_dir1, 2))
        print('sur dir \n', np.round(surf_dir2, 2))
        # print('sur dir \n', np.round(surface_points_1, 2))
        # print('sur dir \n', np.round(surface_points_2, 2))
        raise ValueError("Stop script")

    # plt.figure()
    # plt.quiver(phi_2.reshape(n_points, 1), np.zeros((n_points, 1)), surf_dir2[0, :].reshape(n_points, 1), surf_dir2[1, :].reshape(n_points, 1))
               
    # plt.xlabel('$\phi_2$')
    # plt.title('Surface Direction')
    
    plt.ion()
    plt.show()

    # phi_1 = phi_1*180/pi
    # phi_2 = phi_2*180/pi
    # angle_position = angle_position*180/pi

    show_plot_3d = False
    if show_plot_3d:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(angle_position[0, :, :], angle_position[1, :, :], distance,
                               cmap=cm.coolwarm, # linewidth=0,
                               antialiased=False, alpha=0.9)

        freq_quiver = 3
        ind_ = np.logical_not(np.mod(np.arange(n_points), freq_quiver))
        plt.quiver(angle_position[0, ind_, :][:, ind_], angle_position[1, ind_, :][:, ind_], np.zeros(np.sum(ind_)), d_dphi[0, ind_,  :][:, ind_], d_dphi[1, ind_,  :][:, ind_], np.zeros(np.sum(ind_)), color='k')
        plt.xlabel('$\phi_1$')
        plt.ylabel('$\phi_2$')
        plt.title('Eucledian Distance')
        fig.colorbar(surf, shrink=0.5, aspect=5)


    show_stream_plot = False
    if show_stream_plot:
        plt.figure(figsize=(6,6))
        plt.streamplot(phi_1, phi_2, d_dphi[0, :,  :].T, d_dphi[1, :,  :].T, zorder=-2)
        # plt.streamplot(angle_position[0, :, :], angle_position[1, :, :], d_dphi[0, :,  :], d_dphi[1, :,  :])
        plt.plot(phi_1[ii_min], phi_2[jj_min], '*r', zorder=1)
        plt.grid()
        plt.axis('equal')
        plt.xlim(phi_1[0], phi_1[-1])
        plt.ylim(phi_2[0], phi_2[-1])
        plt.xlabel('$\phi_1$')
        plt.ylabel('$\phi_2$')
        plt.title('Streamline of Gradient')

    show_quiver = True
    if show_quiver:
        plt.figure(figsize=(6,6))
        plt.quiver(angle_position[0, :, :], angle_position[1, :, :], d_dphi[0, :,  :], d_dphi[1, :,  :])
        plt.plot(phi_1[ii_min], phi_2[jj_min], '*r', zorder=1)
        

        plt.grid()
        plt.axis('equal')
        plt.xlim(phi_1[0], phi_1[-1])
        plt.ylim(phi_2[0], phi_2[-1])
        plt.xlabel('$\phi_1$')
        plt.ylabel('$\phi_2$')
        plt.title('Streamline of Gradient')


if 2 in case:
    dim_ang = 2*(obs1.dim-1)
    
    dist_dir = obs2.center_position - obs1.center_position

    ang_1 = np.arctan2(dist_dir[1], dist_dir[0])
    ang_2 = np.arctan2(-dist_dir[1], -dist_dir[0])
    # ang_1 = pi/2
    # ang_2 = -1.73*pi/2
    
    angles_optimization = np.array([[ang_1], [ang_2]])
    distance_opt = np.zeros((0))

    step_size = 0.02
    convergence_err = 1e-3
    
    it_count = 0
    max_it = 1000
    
    while True:
        dist_old = -1

        surf_dir1 = obs1.get_surface_derivative_angle(ang_1, in_global_frame=True)
        surf_dir2 = obs2.get_surface_derivative_angle(ang_2, in_global_frame=True)

        directions_1 = np.array([np.cos(ang_1), np.sin(ang_1)])
        surface_point_1 = obs1.get_intersection_with_surface(direction=directions_1, only_positive_direction=True, in_global_frame=True)

        directions_2 = np.array([np.cos(ang_2), np.sin(ang_2)])
        surface_point_2 = obs2.get_intersection_with_surface(direction=directions_2, only_positive_direction=True, in_global_frame=True)

        dist_dir = (surface_point_2 - surface_point_1)
        distance_opt = np.append(distance_opt, [np.linalg.norm(dist_dir)] )

        d_dphi_opt = 1.0/distance_opt[-1]*np.array([-np.sum(dist_dir*surf_dir1),
                                                    np.sum(dist_dir*surf_dir2)])

        new_angles = angles_optimization[:, -1] - step_size*d_dphi_opt
        
        angles_optimization = np.append(angles_optimization, new_angles.reshape(dim_ang, 1), axis=1)

        ang_1, ang_2 = new_angles[0], new_angles[1]
        
        it_count += 1

        # Gradient descent step
        if it_count>max_it or np.linalg.norm(angles_optimization[:, -1] - angles_optimization[:, -2]) < convergence_err:
            break
        
    print("Converged after {} iterations.".format(it_count))
    
    # plt.figure()
    plt.plot(angles_optimization[0, :], angles_optimization[1, :], 'g', zorder=2)
    plt.plot(angles_optimization[0,-1], angles_optimization[1, -1], '*g', label="Minimum Gradient", zorder=2)
    plt.plot(angles_optimization[0,0], angles_optimization[1, 0], 'og', zorder=2)
    plt.plot(phi_1[ii_min], phi_2[jj_min], '*r', label="Minimum Numerical", zorder=2)
    plt.axis('equal')
    plt.legend(loc=2)
    
    plt.figure()
    plt.plot(np.arange(it_count), distance_opt, 'g', label="Gradient Descend (min={})".format(np.round(distance_opt[-1], 3)))
    plt.plot(np.sum(it_count)-1, distance[ii_min, jj_min], '.r', label="Numerical Distance = {}".format(np.round(distance[ii_min, jj_min], 3)))
    # plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Distance")
    plt.grid()
    plt.legend()

plt.figure()
plt.plot(surface_points_1[0, :], surface_points_1[1, :], 'k', label='Ellipse 1')
plt.plot(surface_points_2[0, :], surface_points_2[1, :], 'k', label='Ellipse 2')
plt.plot(surface_points_1[0, ii_min], surface_points_1[1, ii_min], '*r', label='Minimum @ $\phi_1$={} / $\phi_2$ = {}'.format(np.round(phi_1[ii_min]*180/pi,0), np.round(phi_2[jj_min]*180/pi, 0)))
# plt.plot(surface_points_2[0, jj_min], surface_points_2[1, jj_min], '*r', label='Minimum')
plt.plot(surface_points_2[0, jj_min], surface_points_2[1, jj_min], '*r')
plt.plot(surface_point_1[0], surface_point_1[1], '*g', label="Minimum Gradient @ $\phi_1$={} / $\phi_2$ = {}".format(np.round(ang_1*180/pi, 0), np.round(ang_2*180/pi, 0)))
plt.plot(surface_point_2[0], surface_point_2[1], '*g')
plt.legend(loc=2)
plt.grid()
plt.axis('equal')
# plt.plot(directions_1[0, :], directions_1[1, :], 'o')

plt.ion()
plt.show()
