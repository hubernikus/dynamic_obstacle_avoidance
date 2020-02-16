#!/USSR/bin/python3

'''
Script to show lab environment on computer

@author LukasHuber
@date 2020-01-15
@conact lukas.huber@epfl.ch
'''

# Command to automatically reload libraries -- in ipython before executing
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import *


def main(n_resol=90, *args, **kwargs):
    # x_lim = [0.9, 3.1]
    # y_lim = [0.9, 3.1]

    x_lim = [-4.4, 4.4]
    y_lim = [-4.1, 4.1]

    x_lim = [-5.4, 5.4]
    y_lim = [-5.1, 5.1]

    pos_attractor = [-1.5, 1.0]

    obs = ObstacleContainer()
    
    edge_points = np.array((
        [4.0, 1.0, 1.0, 0.0, 0.0,-4.0, -4.0,-2.5,-2.5, 0.0, 0.0, 4.0],
        [0.0, 0.0, 1.0, 1.0, 1.5, 1.5, -2.0,-2.0,-3.6,-3.6,-3.0,-3.0]
    ))

    # edge_points = np.array([[ 4, 4, 3, 1, -1, -3, -4, -4],
                            # [-4, 4, 4, 2, 2, 4,  4, -4]])
                            

    case_list = {"lab":0, "one_square":1, "one_ellipse":2}
    
    case = "lab"
    # case = "one_square"
    # case = "one_ellipse"
      
    if case_list[case]==case_list["lab"]:

        robot_margin = 0.4 # radius
        # edge_points = np.array((
        # [100.0,-100.0,-100.0, 100.0],
        # [100.0, 100.0, -100.0,-100.0]))

        # frame_id = "world_lab"
        # obs.append( Polygon(edge_points=edge_points, margin_absolut=0.5))
        y_lim = [-4, 2]
        obs.append( Polygon(edge_points=edge_points, is_boundary=True, margin_absolut=robot_margin, center_position=np.array([-1.0, -1]) , name="lab") )

        # Displacement
        
        # Table
        obs.append( Cuboid(center_position=[-2.5, 0.0], axes_length=[0.8, 0.8], margin_absolut=0.4, name="table"))
        
        # Tool-Trolley
        # obs.append( Cuboid(center_position=[0.01, -0.5], orientation=-80/180.*pi, axes_length=[1.2, 0.5]) )
        obs.append( Cuboid(center_position=[0.3, -0.5], axes_length=[0.2, 0.4], margin_absolut=0.4, name="trolley", orientation=90/180.*pi))

        # Human
        obs.append( Ellipse(center_position=[-1.0, -0.5], orientation=-10/180.*pi, axes_length=[0.5, 0.3], margin_absolut=0.4, name="human"))

        Simulation_vectorFields(x_lim, y_lim, n_resol, obs, xAttractor=pos_attractor, saveFigure=False, figName='lab_environment', noTicks=False, automatic_reference_point=True, draw_vectorField=True)

        plt.grid('true')
        
    elif case_list[case]==case_list["one_square"]:
        
        obs.append( Cuboid(center_position=[0.0, 0.0], axes_length=[2.0, 2.0],
                           # margin_absolut=0.0,
                           margin_absolut=1.0, orientation=0*pi/180))

        # obs[-1].set_reference_point(np.array([3, 0]), in_global_frame=False)


        Simulation_vectorFields(x_lim, y_lim, n_resol, obs, xAttractor=pos_attractor, saveFigure=False, figName='lab_environment_three_connected', noTicks=False, automatic_reference_point=True)

        # obs[-1].set_reference_point(np.array([1.4, 1.4]), in_global_frame=False)
        # obs.append( Cuboid(center_position=[-1.0, -1.2], axes_length=[1.0, 1.0], margin_absolut=1.0, orientation=120*pi/180))
        # obs.append( Cuboid(center_position=[0.0, -0.9], axes_length=[1.0, 1.0], margin_absolut=1.0, orientation=0))
        # obs[-1].set_reference_point(np.array([1.4, 1.4]), in_global_frame=False)
    elif case_list[case]==case_list["one_ellipse"]:
        obs.append( Ellipse(
            center_position=[0.0, 0.0],
            axes_length=[2.0, 1.0],
            # margin_absolut=1.0,
            margin_absolut=0.0,
            orientation=00*pi/180))
        
        obs[-1].set_reference_point(np.array([2.9, 2]), in_global_frame=True)
        # obs[-1].set_reference_point(np.array([0, 2]), in_global_frame=True)
        
        Simulation_vectorFields(x_lim, y_lim, n_resol, obs, xAttractor=pos_attractor, saveFigure=False, figName='lab_environment', noTicks=False, automatic_reference_point=True, draw_vectorField=True)

    return 
    
    plt.figure()
    plt.grid(True)

    # coloring_gamma = True
    # if coloring_gamma:
        # Gamma_coloring =

    # pos = np.array([1.0, 0])
    # pos = np.array([-.0, -3.0])
    pos = np.array([-.150, -2.7])
    normal0 = obs[0].get_normal_direction(pos, in_global_frame=True)
    gamma0 = obs[0].get_gamma(pos, in_global_frame=True)
    
    n_resolution = 100
    x_grid = np.linspace(x_lim[0], x_lim[1], n_resolution)
    y_grid = np.linspace(y_lim[0], y_lim[1], n_resolution)

    n_obs = len(obs)
    Gamma_vals = np.zeros((n_resolution, n_resolution, n_obs))
    normals = np.zeros((obs[0].dim, n_resolution, n_resolution, n_obs))
    positions = np.zeros((obs[0].dim, n_resolution, n_resolution))

    for it_obs in range(n_obs):
        for ix in range(n_resolution):
            for iy in range(n_resolution):
                pos = np.array([x_grid[ix], y_grid[iy]])
                
                positions[:, ix, iy] = pos
                
                Gamma_vals[ix, iy, it_obs] = obs[it_obs].get_gamma(pos, in_global_frame=True)
                # normals[:, ix, iy, it_obs] = obs[it_obs].get_normal_direction(pos, in_global_frame=True)

    # Gamma_vals[22, 3] = 100
    Gamma_vals = np.flip(Gamma_vals, axis=1)
    Gamma_vals = np.swapaxes(Gamma_vals, 0, 1)

    merge_gammas = False
    if merge_gammas:
        Gamma_prod = np.ones((n_resolution, n_resolution))
        for oo in range(n_obs):
            Gamma_prod = Gamma_prod*np.max((Gamma_vals[:, :, oo]-1, np.zeros(Gamma_vals[:, :, oo].shape)), axis=0 )
        # Gamma_prod = Gamma_prod*Gamma_vals[:, :, oo]

        # Gamma_prod = (Gamma_prod+1)**(1.0/n_obs)
        # Gamma_prod = (Gamma_prod + 1)**(1.0/n_obs)
        
        # Gamma_vals = Gamma_vals[:, :, 0]
        Gamma_vals = Gamma_prod
    else:
        it_obs = 0
        Gamma_vals = Gamma_vals[:, :, it_obs]

    max_val = 3
    if max_val:
        Gamma_vals[Gamma_vals>max_val]=max_val
    # masked_array = np.ma.masked_where(Gamma_vals<0.99, Gamma_vals)
    masked_array = np.ma.masked_where(Gamma_vals<0.99, Gamma_vals)
    # masked_array = np.ma.masked_where(Gamma_vals>1.01, Gamma_vals)

    cmap = matplotlib.cm.winter
    cmap.set_bad(color='white')
     
    # plt.figure()
    # im = plt.imshow(Gamma_vals, extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]])
    dx2 = (x_grid[1]-x_grid[0])/2.0
    dy2 = (y_grid[1]-y_grid[0])/2.0

    im = plt.imshow(masked_array, cmap=cmap,
                    extent=[x_lim[0]-dx2, x_lim[1]+dx2, y_lim[0]-dy2, y_lim[1]+dy2])
    ax = plt.gca()
    cbar = ax.figure.colorbar(im, ax=ax)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    if False:
        for it_obs in [0]:
            plt.quiver(positions[0, :, :], positions[1, :, :],
                       normals[0, :, :, it_obs], normals[1, :, :, it_obs])

if (str(__name__)==("__main__")):
    main()
