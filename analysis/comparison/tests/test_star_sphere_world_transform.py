"""
Double Blob Obstacle Test
"""
import unittest

from math import pi

import numpy as np
from numpy import linalg as LA

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import Obstacle, Ellipse, Sphere

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from sphere_world_optimizer import SphereWorldOptimizer, ClosedLoopQP
from navigation import NavigationContainer

class StarSphereWorldTransform(unittest.TestCase):
    def test_object_hull_to_circle_trafo(self):
        pass

def draw_displacement():
    # Set to 1000 as describe din paper.
    sphere_world = SphereWorldOptimizer(
        attractor_position=np.array([0, 0]),
        lambda_constant=1000)

    # sphere_world.append(
        # Sphere(
        # center_position=np.array([1, 1]),
        # radius=0.4,
        # ))

    sphere_world.append(
        DoubleBlob(
            a_value=1, b_value=1.1,
            center_position=[0, 3],
            is_boundary=False,
            ))

    sphere_world.append(
        Sphere(
        center_position=np.array([0, 0]),
        radius=8,
        is_boundary=True,
        ))


    plt.plot()

def plot_star_and_sphere_world():
    """ Sphere world & sphere-world."""
    x_lim = [-5, 5]
    y_lim = [-5, 5]

    obstacle_container = NavigationContainer()
    obstacle_container.attractor_position = np.array([0, 5])
    obstacle_container.append(
        Ellipse(
            center_position=np.array([2, 0]), 
            axes_length=np.array([2, 1]),
            )
        )

    obstacle_container.append(
        Ellipse(
            center_position=np.array([-2, 2]), 
            axes_length=np.array([2, 1]),
            orientation=30./180*pi,
            )
        )

    obstacle_container.append(
        DoubleBlob(
            a_value=1, b_value=1.1,
            center_position=np.array([2, 3]),
            is_boundary=False,
            ))


    # obstacle_container.append(
    #     Ellipse(
    #         center_position=np.array([-1, -2.5]), 
    #         axes_length=np.array([2, 1]),
    #         orientation=-50./180*pi,
    #         )
    #     )

    n_resolution = 50
    # velocities = np.zeros(positions.shape)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    
    for it_obs in range(obstacle_container.n_obstacles):
        obstacle_container[it_obs].draw_obstacle(n_grid=n_resolution)
        boundary_points = obstacle_container[it_obs].boundary_points_global
        ax.plot(boundary_points[0, :], boundary_points[1, :], 'k-')
        ax.plot(obstacle_container[it_obs].center_position[0],
                obstacle_container[it_obs].center_position[1], 'k+')

        boundary_displaced = np.zeros(boundary_points.shape)
        for ii in range(boundary_points.shape[1]):
            boundary_displaced[:, ii] = obstacle_container.transform_to_sphereworld(
                boundary_points[:, ii])
                
            # collisions[ii] = obstacle_container.check_collision(positions[:, ii])
            
        ax.plot(boundary_displaced[0, :], boundary_displaced[1, :], 'b')

        ax.plot(obstacle_container[it_obs].center_position[0],
                obstacle_container[it_obs].center_position[1], 'b+')
            
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.plot(obstacle_container.attractor_position[0],
            obstacle_container.attractor_position[1], 'k*')
    
    ax.set_aspect('equal', adjustable='box')
    
    plt.ion()
    plt.show()

    # cbar = fig.colorbar(cs, ticks=np.linspace(-10, 0, 11))
    
    # for ii in range(n_resolution):
        # obstacle_container.check_collision(positions[:, ii])
        # n_resolution =
        # pass




if (__name__) == "__main__":
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    visualize = True
    if visualize:
        plt.close('all')
        plt.ion()

        tester = StarSphereWorldTransform()
        
        # draw_displacement()
        plot_star_and_sphere_world()
        
        plt.show()

    print("Finished running script.")
    # print("No output was produced.")
    # for ii in range(10):
        # print(f"It = {ii}")
        # plt.pause(0.5)
