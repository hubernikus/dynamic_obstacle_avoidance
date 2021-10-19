"""
Double Blob Obstacle Test
"""
import unittest

from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import Obstacle, Ellipse, Sphere
from dynamic_obstacle_avoidance.obstacles import DoubleBlob

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from avoidance_comparison.sphere_world_optimizer import SphereWorldOptimizer, ClosedLoopQP
from avoidance_comparison.navigation import SphereToStarTransformer

# from navigation import NavigationContainer
# from double_blob_obstacle import DoubleBlob


class TestStarSphereWorldTransform(unittest.TestCase):
    def test_object_hull_to_circle_trafo(self):
        pass


def draw_displacement():
    # Set to 1000 as describe din paper.
    sphere_world = SphereWorldOptimizer(
        attractor_position=np.array([0, 0]), lambda_constant=1000
    )

    # sphere_world.append(
    # Sphere(
    # center_position=np.array([1, 1]),
    # radius=0.4,
    # ))

    sphere_world.append(
        DoubleBlob(
            a_value=1,
            b_value=1.1,
            center_position=[0, 3],
            is_boundary=False,
        )
    )

    sphere_world.append(
        Sphere(
            center_position=np.array([0, 0]),
            radius=8,
            is_boundary=True,
        )
    )

    plt.plot()


def plot_star_and_sphere_world(save_figure=False, lambda_constant=None):
    """Sphere world & sphere-world."""
    x_lim = [-5, 5]
    y_lim = [-5, 5]

    obstacle_container = SphereToStarTransformer()
    if lambda_constant is not None:
        obstacle_container.lambda_constant = lambda_constant

    # obstacle_container = NavigationContainer()
    obstacle_container.attractor_position = np.array([0, 5])
    # obstacle_container.append(
    #     Ellipse(
    #         center_position=np.array([2, 0]),
    #         axes_length=np.array([2, 1]),
    #         )
    #     )

    # obstacle_container.append(
    #     Ellipse(
    #         center_position=np.array([-2, 2]),
    #         axes_length=np.array([2, 1]),
    #         orientation=30./180*pi,
    #         )
    #     )

    obstacle_container.append(
        DoubleBlob(
            a_value=1,
            b_value=1.1,
            center_position=np.array([2, 3]),
            is_boundary=False,
        )
    )

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

    point_list = [
        [-1.20, 3.57],
        [3.58, 2.03],
        [-1.07, -2.66],
        [-1.31, 4.52],
        [0.17, 4.87],
        [0, -2],
        [0.92, 2.29],
        [3.43, 3.53],
        [-4, -2],
        [-4, 0],
        [-4, 4],
        [-4.01, 1.19],
    ]

    # point_list = [
    # [-4.01, 1.19],
    # ]

    l_head = 0.2
    for point in point_list:
        point_displaced = obstacle_container.transform_to_sphereworld(
            np.array(point)
        )

        plt.plot(point[0], point[1], "k.")
        plt.plot(point_displaced[0], point_displaced[1], "b.")

        dir_arrow = np.array(
            [point_displaced[0] - point[0], point_displaced[1] - point[1]]
        )

        norm_arrow = LA.norm(dir_arrow)
        if norm_arrow > l_head:  # Nonzero
            new_norm_arrow = max(norm_arrow - l_head, 0)
            dir_arrow = new_norm_arrow * dir_arrow / norm_arrow

            plt.arrow(
                point[0],
                point[1],
                dir_arrow[0],
                dir_arrow[1],
                color="#808080",
                head_width=0.1,
            )
        # plt.plot([point[0], point_displaced[0]],
        # [point[1], point_displaced[1]], '#808080')

    for it_obs in range(obstacle_container.n_obstacles):
        obstacle_container[it_obs].draw_obstacle(n_grid=n_resolution)
        boundary_points = obstacle_container[it_obs].boundary_points_global
        ax.plot(boundary_points[0, :], boundary_points[1, :], "k-")
        ax.plot(
            obstacle_container[it_obs].center_position[0],
            obstacle_container[it_obs].center_position[1],
            "k+",
        )

        boundary_displaced = np.zeros(boundary_points.shape)
        for ii in range(boundary_points.shape[1]):
            boundary_displaced[
                :, ii
            ] = obstacle_container.transform_to_sphereworld(
                boundary_points[:, ii]
            )

            # collisions[ii] = obstacle_container.check_collision(positions[:, ii])
        ax.plot(boundary_displaced[0, :], boundary_displaced[1, :], "b")

        ax.plot(
            obstacle_container[it_obs].center_position[0],
            obstacle_container[it_obs].center_position[1],
            "b+",
        )

    ax.plot([], [], "k", label="Real World Boundary")
    ax.plot([], [], "b", label="Circular World Boundary")

    ax.plot([], [], "k.", label="Initial Position [random points]")
    ax.plot([], [], "b.", label="Positition in Circlar World")

    plt.title(r"$\lambda$={}".format(obstacle_container.lambda_constant))

    ax.legend()

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.plot(
        obstacle_container.attractor_position[0],
        obstacle_container.attractor_position[1],
        "k*",
    )

    ax.set_aspect("equal", adjustable="box")

    plt.ion()
    plt.show()

    # cbar = fig.colorbar(cs, ticks=np.linspace(-10, 0, 11))

    # for ii in range(n_resolution):
    # obstacle_container.check_collision(positions[:, ii])
    # n_resolution =
    # pass
    if save_figure:
        fig_name = (
            "sphere_to_star_world with_lambda"
            + str(obstacle_container.lambda_constant)
            + "_n_obs"
            + str(len(obstacle_container))
        )
        plt.savefig("figures/" + fig_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)
    visualize = True
    if visualize:
        # plt.close('all')
        plt.ion()

        # tester = StarSphereWorldTransform()

        # draw_displacement()
        plot_star_and_sphere_world(save_figure=True, lambda_constant=10)
        plot_star_and_sphere_world(save_figure=True, lambda_constant=100)
        plot_star_and_sphere_world(save_figure=True, lambda_constant=1000)

        plt.show()

    print("Finished running script.")
    # print("No output was produced.")
    # for ii in range(10):
    # print(f"It = {ii}")
    # plt.pause(0.5)
