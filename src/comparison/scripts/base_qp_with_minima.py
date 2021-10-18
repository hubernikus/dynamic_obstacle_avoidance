"""
Replicating the paper of
'Safety of Dynamical Systems With MultipleNon-Convex Unsafe Sets UsingControl Barrier Functions'
"""
from abc import ABC, abstractmethod
import time
import copy

import numpy as np
from numpy import linalg as LA

# from cvxopt.modeling import variable
from cvxopt import solvers, matrix

import matplotlib.pyplot as plt
from matplotlib import cm

from vartools.dynamical_systems import DynamicalSystem, LinearSystem
from vartools.dynamical_systems import plot_dynamical_system_streamplot
from vartools.math import get_numerical_gradient, get_numerical_hessian
from vartools.math import get_numerical_hessian_fast
from vartools.math import get_scaled_orthogonal_projection

from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.obstacles import Obstacle, Sphere
from dynamic_obstacle_avoidance.containers import BaseContainer

from barrier_functions import (
    BarrierFunction,
    CirclularBarrier,
    DoubleBlobBarrier,
)
from barrier_functions import BarrierFromObstacleList

from _base_qp import ControllerQP
from navigation import SphereToStarTransformer
from double_blob_obstacle import DoubleBlob
from sphere_world_optimizer import SphereWorldOptimizer, ClosedLoopQP
from control_dynamics import StaticControlDynamics


class VelocityController(ABC):
    @abstractmethod
    def evaluate(self, position):
        """Returns the obstacle avoidance at a certain position."""
        pass


class NonconvexAvoidanceCBF(VelocityController):
    """
    Attributes
    ----------
    obstacle_container: SphereWorldOptimizer obstacle container.
    qp_control_optimizer: optimizes the path based on dynamics & 'modified' outside
    """

    def __init__(
        self,
        obstacle_container: SphereWorldOptimizer,
        qp_control_optimizer: ControllerQP,
    ):
        self.obstacle_container = obstacle_container

        self.qp_control_optimizer = qp_control_optimizer
        self.qp_control_optimizer.barrier_function = BarrierFromObstacleList(
            self.obstacle_container
        )

    def update(self, position, delta_time=0.01):
        position = self.obstacle_container.transform_to_sphereworld(position)

        velocity = self.qp_control_optimizer.evaluate_base_dynamics(position)
        velocity = self.obstacle_container.transform_to_sphereworld(
            velocity, trafo_type="velocity"
        )

        self.obstacle_container.update(position, velocity, delta_time)

    def update_in_sphere_world(self, position, velocity, delta_time):
        self.obstacle_container.update(position, velocity, delta_time)

    def evaluate(self, position):
        position = self.obstacle_container.transform_to_sphereworld(position)
        velocity = self.qp_control_optimizer.get_optimal_control(position)
        # velocity = np.zeros(self.dimension)
        velocity = self.obstacle_container.transform_from_sphereworld(
            velocity, trafo_type="velocity"
        )
        return velocity

    def evaluate_in_sphere_world(self, position):
        velocity = self.qp_control_optimizer.get_optimal_control(position)
        return velocity


def plot_integrate_trajectory(delta_time=0.005, n_steps=1000):
    # start_position = [-4, 4]
    # start_position = [4, 4]
    x_lim = [-5, 5]
    y_lim = [-2, 6.5]

    dimension = 2

    f_x = LinearSystem(A_matrix=np.array([[-6, 0], [0, -1]]))
    g_x = StaticControlDynamics
    # g_x = LinearSystem(A_matrix=np.eye(dimension))

    # barrier_function = DoubleBlobBarrier(
    # blob_matrix=np.array([[10.0, 0.0],
    # [0.0, -1.0]]),
    # center_position=np.array([0.0, 3.0]))

    barrier_function = CirclularBarrier(
        radius=1.0,
        center_position=np.array([0, 3]),
    )

    dynamics = ClosedLoopQP(
        f_x=f_x, g_x=g_x, barrier_function=barrier_function
    )

    start_position_list = [[4, 4], [-4, 4]]

    fig, ax = plt.subplots(figsize=(7.5, 6))

    for start_position in start_position_list:
        position = np.zeros((dimension, n_steps + 1))
        position[:, 0] = start_position
        for ii in range(n_steps):
            vel = dynamics.evaluate(position[:, ii])
            position[:, ii + 1] = position[:, ii] + vel * delta_time

        ax.plot(position[0, :], position[1, :])
    # ax.plot(barrier_function.center_position[0], barrier_function.center_position[1], 'k*')

    ax.plot(0, 0, "k*")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.grid()


def plot_main_vector_field():
    # dimension = 2
    f_x = LinearSystem(A_matrix=np.array([[-6, 0], [0, -1]]))
    # g_x = LinearSystem(A_matrix=np.eye(dimension))

    # closed_loop_ds = ClosedLoopQP(f_x=f_x, g_x=g_x)
    # closed_loop_ds.evaluate_with_control(position, control)

    plot_dynamical_system_streamplot(
        dynamical_system=f_x, x_lim=[-10, 10], y_lim=[-10, 10]
    )


def plot_barrier_function():
    fig, ax = plt.subplots(figsize=(7.5, 6))

    x_lim = [-5, 5]
    y_lim = [-2, 6]

    n_grid = 100

    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], n_grid),
        np.linspace(y_lim[0], y_lim[1], n_grid),
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    values = np.zeros(positions.shape[1])

    barrier_function = DoubleBlobBarrier(
        blob_matrix=np.array([[10, 0], [0, -1]]),
        center_position=np.array([0, 3]),
    )

    barrier_function = CirclularBarrier(
        radius=1.0, center_position=np.array([0, 3])
    )

    for ii in range(positions.shape[1]):
        values[ii] = barrier_function.get_barrier_value(positions[:, ii])

    plt.grid()
    ax.set_aspect("equal", adjustable="box")


def plot_spherial_dynamic_container():
    """Plot surrounding in different actions."""
    x_lim = [-4, 4]
    y_lim = [-4, 6]

    # Set to 1000 as describe din paper.
    sphere_world = SphereWorldOptimizer(lambda_constant=1000)

    sphere_world.append(
        Sphere(
            center_position=np.array([1, 1]),
            radius=0.4,
        )
    )

    sphere_world.append(
        Sphere(
            center_position=np.array([0, 0]),
            radius=3,
            is_boundary=True,
        )
    )

    sphere_world.transform_obstacles_to_sphere_world()

    pos = np.array([0.5, 0.5])
    vel = np.array([0, 0])

    fig, ax = plt.subplots(figsize=(7.5, 6))
    plt.plot(pos[0], pos[1], "bo")

    for ii in range(len(sphere_world)):
        obs = sphere_world.sphere_world_list[ii]
        obs.draw_obstacle()
        boundary_points = obs.boundary_points_global
        plt.plot(boundary_points[0, :], boundary_points[1, :], "k")
        plt.plot(obs.center_position[0], obs.center_position[1], "k+")

    sphere_world.update(position=pos, velocity=vel)

    for ii in range(len(sphere_world)):
        obs = sphere_world.sphere_world_list[ii]
        obs.draw_obstacle()
        boundary_points = obs.boundary_points_global
        plt.plot(boundary_points[0, :], boundary_points[1, :], "g")
        plt.plot(obs.center_position[0], obs.center_position[1], "g+")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


def plot_obstacles_boundary(ax, controller):
    # Initial set up
    for ii in range(len(controller.obstacle_container)):
        # obs = sphere_world.sphere_world_list[ii]
        obs = controller.obstacle_container[ii]
        obs.draw_obstacle()
        boundary_points = obs.boundary_points_global
        ax.plot(boundary_points[0, :], boundary_points[1, :], "k")
        ax.plot(obs.center_position[0], obs.center_position[1], "k+")


def animation_double_worlds(
    start_position, it_max=100, delta_time=0.01, wait_time=0.1
):
    x_lim = [-2, 6]
    y_lim = [-4, 6]
    dimension = 2

    # Set to 1000 as describe din paper.
    sphere_world = SphereWorldOptimizer(
        attractor_position=np.array([0, 0]), lambda_constant=1000
    )

    sphere_world.append(
        DoubleBlob(
            a_value=1,
            b_value=1.1,
            center_position=[0, 3],
            is_boundary=False,
        )
    )

    f_x = LinearSystem(A_matrix=np.array([[-6, 0], [0, -1]]))

    g_x = StaticControlDynamics(A_matrix=np.eye(dimension))
    # g_x = LinearSystem(A_matrix=np.eye(dimension))

    qp_controller = ClosedLoopQP(f_x=f_x, g_x=g_x)
    # Is this the main controller (?!)
    controller = qp_controller

    # controller_sphere = NonconvexAvoidanceCBF(
    # obstacle_container=sphere_world, qp_control_optimizer=qp_controller
    # )

    # fig, ax = plt.subplots(figsize=(7.5, 6))
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 6))

    n_obs_plus_boundary = len(sphere_world)

    trajectory = np.zeros((dimension, it_max + 1))
    # trajectory[:, 0] = start_position

    traj_spher = np.zeros((dimension, it_max + 1))
    traj_spher[:, 0] = controller.obstacle_container.transform_to_sphereworld(
        start_position
    )

    plt_outline = [None] * n_obs_plus_boundary
    plt_center = [None] * (n_obs_plus_boundary - 1)

    # Main loop
    for it in range(it_max):
        ax = axs[it]
        plot_obstacles_boundary(ax, controller)
        # update_in_sphere_world
        # velocity = controller.evaluate(position=trajectory[:, it])
        # velocity = controller.evaluate(position=trajectory[:, it])
        # trajectory[:, it+1] = trajectory[:, it] + velocity*delta_time

        vel_sphere = controller.evaluate_in_sphere_world(
            position=traj_spher[:, it]
        )
        traj_spher[:, it + 1] = traj_spher[:, it] + vel_sphere * delta_time

        trajectory[
            :, it + 1
        ] = controller.obstacle_container.transform_from_sphereworld(
            traj_spher[:, it + 1]
        )

        plot_obstacles_boundary(ax, controller)

        ax.plot(traj_spher[0, : it + 1], traj_spher[1, : it + 1], "r")
        ax.plot(traj_spher[0, 0], traj_spher[1, 0], "r*")
        ax.plot(traj_spher[0, it + 1], traj_spher[1, it + 1], "ro")

        # plt.show()
        # time.sleep(wait_time)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        print(f"Loop #{it}")

        plt.pause(wait_time)

        ax.clear()

        if not len(plt.get_fignums()):
            # No figure active
            print("Animation ended by closing of figures.")
            break

    if False:
        # Plot everything
        for ii in range(len(sphere_world)):
            obs = sphere_world.sphere_world_list[ii]
            obs.draw_obstacle()
            plt_outline[ii] = boundary_points = obs.boundary_points_global
            plt_center[ii] = ax.plot(
                boundary_points[0, :], boundary_points[1, :], "k"
            )

            if not obs.is_boundary:
                ax.plot(obs.center_position[0], obs.center_position[1], "k+")


if (__name__) == "__main__":
    plt.ion()
    solvers.options["show_progress"] = False
    plot_main_vector_field()

    # plot_barrier_function()
    # plot_integrate_trajectory()

    # plot_spherial_dynamic_container()
    # animation_spherical_wold(start_position=np.array([4.0, 5]))

    plt.show()
