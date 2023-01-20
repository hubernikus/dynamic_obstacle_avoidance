"""
Library for the Rotation (Modulation Imitation) of Linear Systems
"""
# Author: Lukas Huber
# GitHub: hubernikus
# Created: 2021-09-01

import json
import warnings
import copy
import math
from functools import partial
from typing import Protocol, Optional

import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

import matplotlib.pyplot as plt

from vartools.math import get_intersection_with_circle
from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_directional_weighted_sum
from vartools.directional_space import (
    get_directional_weighted_sum_from_unit_directions,
)
from vartools.directional_space import get_angle_space, get_angle_space_inverse
from vartools.directional_space import UnitDirection

# from vartools.directional_space DirectionBase
from vartools.dynamical_systems import DynamicalSystem
from vartools.dynamical_systems import AxesFollowingDynamics
from vartools.dynamical_systems import ConstantValue
from vartools.dynamical_systems import CircularStable

from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.utils import get_weight_from_inv_of_gamma
from dynamic_obstacle_avoidance.utils import get_relative_obstacle_velocity

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import Obstacle

from dynamic_obstacle_avoidance.avoidance import BaseAvoider
from dynamic_obstacle_avoidance.rotational.nonlinear_rotation_avoider import (
    NonlinearRotationalAvoider,
)
from dynamic_obstacle_avoidance.rotational.rotation_container import RotationContainer
from dynamic_obstacle_avoidance.rotational.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)


def integrate_trajectory(
    evaluate_function,
    start_position,
    it_max,
    delta_time=0.01,
    conv_margin=1e-3,
    collision_functor=None,
):
    positions = np.zeros((start_position.shape[0], it_max))
    positions[:, 0] = start_position

    for ii in range(1, it_max):
        velocity = evaluate_function(positions[:, ii - 1])

        if LA.norm(velocity) < conv_margin:
            print(f"Reached local minima at it={ii}")
            return positions[:, : (ii - 1)]

        positions[:, ii] = velocity * delta_time + positions[:, ii - 1]

        # Normalize or not(?) -> for now not...
        if collision_functor is not None and collision_functor(positions[:, ii]):
            print(f"Converged at it={ii}")
            return positions[:, :ii]

    print(f"Maximum iterations of {ii} reached.")
    return positions


def create_six_obstacle_environment(distance_scaling=0.3) -> RotationContainer:
    obstacle_environment = RotationContainer()
    obstacle_environment.append(
        Ellipse(
            center_position=np.array([-0.9, 2.0]),
            axes_length=np.array([0.9, 0.9]),
            distance_scaling=distance_scaling,
            # margin_absolut=0.3,
        )
    )

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([0.9, 2.0]),
            # axes_length=np.array([1.4, 1.4]),
            axes_length=np.array([0.9, 0.9]),
            distance_scaling=distance_scaling,
            # margin_absolut=0.3,
        )
    )

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([-2.6, 0.0]),
            axes_length=np.array([0.8, 1.8]),
            distance_scaling=distance_scaling,
        )
    )

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([-1.4, 0.0]),
            axes_length=np.array([0.8, 1.8]),
            distance_scaling=distance_scaling,
        )
    )

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([0.0, -2.0]),
            axes_length=np.array([1.0, 0.5]),
            distance_scaling=distance_scaling,
        )
    )

    obstacle_environment.append(
        Ellipse(
            center_position=np.array([2.0, 0.0]),
            axes_length=np.array([1.0, 0.5]),
            orientation=45.0 / 180 * math.pi,
            distance_scaling=distance_scaling,
        )
    )
    return obstacle_environment


def _test_circular_dynamics_multiobstacle(
    visualize=False,
    n_resolution: int = 20,
    savefig: bool = False,
    traj_integrate: bool = False,
):
    obstacle_environment = create_six_obstacle_environment()
    circular_ds = CircularStable(radius=2.0, maximum_velocity=2.0)

    rotation_projector = ProjectedRotationDynamics(
        attractor_position=circular_ds.pose.position,
        initial_dynamics=circular_ds,
        reference_velocity=lambda x: x - center_velocity.center_position,
    )

    obstacle_avoider = NonlinearRotationalAvoider(
        initial_dynamics=circular_ds,
        # convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=rotation_projector,
    )

    if visualize:
        x_lim = [-4.0, 4.0]
        y_lim = [-3.0, 3.0]

        vf_color = "blue"
        figname = "nonlinear_infinite_dynamics"
        # vf_color = "black"

        figsize = (8.0, 8.0)

        from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
            plot_obstacle_dynamics,
        )
        from dynamic_obstacle_avoidance.visualization import plot_obstacles

        fig, ax = plt.subplots(figsize=figsize)
        plot_obstacle_dynamics(
            obstacle_container=obstacle_environment,
            dynamics=obstacle_avoider.evaluate,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            n_grid=n_resolution,
            do_quiver=True,
            vectorfield_color=vf_color,
        )
        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            # noTicks=True,
            # show_ticks=False,
        )

    if traj_integrate:
        traj_positions = integrate_trajectory(
            obstacle_avoider.evaluate,
            start_position=np.array([-3, 1.5]),
            it_max=2000,
            delta_time=0.01,
        )
        ax.plot(traj_positions[0, :], traj_positions[1, :], color="black")
        traj_positions = integrate_trajectory(
            obstacle_avoider.evaluate,
            start_position=np.array([-0.5, -0.5]),
            it_max=2000,
            delta_time=0.01,
        )
        ax.plot(traj_positions[0, :], traj_positions[1, :], color="black")


def create_start_positions(
    n_grid, x_lim, y_lim, obstacle_environment, datapath, store_to_file=False
):
    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], n_grid),
        np.linspace(y_lim[0], y_lim[1], n_grid),
    )
    positions = np.array([xx.flatten(), yy.flatten()])

    # Keep only positions with no collision
    ind_outside = np.zeros(positions.shape[1], dtype=bool)
    for pp in range(positions.shape[1]):
        if obstacle_environment.get_minimum_gamma(positions[:, pp]) > 1:
            ind_outside[pp] = 1

    positions = positions[:, ind_outside]

    if store_to_file:
        filename = "initial_positions.csv"
        np.savetxt(
            os.path.join(datapath, "data", filename),
            positions.T,
            header="x, y",
            delimiter=",",
        )


def evaluate_trajectories(
    datapath,
    avoidance_functor,
    outputfolder,
    inputfile="initial_positions.csv",
    parameter_file="comparison_parameters.json",
    store_to_file=True,
    collision_functor=None,
):
    # IDEAS: Metrics to use
    # - Distance to circle (desired path)
    # - Average acceleration
    # - Std acceleration
    # - # of switching
    # - # of stuck in local minima
    # - Deviation / error to desired velocity
    with open(os.path.join(datapath, parameter_file)) as user_file:
        simulation_parameters = json.load(user_file)

    simulation_parameters["it_max"] = 10
    #

    parameter_file
    start_positions = np.loadtxt(
        os.path.join(datapath, "data", inputfile),
        delimiter=",",
        dtype=float,
        skiprows=1,
    )
    start_positions = start_positions.T

    output_path = os.path.join(datapath, "data", outputfolder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for pp in range(start_positions.shape[1]):
        trajectory = integrate_trajectory(
            start_position=start_positions[:, pp],
            evaluate_function=avoidance_functor,
            collision_functor=collision_functor,
            **simulation_parameters,
        )

        filename = f"trajectory{pp:03d}.csv"
        np.savetxt(
            os.path.join(output_path, filename),
            trajectory.T,
            header="x, y",
            delimiter=",",
        )


def evaluate_nonlinear_trajectories():
    obstacle_environment = create_six_obstacle_environment()

    circular_ds = CircularStable(radius=2.0, maximum_velocity=2.0)

    rotation_projector = ProjectedRotationDynamics(
        attractor_position=circular_ds.pose.position,
        initial_dynamics=circular_ds,
        reference_velocity=lambda x: x - center_velocity.center_position,
    )

    obstacle_avoider = NonlinearRotationalAvoider(
        initial_dynamics=circular_ds,
        # convergence_system=convergence_dynamics,
        obstacle_environment=obstacle_environment,
        obstacle_convergence=rotation_projector,
    )

    evaluate_trajectories(
        datapath=datapath,
        outputfolder="nonlinear_avoidance",
        avoidance_functor=obstacle_avoider.evaluate,
        collision_functor=lambda x: obstacle_environment.get_minimum_gamma(x) <= 1,
    )


if (__name__) == "__main__":
    datapath = "/home/lukas/Code/dynamic_obstacle_avoidance/dynamic_obstacle_avoidance/rotational/comparison"

    # create_start_positions(
    #     n_grid=10,
    #     x_lim=[-4, 4],
    #     y_lim=[-4, 4],
    #     obstacle_environment=create_six_obstacle_environment(),
    #     datapath=datapath,
    #     store_to_file=True,
    # )

    # evaluate_nonlinear_trajectories()

    _test_circular_dynamics_multiobstacle(visualize=True, n_resolution=20)
