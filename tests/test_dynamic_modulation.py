#!/USSR/bin/python3
""" Script to show lab environment on computer """

# Author: Lukas Huber
# Date: 2020-01-15
# Email: lukas.huber@epfl.ch
import time

import pytest

from dynamic_obstacle_avoidance.obstacles import Ellipse, Polygon, Cuboid
from dynamic_obstacle_avoidance.containers import ShapelyContainer

from dynamic_obstacle_avoidance.avoidance import (
    obs_avoidance_potential_field,
    obs_avoidance_orthogonal_moving,
    obs_avoidance_interpolation_moving,
)


def test_vectorfield_inside_multiagent(visualize=False):
    obs_list = ShapelyContainer()

    obs_list.append(
        Polygon(
            center_position=np.array([5.0, 5.0]),
            absolute_edge_position=False,
            is_boundary=True,
            tail_effect=False,
            edge_points=np.array([[-5.0, -5.0, 5.0, 5.0], [5.0, -5.0, -5.0, 5.0]]),
        )
    )

    obs_list.append(
        Cuboid(
            center_position=np.array([11, 5]),
            axes_length=np.array([12.0, 2.5]),
            tail_effect=False,
        )
    )

    obs_list.append(
        Ellipse(
            center_position=np.array([4.2383868, -0.64125104]),
            orientation=-0.020000000000000004,
            linear_velocity=np.array([0.06447974, 0.47634182]),
            angular_velocity=-0.2,
            axes_length=np.array([1.32235986, 1.15870828]),
        )
    )

    obs_list.append(
        Ellipse(
            center_position=np.array([2.21540644, 6.3835766]),
            orientation=-0.020000000000000004,
            linear_velocity=np.array([0.13479139, -0.70029393]),
            angular_velocity=-0.2,
            axes_length=np.array([1.29421638, 0.70694301]),
        )
    )

    tt = time.time()
    obs_list.update_reference_points()
    dt = time.time() - tt
    print(f"Time for automated reference point: {dt*1000}ms")

    attractor_position = np.array([8.5, 1.3])

    dynamical_system = LinearSystem(attractor_position=attractor_position)

    x_range = [-1.1, 11.1]
    y_range = [-1.1, 11.1]

    start_position = np.array([9, 9])
    it_max = 1000
    delta_time = 0.01

    dim = 2

    position_list = np.zeros((dim, it_max + 1))
    position_list[:, 0] = start_position

    for ii in range(it_max):
        dyn_sys = dynamical_system.evaluate(position_list[:, ii])
        dyn_sys = obs_avoidance_interpolation_moving(
            position_list[:, ii], dyn_sys, obs_list
        )

        position_list[:, ii + 1] = position_list[:, ii] + dyn_sys * delta_time

        if np.allclose(position_list[:, ii], position_list[:, ii + 1]):
            position_list = position_list[:, :ii]
            print(f"Converged after {ii} iterations.")
            break

    fig, ax = plt.subplots(num=1, figsize=(8, 6))
    ax.plot(position_list[0, :], position_list[1, :], "r--", linewidth=4)

    print("Doing Dynamic.")
    Simulation_vectorFields(
        x_range,
        y_range,
        obs=obs_list,
        pos_attractor=dynamical_system.attractor_position,
        # showLabel=False,
        saveFigure=False,
        obs_avoidance_func=obs_avoidance_interpolation_moving,
        noTicks=False,
        automatic_reference_point=False,
        draw_vectorField=True,
        show_streamplot=False,
        fig_and_ax_handle=(fig, ax),
        normalize_vectors=False,
        point_grid=50,
    )

    print("Doing Orthogonal.")
    Simulation_vectorFields(
        x_range,
        y_range,
        obs=obs_list,
        pos_attractor=dynamical_system.attractor_position,
        # showLabel=False,
        saveFigure=False,
        noTicks=False,
        automatic_reference_point=False,
        draw_vectorField=True,
        show_streamplot=False,
        fig_and_ax_handle=(fig, ax),
        normalize_vectors=False,
        point_grid=50,
        obs_avoidance_func=obs_avoidance_orthogonal_moving,
    )

    print("Doing Repulsion.")
    Simulation_vectorFields(
        x_range,
        y_range,
        obs=obs_list,
        pos_attractor=dynamical_system.attractor_position,
        # showLabel=False,
        saveFigure=False,
        noTicks=False,
        automatic_reference_point=False,
        draw_vectorField=True,
        show_streamplot=False,
        fig_and_ax_handle=(fig, ax),
        normalize_vectors=False,
        point_grid=50,
        obs_avoidance_func=obs_avoidance_potential_field,
    )


if (__name__) == "__main__":
    test_vectorfield_inside_multiagent()
