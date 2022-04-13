#!/USSR/bin/python3.9
""" Test overrotation for ellipses. """
# Author: Lukas Huber
# Created: 2021-08-04
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import warnings

import unittest
from math import pi

import numpy as np
from numpy import linalg as LA

from vartools.dynamical_systems import LinearSystem, ConstantValue
from vartools.directional_space import UnitDirection, DirectionBase

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.avoidance import obstacle_avoidance_rotational
from dynamic_obstacle_avoidance.avoidance import RotationalAvoider
from dynamic_obstacle_avoidance.avoidance.rotational_avoider import get_intersection_with_circle
from dynamic_obstacle_avoidance.containers import RotationContainer

from dynamic_obstacle_avoidance.visualization import (
    Simulation_vectorFields,
    # plot_obstacles,
)


def test_intersection_with_circle():
    # One Dimensional Circle
    start_position = np.array([0.1])
    radius = 2.1
    direction = np.array([-2])

    direction = direction / LA.norm(direction)
    circle_position = get_intersection_with_circle(
        start_position=start_position,
        direction=direction,
        radius=radius,
    )

    dir_new = circle_position - start_position
    dir_new = dir_new / LA.norm(dir_new)
    assert np.allclose(dir_new, direction)
    assert np.isclose(LA.norm(circle_position), radius)

    # Two Dimensional Circle
    start_position = np.array([0.3, 0.5])
    radius = 1.4
    direction = np.array([3, 1])

    direction = direction / LA.norm(direction)
    circle_position = get_intersection_with_circle(
        start_position=start_position,
        direction=direction,
        radius=radius,
    )

    dir_new = circle_position - start_position
    dir_new = dir_new / LA.norm(dir_new)
    assert np.allclose(dir_new, direction)
    assert np.isclose(LA.norm(circle_position), radius)

    # 1 Dimensional Circle with 2 Points
    # One Dimensional Circle
    start_position = np.array([0.1])
    radius = 2.1
    direction = np.array([-2])

    direction = direction / LA.norm(direction)
    circle_positions = get_intersection_with_circle(
        start_position=start_position,
        direction=direction,
        radius=radius,
        only_positive=False,
    )

    dir_new = circle_positions[:, 0] - start_position
    dir_new = dir_new / LA.norm(dir_new)
    assert np.allclose(dir_new, (-1) * direction)

    dir_new = circle_positions[:, 1] - start_position
    dir_new = dir_new / LA.norm(dir_new)
    assert np.allclose(dir_new, direction)

    # Points are both on boundary
    assert np.isclose(LA.norm(circle_positions[:, 0]), radius)
    assert np.isclose(LA.norm(circle_positions[:, 1]), radius)


def test_rotated_convergence_direction_circle():
    initial_dynamics = LinearSystem(attractor_position=np.array([1.5, 0]))

    # ConvergingDynamics=ConstantValue (initial_velocity)
    obstacle = Ellipse(
        center_position=np.array([0, 0]),
        axes_length=np.array([1, 1]),
    )

    weight = 0.5
    position = np.array([-1.0, 0.5])

    inital_velocity = initial_dynamics.evaluate(position)

    normal = obstacle.get_normal_direction(position, in_global_frame=True)
    norm_base = DirectionBase(vector=normal * (-1))
    
    main_avoider = RotationalAvoider()
    convergence_dir = main_avoider._get_rotated_convergence_direction(
        weight=weight,
        convergence_radius=np.pi / 2.0,
        convergence_vector=inital_velocity,
        reference_vector=obstacle.get_reference_direction(position, in_global_frame=True),
        base=norm_base,
    )

    initial_dir = UnitDirection(norm_base).from_vector(inital_velocity)

    assert convergence_dir.norm() > initial_dir.norm(), \
      "Rotational convergence not further away from norm."

    assert np.cross(inital_velocity, convergence_dir.as_vector()) > 0, \
      "Rotation in the wrong direction."

    # Testing the non-linear 'pulling' (based on linear velocity)
    nonlinear_velocity = inital_velocity
    dir_nonlinear = UnitDirection(norm_base).from_vector(nonlinear_velocity)
    
    nonlinear_conv = main_avoider._get_projected_nonlinear_velocity(
        dir_conv_rotated=convergence_dir,
        dir_nonlinear=dir_nonlinear,
        convergence_radius=np.pi / 2,
        weight=weight
    )

    # assert convergence_dir.as_vector()


def test_rotated_convergence_direction_ellipse():
    initial_dynamics = LinearSystem(attractor_position=np.array([1.5, 0]))

    obstacle = Ellipse(
        center_position=np.array([0, 0]),
        axes_length=np.array([1, 2]),
    )

    weight = 0.5
    position = np.array([-1.0, 0.5])

    inital_velocity = initial_dynamics.evaluate(position)

    normal = obstacle.get_normal_direction(position, in_global_frame=True)
    norm_base = DirectionBase(vector=normal * (-1))
    
    main_avoider = RotationalAvoider()
    convergence_dir = main_avoider._get_rotated_convergence_direction(
        weight=weight,
        convergence_radius=np.pi / 2.0,
        convergence_vector=inital_velocity,
        reference_vector=obstacle.get_reference_direction(position, in_global_frame=True),
        base=norm_base,
    )

    initial_dir = UnitDirection(norm_base).from_vector(inital_velocity)

    assert convergence_dir.norm() > initial_dir.norm(), \
      "Rotational convergence not further away from norm."

    assert np.cross(inital_velocity, convergence_dir.as_vector()) > 0, \
      "Rotation in the wrong direction."



def test_single_circle_linear(visualize=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2, 2]),
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([1.5, 0]))

    obstacle_list.set_convergence_directions(ConvergingDynamics=initial_dynamics)
    # ConvergingDynamics=ConstantValue (initial_velocity)

    main_avoider = RotationalAvoider()

    if visualize:
        # Plot Normals
        Simulation_vectorFields(
            x_lim=[-2, 2],
            y_lim=[-2, 2],
            n_resolution=20,
            obstacle_list=obstacle_list,
            noTicks=False,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=initial_dynamics.evaluate,
            obs_avoidance_func=main_avoider.avoid,
            automatic_reference_point=False,
            pos_attractor=initial_dynamics.attractor_position,
            # Quiver or stream plot
            show_streamplot=False,
            # show_streamplot=False,
        )

    # Velocity on surface is tangent after modulation
    rad_x = np.sqrt(2) / 2 + 1e-9
    position = np.array([(-1) * rad_x, rad_x])
    modulated_velocity = obstacle_avoidance_rotational(
        position, initial_dynamics.evaluate(position), obstacle_list
    )
    normal_dir = obstacle_list[0].get_normal_direction(position, in_global_frame=True)

    assert abs(np.dot(modulated_velocity, normal_dir)) < 1e-6

    # No effect when already pointing away (save circle)
    position = np.array([1, 0])
    modulated_velocity = obstacle_avoidance_rotational(
        position,
        initial_dynamics.evaluate(position),
        obstacle_list,
    )
    assert np.allclose(initial_dynamics.evaluate(position), modulated_velocity)

    # Point far away has no/little influence
    position = np.array([1e10, 0])
    modulated_velocity = obstacle_avoidance_rotational(
        position,
        initial_dynamics.evaluate(position),
        obstacle_list,
    )
    assert np.allclose(initial_dynamics.evaluate(position), modulated_velocity)

    # Decreasing influence with decreasing distance
    position = np.array([-1, 0.1])
    mod_vel = obstacle_avoidance_rotational(
        position, initial_dynamics.evaluate(position), obstacle_list
    )
    mod_vel1 = mod_vel / LA.norm(mod_vel)

    position = np.array([-2, 0.1])
    mod_vel = obstacle_avoidance_rotational(
        position,
        initial_dynamics.evaluate(position),
        obstacle_list
        # position, initial_velocity, obstacle_list
    )
    mod_vel2 = mod_vel / LA.norm(mod_vel)

    # Decreasing influence -> closer to 0 [without magnitude]
    velocity = initial_dynamics.evaluate(position)

    assert np.dot(mod_vel1, velocity) < np.dot(mod_vel2, velocity)

    # Rotate to the left on top
    position = np.array([-1, 0.5])
    initial_velocity = initial_dynamics.evaluate(position)

    modulated_velocity = main_avoider.avoid(
        position=position,
        initial_velocity=initial_velocity,
        obstacle_list=obstacle_list
        )
    assert np.cross(initial_velocity, modulated_velocity) > 0, \
      " Rotation in the wrong direction to avoid the circle."

    # Rotate to the right bellow
    position = np.array([-1, -0.5])
    initial_velocity = initial_dynamics.evaluate(position)

    modulated_velocity = main_avoider.avoid(
        position=position,
        initial_velocity=initial_velocity,
        obstacle_list=obstacle_list
        )
    assert np.cross(initial_velocity, modulated_velocity) < 0, \
      " Rotation in the wrong direction to avoid the circle."


def test_single_perpendicular_ellipse(visualize=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([1, 2]),
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([1.5, 0]))

    obstacle_list.set_convergence_directions(ConvergingDynamics=initial_dynamics)
    # ConvergingDynamics=ConstantValue (initial_velocity)

    main_avoider = RotationalAvoider(
        initial_dynamics=initial_dynamics,
        obstacle_environment=obstacle_list
        )

    if visualize:
        Simulation_vectorFields(
            x_lim=[-2, 2],
            y_lim=[-2, 2],
            n_resolution=20,
            obstacle_list=obstacle_list,
            noTicks=False,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=initial_dynamics.evaluate,
            obs_avoidance_func=main_avoider.avoid,
            automatic_reference_point=False,
            pos_attractor=initial_dynamics.attractor_position,
            # Quiver or Streamplot
            show_streamplot=False,
            # show_streamplot=False,
        )
    position = np.array([-1, 0.5])
    initial_velocity = initial_dynamics.evaluate(position)

    # assert np.cross(initial_velocity, modulated_velocity) > 0, \
      # " Rotation in the wrong direction to avoid the ellipse."

    print("<< Ellipse >>")
    modulated_velocity = main_avoider.avoid(
        position=position,
        initial_velocity=initial_velocity,
        obstacle_list=obstacle_list
        )
    print("Velocities: ")
    print(initial_velocity / LA.norm(initial_velocity))
    print(modulated_velocity / LA.norm(modulated_velocity))

    obstacle_list[-1].axes_length = np.array([1, 1])
    print("<< Circular >>")
    modulated_velocity = main_avoider.avoid(
        position=position,
        initial_velocity=initial_velocity,
        obstacle_list=obstacle_list
    )
    print("Velocities: ")
    print(initial_velocity / LA.norm(initial_velocity))
    print(modulated_velocity / LA.norm(modulated_velocity))


def test_double_ellipse(visualize=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([1, 2]),
            name="center_ellipse",
        )
    )

    obstacle_list.append(
        Ellipse(
            center_position=np.array([1, 0]),
            axes_length=np.array([4, 2]),
            orientation=30 / 180.0 * pi,
        )
    )

    # Arbitrary constant velocity
    initial_velocity = np.array([1, 1])

    obstacle_list.set_convergence_directions(
        ConvergingDynamics=ConstantValue(initial_velocity)
    )

    if visualize:
        # Plot Normals
        Simulation_vectorFields(
            x_lim=[-2, 2],
            y_lim=[-2, 2],
            n_resolution=20,
            obstacle_list=obstacle_list,
            noTicks=False,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=lambda x: initial_velocity,
            obs_avoidance_func=obstacle_avoidance_rotational,
            automatic_reference_point=False,
            # pos_attractor=initial_dynamics.attractor_position,
            # Quiver or Streamplot
            show_streamplot=False,
            # show_streamplot=False,
        )

    # Random evaluation
    position = np.array([-4, 2])
    modulated_velocity = obstacle_avoidance_rotational(
        position, initial_velocity, obstacle_list
    )

    # Normal in either case
    position = np.array([-1, 0])
    modulated_velocity = obstacle_avoidance_rotational(
        position, initial_velocity, obstacle_list
    )

    normal_dir = obstacle_list[0].get_normal_direction(position, in_global_frame=True)
    # assert np.isclose(np.dot(modulated_velocity, normal_dir), 0)


def test_stable_linear_avoidance(visualize=False):
    obstacle_list = RotationContainer()
    obstacle_list.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([2, 2]),
        )
    )

    # Arbitrary constant velocity
    initial_dynamics = LinearSystem(attractor_position=np.array([1.5, 0]))

    obstacle_list.set_convergence_directions(ConvergingDynamics=initial_dynamics)
    # ConvergingDynamics=ConstantValue (initial_velocity)

    if visualize:
        # Plot Normals
        Simulation_vectorFields(
            x_lim=[-2, 2],
            y_lim=[-2, 2],
            n_resolution=20,
            obstacle_list=obstacle_list,
            noTicks=False,
            showLabel=False,
            draw_vectorField=True,
            dynamical_system=initial_dynamics.evaluate,
            obs_avoidance_func=obstacle_avoidance_rotational,
            automatic_reference_point=False,
            pos_attractor=initial_dynamics.attractor_position,
            # Quiver or Streamplot
            show_streamplot=False,
            # show_streamplot=False,
        )



if (__name__) == "__main__":
    test_intersection_with_circle()

    test_rotated_convergence_direction_circle()
    test_rotated_convergence_direction_ellipse()
    
    test_single_circle_linear(visualize=False)
    
    # test_single_perpendicular_ellipse(visualize=True)
    
    # test_double_ellipse(visualize=True)
    # test_stable_linear_avoidance(visualize=False)

    print("[Rotational Tests] Done tests")
