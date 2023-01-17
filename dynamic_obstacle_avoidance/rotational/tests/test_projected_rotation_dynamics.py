""" Class to Deviate a DS based on an underlying obtacle.
"""
# %%
import sys
import math
import copy
import os
from typing import Optional

# from enum import Enum

import numpy as np
from numpy import linalg as LA
import warnings

from vartools.dynamical_systems import DynamicalSystem
from vartools.linalg import get_orthogonal_basis

# from vartools.linalg import get_orthogonal_basis
# from vartools.directional_space import get_angle_space

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from dynamic_obstacle_avoidance.rotational.rotational_avoidance import (
    obstacle_avoidance_rotational,
)

from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationXd
from dynamic_obstacle_avoidance.rotational.datatypes import Vector
from dynamic_obstacle_avoidance.rotational.dynamics.projected_rotation_dynamics import (
    ProjectedRotationDynamics,
)


def get_environment_obstacle_top_right():
    # attractor_position = np.array([1.0, -1.0])
    attractor_position = np.array([0.0, 0.0])
    obstacle = Ellipse(
        # center_position=np.array([.0, 0.0]),
        center_position=np.array([3.0, 0.0]),
        axes_length=np.array([2, 3.0]),
        orientation=0 * math.pi / 180.0,
    )

    reference_velocity = np.array([2, 0.1])

    dynamics = ProjectedRotationDynamics(
        obstacle=obstacle,
        attractor_position=attractor_position,
        reference_velocity=reference_velocity,
    )

    return dynamics


def _test_base_gamma(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    visualize_vectors: bool = False,
    n_vectors: int = 8,
    save_figure: bool = False,
    **kwargs,
):
    # No explicit test in here, since only getting the gamma value.
    dynamics = get_environment_obstacle_top_right()

    if visualize:
        # x_lim = [-10, 10]
        # y_lim = [-10, 10]

        nx = ny = n_resolution

        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gammas = np.zeros(positions.shape[1])

        ### Basic Obstacle Transformation ###
        for pp in range(positions.shape[1]):
            gammas[pp] = dynamics.obstacle.get_gamma(
                positions[:, pp], in_global_frame=True
            )

        fig, ax = plt.subplots(figsize=figsize)
        cs = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas.reshape(nx, ny),
            cmap="binary_r",
            vmin=1.0,
            levels=np.linspace(1, 10, 9),
        )
        # cbar = fig.colorbar(cs, ticks=np.linspace(1, 11, 6))

        ax.plot(
            dynamics.attractor_position[0],
            dynamics.attractor_position[1],
            "*",
            color=kwargs["attractor_color"],
            linewidth=12,
            markeredgewidth=1.2,
            markersize=15,
            zorder=3,
        )

        # Opposite point
        ax.plot(
            [dynamics.attractor_position[0], x_lim[0]],
            [dynamics.attractor_position[1], dynamics.attractor_position[1]],
            kwargs["linestyle"],
            color=kwargs["opposite_color"],
            linewidth=3,
            zorder=2,
        )
        # ax.plot(
        #     dynamics.obstacle.center_position[0],
        #     dynamics.obstacle.center_position[1],
        #     "+",
        #     color=kwargs[]
        #     linewidth=12,
        #     markeredgewidth=3.0,
        #     markersize=14,
        #     zorder=3,
        # )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        plot_obstacles(
            ax=ax, obstacle_container=[dynamics.obstacle], alpha_obstacle=1.0
        )

        if visualize_vectors:
            # Plot the vectors
            nx = ny = n_vectors

            x_vals, y_vals = np.meshgrid(
                np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
            )
            positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

            for pp in range(positions.shape[1]):
                pos = dynamics._get_position_after_inflating_obstacle(positions[:, pp])
                velocity = dynamics._get_lyapunov_gradient(pos)
                ax.quiver(
                    pos[0],
                    pos[1],
                    velocity[0],
                    velocity[1],
                    color=kwargs["initial_color"],
                    scale=10.0,
                    width=0.01,
                    zorder=1,
                )

                velocity_rotation = dynamics._get_vector_rotation_of_modulation(
                    pos, velocity
                )

                velocity_mod = velocity_rotation.rotate(vety)
                ax.quiver(
                    pos[0],
                    pos[1],
                    velocity_mod[0],
                    velocity_mod[1],
                    color=kwargs["final_color"],
                    scale=10.0,
                    width=0.01,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if save_figure:
                figure_name = "obstacle_original_space"
                fig.savefig(
                    os.path.join(
                        # os.path.dirname(__file__),
                        "figures",
                        figure_name + figtype,
                    ),
                    bbox_inches="tight",
                )


def test_obstacle_inflation(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    n_vectors: int = 8,
    save_figure: bool = False,
    **kwargs,
):
    dynamics = get_environment_obstacle_top_right()

    if visualize:
        # x_lim = [-10, 10]
        # y_lim = [-10, 10]
        nx = ny = n_resolution

        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gammas = np.zeros(positions.shape[1])

        ### Do before the trafo ###
        gammas_shrink = np.zeros_like(gammas)
        for pp in range(positions.shape[1]):
            # Do the reverse operation to obtain an 'even' grid
            pos_shrink = dynamics._get_position_after_inflating_obstacle(
                positions[:, pp]
            )
            gammas_shrink[pp] = dynamics.obstacle.get_gamma(
                pos_shrink, in_global_frame=True
            )

        # Transpose attractor
        attractor_position = dynamics._get_position_after_deflating_obstacle(
            dynamics.attractor_position
        )

        fig, ax = plt.subplots(figsize=figsize)
        cs = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas_shrink.reshape(nx, ny),
            cmap="binary_r",
            vmin=1.0,
            levels=np.linspace(1, 10, 9),
        )

        # cbar = fig.colorbar(cs, ticks=np.linspace(1, 11, 6))

        ax.plot(
            attractor_position[0],
            attractor_position[1],
            "*",
            color=kwargs["attractor_color"],
            linewidth=12,
            markeredgewidth=1.2,
            markersize=15,
            zorder=3,
        )

        ax.plot(
            dynamics.obstacle.center_position[0],
            dynamics.obstacle.center_position[1],
            "+",
            color=kwargs["obstacle_color"],
            linewidth=12,
            markeredgewidth=3.0,
            markersize=14,
            zorder=3,
        )

        # Opposite point
        ax.plot(
            [attractor_position[0], x_lim[0]],
            [attractor_position[1], dynamics.attractor_position[1]],
            kwargs["linestyle"],
            color=kwargs["opposite_color"],
            linewidth=3,
            zorder=2,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        if n_vectors:
            # Plot the vectors
            nx = ny = n_vectors

            x_vals, y_vals = np.meshgrid(
                np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
            )
            positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

            for pp in range(positions.shape[1]):
                pos = dynamics._get_position_after_inflating_obstacle(positions[:, pp])
                velocity = dynamics._get_lyapunov_gradient(pos)
                ax.quiver(
                    positions[0, pp],
                    positions[1, pp],
                    velocity[0],
                    velocity[1],
                    color=kwargs["initial_color"],
                    scale=10.0,
                    width=0.01,
                    zorder=1,
                )

                velocity_rotation = dynamics._get_vector_rotation_of_modulation(
                    pos, velocity
                )

                velocity_mod = velocity_rotation.rotate(velocity)
                ax.quiver(
                    positions[0, pp],
                    positions[1, pp],
                    velocity_mod[0],
                    velocity_mod[1],
                    color=kwargs["final_color"],
                    scale=10.0,
                    width=0.01,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if save_figure:
                figure_name = "obstacle_deflated_space"
                fig.savefig(
                    os.path.join(
                        # os.path.dirname(__file__),
                        "figures",
                        figure_name + figtype,
                    ),
                    bbox_inches="tight",
                )
        return fig, ax

    # # Test point perpendicular
    # position = np.array([-1.5, -5])

    # Deflating close to the obstacle
    position = dynamics.obstacle.center_position + 1e-1
    deflated_position = dynamics._get_position_after_deflating_obstacle(
        position, in_obstacle_frame=False
    )
    assert np.allclose(deflated_position, dynamics.obstacle.center_position)

    # Attractor is outside the obstacle
    position = np.array([0, 0])
    new_position = dynamics._get_position_after_inflating_obstacle(position)
    assert dynamics.obstacle.get_gamma(new_position, in_global_frame=True) > 1

    restored_position = dynamics._get_position_after_deflating_obstacle(new_position)
    assert np.allclose(position, restored_position)

    # Position relatively close
    position = np.copy(dynamics.obstacle.center_position)
    position[0] = position[0] + 4
    new_position = dynamics._get_position_after_inflating_obstacle(position)
    restored_position = dynamics._get_position_after_deflating_obstacle(new_position)
    assert np.allclose(position, restored_position)


def test_inverse_projection_around_obstacle(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    n_vectors=10,
    save_figure=False,
    **kwargs,
):
    dynamics = get_environment_obstacle_top_right()

    if visualize:
        nx = ny = n_resolution

        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gammas = np.zeros(positions.shape[1])

        ### Do before trafo ###
        attractor_position = dynamics._get_position_after_deflating_obstacle(
            dynamics.attractor_position, in_obstacle_frame=False
        )

        gammas_shrink = np.zeros_like(gammas)
        for pp in range(positions.shape[1]):
            # Do the reverse operation to obtain an 'even' grid
            pos_shrink = dynamics._get_unfolded_position_opposite_kernel_point(
                positions[:, pp],
                attractor_position,
                in_obstacle_frame=False,
            )
            pos_shrink = dynamics._get_position_after_inflating_obstacle(
                pos_shrink,
                in_obstacle_frame=False,
                # in_obstacle_frame=False
            )
            gammas_shrink[pp] = dynamics.obstacle.get_gamma(
                pos_shrink, in_global_frame=True
            )

        fig, ax = plt.subplots(figsize=figsize)
        cs = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas_shrink.reshape(nx, ny),
            cmap="binary_r",
            vmin=1.0,
            levels=np.linspace(1, 10, 9),
        )

        # cbar = fig.colorbar(cs, ticks=np.linspace(1, 11, 6))

        # Attractor line
        ax.plot(
            [x_lim[0], x_lim[0]],
            y_lim,
            kwargs["linestyle"],
            color=kwargs["attractor_color"],
            linewidth=7,
            zorder=3,
        )
        # Split lines
        ax.plot(
            x_lim,
            [y_lim[0], y_lim[0]],
            kwargs["linestyle"],
            color=kwargs["opposite_color"],
            linewidth=7,
            zorder=3,
        )
        ax.plot(
            x_lim,
            [y_lim[1], y_lim[1]],
            kwargs["linestyle"],
            color=kwargs["opposite_color"],
            linewidth=7,
            zorder=3,
        )

        ax.plot(
            dynamics.obstacle.center_position[0],
            dynamics.obstacle.center_position[1],
            "+",
            color=kwargs["obstacle_color"],
            linewidth=12,
            markeredgewidth=3.0,
            markersize=14,
            zorder=3,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        # Plot vectors
        if n_vectors:
            # plot the vectors
            nx = ny = n_vectors

            x_vals, y_vals = np.meshgrid(
                np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
            )
            positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
            for pp in range(positions.shape[1]):
                pos = dynamics._get_position_after_deflating_obstacle(positions[:, pp])
                velocity = dynamics._get_projected_lyapunov_gradient(pos)
                ax.quiver(
                    positions[0, pp],
                    positions[1, pp],
                    velocity[0],
                    velocity[1],
                    color=kwargs["initial_color"],
                    scale=10.0,
                    width=0.01,
                    zorder=1,
                )

                velocity_rotation = dynamics._get_vector_rotation_of_modulation(
                    pos, velocity
                )

                velocity_mod = velocity_rotation.rotate(velocity)
                ax.quiver(
                    positions[0, pp],
                    positions[1, pp],
                    velocity_mod[0],
                    velocity_mod[1],
                    color=kwargs["final_color"],
                    scale=10.0,
                    width=0.01,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if save_figure:
                figure_name = "obstacle_projection_deflation"
                fig.savefig(
                    os.path.join(
                        # os.path.dirname(__file__),
                        "figures",
                        figure_name + figtype,
                    ),
                    bbox_inches="tight",
                )

    attractor_position = dynamics._get_position_after_deflating_obstacle(
        dynamics.attractor_position,
        in_obstacle_frame=False,
    )

    position = np.array([-1.5, -5])
    pos_shrink = dynamics._get_unfolded_position_opposite_kernel_point(
        position,
        attractor_position,
        in_obstacle_frame=False,
    )
    # Due to rotation -> projected to the left of attractor
    assert pos_shrink[1] < 1
    assert pos_shrink[0] < attractor_position[0]

    pos_infl = dynamics._get_position_after_inflating_obstacle(
        pos_shrink,
        in_obstacle_frame=False,
    )
    gammas_shrink = dynamics.obstacle.get_gamma(pos_infl, in_global_frame=True)

    # Un-Projected position is (relatively far) stil
    assert gammas_shrink > 3

    ## Position very south of the obstacle gets projected to the attractor (and vice-versa)
    position_start = (
        dynamics.obstacle.center_position
        + (attractor_position - dynamics.obstacle.center_position) * 100
    )
    pos_shrink = dynamics._get_unfolded_position_opposite_kernel_point(
        position_start,
        attractor_position,
        in_obstacle_frame=False,
    )
    assert np.allclose(pos_shrink, attractor_position)

    # center of the obstacle is the 'stable' point
    position = dynamics.obstacle.center_position
    pos_shrink = dynamics._get_unfolded_position_opposite_kernel_point(
        position,
        attractor_position,
        in_obstacle_frame=False,
    )
    assert np.allclose(pos_shrink, position)


def test_inverse_projection_and_deflation_around_obstacle(
    visualize=False,
    x_lim=[-6, 6],
    y_lim=[-6, 6],
    n_resolution=30,
    figsize=(5, 4),
    n_vectors=10,
    save_figure=False,
    **kwargs,
):
    dynamics = get_environment_obstacle_top_right()

    if visualize:
        nx = ny = n_resolution

        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        gammas = np.zeros(positions.shape[1])

        ### do before trafo ###
        attractor_position = dynamics._get_position_after_deflating_obstacle(
            dynamics.attractor_position
        )

        gammas_shrink = np.zeros_like(gammas)
        for pp in range(positions.shape[1]):
            # do the reverse operation to obtain an 'even' grid
            pos_shrink = positions[:, pp]
            # pos_shrink = np.array([3.0, 1.0])
            pos_shrink = dynamics._get_position_after_deflating_obstacle(pos_shrink)

            if np.allclose(pos_shrink, dynamics.obstacle.center_position):
                gammas_shrink[pp] = 1
                continue

            pos_shrink = dynamics._get_unfolded_position_opposite_kernel_point(
                pos_shrink, attractor_position
            )
            pos_shrink = dynamics._get_position_after_inflating_obstacle(pos_shrink)

            gammas_shrink[pp] = dynamics.obstacle.get_gamma(
                pos_shrink, in_global_frame=True
            )

        fig, ax = plt.subplots(figsize=figsize)
        cs = ax.contourf(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            gammas_shrink.reshape(nx, ny),
            cmap="binary_r",
            vmin=1.0,
            levels=np.linspace(1, 10, 9),
            zorder=-1,
        )

        # cbar = fig.colorbar(cs, ticks=np.linspace(1, 11, 6))

        # attractor line
        ax.plot(
            [x_lim[0], x_lim[0]],
            y_lim,
            kwargs["linestyle"],
            color=kwargs["attractor_color"],
            linewidth=7,
            zorder=3,
        )
        # split lines
        ax.plot(
            x_lim,
            [y_lim[0], y_lim[0]],
            kwargs["linestyle"],
            color=kwargs["opposite_color"],
            linewidth=7,
            zorder=3,
        )
        ax.plot(
            x_lim,
            [y_lim[1], y_lim[1]],
            kwargs["linestyle"],
            color=kwargs["opposite_color"],
            linewidth=7,
            zorder=3,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        plot_obstacles(
            ax=ax, obstacle_container=[dynamics.obstacle], alpha_obstacle=1.0
        )

        # Plot vectors
        if n_vectors:
            # plot the vectors
            nx = ny = n_vectors

            x_vals, y_vals = np.meshgrid(
                np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
            )
            positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
            for pp in range(positions.shape[1]):
                pos = dynamics._get_position_after_inflating_obstacle(positions[:, pp])
                velocity = dynamics._get_projected_lyapunov_gradient(pos)
                ax.quiver(
                    pos[0],
                    pos[1],
                    velocity[0],
                    velocity[1],
                    color=kwargs["initial_color"],
                    scale=10.0,
                    width=0.01,
                )

                velocity_rotation = dynamics._get_vector_rotation_of_modulation(
                    pos, velocity
                )

                velocity_mod = velocity_rotation.rotate(velocity)
                ax.quiver(
                    pos[0],
                    pos[1],
                    velocity_mod[0],
                    velocity_mod[1],
                    color=kwargs["final_color"],
                    scale=10.0,
                    width=0.01,
                )

            ax.set_xticks([])
            ax.set_yticks([])

            if save_figure:
                figure_name = "obstacle_projection_inflated"
                fig.savefig(
                    os.path.join(
                        # os.path.dirname(__file__),
                        "figures",
                        figure_name + figtype,
                    ),
                    bbox_inches="tight",
                )


def test_obstacle_partially_rotated():
    # attractor_position = np.array([1.0, -1.0])
    attractor_position = np.array([0.0, 0.0])
    obstacle = Ellipse(
        center_position=np.array([-5.0, 4.0]),
        axes_length=np.array([2, 3.0]),
        orientation=0 * math.pi / 180.0,
    )

    reference_velocity = np.array([2, 0.1])

    dynamics = ProjectedRotationDynamics(
        obstacle=obstacle,
        attractor_position=attractor_position,
        reference_velocity=reference_velocity,
    )

    # test projection
    dist_surf = 1e-6
    pos_close_to_center = copy.deepcopy(dynamics.obstacle.center_position)
    pos_close_to_center[0] = pos_close_to_center[0] + dist_surf

    pos = dynamics._get_position_after_deflating_obstacle(
        pos_close_to_center, in_obstacle_frame=False
    )

    assert np.allclose(pos, dynamics.obstacle.center_position, atol=dist_surf / 2.0)


def test_obstacle_on_x_transformation():
    """tests if the folding / unfolding are bijective, i.e., same start and end point."""
    # simplified environment
    attractor_position = np.array([0.0, 0.0])
    obstacle = Ellipse(
        center_position=np.array([5.0, 0.0]),
        axes_length=np.array([2, 2.0]),
        orientation=0 * math.pi / 180.0,
    )

    reference_velocity = np.array([-1, 0.0])

    dynamics = ProjectedRotationDynamics(
        obstacle=obstacle,
        attractor_position=attractor_position,
        reference_velocity=reference_velocity,
    )

    # check that the folding / unfolding is bijective
    position = np.array([0, 5])
    relative_position = dynamics.obstacle.pose.transform_position_to_relative(position)
    relative_attr_pos = dynamics.obstacle.pose.transform_position_to_relative(
        attractor_position
    )

    trafo_pos = dynamics._get_folded_position_opposite_kernel_point(
        relative_position, attractor_position=relative_attr_pos
    )
    assert np.allclose(trafo_pos, [0, 1])

    reconstructed_pos = dynamics._get_unfolded_position_opposite_kernel_point(
        trafo_pos, attractor_position=relative_attr_pos
    )

    assert np.allclose(relative_position, reconstructed_pos)


def test_transformation_bijection_for_rotated():
    # rotated obstacle
    relative_attr_pos = np.array([0.0, -4.0])
    obstacle = Ellipse(
        center_position=np.array([0.0, 0.0]),
        axes_length=np.array([1.0, 2.0]),
        orientation=30 * math.pi / 180.0,
    )

    reference_velocity = np.array([0, -1])
    dynamics = ProjectedRotationDynamics(
        obstacle=obstacle,
        attractor_position=np.zeros(2),
        reference_velocity=reference_velocity,
    )
    relative_position = np.array([-4.0, -4.0])

    trafo_pos = dynamics._get_folded_position_opposite_kernel_point(
        relative_position, attractor_position=relative_attr_pos
    )
    assert np.allclose(trafo_pos, [-1, 0])

    reconstructed_pos = dynamics._get_unfolded_position_opposite_kernel_point(
        trafo_pos, attractor_position=relative_attr_pos
    )

    assert np.allclose(relative_position, reconstructed_pos)


def test_full_projection_pipeline():
    # Simplest version
    position_attractor = np.array([-2.0, 0])
    obstacle = Ellipse(
        center_position=np.array([0.0, 0.0]),
        axes_length=np.array([1.0, 1.0]),
        orientation=0,
    )
    reference_velocity = np.array([-1.0, 0])

    dynamics = ProjectedRotationDynamics(
        obstacle=obstacle,
        attractor_position=position_attractor,
        reference_velocity=reference_velocity,
    )

    # Point almost at the attractor
    position = np.copy(position_attractor)
    position[0] = position[0] + 1e-8
    projected_position = dynamics.get_projected_position(position)
    assert math.isclose(projected_position[1], 0), "No variation in y..."
    assert projected_position[0] < 10, "Is expected to be projected to large negatives!"

    # Point almost on the surface of the obstacle
    position = np.copy(obstacle.center_position)
    position[0] = position[0] + obstacle.axes_length[0] / 2.0 + 1e-5
    projected_position = dynamics.get_projected_position(position)

    assert np.allclose(
        position, projected_position
    ), "Projection should have little affect close to the obstacles surface."


def test_full_projection_pipeline_challenging():
    # And now: we move to a slightly more difficult setup
    position_attractor = np.array([0.0, 1.0])

    obstacle = Ellipse(
        center_position=np.array([0.0, 3.0]),
        # Equal axes length, to easily place a point on surface
        axes_length=np.array([0.9, 0.9]),
        orientation=40 / 180.0 * math.pi,
    )

    reference_velocity = np.array([-1.0, -2.0])
    dynamics = ProjectedRotationDynamics(
        obstacle=obstacle,
        attractor_position=position_attractor,
        reference_velocity=reference_velocity,
    )

    # Point almost on the surface of the obstacle
    position = np.copy(obstacle.center_position)
    position[1] = position[1] + obstacle.axes_length[1] / 2.0 + 1e-5
    projected_position = dynamics.get_projected_position(position)
    assert np.allclose(
        position, projected_position
    ), "Projection should have little affect close to the obstacles surface."

    # Point almost at the attractor
    position = np.copy(position_attractor)
    position[1] = position[1] + 1e-7
    projected_position = dynamics.get_projected_position(position)
    assert math.isclose(projected_position[0], 0, abs_tol=1e-4), "No variation in x..."
    assert LA.norm(projected_position[1]) > 10, "No variation expected."


if (__name__) == "__main__":
    setup = {
        "attractor_color": "#db6e14",
        "opposite_color": "#96a83d",
        "obstacle_color": "#b35f5b",
        "initial_color": "#a430b3",
        "final_color": "#30a0b3",
        "figsize": (5, 4),
        "x_lim": [-3, 9],
        "y_lim": [-6, 6],
        "n_resolution": 100,
        "n_vectors": 8,
        "linestyle": ":",
    }
    figtype = "png"

    import matplotlib.pyplot as plt

    plt.ion()
    plt.close("all")

    # _test_base_gamma(visualize=True, visualize_vectors=True, save_figure=True, **setup)
    # test_obstacle_inflation(visualize=True, **setup, save_figure=True)

    # test_obstacle_partially_rotated()
    # test_obstacle_on_x_transformation()
    # test_transformation_bijection_for_rotated()
    # test_transformation(visualize=False)

    # test_inverse_projection_around_obstacle(visualize=False)
    # test_inverse_projection_around_obstacle(visualize=True, **setup, save_figure=True)

    test_inverse_projection_around_obstacle(visualize=True, **setup, save_figure=False)
    # test_inverse_projection_around_obstacle(visualize=False, **setup, save_figure=False)

    # test_obstacle_inflation()

    # test_inverse_projection_and_deflation_around_obstacle(
    #     visualize=1, **setup, save_figure=True
    # )

    # test_full_projection_pipeline()
    # test_full_projection_pipeline_challenging()
    print("Tests done.")
