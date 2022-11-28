#!/USSR/bin/python3.10
""" Test directional orientation system. """
# Author: Lukas Huber
# Created: 2022-11-27
# Github: hubernikus
# License: M (c) 2022

import math

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.dynamical_systems import DynamicalSystem
from vartools.dynamical_systems import CircularStable

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationXd
from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationTree
from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationSequence

from dynamic_obstacle_avoidance.rotational.datatypes import Vector

from dynamic_obstacle_avoidance.visualization import plot_obstacle_dynamics


class ObstacleRotatedDynamics(DynamicalSystem):
    def __init__(
        self, obstacle_container: ObstacleContainer, initial_dynamics: DynamicalSystem
    ) -> None:
        self.obstacle_container = obstacle_container
        self.initial_dynamics = initial_dynamics

        self.center_dynamics = np.zeros((len(self.obstacle_container), self.dimension))

        self.velocity_scaling = 1
        # Factor by which the dot-product gets multiplied before being applied
        self.dotprod_factor = 5

    def dimension(self) -> int:
        return self.obstacle_container.dimension

    def evaluate(self, position: Vector, update_center_dynamics: bool = True) -> Vector:
        # if not (pos_norm := LA.norm(position)):
        #     # TODO: is this the ideal way to do it (?)
        #     return self.initial_dynamics.evaluate(position)
        gammas = np.array(
            [
                obs.get_gamma(position, in_global_frame=True)
                for obs in self.obstacle_container
            ]
        )

        if np.any((ind_zero := gammas) <= 1):
            weights = ind_zero / np.sum(ind_zero)
            weight_sum = 1

        else:
            weights = 1 / (1 - gammas)
            if (weight_sum := np.sum(weights)) >= 1:
                weights = weights / weight_sum
                weight_sum = 1

        if update_center_dynamics:
            center_directions = np.zeros((len(self.obstacle_container), self.dimension))
            for ii, obs in enumerate(self.obstacle_container):
                if not gammas[ii]:
                    continue

                # We don't normalize before the summing, to keep the 'importance'
                self.center_dynamics[:, ii] = self.initial_dynmics.evaluate(
                    obs.center_position
                )
                center_directions[:, ii] = obs.center_position
                # center_norms = LA.norm(center_dirs, axis=0)
                # ind_nonzero = center_norms > 0
                # center_dirs = center_dirs / np.tile(
                #     center_norms[:, ind_nonzero], (self.dimension, 1)
                # )

                # if not np.all(ind_nonzero):
                #     # TODO: maybe evaluate weight based on magnitude of the velocity
                #     breakpoint()

                # ind_nonzero = center_norms > 0
                # center_dirs[:, ind_nonzero] = center_dirs[:, ind_nonzero] / np.tile(
                #     center_norms[:, ind_nonzero], (self.dimension, 1)
                # )

                # if not np.all(ind_nonzero):
                #     # General unit direction
                #     center_dirs[0, np.logical_not(ind_nonzero)] = 1

                # # self.rotation_tree = VectorRotationTree(
                #     root_id=-1, root_direction=pos_norm
                # )
                # self.rotation_tree.add_node(
                #     node_id=ii,
                #     direction=self.center_dynamics[:, ii]
                #     / LA.norm(self.center_dynamics[:, ii]),
                #     parent_id=-1,
                # )

        # center_positions = np.array([obs.center_position for obs in self.obstacle_container]).T
        # center_dists = LA.norm(np.tile(position, (len(self.obstacle_container), 1)).T - center_positions, axis=0)

        # Get mean-basis
        base0 = np.sum(
            self.center_dynamics * np.tile(weights, (self.dimension, 1)), axis=1
        )
        base1 = np.sum(
            center_directions * np.tile(weights, (self.dimension, 1)), axis=1
        )

        if not (base0_norm := LA.norm(base0)) or not (base1_norm := LA.norm(base1)):
            return self.initial_dynamics.evaluate(position)

        # Surface Weight
        base0 = base0 / base0_norm
        base1 = base1 / base1_norm

        vector_rotation = VectorRotationXd.from_directions(base0, base1)
        weight = max(base0_norm, base1_norm) / self.velocity_scaling
        dot_prod_weight = np.dot(base1, base1) + 1
        dot_prod_weight = max(self.dotprod_factor * dot_prod_weight, 1)
        weight = weight * dot_prod_weight

        rotated_velocity = vector_rotation.rotate(
            self.initial_dynamics.evaluate(position), rot_factor=weight
        )
        return rotated_velocity


def test_local_rotation(visualize=False):
    obstacle_list = ObstacleContainer()  #
    obstacle_list.append(
        Ellipse(
            center_position=np.array([5, 4]),
            axes_length=np.array([4, 6]),
            orientation=90 * math.pi / 180.0,
        )
    )

    circular_ds = CircularStable(radius=10, maximum_velocity=1)
    # rotated_ds = ObstacleRotatedDynamics()

    if visualize:
        plt.close("all")
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_obstacle_dynamics(
            obstacle_container=[],
            dynamics=circular_ds.evaluate,
            x_lim=[-15, 15],
            y_lim=[-15, 15],
            n_grid=30,
            ax=ax,
        )
        ax.scatter(
            0,
            0,
            marker="*",
            s=200,
            color="black",
            zorder=5,
        )

    # Position
    position = np.array([0, 0])
    velocity = rotated_ds.evaluate(position)

    position = np.array([1, 0])


if (__name__) == "__main__":
    figtype = ".png"

    test_local_rotation(visualize=True)
    print("Tests done")
