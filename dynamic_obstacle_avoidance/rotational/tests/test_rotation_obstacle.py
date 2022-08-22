""" Obstacle which is created by consecutive rotation around a graph."""
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-08-01

import math

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import Obstacle

# from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationXd
from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationTree
from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationSequence

from dynamic_obstacle_avoidance.rotational.utils import gamma_normal_gradient_descent

from dynamic_obstacle_avoidance.rotational.datatypes import Vector


class SingleLevelObstacle:
    """Assuming obstacle graph with single level, i.e., one root + many branches."""

    def __init__(self, root_obstacle) -> None:
        self._reference_tree = None
        self._root_obstacle = None

        self._rotation_tree = VectorRotationTree()

        self._obstacles = []
        self.append_obstacle_part(obstacle=root_obstacle, is_root=True)

    # def ind(self, ii):
    #     return ii + 1

    def append_obstacle_part(self, obstacle: Obstacle, is_root: bool = False) -> None:
        self._obstacles.append(obstacle)
        # values >= 1 are used for obstacle-id's, to allow for negative values
        graph_id = len(self._obstacles)

        if is_root:
            direction = np.zeros(obstacle.dimension)
            self._root_obstacle = obstacle

            # Assumption of root being at level=0 -> level=-1 is the normal to the root
            self._rotation_tree.add_node(graph_id, direction=direction, level=0)

            self._root_id = graph_id

        else:
            # Assuming parent is the root (!)
            intersection = gamma_normal_gradient_descent(
                [self._root_obstacle, obstacle]
            )
            obstacle.set_reference_point(intersection, in_global_frame=True)

            direction = intersection - self._root_obstacle.center_position

            self._rotation_tree.add_node(
                graph_id, direction=direction, parent_id=self._root_id
            )

    def rotate_with_average(self, initial_direction, node_list, weights):
        pass

    def avoid(self, position: Vector, initial_velocity: Vector) -> Vector:
        # Get 'projected rotation' at each reference point
        # -> project by observing added rotation w.r.t. initial orientation
        # Weighted rotated sum
        # Rotation modulation (in local frame)
        # Average over all rotated-velocities

        # Check two conditions:
        # 1) is the surface point intersecting
        # 2) is the

        # Unsolved problems:
        # How to define the direction of sub-rotation (?)
        # Where is the initial direction for each subsequent obstacle ?
        pass


def _test_simple_multiobstacle():
    # Base-normal
    obstacle0 = Ellipse(
        center_position=np.array([0, 1]), axes_length=np.array([2, 1]), orientation=0
    )

    my_obstacle = SingleLevelObstacle(obstacle0)

    my_obstacle.append_obstacle_part(
        Ellipse(
            center_position=np.array([1, 0.5]),
            axes_length=np.array([2, 1]),
            orientation=math.pi / 2,
        )
    )

    position = np.array([0, 1])
    initial_dynamics = LinearSystem(attractor_position=np.array([0, 0]))

    modulated_velocity = my_obstacle.avoid(
        position, initial_dynamics.evaluate(position)
    )


if (__name__) == "__main__":
    _test_simple_multiobstacle()

    pass
