""" Obstacle which is created by consecutive rotation around a graph."""
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-08-01

import math

import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import Obstacle

# from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationXd
from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationTree
from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationSequence

from vartools.dynamical_systems import LinearSystem

Vector = npt.ArrayLike


class GraphRotationObstacle:
    def __init__(self) -> None:
        self._reference_tree = None

        self._rotation_tree = VectorRotationTree()
        self._obstacles = []

    def append_obstacle_part(self, obstacle: Obstacle, is_root=False) -> None:
        self._obstacles.append(obstacle)
        # values >= 1 are used for obstacle-id's, to allow for negative values
        graph_id = len(self._obstacles) + 1

        if is_root:
            self._rotation_tree.set_root(
                graph_id,
            )
            pass
        else:
            pass

    def rotate_with_average(self, initial_direction, node_list, weights):
        pass

    def avoid(self, position: Vector, initial_velocity: Vector):
        # Rotate initial direction
        #  -> obtain initial velocity for each obstacle
        # Rotation modulation (in local frame)
        # Average over all rotated-velocities
        pass


def _test_simple_multiobstacle():
    # Base-normal
    my_obstacle = GraphRotationObstacle()
    my_obstacle.append_obstacle_core(
        Ellipse(position=np.array(0, 1), axes_length=np.array([2, 1]), orientation=0)
    )

    my_obstacle.append_obstacle_part(
        Ellipse(
            position=np.array(1, 0.5),
            axes_length=np.array([2, 1]),
            orientation=math.pi / 2,
        )
    )

    position = np.array([0, 1])
    initial_dynamics = LinearSystem(attractor_position=np.array([0, 0]))

    # modulated_velocity = my_obstacle.avoid(
    #     position, initial_dynamics.evaluate(position)
    # )

    breakpoint()


if (__name__) == "__main__":
    _test_simple_multiobstacle()

    pass
