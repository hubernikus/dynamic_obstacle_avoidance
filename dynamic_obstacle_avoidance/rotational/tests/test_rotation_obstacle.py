""" Obstacle which is created by consecutive rotation around a graph."""
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-08-01

import math

import numpy as np
from numpy import linalg as LA
import numpy.typing as npt

import matplotlib.pyplot as plt

from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import Obstacle

# from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationXd
from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationTree
from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationSequence

from dynamic_obstacle_avoidance.rotational.utils import gamma_normal_gradient_descent

Vector = npt.ArrayLike


class SingleLevelObtacle:
    """Assuming obstacle graph with single level, i.e., one root + many branches."""

    def __init__(self) -> None:
        # self._obstacles: list[Obstacle] = []
        self._rotation_tree = VectorRotationTree()

        self._root_obs: Obstacle = None
        self._obstacles = []

        self._id_counter = 0

    def append_obstacle_part(self, obstacle: Obstacle, is_root: bool = False) -> None:
        # values >= 1 are used for obstacle-id's, to allow for negative values
        self._obstacles.append(obstacle)
        graph_id = len(self._obstacles) + 1

        self._id_counter += 1

        if is_root:
            direction = np.zeros(obstacle.dimension)
            self._root_obs = obstacle
            self._root_id = graph_id
            self._rotation_tree.set_root(graph_id, direction=direction)
            return

        # Assuming parent is the root (!)
        intersection = gamma_normal_gradient_descent([self._root_obs, obstacle])
        obstacle.set_reference_point(intersection, in_global_frame=True)

        direction = intersection - self._root_obs.center_position
        self._rotation_tree.add_node(
            graph_id, parent_id=self._root_id, direction=direction
        )
        breakpoint()

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
        # 2) (hello)
        for obstacle in self._obstacles:
            if obstacle is self._root_obs:  # Reference check
                pass

        pass


def _test_simple_multiobstacle():
    # Base-normal

    aa_list = [-100, 20]

    for ii, aa in enumerate(aa_list):
        print(ii, aa)

    my_obstacle = SingleLevelObtacle()

    my_obstacle.append_obstacle_part(
        Ellipse(
            center_position=np.array([0, 1]),
            axes_length=np.array([2, 1]),
            orientation=0,
        ),
        is_root=True,
    )

    my_obstacle.append_obstacle_part(
        Ellipse(
            center_position=np.array([1, 0.5]),
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
