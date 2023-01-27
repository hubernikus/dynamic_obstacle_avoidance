"""
Multiple Ellipse in One Obstacle

for now limited to 2D (in order to find intersections easily).
"""

import math
from typing import Optional
import itertools as it

import numpy as np
from numpy import linalg as LA

from vartools.math import get_intersection_with_circle

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationXd
from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationTree

from dynamic_obstacle_avoidance.rotational.geometry import get_intersection

from dynamic_obstacle_avoidance.rotational.datatypes import Vector


def get_intersection_with_ellipse(
    position, direction, obstacle: Ellipse, in_global_frame: bool = False
) -> Optional[np.ndarray]:
    if in_global_frame:
        # Currently only implemented with ellipse
        position = obstacle.pose.transform_position_to_relative(position)
        direction = obstacle.pose.transform_direction_to_relative(direction)

    # Stretch according to ellipse axes (radius)
    rel_pos = position / ellipse.axes_length * 2
    rel_dir = direction / ellipse.axes_length * 2

    # Intersection with unit circle
    surface_rel_pos = get_intersection_with_circle(
        start_position=rel_pos_stretch, direction=rel_dir_stretch, radius=1.0
    )

    if surface_rel_pos is None:
        return None

    # Relative
    surface_pos = surface_rel_pos * ellipse.axes_length * 0.5

    if in_global_frame:
        return obstacle.pose.transform_position_from_relative(surface_pos)

    else:
        return surface_pos


class MultiEllipseObstacle(Obstacle):
    def __init__(self):
        self._obstacle_list = []

        self._root_id: Optional[int] = None
        self._parent_list: list[Optional[int]] = []
        self._children_list: list[list[int]] = []
        pass

    def set_root(self, obs_id: int):
        if self._root_id:
            raise NotImplementedError("Make sure to delete first.")
        self._root_id = obs_id
        self._parent_list[obs_id] = -1

    def set_parent(self, obs_id: int, parent_id: int):
        # This should go automatically at run-time
        if self._parent_list[obs_id]:
            raise NotImplementedError("Make sure to delete first.")

        self._parent_list[obs_id] = parent_id
        self._children_list[parent_id].append(obs_id)

        # Set reference point
        intersection = get_intersection(
            self._obstacle_list[obs_id], self._obstacle_list[parent_id]
        )

        self._obstacle_list[obs_id].set_reference_point(
            intersection, in_global_frame=True
        )

    def append(self, obstacle: Obstacle) -> None:
        self._obstacle_list.append(obstacle)
        self._children_list.append([])
        self._parent_list.append(None)

    def delete_item(self, obs_id: int):
        raise NotImplementedError()

    def get_tangent_direction(self, position, velocity):
        base_velocity = self.get_linearized_velocity(
            self._obstacle_list[self._root_id].center_position
        )

        gamma_values = np.zeros(self.n_components)
        for ii, obs in enumerate(self._obstacle_list):
            gamma_values[ii] = obs.get_gamma(position, in_global_frame=True)
        gamma_weights = self.get_gamma_weights(gamma_values)

        lambda_values = np.zeros(self.n_components)

        self._tangent_tree = VectorRotationTree()
        self._tangent_tree.set_root(
            root_id=-1,
            direction=velocity,
        )
        self._tangent_tree.add_node(
            node_id=self._root_id,
            direction=base_velocity,
        )

        node_list = []
        # for obs_id in it.filterfalse(lambda x: x <= 0, range(len(self._obstacle_list))):
        for obs_id in range(len(self._obstacle_list)):
            node_list.append((obs_id, self._root_id))

            if gamma_weights[obs_id] <= 0:
                continue

            # TODO: What about filtertrue(?)
            self._update_tangent_branch(obs_id, base_velocity)

        weighted_mean = self._tangent_tree.get_weighted_mean(
            node_list=node_list, weights=gamma_weights
        )

        return weighted_mean

    def _update_tangent_branch(self, obs_id: int, base_velocity: np.ndarray) -> None:
        surface_points: list(Vector) = [position]
        normal_directions: List(Vector) = []
        parents_tree: List(int) = [obs_id]

        obs = self._parent_list[parents_tree[-1]]
        normal_directions = [obs.get_normal_direction(position, in_global_frame=True)]

        while parents_tree[-1] != self._root_id:
            parents_tree.append(self._parent_list[parents_tree[-1]])

            if len(parents_tree) > 10:
                # TODO: remove this debug check
                raise Exception()

            obs_parent = self._parent_list[parents_tree[-1]]
            ref_dir = (
                obs_parent.get_reference_point(in_global_frame=True) - surface_points
            )
            intersection = get_intersection_with_ellipse(
                position, ref_dir, obs, in_global_frame=True
            )

            if intersection is None:
                raise Exception()

            surface_points.append(intersection)

            normal_directions = obs.get_normal_direction(
                intersection, in_global_frame=True
            )

        # Reversely traverse the parent tree - to project tangents
        self._tangent_tree.add_node(
            node_id=(obs_id, self._root_id),
            parent_id=self._root_id,
            direction=base_velocity,
        )

        breakpoint()
        # Iterate over all but last one
        tangent = base_velocity
        for ii, rel_id in reversed(enumerate(parents_tree[:-1])):
            tangent = self._get_normalized_tangent_component(
                tangent, normal_directions[ii]
            )

            self._tangent_tree.add_node(
                node_id=(obs_id, rel_id),
                parent_id=(obs_id, parents_tree[ii + 1]),
                direction=base_velocity,
            )

    def _get_normalized_tangent_component(
        self, vector: Vector, normal: Vector
    ) -> Vector:
        rotation_xd = VectorRotationXd.from_directions(vector, normal)

        if math.isclose(rotation_xd, math.pi):
            raise NotImplementedError()

        # rotation_xd.rotation_angle = math.pi / 2.0
        return rotation_xd.base[:, 1]

    def _get_tangent_tree(self):
        pass

    def get_gamma(self, position):
        pass

    def get_normal_direction(self, position):
        pass

    def get_tangent_direction(self, position):
        pass

    def weights(self):
        pass


def plot_multi_obstacle(multi_obstacle, ax=None, **kwargs):
    plot_obstacles(
        obstacle_container=multi_obstacle._obstacle_list,
        ax=ax,
        **kwargs,
    )


def _test_triple_ellipse_environment(visualize=False):
    triple_ellipses = MultiEllipseObstacle()
    triple_ellipses.append(
        Ellipse(
            center_position=np.array([0, 0]),
            axes_length=np.array([8, 3.0]),
            orientation=0,
        )
    )

    triple_ellipses.append(
        Ellipse(
            center_position=np.array([-3.4, 3.4]),
            axes_length=np.array([8, 3.0]),
            orientation=90 * math.pi / 180.0,
        )
    )

    triple_ellipses.append(
        Ellipse(
            center_position=np.array([3.4, 3.4]),
            axes_length=np.array([8, 3.0]),
            orientation=-90 * math.pi / 180.0,
        )
    )

    triple_ellipses.set_root(obs_id=0)
    triple_ellipses.set_parent(obs_id=1, parent_id=0)
    triple_ellipses.set_parent(obs_id=2, parent_id=0)

    if visualize:
        plot_obstacles(
            obstacle_container=triple_ellipses._obstacle_list,
            # ax=ax,
            x_lim=[-6, 6],
            y_lim=[-3, 9],
            draw_reference=True,
            # ** kwargs,
        )

    position = np.array([])


if (__name__) == "__main__":
    _test_triple_ellipse_environment(visualize=True)
