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
from vartools.linalg import get_orthogonal_basis

from dynamic_obstacle_avoidance.utils import compute_weights
from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationXd
from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationTree

from dynamic_obstacle_avoidance.rotational.geometry import get_intersection

from dynamic_obstacle_avoidance.rotational.datatypes import Vector

# TODO:
#   - smoothing to ensure consistency at convergence limit, i.e., add lambda to each branch
#   - use reference direction to do correct decomposition of reference matrix


def get_intersection_with_circle_entry(
    start_position: np.ndarray,
    direction: np.ndarray,
    radius: float,
) -> Optional[np.ndarray]:
    """Returns intersection with circle with center at 0
    of of radius 'radius' and the line defined as 'start_position + x * direction'

    only intersection at closest distance to start_point is returned.
    """
    if not radius:  # Zero radius
        return None

    # Binomial Formula to solve for x in:
    # || dir_reference + x * (delta_dir_conv) || = radius
    AA = np.sum(direction**2)
    BB = 2 * np.dot(direction, start_position)
    CC = np.sum(start_position**2) - radius**2
    DD = BB**2 - 4 * AA * CC

    if DD < 0:
        # No intersection with circle
        return None

    # if only_positive:
    # Only positive direction due to expected negative A (?!) [returns min-distance]..b
    fac_direction = (-BB - np.sqrt(DD)) / (2 * AA)
    point = start_position + fac_direction * direction
    return point


def get_intersection_with_ellipse(
    position, direction, ellipse: Ellipse, in_global_frame: bool = False
) -> Optional[np.ndarray]:
    if in_global_frame:
        # Currently only implemented with ellipse
        position = ellipse.pose.transform_position_to_relative(position)
        direction = ellipse.pose.transform_direction_to_relative(direction)

    # Stretch according to ellipse axes (radius)
    rel_pos = position / ellipse.axes_length
    rel_dir = direction / ellipse.axes_length

    # Intersection with unit circle
    surface_rel_pos = get_intersection_with_circle_entry(
        start_position=rel_pos, direction=rel_dir, radius=0.5
    )

    if surface_rel_pos is None:
        return None

    # Relative
    surface_pos = surface_rel_pos * ellipse.axes_length

    if in_global_frame:
        return ellipse.pose.transform_position_from_relative(surface_pos)

    else:
        return surface_pos


class MultiEllipseObstacle(Obstacle):
    def __init__(self):
        self._obstacle_list = []

        self._root_id: Optional[int] = None
        self._parent_list: list[Optional[int]] = []
        self._children_list: list[list[int]] = []

        self.gamma_power_scaling = 0.5

    @property
    def n_components(self) -> int:
        return len(self._obstacle_list)

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

    def get_tangent_direction(
        self,
        position: Vector,
        velocity: Vector,
        linearized_velocity: Optional[Vector] = None,
    ):

        if linearized_velocity is None:
            base_velocity = self.get_linearized_velocity(
                self._obstacle_list[self._root_id].get_reference_point(
                    in_global_frame=True
                )
            )
        else:
            base_velocity = linearized_velocity

        gamma_values = np.zeros(self.n_components)
        for ii, obs in enumerate(self._obstacle_list):
            gamma_values[ii] = obs.get_gamma(position, in_global_frame=True)
        gamma_weights = compute_weights(gamma_values)

        lambda_values = np.zeros(self.n_components)

        self._tangent_tree = VectorRotationTree()
        self._tangent_tree.set_root(
            root_id=-1,
            direction=velocity,
        )
        self._tangent_tree.add_node(
            parent_id=-1,
            node_id=self._root_id,
            direction=base_velocity,
        )

        # The base node (initial velocity)
        node_list = [-1]

        # for obs_id in it.filterfalse(lambda x: x <= 0, range(len(self._obstacle_list))):
        for obs_id in range(len(self._obstacle_list)):
            if gamma_weights[obs_id] <= 0:
                continue

            node_list.append((obs_id, obs_id))
            self._update_tangent_branch(position, obs_id, base_velocity)

        weights = (
            gamma_weights[gamma_weights > 0]
            * (1 / np.min(gamma_values)) ** self.gamma_power_scaling
        )

        # Remaining weight to the initial velocity
        weights = np.hstack(([1 - np.sum(weights)], weights))

        weighted_tangent = self._tangent_tree.get_weighted_mean(
            node_list=node_list, weights=weights
        )
        # breakpoint()

        return weighted_tangent

    def _update_tangent_branch(
        self, position: Vector, obs_id: int, base_velocity: np.ndarray
    ) -> None:
        surface_points: list(Vector) = [position]
        # normal_directions: List(Vector) = []
        # reference_directions: List(Vector) = []
        parents_tree: List(int) = [obs_id]

        obs = self._obstacle_list[obs_id]
        normal_directions = [obs.get_normal_direction(position, in_global_frame=True)]
        reference_directions = [
            obs.get_reference_direction(position, in_global_frame=True)
        ]

        while parents_tree[-1] != self._root_id:
            obs = self._obstacle_list[parents_tree[-1]]

            new_id = self._parent_list[parents_tree[-1]]
            if new_id is None:
                # TODO: We should not reach this?! -> remove(?)
                breakpoint()
                break

            if len(parents_tree) > 10:
                # TODO: remove this debug check
                raise Exception()

            parents_tree.append(new_id)

            obs_parent = self._obstacle_list[new_id]
            ref_dir = obs.get_reference_point(in_global_frame=True) - surface_points[-1]

            intersection = get_intersection_with_ellipse(
                position, ref_dir, obs_parent, in_global_frame=True
            )

            if intersection is None:
                raise Exception()

            surface_points.append(intersection)

            normal_directions.append(
                obs_parent.get_normal_direction(intersection, in_global_frame=True)
            )

            reference_directions.append(
                obs_parent.get_reference_direction(intersection, in_global_frame=True)
            )

        # Reversely traverse the parent tree - to project tangents
        # First node is connecting to the center-velocity
        self._tangent_tree.add_node(
            node_id=(obs_id, self._root_id),
            parent_id=self._root_id,
            direction=base_velocity,
        )

        # Iterate over all but last one
        tangent = base_velocity
        for ii in reversed(range(len(parents_tree) - 1)):
            rel_id = parents_tree[ii]

            # breakpoint()
            # Re-project tangent
            tangent = self._get_normalized_tangent_component(
                tangent,
                normal=normal_directions[ii],
                reference=reference_directions[ii],
            )

            self._tangent_tree.add_node(
                node_id=(obs_id, rel_id),
                parent_id=(obs_id, parents_tree[ii + 1]),
                direction=tangent,
            )

    def get_linearized_velocity(self, position):
        raise NotImplementedError()

    def _get_normalized_tangent_component(
        self, vector: Vector, normal: Vector, reference: Vector
    ) -> Vector:
        # TODO: use circular-intersection
        basis = get_orthogonal_basis(normal)
        basis[:, 0] = reference

        tmp_tangent = LA.pinv(basis) @ vector
        tmp_tangent[0] = 0  # only in tangent plane
        tangent = basis @ tmp_tangent

        if not (norm_tangent := LA.norm(tangent)):
            raise Exception()

        # rotation_xd = VectorRotationXd.from_directions(normal, vector)
        # if math.isclose(rotation_xd.rotation_angle, math.pi):
        #     raise NotImplementedError()

        # old_tangent = rotation_xd.base[:, 1]
        # # return old_tangent

        # breakpoint()
        # rotation_xd.rotation_angle = math.pi / 2.0
        return tangent / norm_tangent

    def _get_tangent_tree(self):
        pass

    def get_gamma(self, position):
        pass

    def get_normal_direction(self, position):
        pass

    def weights(self):
        pass


def plot_multi_obstacle(multi_obstacle, ax=None, **kwargs):
    plot_obstacles(
        obstacle_container=multi_obstacle._obstacle_list,
        ax=ax,
        **kwargs,
    )


def test_triple_ellipse_environment(visualize=False):
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

    # Example position
    position = np.array([2.0, 6])
    velocity = np.array([1.0, 0])
    linearized_velociy = np.array([1.0, 0])

    if visualize:
        x_lim = [-14, 14]
        y_lim = [-8, 16]
        fig, ax = plt.subplots(figsize=(8, 5))

        plot_obstacles(
            obstacle_container=triple_ellipses._obstacle_list,
            ax=ax,
            x_lim=x_lim,
            y_lim=y_lim,
            draw_reference=True,
            # reference_point_number=True,
            show_obstacle_number=True,
            # ** kwargs,
        )

        plot_obstacle_dynamics(
            obstacle_container=[],
            # obstacle_container=triple_ellipses._obstacle_list,
            dynamics=lambda x: triple_ellipses.get_tangent_direction(
                x, velocity, linearized_velociy
            ),
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            do_quiver=True,
            n_grid=30,
            # vectorfield_color=vf_color,
        )

    # Testing various position around the obstacle
    position = np.array([-1.5, 5])
    averaged_direction = triple_ellipses.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0 and averaged_direction[1] < 0

    position = np.array([-5, 5])
    averaged_direction = triple_ellipses.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0 and averaged_direction[1] > 0

    position = np.array([5.5, 5])
    averaged_direction = triple_ellipses.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0 and averaged_direction[1] < 0

    position = np.array([-5, -0.9])
    averaged_direction = triple_ellipses.get_tangent_direction(
        position, velocity, linearized_velociy
    )
    assert averaged_direction[0] > 0 and averaged_direction[1] < 0


if (__name__) == "__main__":
    import matplotlib.pyplot as plt
    from dynamic_obstacle_avoidance.visualization import plot_obstacles
    from dynamic_obstacle_avoidance.visualization.plot_obstacle_dynamics import (
        plot_obstacle_dynamics,
    )

    # plt.close("all")
    plt.ion()

    test_triple_ellipse_environment(visualize=True)
