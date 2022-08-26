"""
Examples of how to use an obstacle-boundary mix,
i.e., an obstacle which can be entered

This could be bag in task-space, or a complex-learned obstacle.
"""
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-06-21

import copy
import logging
import warnings
import math

import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt

import networkx as nx
from networkx import shortest_path, shortest_path_length

# from networkx import to_dict_of_dicts, to_edgelist

from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_directional_weighted_sum
from vartools.dynamical_systems import plot_dynamical_system, LinearSystem

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid

from dynamic_obstacle_avoidance.containers import BaseContainer

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from dynamic_obstacle_avoidance.rotational.utils import gamma_normal_gradient_descent

from dynamic_obstacle_avoidance.rotational.datatypes import Vector, Orientation


def get_property_of_node_edge(graph, node, key) -> Vector:
    """Returns value of given property of the corresponding node in a networkx.graph,
    assuming the node has only one edge (in case of several edges the first one
    is taken randomly)."""
    # temp_list is needed (otherwise an error is thrown)
    temp_list = list(graph.edges([node], data=True))
    return temp_list[0][2][key]


class MultiHullAndObstacle(Obstacle):
    """This is similar to an 'basic' obstacle but requires some different structure / treatment.

    Note that this is an obstacle made out of (sub-)obstacles!

    Attributes
    ----------
    inner_obstacles: List of obstacle which form the inner hull.
    outer_obstacles: For now the outer_obstacle is limited to a single obstacle (concave).
    """

    # TODO: this could be refactored to be in the outer-obstacle-frame
    def __init__(
        self,
        inner_obstacles: list,
        outer_obstacle: Obstacle,
        center_position: Vector = None,
        orientation: Orientation = None,
        **kwargs,
    ):
        if center_position is None:
            center_position = np.zeros(outer_obstacle.center_position.shape)

        super().__init__(center_position=center_position, orientation=orientation)

        # This obstacle is duality of boundary - obstacle (depending on position)
        self.is_boundary = None

        # Graph to the inner obstacle
        self._graph = None
        self.inner_obstacles = inner_obstacles

        self.outer_obstacle = outer_obstacle
        self._local_outside_attractor = None
        self._attractor_is_outside = None
        # Store the entrances and the position seperately
        # self._entrance_positions = []
        # self._entrance_obstacles = []

        self._entrance_counter = 0

    @property
    def dimension(self) -> int:
        return self.outer_obstacle.dimension

    @property
    def all_obstacles(self) -> list:
        return self.inner_obstacles + [self.outer_obstacle]

    @property
    def n_elements(self) -> int:
        return len(self.inner_obstacles) + 1

    @property
    def _indices_outer(self) -> int:
        # return np.arange(0, len(self.outer_obstacles))
        return len(self.inner_obstacles)

    @property
    def _indices_inner(self) -> np.array:
        return np.arange(self.n_elements - 1)

    @property
    def all_hash_list(self) -> list:
        return self.inner_obstacles + [ii for ii in range(self._entrance_counter)]

    def set_attractor(self, position: Vector, in_global_frame=False) -> None:
        """Set attractor in the global frame.
        for the moment we assume linear dynamics.
        The option to add local dynamics is possible.

        Additionally reset the 'local_attractor', since the attractor position
        has changed."""
        # position = self.transform_global2relative(position)

        if in_global_frame:
            position = self.transform_global2relative(position)

        in_free_space = False  # Track if the attractor is in free space!

        if self.outer_obstacle.is_inside(position, in_global_frame=True):
            # Just to it for the first 'outside'-node
            # => since the behavior outside should be consistent
            self._local_outside_attractor = None
            self._attractor_is_outside = False

        else:
            self._local_outside_attractor = position
            self._attractor_is_outside = True

            in_free_space = True

        for ii, obs in enumerate(self.inner_obstacles):

            if obs.is_inside(position, in_global_frame=True):
                self._graph.nodes[obs]["local_attractor"] = None
                self._graph.nodes[obs]["contains_attractor"] = False

            else:
                self._graph.nodes[obs]["local_attractor"] = position
                self._graph.nodes[obs]["contains_attractor"] = True

                in_free_space = True

        if not in_free_space:
            # Attractor is not in free space
            # Find the closest one instead
            warnings.warn("Attractor is in obstacle-wall.")

            self._evaluate_weights(position)
            ind_max = np.argmin(self.weights)
            if ind_max == self._indices_outer:
                self._local_outside_attractor = position
                self._attractor_is_outside = True

            else:
                obs = self.inner_obstacles[ind_max]
                self._graph.nodes[obs]["local_attractor"] = position
                self._graph.nodes[obs]["contains_attractor"] = True

    def is_inside(self, position, in_global_frame=True):
        """Checks if the obstacle is colliding."""
        in_free_space = False  # Track if the attractor is in free space!

        if not self.outer_obstacle.is_inside(position, in_global_frame=True):
            in_free_space = True

        for ii, obs in enumerate(self.inner_obstacles):
            if not obs.is_inside(position, in_global_frame=True):
                in_free_space = True

        return not in_free_space

    def evaluate(self, position: Vector, in_global_frame: bool = True) -> Vector:
        """Avoids the boundary-hull obstacle."""
        # Assumption that graph is already constructed
        if in_global_frame:
            position = self.transform_global2relative(position)

        self._evaluate_weights(position)

        if not np.sum(self.weights):
            # All weights are zero
            return np.zeros(self.dimension)

        weights_ = []
        velocities_ = []

        for ii, obs in enumerate(self.inner_obstacles):
            if not self.weights[ii]:
                continue

            weights_.append(self.weights[ii])

            if self._graph.nodes[obs]["local_attractor"] is None:
                self.update_shortest_attractor_path(obs)

            velocities_.append(
                self._get_local_dynamics(
                    position,
                    obs_hash=obs,
                    gamma=self.gamma_list[ii],
                )
            )

        if self.weights[self._indices_outer]:
            weights_.append(self.weights[self._indices_outer])

            if self._local_outside_attractor is None:
                self.update_shortest_attractor_path_starting_from_outside(position)

            velocities_.append(
                self._get_local_dynamics(
                    position,
                    obs_hash=None,
                    # weight=weights_[-1],
                    gamma=self.gamma_list[-1],
                )
            )

        velocities_ = np.array(velocities_).T

        mean_direction = np.sum(
            velocities_ * np.tile(weights_, (self.dimension, 1)), axis=1
        )

        mean_magnitude = LA.norm(velocities_, axis=0)
        mean_magnitude = np.sum(mean_magnitude * np.array(weights_))

        dir_norm = LA.norm(mean_direction)
        if not dir_norm and np.min(mean_magnitude):
            # Direction is zero, but not initial one
            breakpoint()
            raise ValueError("Zero direction value obtained.")

        mean_direction = mean_direction / dir_norm * mean_magnitude

        if in_global_frame:
            mean_direction = self.transform_global2relative_dir(mean_direction)

        return mean_direction

    def _get_local_dynamics(
        self,
        position: Vector,
        gamma: float,
        obs_hash: Obstacle = None,
        scaling: float = 1,
        max_velocity: float = 1,
    ) -> Vector:

        # Compute and stretch the linear dynamics
        if obs_hash is None:
            linear_velocity = self._local_outside_attractor - position
            obs = self.outer_obstacle

        else:
            linear_velocity = self._graph.nodes[obs_hash]["local_attractor"] - position
            obs = obs_hash

        linear_velocity = linear_velocity * scaling

        vel_norm = LA.norm(linear_velocity)

        # Rotational Modulation
        normal = obs.get_normal_direction(position, in_global_frame=True)
        tangent_base = get_orthogonal_basis(normal)
        tangent_base[:, 0] = obs.get_reference_direction(position, in_global_frame=True)

        tangent_velocity = LA.pinv(tangent_base) @ linear_velocity
        tangent_velocity[0] = 0
        tangent_velocity = tangent_base @ tangent_velocity

        # Get 'lambda' or the weight of correction
        weight = 1 / gamma

        mean_direction = get_directional_weighted_sum(
            null_direction=obs.get_reference_direction(position, in_global_frame=True),
            weights=np.array([1 - weight, weight]),
            directions=np.vstack((linear_velocity, tangent_velocity)).T,
        )

        if vel_norm > max_velocity:
            linear_velocity = mean_direction / LA.norm(mean_direction) * max_velocity
        else:
            linear_velocity = mean_direction / LA.norm(mean_direction) * vel_norm

        return mean_direction

    def update_shortest_attractor_path_starting_from_outside(
        self, position: Vector
    ) -> None:
        distances = []
        entrances = []
        targets = []

        for ii in range(self._entrance_counter):
            dist_to_entrance = LA.norm(
                position - get_property_of_node_edge(self._graph, ii, "intersection")
            )

            path_dist = shortest_path_length(self._graph, source=ii, weight="distance")
            for obs in self.inner_obstacles:
                distances.append(path_dist[ii] + dist_to_entrance)
                entrances.append(ii)
                targets.append(obs)

        try:
            ind_min = np.argmin(distances)
        except:
            breakpoint()

        path = shortest_path(
            self._graph,
            source=entrances[ind_min],
            target=targets[ind_min],
            weight="distance",
        )

        self._assign_local_attractors_along_path(path)

    def update_shortest_attractor_path(self, obs_hash_source) -> None:
        closest_dist = None
        path_dist = shortest_path_length(
            self._graph, source=obs_hash_source, weight="distance"
        )
        closest_dist = []
        closest_elems = []

        # Check the outsiders
        for obs in self.inner_obstacles:
            # Self check (i.e. obs == obs_source) is not needed, since
            # obs_source has "contains_attractor" == False

            if not self._graph.nodes[obs]["contains_attractor"]:
                continue

            closest_dist.append(path_dist)
            closest_elems.append(obs)

        if self._attractor_is_outside:
            for ii in range(self._entrance_counter):
                dist_attractor = LA.norm(
                    get_property_of_node_edge(self._graph, ii, "intersection")
                    - self._local_outside_attractor
                )

                closest_dist.append(path_dist[ii] + dist_attractor)
                closest_elems.append(ii)

        # Get the shortest path and update
        ind_min = np.argmin(closest_dist)

        path = shortest_path(
            self._graph,
            source=obs_hash_source,
            target=closest_elems[ind_min],
            weight="distance",
        )

        self._assign_local_attractors_along_path(path)

    def _assign_local_attractors_along_path(self, path: list):
        """Input is a NetworXd-paht (list of the node-id's)."""
        for ii in range(0, len(path) - 1):
            # Treat 'int' obstacles differently, since they represent the entrance
            # coming from the same obstacle
            if isinstance(path[ii], int):
                ind0 = 0
                obs = self.outer_obstacle
                attractor = self._local_outside_attractor

            else:
                ind0 = obs = path[ii]
                attractor = self._graph.nodes[ind0]["local_attractor"]

            if attractor is not None:
                # If the first assigned one is found, it implies that the shortest path
                # subsequent shortest path is already assigned
                break

            # Set projected connection element
            ind1 = 0 if isinstance(path[ii + 1], int) else path[ii + 1]

            local_attractor = self._graph.edges[ind0, ind1]["intersection"]
            local_attractor = obs.get_point_on_surface(
                local_attractor, in_global_frame=True
            )

            if isinstance(path[ii], int):
                self._local_outside_attractor = local_attractor
            else:
                self._graph.nodes[ind0]["local_attractor"] = local_attractor

    def _evaluate_weights(
        self,
        position: Vector,
        mult_power_weight: float = 3.0,
        max_power_weight: float = 5.0,
    ) -> None:
        """Position input is in local-frame."""
        self.gamma_list = np.zeros(self.n_elements)
        self.weights = np.zeros(self.n_elements)

        for ii, obs_ii in enumerate(self.all_obstacles):
            self.gamma_list[ii] = obs_ii.get_gamma(position, in_global_frame=True)
        self.weights = np.maximum(self.gamma_list - 1, 0)

        # self.gamma_weight = 1 - 1 / self.gamma_list

        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            # At least one weight has to be bigger than one
            self.weights /= weight_sum

    def plot_obstacle(
        self,
        ax=None,
        x_lim: list = None,
        y_lim: list = None,
        plot_attractors: bool = False,
        outder_linealpha: float = 0.9,
        inner_linealpha: float = 0.9,
    ) -> None:
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        temp_obs = copy.deepcopy(self.outer_obstacle)
        temp_obs.pose = self.pose.transform_pose_to_relative(temp_obs.pose)

        plot_obstacles(
            obstacle_container=[temp_obs],
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            linealpha=outder_linealpha,
            # alpha_obstacle=1.0,
        )

        temp_list = copy.deepcopy(self.inner_obstacles)
        for obs in temp_list:
            obs.is_boundary = False
            obs.pose = self.pose.transform_pose_to_relative(obs.pose)

        plot_obstacles(
            obstacle_container=temp_list,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            alpha_obstacle=1.0,
            obstacle_color="white",
            # draw_reference=True,
            linecolor="#808080",
            linealpha=inner_linealpha,
            draw_center=False,
        )

        if plot_attractors:
            # for obs in self.inner_obstacles:
            for local_attractor in self.local_attractor_iterator():
                if local_attractor is None:
                    continue
                # self._graph.nodes[0]["local_attractor"] = None
                # self._graph.nodes[0]["contains_attractor"] = False

                ax.plot(
                    local_attractor[0],
                    local_attractor[1],
                    "*",
                    color="#808080",  # gray
                    linewidth=11,
                    markersize=11,
                    zorder=4,
                )

            attractor_ = self.get_attractor()
            ax.plot(
                attractor_[0],
                attractor_[1],
                "*",
                color="black",
                linewidth=12,
                markersize=12,
                zorder=4,
            )

    def get_attractor(self):
        """Find the attractor and return it."""
        if self._attractor_is_outside:
            return self._local_outside_attractor

        for obs in self.inner_obstacles:
            if self._graph.nodes[obs]["contains_attractor"]:
                return self._graph.nodes[obs]["local_attractor"]

    def local_attractor_iterator(self):
        for obs in self.inner_obstacles:
            yield self._graph.nodes[obs]["local_attractor"]

        yield self._local_outside_attractor

    def get_gamma(self, position: Vector, in_global_frame: bool = False) -> float:
        """Get distance value with respect to the obstacle."""
        # if in_global_frame:
        #     position = self.transform_global2relative(position)
        raise NotImplementedError()

    def get_normal_direction(
        self, position: Vector, in_global_frame: bool = False
    ) -> float:
        raise NotImplementedError()

    def _update_intersections_with_outer_obstacle(self):
        """Get intersection with the 'free' space"""

        # Reset the counter
        self._entrance_counter = 0

        for ii, obs_ii in enumerate(self.inner_obstacles):
            if LA.norm(obs_ii.center_position - self.outer_obstacle.center_position) < (
                np.min(self.outer_obstacle.axes_length) - np.max(obs_ii.axes_length)
            ):
                continue

            position = gamma_normal_gradient_descent(
                [obs_ii, self.outer_obstacle],
                powers=[-2, -2],  # -> both in free space, i.e. < 0
                factors=[-1, 1],
            )

            if obs_ii.is_inside(position) or obs_ii.is_inside(position):
                continue

            self._graph.add_node(
                self._entrance_counter,
            )
            self._graph.add_edge(
                self._entrance_counter,
                obs_ii,
                distance=LA.norm(position - obs_ii.center_position),
                intersection=position,
            )
            self._entrance_counter += 1

    def evaluate_hirarchy_and_reference_points(self) -> None:
        self._graph = nx.Graph()
        self._graph.add_nodes_from(self.inner_obstacles)

        self._update_intersections_with_outer_obstacle()

        # Define entering & exit points
        self._exit_obstacles = []
        self._exit_references = []

        for ii, obs_ii in enumerate(self.inner_obstacles):
            for jj in range(ii + 1, len(self.inner_obstacles)):
                # Speedy distance check
                obs_jj = self.all_obstacles[jj]
                dist_center = LA.norm(obs_ii.center_position - obs_jj.center_position)
                if dist_center > np.max(obs_ii.axes_length) + np.max(
                    obs_jj.axes_length
                ):
                    continue

                close_position = gamma_normal_gradient_descent(
                    [obs_ii, obs_jj],
                    powers=[-2, -2],  # -> both in free space, i.e. > 0
                    factors=[-1, -1],
                )

                distance = LA.norm(obs_ii.center_position - close_position) + LA.norm(
                    obs_jj.center_position - close_position
                )

                if not (
                    not obs_ii.is_inside(close_position, in_global_frame=True)
                    and not obs_jj.is_inside(close_position, in_global_frame=True)
                ):
                    # Only store points if they are intersecting
                    continue

                self._graph.add_edge(
                    obs_ii,
                    obs_jj,
                    distance=distance,
                    intersection=close_position,
                )

        # if not self._graph.is_connected:
        #     # TODO: find points which don't belong anywhere and either
        #     # 1) add new ellipses
        #     # 2) extend existing ellipses
        #     raise NotImplementedError()


def get_boundary_intersection_weight(
    position: Vector,
    inner_obstacle: Obstacle,
    outer_obstacle: Obstacle,
    power_factor: float = 1.5,
) -> float:
    """Returns the weight of a corresponding intersection obstacle.

    The center / reference point of the inner obstacle has to lie within the outer."""
    gamma_outer = outer_obstacle.get_gamma(position, in_global_frame=True)
    gamma_inner = inner_obstacle.get_gamma(position, in_global_frame=True)

    if gamma_inner <= 1 or gamma_outer >= 1:
        return 0

    projected_position = outer_obstacle.get_point_on_surface(
        position, in_global_frame=True
    )

    gamma_projected = inner_obstacle.get_gamma(projected_position, in_global_frame=True)

    if gamma_projected <= 1:
        return 1

    # w_i = 1 / gamma_projected
    # p_i = gamma_outer
    # return w_i ** (p_i * 2)

    return (1 / gamma_projected) ** (gamma_outer * power_factor)


class MultiRotationalContainer(BaseContainer):
    def __init__(self, *args, **kwargs):
        self._initial_dynamics = None
        self._attractor_position = None

        super().__init__(*args, **kwargs)

    def append(self, *args, **kwargs):
        super().append(*args, **kwargs)

        if self._initial_dynamics is not None:
            warnings.warn(
                "Initial dynamics set before setting the obstacles."
                + "Make sure to update after."
            )

    def set_dynamics(self, dynamics):
        self._initial_dynamics = dynamics

        if dynamics is not None:
            self.set_attractor(dynamics.attractor_position)

    def set_attractor(self, attractor_position: Vector) -> None:
        # Make sure attractor is updated
        if self._attractor_position is None or not np.allclose(
            self._attractor_position, attractor_position
        ):
            self._attractor_position = attractor_position

            for obs in self._obstacle_list:
                obs.evaluate_hirarchy_and_reference_points()
                obs.set_attractor(attractor_position, in_global_frame=True)
                # obs.update_shortest_attractor_path()

    def evaluate(self, position: Vector) -> Vector:
        vectors = np.zeros((position.shape[0], len(self._obstacle_list)))

        for ii, obs in enumerate(self._obstacle_list):
            vectors[:, ii] = obs.evaluate(position, in_global_frame=True)

        return np.mean(vectors, axis=1)


def create_u_shape_obstacle(
    length: float, height: float, wallwidth: float
) -> MultiHullAndObstacle:
    """Returns a U-shaped Multi-Obstacle in two dimensions (object-factory)."""
    # TODO: add position (?)
    outer_obstacle = Cuboid(
        center_position=np.array([0, 0]),
        axes_length=np.array([length, height]),
        is_boundary=False,
    )

    subhull = [
        Cuboid(
            center_position=np.array([0.0, wallwidth]),
            axes_length=np.array([length - 2 * wallwidth, height]),
            is_boundary=True,
        )
    ]

    return MultiHullAndObstacle(outer_obstacle=outer_obstacle, inner_obstacles=subhull)


def test_intersection_weight(visualize=False, save_figure=False):
    my_hullobstacle = create_u_shape_obstacle(4, 2, 0.5)

    position_test = np.array([0, 0.2])
    weight = get_boundary_intersection_weight(
        position_test,
        inner_obstacle=my_hullobstacle.inner_obstacles[0],
        outer_obstacle=my_hullobstacle.outer_obstacle,
    )

    position_test = np.array([0, 0.5])
    weight = get_boundary_intersection_weight(
        position_test,
        inner_obstacle=my_hullobstacle.inner_obstacles[0],
        outer_obstacle=my_hullobstacle.outer_obstacle,
    )

    if visualize:
        x_lim = [-4, 4]
        y_lim = [-3, 3]
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        my_hullobstacle.plot_obstacle(x_lim=x_lim, y_lim=y_lim, ax=ax)

        ax.tick_params(
            axis="both",
            which="major",
            labelbottom=False,
            labelleft=False,
            bottom=False,
            top=False,
            left=False,
            right=False,
        )

        if True:
            return

        n_resolution = 50
        levels = np.linspace(1e-5, 1.0 + 1e-5, 10 + 1)
        cmap = "YlGn"

        nx = n_resolution
        ny = n_resolution
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )

        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
        weights = np.zeros(positions.shape[1])

        for it in range(positions.shape[1]):
            weights[it] = get_boundary_intersection_weight(
                positions[:, it],
                inner_obstacle=my_hullobstacle.inner_obstacles[0],
                outer_obstacle=my_hullobstacle.outer_obstacle,
            )

        # cs0 = axs[oo].contourf(
        cs0 = ax.contourf(
            x_vals,
            y_vals,
            # weights[oo, :].reshape(x_vals.shape),
            weights.reshape(x_vals.shape),
            levels=levels,
            cmap=cmap,
            zorder=1,
            alpha=0.5,
        )

        plt.colorbar(cs0, ax=ax)


def test_single_u_shape(visualize=False):
    my_hullobstacle1 = create_u_shape_obstacle(4, 2, 0.5)
    # my_hullobstacle1.orientation = 0 / 180 * math.pi
    # my_hullobstacle1.position = np.array([1, -1.25])

    # my_hullobstacle2 = create_u_shape_obstacle(4, 2, 0.5)
    # my_hullobstacle2.orientation = 180 / 180 * math.pi
    # my_hullobstacle2.position = np.array([-1, 1.25])
    environment = MultiRotationalContainer()
    environment.append(my_hullobstacle1)

    environment.set_dynamics(LinearSystem(attractor_position=np.array([3, -2.5])))

    if visualize:
        x_lim = [-4, 4]
        y_lim = [-3, 3]
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        environment[0].plot_obstacle(
            x_lim=x_lim, y_lim=y_lim, ax=ax, plot_attractors=True
        )

        # my_hullobstacle2.plot_obstacle(x_lim=[-6, 6], y_lim=[-5, 5], ax=ax)
        plot_dynamical_system(
            environment, x_lim=x_lim, y_lim=y_lim, n_resolution=10, ax=ax
        )


if (__name__) == "__main__":
    plt.close("all")
    # test_single_u_shape(visualize=True)
    test_intersection_weight(visualize=True, save_figure=False)
