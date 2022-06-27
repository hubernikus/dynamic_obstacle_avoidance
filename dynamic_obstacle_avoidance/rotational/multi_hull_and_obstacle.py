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

import numpy as np
import numpy.linalg as LA
import numpy.typing as npt

import networkx as nx
from networkx import shortest_path, shortest_path_length

# from networkx import to_dict_of_dicts, to_edgelist

from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_directional_weighted_sum

from dynamic_obstacle_avoidance.obstacles import Obstacle
from dynamic_obstacle_avoidance.visualization import plot_obstacles

Vector = npt.ArrayLike


def get_property_of_node_edge(graph, node, key) -> Vector:
    """Returns value of given property of the corresponding node in a networkx.graph,
    assuming the node has only one edge (in case of several edges the first one
    is taken randomly)."""
    # temp_list is needed (otherwise an error is thrown)
    temp_list = list(graph.edges([node], data=True))
    return temp_list[0][2][key]


def _gamma_normal_gradient_descent(
    obstacles: Obstacle,
    factors: npt.ArrayLike = None,
    powers: npt.ArrayLike = None,
    it_max: int = 50,
    step_factor: float = 0.1,
    convergence_error: float = 1e-1,
) -> np.ndarray:
    """Returns the intersection-position (or a point if it does not exists),
    for two convex input obstacles.

    Arguments
    ---------
    factors: The factor of the direction;
        > 0 if outside obstacle OR inside boundary
        < 0 otherwise
    powers: Power of the weights, chose |powers| > 1 for better convergence; good
        choice for the values is |powers[ii]|=2. Furthermore:
        > 0 if inside obstacle OR inside boundary
        < 0 otherwise
    """
    position = 0.5 * (obstacles[0].center_position + obstacles[1].center_position)

    if powers is None:
        powers = (2 for _ in range(len(obstacles)))

    if factors is None:
        factors = (1 for _ in range(len(obstacles)))

    dimension = obstacles[0].dimension

    for ii in range(it_max):
        step = np.zeros(dimension)

        # Gamma Gradient and Normal direction (?)
        for ii, obs_ii in enumerate(obstacles):
            stepsize = obs_ii.get_gamma(position, in_global_frame=True) ** powers[ii]
            step += (
                stepsize
                * factors[ii]
                * obs_ii.get_normal_direction(position, in_global_frame=True)
            )

        if LA.norm(step) < convergence_error:
            logging.info(f"Gamma gradient converged at it={ii}")
            break

        position += step_factor * step

    # if (obs_ii.get_gamma(position, in_global_frame=True) < 1
    #     or obs_jj.get_gamma(position, in_global_frame=True)
    # ):
    #     # The points are not actually intersecting

    # else:
    return position


class MultiHullAndObstacle(Obstacle):
    """This is similar to an 'basic' obstacle but requires some different structure / treatment.

    Attributes
    ----------
    inner_obstacles: List of obstacle which form the inner hull.
    outer_obstacles: For now the outer_obstacle is limited to a single obstacle (concave).
    """

    # TODO: this could be refactored to be in the outer-obstacle-frame
    def __init__(
        self,
        inner_obstacles: list,
        outer_obstacle: list,
        center_position: np.ndarray = None,
        **kwargs,
    ):
        if center_position is None:
            center_position = np.zeros(outer_obstacle.center_position.shape)

        super().__init__(center_position=center_position, **kwargs)

        # This obstacle is duality of boundary - obstacle (depending on position)
        self.is_boundary = None

        self.outer_obstacle = outer_obstacle
        self.inner_obstacles = inner_obstacles

        # Graph to the inner obstacle
        self._graph = None

        # Store the entrances and the position seperately
        # self._entrance_positions = []
        # self._entrance_obstacles = []

        self._entrance_counter = 0

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

    def set_attractor(self, position: Vector) -> None:
        """Set attractor in the global frame.
        for the moment we assume linear dynamics.
        The option to add local dynamics is possible.

        Additionally reset the 'local_attractor', since the attractor position
        has changed."""
        position = self.transform_global2relative(position)

        in_free_space = False  # Track if the attractor is in free space!

        if self.outer_obstacle.is_inside(position, in_global_frame=True):
            # Just to it for the first 'outside'-node
            # => since the behavior outside should be consistent
            self._graph.nodes[0]["local_attractor"] = None
            self._graph.nodes[0]["contains_attractor"] = False

        else:
            self._graph.nodes[0]["local_attractor"] = position
            self._graph.nodes[0]["contains_attractor"] = True

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
            warnings.warn("Attractor is in obstacle-wall.")

    def is_inside(self, position, in_global_frame=True):
        """Checks if the obstacle is colliding."""
        in_free_space = False  # Track if the attractor is in free space!

        if not self.outer_obstacle.is_inside(position, in_global_frame=True):
            in_free_space = True

        for ii, obs in enumerate(self.inner_obstacles):
            if not obs.is_inside(position, in_global_frame=True):
                in_free_space = True

        return not in_free_space

    def evaluate(self, position: Vector) -> Vector:
        """Avoids the boundary-hull obstacle."""
        if self.is_inside(position):
            return np.zeros(self.dimension)

        # Assumption that graph is already constructed
        self._evaluate_weights(position)

        weights_ = []
        velocities_ = []

        for ii, obs in enumerate(self.inner_obstacles):
            if not self.weights[ii]:
                continue

            weights_.append(self.weights[ii])

            if self._graph[obs]["local_attractor"] is None:
                self.update_shortest_attractor_path(obs)

            velocities_.append(
                self.get_local_dynamics(position),
                obs=self._graph[obs],
                weight=weights_[-1],
            )

        if self.weights[self._indices_outer]:
            weights_.append(self.weights[self._indices_outer])

            if self._graph.nodes[0]["local_attractor"] is None:
                self.update_shortest_attractor_path_from_outside(position)

            velocities_.append(
                self.get_local_dynamics(
                    position,
                    obs=0,
                    weight=weights_[-1],
                )
            )

        # TODO: can this be obtained using the 'directional'-sum
        if (
            self.weights[self._indices_outer]
            and not self._graph.nodes[0]["contains_attractor"]
        ):
            velocities_.append(
                self.get_local_dynamics(
                    position,
                    obs=0,
                    weights=weights_[self._indices_outer],
                )
            )

        mean_direction = np.sum(
            np.array(velocities_).T * np.tile(weights_, (self.dimension, 1)), axis=0
        )
        breakpoint()
        mean_magnitude = LA.norm(np.array(velocities_).T, axis=0)
        mean_magnitude = np.sum(mean_magnitude * np.array(weights_))

        dir_norm = LA.norm(mean_direction)
        if not dir_norm:
            raise ValueError("Zero direction value obtained.")

        mean_direction = mean_direction / dir_norm * mean_magnitude
        return mean_direction

    def get_local_dynamics(
        self,
        position: Vector,
        obs: Obstacle,
        weight: float,
        scaling: float = 1,
        max_velocity: float = 1,
    ) -> Vector:
        # Compute linear dynamics
        breakpoint()
        linear_velocity = self._graph.nodes[obs]["local_attractor"] - position
        linear_velocity = linear_velocity * scaling

        vel_norm = LA.norm(linear_velocity)
        if vel_norm > max_velocity:
            linear_velocity = linear_velocity / vel_norm * max_velocity

        # Rotational Modulation
        normal = self.obs.get_normal_direction(position, in_global_frame=True)
        tangent_base = get_orthogonal_basis(normal)

        tangent_velocity = tangent_base.T @ linear_velocity
        tangent_velocity[0] = 0
        tangent_velocity = tangent_velocity @ tangent_base

        mean_direction = get_directional_weighted_sum(
            null_direction=obs.get_reference_direction(position, in_global_frame=True),
            weights=[1 - weight, weight],
            directions=[linear_velocity, tangent_velocity],
        )

        return mean_direction

    def update_shortest_attractor_path_from_outside(self, position: Vector) -> None:
        closest_dist = None

        closest_dist = []
        closest_elems = []

        for ii in range(self._entrance_counter):
            dist_entrance = LA.norm(
                position - get_property_of_node_edge(self._graph, ii, "intersection")
            )

            path_dist = shortest_path_length(self._graph, source=ii)
            for ii in range(self._entrance_counter):
                closest_dist.append(path_dist[ii])
                closest_elems.append(ii)
            pass

    def update_shortest_attractor_path(self, obs_source) -> None:
        closest_dist = None
        path_dist = shortest_path_length(self._graph, source=obs_source)
        closest_dist = []
        closest_elems = []

        # path_dists = shortest_path(self._graph, source=obs_source)

        # Check the outsiders
        for obs in self.inner_obstacles:
            # Self check (i.e. obs == obs_source) is not needed, since
            # obs_source has "contains_attractor" == False
            if not self._graph[obs]["contains_attractor"]:
                continue

            closest_dist.append(path_dist)
            closest_elems.append(obs)

        if self._graph[0]["contains_attractor"]:
            for ii in range(self._entrance_counter):
                dist_attractor = LA.norm(
                    get_property_of_node_edge(self._graph, ii, "intersection")
                    - self._graph.node[0]["local_attractor"]
                )

                closest_dist.append(path_dist[ii] + dist_attractor)
                closest_elems.append(ii)

        # Get the shortest path and update
        ind_min = np.argmin(closest_dist)

        path = shortest_path(
            self._graph, soure=obs_source, target=closest_elems[ind_min]
        )

        breakpoint()
        for ii in range(len(path) - 1):
            if self._graph[obs]["local_attractor"] is not None:
                continue
            breakpoint()

            # Set projected connection element
            local_attractor = self._graph.edges[path[ii], path[ii] + 1]["intersection"]

            border_attractor = path[ii].get_point_on_surface(
                local_attractor, in_global_frame=True
            )

            self._graph.nodes[paht[ii]]["border_attractor"]

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
    ) -> None:
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        plot_obstacles(
            obstacle_container=[self.outer_obstacle],
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            # alpha_obstacle=1.0,
        )

        temp_list = copy.deepcopy(self.inner_obstacles)
        for obs in temp_list:
            obs.is_boundary = False

        plot_obstacles(
            obstacle_container=temp_list,
            x_lim=x_lim,
            y_lim=y_lim,
            ax=ax,
            alpha_obstacle=1.0,
            obstacle_color="white",
            draw_reference=True,
        )

        if plot_attractors:
            for node in self._graph.nodes:
                if node[0]["contains_attractor"]:
                    color = "black"
                    linewidth = 18
                    markersize = 18
                else:
                    color = "blue"
                    linewidth = 13
                    markersize = 13

                self._graph.nodes[0]["local_attractor"] = None
                self._graph.nodes[0]["contains_attractor"] = False
                ax.plot(
                    self._graph.nodes[0]["local_attractor"][0],
                    self._graph.nodes[0]["local_attractor"][1],
                    "*",
                    color=color,
                    linewidth=linewidth,
                    markersize=markersize,
                )

    def get_gamma(self, position: Vector, in_global_frame: bool = False) -> float:
        """Get distance value with respect to the obstacle."""
        if in_global_frame:
            position = self.transform_global2relative(position)
        breakpoint()

    def get_normal_direction(
        self, position: Vector, in_global_frame: bool = False
    ) -> float:
        raise NotImplementedError()

    def _update_intersections_with_outer_obstacle(self):
        """Get intersection with the 'free' space"""

        # Reset the counter
        self._entrance_counter = 0
        # self._entrance_obstacles = []
        # self._entrance_positions = []

        for ii, obs_ii in enumerate(self.inner_obstacles):
            if LA.norm(obs_ii.center_position - self.outer_obstacle.center_position) < (
                np.min(self.outer_obstacle.axes_length) - np.max(obs_ii.axes_length)
            ):
                continue

            position = _gamma_normal_gradient_descent(
                [obs_ii, self.outer_obstacle],
                powers=[-2, -2],  # -> both in free space, i.e. < 0
                factors=[-1, 1],
            )

            if obs_ii.is_inside(position) or obs_ii.is_inside(position):
                continue

            # Store intersections
            # self._entrance_obstacles.append(obs_ii)
            # self._entrance_positions.append(position)

            self._graph.add_node(
                self._entrance_counter,
            )
            self._graph.add_edge(
                self._entrance_counter,
                obs_ii,
                weight=LA.norm(position - obs_ii.center_position),
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

                close_position = _gamma_normal_gradient_descent(
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
                    weight=distance,
                    intersection=close_position,
                )

        # if not self._graph.is_connected:
        #     # TODO: find points which don't belong anywhere and either
        #     # 1) add new ellipses
        #     # 2) extend existing ellipses
        #     raise NotImplementedError()
