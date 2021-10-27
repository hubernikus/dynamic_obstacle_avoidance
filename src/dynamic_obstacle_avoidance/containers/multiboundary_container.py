"""
Container for Obstacle to treat the intersction (and exiting) between different walls.
"""
# Author: Lukas Huber
# Mail hubernikus@gmail.com
# License: BSD (c) 2021

import warnings
import numpy as np

from dynamic_obstacle_avoidance.containers import RotationContainer

from vartools.dynamical_systems import LinearSystem


class MultiBoundaryContainer(RotationContainer):
    """Container to treat multiple boundaries / walls."""

    def __init__(self, obs_list=None, attractor_position=None, *args, **kwargs):
        super().__init__(obs_list=obs_list, *args, **kwargs)

        if obs_list is not None:
            raise NotImplementederror()

        else:
            self._parent_array = np.zeros(0, dtype=int)
            self._children_list = []
            self._parent_intersection_point = []

        if attractor_position is not None or not hasattr(self, "_attractor_position"):
            self._attractor_position = attractor_position

    def append(self, value, parent=None):
        """Add new obstacle & adapting container-properties"""
        super().append(value)

        # if update_parent_list:
        self._parent_array = np.hstack((self._parent_array, [-1]))
        self._children_list.append([])
        self._parent_intersection_point.append(None)

        if parent is not None:
            # Automatically define graph
            if parent == -1:
                self.extend_graph(
                    child=self.n_obstacles - 1, parent=self.n_obstacles - 2
                )

    def __delitem__(self, key):
        super().append(value)
        breakpoint()  # Test if it was executed correctly
        del self._children_list[key]
        self._parent_array = np.delete(self._parent_array, key)

        del self._parent_intersection_point[key]

    def get_parent(self, it):
        """Returns parent (int) of the Node [it] as input (int)"""
        return self._parent_array[it]

    def get_children(self, it):
        return self._children_list[it]

    def extend_graph(self, child, parent):
        """Use internal functions to update parents & children."""
        self.set_parent(it=child, parent=parent)
        self.add_children(it=parent, children=[child])

    def get_root(self, return_multi_values=False):
        """Get root element from parent array.
        Option to return multiple root value (if existing)."""
        root = np.where(self._parent_array < 0)[0]

        if return_multi_values:
            return root
        elif root.shape[0] > 1:
            warnings.warn("Several root-element found. Only first one returned.")

        return root[0]

    def set_parent(self, it, parent):
        self._parent_array[it] = parent

    def add_children(self, it, children):
        if isinstance(children, int):
            children = [children]
        self._children_list[it] = self._children_list[it] + children

    def get_level_numbers(self):
        """Get a number for each obstacle in self which corresponds to level."""
        ind_parents = self.get_root(return_multi_values=True)
        ind_parents = ind_parents.tolist()
        print(ind_parents)

        level_value = (-1) * np.ones(len(self._obstacle_list), dtype=int)
        it_level = 0
        while len(ind_parents):  # nonzero-iterator
            print("ind_partents", ind_parents)
            ind_children = []
            for it_p in ind_parents:
                level_value[it_p] = it_level
                ind_children = ind_children + self.get_children(it=it_p)
            it_level += 1
            ind_parents = ind_children
        return level_value

    def _get_intersection(self, it_obs1, it_obs2):
        """Get the intersection betweeen two obstacles contained in the list.
        The intersection is numerically based on the drawn points."""
        self[it_obs1].create_shape()
        self[it_obs2].create_shape()
        intersect = self[it_obs1].shape.intersection(self[it_obs2].shape)
        intersections = np.array(intersect.exterior.coords.xy)
        intersection = np.mean(intersections, axis=1)

        return intersection

    def get_boundary_list(self):
        """Returns obstacle list containing all boundary-elements."""
        # TODO MAYBE: store boundaries in separate list (?)
        return [self[ii] for ii in range(self.n_obstacles) if self[ii].is_boubndary]

    def get_boundary_ind(self):
        """Returns indeces of the current container which are equivalent to obstacles."""
        return np.array([self[ii].is_boundary for ii in range(self.n_obstacles)])

    # def update_convergence_attractor_tree(self):
    def update_intersection_graph(self, attractor_position=None):
        """Caclulate the intersection with each of the children."""
        for ii in range(self.n_obstacles):
            if self.get_parent(ii) < 0:  # Root element
                self._parent_intersection_point[ii] = None
            else:
                self._parent_intersection_point[ii] = self._get_intersection(
                    it_obs1=ii, it_obs2=self.get_parent(ii)
                )

                self._attractor_position = attractor_position

    def update_convergence_direction(self, dynamical_system):
        """Set the convergence direction to the evaluated 'convergence' DS."""
        for obs in self.n_obstacles:
            obs.get_convergence_direction(dynamical_system=dynamical_system)

    def update_relative_reference_point(self, position, gamma_margin_close_wall=1e-6):
        """Get the local reference point as described in active-learning."""
        ind_boundary = self.get_boundary_ind()
        gamma_list = np.zeros(self.n_obstacles)
        for ii in range(self.n_obstacles):
            gamma_list[ii] = self[ii].get_gamma(
                position, in_global_frame=True, relative_gamma=False
            )

        ind_inside = np.logical_and(gamma_list > 1, ind_boundary)
        ind_close = np.logical_and(gamma_list > gamma_margin_close_wall, ind_boundary)

        num_close = np.sum(ind_close)

        for ii, ii_self in zip(
            range(np.sum(ind_inside)), np.arange(self.n_obstacles)[ind_inside]
        ):
            # Displacement_weight for each obstacle
            # TODO: make sure following function is for obstacles other than ellipses (!)
            boundary_point = self[ii_self]._get_intersection_with_surface(
                direction=(position - self[ii_self].center_position),
                in_global_frame=True,
            )

            weights = np.zeros(num_close)

            dist_boundary_point = np.linalg.norm(
                boundary_point - self[ii_self].center_position
            )
            dist_point = np.linalg.norm(position - self[ii_self].center_position)

            for jj, jj_self in zip(
                range(num_close), np.arange(self.n_obstacles)[ind_close]
            ):
                if ii_self == jj_self:
                    continue
                gamma_boundary_point = self[jj_self].get_gamma(
                    boundary_point, in_global_frame=True, relative_gamma=False
                )

                # Only obstacles are considered which intersect at the (projected) boundary point
                if gamma_boundary_point < 1:
                    continue

                # Weight for the distance to the surface
                weight_1 = 1 - (dist_point) / (dist_boundary_point)

                # Weight for importance based on corresponding boundary-point
                mult_power_weight = 3.0
                max_power_weight = 5.0
                weight_2 = min(
                    (gamma_boundary_point - 1) * mult_power_weight,
                    max_power_weight,
                )

                # Power weights
                f_gap = dist_boundary_point / (dist_boundary_point - dist_point)
                f_gamma = dist_boundary_point / (dist_point)
                ff = f_gap ** (weight_2) * f_gamma

                x_hat_dist = (dist_boundary_point - dist_point * ff) / (1 - ff)
                weights[jj] = x_hat_dist / dist_point

                # print('weights final', weights[jj])
                if weights[jj] > 1 or weights[jj] < 0:
                    # DEBUG
                    breakpoint()
                    a = 0  # To not skip to beginning

            rel_reference_weight = np.max(weights)

            if rel_reference_weight > 1:
                # TODO: remove aftr debugging..
                breakpoint()
                raise ValueError("Weight greater than 1...")

            self[ii_self].global_relative_reference_point = (
                rel_reference_weight * position
                + (1 - rel_reference_weight) * self[ii_self].global_reference_point
            )

            dist_rel_ref = np.linalg.norm(
                self[ii_self].global_relative_reference_point
                - self[ii_self].center_position
            )
            relative_gamma = (dist_boundary_point - dist_rel_ref) / (
                dist_point - dist_rel_ref
            )

            self[ii_self].set_relative_gamma_at_position(
                position=position, relative_gamma=relative_gamma
            )

            # print(f"obs={ii_self} has rel_gamam={relative_gamma}")
            # breakpoint()

        for ii_self in np.arange(self.n_obstacles)[~ind_inside]:
            self[ii_self].reset_relative_reference()

    def reset_relative_references(self):
        """Reset all relative references."""
        for ii_self in range(self.n_obstacles):
            self[ii_self].reset_relative_reference()

    # def get_convergence_direction(self, position, it_obs, attractor_position=None):
    #     """ Get the (null) direction for a specific obstacle in the multi-body-boundary
    #     container which serves for the rotational-modulation. """
    #     if attractor_position is not None:
    #         self._attractor_position = attractor_position

    #     # Project point on surface
    #     if self._parent_intersection_point[it_obs] is None:
    #         if self._attractor_position is None:
    #             raise ValueError("Need 'attractor_position' to evaluate the desired direction.")
    #         local_attractor = self._attractor_position

    #     else:
    #         local_attractor = self[it_obs]._get_intersection_with_surface(
    #             direction=(self._parent_intersection_point[it_obs]-self[it_obs].center_position),
    #             in_global_frame=True)

    #     direction = LinearSystem(attractor_position=local_attractor).evaluate(position)
    #     return direction

    def get_parent_intersection(self, it_child):
        return self._parent_intersection_point[it_child]

    def get_convergence_boundary_point(
        self, *, it_child: int, project_on_child: bool, it_parent: bool = None
    ):
        # def get_convergence_boundary_point(self, *, it_child, project_on_child, it_parent =0):
        """Return intersection point projected on boundary of one of the obstacles."""
        # TODO: is projection really needed (?!)
        intersection_point = self._parent_intersection_point[it_child]

        if project_on_child is False and it_parent is None:
            raise ValueError("Parent index is needed.")

        it_surf_obs = it_child if project_on_child else it_parent
        surface_projection = self[it_surf_obs].get_intersection_with_surface(
            direction=(intersection_point - self[it_surf_obs].center_position),
            in_global_frame=True,
        )

        return surface_projection

    def plot_convergence_attractor(self, ax, attractor_position):
        """Plot the local-graph for all obstacles"""
        for ii in range(self.n_obstacles):
            if self.get_parent(ii) < 0:  # is the root
                ax.plot(
                    [attractor_position[0], self[ii].position[0]],
                    [attractor_position[1], self[ii].position[1]],
                    "-",
                    color="#808080",
                )
                ax.plot(attractor_position[0], attractor_position[1], "k*")
            else:
                local_attractor = self._parent_intersection_point[ii]
                ax.plot(
                    [local_attractor[0], self[ii].position[0]],
                    [local_attractor[1], self[ii].position[1]],
                    "-",
                    color="#808080",
                )

                ax.plot(local_attractor[0], local_attractor[1], "k*")

                ii_parent = self.get_parent(ii)
                ax.plot(
                    [local_attractor[0], self[ii_parent].position[0]],
                    [local_attractor[1], self[ii_parent].position[1]],
                    "-",
                    color="#808080",
                )
            ax.plot(self[ii].position[0], self[ii].position[1], "k+")
