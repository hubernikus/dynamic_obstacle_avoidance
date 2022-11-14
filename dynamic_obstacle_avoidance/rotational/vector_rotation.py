#!/USSR/bin/python3
""" Create the rotation space which is so much needed. ... """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-07-07

# Use python 3.10 [annotations / typematching]
from __future__ import annotations  # Not needed from python 3.10 onwards

import copy
import warnings

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy import linalg as LA

import networkx as nx

from vartools.linalg import get_orthogonal_basis

from dynamic_obstacle_avoidance.rotational.utils import get_orthonormal_spanning_basis
from dynamic_obstacle_avoidance.rotational.datatypes import Vector, VectorArray

NodeType = int


def rotate_direction(
    direction: Vector, base: VectorArray, rotation_angle: float
) -> Vector:
    """Returns the rotated of the input vector with respect to the base and rotation angle."""
    # Normalize just to make sure.
    if not (dir_norm := LA.norm(direction)):
        raise ZeroDivisionError()

    direction = direction / dir_norm

    dot_prods = np.dot(base.T, direction)
    angle = math.atan2(dot_prods[1], dot_prods[0]) + rotation_angle

    # Convert angle to the two basis-axis
    out_direction = math.cos(angle) * base[:, 0] + math.sin(angle) * base[:, 1]
    out_direction *= math.sqrt(sum(dot_prods**2))

    # Finally, add the orthogonal part (no effect in 2D, but important for higher dimensions)
    out_direction += direction - np.sum(dot_prods * base, axis=1)
    return out_direction


def rotate_array(
    directions: VectorArray,
    base: VectorArray,
    rotation_angle: float,
) -> VectorArray:
    """Rotate upper level base with respect to total."""
    dimension, n_dirs = directions.shape

    directions = directions / LA.norm(directions, axis=0)

    # Matrix dimensions: [2 x n_dirs ] <- [dimension x 2 ].T @ [dimension x n_dirs]
    dot_prods = np.dot(base.T, directions)
    angles = np.arctan2(dot_prods[1, :], dot_prods[0, :]) + rotation_angle

    # Compute output from rotation
    out_vectors = np.tile(base[:, 0], (n_dirs, 1)).T * np.tile(
        np.cos(angles), (dimension, 1)
    ) + np.tile(base[:, 1], (n_dirs, 1)).T * np.tile(np.sin(angles), (dimension, 1))
    out_vectors *= np.tile(np.sqrt(np.sum(dot_prods**2, axis=0)), (dimension, 1))

    # Finally, add the orthogonal part (no effect in 2D, but important for higher dimensions)
    out_vectors += directions - (base @ dot_prods)

    return out_vectors


@dataclass
class VectorRotationXd:
    """This approach allows successive modulation which can be added up.

    Attributes
    ----------
    base array of size [dimension x 2]: The (orthonormal) base constructed from the to
        input directions
    rotation_angle (float): The rotation angle resulting from the two input directions
    """

    base: VectorArray
    rotation_angle: float

    @classmethod
    def from_directions(cls, vec_init: Vector, vec_rot: Vector) -> VectorRotationXd:
        """Alternative constructor base on two input vectors which define the
        initialization."""

        # # Normalize both vectors
        vec_init = vec_init / LA.norm(vec_init)
        vec_rot = vec_rot / LA.norm(vec_rot)

        dot_prod = np.dot(vec_init, vec_rot)
        if dot_prod == (-1):
            warnings.warn("Antiparallel vectors")

        if abs(dot_prod) < 1:
            vec_perp = vec_rot - vec_init * dot_prod
            vec_perp = vec_perp / LA.norm(vec_perp)

        else:
            # (Anti-)parallel vectors => calculate random perpendicular vector
            vec_perp = np.zeros(vec_init.shape)
            if not LA.norm(vec_init[:2]):
                vec_perp[0] = 1
            else:
                vec_perp[0] = vec_init[1]
                vec_perp[1] = vec_init[0] * (-1)
                vec_perp[:2] = vec_perp[:2] / LA.norm(vec_perp[:2])

        angle = np.arccos(min(max(dot_prod, -1), 1))
        return cls(base=np.array([vec_init, vec_perp]).T, rotation_angle=angle)

    # def __mult__(self, factor) -> VectorRotationXd:
    #     instance_copy = copy.deepcopy(self)
    #     instance_copy.rotation_angle = instance_copy.rotation_angle * factor
    #     return instance_copy

    @property
    def base0(self):
        return self.base[:, 0]

    @property
    def dimension(self):
        try:
            return self.base.shape[0]
        except AttributeError:
            warnings.warn("base has not been defined")
            return None

    def get_second_vector(self) -> Vector:
        """Returns the second vector responsible for the rotation"""
        return rotate_direction(
            direction=self.base[:, 0],
            rotation_angle=self.rotation_angle,
            base=self.base,
        )

    def rotate(self, direction, rot_factor: float = 1):
        """Returns the rotated of the input vector with respect to the base and rotation angle
        rot_factor: factor gives information about extension of rotation"""
        return rotate_direction(
            direction=direction,
            rotation_angle=rot_factor * self.rotation_angle,
            base=self.base,
        )

    def rotate_vector_rotation(
        self, rotation: VectorRotationXd, rot_factor: float = 1
    ) -> VectorRotationXd:
        rotation = copy.deepcopy(rotation)
        rotation.base = rotate_array(
            directions=rotation.base,
            base=rotation.base,
            rotation_angle=rot_factor * self.rotation_angle,
        )
        return rotation

    def inverse_rotate(self, direction):
        return rotate_direction(
            direction=direction,
            rotation_angle=(-1) * self.rotation_angle,
            base=self.base,
        )


class VectorRotationSequence:
    """
    Vector-Rotation environment based on multiple vectors

    Attributes
    ----------
    vectors_array (np.array of shape [dimension x n_rotations + 1]):
        (storing) the inital array of vectors
    basis_array (numpy array of  shape [dimension x n_rotations x 2]):
        contains the basis of all rotations
    rotation_angles: The rotation between going from one to the next basis
    """

    def __init__(self, vectors_array: np.array) -> None:
        # Normalize
        self.vectors_array = vectors_array / LA.norm(vectors_array, axis=0)

        dot_prod = np.sum(
            self.vectors_array[:, 1:] * self.vectors_array[:, :-1], axis=0
        )

        if np.sum(dot_prod == (-1)):  # Any of the values
            raise ValueError("Antiparallel vectors.")

        # Evaluate basis and angles
        vec_perp = self.vectors_array[:, 1:] - self.vectors_array[:, :-1] * dot_prod
        vec_perp = vec_perp / LA.norm(vec_perp, axis=0)

        self.basis_array = np.stack((self.vectors_array[:, :-1], vec_perp), axis=2)
        self.rotation_angles = np.arccos(dot_prod)

    @property
    def n_rotations(self):
        return self.basis_array.shape[1]

    @property
    def dimension(self):
        return self.basis_array.shape[0]

    def base(self) -> Vector:
        return self.basis_array[:, [0, -1]]

    def append(self, direction: Vector) -> None:
        self.basis_array = np.hstack((self.basis_array, direction.reshape(-1, 1)))

        raise NotImplementedError("Finish updating basis and rotation angles.")

    def rotate(self, direction: Vector, rot_factor: float = 1) -> Vector:
        """Rotate over the whole length of the vector."""
        weights = np.zeros(self.n_rotations)
        weights[-1] = rot_factor
        return self.rotate_weighted(direction, weights=weights)

    def rotate_weighted(self, direction: Vector, weights: list[float] = None) -> Vector:
        """
        Returns the rotated direction vector with repsect to the (rotation-)weights

        weights (list of floats (>=0) with length [self.n_rotations]): indicates fraction
        of each rotation which is applied.
        """
        # Starting at the root
        cumulated_weights = np.cumsum(weights[::-1])[::-1]

        if not math.isclose(cumulated_weights[0], 1):
            warnings.warn("Weights are summing up to more than 1.")

        temp_base = np.copy(self.basis_array)
        if weights is None:
            temp_angle = self.rotation_angles

        else:
            temp_angle = self.rotation_angles * cumulated_weights

            # Update the basis of rotation weights from top-to-bottom
            # by rotating upper level base with respect to total
            for ii in reversed(range(self.n_rotations - 1)):
                temp_base[:, (ii + 1) :, :] = rotate_array(
                    directions=temp_base[:, (ii + 1) :, :].reshape(self.dimension, -1),
                    base=temp_base[:, ii, :],
                    rotation_angle=self.rotation_angles[ii]
                    * (1 - cumulated_weights[ii]),
                ).reshape(self.dimension, -1, 2)

        # Finally: rotate from bottom-to-top
        for ii in range(self.n_rotations):
            direction = rotate_direction(
                direction=direction,
                rotation_angle=temp_angle[ii],
                base=temp_base[:, ii, :],
            )
        return direction


class VectorRotationTree:
    """
    VectorRotation but originating structured in a tree

    Positive node number reference the corresponding reference-id
    (negative numbers are used for the vectors)
    """

    # TODO: what happens if an obstacle is at angle 'pi'?
    # as it might happend at  zero-level

    def __init__(self, root_id: int = None, root_direction: Vector = None) -> None:
        self._graph = nx.DiGraph()

        if root_id is not None:
            self.set_root(root_id, root_direction)

    def set_root(self, root_id, direction):
        # To easier find the root again (!)
        self._graph.add_node(
            root_id,
            level=0,
            direction=direction,
            weight=0,
            orientation=VectorRotationXd.from_directions(direction, direction),
        )

        # To easier find the root again (!)
        self._root_id = root_id

    @property
    def root(self, root_id: int, direction: Vector) -> NodeType:
        return self._graph.nodes(self._root_id)

    @property
    def Graph(self):
        # rename to _G (?)
        return self._graph

    def add_node(
        self,
        node_id: NodeType,
        direction: Vector = None,
        parent_id: NodeType = None,
        child_id: NodeType = None,
        level: int = None
        # rotation_limit: float = math.pi * 0.75,
    ) -> None:

        if parent_id is not None:
            self._graph.add_edge(
                parent_id,
                node_id,
            )

            level = self._graph.nodes[parent_id]["level"] + 1

        elif child_id is not None:
            self._graph.add_edge(node_id, child_id)
            self.set_direction(direction, node_id)

        elif level is None:
            raise ValueError(
                "Argument 'level' is needed, if no parent or child is provided"
            )

        self._graph.add_node(
            node_id,
            level=level,
            direction=None,
            weight=0,
            orientation=None,
        )

        if direction is not None:
            self.set_direction(node_id, direction)

        # TODO: what happens when you overwrite a node (?)

    def set_direction(self, node_id: int, direction: Vector) -> None:
        self._graph.nodes[node_id]["direction"] = direction

        # TODO: Not all rotation would have to be reset (only when higher weight is changed..)
        # Do better checks to accelerate
        successors = self.get_all_childs_children(node_id)
        for succ in successors:
            succ["orientation"] = None

    def evaluate_all_orientations(
        self, sorted_list: list[NodeType] = None, pi_margin: float = np.pi * 0.75
    ) -> None:
        """Updates all orientations of the '_graph' class.
        -> store the new direction in the graph as 'part_direction'
        """
        # TODO: only store partial_direction here (!)
        if sorted_list is None:
            sorted_list = self.get_nodes_ascending()

        # Special values for root-node
        self._graph.nodes[sorted_list[0]]["part_direction"] = self._graph.nodes[
            sorted_list[0]
        ]["direction"]

        self._graph.nodes[sorted_list[0]][
            "orientation"
        ] = VectorRotationXd.from_directions(
            self._graph.nodes[sorted_list[0]]["direction"],
            self._graph.nodes[sorted_list[0]]["direction"],
        )

        for n_id in self._graph.nodes:
            for c_id in self._graph.successors(n_id):
                self._graph.nodes[c_id][
                    "orientation"
                ] = VectorRotationXd.from_directions(
                    self._graph.nodes[n_id]["part_direction"],
                    self._graph.nodes[c_id]["direction"],
                )

                # At which angle the rotation should be reduced to obtain a continuous behavior
                if (
                    pi_margin
                    and self._graph.nodes[c_id]["orientation"].rotation_angle
                    > pi_margin
                ):
                    breakpoint()
                    weight = self._graph.nodes[c_id]["orientation"].rotation_angle
                    weight = (math.pi - abs(weight)) / (math.pi - pi_margin)
                    self._graph.nodes[c_id]["orientation"].rotation_angle *= weight

                    self._graph.nodes[c_id]["part_direction"] = self._graph.nodes[c_id][
                        "orientation"
                    ].get_second_vector()

                else:
                    self._graph.nodes[c_id]["part_direction"] = self._graph.nodes[c_id][
                        "direction"
                    ]

    def reset_node_weights(self) -> None:
        for node_id in self._graph.nodes:
            self._graph.nodes[node_id]["weight"] = 0

    def set_node(
        self, node_id: NodeType, parent_id: NodeType, direction: Vector
    ) -> None:
        # TODO: implement such that it updates lower and higher nodes (!)
        raise NotImplementedError()

    @property
    def dimension(self) -> int:
        try:
            return self._graph.nodes[0]["direction"].shape[0]
        except AttributeError:
            warnings.warn("base or property has not been defined")
            return None

    def get_nodes_ascending(self) -> list[NodeType]:
        # Ascending sorted node-list
        level_list = [self._graph.nodes[node]["level"] for node in self._graph.nodes]
        node_unsorted = [node for node in self._graph.nodes]
        return [node_unsorted[ii] for ii in np.argsort(level_list)]

    def get_all_childs_children(self, node: NodeType) -> list[NodeType]:
        """Returns list of nodes which are in the directional line of the argument node."""
        successor_list = [ii for ii in self._graph.successors(node)]

        ii = 0
        while ii < len(successor_list):
            # Add all children elements to the list
            successor_list += [jj for jj in self._graph.successors(successor_list[ii])]
            ii += 1

        return successor_list

    def get_weighted_mean(self, node_list: list[int], weights: list[float]) -> Vector:
        """Evaluate the weighted mean of the graph."""

        if (weight_sum := np.sum(weights)) != 1:
            warnings.warn(f"Sum of weights {weight_sum} is not equal to one.")

        self.reset_node_weights()

        # Weights are stored in the predecessing nodes of the corresponding edge
        for ii, node in enumerate(node_list):
            self._graph.nodes[node]["weight"] = weights[ii]

        # Update cumulated weights
        sorted_list = self.get_nodes_ascending()

        self.evaluate_all_orientations(sorted_list)

        # for node_id in sorted_list[::-1]:
        for node_id in reversed(sorted_list):
            # Reverse update the weights
            for pred_id in self._graph.predecessors(node_id):
                # There is only one predecessor,
                # Where are the weights stored / where are the rotations stored (?)
                self._graph.nodes[pred_id]["weight"] += self._graph.nodes[node_id][
                    "weight"
                ]

            # Update orientation and create 'partial' orientations
            self._graph.nodes[node_id]["part_orientation"] = VectorRotationXd(
                base=self._graph.nodes[node_id]["orientation"].base,
                rotation_angle=(
                    self._graph.nodes[node_id]["orientation"].rotation_angle
                    * self._graph.nodes[node_id]["weight"]
                ),
            )

            if (
                not self._graph.successors(node_id)
                or self._graph.nodes[node_id]["weight"] <= 0
                or self._graph.nodes[node_id]["weight"] >= 1
            ):
                # No successor nodes (or successors with only zero weight !? )
                # or full rotation is being kept
                # TODO: why is the second condition not working ?!
                continue

            successors = self.get_all_childs_children(node_id)

            _succ_basis = [
                self._graph.nodes[succ]["part_orientation"].base for succ in successors
            ]
            if not len(_succ_basis):
                continue

            # Make sure dimension is the first axes for future array restructuring
            succ_basis = np.swapaxes(_succ_basis, 0, 1)

            # Backwards rotate such that it's aligned with the new angle
            succ_basis = rotate_array(
                directions=succ_basis.reshape(self.dimension, -1),
                base=self._graph.nodes[node_id]["orientation"].base,
                rotation_angle=(
                    self._graph.nodes[node_id]["part_orientation"].rotation_angle
                    - self._graph.nodes[node_id]["orientation"].rotation_angle
                ),
            ).reshape(self.dimension, -1, 2)

            for ii, succ in enumerate(successors):
                self._graph.nodes[succ]["part_orientation"].base = succ_basis[:, ii, :]

        return self.evaluate_graph_summing(sorted_list)

    def evaluate_graph_summing(self, sorted_list) -> Vector:
        """Graph summing under assumption of shared-basis at each level.

        => the number of calculations is $2x (n_{childrend} of node) \forall node \in nodes $
        i.e. does currently not scale well
        But calculations are simple, i.e., this could be sped upt with cython / C++ / Rust
        """
        level_list = [self._graph.nodes[node_id]["level"] for node_id in sorted_list]

        # Bottom up calculation - from lowest level to highest level
        # at each level, take the weighted average of all rotations
        for level in set(level_list):

            nodelevel_ids = []
            for node_id, lev in zip(sorted_list, level_list):
                if lev == level and self._graph.nodes[node_id]["weight"]:
                    nodelevel_ids.append(node_id)

            if not nodelevel_ids:
                continue

            # Each round higher levels are un-rotated to share the same basis
            shared_first_basis = self._graph.nodes[nodelevel_ids[0]][
                "part_orientation"
            ].base[:, 0]
            shared_basis = get_orthogonal_basis(shared_first_basis)

            # Get the rotation-vector (second -base vector) of all of the
            # same-level rotation-structs in the local_basis
            local_basis = np.array(
                [
                    self._graph.nodes[jj]["part_orientation"].base[:, 1]
                    for jj in nodelevel_ids
                ]
            ).T
            # local_basis = shared_basis.T @ local_basis

            # Add the rotation angles up
            local_basis *= np.array(
                [
                    self._graph.nodes[jj]["part_orientation"].rotation_angle
                    for jj in nodelevel_ids
                ]
            )
            local_mean_basis = np.sum(local_basis, axis=1)
            new_angle = LA.norm(local_mean_basis)
            if new_angle:  # Nonzero
                # local_mean_basis[0] = 0  # Really (?)
                # averaged_direction = shared_basis @ (local_mean_basis / new_angle)
                averaged_direction = local_mean_basis / new_angle
            else:
                # No rotation, hence it's the first vector
                averaged_direction = shared_basis[:, 0]

            # Rotate all following rotation-levels back
            all_successors = []
            all_basis = np.zeros((self.dimension, 0))
            for node_id in nodelevel_ids:
                # Transform all child angles to first base-direction
                successors = self.get_all_childs_children(node_id)

                if not successors:
                    # No successors
                    continue

                _succ_basis = [
                    self._graph.nodes[succ]["part_orientation"].base
                    for succ in successors
                ]
                # Make sure dimension is the first axes for future array restructuring
                succ_basis = np.swapaxes(_succ_basis, 0, 1)

                succ_basis = rotate_array(
                    directions=succ_basis.reshape(self.dimension, -1),
                    base=self._graph.nodes[node_id]["part_orientation"].base,
                    rotation_angle=(-1)
                    * self._graph.nodes[node_id]["part_orientation"].rotation_angle,
                )

                all_successors += successors
                all_basis = np.hstack((all_basis, succ_basis))

            if not all_successors:
                # Zero list -> check the next level
                continue

            if new_angle:
                # Transform to the new basis-direction
                all_basis = rotate_array(
                    # directions=all_basis.reshape(self.dimension, -1),
                    directions=all_basis,
                    base=np.vstack((shared_first_basis, averaged_direction)).T,
                    rotation_angle=new_angle,
                ).reshape(self.dimension, -1, 2)

            else:
                # Zero transformation angle, hence just reshape
                all_basis = all_basis.reshape(self.dimension, -1, 2)

            for ii, node in enumerate(all_successors):
                self._graph.nodes[node]["part_orientation"].base = all_basis[:, ii, :]

        final_rotation = VectorRotationXd(
            base=np.vstack((shared_first_basis, averaged_direction)).T,
            rotation_angle=new_angle,
        )

        return final_rotation.rotate(shared_first_basis)

    def rotate(
        self, initial_vector: Vector, node_list: list[int], weights: list[float]
    ) -> Vector:
        """Returns the rotated vector based on the mean-direction.

        Assumption that the initial"""
        rotated_dir = self.get_weighted_mean()
        temp_rotation = VectorRotationXd.from_directions(
            self.root["orientation"].base[:, 0], rotated_dir
        )

        return rotate_direction(
            direction=initial_vector,
            base=temp_rotation.base,
            rotation_angle=temp_rotation.rotation_angle,
        )

    def inverse_rotate(
        self, initial_vector: Vector, node_list: list[int], weights: list[float]
    ) -> Vector:
        """Returns the rotated vector based on the mean-direction.

        Assumption that the initial"""
        rotated_dir = self.get_weighted_mean()
        temp_rotation = VectorRotationXd.from_directions(
            rotated_dir, self.root["orientation"].base[:, 0]
        )

        return rotate_direction(
            direction=initial_vector,
            base=temp_rotation.base,
            rotation_angle=temp_rotation.rotation_angle,
        )

    def get_rotation_weights(self, parent_id: int, direction: Vector) -> float:
        pass

    def rotate_weighted(self, node_id_list: list[int], weights: list[float]):
        # For safe rotation at the back
        raise NotImplementedError()
