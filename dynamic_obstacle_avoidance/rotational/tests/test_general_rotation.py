#!/USSR/bin/python3
""" Create the rotation space which is so much needed. ... """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-07-07

# Use python 3.10 [annotations / typematching]
from __future__ import annotations  # Not needed from python 3.10 onwards

import copy
import warnings

# import logging
import math
from dataclasses import dataclass

import numpy as np
from numpy import linalg as LA

import networkx as nx

# import igraph

import matplotlib.pyplot as plt
import numpy.typing as npt

from vartools.linalg import get_orthogonal_basis

Vector = npt.ArrayLike
VectorArray = npt.ArrayLike

NodeType = int


def get_orthonormal_spanning_basis(vector1, vector2, /):
    """Returns a orthonormal basis from to orthonormal input vectors."""
    dot_prod = np.dot(vector1, vector2)

    if abs(dot_prod) < 1:
        vec_perp = vector2 - vector1 * dot_prod
        vec_perp = vec_perp / LA.norm(vec_perp)
    else:
        # (Anti-)parallel vectors => take random perpendicular vector
        vec_perp = np.zeros(vector1.shape)
        if not LA.norm(vector1[:2]):
            vec_perp[0] = 1
        else:
            vec_perp[0] = vector1[1]
            vec_perp[1] = vector1[0] * (-1)
            vec_perp[:2] = vec_perp[:2] / LA.norm(vec_perp[:2])

    return np.vstack((vector1, vec_perp)).T


def rotate_direction(direction: Vector, base: VectorArray, rotation_angle: float):
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


class VectorRotationTree:
    """
    VectorRotation but originating structured in a tree

    Positive node number reference the corresponding reference-id
    (negative numbers are used for the vectors)
    """

    # TODO: what happens if an obstacle is at angle 'pi'?
    # as it might happend at  zero-level

    def __init__(self, root_id: int, root_direction: Vector) -> None:
        self._graph = nx.DiGraph()
        self._graph.add_node(
            root_id,
            level=0,
            direction=root_direction,
            weight=0,
            orientation=VectorRotationXd(
                base=np.tile(root_direction, (2, 1)).T, rotation_angle=0
            ),
        )

    # def set_root(self, root_id: int, direction: Vector) -> None:
    #     self._graph.add_node(node_id, level=0, direction=direction)

    def add_node(
        self,
        node_id: NodeType,
        direction: Vector,
        parent_id: NodeType,
        rotation_limit: float = math.pi * 0.75,
    ) -> None:

        new_rotation = VectorRotationXd.from_directions(
            self._graph.nodes[parent_id]["direction"],
            direction,
        )

        if rotation_limit and abs(new_rotation.rotation_angle) > rotation_limit:
            rot_factor = (
                2 * math.pi - abs(new_rotation.rotation_angle)
            ) / rotation_limit

            direction = new_rotation.rotate(
                new_rotation.basis[:, 0], rot_factor=rot_factor
            )

            new_rotation.angle = new_rotation.rotation_angle * rot_factor

            warnings.warn(
                f"Rotation is the limit={rotation_limit}. "
                + "Adaptation weight is used. "
            )

        self._graph.add_node(
            node_id,
            level=self._graph.nodes[parent_id]["level"] + 1,
            direction=direction,
            weight=0,
            orientation=new_rotation,
        )

        self._graph.add_edge(
            parent_id,
            node_id,
        )
        # TODO: what happens when you overwrite a node (?)

    def reset_node_weights(self):
        for node_id in self._graph.nodes:
            self._graph.nodes[node_id]["weight"] = 0

    def set_node(self, node_id, parent_id, direction):
        # TODO: implement such that it updates lower and higher nodes (!)
        raise NotImplementedError()

    @property
    def dimension(self):
        try:
            return self._graph.nodes[0]["direction"].shape[0]
        except AttributeError:
            warnings.warn("base or property has not been defined")
            return None

    def get_nodes_ascending(self) -> list(NodeType):
        # Ascending sorted node-list
        level_list = [self._graph.nodes[node]["level"] for node in self._graph.nodes]
        node_unsorted = [node for node in self._graph.nodes]
        return [node_unsorted[ii] for ii in np.argsort(level_list)]

    def get_all_childs_children(self, node: NodeType) -> list(NodeType):
        """Returns list of nodes which are in the directional line of the argument node."""
        successor_list = [ii for ii in self._graph.successors(node)]

        ii = 0
        while ii < len(successor_list):
            # Add all children elements to the list
            successor_list += [jj for jj in self._graph.successors(successor_list[ii])]
            ii += 1

        return successor_list

    def get_weighted_mean(self, node_list: list(int), weights: list(float)) -> Vector:
        """Evaluate the weighted mean of the graph."""

        if (weight_sum := np.sum(weights)) != 1:
            warnings.warn(f"Sum of weights {weight_sum} is not equal to one.")

        self.reset_node_weights()

        # Weights are stored in the predecessing nodes of the corresponding edge
        for ii, node in enumerate(node_list):
            self._graph.nodes[node]["weight"] = weights[ii]

        # Update cumulated weights
        sorted_list = self.get_nodes_ascending()
        for node_id in sorted_list[::-1]:
            # only one predecessor
            for pred_id in self._graph.predecessors(node_id):
                # for succ_id in self._graph.predecessors(node_id):
                # Where are the weights stored / where are the rotations stored (?)
                self._graph.nodes[pred_id]["weight"] += self._graph.nodes[node_id][
                    "weight"
                ]

        # Create 'partial' orientations
        for node_id in reversed(sorted_list):
            if "orientation" not in self._graph.nodes[node_id]:
                self._graph.nodes[node_id]["part_orientation"] = create_zero_rotation(
                    dimension=3
                )
                continue

            self._graph.nodes[node_id]["part_orientation"] = VectorRotationXd(
                base=self._graph.nodes[node_id]["orientation"].base,
                rotation_angle=(
                    self._graph.nodes[node_id]["orientation"].rotation_angle
                    * self._graph.nodes[node_id]["weight"]
                ),
            )

            if (
                not self._graph.successors(node_id)
                or self._graph.nodes[node_id]["weight"] == 1
            ):
                # No sucessor nodes or full rotation is being kept
                continue

            successors = self.get_all_childs_children(node_id)

            _succ_basis = [
                self._graph.nodes[succ]["part_orientation"].base for succ in successors
            ]
            if not len(_succ_basis):
                continue

            # Make sure dimension is the first axes for future array restructuring
            succ_basis = np.swapaxes(_succ_basis, 0, 1)

            # backwards rotate such that it's aligned with the new angle
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

        -> the number of calculations is $2x (n_{childrend} of node) \forall node \in nodes $
        i.e. does currently not scale well
        But calculations are simple, i.e., this could be sped upt with cython / C++ / Rust
        """
        level_list = [self._graph.nodes[node_id]["level"] for node_id in sorted_list]

        # Bottom up calculation - from lowest level to highest level
        # at each level, take the weighted average of all rotations
        for level in set(level_list):
            nodelevel_ids = [
                sorted_list[jj] for jj, lev in enumerate(level_list) if lev == level
            ]

            # Each round higher levels are un-rotated to share the same basis
            shared_first_basis = self._graph.nodes[nodelevel_ids[0]]["part_orientation"].base[:, 0]
            shared_basis = get_orthogonal_basis(shared_first_basis)

            # Get the rotation-vector (second -base vector) of all of the
            # same-level rotation-structs in the local_basis
            basis_array = np.array(
                [
                    self._graph.nodes[ii]["part_orientation"].base[:, 1]
                    for ii in nodelevel_ids
                ]
            ).T
            local_basis = shared_basis.T @ basis_array

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
                averaged_direction = shared_basis @ (local_mean_basis / new_angle)
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

            breakpoint()
            if not all_successors:
                final_rotation = VectorRotationXd(
                    base=np.vstacke(shared_first_basis, averaged_direction).T,
                    rotation_angle=new_angle,
                )
                    
                # Reached the end of the graph - there are no successors anymore
                return final_rotation.rotate(shared_first_basis)

            if new_angle:
                # Transform to the new basis-direction
                if np.dot(shared_first_basis, averaged_direction):
                    # TODO: remove
                    breakpoint() 
                # new_base = get_orthonormal_spanning_basis(
                #     self._graph.nodes[nodelevel_ids[0]]["part_orientation"].base[:, 0],
                #     averaged_direction,
                # )

                all_basis = rotate_array(
                    # directions=all_basis.reshape(self.dimension, -1),
                    directions=all_basis,
                    base=new_base,
                    rotation_angle=new_angle,
                ).reshape(self.dimension, -1, 2)

            else:
                # Zero transformation angle, hence just reshape
                all_basis = all_basis.reshape(self.dimension, -1, 2)

            for ii, node in enumerate(all_successors):
                self._graph.nodes[node]["part_orientation"].base = all_basis[:, ii, :]

        # Convergence and return should happen within the loop
        raise Exception("No weighted mean found.")

    def get_rotation_weights(self, parent_id: int, direction: Vector) -> float:
        pass

    def rotate_weighted(self, node_id_list: list(int), weights: list(float)):
        # For safe rotation at the back
        raise NotImplementedError()


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
            # raise ValueError("Antiparallel vectors")
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

        return cls(
            base=np.array([vec_init, vec_perp]).T, rotation_angle=np.arccos(dot_prod)
        )

    # @classmethod
    # def zero_rotation(cls, vec_init: Vector) -> VectorRotationXd:
    # return cls(base=np.repeat(vec_init, axis=1), rotation=0)

    @property
    def dimension(self):
        try:
            return self.base.shape[0]
        except AttributeError:
            warnings.warn("base has not been defined")
            return None

    def rotate(self, direction, rot_factor: float = 1):
        """Returns the rotated of the input vector with respect to the base and rotation angle
        rot_factor: factor gives information about extension of rotation"""
        return rotate_direction(
            direction=direction,
            rotation_angle=rot_factor * self.rotation_angle,
            base=self.base,
        )

    def inverse_rotate(self, direction):
        return rotate_direction(
            direction=direction,
            rotation_angle=(-1) * self.rotation_angle,
            base=self.base,
        )


def test_cross_rotation_2d(visualize=False, savefig=False):
    vec0 = np.array([1, 0.3])
    vec1 = np.array([1.0, -1])

    vector_rotation = VectorRotationXd.from_directions(vec0, vec1)

    vec0 /= LA.norm(vec0)
    vec1 /= LA.norm(vec1)

    # Reconstruct vector1
    vec_rot = vector_rotation.rotate(vec0)
    assert np.allclose(vec1, vec_rot), "Rotation was not reconstructed."

    vecs_test = [
        [1, -1.2],
        [-1.2, -1],
        [-1.2, 1.3],
    ]

    cross_prod_base = np.cross(vec0, vec1)

    vecs_rot_list = []
    for ii, vec in enumerate(vecs_test):
        vec_test = np.array(vecs_test[ii])
        vec_test /= LA.norm(vec_test)
        vec_rot = vector_rotation.rotate(vec_test)

        assert np.isclose(
            cross_prod_base, np.cross(vec_test, vec_rot)
        ), "Vectors are not close"

        # assert np.isclose(
        #     vector_rotation.rotation_angle, np.arccos(np.dot(vec_test, vec_rot))
        # ), "Not the correct rotation."

        # For visualization purposes
        vecs_rot_list.append(vec_rot)

    if visualize:
        arrow_props = {"head_length": 0.1, "head_width": 0.05}

        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

        ax = axs[0, 0]
        vec_1 = vector_rotation.base[:, 0]
        ax.arrow(
            0,
            0,
            vec_1[0],
            vec_1[1],
            color="k",
            **arrow_props,
        )

        vec_perp = vector_rotation.base[:, 1]
        ax.arrow(
            0,
            0,
            vec_perp[0],
            vec_perp[1],
            color="k",
            label="Base",
            **arrow_props,
        )

        ax.arrow(0, 0, vec0[0], vec0[1], color="g", label="Vector 0", **arrow_props)
        ax.arrow(0, 0, vec1[0], vec1[1], color="b", label="Vector 1", **arrow_props)
        ax.legend()

        ax = axs[0, 1]
        axs_test = axs.flatten()[1:]
        for ii, ax in enumerate(axs_test):
            vec_test = vecs_test[ii] / LA.norm(vecs_test[ii])
            ax.arrow(
                0,
                0,
                vec_test[0],
                vec_test[1],
                color="g",
                label="Initial",
                **arrow_props,
            )
            vec_rot = vecs_rot_list[ii]

            ax.arrow(
                0, 0, vec_rot[0], vec_rot[1], color="b", label="Rotated", **arrow_props
            )
            ax.legend()

        for ax in axs.flatten():
            ax.axis("equal")
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.grid()
            # ax.legend()

        if savefig:
            figure_name = "rotation_with_perpendicular_basis"
            plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def test_zero_rotation():
    vec_init = np.array([1, 0])
    vec_rot = np.array([1, 0])

    vector_rotation = VectorRotationXd.from_directions(vec_init, vec_rot)

    vec_test = np.array([1, 0])
    vec_rotated = vector_rotation.rotate(vec_test)
    assert np.allclose(vec_test, vec_rotated)

    vec_test = np.array([0, 1])
    vec_rotated = vector_rotation.rotate(vec_test)
    assert np.allclose(vec_test, vec_rotated)

    
def test_mirror_rotation():
    vec1 = np.array([0, 1])
    vec2 = np.array([1, 0])

    vector_rotation1 = VectorRotationXd.from_directions(vec1, vec2)
    vector_rotation2 = VectorRotationXd.from_directions(vec2, vec1)

    vec_rand = np.ones(2) / np.sqrt(2)

    # Back and forward rotation
    vec_rot = vector_rotation2.rotate(vector_rotation1.rotate(vec_rand))

    assert np.allclose(vec_rand, vec_rot)
    

def test_cross_rotation_3d():
    vec_init = np.array([1, 0, 0])
    vec_rot = np.array([0, 1, 0])

    vector_rotation = VectorRotationXd.from_directions(vec_init, vec_rot)

    vec_test = np.array([-1, 0, 0])
    vec_rotated = vector_rotation.rotate(vec_test)
    assert np.allclose(vec_rotated, [0, -1, 0]), "Not correct rotation."

    vec_test = np.array([0, 0, 1])
    vec_rotated = vector_rotation.rotate(vec_test)
    assert np.allclose(vec_rotated, vec_test), "No rotation expected."

    vec_test = np.ones(3)
    vec_test = vec_test / LA.norm(vec_test)
    vec_rotated = vector_rotation.rotate(vec_test)
    assert np.isclose(LA.norm(vec_rotated), 1), "Unit norm expected."


def test_multi_rotation_array():
    # Rotation from 1 to final
    vector_seq = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0],
        ]
    ).T

    rotation_seq = VectorRotationSequence(vector_seq)
    rotated_vec = rotation_seq.rotate(direction=np.array([1, 0, 0]))
    assert np.allclose(rotated_vec, [0, -1, 0]), "Unexpected rotation."

    rotation_seq = VectorRotationSequence(vector_seq)
    rotated_vec = rotation_seq.rotate_weighted(
        direction=np.array([1, 0, 0]), weights=np.array([0.5, 0.5, 0, 0])
    )
    out_vec = np.array([0, 1, 1]) / np.sqrt(2)
    assert np.allclose(rotated_vec, out_vec), "Not rotated into second plane."


def test_rotation_tree():
    new_tree = VectorRotationTree(root_id=0, root_direction=np.array([0, 1]))

    new_tree.add_node(node_id=10, direction=np.array([1, 0]), parent_id=0)
    new_tree.add_node(node_id=20, direction=np.array([1, 0]), parent_id=10)

    new_tree.add_node(node_id=30, direction=np.array([-1, 0]), parent_id=0)
    new_tree.add_node(node_id=40, direction=np.array([-1, 0]), parent_id=30)

    # Full rotation in one direction
    weighted_mean = new_tree.get_weighted_mean(
        node_list=[0, 20, 40],
        weights=[0, 0, 1],
    )

    assert np.isclose(new_tree._graph.nodes[0]["weight"], 1)
    assert np.isclose(new_tree._graph.nodes[10]["weight"], 0)
    assert np.isclose(new_tree._graph.nodes[30]["weight"], 1)
    assert np.isclose(new_tree._graph.nodes[40]["weight"], 1)

    assert np.allclose(new_tree._graph.nodes[40]["direction"], weighted_mean)

    # Equal rotation
    weighted_mean = new_tree.get_weighted_mean(
        node_list=[0, 20, 40],
        weights=np.ones(3) / 3,
    )
    assert np.isclose(new_tree._graph.nodes[0]["weight"], 1)
    assert np.isclose(new_tree._graph.nodes[10]["weight"], 1.0 / 3)
    assert np.isclose(new_tree._graph.nodes[30]["weight"], 1.0 / 3)

    assert np.allclose(new_tree._graph.nodes[0]["direction"], weighted_mean)

    # Left / right rotation
    weighted_mean = new_tree.get_weighted_mean(
        node_list=[0, 20, 40],
        weights=[0.0, 0.5, 0.5],
    )
    assert np.isclose(new_tree._graph.nodes[0]["weight"], 1)

    assert np.allclose(new_tree._graph.nodes[0]["direction"], weighted_mean)

    # Left shift (half)
    weighted_mean = new_tree.get_weighted_mean(
        node_list=[0, 20, 40],
        weights=[0.5, 0.0, 0.5],
    )
    final_dir = (
        new_tree._graph.nodes[0]["direction"] + new_tree._graph.nodes[40]["direction"]
    )
    final_dir = final_dir / LA.norm(final_dir)

    # Check direction
    assert np.allclose(final_dir, weighted_mean)

    
def test_two_ellipse_with_normal_obstacle():
    # Simple obstacle which looks something like this:
    # ^   <-o
    # |     |
    # o  -  o
    new_tree = VectorRotationTree(root_id=0, root_direction=np.array([0, 1]))
    new_tree.add_node(node_id=1, direction=np.array([1, 0]), parent_id=0)
    new_tree.add_node(node_id=2, direction=np.array([-.2, 1]), parent_id=1)
    new_tree.add_node(node_id=3, direction=np.array([-1, 0]), parent_id=2, rotation_limit=False)

    weighted_mean = new_tree.get_weighted_mean(
        node_list=[0, 3],
        weights=[0.5, 0.5],
    )
    
    breakpoint()

    
if (__name__) == "__main__":
    plt.close("all")
    plt.ion()
    # test_cross_rotation_2d(visualize=False, savefig=0)
    # test_zero_rotation()
    # test_cross_rotation_3d()
    # test_multi_rotation_array()

    # test_rotation_tree()
    test_two_ellipse_with_normal_obstacle()

    print("\nDone with tests.")
