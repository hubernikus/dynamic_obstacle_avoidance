#!/USSR/bin/python3
""" Create the rotation space which is so much needed. ... """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-07-07

import warnings
import math

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
import numpy.typing as npt

Vector = npt.ArrayLike
VectorArray = npt.ArrayLike


class SummedVectorRoation:
    pass


class VectorRotationSequence:
    def __init__(self, vectors_array: np.array) -> None:
        # Normalize
        self.vectors_array = vectors_array / LA.norm(vectors_array, axis=1)

        dot_prod = np.dot(self.vectors_array[:, 1:], self.vectors_array[:, :-1])

        if np.sum(np.abs(dot_prod)) == 1:
            raise ValueError("Antiparallel vectors.")

        # Evaluate base and angles
        vec_perp = self.vectors_array[:, 1:] - self.vectors_array[:, :-1] * dot_prod
        vec_perp = vec_perp / LA.norm(vec_perp)

        self.bases = np.stack((self.vectors_array[:, :-1], vec_perp), axis=2)
        self.rot_angles = np.arccos(dot_prod)

    def base(self) -> Vector:
        return self.bases_array[:, [0, -1]]

    def append(self, direction: Vector) -> None:
        self.bases_array = np.hstack((self.bases_array, direction.reshape(-1, 1)))

    def _update_partial_rotations(self, rot_factor):
        self._temp_angles = None
        self._temp_bases = None
        pass

    def rotate(self, direction: Vector, rot_factor: float = 1) -> None:
        # if not math.isclose(abs(rot_factor), 1):
        pass

    def weighted_rotate(self, direction: Vector, weights: list):

        # Starting at the root
        cumsum_weights = np.cumsum(weights[::-1])[::-1]

        if cumsum_weights[0] > 1:
            warnings.warn("Weights are summing up to more than 1.")


class VectorRotationXd:
    """This approach allows successive modulation which can be added up (!)

    Compared to directional space (!)."""

    def __init__(self, vec_init: Vector, vec_rot: Vector):

        # Normalize both vectors
        vec_init = vec_init / LA.norm(vec_init)
        vec_rot = vec_rot / LA.norm(vec_rot)

        dot_prod = np.dot(vec_init, vec_rot)
        if abs(dot_prod) == 1:
            raise ValueError("Antiparallel vectors")

        vec_perp = vec_rot - vec_init * dot_prod
        vec_perp = vec_perp / LA.norm(vec_perp)

        self.base = np.array([vec_init, vec_perp]).T

        # self.dot_prod = dot_prod
        self.rot_angle = np.arccos(dot_prod)

    def rotate(self, vector, rot_factor: float = 1):
        """Returns the rotated of the input vector with respect to the base and rotation angle

        Normalize just to make sure.

        rot_factor: factor gives information about extension of rotation"""
        vector = vector / LA.norm(vector)

        dot_prods = np.dot(self.base.T, vector)
        angle = math.atan2(dot_prods[1], dot_prods[0])

        angle += self.rot_angle * rot_factor

        # Compute output from rotation
        out_vector = (
            math.cos(angle) * self.base[:, 0] + math.sin(angle) * self.base[:, 1]
        )
        # Scale with dot_prods
        out_vector *= math.sqrt(sum(dot_prods**2))
        # Finally, add the orthogonal part (no effect in 2D, but important for higher dimensions)
        out_vector += vector - np.sum(dot_prods * self.base, axis=1)
        return out_vector

    def inverse_rotate(self, vector):
        return self.rotate(vector, rot_factor=(-1))


def test_cross_rotation_2d(visualize=False, savefig=False):
    vec0 = np.array([1, 0.3])
    vec1 = np.array([1.0, -1])

    vector_rotation = VectorRotationXd(vec0, vec1)

    vec0 /= LA.norm(vec0)
    vec1 /= LA.norm(vec1)
    cross_prod_base = np.cross(vec0, vec1)

    arrow_props = {"head_length": 0.1, "head_width": 0.05}

    vecs_test = [
        [1, -1.2],
        [-1.2, -1],
        [-1.2, 1.3],
    ]

    vecs_rot_list = []
    for ii, vec in enumerate(vecs_test):
        vec_test = np.array(vecs_test[ii])
        vec_test /= LA.norm(vec_test)
        vec_rot = vector_rotation.rotate(vec_test)

        assert np.isclose(
            cross_prod_base, np.cross(vec_test, vec_rot)
        ), "Vectors are not close"

        # For visualization purposes
        vecs_rot_list.append(vec_rot)

    if visualize:
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


def test_cross_rotation_3d():
    vec_init = np.array([1, 0, 0])
    vec_rot = np.array([0, 1, 0])

    vector_rotation = VectorRotationXd(vec_init, vec_rot)

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


if (__name__) == "__main__":
    plt.close("all")
    test_cross_rotation_2d(visualize=True, savefig=1)
    test_cross_rotation_3d()

    print("\nDone with tests.")
