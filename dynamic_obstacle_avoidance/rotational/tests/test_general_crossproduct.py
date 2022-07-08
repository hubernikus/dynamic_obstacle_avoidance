#!/USSR/bin/python3
""" Create the rotation space which is so much needed. ... """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-07-07

import math

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
import numpy.typing as npt

Vector = npt.ArrayLike


class VectorRotationXd:
    """This approach allows successive modulation which can be added up (!)"""

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

        self.dot_prod = dot_prod

        self.rot_angle = np.arccos(self.dot_prod)

    @property
    def weights(self):
        return np.array([self.dot_prod, math.sqrt(1 - self.dot_prod * self.dot_prod)])

    def rotate(self, vector, rot_factor: float = 1):
        """Normalize just to make sure.

        rot_factor: factor gives information about extension of rotation"""
        vector /= LA.norm(vector)
        dot_prods = np.dot(self.base.T, vector)
        angle = np.arctan2(dot_prods[1], dot_prods[0])

        angle += self.rot_angle * rot_factor

        # Output vector is calculated from rotation + the orthogonal part
        # (no effect in 2D, but important for higher dimensions)
        out_vector = np.cos(angle) * self.base[:, 0] + np.sin(angle) * self.base[:, 1]
        out_vector += vector - np.sum(dot_prods * self.base, axis=1)
        return out_vector

    def inverse_rotate(self, vector):
        return self.rotate(vector, rot_factor=(-1))


def test_cross_rotation_2d(visualize=True):
    vec_init = np.array([1, 0.3])
    vec_rot = np.array([0.4, -1])

    vector_rotation = VectorRotationXd(vec_init, vec_rot)

    # vec_init /= LA.norm(vec_init)
    # vec_rot /= LA.norm(vec_init)
    cross_prod_base = np.cross(vec_init / LA.norm(vec_init), vec_rot / LA.norm(vec_rot))

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
        label="base",
        **arrow_props,
    )

    ax.arrow(0, 0, vec_init[0], vec_init[1], color="g", label="Init", **arrow_props)
    ax.arrow(0, 0, vec_rot[0], vec_rot[1], color="b", label="Rotat", **arrow_props)

    vec_test = np.array([1, -1.2])
    ax = axs[0, 1]
    axs_test = axs.flatten()[1:]
    vecs_test = [
        [1, -1.2],
        [-1.2, -1],
        [-1.2, 1.3],
    ]
    for ii, ax in enumerate(axs_test):
        vec_test = np.array(vecs_test[ii])

        vec_test /= LA.norm(vec_test)
        ax.arrow(0, 0, vec_test[0], vec_test[1], color="g", label="Init", **arrow_props)
        vec_rot = vector_rotation.rotate(vec_test)
        ax.arrow(0, 0, vec_rot[0], vec_rot[1], color="b", label="Rot", **arrow_props)
        ax.legend()

        assert np.isclose(
            cross_prod_base, np.cross(vec_test, vec_rot)
        ), "Vectors are not close"

    for ax in axs.flatten():
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.axis("equal")
        ax.grid()
        # ax.legend()


if (__name__) == "__main__":
    plt.close("all")
    test_cross_rotation_2d()
