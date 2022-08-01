#!/USSR/bin/python3
""" Create the rotation space which is so much needed. ... """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-07-07

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationXd
from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationTree
from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationSequence


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
    new_tree.add_node(node_id=2, direction=np.array([-0.2, 1]), parent_id=1)
    new_tree.add_node(
        node_id=3, direction=np.array([-1, 0]), parent_id=2, rotation_limit=False
    )


def test_multi_normal_tree():
    # Base-normal
    new_tree = VectorRotationTree(root_id=0, root_direction=np.array([0, 1]))

    # 1st object + normal
    new_tree.add_node(node_id=1, direction=np.array([1, 0]), parent_id=0)
    new_tree.add_node(node_id=2, direction=np.array([0.2, 1.0]), parent_id=1)
    new_tree.add_node(node_id=3, direction=np.array([-1.0, 0.0]), parent_id=2)

    # 2nd object + normal
    new_tree.add_node(node_id=4, direction=np.array([0.0, 1.0]), parent_id=1)
    new_tree.add_node(node_id=5, direction=np.array([-1.0, -0.2]), parent_id=4)
    new_tree.add_node(node_id=6, direction=np.array([0, -1.0]), parent_id=5)

    weighted_mean = new_tree.get_weighted_mean(
        node_list=[0, 3],
        weights=[0.5, 0.5],
    )

    final_dir = (
        new_tree._graph.nodes[0]["direction"] + new_tree._graph.nodes[3]["direction"]
    )
    final_dir = final_dir / LA.norm(final_dir)
    assert np.allclose(weighted_mean, final_dir)

    # 180 turn and several branches
    weighted_mean = new_tree.get_weighted_mean(
        node_list=[0, 6],
        weights=[0.5, 0.5],
    )
    assert np.allclose(np.array([-1.0, 0]), weighted_mean)


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()
    # test_cross_rotation_2d(visualize=False, savefig=0)
    # test_zero_rotation()
    # test_cross_rotation_3d()
    # test_multi_rotation_array()

    # test_rotation_tree()
    # test_rotation_tree()
    # test_multi_normal_tree()

    print("\nDone with tests.")
