"""
Functions for the evluation of the hessian

NOTE: this class requires the sympy module for jacobian evaluation
      (if desired -> move it somewhere else)
"""
import os

import numpy as np
from numpy import linalg as LA

import sympy

import matplotlib.pyplot as plt

from robot_avoidance.model_robot import ModelRobot2D


def _sympy_get_rotation_matrix_from_euler(val):
    # from sympy import sympy.co, sin, Matrix
    rot = sympy.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    if val[0]:
        ang = val[0]
        rot1 = sympy.Matrix(
            [
                [1, 0, 0],
                [0, sympy.cos(ang), -sympy.sin(ang)],
                [0, sympy.sin(ang), sympy.cos(ang)],
            ]
        )
        rot = rot @ rot1

    if val[1]:
        ang = val[1]
        rot1 = sympy.Matrix(
            [
                [sympy.cos(ang), 0, sympy.sin(ang)],
                [0, 1, 0],
                [-sympy.sin(ang), 0, sympy.cos(ang)],
            ]
        )
        rot = rot @ rot1

    if val[2]:
        ang = val[2]
        rot1 = sympy.Matrix(
            [
                [sympy.cos(ang), -sympy.sin(ang), 0],
                [sympy.sin(ang), sympy.cos(ang), 0],
                [0, 0, 1],
            ]
        )
        rot = rot @ rot1
    return rot


def _get_sympy_angles(robot):
    angles = []
    for ii in range(robot.n_links):
        angles.append(sympy.symbols("qq[" + str(ii) + "]"))
    return angles


def _get_sympy_transformation_matrix(robot):
    link_lengths = []
    for ii in range(robot.n_links):
        link_lengths.append(sympy.symbols("ll[" + str(ii) + "]"))
        # link_lengths.append(sympy.symbols("l"+str(ii)))
        # angles.append(sympy.symbols("q"+str(ii)))
    angles = _get_sympy_angles(robot)

    dim = 3
    tot_trafo = sympy.eye((dim + 1))
    tot_trafo[:dim, -1] = [link_lengths[-1], 0, 0]

    for ii in reversed(range(1, robot.n_joints)):
        tranformation_matrix = sympy.eye((dim + 1))
        tranformation_matrix[:dim, :dim] = _sympy_get_rotation_matrix_from_euler(
            [0, 0, angles[ii]]
        )
        tranformation_matrix[:dim, -1] = sympy.Matrix([[link_lengths[ii - 1], 0, 0]]).T
        tot_trafo = tranformation_matrix @ tot_trafo

    tranformation_matrix = sympy.eye((dim + 1))
    tranformation_matrix[:dim, :dim] = _sympy_get_rotation_matrix_from_euler(
        [0, 0, angles[0]]
    )
    tot_trafo = tranformation_matrix @ tot_trafo

    return tot_trafo


def analytic_evaluation_jacobian(robot, symplify_expression: bool = True):
    """Symbolic evaluation of the jacobian.

    Arguments
    ---------
    symplify_expression: bool to decided whether to simplify analtic expression
        (since it's relatively time-intense (<5 seconds))
    """
    tot_trafo = _get_sympy_transformation_matrix(robot)
    angles = _get_sympy_angles(robot)

    # Differentiation
    jacobian = sympy.zeros(2 + 1, robot.n_joints)
    for ii in range(robot.n_joints):
        jacobian[:2, ii] = sympy.diff(tot_trafo[:2, -1], angles[ii])
        jacobian[2, ii] = 1  # is it 1 in 2D (?!)

    if symplify_expression:
        jacobian = sympy.simplify(jacobian)

    jacobian_file = os.path.join(
        "src", "robot_avoidance", "jacobians", robot.name + ".py"
    )

    with open(jacobian_file, "w") as outfile:
        # Define header
        outfile.write("import numpy as np \n")
        outfile.write("from numpy import cos, sin \n\n")

        outfile.write("def _get_jacobian(ll, qq):\n")
        outfile.write("    return np.array([")
        for ii in range(jacobian.shape[0]):
            line_str_list = []
            line_str_list.append("[")
            for jj in range(jacobian.shape[1]):
                line_str_list.append(str(jacobian[ii, jj]))

                if jj != jacobian.shape[1] - 1:
                    line_str_list.append(", ")

            # If not last iteration element
            if jj < jacobian.shape[1] - 1:
                line_str_list.append("], ")
            else:
                line_str_list.append("],")
            line_str_list.append("\n")

            outfile.write("".join(line_str_list))
        outfile.write("])")

    print("Jacobian function written to file.")
