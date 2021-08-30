"""
Functions for the evluation of the hessian

NOTE: this class requires the sympy module for jacobian evaluation
      (if desired -> move it somewhere else)
"""
import numpy as np
from numpy import linalg as LA

import sympy

import matplotlib.pyplot as plt

from model_robot import ModelRobot2D

def _sympy_get_rotation_matrix_from_euler(val):
    # from sympy import sympy.co, sin, Matrix
    rot = sympy.Matrix([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])

    if val[0]:
        ang = val[0]
        rot1 = sympy.Matrix([[1, 0, 0],
                             [0, sympy.cos(ang), -sympy.sin(ang)],
                             [0, sympy.sin(ang), sympy.cos(ang)]])
        rot = rot @ rot1

    if val[1]:
        ang = val[1]
        rot1 = sympy.Matrix([[ sympy.cos(ang), 0, sympy.sin(ang)],
                             [0, 1, 0],
                             [-sympy.sin(ang), 0, sympy.cos(ang)]])
        rot = rot @ rot1

    if val[2]:
        ang = val[2]
        rot1 = sympy.Matrix([[ sympy.cos(ang), -sympy.sin(ang), 0],
                             [sympy.sin(ang), sympy.cos(ang), 0],
                             [0, 0, 1]])
        rot = rot @ rot1
    return rot

def _get_sympy_angles(robot):
    angles = []
    for ii in range(robot.n_links):
        angles.append(sympy.symbols("qq["+str(ii)+"]"))
    return angles

def _get_sympy_transformation_matrix(robot):
    link_lengths = []
    for ii in range(robot.n_links):
        link_lengths.append(sympy.symbols("ll["+str(ii)+"]"))
        # link_lengths.append(sympy.symbols("l"+str(ii)))
        # angles.append(sympy.symbols("q"+str(ii)))
    angles = _get_sympy_angles(robot)

    dim = 3
    tot_trafo = sympy.eye((dim+1))
    tot_trafo[:dim, -1] = [link_lengths[-1], 0, 0]
    
    for ii in reversed(range(1, robot.n_joints)):
        tranformation_matrix = sympy.eye((dim+1))
        tranformation_matrix[:dim, :dim] = _sympy_get_rotation_matrix_from_euler(
            [0, 0, angles[ii]])
        tranformation_matrix[:dim, -1] = sympy.Matrix([[link_lengths[ii-1], 0, 0]]).T
        tot_trafo =  tranformation_matrix @ tot_trafo

    tranformation_matrix = sympy.eye((dim+1))
    tranformation_matrix[:dim, :dim] = _sympy_get_rotation_matrix_from_euler(
            [0, 0, angles[0]])
    tot_trafo =  tranformation_matrix @ tot_trafo
        
    return tot_trafo

def analytic_evaluation_jacobian(robot):
    """ Symbolic evaluation of the jacobian."""
    tot_trafo = _get_sympy_transformation_matrix(robot)
    import sympy
    angles = _get_sympy_angles(robot)

    # base_rot_matr = _sympy_get_rotation_matrix_from_euler(alpha, beta, gamma)
    # Jacobian implemented only for the 2D case
    jacobian = sympy.zeros(2+1, robot.n_joints)
    for ii in range(robot.n_joints):
        jacobian[:2, ii] = sympy.diff(tot_trafo[:2, -1] , angles[ii])
        jacobian[2, ii] = 1 # is it 1 in 2D (?!)

    jacobian_file = "get_2d_jacobian_matrix.py"
    with open(jacobian_file, 'w') as outfile:
        # Define header
        outfile.write("import numpy as np \n")
        outfile.write("from numpy import cos, sin \n\n")

        outfile.write("def get_2d_jacobian_matrix(ll, qq):\n")
        outfile.write("    return np.array([")
        for ii in range(jacobian.shape[0]):
            line_str_list = [] 
            line_str_list.append("[")
            for jj in range(jacobian.shape[1]):
                line_str_list.append(str(jacobian[ii, jj]))

                if jj != jacobian.shape[1]-1:
                    line_str_list.append(", ")

            # If not last iteration element
            if jj < jacobian.shape[1]-1:
                line_str_list.append("], ")
            else:
                line_str_list.append("],")
            line_str_list.append("\n")

            outfile.write("".join(line_str_list))
        outfile.write("])")
    print("Done")


def test_similarity_of_analytic_and_numerical_rotation_matr(visualize=False):
    """ Test the forward transform-matrix for different joint-state configurations. """
    my_robot = ModelRobot2D()

    initial_pos_list = [
        [30, -10, -20, 30],
        [10, 20, 30, 30],
        [10, -60, 130, 30],
        [90, 120, 170, -30],
        ]
    
    for initial_pos in initial_pos_list:
        initial_pos = np.array(initial_pos)
        my_robot.set_joint_state(initial_pos, input_unit='deg')

        trafo_matr = _get_sympy_transformation_matrix(my_robot)
        init_pose = my_robot._joint_state
        for ii in range(init_pose.shape[0]):
            qq = sympy.symbols("qq["+str(ii)+"]")
            trafo_matr = trafo_matr.subs(qq, init_pose[ii])
            ll = sympy.symbols("ll["+str(ii)+"]")
            trafo_matr = trafo_matr.subs(ll, my_robot._link_lengths[ii])

        trafo_matr_eval = np.round(np.array(trafo_matr.evalf()).astype(float), 3)
        position_analytical = trafo_matr_eval[:2, -1]

        my_robot.set_joint_state(initial_pos, input_unit='deg')
        ee_pos0 = my_robot.get_ee_in_base()

        if visualize:
            import matplotlib.pyplot as plt
            x_lim = [-0.2, 4.5]
            y_lim = [-4, 4]
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            my_robot.draw_robot(ax=ax)

        # print(f"{position_analytical=}")
        # print(f"{ee_pos0=}")
        assert np.allclose(ee_pos0, position_analytical, rtol=1e-2), \
               "Analytical & numerical Transformation are not close to each other..."

    print("Test for analyitical vs. numerical transform is done.")


if (__name__) == "__main__":
    # print("Done")
    # analytic_evaluation_jacobian(robot=ModelRobot2D())
    plt.close('all')
    plt.ion()
    
    test_similarity_of_analytic_and_numerical_rotation_matr(visualize=False)

    pass
