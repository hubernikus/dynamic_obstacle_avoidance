"""
Visual of 'weighting' function to help with debugging.
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021
import unittest

import copy
from math import pi

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
import matplotlib as mpl

from vartools.directional_space import UnitDirection, DirectionBase
from vartools.directional_space.visualization import circular_space_setup

from dynamic_obstacle_avoidance.avoidance.rotation import _get_projection_of_inverted_convergence_direction
from dynamic_obstacle_avoidance.avoidance.rotation import _get_projected_nonlinear_velocity

# plt.close('all')
plt.ion()

def visualize_displacements(
    inv_nonlinear_list: list, inv_conv_rotated_list: list,
    inv_convergence_radius=pi/2,
    radius_name=None, save_figure=False, visualize=False
    ):
    dim = 3
    
    inv_nonlinear_list = copy.deepcopy(inv_nonlinear_list)
    inv_conv_rotated_list = copy.deepcopy(inv_conv_rotated_list)

    if not isinstance(inv_nonlinear_list[0], DirectionBase):
        base = DirectionBase(np.eye(dim))
        
        for ii, inv in enumerate(inv_nonlinear_list):
            inv_nonlinear_list[ii] = UnitDirection(base).from_angle(np.array(inv))
            
        for ii, inv in enumerate(inv_conv_rotated_list):
            inv_conv_rotated_list[ii] = UnitDirection(base).from_angle(np.array(inv))

    if visualize:
        fig, ax = plt.subplots(figsize=(7.2, 7))

    for it_nonl, inv_nonlinear in enumerate(inv_nonlinear_list):
        for it_conv, inv_conv_rotated in enumerate(inv_conv_rotated_list):
            inv_conv_proj = _get_projection_of_inverted_convergence_direction(
                inv_conv_rotated=inv_conv_rotated,
                inv_nonlinear=inv_nonlinear,
                inv_convergence_radius=inv_convergence_radius,
                )

            dir_nonl_rotated = _get_projected_nonlinear_velocity(
                dir_conv_rotated=inv_conv_rotated.invert_normal(),
                dir_nonlinear=inv_nonlinear.invert_normal(),
                weight=1, convergence_radius=(pi-inv_convergence_radius),
                )
            inv_nonl_rotated = dir_nonl_rotated.invert_normal()

            if not visualize:
                continue

            plt.plot([inv_nonlinear._angle[0], inv_conv_rotated._angle[0]],
                     [inv_nonlinear._angle[1], inv_conv_rotated._angle[1]],
                     '-', color='k', alpha=0.5)

            plt.plot(inv_conv_rotated._angle[0], inv_conv_rotated._angle[1], 'o', color='b')
            plt.plot(inv_conv_proj._angle[0], inv_conv_proj._angle[1], '+', color='b',
                     markersize=8, markeredgewidth=3)
            
            plt.plot(inv_nonlinear._angle[0], inv_nonlinear._angle[1], 'o', color='r',)
            plt.plot(inv_nonl_rotated._angle[0], inv_nonl_rotated._angle[1], 'x', color='r',
                     markersize=8, markeredgewidth=3)

    if not visualize:
        return
    
    plt.plot([], [], 'o', color='r', label=r"$f_{nonl}^\angle$")
    plt.plot([], [], 'o', color='b', label=r"$f_{conv}^\angle$")
    plt.plot([], [], '+', color='b' , label=r"$f_{c,proj}^\angle$",
             markersize=8, markeredgewidth=3)
    plt.plot([], [], 'x', color='r' , label=r"$f_{n,proj}^\angle$",
             markersize=8, markeredgewidth=3)
    # plt.plot([], [], '--', color='k', legend=r"$f_{proj}^\angle$")

    ax.legend()

    circular_space_setup(ax, circle_background_color='#bee8c2',
                         outer_boundary_color='#ffbebd',
                         circ_radius=inv_convergence_radius)

    if save_figure:
        figure_name = "multiple_nonlinear_total_displacement" + radius_name
        plt.savefig("figures/" + figure_name+radius_name + ".png", bbox_inches='tight')

class TestProjectionOfDisplacement(unittest.TestCase):
    def check_if_inbetween(self, vec1, vec2, vec3):
        """ Check if vec2 lies inbetween vec1 & vec3 on one line. """
        vec21 = (vec2 - vec1).as_angle()
        vec32 = (vec3 - vec2).as_angle()

        # Only for nonzero vectors
        if not LA.norm(vec21) or LA.norm(vec32):
            return

        dot_normalized = np.dot(vec32, vec21) / (LA.norm(vec32), LA.norm(vec21))
        serf.assertTrue(np.isclose(dot_normalized, 1))
        
    def test_radius_pi_quarter(self):
        dim = 3
        base = DirectionBase(np.eye(dim))
        inv_conv_rotated = UnitDirection(base).from_angle(np.array([-0.2*pi, pi*0.5]))
        inv_nonlinear = UnitDirection(base).from_angle(np.array([0.4*pi, 0.6*pi]))
        inv_convergence_radius = pi/4

        inv_conv_proj = _get_projection_of_inverted_convergence_direction(
            inv_conv_rotated=inv_conv_rotated,
            inv_nonlinear=inv_nonlinear,
            inv_convergence_radius=inv_convergence_radius,
            )

        dir_nonl_rotated = _get_projected_nonlinear_velocity(
            dir_conv_rotated=inv_conv_rotated.invert_normal(),
            dir_nonlinear=inv_nonlinear.invert_normal(),
            weight=1, convergence_radius=(pi-inv_convergence_radius),
            )

        inv_nonl_rotated = dir_nonl_rotated.transform_to_base(base)

        self.assertTrue(dir_nonl_rotated.norm() > inv_convergence_radius)
        self.assertTrue(np.allclose(inv_conv_proj.as_angle(), inv_conv_rotated.as_angle()))
        # self.assertTrue(inv_conv_proj)

    def test_radius_pi_half(self):
        dim = 3
        base = DirectionBase(np.eye(dim))
        
        inv_conv_rotated = UnitDirection(base).from_angle(np.array([-0.8*pi, -pi*0.2]))
        inv_nonlinear = UnitDirection(base).from_angle(np.array([0.4*pi, 0.6*pi]))
        inv_convergence_radius = pi/2

        inv_conv_proj = _get_projection_of_inverted_convergence_direction(
            inv_conv_rotated=inv_conv_rotated,
            inv_nonlinear=inv_nonlinear,
            inv_convergence_radius=inv_convergence_radius,
            )

        dir_nonl_rotated = _get_projected_nonlinear_velocity(
            dir_conv_rotated=inv_conv_rotated.invert_normal(),
            dir_nonlinear=inv_nonlinear.invert_normal(),
            weight=1, convergence_radius=(pi-inv_convergence_radius),
            )
        inv_nonl_rotated = dir_nonl_rotated.transform_to_base(base)

        # self.assertTrue(np.isclose(dir_nonl_rotated.norm(), inv_convergence_radius))
        self.check_if_inbetween(inv_conv_proj, inv_nonl_rotated, inv_nonlinear)
        self.check_if_inbetween(inv_conv_rotated, inv_conv_proj, inv_nonlinear)

    def test_radius_pi_three_quarter(self):
        dim = 3
        base = DirectionBase(np.eye(dim))
        inv_conv_rotated = UnitDirection(base).from_angle(np.array([-0.2*pi, pi*0.5]))
        inv_nonlinear = UnitDirection(base).from_angle(np.array([0.4*pi, 0.6*pi]))
        inv_convergence_radius = 3*pi/2

        inv_conv_proj = _get_projection_of_inverted_convergence_direction(
            inv_conv_rotated=inv_conv_rotated,
            inv_nonlinear=inv_nonlinear,
            inv_convergence_radius=inv_convergence_radius,
            )

        dir_nonl_rotated = _get_projected_nonlinear_velocity(
            dir_conv_rotated=inv_conv_rotated.invert_normal(),
            dir_nonlinear=inv_nonlinear.invert_normal(),
            weight=1, convergence_radius=(pi-inv_convergence_radius),
            )

        inv_nonl_rotated = dir_nonl_rotated.invert_normal()

        self.assertTrue(np.allclose(inv_conv_rotated.as_angle(), inv_conv_proj.as_angle()))
        self.assertTrue(np.allclose(inv_nonlinear.as_angle(), inv_nonl_rotated.as_angle()))

    def test_radius_pi(self):
        dim = 3
        base = DirectionBase(np.eye(dim))
        inv_conv_rotated = UnitDirection(base).from_angle(np.array([-0.2*pi, pi*0.5]))
        inv_nonlinear = UnitDirection(base).from_angle(np.array([0.4*pi, 0.6*pi]))
        inv_convergence_radius = pi

        inv_conv_proj = _get_projection_of_inverted_convergence_direction(
            inv_conv_rotated=inv_conv_rotated,
            inv_nonlinear=inv_nonlinear,
            inv_convergence_radius=inv_convergence_radius,
            )

        dir_nonl_rotated = _get_projected_nonlinear_velocity(
            dir_conv_rotated=inv_conv_rotated.invert_normal(),
            dir_nonlinear=inv_nonlinear.invert_normal(),
            weight=1, convergence_radius=(pi-inv_convergence_radius),
            )

        inv_nonl_rotated = dir_nonl_rotated.invert_normal()

        self.assertTrue(np.allclose(inv_conv_rotated.as_angle(), inv_conv_proj.as_angle()))
        self.assertTrue(np.allclose(inv_nonlinear.as_angle(), inv_nonl_rotated.as_angle()))

    def test_radius_null(self):
        dim = 3
        base = DirectionBase(np.eye(dim))
        inv_conv_rotated = UnitDirection(base).from_angle(np.array([0.3*pi, pi*0.3]))
        inv_nonlinear = UnitDirection(base).from_angle(np.array([0.3*pi, 0.3*pi]))
        inv_convergence_radius = 0.001

        inv_conv_proj = _get_projection_of_inverted_convergence_direction(
            inv_conv_rotated=inv_conv_rotated,
            inv_nonlinear=inv_nonlinear,
            inv_convergence_radius=inv_convergence_radius,
            )

        dir_nonl_rotated = _get_projected_nonlinear_velocity(
            dir_conv_rotated=inv_conv_rotated.invert_normal(),
            dir_nonlinear=inv_nonlinear.invert_normal(),
            weight=1, convergence_radius=(pi-inv_convergence_radius),
            )

        inv_nonl_rotated = dir_nonl_rotated.invert_normal()

        self.assertTrue(np.allclose(inv_conv_rotated.as_angle(), inv_conv_proj.as_angle()))
        self.assertTrue(np.allclose(inv_nonlinear.as_angle(), inv_nonl_rotated.as_angle()))

    
if (__name__)=='__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    
    visualize = False
    if visualize:
        # radius_list = [pi/2]
        radius_list = [pi/4, pi/2, 3*pi/4]
        radius_name_list = [
            "_radius_of_pi_quarter", "_radius_of_pi_half" "_radius_of_three_pi_quarter"]
        inv_conv_rotated_list = [
            # [-0.2*pi, pi*0.5],
            # [-0.1*pi, -pi*0.2],
            [-0.8*pi, -pi*0.2],
            # [0.4*pi, -pi*0.4],
            # [0.7*pi, -pi*0.6],
            # [0.9*pi, -pi*0.1],
            # [0.6*pi, pi*0.7],
            ]
        
        inv_nonlinear_list = [[0.4*pi, 0.6*pi]]
        for radius_name, radius in zip(radius_name_list, radius_list):
            visualize_displacements(
                inv_nonlinear_list=inv_nonlinear_list,
                inv_conv_rotated_list=inv_conv_rotated_list,
                inv_convergence_radius=radius,
                radius_name=radius_name,
                save_figure=False,
                visualize=True,
                )
