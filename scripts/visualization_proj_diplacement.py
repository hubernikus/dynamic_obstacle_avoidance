""" Visual of 'weighting' function to help with debugging."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
import matplotlib as mpl

from math import pi

import numpy as np
from numpy import linalg as LA

from vartools.directional_space import UnitDirection, DirectionBase
from vartools.directional_space.visualization import circular_space_setup

plt.close('all')
plt.ion()

def visualize_displacements(
    inv_nonlinear_list: list,
    inv_conv_rotated_list: list,
    ):
    from dynamic_obstacle_avoidance.avoidance.rotation import _get_projection_of_inverted_convergions_direction
    from dynamic_obstacle_avoidance.avoidance.rotation import _get_projected_nonlinear_velocity
    dim = 3
    
    if not isinstance(inv_nonlinear_list[0], DirectionBase):
        base = DirectionBase(np.eye(dim))
        
        for ii, inv in enumerate(inv_nonlinear_list):
            inv_nonlinear_list[ii] = UnitDirection(base).from_angle(np.array(inv))
            
        for ii, inv in enumerate(inv_conv_rotated_list):
            inv_conv_rotated_list[ii] = UnitDirection(base).from_angle(np.array(inv))
        
    fig, ax = plt.subplots(figsize=(7.2, 7))

    for it_nonl, inv_nonlinear in enumerate(inv_nonlinear_list):
        for it_conv, inv_conv_rotated in enumerate(inv_conv_rotated_list):
            inv_conv_proj = _get_projection_of_inverted_convergions_direction(
                inv_conv_rotated=inv_conv_rotated,
                inv_nonlinear=inv_nonlinear,
                inv_convergence_radius=pi/2,
                )

            dir_nonl_rotated = _get_projected_nonlinear_velocity(
                dir_conv_rotated=inv_conv_rotated.invert_normal(),
                dir_nonlinear=inv_nonlinear.invert_normal(),
                weight=1, convergence_radius=pi/2,
                )
            inv_nonl_rotated = dir_nonl_rotated.invert_normal()

            plt.plot([inv_nonlinear._angle[0], inv_conv_rotated._angle[0]],
                     [inv_nonlinear._angle[1], inv_conv_rotated._angle[1]],
                     '-', color='k', alpha=0.5)

            plt.plot(inv_conv_rotated._angle[0], inv_conv_rotated._angle[1], 'o', color='b')
            plt.plot(inv_conv_proj._angle[0], inv_conv_proj._angle[1], '+' , color='b')
            
            plt.plot(inv_nonlinear._angle[0], inv_nonlinear._angle[1], 'o', color='r')
            plt.plot(inv_nonl_rotated._angle[0], inv_nonl_rotated._angle[1], 'x', color='r') 
            
            
    plt.plot([], [], 'o', color='r', label=r"$f_{nonl}^\angle$")
    plt.plot([], [], 'o', color='b', label=r"$f_{conv}^\angle$")
    plt.plot([], [], '+', color='b' , label=r"$f_{c,proj}^\angle$")
    plt.plot([], [], 'x', color='r' , label=r"$f_{n,proj}^\angle$")
    # plt.plot([], [], '--', color='k', legend=r"$f_{proj}^\angle$")

    ax.legend()

    circular_space_setup(ax, circle_background_color='#bee8c2', outer_boundary_color='#ffbebd')

if (__name__)=='__main__':
    visualize_displacements(
        inv_nonlinear_list=[[0.4*pi, 0.6*pi]],
        inv_conv_rotated_list=[
        [-0.2*pi, pi*0.5],
        [-0.1*pi, -pi*0.2],
        [-0.8*pi, -pi*0.2],
        [0.4*pi, -pi*0.4],
        ]
        )
    figure_name = "multiple_nonlinear_total_displacement"
    plt.savefig("figures/" + figure_name + ".png", bbox_inches='tight')
