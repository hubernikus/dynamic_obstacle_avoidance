#!/USSR/bin/python3.9
""" Test the directional space. """

__author__ = "LukasHuber"
__date__ = "2021-05-18"
__email__ = "lukas.huber@epfl.ch"

import unittest
# from math import pi

import numpy as np

# from dynamic_obstacle_avoidance.obstacles import Ellipse
# from dynamic_obstacle_avoidance.obstacles import MultiBoundaryContainer

def weight_invgamma(inv_gamma, pow_fac):
    return weight ** pow_fac

def weight_dist(dist, inv_gamma, pow_fac):
    return (1.0/(1- dist/(1-inv_gamma)))

class TestRotational(unittest.TestCase):
    @classmethod
    def rotation_weight(cls, save_figure=True):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  
        from matplotlib import cm

        import numpy as np

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        n_grid = 20
        dist0 = np.linspace(1e-6, 1-1e-6, n_grid)
        weight = np.linspace(1e-6, 1-1e-6, n_grid)
        weight, dist0 = np.meshgrid(weight, dist0)
        gamma = 1./weight

        # Make data.
        # val = weight**weight_fac * (dist0/(1-dist0/(1-weight)))**power_frac
        # power_frac = 1.0
        # weight0 = (1 - dist0)/(1 + (gamma/(gamma-1)))**power_frac
        # weight0 = (1 - (1-dist0)/( 1 + (gamma/(gamma-1))))**power_frac
        # weight0 = np.ones(weight.shape)

        # weight_fac = 5.0
        # weight1 = weight
        # weight1 = np.ones(weight.shape)
        
        # val =  weight0**power_frac * weight1**weight_fac

        pow_factor = 3.0
        val = weight ** (1.0/(pow_factor*dist0))
        # val = 1 - weight ** (dist0)
        # Plot the surface.
        surf = ax.plot_surface(dist0, weight, val,
                               cmap=cm.YlGnBu,
                               linewidth=0.2, edgecolors='k')
                               # antialiased=False)
        # breakpoint()
        import matplotlib as mpl
        mpl.rc('font',family='Times New Roman')
        ax.set_xlabel(r'Weight $1/\Gamma(\xi)$')
        ax.set_ylabel(r'Relative Rotation $\tilde d (\xi)$')
        ax.set_zlabel(r'Rotational Weights $w_r(\Gamma, \tilde{d})$')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_zlim([0, 1])

        if save_figure:
            figure_name = "rotational_weight_with_power10_" + int(power_factor*10)
            plt.savefig("figures/" + figure_name + ".png", bbox_inches='tight')
        
if __name__ == '__main__':
    visualize = True
    if visualize:
        TestRotational.rotation_weight(save_figure=False)
