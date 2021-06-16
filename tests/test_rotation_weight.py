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

class TestRotational(unittest.TestCase):
    @classmethod
    def rotation_weight(cls):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  
        from matplotlib import cm

        import numpy as np

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        n_grid = 10
        dist0 = np.linspace(0, 1, n_grid)
        weight = np.linspace(0, 1, n_grid)
        weight, dist0 = np.meshgrid(weight, dist0)

        # Make data.
        val = weight**2*dist0 / (1 + weight - dist0)
        # Plot the surface.
        surf = ax.plot_surface(dist0, weight, val,
                               # cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        # breakpoint()
        ax.set_xlabel('Distance')
        ax.set_ylabel('Weight')
        
if __name__ == '__main__':
    visualize = True
    if visualize:
        TestRotational.rotation_weight()
