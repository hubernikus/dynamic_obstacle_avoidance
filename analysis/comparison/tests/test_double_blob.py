"""
Double Blob Obstacle Test
"""
import unittest

import numpy as np
from numpy import linalg as LA

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from double_blob_obstacle import DoubleBlob


class DoubleBlobTest(unittest.TestCase):
    def test_double_blob_creation(self):
        double_blob = DoubleBlob(
            a_value=1, b_value=1.1,
            center_position=[0, 3]
            )

        # Test draw obstacle function
        double_blob.draw_obstacle()

    def test_gamma_of_double_blob(self):
        double_blob = DoubleBlob(
            a_value=1, b_value=1.1,
            center_position=[0, 3]
            )

        position = np.array([-1.41, 3.99])
        gamma = double_blob.get_gamma(position, in_global_frame=True)
        self.assertTrue(gamma>1)

        position = np.array([-0.24, 2.31])
        gamma = double_blob.get_gamma(position, in_global_frame=True)
        self.assertTrue(gamma>1)

        position = np.array([0.78, 2.85])
        gamma = double_blob.get_gamma(position, in_global_frame=True)
        self.assertTrue(gamma<1)

    
def draw_double_blob():
    import matplotlib.pyplot as plt
    from matplotlib import cm

    x_lim = [-2, 2]
    y_lim = [-0.5, 6]

    double_blob = DoubleBlob(a_value=1, b_value=1.1)

    obs_list = ObstacleContainer()
    obs_list.append(
        DoubleBlob(a_value=1, b_value=1.1,
                   center_position=[0, 3]
                   ))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    plot_obstacles(ax, obs_list, x_lim, y_lim)


def draw_barrier_value():
    n_grid = 100
    x_vals, y_vals = np.meshgrid(np.linspace(x_lim[0], x_lim[1], n_grid),
                                 np.linspace(y_lim[0], y_lim[1], n_grid),
                                 )
    
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    values = np.zeros(positions.shape[1])

    for ii in range(positions.shape[1]):
        values[ii] = obs_list[0].get_gamma(positions[:, ii], in_global_frame=True)

    cs = ax.contourf(positions[0, :].reshape(n_grid, n_grid),
                    positions[1, :].reshape(n_grid, n_grid),
                    values.reshape(n_grid, n_grid),
                    # np.linspace(-10.0, 100.0, 11),
                    np.linspace(-10.0, 10.0, 11),
                    # vmin=-0.1, vmax=0.1,
                    # np.linspace(-10, 10.0, 101),
                    # cmap=cm.YlGnBu,
                    # linewidth=0.2, edgecolors='k'
                    )
    
    cbar = fig.colorbar(cs,
                        # ticks=np.linspace(-10, 0, 11)
                        )

    # plt.grid()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    

if (__name__) == "__main__":
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)
    visualize = True
    if visualize:
        plt.close('all')
        plt.ion()
        draw_double_blob()
        plt.show()
