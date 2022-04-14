"""
Double Blob Obstacle Test
"""
import unittest

import numpy as np
from numpy import linalg as LA

from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from dynamic_obstacle_avoidance.obstacles import DoubleBlob


class TestDoubleBlob(unittest.TestCase):
    def test_double_blob_creation(self):
        double_blob = DoubleBlob(a_value=1, b_value=1.1, center_position=[0, 3])

        # Test draw obstacle function
        # double_blob.draw_obstacle()

    def test_gamma_of_double_blob(self):
        double_blob = DoubleBlob(a_value=1, b_value=1.1, center_position=[0, 3])

        position = np.array([-1.41, 3.99])
        gamma = double_blob.get_gamma(position, in_global_frame=True)
        self.assertTrue(gamma > 1)

        position = np.array([-0.24, 2.31])
        gamma = double_blob.get_gamma(position, in_global_frame=True)
        self.assertTrue(gamma > 1)

        position = np.array([0.78, 2.85])
        gamma = double_blob.get_gamma(position, in_global_frame=True)
        self.assertTrue(gamma < 1)

    def test_normal_double_blob(self, visualize=False):
        """Test if the normal is pointing outwards on the double-blob obstacle."""
        obs = DoubleBlob(a_value=1, b_value=1.1, center_position=np.array([0.0, 3.0]))

        position = np.array([1, 2])

        normal = obs.get_normal_direction(position=position, in_global_frame=True)

        ref_dir = obs.get_reference_direction(position=position, in_global_frame=True)

        self.assertTrue(
            normal.dot(ref_dir) >= 0,
            f"Normal/reference error at position={position}",
        )

        if not visualize:
            return

        x_lim = [-2.05, 2.05]
        y_lim = [-0.3, 6.3]
        dim = 2

        n_grid = 10

        x_vals = np.linspace(x_lim[0], x_lim[1], n_grid)
        y_vals = np.linspace(y_lim[0], y_lim[1], n_grid)

        positions = np.zeros((dim, n_grid, n_grid))
        normals = np.zeros(positions.shape)
        ref_dirs = np.zeros(positions.shape)

        obs.get_normal_direction(position=np.array([-1.61, 3.36]), in_global_frame=True)

        for ix in range(10):
            for iy in range(10):
                positions[:, ix, iy] = np.array([x_vals[ix], y_vals[iy]])

                normals[:, ix, iy] = obs.get_normal_direction(
                    position=positions[:, ix, iy], in_global_frame=True
                )

                ref_dirs[:, ix, iy] = obs.get_reference_direction(
                    position=positions[:, ix, iy], in_global_frame=True
                )

                self.assertTrue(
                    normals[:, ix, iy].dot(ref_dirs[:, ix, iy]) >= 0,
                    f"Normal/reference error at position={positions[:, ix, iy]}",
                )

        if visualize:
            fig, ax = plt.subplots()

            ax.quiver(
                positions[0, :, :],
                positions[1, :, :],
                normals[0, :, :],
                normals[1, :, :],
                color="green",
            )

            ax.quiver(
                positions[0, :, :],
                positions[1, :, :],
                ref_dirs[0, :, :],
                ref_dirs[1, :, :],
                color="blue",
            )

            obs.plot_obstacle(ax=ax, fill_color=None, outline_color="black")

            ax.set_aspect("equal", adjustable="box")


def draw_double_blob():
    import matplotlib.pyplot as plt
    from matplotlib import cm

    x_lim = [-2, 2]
    y_lim = [-0.5, 6]

    obs_list = ObstacleContainer()
    obs_list.append(DoubleBlob(a_value=1, b_value=1.1, center_position=[0, 3]))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    plot_obstacles(ax, obs_list, x_lim, y_lim)


if (__name__) == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

    # visualize = True
    visualize = False
    if visualize:
        import matplotlib.pyplot as plt

        plt.close("all")
        plt.ion()
        # draw_double_blob()

        # draw_barrier_value()
        my_tester = TestDoubleBlob()
        # my_tester.test_double_blob_creation()

        my_tester.test_normal_double_blob(visualize=False)

        plt.show()
