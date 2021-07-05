# Done#!/USSR/bin/python3.9
""" Test overrotation for ellipses. """
# Author: Lukas Huber
# Date: 2021-05-18
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import unittest
from math import pi

import numpy as np

from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.containers import MultiBoundaryContainer
from dynamic_obstacle_avoidance.visualization.gamma_field_visualization import gamma_field_visualization
# from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.visualization import Simulation_vectorFields, plot_obstacles

from dynamic_obstacle_avoidance.avoidance.multihull_convergence import get_desired_radius, multihull_attraction
# from dynamic_obstacle_avoidance.avoidance.rotation import get_desired_radius


def get_positions(x_lim, y_lim, n_resolution, flattened=False):
    dimension = 2
    nx, ny = n_resolution, n_resolution
    x_vals, y_vals = np.meshgrid(np.linspace(x_lim[0], x_lim[1], nx),
                                 np.linspace(y_lim[0], y_lim[1], ny))
    
    positions = np.vstack((x_vals.reshape(1,-1), y_vals.reshape(1,-1)))

    if not flattened:
        positions = positions.reshape(2, n_resolution, n_resolution)

    return positions


class TestOverrotation(unittest.TestCase):
    def test_single_ellipse_radius(self, assert_check=True, visualize=False, save_figure=False):
        """ Cretion & adapation of MultiWall-Surrounding """
        dim = 2
        x_lim = [-10, 10]
        y_lim = [-10, 10]
        n_resolution = 10
    
        # InitialDynamics = LinearSystem(attractor_position=np.array([6, -5]))
        InitialDynamics = LinearSystem(attractor_position=np.array([5.5, -4.5]),
                                       maximum_velocity=0.5)
        
        obstacle_list = MultiBoundaryContainer()
        obstacle_list.append(
            Ellipse(
            center_position=np.array([0, 0]), 
            axes_length=np.array([5, 8]),
            orientation=50./180*pi,
            is_boundary=True,
            )
        )

        obstacle_list.set_convergence_directions(InitialDynamics)

        visuzlize = visualize or save_figure 
        if visualize:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        positions = get_positions(x_lim, y_lim, n_resolution, flattened=True)

        conv_radius = np.zeros((positions.shape[1]))
        gamma = np.zeros((positions.shape[1]))

        initial_velocities = np.zeros(positions.shape)
        rotated_velocities = np.zeros(positions.shape)

        normal_directions = np.zeros(positions.shape)

        for it_obs, obs in zip(range(len(obstacle_list)), obstacle_list):
            for it in range(n_resolution*n_resolution):

                normal_directions[:, it] = obstacle_list[-1].get_normal_direction(
                    positions[:, it], in_global_frame=True)
                gamma[it] = obs.get_gamma(positions[:, it], in_global_frame=True)
                
                conv_radius[it] = get_desired_radius(
                    position=positions[:, it],
                    gamma_value=gamma[it],
                    it_obs=it_obs,
                    obstacle_list=obstacle_list,
                    )

                initial_velocities[:, it] = InitialDynamics.evaluate(positions[:, it])
                rotated_velocities[:, it] = multihull_attraction(
                    positions[:, it], initial_velocities[:, it], obstacle_list)
                # breakpoint()
                if assert_check and it>=1:
                    # Exclude numerical errors...
                    if not np.isclose(conv_radius[it], conv_radius[it-1]):
                        # Since no 'additional' weight; unique function of gamma
                        self.assertEqual(conv_radius[it] > conv_radius[it-1], gamma[it] < gamma[it-1],
                                         "Increasing convergence-raius with decreasing gamma needed.")

            # breakpoint()

            if visualize:
                cs = ax.contourf(positions[0, :].reshape(n_resolution, n_resolution),
                                 positions[1, :].reshape(n_resolution, n_resolution),
                                 conv_radius.reshape(n_resolution, n_resolution), 
                                 np.arange(0.0, np.pi, np.pi/10.0),
                                 cmap='hot_r',
                                 extend='max', alpha=0.6, zorder=2)

                plot_obstacles(ax=ax, obs=obstacle_list, x_range=x_lim, y_range=y_lim,
                               noTicks=True, showLabel=False, draw_wall_reference=True)
                
                ax.quiver(positions[0, :], positions[1, :],
                          rotated_velocities[0, :], rotated_velocities[1, :],
                          color='black', zorder=3)

                ax.quiver(positions[0, :], positions[1, :],
                          normal_directions[0, :], normal_directions[1, :],
                          color='white', zorder=3)

        if visualize:
            cbar_ax = fig.add_axes([0.925, 0.18, 0.02, 0.65])
            cbar = fig.colorbar(cs, cax=cbar_ax, ticks=np.linspace(0, np.pi, 5))
            # cbar.ax.set_yticklabels(["0", "pi/4", "pi/2", "3pi/4", "pi"])
            # cbar = fig.colorbar(cs, cax=cbar_ax, ticks=np.linspace(0, np.pi, 5))
            cbar.ax.set_yticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3 \pi}{4}$"])
            plt.ion()
            plt.show()

            if save_figure:
                figure_name = "single_ellipse_radius_value"
                plt.savefig("figures/" + figure_name + ".png", bbox_inches='tight')


    def test_two_ellipse_radius(self, assert_check=True, visualize=False, save_figure=False):
        """ Cretion & adapation of MultiWall-Surrounding """
        dim = 2
        x_lim = [-10, 10]
        y_lim = [-10, 10]
        n_resolution = 50
    
        # InitialDynamics = LinearSystem(attractor_position=np.array([6, -5]))
        InitialDynamics = LinearSystem(attractor_position=np.array([5.5, -4.5]))
        
        obstacle_list = MultiBoundaryContainer()
        obstacle_list.append(
            Ellipse(
            center_position=np.array([-2, 0]), 
            axes_length=np.array([5, 3]),
            orientation=00./180*pi,
            is_boundary=True,
            )
        )

        obstacle_list.append(
            Ellipse(
            center_position=np.array([2, 0]), 
            axes_length=np.array([3, 5]),
            orientation=0./180*pi,
            is_boundary=True,
            ),
            parent=-1,
        )

        obstacle_list.update_intersection_graph()
        obstacle_list.set_convergence_directions(InitialDynamics)
        
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(obstacle_list), figsize=(14, 8))
        
        positions = get_positions(x_lim, y_lim, n_resolution, flattened=True)
        conv_radius = np.zeros((positions.shape[1]))
        gammas = np.zeros((positions.shape[1]))
        
        for it_obs, obs in zip(range(len(obstacle_list)), obstacle_list):
            for it in range(positions.shape[1]):
                # it_obs = 1
                # obs = obstacle_list[it_obs]
                # positions[:, it] = [-4.6, -5.03]
                
                gammas[it] = obs.get_gamma(positions[:, it], in_global_frame=True)
                
                conv_radius[it] = get_desired_radius(
                    position=positions[:, it],
                    gamma_value=gammas[it],
                    it_obs=it_obs,
                    obstacle_list=obstacle_list,
                    dotprod_weight=1, gamma_weight=1)

            # breakpoint()
            cs = axs[it_obs].contourf(positions[0, :].reshape(n_resolution, n_resolution),
                                      positions[1, :].reshape(n_resolution, n_resolution),
                                      conv_radius.reshape(n_resolution, n_resolution), 
                                      np.arange(0.0, np.pi, np.pi/10.0),
                                      cmap='hot_r',
                                      extend='max', alpha=1.0, zorder=2)
            
            plot_obstacles(ax=axs[it_obs], obs=obstacle_list, x_range=x_lim, y_range=y_lim,
                           noTicks=True, showLabel=False, draw_wall_reference=True,
                           obstacle_color='#000000')

        cbar_ax = fig.add_axes([0.925, 0.18, 0.02, 0.65])
        cbar = fig.colorbar(cs, cax=cbar_ax, ticks=np.linspace(0, np.pi, 5))
        cbar.ax.set_yticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3 \pi}{4}$"])
        plt.ion()
        plt.show()

        if save_figure:
            figure_name = "double_ellipse_radius_value"
            plt.savefig("figures/" + figure_name + ".png", bbox_inches='tight')


    def test_vectorfield(self, assert_check=True, visualize=False, save_figure=False):
        Simulation_vectorFields(
            x_lim, y_lim, n_resolution, obstacle_list,
            saveFigure=False, 
            noTicks=True, showLabel=False,
            draw_vectorField=True,
            dynamical_system=InitialDynamics.evaluate,
            obs_avoidance_func=obstacle_avoidance_rotational,
            automatic_reference_point=False,
            pos_attractor=InitialDynamics.attractor_position,
            # fig_and_ax_handle=(fig, ax),
            show_streamplot=False,
            # show_streamplot=True,
            vector_field_only_outside=False,
        )
        

if __name__ == '__main__':
    # Allow running in ipython (!)
    # unittest.main(argv=['first-arg-is-ignored'], exit=False)
    # unittest.main()
    
    visualize = True
    if visualize:
        Tester = TestOverrotation()
        Tester.test_single_ellipse_radius(visualize=True, assert_check=False, save_figure=True)
        # Tester.test_two_ellipse_radius(visualize=True, save_figure=True)

