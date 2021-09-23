#!/USSR/bin/python3.9
""" Test overrotation for ellipses. """
# Author: Lukas Huber
# Created: 2021-05-18
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import unittest
from math import pi

import numpy as np
from numpy import linalg as LA

from vartools.dynamical_systems import LinearSystem
from vartools.directional_space import UnitDirection, DirectionBase

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.containers import MultiBoundaryContainer
from dynamic_obstacle_avoidance.visualization.gamma_field_visualization import gamma_field_visualization
from dynamic_obstacle_avoidance.visualization import Simulation_vectorFields, plot_obstacles

from dynamic_obstacle_avoidance.avoidance.multihull_convergence import get_desired_radius, multihull_attraction

from dynamic_obstacle_avoidance.avoidance.rotation import get_intersection_with_circle
from dynamic_obstacle_avoidance.avoidance.rotation import directional_convergence_summing

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
    def test_intersection_with_circle_2d(self):
        # 2D test right intersection
        start = np.array([-2, 0])
        center = np.array([0, 0])
        radius = 1
        
        points = get_intersection_with_circle(
            start_position=start,
            direction=center-start,
            radius=1,
            only_positive=False,
            )
        self.assertTrue(np.allclose([-1, 0], points[:, 0]))
        self.assertTrue(np.allclose([ 1, 0], points[:, 1]))

        # Only positive direction
        points = get_intersection_with_circle(
            start_position=start,
            direction=center-start,
            radius=1,
            only_positive=True,
            )
        self.assertTrue(np.allclose([1, 0], points))

        # Inside point (positive direction) ['randomly'-chosen numbers]
        start = np.array([0.2, -0.1])
        direction = np.array([2, 3])

        points = get_intersection_with_circle(
            start_position=start,
            direction=direction,
            radius=2.3,
            )
        self.assertEqual(points.shape, start.shape)
        ratio = (points-start)/direction
        
        self.assertEqual(ratio[0], ratio[1])
        self.assertTrue(ratio[0] > 0)

    def test_directional_deviation_weight(self, visualize=False, save_figure=False):
        from dynamic_obstacle_avoidance.avoidance.rotation import _get_directional_deviation_weight
        
        w_conv = _get_directional_deviation_weight(weight=1, weight_deviation=1)
        self.assertTrue(w_conv==1)

        w_conv = _get_directional_deviation_weight(weight=0, weight_deviation=1)
        self.assertTrue(w_conv==1)
        
        w_conv = _get_directional_deviation_weight(weight=1, weight_deviation=0)
        self.assertTrue(w_conv==0)

        w_conv = _get_directional_deviation_weight(weight=0, weight_deviation=0)
        self.assertTrue(w_conv==0)
        
        w_low = _get_directional_deviation_weight(weight=0.3, weight_deviation=0.3)
        w_high = _get_directional_deviation_weight(weight=0.3, weight_deviation=0.7)
        self.assertTrue(0 < w_low < w_high < 1)

        w_low = _get_directional_deviation_weight(weight=0.3, weight_deviation=0.3)
        w_high = _get_directional_deviation_weight(weight=0.7, weight_deviation=0.3)
        self.assertTrue(0 < w_low < w_high < 1)
        

    def test_directional_convergence_summing(self):
        """ Test directional convergence summing in 3D."""
        # Weighting der
        dim = 3
        base = DirectionBase(matrix=np.eye(dim))
    
        convergence_vector = np.array([1, 0.1, 0])
        convergence_vector = convergence_vector / LA.norm(convergence_vector)
        reference_vector = [1, 0, 0]
        weight = 0.0
        # nonlinear_velocity = convergence_vector
        
        converged_vector = directional_convergence_summing(
            convergence_vector=convergence_vector,
            reference_vector=reference_vector,
            base=base, weight=weight,
            convergence_radius=pi/2)

        # Zero movement of the converging vector
        self.assertTrue(np.allclose(convergence_vector, converged_vector.as_vector()))

        weight = 1
        converged_vector = directional_convergence_summing(
            convergence_vector=convergence_vector,
            reference_vector=reference_vector,
            base=base, weight=weight,
            convergence_radius=pi/2)

        # Project on pi/2 (tangent) space
        self.assertTrue(np.isclose(np.dot(base[0], converged_vector.as_vector()), 0))
        # Project away from 'reference-vector'
        self.assertTrue(np.dot(reference_vector, convergence_vector) >
                        np.dot(reference_vector, converged_vector.as_vector()))

        # # Increasingly going closer
        # convergence_velocity = np.array([8., 4.])
        # initial_velocity = np.array([ 0.99173346, -0.12831505])
        # reference_dir = np.array([-0.91914503, -0.3939193 ])
        # inv_gamma_weight = 3.264726439135561
        # normal_orthogonal_matrix = np.array([[-9.99999977e-01,  2.13942402e-04],
        #                                      [-2.13942402e-04, -9.99999977e-01]])
        # convergence_radius = 2.54063962480226

        # rotated_velocities = directional_convergence_summing(
        #     convergence_vector=convergence_velocity,
        #     reference_vector=reference_dir,
        #     weight=inv_gamma_weight,
        #     # weight=1, # 
        #     nonlinear_velocity=initial_velocity,
        #     base=DirectionBase(matrix=normal_orthogonal_matrix),
        #     convergence_radius=convergence_radius)
        # breakpoint()
        
        
    def plottest_single_ellipse_radius(self, assert_check=True, visualize=False, save_figure=False):
        """ Cretion & adapation of MultiWall-Surrounding """
        dim = 2
        
        x_lim = [-10, 10]
        y_lim = [-10, 10]
        
        n_resolution = 10
    
        # InitialDynamics = LinearSystem(attractor_position=np.array([6, -5]))
        InitialDynamics = LinearSystem(attractor_position=np.array([5.5, -4.5]),
                                       A_matrix=np.array([[-1, 0.9], [-0.9, -1]]),
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
        ConverginSystem = LinearSystem(attractor_position=np.array([5.5, -4.5]),
                                       maximum_velocity=0.5)

        obstacle_list.set_convergence_directions(ConverginSystem)

        visualize = visualize or save_figure 
        if visualize:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(14, 8))
            ax = axs[1]
        
        positions = get_positions(x_lim, y_lim, n_resolution, flattened=True)

        conv_radius = np.zeros((positions.shape[1]))
        gamma = np.zeros((positions.shape[1]))

        initial_velocities = np.zeros(positions.shape)
        rotated_velocities = np.zeros(positions.shape)

        normal_directions = np.zeros(positions.shape)

        for it_obs, obs in zip(range(len(obstacle_list)), obstacle_list):
            for it in range(n_resolution*n_resolution):
                # positions[:, it] = [-3.38, -7.66]
                normal_directions[:, it] = -obstacle_list[-1].get_normal_direction(
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
                
                if assert_check and it>=1:
                    # Exclude numerical errors...
                    if not np.isclose(conv_radius[it], conv_radius[it-1]):
                        # Since no 'additional' weight; unique function of gamma
                        self.assertEqual(
                            conv_radius[it] > conv_radius[it-1], gamma[it] < gamma[it-1],
                            "Increasing convergence-raius with decreasing gamma needed.")

            if visualize:
                cs = ax.contourf(positions[0, :].reshape(n_resolution, n_resolution),
                                 positions[1, :].reshape(n_resolution, n_resolution),
                                 conv_radius.reshape(n_resolution, n_resolution), 
                                 np.arange(0.0, np.pi, np.pi/10.0),
                                 cmap='hot_r',
                                 extend='max', alpha=0.6, zorder=2)

                plot_obstacles(ax=ax, obs=obstacle_list, x_range=x_lim, y_range=y_lim,
                               noTicks=True, showLabel=False, draw_wall_reference=True)

                ax.plot(InitialDynamics.attractor_position[0],
                        InitialDynamics.attractor_position[1], 'k*',
                        linewidth=18.0, markersize=18, zorder=5)
                
                ax.quiver(positions[0, :], positions[1, :],
                          rotated_velocities[0, :], rotated_velocities[1, :],
                          color='black', zorder=3)

                ax.quiver(positions[0, :], positions[1, :],
                          normal_directions[0, :], normal_directions[1, :],
                          color='white', zorder=3)

                # Default 
                plot_obstacles(ax=axs[0], obs=obstacle_list, x_range=x_lim, y_range=y_lim,
                               noTicks=True, showLabel=False, draw_wall_reference=True,
                               alpha_obstacle=0.3,
                               # obstacle_color='white'
                               )
                
                axs[0].quiver(positions[0, :], positions[1, :],
                              initial_velocities[0, :], initial_velocities[1, :],
                              color='black', zorder=3)
                
                axs[0].plot(InitialDynamics.attractor_position[0],
                            InitialDynamics.attractor_position[1], 'k*',
                            linewidth=18.0, markersize=18, zorder=5)
                
        if visualize:
            cbar_ax = fig.add_axes([0.925, 0.18, 0.02, 0.65])
            cbar = fig.colorbar(cs, cax=cbar_ax, ticks=np.linspace(0, np.pi, 5))
            # cbar.ax.set_yticklabels(["0", "pi/4", "pi/2", "3pi/4", "pi"])
            # cbar = fig.colorbar(cs, cax=cbar_ax, ticks=np.linspace(0, np.pi, 5))
            cbar.ax.set_yticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3 \pi}{4}$"])
            plt.ion()
            plt.show()

            if save_figure:
                figure_name = "single_ellipse_radius_value_with_normal"
                plt.savefig("figures/" + figure_name + ".png", bbox_inches='tight')

    def plottest_two_ellipse_radius(self, assert_check=True, visualize=False, save_figure=False):
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


    def plottest_vectorfield(self, assert_check=True, visualize=False, save_figure=False):
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
        

if (__name__) == '__main__':
    # Allow running in ipython (!)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    # unittest.main()
    
    # visualize = False
    # visualize = True
    visualize = False
    if visualize:
        Tester = TestOverrotation()
        # Tester.test_intersection_with_circle_2d()
        # Tester.test_interesection_with_circle_specific()

        # Tester.test_directional_deviation_weight(visualize=False)
        # Tester.test_angle_space_distance()
        
        # Tester.test_nonlinear_inverted_weight(visualize=True)
        # Tester.test_directional_convergence_summing()
        
        # Tester.test_single_ellipse_radius(visualize=True, assert_check=False, save_figure=True)
        # Tester.test_directional_convergence_summing()
        
        # Tester.test_two_ellipse_radius(visualize=True, save_figure=False)
        Tester.test_directional_convergence_summing()

        print("All selected tests executed with success.")
