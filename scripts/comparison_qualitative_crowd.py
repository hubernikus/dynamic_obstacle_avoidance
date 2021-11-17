#!/USSR/bin/python3
''' Script to show lab environment on computer '''
import warnings
import copy

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

# Custom libraries
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import *  #
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import Ellipse
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import CircularObstacle
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import *
from dynamic_obstacle_avoidance.obstacle_avoidance.comparison_algorithms import obs_avoidance_potential_field, obs_avoidance_orthogonal_moving
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import obs_avoidance_interpolation_moving, obs_check_collision_2d
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import GradientContainer
from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import linear_ds_max_vel
from dynamic_obstacle_avoidance.obstacle_avoidance.metric_evaluation import MetricEvaluator

__author__ = "LukasHuber"
__date__ = "2020-01-15"
__email__ = "lukas.huber@epfl.ch"


plt.close('all')
plt.ion()


class DynamicSimulation():
    def __init__(self, environment, attractor_position):
        self.environment = environment
        self.attractor_position = attractor_position
        
        # Simulation parameter
        self.animation_paused = False

    def on_click(self, event):
        if self.animation_paused:
            self.animation_paused = False
        else:
            self.animation_paused = True

    def run(self, start_position,
            it_max=200, dt_simu=0.08, dt_sleep=0.05,
            x_lim=[-3.5, 3.5], y_lim=[-1.0, 11],
            create_video=True,
            ):
        """ """
        evaluation_funcs = [obs_avoidance_interpolation_moving,
                            obs_avoidance_orthogonal_moving,
                            obs_avoidance_potential_field,
                            ]
        evals_titles = ["Dynamic", "Orthogonal", "Repulsion"]
        n_methods = len(evaluation_funcs)
        
        # Two dimensional case
        dim = 2

        trajectories = np.zeros((dim, it_max+1, n_methods))
        fig, axs = plt.subplots(1, n_methods, figsize=(16, 8), num=n_methods)
        cid = fig.canvas.mpl_connect("button_press_event", self.on_click)
        
        for aa in range(n_methods):
            trajectories[:, 0, aa] = start_position

        obj_list = []

        im_list = []
        ii = 0
        while (ii < it_max):
            if self.animation_paused:
                plt.pause(dt_sleep)
                if not plt.fignum_exists(fig.number):
                    print("Stopped animation on closing of the figure..")
                    break
                continue

            for obs in self.environment:
                obs.update_position(t=ii*dt_simu, dt=dt_simu)

            for aa, func in enumerate(evaluation_funcs):
                # reset refs
                # for obs in self.environment:
                    # obs.set_refence_point([0, 0])
                if any(
                    obs.get_gamma(trajectories[:, ii, aa], in_global_frame=True) < 1
                    for obs in self.environment
                    ):
                    # Skip loop
                    trajectories[:, ii+1, aa] = trajectories[:, ii, aa]
                    # print("Skip loop")
                
                else:
                    # No colliision happend
                    initial_vel = linear_ds_max_vel(
                        position=trajectories[:, ii, aa],
                        attractor=self.attractor_position,
                        vel_max=1.0,
                        )

                    mod_vel = func(trajectories[:, ii, aa],
                                   initial_vel,
                                   self.environment)
                
                    trajectories[:, ii+1, aa] = (
                        trajectories[:, ii, aa] + dt_simu*mod_vel)

                axs[aa].clear()
                axs[aa].plot(trajectories[0, :ii+1, aa],
                             trajectories[1, :ii+1, aa],
                             '--', color="black")
                
                axs[aa].plot(trajectories[0, ii+1, aa],
                             trajectories[1, ii+1, aa],
                             'o', color="black")

                plt.sca(axs[aa])
                Simulation_vectorFields(
                    x_lim, y_lim, obs=self.environment,
                    xAttractor=self.attractor_position,
                    saveFigure=False,
                    obs_avoidance_func=func,
                    show_streamplot=False,
                    fig_and_ax_handle=(fig, axs[aa]),
                    draw_vectorField=False,
                    showLabel=False,
                    point_grid=20,
                    )
                
                axs[aa].grid()
                axs[aa].set_aspect("equal", adjustable="box")
                axs[aa].set_xlim(x_lim)
                axs[aa].set_ylim(y_lim)
                axs[aa].set_title(evals_titles[aa])

            # obj_list.append(axs[aa].get_children)
            # Check convergence
            if np.allclose(trajectories[:, ii, :], trajectories[:, ii+1, :]):
                print("All trajectories converged.")
                break

            plt.pause(dt_sleep)
            
            if not plt.fignum_exists(fig.number):
                print("Stopped animation on closing of the figure..")
                break

            ii += 1

def main_dynamic(robot_margin=0.3, human_radius=0.35):
    environment = GradientContainer()
    environment.append(CircularObstacle(
        center_position=np.array([0.5, 8]),
        linear_velocity=np.array([0.1, -0.8]),
        margin_absolut=robot_margin,
        radius=human_radius
    ))

    environment.append(CircularObstacle(
        center_position=np.array([-0.4, 8]),
        linear_velocity=np.array([0.1, -0.77]),
        margin_absolut=robot_margin,
        radius=human_radius
    ))

    environment.append(CircularObstacle(
        center_position=np.array([2, 4]),
        linear_velocity=np.array([-.01, -0.5]),
        margin_absolut=robot_margin,
        radius=human_radius
    ))

    environment.append(CircularObstacle(
        center_position=np.array([-3, 9]),
        linear_velocity=np.array([.01, -0.45]),
        margin_absolut=robot_margin,
        radius=human_radius
    ))

    environment.append(CircularObstacle(
        center_position=np.array([2, 12]),
        linear_velocity=np.array([-.2, -0.8]),
        margin_absolut=robot_margin,
        radius=human_radius
    ))

    environment.append(CircularObstacle(
        center_position=np.array([-1, 14]),
        linear_velocity=np.array([.01, -0.6]),
        margin_absolut=robot_margin,
        radius=human_radius
    ))

    my_simulation = DynamicSimulation(
        environment,
        attractor_position=np.array([0, 10]))
    
    my_simulation.run(start_position=np.array([0, 0]))


def main_static(robot_margin=0.3, human_radius=0.35):
    environment = GradientContainer()
    environment.append(CircularObstacle(
        center_position=np.array([-1.2, 5]),
        linear_velocity=np.array([0, 0]),
        margin_absolut=robot_margin,
        radius=human_radius
    ))

    environment.append(CircularObstacle(
        center_position=np.array([0.4, 5]),
        linear_velocity=np.array([0, 0]),
        margin_absolut=robot_margin,
        radius=human_radius
    ))

    environment.append(CircularObstacle(
        center_position=np.array([-0.4, 8]),
        linear_velocity=np.array([0, 0]),
        margin_absolut=robot_margin,
        radius=human_radius
    ))

    environment.append(CircularObstacle(
        center_position=np.array([0.6, 8]),
        linear_velocity=np.array([0, 0]),
        margin_absolut=robot_margin,
        radius=human_radius
    ))

    environment.append(CircularObstacle(
        center_position=np.array([0.9, 7]),
        linear_velocity=np.array([0, 0]),
        margin_absolut=robot_margin,
        radius=human_radius
    ))

    environment.append(CircularObstacle(
        center_position=np.array([-2, 3]),
        linear_velocity=np.array([0, 0]),
        margin_absolut=robot_margin,
        radius=human_radius
    ))

    environment.append(CircularObstacle(
        center_position=np.array([1, 2]),
        linear_velocity=np.array([0, 0]),
        margin_absolut=robot_margin,
        radius=human_radius
    ))

    my_simulation = DynamicSimulation(
        environment,
        attractor_position=np.array([0, 10]))
    
    my_simulation.run(start_position=np.array([0, 0]))
    

if (__name__) == "__main__":
    plt.ion()
    # main_static()
    main_dynamic()
