#!/USSR/bin/python3
''' Script to show lab environment on computer '''
import warnings
import copy
import datetime


import numpy as np
from numpy import pi

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.image as mpimg

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

        self.evaluation_funcs = [obs_avoidance_interpolation_moving,
                                 obs_avoidance_orthogonal_moving,
                                 obs_avoidance_potential_field,
                                 ]
        self.evals_titles = ["Dynamic", "Orthogonal", "Repulsion"]
        self.n_methods = len(self.evaluation_funcs)

        self.dim = 2


    def on_click(self, event):
        # if self.animation_run:
            # self.anim.stop()
            
        if self.animation_paused:
            self.animation_paused = False
        else:
            self.animation_paused = True

    def run(self, start_position,
            it_max=220, dt_simu=0.12, dt_sleep=0.12,
            x_lim=[-3.5, 3.5], y_lim=[-1.0, 11],
            save_animation=False,
            animation_name=None,
            ):
        """ """
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.dt_simu = dt_simu

        self.trajectories = np.zeros((self.dim, it_max+1, self.n_methods))
        self.fig, self.axs = plt.subplots(1, self.n_methods, figsize=(16, 8))
        cid = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        
        for aa in range(self.n_methods):
            self.trajectories[:, 0, aa] = start_position

        # Setup with QOLO
        self.img_qolo = mpimg.imread(os.path.join("figures", "Qolo_T_CB_top_bumper.png"))

        self.intersection_points = [[] for aa in range(self.n_methods)]
        
        if save_animation:
            if animation_name is None:
                now = datetime.datetime.now()
                animation_name = f"animation_{now:%Y-%m-%d_%H-%M-%S}"
                
            # Set filetype
            file_type = ".mp4"
            animation_name = animation_name + file_type
            
            self.animation_run = True
            
            self.anim = animation.FuncAnimation(
                self.fig,
                self.update_step,
                frames=it_max,
                interval=dt_sleep*1000, # Conversion [s] -> [ms]
                )
            print("Done with the animation.")
            
            self.anim.save(os.path.join("figures", animation_name),
                           metadata={'artist':'Lukas Huber'},
                           # save_count=2,
                           )
            
            # plt.close('all')
            print("Done it all.")
            
        else:
            ii = 0
            while (ii < it_max):
                if self.animation_paused:
                    plt.pause(dt_sleep)
                    if not plt.fignum_exists(self.fig.number):
                        print("Stopped animation on closing of the figure..")
                        break
                    continue
                
                self.update_step(ii, animation_run=False)

                # Check convergence
                if np.allclose(self.trajectories[:, ii, :], self.trajectories[:, ii+1, :]):
                    print(f"All trajectories converged at it={ii}.")
                    break

                plt.pause(dt_sleep)

                if not plt.fignum_exists(self.fig.number):
                    print("Stopped animation on closing of the figure..")
                    break
                ii += 1

    def update_step(self, ii, animation_run=True, print_modulo=10) -> list:
        """ One step of the simulation."""
        if print_modulo:
            if not ii % print_modulo:
                print(f"it={ii}")

        for obs in self.environment:
            obs.update_position(t=ii*self.dt_simu, dt=self.dt_simu)

        for aa, func in enumerate(self.evaluation_funcs):
            # Reset references
            if any(
                obs.get_gamma(self.trajectories[:, ii, aa], in_global_frame=True) < 1
                for obs in self.environment
                ):
                # Skip loop
                self.trajectories[:, ii+1, aa] = self.trajectories[:, ii, aa]

                self.intersection_points[aa].append(self.trajectories[:, ii, aa])

            else:
                # No collision happend
                initial_vel = linear_ds_max_vel(
                    position=self.trajectories[:, ii, aa],
                    attractor=self.attractor_position,
                    vel_max=1.0,
                    )

                mod_vel = func(self.trajectories[:, ii, aa],
                               initial_vel,
                               self.environment)

                self.trajectories[:, ii+1, aa] = (
                    self.trajectories[:, ii, aa] + self.dt_simu*mod_vel)

            self.axs[aa].clear()
            self.axs[aa].plot(self.trajectories[0, :ii+1, aa],
                         self.trajectories[1, :ii+1, aa],
                         '--', color="black")

            self.axs[aa].plot(self.trajectories[0, ii+1, aa],
                         self.trajectories[1, ii+1, aa],
                         'o', color="black")

            # plt.sca(self.axs[aa])
            Simulation_vectorFields(
                self.x_lim, self.y_lim, obs=self.environment,
                xAttractor=self.attractor_position,
                saveFigure=False,
                obs_avoidance_func=func,
                show_streamplot=False,
                fig_and_ax_handle=(self.fig, self.axs[aa]),
                draw_vectorField=False,
                showLabel=False,
                point_grid=20,
                )

            # self.axs[aa].grid()
            self.axs[aa].set_aspect("equal", adjustable="box")
            self.axs[aa].set_xlim(self.x_lim)
            self.axs[aa].set_ylim(self.y_lim)
            self.axs[aa].set_title(self.evals_titles[aa])

            if len(self.intersection_points[aa]):
                temp_points = np.array(self.intersection_points[aa]).T
                
                self.axs[aa].plot(temp_points[0, :], temp_points[1, :], 'ro')

            self.plot_qolo(self.axs[aa], self.trajectories[:, ii+1, aa], mod_vel)

    def plot_qolo(self, ax, position, velocity):
        length_x = 1.0
        length_y = (1.0)*self.img_qolo.shape[0]/self.img_qolo  .shape[1] * length_x
        
        rot = np.arctan2(velocity[1], velocity[0])
        
        arr_img_rotated = ndimage.rotate(self.img_qolo, rot*180.0/pi, cval=1.0, order=1)
        
        lenght_x_rotated = (np.abs(np.cos(rot))*length_x + 
                            np.abs(np.sin(rot))*length_y )
        
        lenght_y_rotated = (np.abs(np.sin(rot))*length_x + 
                            np.abs(np.cos(rot))*length_y )

        ax.imshow(arr_img_rotated,
                  extent=[position[0]-lenght_x_rotated/2.0,
                          position[0]+lenght_x_rotated/2.0,
                          position[1]-lenght_y_rotated/2.0,
                          position[1]+lenght_y_rotated/2.0])


def main_dynamic(robot_margin=0.3, human_radius=0.35,
                 save_animation=False, it_max=180):
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
    
    my_simulation.run(start_position=np.array([0, 0]),
                      save_animation=save_animation, it_max=it_max)


def main_static(robot_margin=0.3, human_radius=0.35,
                save_animation=False, it_max=180):
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
    
    my_simulation.run(start_position=np.array([0, 0]),
                      save_animation=save_animation, it_max=it_max)
    

if (__name__) == "__main__":
    plt.ion()
    # Diable warnings
    import warnings
    warnings.filterwarnings("ignore")

    main_static(save_animation=True, it_max=140)
    # main_dynamic(save_animation=False, it_max=120)
