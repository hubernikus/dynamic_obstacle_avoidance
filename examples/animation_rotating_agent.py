# Author: Federico Conzelmann
# Email: federico.conzelmann@epfl.ch
# Created: 2021-11-01

from math import pi
from math import atan2
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.obstacles import Polygon, Cuboid, Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.avoidance import DynamicCrowdAvoider
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from vartools.dynamical_systems import LinearSystem


def calculate_delta(pos_list):
    return np.array([pos_list[1, 0] - pos_list[0, 0], pos_list[1, 1] - pos_list[0, 1]]) / 2.0


class DynamicalSystemAnimation:
    def __init__(self):
        self.animation_paused = False

    def on_click(self, event):
        if self.animation_paused:
            self.animation_paused = False
        else:
            self.animation_paused = True

    def run(
            self, initial_dynamics, obstacle_environment,
            obs_w_multi_agent,
            start_position=None,
            x_lim=None, y_lim=None,
            it_max=1000, dt_step=0.03, dt_sleep=0.1
    ):

        num_obs = len(obstacle_environment)
        num_agent = len(start_position)
        dim = 2

        if y_lim is None:
            y_lim = [-0.5, 2.5]
        if x_lim is None:
            x_lim = [-1.5, 2]
        if start_position is None:
            start_position = np.zeros((num_obs, dim))

        dynamic_avoider = DynamicCrowdAvoider(initial_dynamics=initial_dynamics, environment=obstacle_environment)
        position_list = np.zeros((num_agent, dim, it_max))
        relative_agent_pos = np.zeros((num_agent, dim))
        relative_agent_pos[0, :] = obstacle_environment[0].center_position - start_position[0, :]
        relative_agent_pos[1, :] = obstacle_environment[0].center_position - start_position[1, :]
        position_list[:, :, 0] = start_position

        fig, ax = plt.subplots(figsize=(10, 8))
        cid = fig.canvas.mpl_connect('button_press_event', self.on_click)

        ii = 0
        while ii < it_max:
            if self.animation_paused:
                plt.pause(dt_sleep)
                if not plt.fignum_exists(fig.number):
                    print("Stopped animation on closing of the figure..")
                    break
                continue

            ii += 1
            if ii > it_max:
                break

            # Here come the main calculation part
            for obs in range(num_obs):
                num_agents_in_obs = len(obs_w_multi_agent[obs])
                if num_agents_in_obs > 1:
                    weights = 1 / len(obs_w_multi_agent)
                    velocity = np.ndarray((num_agents_in_obs, dim))
                    for agent in obs_w_multi_agent[obs]:
                        temp_env = obstacle_environment[0:obs] + obstacle_environment[obs + 1 :]
                        velocity[agent, :] = dynamic_avoider.evaluate_for_crowd_agent(position_list[agent, :, ii - 1], agent, temp_env, True)
                        # position_list[agent, :, ii] = velocity[agent, :] * dt_step + position_list[agent, :, ii - 1]
                        # obstacle_environment[obs].center_position +=
                        # print(calculate_delta(position_list[obs_w_multi_agent[obs], :, ii]))

                    obs_vel = weights * velocity.sum(axis=0)
                    # now what ?
                    angular_vel = np.zeros(num_agents_in_obs)
                    for agent in obs_w_multi_agent[obs]:
                        angular_vel[agent] = weights * np.cross(
                            (obstacle_environment[obs].center_position - position_list[agent, :, ii - 1]),
                            (velocity[agent, :] - obs_vel))

                    angular_vel_obs = angular_vel.sum()
                    obstacle_environment[obs].linear_velocity = obs_vel
                    obstacle_environment[obs].angular_velocity = angular_vel_obs
                    obstacle_environment[obs].do_velocity_step(dt_step)
                    for agent in obs_w_multi_agent[obs]:
                        position_list[agent, :, ii] = obstacle_environment[obs].transform_relative2global(relative_agent_pos[agent, :])
                else:
                    raise Exception("Not implemented")

                # obstacle_environment[obs].center_position = position_list[obs_w_multi_agent[obs][0], :, ii] + calculate_delta(position_list[obs_w_multi_agent[obs], :, ii])

            # angle = atan2(position_list[0, 1, ii] - position_list[1, 1, ii], position_list[0, 0, ii] - position_list[1, 0, ii])
            # obstacle_environment[0].orientation = angle

            # Clear right before drawing again
            ax.clear()

            # Drawing and adjusting of the axis
            for agent in range(num_agent):
                plt.plot(position_list[agent, 0, :ii], position_list[agent, 1, :ii], ':',
                         color='#135e08')
                plt.plot(position_list[agent, 0, ii], position_list[agent, 1, ii],
                         'o', color='#135e08', markersize=12, )
                plt.arrow(position_list[agent, 0, ii], position_list[agent, 1, ii], velocity[agent, 0], velocity[agent, 1])

            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

            plot_obstacles(ax, obstacle_environment, x_lim, y_lim, showLabel=False)

            for agent in range(num_agent):
                ax.plot(initial_dynamics[agent].attractor_position[0],
                        initial_dynamics[agent].attractor_position[1], 'k*', markersize=8, )
            ax.grid()

            ax.set_aspect('equal', adjustable='box')
            # breakpoiont()

            # Check convergence
            if np.sum(np.abs(velocity)) < 1e-2:
                print(f"Converged at it={ii}")
                break

            plt.pause(dt_sleep)
            if not plt.fignum_exists(fig.number):
                print("Stopped animation on closing of the figure..")
                break


def multiple_robots():
    obstacle_pos = np.array([-2.0, 0.0])
    agent_pos = np.array([[-2.25, 0.0], [-1.75, 0.0]])
    attractor_pos = np.array([[0.0, 0.0], [0.0, -0.5]])
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(Cuboid(
        axes_length=[1.5, 0.6],
        center_position=obstacle_pos,
        margin_absolut=0,
        orientation=0,
        tail_effect=False,
        repulsion_coeff=1.4,
    ))
    initial_dynamics = [LinearSystem(
        attractor_position=attractor_pos[0],
        maximum_velocity=1, distance_decrease=0.3
    ), LinearSystem(
        attractor_position=attractor_pos[1],
        maximum_velocity=1, distance_decrease=0.3
    )]

    obs_multi_agent = {0: [0, 1]}

    DynamicalSystemAnimation().run(
        initial_dynamics,
        obstacle_environment,
        obs_multi_agent,
        agent_pos,
        x_lim=[-3, 1],
        y_lim=[-2, 2],
        dt_step=0.05,
    )


if __name__ == "__main__":
    plt.close('all')
    plt.ion()

    multiple_robots()
