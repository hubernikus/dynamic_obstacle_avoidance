#!/USSR/bin/python3
""" Script to show lab environment on computer """

# Author: Lukas Huber
# Date: 2020-01-15
# Email: lukas.huber@epfl.ch

import warnings
import copy

import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

from vartools.dynamical_systems import LinearSystem


from dynamic_obstacle_avoidance.obstacles import Ellipse, Polygon, Cuboid
from dynamic_obstacle_avoidance.containers import ShapelyContainer
from dynamic_obstacle_avoidance.metric_evaluation import MetricEvaluator
from dynamic_obstacle_avoidance.utils import obs_check_collision_2d
from dynamic_obstacle_avoidance.avoidance import (
    obs_avoidance_potential_field,
    obs_avoidance_orthogonal_moving,
    obs_avoidance_interpolation_moving,
)
from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
)

plt.close("all")
plt.ion()


class DynamicEllipse(Ellipse):
    # Movement is in the brownian motion style
    max_velocity = 0.8
    variance_velocity = 0.1

    max_angular_velocity = 0.2
    variance_angular_velocity = 0.2

    # Random creation
    axis_minimum = 0.3
    axis_range = [0.5, 2.0]
    expansion_variance = 0.1
    max_expansion_rate = 0.1

    save_trajectory = True

    def __init__(self, x_range, y_range, *args, **kwargs):
        self.dim = 2
        self.x_range = x_range
        self.y_range = y_range

        freq_oscilation = None

        position = np.random.rand(self.dim)
        position[0] = position[0] * (x_range[1] - x_range[0]) + x_range[0]
        position[1] = position[1] * (y_range[1] - y_range[0]) + y_range[0]

        orientation = 0
        axes_length = np.random.uniform(size=(self.dim))
        axes_length = (
            axes_length * (self.axis_range[1] - self.axis_range[0]) + self.axis_range[0]
        )

        self.expansion_vel = (
            np.random.uniform(size=(self.dim)) * self.expansion_variance
        )

        self.is_dynamic = 1

        if self.is_dynamic:
            vel_dir = np.random.uniform(low=0, high=2 * np.pi)
            linear_velocity = np.array(
                [np.cos(vel_dir), np.sin(vel_dir)]
            ) * np.random.uniform(low=0, high=self.max_velocity)

            angular_velocity = np.random.uniform(
                low=-self.max_angular_velocity, high=-self.max_angular_velocity
            )

        else:
            linear_velocity = np.zeros(self.dim)
            angular_velocity = 0

        # print('Is dynamic', self.is_dynamic)

        super().__init__(
            axes_length=axes_length,
            center_position=position,
            orientation=orientation,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            is_dynamic=self.is_dynamic,
            *args,
            **kwargs,
        )

        if self.save_trajectory:
            self.position_list = np.zeros((self.dim, 0))

        # Save line-handle when plotting
        self.line_handle = None

    def update_step(self, time_step=0.1, acceleration_rand=0.1, expansion_acc_rand=0.1):
        """Update velocity & expansion."""

        # Random walk for velocity
        # print('linear_velocity', self.linear_velocity)
        self.linear_velocity = (
            self.linear_velocity + np.random.randn(self.dim) * self.variance_velocity
        )
        vel_mag = np.linalg.norm(self.linear_velocity)
        if vel_mag > self.max_velocity:
            self.linear_velocity = self.linear_velocity / vel_mag * self.max_velocity

        self.angular_velocity = (
            self.angular_velocity + np.random.randn(1) * self.variance_angular_velocity
        )
        if abs(self.angular_velocity) > self.max_angular_velocity:
            self.angular_velocity = np.copysign(
                self.max_angular_velocity, self.angular_velocity
            )

        self.position = self.position + time_step * self.linear_velocity
        self.orientation = self.orientation + time_step * self.angular_velocity

        # Random change of axes
        self.axes_length = self.axes_length

        for ii in range(self.dim):
            delta_vel_range = [
                self.axis_range[0] - self.axes_length[ii],
                self.axis_range[1] - self.axes_length[ii],
            ]

            self.expansion_vel[ii] = self.expansion_vel[
                ii
            ] + time_step * np.random.uniform(
                low=delta_vel_range[0], high=delta_vel_range[1]
            )

            if self.expansion_vel[ii] > self.max_expansion_rate:
                self.expansion_vel[ii] = self.max_expansion_rate

            self.axes_length[ii] = (
                self.axes_length[ii] + time_step * self.expansion_vel[ii]
            )

            if self.axes_length[ii] < self.axis_minimum:
                self.axes_length[ii] = self.axis_minimum
                self.expansion_vel[ii] = 0

        if self.save_trajectory:
            self.position_list = np.vstack((self.position_list.T, self.position)).T

        self.check_boundary_collision()
        # Draw obstacles
        self.draw_obstacle()

    def check_boundary_collision(self, boundary_obstacle=None):
        # Check limits in infinte-universe manner

        if self.x_range[0] > self.position[0]:
            self.position[0] = self.x_range[1] - (self.position[0] - self.x_range[0])
        elif self.x_range[1] < self.position[0]:
            self.position[0] = self.x_range[0] + (self.x_range[1] - self.position[0])

        if self.y_range[0] > self.position[1]:
            self.position[1] = self.y_range[1] - (self.position[1] - self.y_range[0])
        elif self.y_range[1] < self.position[1]:
            self.position[1] = self.y_range[0] + (self.y_range[1] - self.position[1])

    def reset_position_list(self):
        if self.save_trajectory:
            self.position_list = np.zeros((self.dim, 0))


class ObstacleAvoidanceAgent:
    def __init__(
        self,
        start_position,
        avoidance_function=None,
        default_velocity=None,
        dim=2,
        attractor=None,
        name=None,
    ):
        # print('The birth of a new autonomous agent is being witnessed.')
        self.dim = dim

        self.default_velocity = default_velocity
        self.avoidance_function = avoidance_function

        self.position_list = np.reshape(start_position, (self.dim, 1))
        self.position = start_position
        self.velocity = np.zeros(self.dim)

        self.attractor = attractor

        self.has_converged = False
        self.has_collided = False
        self.is_in_local_minma = False

        self.name = name

        self.MetricTracker = MetricEvaluator()

    @property
    def has_stopped(self):
        return self.has_converged or self.has_collided or self.is_in_local_minma

    def update_step(
        self,
        obstacle_list=[],
        delta_time=0.1,
        initial_velocity=None,
        check_convergence=True,
        time=None,
        vel_min_margin=0.01,
    ):

        if self.has_stopped and check_convergence:
            return

        if initial_velocity is None:
            if self.default_velocity is None:
                raise NotImplementedError("Undefined velicty")

            initial_velocity = self.default_velocity(self.position)

        initial_vel_mag = np.linalg.norm(initial_velocity)

        self.velocity = self.avoidance_function(
            self.position, initial_velocity, obstacle_list
        )

        vel_mag = np.linalg.norm(self.velocity)
        if vel_mag > initial_vel_mag:  # Too fast
            self.velocity = self.velocity / vel_mag * initial_vel_mag

        if vel_mag < vel_min_margin:  # Nonzero with margin (Local minima)
            print("Agent velocity is zero. Stopping exectuion.")
            self.is_in_local_minma = True

        self.position = self.position + self.velocity * delta_time

        # Save to list for tracking and storage
        self.position_list = np.vstack((self.position_list.T, self.position)).T

        self.MetricTracker.update_list(
            position=self.position, velocity=self.velocity, time=time
        )

        # import pdb; pdb.set_trace()
        if self.attractor is not None:
            self.check_convergence()

        # print('A big step for the creator.')

    def check_convergence(self, convergence_margin=0.1):
        if np.linalg.norm(self.attractor - self.position) < convergence_margin:
            self.has_converged = True
            print("Agent {} has converged.".format(self.name))

    def check_collision(self, obstacle_list):
        if self.has_stopped:
            # if self.has_collided:
            return

        if not position_is_in_free_space(self.position, obstacle_list):
            self.has_collided = True
            print("Agent {} has collided.".format(self.name))

    def evaluate_metric(self, delta_time=None):
        metrics = self.MetricTracker.evaluate_metrics(delta_time=delta_time)
        metrics["name"] = self.name

        # Metrics
        metrics["has_collided"] = self.has_collided
        metrics["has_converged"] = self.has_converged
        metrics["is_in_local_minima"] = self.is_in_local_minma
        metrics["not_converged"] = not (self.has_stopped)

        return metrics


def position_is_in_free_space(position, obstacle_list):
    """Check for all obstacles if position is intersecting."""
    for obs in obstacle_list:
        gamma = obs.get_gamma(position, in_global_frame=True)
        if gamma < 1:
            return False
    return True


def compare_algorithms_random(
    max_it=1000,
    delta_time=0.01,
    max_num_obstacles=5,
    dim=2,
    visualize_scene=True,
    random_seed=None,
    fig_and_ax_handle=None,
    fig_num=None,
    plot_last_image=False,
    show_legend=True,
):
    """Compare the algorithms with a random environment setup."""

    if random_seed is not None:
        np.random.seed(random_seed)

    x_range = [-1, 11]
    y_range = [-1, 11]

    obs_list = ShapelyContainer()

    obs_xaxis = [x_range[0] + 1, x_range[1] - 1]
    obs_yaxis = [y_range[0] + 1, y_range[1] - 1]
    edge_points = np.array(
        [
            [obs_xaxis[0], obs_xaxis[0], obs_xaxis[1], obs_xaxis[1]],
            [obs_yaxis[1], obs_yaxis[0], obs_yaxis[0], obs_yaxis[1]],
        ]
    )

    obs_list.append(
        Polygon(
            edge_points=edge_points,
            is_boundary=True,
            tail_effect=False,
        )
    )

    # Two obstacles instead of one gives a better performance at the 'star-side'
    center_point = copy.deepcopy(obs_list[-1].center_position)

    if True:
        attractor_position = np.array([7.5, 1.7])

        obs_list.append(
            Cuboid(
                axes_length=[12, 2.5],
                center_position=[11, 5],
                tail_effect=False,
            )
        )

    if False:
        attractor_position = np.array([2.5, 1.3])

        obs_list.append(
            Cuboid(
                axes_length=[10, 2],
                center_position=[11, 7.0],
                tail_effect=False,
            )
        )
        obs_list.append(
            Cuboid(
                axes_length=[9, 2],
                center_position=[-1, 3.5],
                tail_effect=False,
            )
        )

    num_obstacles = 2
    for oo in range(num_obstacles):
        obs_list.append(
            DynamicEllipse(
                x_range=x_range,
                y_range=y_range,
                tail_effect=False,
            )
        )

    if False:
        from dynamic_obstacle_avoidance.visualization.gamma_field_visualization import (
            gamma_field_visualization,
        )

        gamma_field_visualization([-1, 11], [-1, 11], obstacle=obs_list[-1])
        return

    if False:
        pos = np.array([2.5, 9.5])
        vel = attractor_position - pos
        vel = vel / np.linalg.norm(vel)
        # xd = obs_avoidance_orthogonal_moving(pos, vel, obs_list)
        xd = obs_avoidance_potential_field(pos, vel, obs_list)
        import pdb

        pdb.set_trace()

    if False:
        Simulation_vectorFields(
            x_range,
            y_range,
            obs=obs_list,
            pos_attractor=attractor_position,
            saveFigure=False,
            # obs_avoidance_func=obs_avoidance_interpolation_moving,
            # obs_avoidance_func=obs_avoidance_orthogonal_moving,
            obs_avoidance_func=obs_avoidance_potential_field,
            # noTicks=False, automatic_reference_point=True,
            # draw_vectorField=False,
            show_streamplot=True,
            # fig_and_ax_handle=(fig, ax),
            point_grid=100,
            normalize_vectors=False,
        )
        return

    dynamical_system = LinearSystem(attractor_position=attractor_position)

    # Try to find a point in free-space
    start_point_found = False
    for it_count in range(100):
        x_starting = x_range
        y_starting = [7, 10]

        start_position = np.random.uniform(size=(dim))
        start_position[0] = (
            x_starting[0] + (x_starting[1] - x_starting[0]) * start_position[0]
        )
        start_position[1] = (
            y_starting[0] + (y_starting[1] - y_starting[0]) * start_position[1]
        )

        if position_is_in_free_space(start_position, obs_list):
            start_point_found = True
            break

    if not start_point_found:
        warnings.warn("No free position found")
        return

    # Define different obstacle avoidance agents
    agents = []
    agents.append(
        ObstacleAvoidanceAgent(
            start_position=start_position,
            name="Dynamic",
            avoidance_function=obs_avoidance_interpolation_moving,
            attractor=dynamical_system.attractor_position,
        )
    )

    agents.append(
        ObstacleAvoidanceAgent(
            start_position=start_position,
            name="Orthogonal",
            avoidance_function=obs_avoidance_orthogonal_moving,
            attractor=dynamical_system.attractor_position,
        )
    )

    agents.append(
        ObstacleAvoidanceAgent(
            start_position=start_position,
            name="Repulsion",
            avoidance_function=obs_avoidance_potential_field,
            attractor=dynamical_system.attractor_position,
        )
    )

    initial_distance = np.linalg.norm(
        dynamical_system.attractor_position - start_position
    )

    if visualize_scene or plot_last_image:
        if fig_num is None:
            fig_num = 1001

        if fig_and_ax_handle is None:
            fig, ax = plt.subplots(num=fig_num, figsize=(8, 6))
        else:
            fig, ax = fig_and_ax_handle

    # Main loop
    ii = 0
    while ii < max_it:
        # Iterate
        print(f"it={ii}")
        ii += 1
        # for ii in range(max_it):
        for obs in obs_list:
            if obs.is_dynamic:
                obs.update_step()

        obs_list.update_reference_points()

        if visualize_scene:
            ax.cla()

            Simulation_vectorFields(
                x_range,
                y_range,
                obs=obs_list,
                pos_attractor=dynamical_system.attractor_position,
                showLabel=False,
                saveFigure=False,
                obs_avoidance_func=obs_avoidance_interpolation_moving,
                # noTicks=False,
                automatic_reference_point=False,
                draw_vectorField=True,
                show_streamplot=False,
                fig_and_ax_handle=(fig, ax),
                normalize_vectors=False,
                point_grid=10,
            )

            if True:
                plt.plot(
                    obs_list[-1].center_position[0],
                    obs_list[-1].center_position[1],
                    "k+",
                )

            if False:
                # if obs_list[0].save_trajectory:
                # Show all or nothing
                for obs in obs_list:
                    if obs.is_dynamic:
                        plt.plot(
                            obs.position_list[0, :], obs.position_list[1, :], "k--"
                        )
        for agent in agents:
            # initial_velocity = linear_ds_max_vel(
            # position=agent.position, attractor=attractor_position, vel_max=1.0)
            initial_velocity = dynamical_system.evaluate(agent.position)
            agent.update_step(
                obs_list, initial_velocity=initial_velocity, time=delta_time * ii
            )
            agent.check_collision(obs_list)

            if visualize_scene:
                (agent.line_handle,) = plt.plot(
                    agent.position_list[0, :],
                    agent.position_list[1, :],
                    "--",
                    label=agent.name,
                )

        breakpoint()

        if visualize_scene and show_legend:
            plt.legend(loc="center right")

        if visualize_scene and not plt.fignum_exists(fig_num):
            print(f"Simulation ended with closing of figure")
            plt.pause(0.01)
            plt.close("all")
            break

        all_stopped = True
        for agent in agents:
            if not agent.has_stopped:
                all_stopped = False
                break

        if all_stopped or ii >= max_it:
            if not visualize_scene and plot_last_image:
                ii -= 1
                visualize_scene = True
                continue

            if all_stopped:
                print("All agents stopped.")
            else:
                print("Reached maximum number of iterations.")
            break

        if visualize_scene:
            plt.pause(0.01)

    agent_metrics = []
    for agent in agents:
        agent_metrics.append(agent.evaluate_metric(delta_time=delta_time))

    if plot_last_image and not show_legend:
        # Return legend label
        return [agent.line_handle for agent in agents]
    else:
        return agent_metrics, initial_distance


def multiple_random_runs(num_runs=3, visualize=False):
    """What is happening."""
    # First one just to get a 'default-metric'
    num_agent_type = 3

    # 3 algorithms to compare
    list_agent_metrics = []
    outcome_keys = [
        "total",
        "has_converged",
        "has_collided",
        "is_in_local_minima",
        "not_converged",
    ]
    outcome_dict = {}
    for key in outcome_keys:
        outcome_dict[key] = 0

    list_simulation_outcome = [
        copy.deepcopy(outcome_dict) for ii in range(num_agent_type)
    ]

    list_initial_distance = []

    # for ii in range(num_runs):
    ii = 0
    while len(list_initial_distance) == 0 or ii < num_runs:
        ii += 1
        agent_metrics, initial_distance = compare_algorithms_random(
            visualize_scene=visualize,
        )

        # if all([agent['has_converged'] for agent in agent_metrics]):
        if all([agent["has_converged"] for agent in agent_metrics]):
            list_agent_metrics.append(agent_metrics)
            list_initial_distance.append(initial_distance)

        for aa in range(len(agent_metrics)):
            for outcome_key in [
                "has_converged",
                "has_collided",
                "is_in_local_minima",
                "not_converged",
            ]:
                if agent_metrics[aa][outcome_key]:
                    list_simulation_outcome[aa][outcome_key] += 1

            list_simulation_outcome[aa]["total"] += 1

    if not len(list_initial_distance):  # zero length
        warnings.warn("No list detected.")
        return

    list_initial_distance = np.array(list_initial_distance)

    if True:
        return (list_agent_metrics, list_simulation_outcome, list_initial_distance)


def evaluation_metrics(metrics):
    # breakpoint()
    list_agent_metrics, list_simulation_outcome, list_initial_distance = metrics
    # Set up dictionary
    eval_dict = {}

    num_success_metrics = len(list_agent_metrics)

    # Create lists for each key
    for aa, agent_dict in zip(
        np.arange(len(list_agent_metrics[0])), list_agent_metrics[0]
    ):
        # import pdb; pdb.set_trace()
        for key in agent_dict.keys():

            if key == "name":
                # Treat name key differently
                key_str = "name"
                if key_str not in eval_dict:
                    eval_dict[key_str] = []
                eval_dict[key_str].append(agent_dict[key_str])
                continue

            if isinstance(agent_dict[key], dict):
                for subkey in agent_dict[key]:
                    key_str = key + "_" + subkey

                    data = [
                        list_agent_metrics[it][aa][key][subkey]
                        for it in range(len(list_agent_metrics))
                    ]

                    if key_str not in eval_dict:
                        eval_dict[key_str] = []

                    eval_dict[key_str].append({})
                    eval_dict[key_str][-1]["mean"] = np.mean(data)
                    eval_dict[key_str][-1]["std"] = np.std(data)

            else:
                key_str = key
                # data = [agent_metrics[aa][key] for agent_metrics in list_agent_metrics]
                data = [
                    list_agent_metrics[it][aa][key]
                    for it in range(len(list_agent_metrics))
                ]

                if key_str == ["distance"] or key_str == ["duration"]:
                    data = np.array(data) / list_initial_distance

                if not key_str in eval_dict:
                    eval_dict[key_str] = []

                eval_dict[key_str].append({})
                eval_dict[key_str][-1]["mean"] = np.mean(data)
                eval_dict[key_str][-1]["std"] = np.std(data)

    separator_str = " & "
    end_of_line = " \\\\ \n   "
    separator_sum_var = " $\pm$ "
    ind_separator = int((-1) * len(separator_str))
    ind_eol = int((-1) * len(end_of_line))

    table_order_list = [
        "name",
        "distance",
        "duration",
        "linear_velocity_mean",
        "linear_velocity_std",
    ]

    table_str = ""
    for elem in table_order_list:
        table_str = table_str + elem + separator_str
    # String without last elements
    table_str = table_str[:ind_separator] + end_of_line

    round_dec = 2
    for aa in range(len(eval_dict["name"])):
        for key in table_order_list:
            if key == "name":
                table_str = table_str + eval_dict[key][aa] + separator_str
                continue
            # elif key == 'duration':
            # table_str = table_str + str(eval_dict[key][aa])
            # continue

            table_str = (
                table_str
                + str(round(eval_dict[key][aa]["mean"], round_dec))
                + separator_sum_var
                + str(round(eval_dict[key][aa]["std"], round_dec))
                + separator_str
            )

        # Remove separators at end of line
        table_str = table_str[:ind_separator] + end_of_line

    table_success_str = ""
    for key in list_simulation_outcome[0].keys():
        if key == "total":
            pass
        table_success_str = table_success_str + key + separator_str
    # String wihtout separator at end of line
    table_success_str = table_success_str[:ind_separator] + end_of_line

    # Get number of total runs success (same for all agents')
    num_total = list_simulation_outcome[0]["total"]

    for aa in range(len(eval_dict["name"])):
        # Add name at beginning of line
        table_success_str = table_success_str + eval_dict["name"][aa] + separator_str

        for key in list_simulation_outcome[0].keys():
            if key == "total":
                # Don't print total
                continue

            if key == "not_converged":
                # Unknown reason for no convergence
                continue

            table_success_str = (
                table_success_str
                + str(round(100.0 * list_simulation_outcome[aa][key] / num_total))
                + "\\%"
                + separator_str
            )

        table_success_str = table_success_str[:ind_separator] + end_of_line

    # Remove last new-line from tstring
    # table_str = table_str[:ind_eol]
    # table_success_str = table_success_str[:ind_eol]

    # breakpoint()
    return (
        table_str,
        table_success_str,
        {"num_success_metrics": num_success_metrics, "num_total": num_total},
    )


# def expanding_circle():
# pass


def compare_algorithms_plot():
    """ """
    # create empty obstacle list
    obs = ShapelyContainer()

    obs.append(
        Ellipse(
            center_position=[3.5, 0.4],
            orientation=30.0 / 180.0 * pi,
            axes_length=[1.2, 2.0],
        )
    )

    x_lim = [-0.1, 11]
    y_lim = [-4.5, 4.5]

    xAttractor = np.array([8.0, -0.1])
    # x_lim, y_lim = [-0, 7.1], [-0.1, 7.1]

    num_point_grid = 100

    vel = np.array([1.0, 0])
    pos = np.array([0.0, 1.0])
    vel_mod = obs_avoidance_interpolation_moving(pos, vel, obs=obs)
    print(vel_mod)

    vel_mod = obs_avoidance_orthogonal_moving(pos, vel, obs=obs)
    print(vel_mod)

    if True:
        Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            xAttractor=xAttractor,
            saveFigure=False,
            figName="compare_algorithms_potential_field",
            obs_avoidance_func=obs_avoidance_potential_field,
            point_grid=num_point_grid,
            # show_streamplot=False,
            noTicks=False,
            # draw_vectorField=True,  automatic_reference_point=True, point_grid=N_resol
        )

    if False:
        Simulation_vectorFields(
            x_lim,
            y_lim,
            obs=obs,
            xAttractor=xAttractor,
            saveFigure=False,
            figName="compare_algorithms_orthogonal",
            obs_avoidance_func=obs_avoidance_orthogonal_moving,
            # obs_avoidance_func=obs_avoidance_interpolation_moving,
            # draw_vectorField=True,
            # automatic_reference_point=True,
            show_streamplot=True,
            point_grid=num_point_grid,
        )


def comparison_subplots(
    rand_seed_0=None, rand_seed_1=1, fig_num=1001, save_figure=False
):
    """Create Figure with several stopping times."""
    it_plot = 0
    # fig, ax = plt.subplots(figsize=(14, 5), num=fig_num)
    fig, ax = plt.subplots(figsize=(7, 3), num=fig_num)

    n_cols = 3
    n_rows = 1

    if rand_seed_0 is not None:
        np.random.seed(rand_seed_0)

    it_plot += 1
    ax = plt.subplot(n_rows, n_cols, it_plot)
    compare_algorithms_random(
        max_it=10,
        visualize_scene=False,
        fig_and_ax_handle=(fig, ax),
        fig_num=fig_num,
        plot_last_image=True,
        show_legend=False,
    )

    it_plot += 1
    np.random.seed(rand_seed_0)
    stop_time = 10
    ax = plt.subplot(n_rows, n_cols, it_plot)
    compare_algorithms_random(
        max_it=50,
        visualize_scene=False,
        fig_and_ax_handle=(fig, ax),
        fig_num=fig_num,
        plot_last_image=True,
        show_legend=False,
    )
    ax_middle = ax

    it_plot += 1
    np.random.seed(rand_seed_0)
    stop_time = 10
    ax = plt.subplot(n_rows, n_cols, it_plot)
    line_labels = compare_algorithms_random(
        max_it=100,
        visualize_scene=False,
        fig_and_ax_handle=(fig, ax),
        fig_num=fig_num,
        plot_last_image=True,
        show_legend=False,
    )

    # plt.legend(handles=line_labels, bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.legend(handles=line_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax_middle.legend(
        handles=line_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        fancybox=False,
        shadow=False,
        ncol=3,
    )

    if save_figure:
        plt.savefig("figures/" + "subplot_comparison" + ".png", bbox_inches="tight")


if (__name__) == "__main__":

    if False:
        # Visual evaluation with three plots
        # comparison_subplots(rand_seed_0=9, save_figure=False)
        comparison_subplots(rand_seed_0=56, save_figure=False)
        # comparison_subplots(rand_seed_0=56, save_figure=False)

    if True:
        # Visualize 1 run
        np.random.seed(2)
        metrics_tuple = multiple_random_runs(num_runs=1, visualize=True)
    # Numerical evaluation
    if False:
        # Specific seed is chosen for 'replicability of numbers' of report
        np.random.seed(0)
        # compare_algorithms_plot()

        # for ii in range(1):
        # compare_algorithms_random()

        metrics_tuple = multiple_random_runs(num_runs=300)
        tables_results = evaluation_metrics(metrics_tuple)

        print("Info")
        print(tables_results[2])

        print("Table success")
        print(tables_results[1])

        print("Table Metrics")
        print(tables_results[0])
