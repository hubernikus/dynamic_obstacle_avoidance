# !/usr/bin/env python2

"""
`Dataset UCY evaluation
"""
__author__ = "Lukas Huber"
__date__ = "2021-01-17"
__email__ = "lukas.huber@epfl.ch"

import sys
import os
import yaml
import copy
import time
from datetime import datetime

# from PIL import Image
import glob

import numpy as np
from math import pi

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation

from scipy import ndimage
from scipy.spatial.transform import Rotation

from threading import Lock

lock = Lock()

if sys.version_info < (3, 0):
    from itertools import izip as zip


# Check if obstacle avoidance library is installed
try:
    import dynamic_obstacle_avoidance
except:
    print("Importing path to obstacle avoidance library")

    # Add obstacle avoidance without 'setting' up
    path_avoidance = os.path.join(
        directory_path, "scripts", "dynamic_obstacle_avoidance", "src"
    )
    if not path_avoidance in sys.path:
        sys.path.append(path_avoidance)

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import (
    get_linear_ds,
    make_velocity_constant,
    linear_ds_max_vel,
    linearAttractor_const,
)

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import (
    get_linear_ds,
    make_velocity_constant,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.gradient_container import (
    GradientContainer,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.human_ellipse import (
    TrackedPedestrian,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import Obstacle
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import (
    Ellipse,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_polygon import (
    Polygon,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import (
    obs_avoidance_interpolation_moving,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.angle_math import (
    angle_difference_directional,
    transform_polar2cartesian,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.crowd_learning_container import (
    CrowdLearningContainer,
    CrowdCircleContainer,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import (
    CircularObstacle,
)

from dynamic_obstacle_avoidance.visualization.vector_field_visualization import (
    Simulation_vectorFields,
    plot_obstacles,
    plt_speed_line_and_qolo,
)

plt.ion()

### CONSTANT ROBOT PARAMETERS
ROBOT_MARGIN = 0.4  # m
HUMAN_RADIUS = 0.35

RADIUS_HUMAN_PLUS_MARGIN = 0.6

num_crowd_close = 10

SAFE_ANIMATION = False
SAFE_FIG = False

DES_SPEED = 0.5
MAX_SPEED = 0.8

# CONSTANT PARAMETERS
DIM = 2

DISTANCE_SCALING = 1.0 / 100

IMAGE_DIR = os.path.join("/home/lukas/Code/dynamic_obstacle_avoidance", "data")


class DynamicAnimationQOLO:
    def __init__(
        self,
        position_init,
        attractor_position=[0, 0],
        x_lim=None,
        y_lim=None,
        obstacle_list=None,
    ):

        self.position_init = position_init
        self.attractor_position = attractor_position

        # Set axes
        self.x_lim = x_lim
        self.y_lim = y_lim

        if obstacle_list is not None:
            self.obstacle_list = obstacle_list

    def on_click(self, event):
        # Pause when one clicks on image
        self.is_pause ^= True

    def run_dynamic_animation(
        self,
        dt_simulation=0.01,
        max_it=500,
        scale_qolo=1,
        des_speed=0.3,
        dt_sleep=0.01,
        show_plot_vectorfield=False,
    ):
        """Dynamic animation of the crowd moving."""
        fig_num = 1001  # TODO: convert to name

        if plt.fignum_exists(fig_num):
            plt.close(fig_num)

        if not show_plot_vectorfield:
            fig, ax = plt.subplots(num=fig_num)
        else:
            fig, (ax_vec, ax) = plt.subplots(1, 2, num=fig_num, figsize=(14, 6))

        # Enable Pause on button-click
        self.is_pause = False
        fig.canvas.mpl_connect("button_press_event", self.on_click)

        plt.show()

        if False:
            # if self.x_lim is not None:
            ax.set_xlim(self.x_lim)
            ax.set_ylim(self.y_lim)

            if show_plot_vectorfield:
                ax_vec.set_xlim(self.x_lim)
                ax_vec.set_ylim(self.y_lim)

        # Create all Lines
        max_ii = 1

        self.agent = ObjectQOLO(
            center_position=self.position_init,
            attractor_position=self.attractor_position,
            scale_qolo=scale_qolo,
        )
        self.agent.ax = ax

        it_count = 0

        while True:

            if self.is_pause:
                plt.pause(dt_sleep)
                continue

            print(f"It {it_count}")

            # Create simualtion time
            simulation_time = it_count * dt_simulation
            it_count += 1

            for obs in self.obstacle_list:
                if obs.is_deforming:
                    obs.update_deforming_obstacle(simulation_time)

            # Clear axes
            ax.cla()

            # Draw Robot Moving next to the vectorfield
            self.agent.iterate_pos(
                obstacle_list=self.obstacle_list,
                des_speed=des_speed,
                dt=dt_simulation,
            )

            self.agent.display_agent(point_only=False)

            Simulation_vectorFields(
                self.x_lim,
                self.y_lim,
                obs=self.obstacle_list,
                xAttractor=self.attractor_position,
                saveFigure=False,
                figName="",
                noTicks=True,
                showLabel=False,
                draw_vectorField=False,
                show_streamplot=False,
                point_grid=0,
                normalize_vectors=False,
                reference_point_number=False,
                drawVelArrow=True,
                automatic_reference_point=False,
                fig_and_ax_handle=(fig, ax),
                gamma_distance=RADIUS_HUMAN_PLUS_MARGIN,
            )

            if show_plot_vectorfield:
                ax_vec.cla()
                Simulation_vectorFields(
                    self.x_lim,
                    self.y_lim,
                    obs=self.obstacle_list,
                    xAttractor=self.attractor_position,
                    saveFigure=False,
                    figName="",
                    noTicks=True,
                    showLabel=False,
                    draw_vectorField=True,
                    show_streamplot=False,
                    point_grid=10,
                    normalize_vectors=True,
                    reference_point_number=False,
                    drawVelArrow=True,
                    automatic_reference_point=False,
                    fig_and_ax_handle=(fig, ax_vec),
                    gamma_distance=RADIUS_HUMAN_PLUS_MARGIN,
                )

            if it_count > max_it:
                print(f"Simulation ended after maximum of {max_it} iterations.")
                break

            if not plt.fignum_exists(fig_num):
                print(f"Simulation ended with closing of figure")
                break

            if self.agent.check_if_converged():
                print(f"Convergence to attractor reached after {it_count} iterations.")
                break

            plt.pause(dt_sleep)

    @property
    def active_people(self):
        """Get only crowd which is currently 'actively' moving."""
        # Careful this can get relatively expensive

        active_peops = []
        for person in self.dynamic_people:
            if person.is_active:
                active_peops.append(person)
        return active_peops


class ObjectQOLO(Obstacle):
    """QOLO or different agent as <<obstacle>>."""

    image_name = "Qolo_T_CB_top_bumper.png"

    def __init__(self, attractor_position=None, ax=None, scale_qolo=1, **kwargs):
        if sys.version_info > (3, 0):
            super().__init__(name="QOLO", orientation=0, **kwargs)
        else:
            super(CircularObstacle, self).__init__(name="QOLO", orientation=0, **kwargs)

        if attractor_position is not None:
            self.attractor_position = attractor_position

        if ax is not None:
            # Axes
            self.ax = ax

        self.margin = ROBOT_MARGIN

        self.arr_img = mpimg.imread(os.path.join(IMAGE_DIR, self.image_name))
        self.length_x = 1.0 * scale_qolo
        self.length_y = (
            (1.0) * self.arr_img.shape[0] / self.arr_img.shape[1] * self.length_x
        )

    @property
    def ax(self):
        return self._ax

    @ax.setter
    def ax(self, value):
        self._ax = value

    def check_if_converged(self, convergence_margin=0.1):
        """Check if agent has reached convergence"""
        dist_attr = np.linalg.norm(self.position - self.attractor_position)

        return dist_attr < convergence_margin

    def iterate_pos(
        self,
        obstacle_list,
        ds_initial=None,
        attractor=None,
        dt=0.1,
        des_speed=0.3,
    ):
        """Iterate agent pose based on moulation"""

        if attractor is not None:
            self.attractor_position = attractor

        if ds_initial is None:
            ds_initial = linear_ds_max_vel(
                self.position, attractor=self.attractor_position
            )

        ds_modulated = obs_avoidance_interpolation_moving(
            self.position,
            ds_initial,
            obstacle_list,
            tangent_eigenvalue_isometric=False,
            repulsive_obstacle=False,
        )

        # Only slow down at attaractor
        ds_modulated = make_velocity_constant(
            ds_modulated,
            self.position,
            self.attractor_position,
            constant_velocity=des_speed,
        )

        self.linear_velocity = ds_modulated
        self.orientation = np.arctan2(self.linear_velocity[1], self.linear_velocity[0])

        # import pdb; pdb.set_trace()

        self.position = self.position + self.linear_velocity * dt

    def display_agent(
        self, rotation=None, ax=None, display_velocity=True, point_only=False
    ):
        """Plot picture d (or only a point represetation) of the agent and it's velocity."""

        if ax is not None:
            self.ax = ax

        self.ax.quiver(
            self.position[0],
            self.position[1],
            self.linear_velocity[0],
            self.linear_velocity[1],
            color="b",
            alpha=0.7,
            scale=5.0,
            label="Mouldated DS",
        )

        if point_only:
            self.ax.scatter(self.position[0], self.position[1], s=100, color="k")

        else:
            rot = self.orientation
            arr_img_rotated = ndimage.rotate(self.arr_img, rot * 180.0 / pi, cval=255)

            lenght_x_rotated = (
                np.abs(np.cos(rot)) * self.length_x
                + np.abs(np.sin(rot)) * self.length_y
            )

            lenght_y_rotated = (
                np.abs(np.sin(rot)) * self.length_x
                + np.abs(np.cos(rot)) * self.length_y
            )

            self.ax.imshow(
                arr_img_rotated,
                extent=[
                    self.position[0] - lenght_x_rotated / 2.0,
                    self.position[0] + lenght_x_rotated / 2.0,
                    self.position[1] - lenght_y_rotated / 2.0,
                    self.position[1] + lenght_y_rotated / 2.0,
                ],
            )


class LineObject(CircularObstacle):
    # Delta time between frames [s]
    seconds_per_frame = 1
    # if 'DISTANCE_SCALING' in locals():
    distance_scaling = 1 / 100
    # else:
    # distance_scaling = 1

    def __init__(self, people_path, ax):
        # Dimensionxo
        self.dim = 2

        # Set some properties
        # self.radius = 0.35
        self.radius = 0.6
        self.margin_absolut = 0.5

        self._path_dictionary = people_path
        self.frame_list_it = 0

        self.position = np.zeros(DIM)
        self.orientation = 0
        self.linear_velocity = np.zeros(DIM)

        if sys.version_info > (3, 0):
            super().__init__(
                radius=self.radius,
                center_position=self.position,
                orientation=self.orientation,
                margin_absolut=self.margin_absolut,
            )
        else:
            super(CircularObstacle, self).__init__(
                radius=self.radius,
                center_position=self.position,
                orientation=self.orientation,
                margin_absolut=self.margin_absolut,
            )

        self.is_human = True
        self.tail_effect = False
        self.sigma = 5  # exponential weight for veloctiy reduction
        self.reactivity = 2  # veloctiy reduction
        self.repulsion_coeff = 1.5

        self.line_points = np.zeros((DIM, 0))

        (self.line_object,) = plt.plot([], [])
        # if type(ax) is tuple:
        self.point_object = ax.scatter([], [], s=20)

    def get_position(self, it):
        """Get the position already scaled"""
        return self._path_dictionary["positions"][:, it] * self.distance_scaling

    @property
    def positions(self):
        return self._path_dictionary["positions"]

    @property
    def orientations(self):
        return self._path_dictionary["orientations"]

    @property
    def frame_list(self):
        return self._path_dictionary["frame_id"]

    @property
    def is_active(self):
        if self.frame_list_it is None or self.frame_list_it is 0:
            return False
        else:
            return True

    def update_plot(self, current_frame, mode=None, dt=1.0):
        """Update the plto library."""
        self.update_velocity_orientation(current_frame)

        if self.frame_list_it is None or self.frame_list_it is 0:
            return

        self.position = self.position + self.linear_velocity * dt

        # Update visualization list & object
        self.line_points = np.vstack((self.line_points.T, self.position)).T
        self.line_object.set_data(self.line_points[0, :], self.line_points[1, :])

        self.point_object.set_offsets(self.line_points[:, -1].T)

        # print('pos', self.position)
        # print('vel', self.velocity)
        # return

    def update_velocity_orientation(self, current_frame):
        if self.frame_list_it is 0:
            if current_frame < self.frame_list[0]:
                return

            else:
                # First frame
                self.frame_list_it += 1  # Start at one
                self.position = self.get_position(it=0)

        elif self.frame_list_it is None:
            # Past last frame
            self.frame_list_it = None
            return

        elif current_frame >= self.frame_list[self.frame_list_it]:
            self.frame_list_it += 1

            if self.frame_list_it == len(self.frame_list):
                # TODO: sure that already none here?
                self.frame_list_it = None
                return

        # Desired time to next
        delta_frame = self.frame_list[self.frame_list_it] - current_frame
        delta_t = delta_frame * self.seconds_per_frame

        # 2D-Velocity
        self.linear_velocity = (
            self.get_position(it=self.frame_list_it) - self.position
        ) / delta_t

        if np.isnan(self.linear_velocity[0]):
            import pdb

            pdb.set_trace()

    def initalize_line():
        pass

    def avoid_robot(self, method=""):
        pass


if (__name__) == "__main__":
    # Main evaluation

    if True:
        file_name = "uni_examples.vsp"
        x_lim = [-500.0, 500.0]
        y_lim = [-250.0, 250.0]

        x_lim = [xx * DISTANCE_SCALING for xx in x_lim]
        y_lim = [yy * DISTANCE_SCALING for yy in y_lim]

    attractor_position = np.array([4.0, 0.0])

    obstacle_list = []


print("\n... finished script. Tune in another time!")
plt.close("all")
