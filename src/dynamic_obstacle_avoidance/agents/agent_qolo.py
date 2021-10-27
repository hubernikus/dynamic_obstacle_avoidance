""" Object to represent agents. Here specifically the QOLO-agent"""

__author__ = "Lukas Huber"
__date__ = "2021-01-17"
__mail__ = "lukas.huber@epfl.ch"


import os
import sys
import warnings

import numpy as np
import matplotlib.image as mpimg

from dynamic_obstacle_avoidance.obstacle_avoidance.ellipse_obstacles import (
    Obstacle,
)


class AgentQOLO(Obstacle):
    """QOLO or different agent as <<obstacle>>."""

    image_name = "Qolo_T_CB_top_bumper.png"

    def __init__(
        self, robot_margin=0, attractor_position=None, ax=None, image_dir=None, **kwargs
    ):
        if sys.version_info > (3, 0):
            super().__init__(name="QOLO", **kwargs)
        else:
            super(AgentQOLO, self).__init__(name="QOLO", **kwargs)

        if attractor_position is not None:
            self.attractor_position = attractor_position

        if ax is not None:
            # Axes
            self.ax = ax

        self.margin = robot_margin

        if image_dir is None:
            self.arr_img = None
        else:
            self.arr_img = mpimg.imread(os.path.join(image_dir, self.image_name))
            self.length_x = 1.4
            self.length_y = (
                (1.0) * self.arr_img.shape[0] / self.arr_img.shape[1] * self.length_x
            )

        # Default control point
        self.control_point_local = np.zeros(self.dim)

    @property
    def ax(self):
        return self._ax

    @ax.setter
    def ax(self, value):
        self._ax = value

    @property
    def control_point_local(self):
        return self._control_point_local

    @control_point_local.setter
    def control_point_local(self, value):
        self._control_point_local = value

    @property
    def control_point_global(self):
        return self.transform_relative2global(self._control_point_local)

    # @property
    # def relative_control_point_global(self):

    def get_control_point_attractor(self, attractor, in_global_frame=True):
        """Calculate the attractor for the control point."""

        if not in_global_frame:
            raise ("Only in global frame")

        return attractor + self.transform_relative2global_dir(self.control_point_local)

    def iterate_pos(self, obstacle_list, ds_initial=None, attractor=None, dt=0.1):
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

        self.linear_velocity = ds_modulated
        self.orientation = np.arctan2(self.linear_velocity[1], self.linear_velocity[0])

        self.position = self.position + self.linear_velocity * dt

    def display_agent(self, rotation=None, ax=None, display_velocity=True):
        """Plot picture of agent"""

        if ax is not None:
            self.ax = ax

        if self.arr_img is None:
            self.ax.scatter(self.position[0], self.position[1], "k")

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


class ObjectQOLO(AgentQOLO):
    warnings.warn("ObjectQOLO depreciated, use AgentQOLO instead")
    pass
