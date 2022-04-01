"""
Obstacle Avoidance Algorithm script with vecotr field

@author LukasHuber
@date 2018-02-15
"""


# General classes
import numpy as np
from math import pi
import copy
import time

import matplotlib.pyplot as plt

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import (
    make_velocity_constant,
)

# from dynamic_obstacle_avoidance.dynamical_system import *
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import (
    obs_check_collision,
)
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_dynamic_center_3d import *

from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle_container import (
    ObstacleContainer,
)

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import (
    linear_ds_max_vel,
)

from dynamic_obstacle_avoidance.visualization.animated_simulation import (
    run_animation,
)


class Visualization3dLevel:
    def __init__(
        self,
        x_range=[-0.3, 0.3],
        y_range=[-0.3, 0.3],
        z_range=[-1, 1],
        obs=ObstacleContainer([]),
        figureSize=(8, 8),
    ):

        self.dim = 3  # projection from 3dnh onto 2d

        # Numerical hull of ellipsoid
        self.figureSize = figureSize
        self.obs = obs

        self.x_range, self.y_range, self.z_range = x_range, y_range, z_range

    def vectorfield2d(self, save_figure=False, attractor=None, z_value=None):

        if attractor is None:
            pos_attractor = np.array([0.075, 0.075, 0.15])

        if z_value is None:
            if not self.z_range is None:
                z_value = self.z_range

            else:
                print("WARNING - NO DRAWING METHOD CHOSEN")
                return

        for n in range(len(self.obs)):
            # Create obstacle boundary
            print("z val", z_value)
            self.obs[n].draw_obstacle(
                numPoints=50, z_val=z_value
            )  # 50 points resolution

        # import pdb; pdb.set_trace()
        # Plot obstacle
        self.draw_2d_obstacles(z_val=z_value, save_figure=save_figure)

        # print(z_value)
        # import pdb; pdb.set_trace()
        self.draw_2d_vectorfield(
            z_val=z_value,
            pos_attractor=pos_attractor,
            num_grid=40,
            stream_plot=False,
        )

        if False:  # Compute only one Point
            # position = np.array([0.075, 0.075, 0.15])
            # position = np.array([-0.0167, -0.0501, 0.15])
            position = np.array([0.01669, -0.04997, 0.15])
            normal = self.obs[0].get_normal_direction(position)
            Gamma = self.obs[0].get_gamma(position, in_global_frame=True)
            vel_init = linear_ds_max_vel(position, pos_attractor)
            vel_modu = obs_avoidance_interpolation_moving(
                position, vel_init, obs=self.obs
            )
            import pdb

            pdb.set_trace()  ## DEBUG ##

    # else:
    # self.obs[n].draw_obstacle(numPoints=50) # 50 points resolution
    # print("NO DRAWING METHOD CHOSEN")

    def animate2d(self, x_init, attractor_position=np.array([0, 0, 0])):
        run_animation(
            x_init,
            self.obs,
            x_range=self.x_range,
            y_range=self.y_range,
            dt=0.05,
            N_simuMax=1040,
            convergenceMargin=0.3,
            sleepPeriod=0.01,
            attractorPos=attractor_position,
            animationName="surgery_setup",
            saveFigure=False,
            dimension=2,
        )

    def draw_2d_obstacles(self, z_val=None, save_figure=False):
        """Draws one level of the obstacle"""

        self.fig, self.ax = plt.subplots(figsize=self.figureSize)

        obs_polygon = []
        obs_polygon_sf = []

        for n in range(len(self.obs)):

            x_obs_sf = self.obs[n].x_obs  # todo include in obs_draw_ellipsoid

            plt.plot(x_obs_sf[0, :], x_obs_sf[1, :], "k--")

            obs_polygon.append(
                plt.Polygon(self.obs[n].x_obs[:2, :].T, alpha=0.8, zorder=2)
            )

            if self.obs[n].is_boundary:
                boundary_polygon = plt.Polygon(
                    np.vstack(
                        (
                            np.array(
                                [
                                    [
                                        self.x_range[0],
                                        self.x_range[1],
                                        self.x_range[1],
                                        self.x_range[0],
                                    ],
                                    [
                                        self.y_range[0],
                                        self.y_range[0],
                                        self.y_range[1],
                                        self.y_range[1],
                                    ],
                                ]
                            ).T
                        )
                    ),
                    alpha=0.5,
                    zorder=-2,
                )

                boundary_polygon.set_color(np.array([176, 124, 124]) / 255.0)

                obs_polygon[n].set_color("white")

            else:
                if len(obstacleColor) == len(self.obs):
                    obs_polygon[n].set_color(obstacleColor[n])
                else:
                    obs_polygon[n].set_color(np.array([176, 124, 124]) / 255)

            obs_polygon_sf.append(
                plt.Polygon(self.obs[n].x_obs[:2, :].T, zorder=1, alpha=0.2)
            )
            obs_polygon_sf[n].set_color([1, 1, 1])

            if self.obs[n].is_boundary:
                plt.gca().add_patch(boundary_polygon)

            plt.gca().add_patch(obs_polygon_sf[n])
            plt.gca().add_patch(obs_polygon[n])

            self.ax.plot(
                self.obs[n].reference_point[0],
                self.obs[n].reference_point[1],
                "k+",
                linewidth=18,
                markeredgewidth=4,
                markersize=13,
            )

        plt.axis("equal")
        self.ax.set_xlim(self.x_range)
        self.ax.set_ylim(self.y_range)

        # import pdb; pdb.set_trace()

        if save_figure:
            inf_str = "".join(str(e) for e in self.obs[0].inflation_parameter)
            figName = "surgery_setup_param" + inf_str
            try:
                plt.savefig("figures/" + figName + ".png", bbox_inches="tight")
            except:
                plt.savefig("../figures/" + figName + ".png", bbox_inches="tight")

        else:
            plt.ion()
            plt.show()

        # import pdb; pdb.set_trace()

    def draw_2d_vectorfield(
        self,
        z_val=0,
        num_grid=10,
        dim=3,
        pos_attractor=[0.1, 0.1, 0],
        stream_plot=True,
    ):
        # xx = np.linspace(self.x_range[0], self.x_range[1], num_grid)
        # yy = np.linspace(self.y_range[0], self.y_range[1], num_grid)

        positions = np.zeros((dim, num_grid, num_grid))
        vel_init = np.zeros((dim, num_grid, num_grid))
        vel_modul = np.zeros((dim, num_grid, num_grid))
        vel_modul_stand = np.zeros((dim, num_grid, num_grid))

        num_x, num_y = num_grid, num_grid
        YY, XX = np.mgrid[
            self.y_range[0] : self.y_range[1] : num_y * 1j,
            self.x_range[0] : self.x_range[1] : num_x * 1j,
        ]
        ZZ = np.ones(XX.shape) * z_val

        Gammas = np.zeros((num_grid, num_grid))
        for ix in range(num_grid):
            for iy in range(num_grid):
                positions[:, ix, iy] = [XX[ix, iy], YY[ix, iy], ZZ[ix, iy]]

                Gammas[ix, iy] = self.obs[0].get_gamma(
                    positions[:, ix, iy], in_global_frame=False
                )
                # print('pos = {} ---- gamma = {}'.format(positions[:, ix, iy], Gammas[ix, iy]))
                # import pdb; pdb.set_trace()
                if Gammas[ix, iy] < 1:
                    continue

                vel_init[:, ix, iy] = get_linear_ds(positions[:, ix, iy], pos_attractor)
                vel_modul[:, ix, iy] = obs_avoidance_interpolation_moving(
                    positions[:, ix, iy],
                    vel_init[:, ix, iy],
                    obs=self.obs,
                    repulsive_obstacle=False,
                )
                vel_modul_stand[:, ix, iy] = make_velocity_constant(
                    vel_modul[:, ix, iy],
                    positions[:, ix, iy],
                    pos_attractor,
                    constant_velocity=0.1,
                    slowing_down_radius=0.01,
                )

                # print('pos', positions[:, ix, iy])
                # print('vel init', vel_init[:, ix, iy])
                # print('vel modu', vel_modul[:, ix, iy])
                # print('\n\n')

        # import pdb; pdb.set_trace()

        # self.ax.quiver(positions[0, :, :].flatten(), positions[1, :, :].flatten(), normals_surface[0, :, :].flatten(), normals_surface[1, :, :].flatten(), color='b')
        # self.ax.quiver(positions[0, :, :].flatten(), positions[1, :, :].flatten(), vel_modul[0, :, :].flatten(), vel_modul[1, :, :].flatten(), color='b')
        # print('norm vel \n', np.round(LA.norm(vel_modul, axis=0), 2), )

        if stream_plot:
            self.ax.streamplot(
                positions[0, :, :],
                positions[1, :, :],
                vel_modul_stand[0, :, :],
                vel_modul_stand[1, :, :],
                color="b",
            )
        else:
            self.ax.quiver(
                positions[0, :, :].flatten(),
                positions[1, :, :].flatten(),
                vel_modul_stand[0, :, :].flatten(),
                vel_modul_stand[1, :, :].flatten(),
                color="b",
            )

        plt.plot(pos_attractor[0], pos_attractor[1], "k*")
        plt.show()
        # print('norm vel \n', np.round(LA.norm(vel_modul, axis=0), 2), )
        # print('Gammas \n', np.round(Gammas, 2))
        return

    def draw_gamma_field(self, z_val=0, num_grid=40, dim=3):
        xx = np.linspace(self.x_range[0], self.x_range[1], num_grid)
        yy = np.linspace(self.y_range[0], self.y_range[1], num_grid)
        zz = z_val

        positions = np.zeros((dim, num_grid, num_grid))
        # vel_init = np.zeros((dim, num_grid, num_grid))
        # vel_modul = np.zeros((dim, num_grid, num_grid))

        Gammas = np.zeros((num_grid, num_grid))
        normals_surface = np.zeros((dim, num_grid, num_grid))

        for ix in range(num_grid):
            for iy in range(num_grid):
                positions[:, ix, iy] = [xx[ix], yy[iy], zz]

                normals_surface[:, ix, iy] = self.obs[0].get_normal_direction(
                    positions[:, ix, iy]
                )
                Gammas[ix, iy] = self.obs[0].get_gamma(positions[:, ix, iy])

        self.ax.quiver(
            positions[0, :, :].flatten(),
            positions[1, :, :].flatten(),
            normals_surface[0, :, :].flatten(),
            normals_surface[1, :, :].flatten(),
            color="b",
        )
        cs = self.ax.imshow(
            Gammas,
            extent=[xx[0], xx[-1], yy[0], yy[-1]],
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            vmax=3,
        )
        # positions[0, :, :].flatten(), positions[1, :, :].flatten(),
        # cs = ax.contourf(xx_r, yy_r, predict_class, cmap=plt.cm.coolwarm, alpha=0.8)
        cbar = self.fig.colorbar(cs)

        plt.show()

    def draw_2d_line(self, n_pos=2, point1=[-0.03, -0.02], point2=[-0.02, -0.0119]):
        # point1 = [-0.0, -0.02]
        # point2 = [-0.02, -0.0]

        # point2 = [-0.011, -0.005]

        x_vals = [point1[0], point2[0]]
        y_vals = [point1[1], point2[1]]
        # x_vals = [-0.02,-0.01]
        # y_vals = [-0.01, -0.02]
        z_vals = [0, 0]
        positions = np.vstack(
            (
                np.linspace(x_vals[0], x_vals[1], n_pos),
                np.linspace(y_vals[0], y_vals[1], n_pos),
                np.linspace(z_vals[0], z_vals[1], n_pos),
            )
        )

        normals_surface = np.zeros((3, n_pos))
        for ii in range(n_pos):
            # positions[:, ix, iy] = [xx[ix], yy[iy], zz]
            # vel_init[:, ix, iy] = linear_ds_max_vel(positions[:, ix, iy])
            # vel_modul[:, ix, iy] =  obs_avoidance_interpolation_moving(vel_init[:, ix, iy])
            normals_surface[:, ii] = self.obs[0].get_normal_direction(positions[:, ii])

        self.ax.quiver(
            positions[0, :],
            positions[1, :],
            normals_surface[0, :],
            normals_surface[1, :],
            color="k",
        )
        plt.show()
