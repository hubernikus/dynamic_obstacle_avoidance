"""
Dynamic Simulation - Obstacle Avoidance Algorithm
"""

# Command to automatically reload libraries -- in ipython before exectureion
import numpy as np
from numpy import pi

import time
import warnings

__author = "LukasHuber"
__date__ = "2019-05-24"

# Visualization libraries
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib._pylab_helpers
from matplotlib import animation

# 3D Animatcoion utils
# from mpl_toolkits.mplot3d import Axes3D
# import mpl_toolkits.mplot3d.art3d as art3d

from dynamic_obstacle_avoidance.dynamical_system.dynamical_system_representation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obstacle import *
from dynamic_obstacle_avoidance.obstacle_avoidance.modulation import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.obstacle_avoidance.obs_dynamic_center_3d import *
from dynamic_obstacle_avoidance.obstacle_avoidance.linear_modulations import *

from dynamic_obstacle_avoidance.obstacle_avoidance.dynamic_boundaries_polygon import (
    DynamicBoundariesPolygon,
)

plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"

##### Anmation Function #####
class AnimatedMultibody:
    """
    An animated scatter plot using matplotlib.animations.FuncAnimation.
    """

    def __init__(
        self,
        x0=None,
        obs=[],
        N_simuMax=600,
        dt=0.01,
        attractorPos=None,
        convergenceMargin=0.01,
        x_range=[-10, 10],
        y_range=[-10, 10],
        zRange=[-10, 10],
        sleepPeriod=0.03,
        RK4_int=False,
        dynamicalSystem=linearAttractor,
        hide_ticks=True,
        figSize=(8, 7),
        dimension=None,
        velocity_max=None,
    ):

        if x0 is None:
            self.dim = 2  # Default
            self.infitineLoop = True
        else:
            self.infitineLoop = False
            self.dim = x0.shape[0]
        self.print_count = False

        if not dimension is None:
            self.dim = dimension

        # Initialize class variables
        self.obs = obs
        self.N_simuMax = N_simuMax
        self.dt = dt

        self.velocity_max = velocity_max

        if attractorPos is None:
            self.attractorPos = np.zeros((self.dim, len(self.obs)))
        else:
            self.attractorPos = np.array(attractorPos)

        self.sleepPeriod = sleepPeriod

        self.hide_ticks = hide_ticks

        # last three values are observed for convergence
        self.convergenceMargin = convergenceMargin
        self.lastConvergences = [convergenceMargin for i in range(3)]

        # Get current simulation time
        self.old_time = time.time()

        self.N_points = x0.shape[1]

        self.x_pos = np.zeros((self.dim, self.N_simuMax + 2, self.N_points))

        self.x_pos[:, 0, :] = x0

        self.xd_ds = np.zeros((self.dim, self.N_simuMax + 1, self.N_points))
        self.t = np.linspace(0, self.N_simuMax + 1, num=self.N_simuMax + 1) * dt

        # Simulation parameters
        self.RK4_int = RK4_int
        self.dynamicalSystem = dynamicalSystem

        self.converged = False

        self.iSim = 0

        self.lines = []  # Container to keep line plots
        self.startPoints = []  # Container to keep line plots
        self.endPoints = []  # Container to keep line plots
        self.patches = []  # Container to keep patch plotes
        self.contour = []
        self.centers = []
        self.cent_dyns = []

        # Setup the figure and axes.
        if self.dim == 2:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.set_size_inches(figSize)

        self.ax.set_xlim(x_range)
        self.ax.set_ylim(y_range)
        # self.ax.set_xlabel('x1')
        # self.ax.set_ylabel('x2')

        if self.dim == 3:
            self.ax.set_zlim(zRange)
            self.ax.set_zlabel("x3")
            # self.ax.view_init(elev=0.3, aim=0.4)

        # Set axis etc.
        plt.gca().set_aspect("equal", adjustable="box")

        # Adjust dynamic center intersection_obs = obs_common_section(self.obs)
        # dynamic_center_3d(self.obs, intersection_obs)

        # Button click variables
        self.pause_start = 0
        self.pause = False

        # Then setup FuncAnimation
        self.tt = np.linspace(0, 2 * np.pi)
        self.x = np.sin(self.tt)

        if self.infitineLoop:
            self.ani = FuncAnimation(
                self.fig,
                self.update,
                interval=1,
                frames=None,
                repeat=False,
                init_func=self.setup_plot,
                blit=True,
                save_count=self.N_simuMax - 2,
            )
        else:
            self.ani = FuncAnimation(
                self.fig,
                self.update,
                interval=1,
                frames=None,
                repeat=False,
                init_func=self.setup_plot,
                blit=True,
                save_count=self.N_simuMax - 2,
            )

    def update(self, iSim):
        # print('Start loop {}'.format(iSim))
        if self.pause:  # NO ANIMATION -- PAUSE
            self.old_time = time.time()
            return (
                self.lines
                + self.obs_polygon
                + self.contour
                + self.centers
                + self.cent_dyns
                + self.startPoints
                + self.endPoints
                + self.attr_pos
            )

        if not (iSim % 10) and self.print_count:  # Display every tenth loop count
            print(
                "loop count={} - frame ={}-Simulation time ={}".format(
                    self.iSim, iSim, np.round(self.dt * self.iSim, 3)
                )
            )

        # intersection_obs = obs_common_section(self.obs)
        # dynamic_center_3d(self.obs, intersection_obs)

        if self.RK4_int:  # Runge kutta integration
            for j in range(self.N_points):
                ind_obs = np.hstack((np.arange(0, j), np.arange(j + 1, self.N_points)))
                obs = [self.obs[ii] for ii in ind_obs]

                self.x_pos[:, self.iSim + 1, j] = obs_avoidance_rk4(
                    self.dt,
                    self.x_pos[:, self.iSim, j],
                    obs,
                    x0=self.attractorPos[:, j],
                    obs_avoidance=obs_avoidance_interpolation_moving,
                )

        else:  # Simple euler integration
            # Calculate DS
            for j in range(self.N_points):
                ind_obs = np.hstack((np.arange(0, j), np.arange(j + 1, self.N_points)))
                obs = [self.obs[ii] for ii in ind_obs]

                xd_temp = linearAttractor(
                    self.x_pos[:, self.iSim, j], self.attractorPos[:, j]
                )

                self.xd_ds[:, self.iSim, j] = obs_avoidance_interpolation_moving(
                    self.x_pos[:, self.iSim, j],
                    xd_temp,
                    obs,
                    velocicity_max=self.velocity_max,
                )
                self.x_pos[:, self.iSim + 1, :] = (
                    self.x_pos[:, self.iSim, :] + self.xd_ds[:, self.iSim, :] * self.dt
                )

        self.t[self.iSim + 1] = (self.iSim + 1) * self.dt

        # Update plots
        for j in range(self.N_points):
            self.lines[j].set_xdata(self.x_pos[0, : self.iSim + 1, j])
            self.lines[j].set_ydata(self.x_pos[1, : self.iSim + 1, j])
            if self.dim == 3:
                self.lines[j].set_3d_properties(zs=self.x_pos[2, : self.iSim + 1, j])

            self.endPoints[j].set_xdata(self.x_pos[0, self.iSim + 1, j])
            self.endPoints[j].set_ydata(self.x_pos[1, self.iSim + 1, j])
            if self.dim == 3:
                self.endPoints[j].set_3d_properties(zs=self.x_pos[2, self.biSim + 1, j])

        # ========= Check collision ----------
        # collisions = obs_check_collision(self.x_pos[:,self.iSim+1,:], obs)
        # collPoints = np.array()

        for o in range(len(self.obs)):  # update obstacles if moving
            self.obs[o].update_position_and_orientation(
                position=self.x_pos[:, self.iSim + 1, o], orientation=0
            )

            self.centers[o].set_xdata(self.obs[o].center_position[0])
            self.centers[o].set_ydata(self.obs[o].center_position[1])
            if self.dim == 3:
                self.centers[o].set_3d_properties(zs=obs[o].center_position[2])

            if np.linalg.norm(self.obs[o].linear_velocity) or np.linalg.norm(
                self.obs[o].angular_velocity
            ):
                self.cent_dyns[o].set_xdata(self.obs[o].global_reference_point[0])
                self.cent_dyns[o].set_ydata(self.obs[o].global_reference_point[1])

                if self.dim == 2:
                    self.obs_polygon[o].xy = self.obs[o].x_obs[:2, :].T
                    self.contour[o].set_xdata(self.obs[o].x_obs_sf[0, :])
                    self.contour[o].set_ydata(self.obs[o].x_obs_sf[1, :])
                else:
                    self.obs_polygon[o].xyz = self.obs[o].x_obs[:3, :].T
                    self.cent_dyns[o].set_3d_properties(
                        zs=self.obs[o].reference_point[2]
                    )

        self.iSim += 1  # update simulation counter

        # Convergence is not discovered during video-saving
        self.check_convergence()  # Check convergence

        # Pause for constant simulation speed
        self.old_time = self.sleep_const(self.old_time)

        self.t[self.iSim + 1] = (self.iSim + 1) * self.dt

        # print('End loop {}'.format(iSim))
        if not self.iSim % 20:
            print("Loop {}/{}".format(self.iSim, self.N_simuMax))

        return (
            self.lines
            + self.obs_polygon
            + self.contour
            + self.centers
            + self.cent_dyns
            + self.startPoints
            + self.endPoints
            + self.attr_pos
        )

    def setup_plot(self):
        print("Start setup...")
        # Draw obstacle as polygon
        self.obs_polygon = []

        # Numerical hull of ellipsoid
        for n in range(len(self.obs)):

            if isinstance(self.obs[n], DynamicBoundariesPolygon):
                it_point = 0
                self.obs[n].draw_obstacle(
                    z_val=self.x_pos[1, self.iSim + 1, it_point]
                )  # Update obstacles)
            else:
                self.obs[n].draw_obstacle(numPoints=50)  # 50 points resolution

        for n in range(len(self.obs)):
            if self.dim == 2:
                emptyList = [[0, 0] for i in range(50)]

                if self.obs[n].is_boundary:
                    x_range = self.ax.get_xlim()
                    y_range = self.ax.get_ylim()
                    outer_boundary = np.array(
                        [
                            [x_range[0], x_range[1], x_range[1], x_range[0]],
                            [y_range[0], y_range[0], y_range[1], y_range[1]],
                        ]
                    ).T

                    boundary_polygon = plt.Polygon(outer_boundary, alpha=0.5, zorder=-2)
                    boundary_polygon.set_color(np.array([176, 124, 124]) / 255.0)
                    plt.gca().add_patch(boundary_polygon)  # No track of this one

                    self.obs_polygon.append(
                        plt.Polygon(self.obs[n].x_obs[:, :2], alpha=1.0, zorder=-1)
                    )
                    self.obs_polygon[n].set_color(np.array([1.0, 1.0, 1.0]))
                else:
                    self.obs_polygon.append(
                        plt.Polygon(
                            emptyList,
                            animated=True,
                        )
                    )

                    self.obs_polygon[n].set_color(np.array([176, 124, 124]) / 255.0)
                    self.obs_polygon[n].set_alpha(0.8)
                patch_o = plt.gca().add_patch(self.obs_polygon[n])
                self.patches.append(patch_o)

                if self.obs[n].x_end > 0:
                    (cont,) = plt.plot([], [], "k--", animated=True)
                else:
                    # cont, = plt.plot([self.obs[n].x_obs_sf[ii][0] for ii in range(len(self.obs[n].x_obs_sf))],
                    # [self.obs[n].x_obs_sf[ii][1] for ii in range(len(self.obs[n].x_obs_sf))],
                    # 'k--', animated=True)
                    (cont,) = plt.plot([], [], "k--", animated=True)
                self.contour.append(cont)

            else:  # 3d
                N_resol = 50  # TODO  save as part of obstacle class internally from assigining....
                self.obs_polygon.append(
                    self.ax.plot_surface(
                        np.reshape(
                            [obs[n].x_obs[i][0] for i in range(len(obs[n].x_obs))],
                            (N_resol, -1),
                        ),
                        np.reshape(
                            [obs[n].x_obs[i][1] for i in range(len(obs[n].x_obs))],
                            (N_resol, -1),
                        ),
                        np.reshape(
                            [obs[n].x_obs[i][2] for i in range(len(obs[n].x_obs))],
                            (N_resol, -1),
                        ),
                    )
                )

            # Center of obstacle
            (center,) = self.ax.plot([], [], "k.", animated=True)
            self.centers.append(center)

            # if hasattr(self.obs[n], 'center_dyn'):# automatic adaptation of center
            (cent_dyn,) = self.ax.plot(
                [],
                [],
                "k+",
                animated=True,
                linewidth=18,
                markeredgewidth=4,
                markersize=13,
            )

            self.cent_dyns.append(cent_dyn)

        for ii in range(self.N_points):
            (line,) = plt.plot([], [], "--", lineWidth=4, animated=True)
            self.lines.append(line)
            # point, = plt.plot(self.x_pos[0,0,ii],self.x_pos[1,0,ii], '*k', markersize=10, animated=True)
            (point,) = plt.plot([], [], "*k", markersize=10, animated=True)
            if self.dim == 3:
                (point,) = plt.plot(
                    self.x_pos[0, 0, ii],
                    self.x_pos[1, 0, ii],
                    self.x_pos[2, 0, ii],
                    "*k",
                    markersize=10,
                )
            self.startPoints.append(point)

            (point,) = plt.plot([], [], "bo", markersize=0, animated=True)
            self.endPoints.append(point)

        if self.dim == 2:
            (attr,) = plt.plot(
                self.attractorPos[0, :],
                self.attractorPos[1, :],
                "k*",
                linewidth=7.0,
                markeredgewidth=4,
                markersize=13,
            )
            self.attr_pos = [attr]
        else:
            (self.attr_pos,) = plt.plot(
                [self.attractorPos[0, :]],
                [self.attractorPos[1, :]],
                [self.attractorPos[2, :]],
                "k*",
                linewidth=7.0,
            )
            self.attr_pos = [attr]

        if self.hide_ticks:
            plt.tick_params(
                axis="both",
                which="major",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )

        self.fig.canvas.mpl_connect(
            "button_press_event", self.onClick
        )  # Button click enabled

        print("... finished setup")
        return (
            self.lines
            + self.obs_polygon
            + self.contour
            + self.centers
            + self.cent_dyns
            + self.startPoints
            + self.endPoints
            + self.attr_pos
        )

    def check_convergence(self, infitineLoop=False):
        self.lastConvergences[0] = self.lastConvergences[1]
        self.lastConvergences[1] = self.lastConvergences[2]

        self.lastConvergences[2] = np.sum(
            abs(self.x_pos[:, self.iSim, :] - self.attractorPos)
        )

        if (sum(self.lastConvergences) < self.convergenceMargin) or (
            self.iSim + 1 >= self.N_simuMax
        ):

            if infitineLoop:
                self.iSim = 0
                for ii in range(self.N_points):
                    self.x_pos[0, 0, ii] = self.attractorPos[0, :]
                    self.x_pos[1, 0, ii] = self.attractorPos[1, :]

                    # self.startPoints[ii].set_data(self.x_pos[0,0,ii], self.x_pos[1,0,ii])
                    self.startPoints[ii].set_xdata(self.x_pos[0, 0, ii])
                    self.startPoints[ii].set_ydata(self.x_pos[1, 0, ii])

                new_attractor_is_chosen = False
                while not new_attractor_is_chosen:
                    self.attractorPos[0, :] = (
                        np.random.rand(1)[0]
                        * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
                        + self.ax.get_xlim()[0]
                    )
                    self.attractorPos[1, :] = (
                        np.random.rand(1)[0]
                        * (self.ax.get_ylim()[1] - self.ax.get_ylim()[0])
                        + self.ax.get_ylim()[0]
                    )

                    new_attractor_is_chosen = True
                    for oo in range(len(self.obs)):
                        if self.obs[oo].get_gamma(self.attractorPos[:, oo]) < 1:
                            new_attractor_is_chosen = False
                            break

                self.attr_pos[0].set_data(
                    self.attractorPos[
                        0,
                    ],
                    self.attractorPos[1],
                )

            else:
                self.ani.event_source.stop()

                if self.iSim >= self.N_simuMax - 1:
                    print(
                        "Maximum number of {} iterations reached without convergence.".format(
                            self.N_simuMax
                        )
                    )
                    self.ani.event_source.stop()
                else:
                    print(
                        "Convergence with tolerance of {} reached after {} iterations.".format(
                            sum(self.lastConvergences), self.iSim + 1
                        )
                    )
                    self.ani.event_source.stop()

    def set_velocity(self, obs_number=0, vel_x=0.0, vel_y=0.0, vel_rot=0):
        self.obs[obs_number].xd = np.array([vel_x, vel_y])
        self.obs[obs_number].w = vel_rot

        plt.show()

    def show(self):
        plt.show()

    def sleep_const(self, old_time=0):
        next_time = old_time + self.sleepPeriod

        now = time.time()

        sleep_time = next_time - now  # get sleep time
        sleep_time = min(
            max(sleep_time, 0), self.sleepPeriod
        )  # restrict to sensible range

        time.sleep(sleep_time)

        return next_time

    def onClick(self, event):
        # Pause when one clicks on image
        self.pause ^= True
        if self.pause:
            self.pause_start = time.time()
        else:
            dT = time.time() - self.pause_start
            if dT < 0.3:  # Break simulation at double click
                print("Animation is exited.")
                self.ani.event_source.stop()


def run_animation_multibody(
    *args,
    animationName="test",
    saveFigure=False,
    return_animationObject=False,
    **kwargs
):
    # This function is called from other scripts
    # Somehow the class initialization <<Animated(**)>>
    # and anim.show() has to be in the same file/function
    print("Run it...")

    plt.close("all")
    # plt.ion()
    plt.ioff()

    anim = AnimatedMultibody(*args, **kwargs)  # Initialize

    # animation cannot run properly when getting saved at the same time to file.
    # i.e. choose one or the other for each run
    if saveFigure:
        print("Saving figures")
        try:  # avoid error warnings
            anim.ani.save("../figures/" + animationName + ".mp4", dpi=100, fps=50)
            print("Saving figure succesful.")
        except:
            warnings.warn("\n\n Saving not succesfull.")
            # raise RuntimeError('WARNING: saving not succesfull.')
            raise
        # print('Print finished saving')
        plt.close("all")

    elif return_animationObject:
        print("Returning animation object")
        return anim
    else:
        print("Starting animation")

        try:  # Avoid long error when shutting down.
            anim.show()
            print("Finished or converged.")
        except:
            print("\nWARNING: animation was interrupted.")
            # raise # Display for debugging


def test_function():
    t = np.linspace(0, 2 * np.pi)
    x = np.sin(t)

    fig, ax = plt.subplots()
    (l,) = ax.plot([0, 2 * np.pi], [-1, 1])

    animate = lambda i: l.set_data(t[:i], x[:i])
    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=len(t))

    return ani
