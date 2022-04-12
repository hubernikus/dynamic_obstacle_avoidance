"""
@author lukas huber
email: hubernikus@gmail.com
"""
import warnings

import numpy as np
from numpy import pi

from matplotlib import pyplot as plt
from matplotlib import animation

# import matplotlib.cm as cm
# colors = cm.rainbow(np.linspace(0, 1, 10))

from pylab import imread, subplot, imshow, show

dim = 2  # two dimensional movement
fig = plt.figure(figsize=(14, 9))

import_image = True
if import_image:
    image_temp = imread("movid_19.png")

    resol_y = image_temp.shape[1]

    n_x = image_temp.shape[1] * 1.3
    n_y = n_x * 0.70  # quadratic

    # dx = (n_x-image_temp.shape[1])/0.5 + 0.5
    dx = (n_x - image_temp.shape[1]) * 0.5 + 0.5
    # dx = 0.15*image_temp.shape[1] + 0.5
    dy = (n_y - image_temp.shape[0]) / 2 + 0.5

    image_temp = np.flip(image_temp, axis=0)
    image_temp = np.logical_not(image_temp)

    n_points = np.sum(image_temp)

    image_temp = image_temp.flatten()
    pos = np.arange(image_temp.shape[0])[image_temp]

    pos_y = np.floor(pos / resol_y)
    pos_x = pos - pos_y * resol_y

    pos_points = np.vstack((pos_x + dx, pos_y + dy))

    ax = plt.axes(xlim=(0, n_x), ylim=(0, n_y))

else:
    pos_points = None
    ax = plt.axes(xlim=(0, 45), ylim=(0, 30))

# First set up the figure, the axis, and the plot element we want to animate
ax.set_aspect("equal")
# plt.grid()
ax.set_xticks([])
ax.set_yticks([])


(line,) = plt.plot([], [])
# plt.axis('equal')


class CrowdSimulation:
    # initialization function: plot the background of each frame

    def __init__(self, n_points=10, pos_points=None):
        self.marker_width = 8

        if pos_points is None:
            self.n_points = n_points

            self.position = np.random.rand(dim, self.n_points)
            self.position[0, :] = (
                self.position[0, :] * (ax.get_xlim()[1] - ax.get_xlim()[0])
                + ax.get_xlim()[0]
            )
            self.position[1, :] = (
                self.position[1, :] * (ax.get_ylim()[1] - ax.get_ylim()[0])
                + ax.get_ylim()[0]
            )
        else:
            self.n_points = pos_points.shape[1]
            self.position = pos_points

        self.scatter_color = np.zeros((self.n_points, 4))
        self.scatter_color[:, 3] = 1
        self.point = ax.scatter(
            [], [], color="k", linewidths=1, s=self.marker_width**2
        )
        # self.position[:, 0] = 0

        self.dynamic_vel = np.zeros((dim, self.n_points))

        self.sick_status = np.ones(self.n_points) * (-1)
        self.is_vaccinated = np.zeros(self.n_points, dtype=bool)
        self.is_healed = np.zeros(self.n_points, dtype=bool)
        self.sick_status[0] = 1  # First infection

        self.sick_cols = np.array([[32, 165, 41], [138, 165, 32]]) / 255.0
        self.healthy_col = np.array([61, 68, 162]) / 255.0
        self.healed_col = np.array([196, 165, 39]) / 255.0
        self.vaccinated_col = np.array([196, 65, 39]) / 255.0

        self.min_velocity = 1
        self.max_velocity = 10
        self.sigma_dir = 20
        self.sigma_mag = 20
        self.sigma_acc = 1

        self.mag_velocity = np.random.rand(self.n_points) * (self.max_velocity)
        self.dir_velocity = np.random.rand(self.n_points) * pi - pi

    def setup(self):
        self.point.set_offsets(self.position.T)
        self.point.set_color(self.scatter_color)
        return line, self.point

    # animation function.  This is called sequentially
    def animate(
        self,
        it_anmimation,
        dt=0.05,
    ):

        # dt proportional to interval?
        dynamic_acc = np.random.normal(
            0.0, self.sigma_acc, dim * self.n_points
        ).reshape(dim, self.n_points)
        self.dynamic_vel = self.dynamic_vel + dt * dynamic_acc

        velocity_magnitude = np.linalg.norm(self.dynamic_vel, axis=0)
        ind_too_fast = velocity_magnitude > self.max_velocity

        if any(ind_too_fast):
            self.dynamic_vel[:, ind_too_fast] = (
                self.dynamic_vel[:, ind_too_fast]
                / velocity_magnitude[ind_too_fast]
                * self.max_velocity
            )
        # self.mag_velocity  = self.mag_velocity + np.random.normal(0.0, self.sigma_mag, self.n_points)
        # self.mag_velocity[self.mag_velocity>self.max_velocity] = self.max_velocity
        # self.mag_velocity[self.mag_velocity<self.min_velocity] = self.min_velocity

        # self.dir_velocity  = self.dir_velocity + np.random.normal(0.0, self.sigma_dir, self.n_points)
        # self.dynamic_vel = np.vstack((np.cos(self.dir_velocity), np.sin(self.dir_velocity)))*self.mag_velocity

        self.position = self.position + self.dynamic_vel * dt

        x_lim = ax.get_xlim()
        diam = (x_lim[1] - x_lim[0]) / ax.get_window_extent().width * self.marker_width
        diam = diam / 2.1242697822623477 * 3.0  # observd value

        self.verify_wall_collision(diameter=diam)
        # self.verify_inter_collision()
        self.verify_inter_collision(min_distance=diam)

        self.update_health()

        self.point.set_offsets(self.position.T)
        self.point.set_color(self.scatter_color)

        return line, self.point

    # @staticmethod
    def angle_modulo(self, angles):
        return (angles + pi) % (2 * pi) - pi

    def refelect_direction_at_edge(self, angles, angle_normal_edge):
        angles = self.angle_modulo(angles - angle_normal_edge)

        return -angles + angle_normal_edge

    def update_health(self, healing_steps=0.001):
        ind_sick = self.sick_status > 0

        self.sick_status[ind_sick] = self.sick_status[ind_sick] - healing_steps

        still_sick = self.sick_status[ind_sick] > 0

        if np.sum(still_sick):
            ind_still = np.arange(ind_sick.shape[0])[ind_sick][still_sick]
            sick_status = np.tile(self.sick_status[ind_still], (3, 1)).T
            self.scatter_color[ind_still, :3] = sick_status * np.tile(
                self.sick_cols[0, :], (np.sum(still_sick), 1)
            ) + (1 - sick_status) * np.tile(
                self.sick_cols[1, :], (np.sum(still_sick), 1)
            )

        if np.sum(np.logical_not(still_sick)):
            ind_healed = np.arange(ind_sick.shape[0])[ind_sick][
                np.logical_not(still_sick)
            ]
            self.is_healed[ind_healed] = True
            self.scatter_color[ind_healed, :3] = np.tile(
                self.healed_col, (np.sum(ind_healed.shape[0]), 1)
            )

    def verify_inter_collision(self, min_distance=0.3):
        """Collision between two bubbles.
        Momentum is not contained but the individual velocity is kept constance
        """

        positions = self.position
        n_points = positions.shape[1]

        # TODO: done to many time.. only to upper triangle
        ind_uptriangle = (
            (np.triu(np.ones(n_points)) - np.diag(np.ones(n_points)))
            .astype(bool)
            .flatten()
        )
        ind_numb_uptriangle = np.arange(ind_uptriangle.shape[0])[ind_uptriangle]

        pos_arr1 = np.swapaxes(np.tile(positions, (n_points, 1, 1)), 0, 1).reshape(
            dim, -1
        )[:, ind_uptriangle]
        pos_arr2 = np.swapaxes(np.tile(positions.T, (n_points, 1, 1)), 0, 2).reshape(
            dim, -1
        )[:, ind_uptriangle]
        connections = pos_arr2 - pos_arr1
        dist = np.linalg.norm(connections, axis=0)

        ind_close = ind_numb_uptriangle[dist < min_distance]
        indices1 = np.floor(ind_close / n_points).astype(int)
        indices2 = ind_close - indices1 * n_points

        # E_matr = np.
        diag_matr = np.diag([-1, 1])  # inverse along direction
        for ind1, ind2 in zip(indices1, indices2):
            # if not dist[ii]:
            # warnings.warn("Zero dist.")
            direction = positions[:, ind1] - positions[:, ind2]
            mag_dir = np.linalg.norm(direction)
            if not mag_dir:
                warnings.warn("Zero Magnitude")
            else:
                direction = direction / mag_dir

            decomp_matr = np.vstack((direction, [-direction[1], direction[0]])).T

            modul_matr = decomp_matr @ diag_matr @ decomp_matr.T

            if np.dot(self.dynamic_vel[:, ind1], -direction) > 0:
                self.dynamic_vel[:, ind1] = modul_matr @ self.dynamic_vel[:, ind1]
            if np.dot(self.dynamic_vel[:, ind2], direction) > 0:
                self.dynamic_vel[:, ind2] = modul_matr @ self.dynamic_vel[:, ind2]

            for ii, jj in zip([ind1, ind2], [ind2, ind1]):
                if self.sick_status[ii] > 0:
                    if self.sick_status[jj] < 0 and not (
                        self.is_healed[jj] or self.is_vaccinated[jj]
                    ):
                        self.sick_status[jj] = 1

    def verify_wall_collision(self, diameter=0):
        radius = diameter / 2.0

        xlim = ax.get_xlim() + np.array([radius, -radius])
        ylim = ax.get_ylim() + np.array([radius, -radius])
        ind_outside = self.position[0, :] < (xlim[0])
        if any(ind_outside):
            self.position[0, ind_outside] = 2 * xlim[0] - self.position[0, ind_outside]

            # self.dir_velocity = self.refelect_direction_at_edge(self.dir_velocity, 0)
            self.dynamic_vel[0, ind_outside] = -self.dynamic_vel[0, ind_outside]

        ind_outside = self.position[0, :] > xlim[1]
        if any(ind_outside):
            self.position[0, ind_outside] = 2 * xlim[1] - self.position[0, ind_outside]
            # self.dir_velocity = self.refelect_direction_at_edge(self.dir_velocity, pi)
            self.dynamic_vel[0, ind_outside] = -self.dynamic_vel[0, ind_outside]

        ind_outside = self.position[1, :] < ylim[0]
        if any(ind_outside):
            self.position[1, ind_outside] = 2 * ylim[0] - self.position[1, ind_outside]
            # self.dir_velocity = self.refelect_direction_at_edge(self.dir_velocity, pi/2.0)
            self.dynamic_vel[1, ind_outside] = -self.dynamic_vel[1, ind_outside]

        ind_outside = self.position[1, :] > ylim[1]
        if any(ind_outside):
            self.position[1, ind_outside] = 2 * ylim[1] - self.position[1, ind_outside]
            # self.dir_velocity = self.refelect_direction_at_edge(self.dir_velocity, 3*pi/2)
            self.dynamic_vel[1, ind_outside] = -self.dynamic_vel[1, ind_outside]

    def check_people(self):
        pass

    def save_path(self):
        pass

    def import_path():
        pass


# call the animator.  blit=True means only re-draw the parts that have changed.
# try:
if True:
    Simulator = CrowdSimulation(pos_points=pos_points)
    # Simulator = CrowdSimulation()
    anim = animation.FuncAnimation(
        fig,
        Simulator.animate,
        init_func=Simulator.setup,
        frames=1000,
        interval=20,
        blit=True,
    )

    save_anim = True
    if save_anim:
        print("Saving in progress...")
        anim.save("pandemic_animation.mp4", fps=30, extra_args=["-vcodec", "libx264"])
        print("Saving finished. See you next time.")
    else:
        plt.show()
# except:
# print("ERROR")


# Save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
