import numpy as np
import matplotlib.pyplot as plt


def ds_sinus(
    pos, x_position_attractor=6, sinus_magnitude=2, maximum_velocity=0.5, y_scaling=2
):
    """2D dynamical system which follows sinus wave form

    Paramteter input
    position: np.array of shape (2.)
    x_position: float"""

    desired_y = np.sin(pos[0] / x_position_attractor * pi) * sinus_magnitude

    vel = np.zeros(2)
    vel[1] = desired_y - pos[1]
    vel[0] = x_position_attractor - pos[0]

    if np.abs(vel[0]) > maximum_velocity:
        vel[0] = np.copysign(maximum_velocity, vel[0])

    vel[1] = vel[1] * y_scaling

    # Normalize
    norm_vel = np.linalg.norm(vel)
    if norm_vel > maximum_velocity:
        vel = vel / norm_vel * maximum_velocity

    return vel


dim = 2  # space dimension
x_range = [-1, 7]
y_range = [-3, 3]

attractor_pos = np.array([6, 0])

n_grid = 20

# x_grid = np.linspace(x_range[0], x_range[1], n_grid)
# y_grid = np.linspace(y_range[0], y_range[1], n_grid)

num_x = num_y = n_grid
YY, XX = np.mgrid[
    y_range[0] : y_range[1] : num_y * 1j, x_range[0] : x_range[1] : num_x * 1j
]

pos = np.dstack((XX, YY))
vel = np.zeros((n_grid, n_grid, dim))

for ix in range(n_grid):
    for iy in range(n_grid):
        vel[ix, iy, :] = ds_sinus(pos[ix, iy, :], attractor_pos[0], 2)


# Draw plot
plt.ion()
# fig, ax = plt.figure()
plt.figure()
plt.quiver(XX, YY, vel[:, :, 0], vel[:, :, 1])
plt.plot(attractor_pos[0], attractor_pos[1], "k*", markeredgewidth=4, markersize=13)
plt.plot(0, 0, "k.", markeredgewidth=4, markersize=13)


print("The code finished correctly.")
