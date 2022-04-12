#!/USSR/bin/python3

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import time

from dynamic_obstacle_avoidance.obstacle_avoidance.dynamic_boundaries_polygon import (
    DynamicBoundariesPolygon,
)

obs = DynamicBoundariesPolygon(is_surgery_setup=True)

# Resolution
n_steps = int(2e1)

# Setup plot
fig = plt.figure(figsize=plt.figaspect(0.8))
ax = fig.add_subplot(1, 1, 1, projection="3d")


while True:
    # bdya = 0*[1e-2, 3e-2] .* rand([4,1])
    random_params = np.random.rand(4)
    random_params = np.zeros(4)
    # random_params = [0.12, 0.234, 0.04, 0.9]
    print("Random", np.round(random_params, 2))

    # Parameters
    ya_init = np.array([1e-2, 3e-2])

    ya = np.copy(ya_init)
    dya = np.tile(ya, (obs.n_planes, 1)) * np.tile(random_params, (2, 1)).T
    # dya = np.tile([1e-2, 3e-2], (n_planes, 1)) * np.tile(np.linspace(0, 1, 4), (2,1)).T

    # Side plates bottom part // specific movement
    # dya[1, 0] = dya[0, 0] + dya[2, 0]
    # dya[3, 0] = dya[1, 0]

    for it_plane in range(obs.n_planes):
        # for it_plane in range(2):
        x = np.zeros((n_steps, n_steps))
        y = np.zeros((n_steps, n_steps))
        z = np.zeros((n_steps, n_steps))

        z_vals = np.linspace(0, obs.height, n_steps)
        # now = time.time()
        for it_z in range(n_steps):
            z[:, it_z] = np.ones(n_steps) * z_vals[it_z]
            wz = (obs.width[1] - obs.width[0]) / obs.height * z_vals[it_z] + obs.width[
                0
            ]

            x_vals = np.linspace(-wz / 2, wz / 2, n_steps)
            y_vals = np.linspace(-wz / 2, wz / 2, n_steps)

            for it_xy in range(n_steps):
                position = np.array([x_vals[it_xy], y_vals[it_xy], z[it_xy, it_z]])
                position = obs.get_surface_position(
                    position,
                    inflation_parameter=dya[it_plane, :],
                    plane_index=it_plane,
                    in_global_frame=True,
                )

                x[it_xy, it_z] = position[0]
                y[it_xy, it_z] = position[1]

        # dt = time.time() - now
        # print('Evaluation of {} points took {} ms'.format(n_steps*n_steps, round(dt*1000,2)))

        surf = ax.plot_surface(
            x,
            y,
            z,
            rstride=1,
            cstride=1,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
            alpha=0.7,
        )

        print("plane it={}".format(it_plane))
        print("min_x", np.min(x))
        print("max_x", np.max(x))
        print("min_y", np.min(y))
        print("max_y", np.max(y))

    # view([-1 -1 2.2])
    # ax.set_xticks([]);  ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_aspect("equal")

    plt.pause(5e-1)

    if not len(plt.get_fignums()):
        print()
        print("Animation ended through closing of figures.")
        break

    ax.cla()  # Clear axes

plt.show()
