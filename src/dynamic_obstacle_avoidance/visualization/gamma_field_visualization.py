""" Gammafield evaluation for different Scenarios. """
import numpy as np
import matplotlib.pyplot as plt

def gamma_field_visualization(obstacle, grid_number=30, x_lim=None, y_lim=None,
                              dim=2, fig=None, ax=None):
    """ Draw the gamma of one obstacle. """
    if fig is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)

    else:
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        
    x_vals = np.linspace(x_lim[0], x_lim[1], grid_number)
    y_vals = np.linspace(y_lim[0], y_lim[1], grid_number)

    gamma_values = np.zeros((grid_number, grid_number))
    positions = np.zeros((dim, grid_number, grid_number))

    for ix in range(grid_number):
        for iy in range(grid_number):
            positions[:, ix, iy] = [x_vals[ix], y_vals[iy]]
            
            gamma_values[ix, iy] = obstacle.get_gamma(positions[:, ix, iy], in_global_frame=True)

    cs = ax.contourf(positions[0, :, :], positions[1, :, :],  gamma_values, 
                     np.arange(1.0, 2.0, 0.1),
                     extend='max', alpha=0.6, zorder=-3)

    cbar = fig.colorbar(cs)

