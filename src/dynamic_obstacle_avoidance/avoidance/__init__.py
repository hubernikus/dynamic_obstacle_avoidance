"""
The :mod:`dynamic_obstacle_avoidance` module implements mixture modeling algorithms.
"""
# Various modulation functions

from .modulation import obs_avoidance_interpolation_moving
from .repulsion_modulation import obs_avoidance_nonlinear_hirarchy
from .comparison_algorithms import obs_avoidance_potential_field, obs_avoidance_orthogonal_moving
from .rotation import obstacle_avoidance_rotational

from .rk4 import obs_avoidance_rk4, obs_avoidance_rungeKutta

__all__ = ['obs_avoidance_rk4',
           'obs_avoidance_rungeKutta',
           'obs_avoidance_interpolation_moving',
           'obstacle_avoidance_rotational',
           'obs_avoidance_nonlinear_hirarchy',
           'obs_avoidance_potential_field',
           'obs_avoidance_orthogonal_moving',
           ]




