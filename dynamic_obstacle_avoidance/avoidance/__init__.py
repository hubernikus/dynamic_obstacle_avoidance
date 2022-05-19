"""
The :mod:`dynamic_obstacle_avoidance` module implements mixture modeling algorithms.
"""
# Various modulation functions

# from .utils import *   # Added this to avoid 'circular'-error
from .modulation import obs_avoidance_interpolation_moving
from .repulsion_modulation import obs_avoidance_nonlinear_hirarchy
from .comparison_algorithms import (
    obs_avoidance_potential_field,
    obs_avoidance_orthogonal_moving,
)

# Addition classes / functions
from .rk4 import obs_avoidance_rk4, obs_avoidance_rungeKutta

# Avoider Classes
from .modulation import ModulationAvoider
from .base_avoider import BaseAvoider
from .obstacle_avoider import ObstacleAvoiderWithInitialDynamcis
from .dynamic_crowd_avoider import DynamicCrowdAvoider

__all__ = [
    "obs_avoidance_rk4",
    "obs_avoidance_rungeKutta",
    "obs_avoidance_interpolation_moving",
    "obs_avoidance_nonlinear_hirarchy",
    "obs_avoidance_potential_field",
    "obs_avoidance_orthogonal_moving",
    "ObstacleAvoiderWithInitialDynamcis",
    "DynamicCrowdAvoider",
    "ModulationAvoider",
    "BaseAvoider",
]
