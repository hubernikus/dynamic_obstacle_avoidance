""" Library for the Rotation (Modulation Imitation) of Linear Systems
"""
# Author: Lukas Huber
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

# TODO: Speed up using cython / cpp e.g. eigen?
# TODO: list / array stack for lru_cache to speed

import warnings
from .rotational_avoider import RotationalAvoider


def obstacle_avoidance_rotational(*args, **kwargs):
    warnings.warn("Depreciated - switch to 'Avoider'.")

    _avoider = RotationalAvoider()
    return _avoider.avoid(*args, **kwargs)
