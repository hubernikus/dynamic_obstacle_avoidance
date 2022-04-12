"""
Cross-Like Obstacle
"""
# Author: Lukas Huber
# Date: created: 2021-10-21
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import numpy as np
from numpy import linalg as LA

# import shapely

from vartools.angle_math import *

from ._base import GammaType
from .polygon import Polygon


class Cross(Polygon):
    def __init__(
        self,
        axes_length: np.ndarray = [1.0, 1.0],
        branch_width: np.ndarray = [0.2, 0.2],
        *args,
        **kwargs
    ):
        """
        This class defines obstacles to modulate the DS around it
        At current stage the function focuses on Ellipsoids,
        but can be extended to more general obstacles
        """
        axes_length = np.array(axes_length) * 0.5
        branch_width = np.array(branch_width) * 0.5
        edge_points = [
            [axes_length[0], -branch_width[1]],
            [axes_length[0], branch_width[1]],
            [branch_width[0], branch_width[1]],
            [branch_width[0], axes_length[1]],
            [-branch_width[0], axes_length[1]],
            [-branch_width[0], branch_width[1]],
            [-axes_length[0], branch_width[1]],
            [-axes_length[0], -branch_width[1]],
            [-branch_width[0], -branch_width[1]],
            [-branch_width[0], -axes_length[1]],
            [branch_width[0], -axes_length[1]],
            [branch_width[0], -branch_width[1]],
        ]

        super().__init__(
            *args,
            edge_points=np.array(edge_points).T,
            absolute_edge_position=False,
            **kwargs
        )
