"""
The :mod:`obstacles` module implements various types of obstacles.
"""
# Various Obstacle Descriptions
from ._base import Obstacle
from .ellipse import Ellipse, CircularObstacle
from .polygon import Polygon, Cuboid
from .flower import StarshapedFlower
from .human_ellipse import TrackedPedestrian, HumanEllipse
from .boundary_cuboid_with_gap import BoundaryCuboidWithGaps

__all__ = ['Obstacle',
           'Ellipse',
           'CircularObstacle',
           'Cuboid',
           'Polygon',
           'StarshapedFlower',
           ]

