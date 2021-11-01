"""
The :mod:`obstacles` module implements various types of obstacles.
"""
# Various Obstacle Descriptions
from ._base import Obstacle, GammaType
from .ellipse import Ellipse, CircularObstacle, Sphere
from .polygon import Polygon
from .cuboid import Cuboid
from .cross import Cross
from .flower import StarshapedFlower
from .human_ellipse import TrackedPedestrian, HumanEllipse
from .boundary_cuboid_with_gap import BoundaryCuboidWithGaps
from .flat_plane import FlatPlane
from .double_blob_obstacle import DoubleBlob


__all__ = [
    "Obstacle",
    "Ellipse",
    "Sphere",
    "CircularObstacle",
    "Cuboid",
    "Cross",
    "Polygon",
    "StarshapedFlower",
    "FlatPlane",
    "DoubleBlob",
    "GammaType",
]
