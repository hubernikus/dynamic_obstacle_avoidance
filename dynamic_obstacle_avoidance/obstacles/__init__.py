"""
The :mod:`obstacles` module implements various types of obstacles.
"""
# Various Obstacle Descriptions
from ._base import Obstacle, GammaType
from ._base import get_intersection_position
from .ellipse import Ellipse, Sphere, CircularObstacle
from .polygon import Polygon
from .cuboid import Cuboid
from .cross import Cross
from .flower import StarshapedFlower
from .human_ellipse import TrackedPedestrian, HumanEllipse
from .boundary_cuboid_with_gap import BoundaryCuboidWithGaps
from .flat_plane import FlatPlane
from .double_blob_obstacle import DoubleBlob

# Multidimensional Obstacles
from .cuboid_xd import CuboidXd
from .ellipse_xd import EllipseWithAxes
from .hyper_shpere import HyperSphere


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
    "CuboidXd",
    "EllipseWithAxes",
    "HyperSphere",
    "get_intersection_position",
]
