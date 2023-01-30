"""
Geometry tools for intersection etc.
"""
import math
from typing import Optional

import numpy as np
from numpy import linalg as LA

from shapely import affinity
from shapely.geometry.point import Point


from dynamic_obstacle_avoidance.obstacles import Obstacle, Ellipse, EllipseWithAxes


def create_shapely_ellipse(ellipse: Obstacle):
    """Create object (shape) based on the shapely library."""
    if ellipse.dim != 2:
        raise NotImplementedError("Shapely object only existing for 2D")

    # Point is set at zero, and only moved when needed
    if ellipse.dimension != 2:
        raise NotImplementedError()

    shapely_ellipse = Point(np.zeros(ellipse.dimension)).buffer(1)

    if isinstance(ellipse, Ellipse):
        axes = ellipse.axes_length + ellipse.margin_absolut

    elif isinstance(ellipse, EllipseWithAxes):
        axes = ellipse.axes_length * 0.5 + ellipse.margin_absolut
        # axes = ellipse.axes_length + ellipse.margin_absolut
    else:
        raise NotImplementedError()

    shapely_ellipse = affinity.scale(shapely_ellipse, axes[0], axes[1])
    shapely_ellipse = affinity.rotate(
        shapely_ellipse, ellipse.orientation * 180 / math.pi
    )

    shapely_ellipse = affinity.translate(
        shapely_ellipse, ellipse.center_position[0], ellipse.center_position[1]
    )

    return shapely_ellipse


def get_intersection_of_obstacles(
    obstacle1: Obstacle, obstacle2: Obstacle
) -> Optional[np.ndarray]:
    """Get the intersection betweeen two obstacles contained in the list.
    The intersection is numerically based on the drawn points."""
    if isinstance(obstacle1, Ellipse) or isinstance(obstacle1, EllipseWithAxes):
        shape1 = create_shapely_ellipse(obstacle1)

    else:
        obstacle1.create_shape()
        shape1 = obstacle1.shape

    if isinstance(obstacle1, Ellipse) or isinstance(obstacle1, EllipseWithAxes):
        shape2 = create_shapely_ellipse(obstacle2)
    else:
        obstacle2.create_shape()
        shape2 = obstacle2.shape

    intersect = shape1.intersection(shape2)
    intersections = np.array(intersect.exterior.coords.xy)

    if not intersections.shape[1]:
        return None

    intersection = np.mean(intersections, axis=1)
    return intersection
