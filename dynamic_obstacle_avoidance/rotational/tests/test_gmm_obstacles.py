#!/USSR/bin/python3
""" Sample the space and decide if points are collision-full or free. """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-06-08
import logging

import numpy as np
from numpy import linalg

from sklearn.mixture import GaussianMixture

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse

from dynamic_obstacle_avoidance.rotational.gmm_obstacle import MultiGmmObstacle


def test_simple_data():
    pass


def test_same_obstacle_from_gmm():
    """ Test to verify the consistent behavior between the Gmm-obstacle and the corresponding
    analytic one."""
    dimension = 2
    
    my_obstacle = MultiGmmObstacle(n_gmms=1)
    my_obstacle._gmm = GaussianMixture(n_components=my_obstacle.n_gmms)
    my_obstacle._gmm.means_ = [np.ones(dimension)]
    my_obstacle._gmm.covariances_ = [np.eye(dimension)]
    my_obstacle._gmm.weights = [1]

    simple_ellipses = my_obstacle.transform_to_analytic_ellipses()

    breakpoint()
    pass


def test_gradient_descent():
    pass


if (__name__) == "__main__":
    pass
