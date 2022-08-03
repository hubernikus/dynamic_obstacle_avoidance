"""
Examples of how to use an obstacle-boundary mix,
i.e., an obstacle which can be entered

This could be bag in task-space, or a complex-learned obstacle.
"""
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-06-21

import numpy as np
import numpy.linalg as LA
import numpy.typing as npt

from dynamic_obstacle_avoidance.obstacles import Obstacle

Vector = npt.ArrayLike


def gamma_normal_gradient_descent(
    obstacles: Obstacle,
    factors: npt.ArrayLike = None,
    powers: npt.ArrayLike = None,
    it_max: int = 50,
    step_factor: float = 0.1,
    convergence_error: float = 1e-1,
) -> Vector:
    """Returns the intersection-position (or a point if it does not exists),
    for two convex input obstacles.

    Arguments
    ---------
    factors: The factor of the direction;
        > 0 if outside obstacle OR inside boundary
        < 0 otherwise
    powers: Power of the weights, chose |powers| > 1 for better convergence; good
        choice for the values is |powers[ii]|=2. Furthermore:
        > 0 if inside obstacle OR inside boundary
        < 0 otherwise
    """
    position = 0.5 * (obstacles[0].center_position + obstacles[1].center_position)

    if powers is None:
        powers = (2 for _ in range(len(obstacles)))

    if factors is None:
        factors = (1 for _ in range(len(obstacles)))

    dimension = obstacles[0].dimension

    for ii in range(it_max):
        step = np.zeros(dimension)

        # Gamma Gradient and Normal direction (?)
        for ii, obs_ii in enumerate(obstacles):
            stepsize = obs_ii.get_gamma(position, in_global_frame=True) ** powers[ii]
            step += (
                stepsize
                * factors[ii]
                * obs_ii.get_normal_direction(position, in_global_frame=True)
            )

        if LA.norm(step) < convergence_error:
            logging.info(f"Gamma gradient converged at it={ii}")
            break

        position += step_factor * step

    return position
