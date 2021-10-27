"""
Functions for the evluation of the hessian

NOTE: this class requires the sympy module for jacobian evaluation
      (if desired -> move it somewhere else)
"""

from robot_avoidance.analytic_evaluation_jacobian import (
    analytic_evaluation_jacobian,
)

if (__name__) == "__main__":
    # analytic_evaluation_jacobian(robot=ModelRobot2D())

    plt.close("all")
    plt.ion()

    pass
