"""
Replicating the paper of
'Safety of Dynamical Systems With MultipleNon-Convex Unsafe Sets UsingControl Barrier Functions'
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
from matplotlib import cm

from vartools.dynamical_systems import DynamicalSystem, LinearSystem
from vartools.dynamical_systems import plot_dynamical_system_streamplot
from vartools.math import get_numerical_gradient, get_numerical_hessian
from vartools.math import get_numerical_hessian_fast

from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.obstacles import Obstacle

from barrier_functions import BarrierFunction, CirclularBarrier, DoubleBlobBarrier

# from cvxopt.modeling import variable
from cvxopt import solvers, matrix

@dataclass
class GammaOperator_g_v():
    """
    g: matrix function output [n x m]
    vector: (v) xvector of dimension [m] / 
    """
    g: Callable
    vector: np.ndarray

    def evaluate(self, position):
        g_x = g.evaluate(position)
        
        # gradient_of_g  => in R^[n x m x n] 
        gradient_of_g = g.get_gradient(position)

        dimension = g_x.shape[0]
        m_dim = g_x.shape[1]

        Gamma = np.zeros((dimension, dimension))
        for ii in range(m_dim):
            Gamma += (g_x[:, ii].T @ vector @ np.eye(dimension) + g_x[:, ii] @ vector.T).dot(
                gradient_of_g[:, ii, :])
        return Gamma
            
class ControlLyapunovFunction():
    def __init__(self, ll=[1, 6]):
        self.ll = ll
        self.dimension = 2

    def get_control_lyapunov_value(self, position):
        return 0.5*np.sum(self.ll*position**2)

    def evaluate_gradient(self, position):
        return self.ll*position

    def get_hessian(self, position):
        return np.diag(self.ll)

@dataclass
class ControlLyapunovQP():
    """
    minimize ||u||^2 + q||w||^2  p delta^2

    s.t. L_f V + L_g V u + ∇_Q V.T ω + γ(V) ≤ δ             (CLF)
         L_f h + L_g h u + α(h) ≥ 0                         (CBF1)
         L_f h_D + L_g h_D u + ∇_Q h_D^T ω + β(h_D) ≥ 0     (CBF2)
    """
    barrier_function: BarrierFunction
    control_lyapunov_function: ControlLyapunovFunction
    p: float = 5
    q: float = 5
    epsilon: float = 0.1

    # TODO:
    # h_D
    # ∇_Q
    # ω / omega: -> virtual control signal
    
    # -> Needs internally
    # Sigma(h) / σ(h) [can kinda guess from the constraints]
    # D(x, Q) & ∇D & ∇_Q D
    # Q(x)
    # O_n (x) (?!)
    # P_f -> scaled orthogonal projection of f
    # P_G -> scaled orthogonal projection of G (? how! matrix..!)
    # Gamma -> matrix / vector function
    #
    # G = g^t g
    def sigma(self, h, delta=1):
        """ Returns σ(h)  s.t.  (i) σ(h) > 0, (ii) σ'(0) = 0 and (iii) lim h→∞ σ(h) = 0."""
        return  np.exp(-0.5*(h/delta)**2)

    
    def extended_class_function(self, barrier_function_value):
        """ Not described in paper - assumption of zero value. 'lambda'-function """
        return barrier_function_value
    

class ClosedLoopQP():
    """
    System of the form:

    dot x = f_x * x + g_x * u

    with a controller to avoid the barrier function h(x)
    
    Properties
    ----------
    f_x
    g_x
    barrier_function
    
    """
    def __init__(self, f_x, g_x, barrier_function: Obstacle):
        self.f_x = f_x
        self.g_x = g_x
        self.barrier_function = barrier_function

    # def set_control(self, control):
        # self._control = control

    def get_optimal_control(self, position):
        gradient = self.barrier_function.evaluate_gradient(position)

        # Lie derivative of h: L_f h(x) / L_g h(x)
        lie_of_h_wrt_f = gradient.dot(self.evaluate_base_dynamics(position))
        lie_of_h_wrt_g = gradient.dot(self.evaluate_control_dynamics(position))

        gamma_of_h = self.extended_class_function(
            self.barrier_function.get_barrier_value(position))

        # Create QP-solver of the form
        # min ( 1/2 x.T P x + q.T x ) 
        # s.t G x < h
        #     A x = b
        #
        # sol = solver.qp(P, q, G, h)
        # sol = solver.qp(P, q, G, h, A, b)
        P = matrix([[1.0]])
        q = matrix([0.0])

        G = (-1)*matrix([lie_of_h_wrt_g])
        h = matrix([[lie_of_h_wrt_f + gamma_of_h]])
        
        sol = solvers.qp(P, q, G, h)
        return sol['x']

    def extended_class_function(self, barrier_function_value):
        """ Not described in paper - assumption of zero value. 'lambda'-function """
        return barrier_function_value
    
    def evaluate(self, position):
        """ Evaluate with 'temporary control'."""
        control = self.get_optimal_control(position)
        return self.evaluate_with_control(position, control=control)
    
    def evaluate_base_dynamics(self, position):
        return self.f_x.evaluate(position)
    
    def evaluate_control_dynamics(self, position):
        return self.g_x.evaluate(position)
    
    def evaluate_with_control(self, position, control):
        """ Controller acts on internal dynamics. """
        return self.f_x.evaluate(position) + self.g_x.evaluate(position)*control


def plot_integrate_trajectory(delta_time=0.005, n_steps=1000):
    # start_position = [-4, 4]
    # start_position = [4, 4]
    x_lim = [-5, 5]
    y_lim = [-2, 6.5]

    dimension = 2
    
    f_x = LinearSystem(
        A_matrix=np.array([[-6, 0],
                           [0, -1]])
        )
    g_x = LinearSystem(A_matrix=np.eye(dimension))

    # barrier_function = DoubleBlobBarrier(
        # blob_matrix=np.array([[10.0, 0.0],
                              # [0.0, -1.0]]),
        # center_position=np.array([0.0, 3.0]))

    barrier_function = CirclularBarrier(
        radius=1.0,
        center_position=np.array([0, 3]),
        )

    dynamics = ClosedLoopQP(f_x=f_x, g_x=g_x, barrier_function=barrier_function)

    start_position_list = [
        [4, 4],
        [-4, 4]
        ]
    
    fig, ax = plt.subplots(figsize=(7.5, 6))

    for start_position in start_position_list:
        position = np.zeros((dimension, n_steps+1))
        position[:, 0] = start_position
        for ii in range(n_steps):
            vel = dynamics.evaluate(position[:, ii])
            position[:, ii+1] = position[:, ii] + vel*delta_time
            
        ax.plot(position[0, :], position[1, :])
    # ax.plot(barrier_function.center_position[0], barrier_function.center_position[1], 'k*')
    
    ax.plot(0, 0, 'k*')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.grid()
    
    
def plot_main_vector_field():
    dimension = 2
    f_x = LinearSystem(
        A_matrix=np.array([[-6, 0],
                           [0, -1]])
        )
    g_x = LinearSystem(A_matrix=np.eye(dimension))

    # closed_loop_ds = ClosedLoopQP(f_x=f_x, g_x=g_x)
    # closed_loop_ds.evaluate_with_control(position, control)

    plot_dynamical_system_streamplot(
        dynamical_system=f_x, x_lim=[-10, 10], y_lim=[-10, 10])


def plot_barrier_function():
    fig, ax = plt.subplots(figsize=(7.5, 6))
    
    x_lim = [-5, 5]
    y_lim = [-2, 6]

    n_grid = 100
    
    x_vals, y_vals = np.meshgrid(np.linspace(x_lim[0], x_lim[1], n_grid),
                                 np.linspace(y_lim[0], y_lim[1], n_grid),
                                 )
    
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    values = np.zeros(positions.shape[1])

    barrier_function = DoubleBlobBarrier(
        blob_matrix=np.array([[10, 0],
                              [0, -1]]),
        center_position=np.array([0, 3]))

    barrier_function = CirclularBarrier(
        radius=1.0,
        center_position=np.array([0, 3])
        )
    for ii in range(positions.shape[1]):
        values[ii] = barrier_function.get_barrier_value(positions[:, ii])

    cs = ax.contourf(positions[0, :].reshape(n_grid, n_grid),
                    positions[1, :].reshape(n_grid, n_grid),
                    values.reshape(n_grid, n_grid),
                    np.linspace(-10.0, 0.0, 11),
                    # np.linspace(-10.0, 0.0, 2),
                    # vmin=-0.1, vmax=0.1,
                    # np.linspace(-10, 10.0, 101),
                    # cmap=cm.YlGnBu,
                    # linewidth=0.2, edgecolors='k'
                    )
    
    cbar = fig.colorbar(cs,
                        # ticks=np.linspace(-10, 0, 11)
                        )

    plt.grid()
    ax.set_aspect('equal', adjustable='box')


def compare_gradients():
    """ Similar to the testing-script from the 'various-tools' library. """
    blob_obstacle = DoubleBlobBarrier(
        blob_matrix=np.array([[10, 0],
                              [0, -1]]),
        center_position=np.array([0, 3]))
    
    positions = np.array(
        [[2,-3],
         [0, 0],
         [1, 1]]
        ).T
    
    for it_pos in range(positions.shape[1]):
        position = positions[:, it_pos]
        
        gradient_analytic = blob_obstacle.evaluate_gradient(position=position)
        gradient_numerical = get_numerical_gradient(
            position=position,
            function=blob_obstacle.get_barrier_value, delta_magnitude=1e-6)

        assert(np.allclose(gradient_analytic, gradient_numerical, rtol=1e-5))

        print(f"Gradient Analytic: {gradient_analytic}")
        print(f"Gradient Numerical: {gradient_numerical}")


def compare_get_numerical_hessian():
    barrier_function = CirclularBarrier(radius=1.5)

    position = np.array([0, 0])
    hessian_numerical = get_numerical_hessian(
        function=barrier_function.get_barrier_value, position=position)                             
    hessian_numerical_fast = get_numerical_hessian_fast(
        function=barrier_function.get_barrier_value, position=position)                             
        
    hessian_analytical = barrier_function.get_hessian(position=position)
    hessian_analytical = barrier_function.get_hessian(position=position)

    print("is close here", np.allclose(hessian_numerical, hessian_analytical, rtol=1e-4))

    print(f"{hessian_numerical=}")
    print(f"{hessian_analytical=}")
    print(f"{hessian_numerical_fast=}")
    # compare speed
    

if (__name__) == "__main__":
    plt.ion()
    solvers.options['show_progress'] = False
    # plot_main_vector_field()
    
    # plot_barrier_function()
    plot_integrate_trajectory()
    
    # compare_gradients()
    # compare_get_numerical_hessian()

    plt.show()
