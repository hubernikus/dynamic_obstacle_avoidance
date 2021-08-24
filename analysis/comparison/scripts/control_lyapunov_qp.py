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
from vartools.math import get_numerical_gradient_of_vectorfield
from vartools.math import get_numerical_hessian_fast
from vartools.math import get_scaled_orthogonal_projection

from vartools.states import ObjectPose

from dynamic_obstacle_avoidance.obstacles import Obstacle

from _base_qp import ControllerQP
from barrier_functions import BarrierFunction, CirclularBarrier, DoubleBlobBarrier

# Add folder to path
# import sys
# path_qp = os.path.join(
    # "/home/lukas/Code/dynamic_obstacle_avoidance", "analysis", "comparison_qp")
# if not path_qp in sys.path:
    # sys.path.append(path_qp)
    
# from cvxopt.modeling import variable
from cvxopt import solvers, matrix

def O_n(vector):
    """ cross-product operator (?!) ^w x = O_n(x) w
    Rotation based on lie group"""
    if vector.shape[0] == 2:
        return np.array([[-vector[1]],
                         [ vector[0]]])
    else:
        raise NotImplementedError("Warning not defined for higher dimenions.")

class ControlDynamics():
    pass

class StaticControlDynamics(ControlDynamics):
    def __init__(self, A_matrix):
        self.A_matrix = np.array(A_matrix)

    @property
    def dimension(self):
        return self.A_matrix.shape[1]
        
    def evaluate(self, position):
        return self.A_matrix
        

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

    def evaluate(self, position):
        return self.get_control_lyapunov_value(position)
    
    def get_control_lyapunov_value(self, position):
        return 0.5*np.sum(self.ll*position**2)

    def evaluate_gradient(self, position):
        return self.ll*position

    def get_hessian(self, position):
        return np.diag(self.ll)

class ControlLyapunovQP(ControllerQP):
    """
    minimize ||u||^2 + q||w||^2  p delta^2

    s.t. L_f V + L_g V u + ∇_Q V.T ω + γ(V) ≤ δ             (CLF)
         L_f h + L_g h u + α(h) ≥ 0                         (CBF1)
         L_f h_D + L_g h_D u + ∇_Q h_D^T ω + β(h_D) ≥ 0     (CBF2)
    """
    def __init__(self, f_x, g_x, barrier_function: BarrierFunction,
                 control_lyapunov_function: ControlLyapunovFunction,
                 p: float = 5, q: float = 5, epsilon: float = 0.1):
        super().__init__(f_x=f_x, g_x=g_x, barrier_function=barrier_function)
        self.control_lyapunov_function = control_lyapunov_function
        self.p = p
        self.q = q
        self.epsilon = epsilon

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

    def get_optimal_control(self, position):
        h_value = self.barrier_function.evaluate(position)
        gradient_h = self.barrier_function.evaluate_gradient(position)
        hessian_h = self.barrier_function.get_hessian(position)

        control_lyapunov = self.control_lyapunov_function.evaluate(position)
        gradient_V = self.control_lyapunov_function.evaluate_gradient(position)
        hessian_V = self.control_lyapunov_function.get_hessian(position)
        
        # f(x) in R^[dimension]
        base_dynamics = self.evaluate_base_dynamics(position)
        f_x = base_dynamics
        grad_f = get_numerical_gradient_of_vectorfield(
            position=position, function=self.evaluate_base_dynamics)

        # g(x) in R^[dimension x input]
        control_dynamics= self.evaluate_control_dynamics(position)
        GG = control_dynamics @ control_dynamics.T

        gamma_g_grad_h = self.gamma_g_v(position=position,
                                        vector=gradient_h,
                                        matrix_function=self.g_x)
        
        gamma_g_grad_V = self.gamma_g_v(position=position,
                                        vector=gradient_V,
                                        matrix_function=self.g_x)
        
        P_f = get_scaled_orthogonal_projection(f_x)
        P_G_grad_h = get_scaled_orthogonal_projection(GG @ gradient_h)
        P_G_grad_V = get_scaled_orthogonal_projection(GG @ gradient_V)
        
        D_value = 1.0/2.0 * gradient_V.T @ GG @ (P_f + P_G_grad_h) @ GG @ gradient_V
        
        grad_D = ((hessian_h @ GG + gamma_g_grad_V.T) @ (P_f + P_G_grad_h) @ GG @ gradient_V
                  + (hessian_h @ GG + gamma_g_grad_h.T) @ P_G_grad_V @ GG @ gradient_h
                  + grad_f.T @ P_G_grad_V @ f_x)
        
        grad_Q_D = ((hessian_h @ O_n(position) - O_n(gradient_V)).T @ GG @ (P_f + P_G_grad_h)
                    @ GG @ gradient_V)

        h_D = self.sigma(h_value) * (D_value - self.epsilon)
        grad_h_D = (self.sigma(h_value) * grad_D + self.derivative_of_sigma(h_value)
                    * (D_value - self.epsilon) * gradient_h)
        grad_q_h_D = self.sigma(h_value) * grad_Q_D

        # TODO: how to fined V_r?
        # it is by evaluating 'virtual' rotation
        grad_V = Q.T @ grad_V_r
        grad_Q_V = O_n(position).T @ grad_V

        # x = [u1, u2, w, delta]
        dimension_opt = 4
        P = matrix(np.diag([1, 1, self.q, self.p]))
        q = matrix(np.zeros((dimension_opt, 1)))
        
        # CLF
        lie_of_V_wrt_f = gradient_V.dot(base_dynamics)
        lie_of_V_wrt_g = gradient_V.dot(control_dynamics)
        # Delta_Q (?)
        gamma_v = self.gamma_function(control_lyapunov)

        G_CLF = np.zeros([dimension_opt])
        G_CLF[0:2] = lie_of_V_wrt_g
        G_CLF[2] = grad_Q_V.T
        G_CLF[3] = (-1)
        
        h_CLF = (-1)*(lie_of_V_wrt_f + gamma_v)

        # CBF1
        lie_of_h_wrt_f = gradient_h.dot(control_dynamics)
        lie_of_h_wrt_g = gradient_h.dot(base_dynamics)
        alpha_h = alpha_function(h_value)

        G_CBF1 = np.zeros([dimension_opt])
        G_CBF1[0:2] = (-1)*lie_of_h_wrt_g
        h_CBF1 = alpha_h + lie_of_h_wrt_f

        # CBF2
        lie_of_hD_wrt_f = grad_h_D.dot(control_dynamics)
        lie_of_hD_wrt_g = grad_h_D.dot(base_dynamics)
        # grad_q_h_D_w =
        beta_h_D = self.beta_function(h_D)
        
        G_CBF2 = np.zeros([dimension_opt])
        G_CBF2[0:2] = (-1)*lie_of_hD_wrt_g
        G_CBF2[2] = (-1)*grad_q_h_D_w
        h_CBF2 = lie_of_hD_wrt_f + beta_h_D

        G = np.vstack((G_CLF, G_CBF1, G_CBF2))
        h = np.array(h_CLF, h_CBF1, h_CBF2)

        sol = solvers.qp(P, q, G, h)
        return sol['x'][0:2]

    def gamma_g_v(self, position, vector, matrix_function):
        """
        position/ pos
        vector/ v
        matrix_function / g(x)

        Returns
        -------
        np.array of the form [n x x]
        """
        # Derivative of function
        if isinstance(matrix_function, StaticControlDynamics):
            # zero value
            dim = matrix_function.dimension
            return np.zeros((dim, dim))

        breakpoint()

        # deriv_g = self.get_derivative_of_dynamics(position, matrix_function)
        # g_x = # g_x = matrix_function.evaluate(position)

        if not LA.norm(deriv_g): # zero matrix
            return np.zeros()

        nn = vector.shape[0]
        gamma_value = np.zeros((nn, nn))
        for ii in range(nn):
            gamma_value += ((g_x[:, ii] @ vector @ np.eye(nn)
                            + g_x[:, ii].reshape(1, nn) @ vector.reshape(nn, 1))
                            @ deriv_g[:, ii, :])
        return gamma_value
        
    def sigma(self, h, delta=1):
        """ Returns σ(h)  s.t.  (i) σ(h) > 0, (ii) σ'(0) = 0 and (iii) lim h→∞ σ(h) = 0."""
        # No specific function given -> interpretation
        return  np.exp(-0.5*(h/delta)**2)

    def derivative_of_sigma(self, h, delta=1, delta_pos=1e-6):
        return (self.sigma(h+delta_pos/2.0)-self.sigma(h-delta_pos/2.0))/delta_pos

    def extended_class_function(self, barrier_function_value):
        """ Not described in paper - assumption of zero value. 'lambda'-function """
        return barrier_function_value

    def alpha_function(self, value):
        return value

    def beta_function(self, value):
        return value
    
    def gamma_function(self, value):
        return value
    

def plot_integrate_trajectory(delta_time=0.005, n_steps=1000):
    # start_position = [-4, 4]
    # start_position = [4, 4]
    x_lim = [-5, 5]
    y_lim = [-2, 6.5]

    dimension = 2

    f_x = LinearSystem(A_matrix=np.zeros((dimension, dimension)))
    g_x = StaticControlDynamics(A_matrix=np.eye(dimension))
    # g_x = LinearSystem(A_matrix=np.eye(dimension))

    barrier_function = CirclularBarrier(
        radius=1.5,
        center_position=np.array([0, 3]),
        )

    control_lyapunov_function = ControlLyapunovFunction(ll=[6, 1])

    dynamics = ControlLyapunovQP(
        f_x=f_x,
        g_x=g_x,
        barrier_function=barrier_function,
        control_lyapunov_function=control_lyapunov_function,
        p=5, q=5, epsilon=0.1)

    start_position_list = [
        # [4, 4],
        [-4, 6]
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
    # g_x = LinearSystem(A_matrix=np.eye(dimension))

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


if (__name__) == "__main__":
    plt.ion()
    solvers.options['show_progress'] = False
    # plot_main_vector_field()
    
    # plot_barrier_function()
    plot_integrate_trajectory()

    plt.show()
