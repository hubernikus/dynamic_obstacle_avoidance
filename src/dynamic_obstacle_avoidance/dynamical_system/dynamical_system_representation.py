'''

Library of different dynamical systems

@author Lukas Huber
@date 2018-02-15
'''

import numpy as np
import numpy.linalg as LA

# TODO: clean up and restructure files.

def linear_ds(position, attractor=None):
    ''' Linear Dynamical System'''
    if attractor is None:
        return (-1)*position
    else:
        return attractor-position

def linear_ds_max_vel(position, attractor=np.array([0,0]), max_vel=0.5, slow_down_region=0.5):
    ''' Linear Dynamical System with decreasing velocity close to the attractor,
    but constant (maximal) velocity, everywhere else.'''
    velocity = attractor-position

    distance = np.linalg.norm(attractor-position)
    if distance < slow_down_region:
        max_vel = max_vel*distance/slow_down_region
        
    norm_vel = velocity
    if norm_vel>max_vel:
        velocity = velocity/norm_vel*max_vel

    return velocity


def limit_velocity(velocity, position, final_position, max_vel=0.07, slow_down_dist=0.1):
    ''' Limit velocity with convergence to 0 around the final position.'''
    dist = final_position-position
    dist_norm = np.linalg.norm(dist)
    vel_norm = np.linalg.norm(velocity)

    if not dist_norm or not vel_norm:
        vel = np.zeros(3)
    elif dist_norm < slow_down_dist:
        vel = velocity/vel_norm*max_vel*dist_norm/slow_down_dist
    else:
        vel = velocity/vel_norm*max_vel
    return vel



def linearAttractor(x, x0=None):
    # change initial value for n dimensions

    dim = x.shape[0]

    if x0 is None:
        x0 = np.zeros(dim)
    
    #M = x.shape[1]
    M= 1
    X0 = np.kron( np.ones((1,M)), x0 )

    xd = -(x-x0)
    
    return xd

def linearAttractor_const(x, x0 = 'default', v_ref=0, velConst=0.3, distSlow=0.01):
    # change initial value for n dimensions
    # TODO -- constant velocity // maximum velocity
    
    dx = x0-x + v_ref
    dx_mag = np.sqrt(np.sum(dx**2))
    
    dx = min(velConst, 1/dx_mag*velConst)*dx

    return dx
        

def nonlinear_wavy_DS(x, x0=[0,0]):

    xd = np.zeros((np.array(x).shape))
    if len(xd.shape)>1:
        xd[0,:] = - x[1,:] * np.cos(x[0,:]) - x[0,:]
        xd[1,:] = - x[1,:]
    else:
        xd[0] = - x[1] * np.cos(x[0]) - x[0]
        xd[1] = - x[1]
    return xd


def nonlinear_stable_DS(x, x0=[0,0], pp=3 ):
    xd = np.zeros((np.array(x).shape))
    if len(xd.shape)>1:
        xd[0,:] = - x[1,:]
        xd[1,:] = - np.copysign(x[1,:]**pp, x[1,:])
    else:
        xd[0] = - x[0]
        xd[1] = - np.copysign(np.abs(x[1])**pp, x[1])
    return xd


def constVelocity_distance(dx, x, x0=[0,0], velConst = 1.0, distSlow=0.1):
    #return dx
    dx_mag = LA.norm(dx)
    if not dx_mag:
        return dx
    dx = dx/dx_mag
            
    xt_mag = LA.norm(x-x0)
    
    return np.min([1, xt_mag/distSlow])*velConst*dx

def make_velocity_constant(vel, position, position_attractor, constant_velocity=0.3, slowing_down_radius=0.01):
    vel_magn = np.linalg.norm(vel)

    if vel_magn:
        dist2attr = np.linalg.norm(position-position_attractor)
        if dist2attr < slowing_down_radius:
            constant_velocity = constant_velocity*dist2attr/slowing_down_radius
        vel = vel/vel_magn * constant_velocity
        
    return vel

def constVelocity(dx, x, x0=[0,0], velConst=0.4, distSlow=0.01):
    dx_mag = np.linalg.norm(dx)

    if dx_mag: # nonzero value
        dist_attr = np.linalg.norm(x-x0)
        if dist_attr < distSlow:
            velConst = velConst*dist_attr/distSlow

        dx = dx/dx_magn*velConst
        # dx = min(1, 1/dx_mag)*velConst*dx

    return dx


def constVel(xd, const_vel=2.0):
    # dim = np.array(xd).shape[0]
    
    xd_norm = np.sqrt(np.sum(xd**2))
    if xd_norm==0: return xd

    return xd/xd_norm*const_vel


def linearDS_constVel(x, x_attr=None, const_vel=2.0, A=None, x0=None):
    dim = np.array(x).shape[0]
    
    x_shape = x.shape
    x = np.squeeze(x.reshape(dim, 1,-1))
    
    if type(x_attr)==type(None):
        if type(x0) == type(None):
            x_attr = np.zeros(dim)
        else:
            x_attr = x0
    
    if len(x.shape)==1: # in case of 1D input array
        x = np.tile(x, (1,1)).T

    xd = -(x - np.tile(x_attr, (x.shape[1], 1)).T)

    if type(A)!=type(None):
        # Higher dimensional matrix multiplication
        xd = A.dot(xd) 
    
    xd_norm = LA.norm(xd,axis=0)
        
    xd[:, xd_norm>0] = xd[:, xd_norm>0]/np.tile(xd_norm[xd_norm>0], (dim,1))*const_vel
    
    return xd.reshape(x_shape)

    
def velConst_attr(x, vel, x0=False, velConst=6, distSlow=0.5):
    # change initial value for n dimensions
    # TODO -- constant velocity // maximum velocity
    if type(x0)==bool:
        dim = np.array(x).shape[0]
        x0 = np.zeros(dim)
        
    delta_x = x0-x
    dist_mag = np.sqrt(np.sum(delta_x**2))
    if dist_mag: # nonzero value
        new_mag = np.min([velConst, dist_mag/distSlow*velConst])


    vel_mag = np.sqrt(np.sum(vel**2))
    if vel_mag:
        vel = vel/vel_mag*new_mag
    
    return vel




