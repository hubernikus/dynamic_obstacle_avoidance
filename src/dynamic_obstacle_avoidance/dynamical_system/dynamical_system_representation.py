'''

Library of different dynamical systems

@author Lukas Huber
@date 2018-02-15
'''

import numpy as np
import numpy.linalg as LA


def linearAttractor(x, x0='default'):
    # change initial value for n dimensions

    dim = x.shape[0]

    if type(x0)==str and x0=='default':
        x0 = dim*[0]
    
    #M = x.shape[1]
    M= 1
    X0 = np.kron( np.ones((1,M)), x0 )

    xd = -(x-x0)
    
    return xd

def linearAttractor_const(x, x0 = 'default', velConst=0.3, distSlow=0.01):
    # change initial value for n dimensions
    # TODO -- constant velocity // maximum velocity
    
    dx = x0-x
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


def constVelocity(dx, x, x0=[0,0], velConst = 0.4, distSlow=0.01):
    dx_mag = np.sqrt(np.sum(np.array(dx)**2))
    
    if dx_mag: # nonzero value
        dx = min(1, 1/dx_mag)*velConst*dx

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

