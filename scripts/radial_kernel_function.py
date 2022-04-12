#!/usr/bin/python3

"""
@author LukasHuber
@date 2019-05-24
"""

import sys
import numpy as np
from numpy import pi

import matplotlib.pyplot as plt

plt.close("all")
plt.ion()

from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


def get_radius_of_angle(
    angle, radius_mean=0.6, radius_magnitude=0.2, number_of_edges=5, orientation=0
):
    return radius_mean + radius_magnitude * np.cos(
        (orientation + angle) * number_of_edges
    )


def angle_difference_abs(angle1, angle2):
    """
    Difference between two angles [0,pi[
    (commutative)
    """
    angle_diff = np.abs(angle2 - angle1)

    ind_angle = angle_diff >= pi
    if np.sum(ind_angle):
        angle_diff[ind_angle] = 2 * pi - angle_diff[ind_angle]

    return angle_diff


# INPUT: (n_samples_1, n_features), (n_samples_2, n_features)
# OUTPUT: (n_samples_1, n_samples_2)


def reshape(XX, YY):
    n_samples_1 = XX.shape[0]
    n_samples_2 = YY.shape[0]
    XX = np.tile(XX, (n_samples_2, 1, 1))
    XX = np.swapaxes(XX, 0, 1)
    YY = np.tile(YY, (n_samples_1, 1, 1))
    return XX, YY


def kernel_rbf(XX, YY, gamma=20):
    XX, YY = reshape(XX, YY)
    # RBF Kernel
    # kk = np.exp(-1/(2*sigma*sigma)*mag2)
    mag2 = np.sum((XX - YY) ** 2, axis=2)
    kk = np.exp(-gamma * mag2)
    return kk


def kernel_polynomial(XX, YY, poly_power=1, center=0):
    XX, YY = reshape(XX, YY)
    # Polynomial Kernel
    dot_prod = np.sum((XX * YY) ** poly_power, axis=2)
    kk = dot_prod


def kernel_spherical(XX, YY, sigma=10):
    XX, YY = reshape(XX, YY)
    # Spherical
    sigma = 10
    dist = np.sqrt(np.sum((XX - YY) ** 2, axis=2))
    kk = 1 - 3.0 / 2 * dist / sigma + 0.5 * (dist / sigma) ** 2
    return kk


def kernel_circular(XX, YY, sigma=10):
    XX, YY = reshape(XX, YY)
    # Spherical
    sigma = 10
    dist = np.sqrt(np.sum((XX - YY) ** 2, axis=2))
    kk = 1 - 3.0 / 2 * dist / sigma + 0.5 * (dist / sigma) ** 2
    return kk


def kernel_radial(XX, YY, gamma=20):
    XX, YY = reshape(XX, YY)

    ## Angular
    XX_magnitude = np.linalg.norm(XX, axis=2)
    YY_magnitude = np.linalg.norm(YY, axis=2)

    dot_prod = np.sum((XX * YY), axis=2)
    dot_prod = dot_prod / (XX_magnitude * YY_magnitude)

    # dot_prod = dot_prod-1
    # kk = np.exp(-gamma*dot_prod)
    kk = dot_prod
    # kk = dot_prod + YY_magnitude

    return kk


def kernel_radial2(XX, YY, gamma=20):
    XX, YY = reshape(XX, YY)

    ## Angular
    XX_magnitude = np.linalg.norm(XX, axis=2)
    YY_magnitude = np.linalg.norm(YY, axis=2)

    dot_prod = np.sum((XX * YY), axis=2)
    dot_prod = dot_prod / (XX_magnitude * YY_magnitude)

    # dot_prod = dot_prod-1
    # kk = np.exp(-gamma*dot_prod)
    # kk = dot_prod
    # kk = dot_prod + XX_magnitude + YY_magnitude

    dot_prod += 1
    kk = dot_prod

    return kk


def kernel_radial_mag(XX, YY, gamma=10):
    x_mean = np.mean(XX[:, 1])
    y_mean = np.mean(YY[:, 1])

    XX, YY = reshape(XX, YY)
    rad_diff = angle_difference_abs(XX[:, :, 0], YY[:, :, 0])

    # kk = np.exp(-gamma*rad_diff*rad_diff)
    kk = np.exp(-gamma * rad_diff * rad_diff)

    # kk = XX[:,:,1] + YY[:,:,1] - x_mean - y_mean

    # kk = kk + XX[:, :, 1] + YY[:, :, 1]
    # kk = XX[:, :, 1] + YY[:, :, 1]

    return kk


function_kernel = kernel_radial_mag

x_limit = [-1.0, 1.0]
y_limit = [-1.0, 1.0]

# sample points
n_samples = 1000
dimension = 2

XX = np.random.rand(dimension, n_samples)
XX = XX - 0.5  # TODO with general limits
XX[0, :] = XX[0, :] * (x_limit[1] - x_limit[0])
XX[1, :] = XX[1, :] * (y_limit[1] - y_limit[0])
rad = np.linalg.norm(XX, axis=0)

angle = np.arctan2(XX[1, :], XX[0, :])
rad_surf = get_radius_of_angle(angle)

label = rad < rad_surf

if True:
    plt.figure()
    plt.plot(XX[0, label == 0], XX[1, label == 0], "b.")
    plt.plot(XX[0, label == 1], XX[1, label == 1], "r.")
    plt.grid()
    plt.axis("equal")
    plt.xlim(x_limit)
    plt.ylim(y_limit)
    plt.show()

show_cartesian_simulation = False
if show_cartesian_simulation:

    kernel = 1.0 * RBF(1.0)
    # classifier = GaussianProcessClassifier(kernel=kernel, random_state=0)
    classifier = svm.SVC(kernel="rbf", gamma=20, C=100.0)
    model = classifier.fit(XX.T, label.T)
    print("n support vectors_", classifier.support_vectors_.shape[0])

    n_resol = 30
    xx, yy = np.meshgrid(
        np.linspace(x_limit[0], x_limit[1], n_resol),
        np.linspace(y_limit[0], y_limit[1], n_resol),
    )

    if True:
        prediction_gp = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
        prediction_gp = prediction_gp.reshape(xx.shape)

        fig, ax = plt.subplots()
        cs = ax.contourf(xx, yy, prediction_gp, cmap=plt.cm.coolwarm, alpha=0.8)
        cbar = fig.colorbar(cs)
        plt.axis("equal")

    if True:
        predict_score = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
        predict_score = predict_score.reshape(xx.shape)

        fig, ax = plt.subplots()
        cs = ax.contourf(xx, yy, predict_score, cmap=plt.cm.coolwarm, alpha=0.8)
        cbar = fig.colorbar(cs)
        # plt.colorbar()
        plt.axis("equal")

# RADIAL
show_radial_simulation = False
if show_radial_simulation:
    ang = np.arctan2(XX[1, :], XX[0, :])
    mag = np.linalg.norm(XX, axis=0)

    plt.figure()
    plt.plot(ang[label == 0], mag[label == 0], "b.")
    plt.plot(ang[label == 1], mag[label == 1], "r.")
    plt.xlim(-pi, pi)

    XX_r = np.vstack((ang, mag))
    cassifier_svm = svm.SVC(kernel=function_kernel, C=200.0)
    model = cassifier_svm.fit(XX_r.T, label.T)

    n_resol = 30
    pp, rr = np.meshgrid(np.linspace(-pi, pi, n_resol), np.linspace(0, 2, n_resol))

    predict_score = cassifier_svm.decision_function(np.c_[pp.ravel(), rr.ravel()])
    predict_score = predict_score.reshape(rr.shape)

    fig, ax = plt.subplots()
    cs = ax.contourf(pp, rr, predict_score, cmap=plt.cm.coolwarm, alpha=0.8)
    cbar = fig.colorbar(cs)
    # plt.axis('equal')
    plt.xlim(-pi, pi), plt.ylim(0, 1)

    predict_class = cassifier_svm.predict(np.c_[pp.ravel(), rr.ravel()])
    predict_class = predict_class.reshape(rr.shape)

    fig, ax = plt.subplots()
    cs = ax.contourf(pp, rr, predict_class, cmap=plt.cm.coolwarm, alpha=0.8)
    cbar = fig.colorbar(cs)
    # plt.axis('equal')
    plt.xlim(-pi, pi), plt.ylim(0, 1)

    fig, ax = plt.subplots()
    xx_r = np.cos(pp) * rr
    yy_r = np.sin(pp) * rr
    cs = ax.contourf(xx_r, yy_r, predict_class, cmap=plt.cm.coolwarm, alpha=0.8)
    cbar = fig.colorbar(cs)
    # plt.axis('equal')
    plt.xlim(-2, 2), plt.ylim(-2, 2)


# RADIAL - CUSTOM
show_custom_simulation = True
if show_custom_simulation:
    ang = np.arctan2(XX[1, :], XX[0, :])
    mag = np.linalg.norm(XX, axis=0)

    plt.figure()
    plt.plot(ang[label == 0], mag[label == 0], "b.")
    plt.plot(ang[label == 1], mag[label == 1], "r.")
    plt.xlim(-pi, pi)

    XX_r = np.vstack((ang, mag))
    cassifier_svm = svm.SVC(kernel=function_kernel, C=200.0)
    model = cassifier_svm.fit(XX_r.T, label.T)

    n_resol = 30
    pp, rr = np.meshgrid(np.linspace(-pi, pi, n_resol), np.linspace(0, 2, n_resol))

    predict_score = cassifier_svm.decision_function(np.c_[pp.ravel(), rr.ravel()])
    predict_score = predict_score.reshape(rr.shape)

    fig, ax = plt.subplots()
    cs = ax.contourf(pp, rr, predict_score, cmap=plt.cm.coolwarm, alpha=0.8)
    cbar = fig.colorbar(cs)
    # plt.axis('equal')
    plt.xlim(-pi, pi), plt.ylim(0, 1)

    predict_class = cassifier_svm.predict(np.c_[pp.ravel(), rr.ravel()])
    predict_class = predict_class.reshape(rr.shape)

    fig, ax = plt.subplots()
    cs = ax.contourf(pp, rr, predict_class, cmap=plt.cm.coolwarm, alpha=0.8)

    cbar = fig.colorbar(cs)
    # plt.axis('equal')
    plt.xlim(-pi, pi), plt.ylim(0, 1)

    fig, ax = plt.subplots()
    xx_r = np.cos(pp) * rr
    yy_r = np.sin(pp) * rr
    cs = ax.contourf(xx_r, yy_r, predict_class, cmap=plt.cm.coolwarm, alpha=0.8)
    cbar = fig.colorbar(cs)
    # plt.axis('equal')
    plt.xlim(-2, 2), plt.ylim(-2, 2)
