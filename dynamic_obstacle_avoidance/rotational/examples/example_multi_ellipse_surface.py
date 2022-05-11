#!/USSR/bin/python3
""" Sample the space and decide if points are collision-full or free. """
# Author: Lukas Huber
# Github: hubernikus
# Created: 2022-04-23

import logging

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.mixture import GaussianMixture
from sklearn import svm


from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles import CuboidXd as Cuboid
from dynamic_obstacle_avoidance.containers import ObstacleContainer

from dynamic_obstacle_avoidance.visualization import plot_obstacles


def collision_sample_space(obstacle_container, num_samples, x_lim, y_lim):
    """ Returns random points based on collision with
    [ >0 : free space // <0 : within obstacle ] """
    dimension = 2
    rand_points = np.random.rand(dimension, num_samples)
    
    rand_points[0] = rand_points[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
    rand_points[1] = rand_points[1] * (y_lim[1] - y_lim[0]) + y_lim[0]

    value = obstacle_container.get_minimum_gamma(rand_points)
    
    # value = value - 1
    value = (value > 1).astype(int)
    value = 2 * value - 1   # Value is in [-1, 1]
    
    return rand_points, value


class GaussianMixtureClassifier:
    # class GaussianMixtureClassifier(BaseEstimator):
    def __init__(self, n_components, n_classes=2, **kwargs):
        self.n_classes = n_classes
        self._gmm_models = [
            GaussianMixture(n_components=n_components, **kwargs) for ii in range(n_classes)
        ]

    def fit(self, X, y, sample_weight=None):
        # TODO: what is sample_weight for?
        
        labels = np.unique(y).astype(int)
        if len(labels) > self.n_classes:
            raise ValueError("Too many labels for the proposed model.")

        for ii, gmm in enumerate(self._gmm_models):
            gmm.fit(X[y == labels[ii], :])

    def predict_proba(self, X):
        """ Returns pseudo-probability by summing individual Gaussians. """
        n_points = X.shape[0]
        probability_tot = np.zeros((n_points, self.n_classes))
        
        for ii, gmm in enumerate(self._gmm_models):
            probability_feature = gmm.predict_proba(X)
            probability_tot[:, ii] = np.sum(
                probability_feature * np.tile(gmm.weights_, (n_points, 1)),
                axis=1
            )
        return probability_tot

    def predict(self, X):
        return np.max(self.predict_proba(X), axis=0)

    def set_params(self, **params):
        for gmm in self._gmm_models:
            gmm.set_params(**params)

    def plot_model(self):
        fig, ax = plt.subplots()

        colors = ["navy", "turquoise", "darkorange"]
        for ii, gmm in enumerate(self._gmm_models):
            color = colors[ii]
            for jj in range(len(gmm.covariances_)):
                covariances = gmm.covariances_[jj][:2, :2]
            
                v, w = np.linalg.eigh(covariances)
                u = w[0] / np.linalg.norm(w[0])
            
                angle = np.arctan2(u[1], u[0])
                angle = 180 * angle / np.pi  # convert to degrees
                v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
                ell = mpl.patches.Ellipse(
                    gmm.means_[jj, :2], v[0], v[1], 180 + angle, color=color
                    )

                ell.set_clip_box(ax.bbox)
                ell.set_alpha(0.5)
                ax.add_artist(ell)
                
        ax.set_aspect("equal", "datalim")
                

class EnvironmentLearner:
    def __init__(self, data_points, label):
        self.data_points = data_points
        self.label = label

        self.my_classifier = None
        
        self.learn_gmm()
        # self.learn_svc()
        # self.learn_svr()

    def learn_svr(self):
        self.my_classifier = svm.SVR(
            C=1.0, kernel='rbf', gamma=1.0, epsilon=0.1,
        )

        self.my_classifier.fit(X=self.data_points.T, y=self.label)

    def learn_svc(self):
        self.my_classifier = svm.SVC(
            C=1.0, kernel='rbf', probability=True
        )

        # self.my_classifier.fit(X=self.data_points.T, y=self.label)
        self.my_classifier.fit(X=self.data_points.T, y=self.label)

    def learn_gmm(self, n_gmms=3):
        self.my_classifier = GaussianMixtureClassifier(
            n_components=n_gmms, n_classes=2
        )
        
        self.my_classifier.fit(X=self.data_points.T, y=self.label)


def get_three_elipse_learner():
    x_lim = [-2, 7]
    y_lim = [-6.5, 6.5]

    environment = ObstacleContainer()
    environment.append(
        Cuboid(
            center_position=[4.5, 0],
            axes_length=[2, 8],
        ))

    environment.append(
        Cuboid(
            center_position=[2, 3],
            axes_length=[5, 2],
        ))

    environment.append(
        Cuboid(
            center_position=[2, -3],
            axes_length=[5, 2],
        ))

    plot_obstacles(obstacle_container=environment, x_lim=x_lim, y_lim=y_lim)
    data_points, label = collision_sample_space(
        obstacle_container=environment,
        num_samples=1000,
        x_lim=x_lim,
        y_lim=y_lim
    )

    min_label = np.min(label)
    max_label = np.max(label)
    label_range = np.maximum(abs(min_label), max_label)

    _, ax = plt.subplots()
    ax.scatter(
        data_points[0, :],
        data_points[1, :],
        s=10,
        c=label,
        cmap='seismic',
        # norm=mpl.colors.Normalize(vmin=(-1)*label_range, vmax=label_range)
    )
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    my_learner = EnvironmentLearner(data_points, label)
    return my_learner


def plot_obstacle_regression(my_classifier, n_resolution=30, x_lim=[-10, 10], y_lim=[-10, 10]):
    nx = n_resolution
    ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx),
        np.linspace(y_lim[0], y_lim[1], ny),
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    # Regression
    regr_value = my_classifier.predict(positions.T)
    fig, ax = plt.subplots()
    ax.contourf(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        regr_value.reshape(nx, ny),
        cmap='RdBu',
        # cmap='seismic_r',
    )

    
def plot_obstacle_classification(
    my_classifier, n_resolution=30, x_lim=[-10, 10], y_lim=[-10, 10]
):
    nx = n_resolution
    ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx),
        np.linspace(y_lim[0], y_lim[1], ny),
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    class_weights = my_classifier.predict_proba(positions.T)
    # class_weights = my_classifier.predict(positions.T)

    colors = []
    colors.append([0.0, 0.0, 1.0])
    colors.append([1.0, 0.0, 0.0])
    
    dim_rgb = 3
    n_samples = class_weights.shape[0]

    color_tot = np.zeros((n_samples, dim_rgb))
    for ii in range(len(colors)):
        color_tot = color_tot + (np.tile(class_weights[:, ii], (dim_rgb, 1)).T
            * np.tile(colors[ii], (n_samples, 1)))
    
    # my_classifier.plot_model()

    # breakpoint()
    fig, ax = plt.subplots()
    plt.imshow(color_tot.reshape(nx, ny, 3))

    # ax.contourf(
        # positions[0, :].reshape(nx, ny),
        # positions[1, :].reshape(nx, ny),
        # c=color_tot
    # )
    
    
if (__name__) == "__main__":
    plt.close('all')
    my_learner = get_three_elipse_learner()
    # my_learner.learn_gmm()

    # plot_obstacle_regression(my_learner.my_classifier)
    plot_obstacle_classification(my_learner.my_classifier)
