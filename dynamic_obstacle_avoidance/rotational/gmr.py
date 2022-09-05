""" Gaussian Mixture Regression"""

import warnings
from math import pi
from dataclasses import dataclass

import numpy as np
from numpy import linalg as LA

from sklearn import mixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import train_test_split

# from sklearn.mixture import GaussianMixture


class GaussianMixtureRegression:
    def __init__(self, n_components: int = 5, covariance_type: str = "full"):

        self.dpgmm = mixture.BayesianGaussianMixture(
            n_components=n_components, covariance_type=covariance_type
        )

    @property
    def n_components(self) -> int:
        return self.dpgmm.n_components

    @n_components.setter
    def n_components(self, value: int) -> None:
        self.dpgmm.n_components = value

    @property
    def covariance_type(self) -> str:
        return self.dpgmm.covariance_type

    @covariance_type.setter
    def covariance_type(self, value: str) -> None:
        self.dpgmm.covariance_type = value

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Regress based on the data given."""
        self.n_samples_fit_ = X.shape[0]

        if y.shape[0] != self.n_samples_fit_:
            raise ValueError("Input data is not consistent.")

        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = y.shape[1]

        self.indexes_in_ = np.arange(self.n_features_in_)
        self.indexes_out_ = np.arange(
            self.n_features_in_, self.n_features_in_ + self.n_features_out_
        )

        self.dpgmm.fit(np.hstack((X, y)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Evaluate the regress field at all the points X"""

        # Gausian Mixture Model Properties
        beta = self._predict_mixing_weights(X)
        mu_yx = self._predict_mean_yx(X)

        regression_value = np.sum(
            np.tile(beta.T, (self.n_features_out_, 1, 1)) * mu_yx, axis=2
        ).T

        return regression_value

    def _predict_mixing_weights(
        self,
        X: np.ndarray,
        normalize_probability: bool = True,
        weight_factor: float = 4.0,
    ):
        """Get input positions X of the form [dimension, number of samples]."""
        # TODO: try to learn the 'weight_factor' [optimization problem?]
        dim_in = self.indexes_in_.shape[0]

        prob_gaussian = self._predict_gaussian_probability(X)

        alpha_times_prob = (
            np.tile(self.dpgmm.weights_, (X.shape[0], 1)).T * prob_gaussian
        )

        if normalize_probability:
            beta = alpha_times_prob / np.tile(
                np.sum(alpha_times_prob, axis=0), (self.n_components, 1)
            )
        else:
            beta = alpha_times_prob
            max_weight = np.max(self.dpgmm.weights_)
            beta = beta / max_weight * weight_factor**dim_in

            sum_beta = np.sum(beta, axis=0)
            ind_large = sum_beta > 1
            beta[:, ind_large] = beta[:, ind_large] / np.tile(
                sum_beta[ind_large], (self.n_components, 1)
            )
        return beta

    def _predict_gaussian_probability(self, X: np.ndarray = None) -> np.ndarray:
        """Returns the array of 'mean'-values based on input positions.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_input_features)
        List of n_features-dimensional input data. Each column
            corresponds to a single data point.

        Returns
        -------
        prob_gauss (beta): array of shape (n_samples)
            The weights (similar to prior) which is gaussian is assigned.
        """
        covariance_matrices = self.dpgmm.covariances_[:, self.indexes_in_, :][
            :, :, self.indexes_in_
        ]

        mean = self.dpgmm.means_[:, self.indexes_in_]

        n_samples = X.shape[0]
        dim_in = X.shape[1]

        # Calculate weight (GAUSSIAN ML)
        prob_gauss = np.zeros((self.n_components, n_samples))

        for gg in range(self.n_components):
            covariance = covariance_matrices[gg, :, :]
            fac = 1 / (
                (2 * pi) ** (dim_in * 0.5) * (np.linalg.det(covariance)) ** (0.5)
            )

            dX = X - np.tile(mean[gg, :], (n_samples, 1))

            val_pow_fac = np.sum(
                np.tile(np.linalg.pinv(covariance), (n_samples, 1, 1))
                * np.swapaxes(np.tile(dX, (dim_in, 1, 1)), 0, 1),
                axis=2,
            )

            val_pow = np.exp(-np.sum(dX * val_pow_fac, axis=1))
            prob_gauss[gg, :] = fac * val_pow

        return prob_gauss

    def _predict_mean_yx(self, X: np.ndarray) -> np.ndarray:
        """Returns the array of 'mean'-values based on input positions.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_input_features)
        List of n_features-dimensional input data. Each column
            corresponds to a single data point.

        Returns
        -------
        mu_yx: array-like of shape (n_samples, n_output_features)
            List of n_features-dimensional output data. Each column
            corresponds to a single data point.
        """
        n_samples = X.shape[0]

        mu_yx = np.zeros((self.n_features_out_, n_samples, self.n_components))
        mu_yx_hat = np.zeros((self.n_features_out_, n_samples, self.n_components))

        for gg in range(self.n_components):
            mu_yx[:, :, gg] = np.tile(
                self.dpgmm.means_[gg, self.indexes_out_], (n_samples, 1)
            ).T
            matrix_mult = self.dpgmm.covariances_[gg][self.indexes_out_, :][
                :, self.indexes_in_
            ].dot(
                LA.pinv(
                    self.dpgmm.covariances_[gg][self.indexes_in_, :][
                        :, self.indexes_in_
                    ]
                )
            )

            mu_yx[:, :, gg] += matrix_mult.dot(
                (X - np.tile(self.dpgmm.means_[gg, self.indexes_in_], (n_samples, 1))).T
            )

            # ### START REMOVE ###
            # for nn in range(n_samples):  # TODO #speed - batch process!!
            #     mu_yx_hat[:, nn, gg] = self.dpgmm.means_[
            #         gg, self.indexes_out_
            #     ] + self.dpgmm.covariances_[gg][self.indexes_out_, :][
            #         :, self.indexes_in_
            #     ] @ np.linalg.pinv(
            #         self.dpgmm.covariances_[gg][self.indexes_in_, :][
            #             :, self.indexes_in_
            #         ]
            #     ) @ (
            #         X[nn, :] - self.dpgmm.means_[gg, self.indexes_in_]
            #     )

        # if np.sum(mu_yx - mu_yx_hat) > 1e-6:
        #     breakpoint()
        # else:
        #     # TODO: remove when warning never shows up anymore
        #     warnings.warn("Remove looped multiplication, since is the same...")
        return mu_yx

    def get_covariance_out(self, stretch_input_values: bool = False):
        """Returns the array of 'mean'-values based on input positions.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_input_features)
        List of n_features-dimensional input data. Each column
            corresponds to a single data point.

        Returns
        -------
        mu_yx: array-like of shape (n_samples, n_output_features)
            List of n_features-dimensional output data. Each column
            corresponds to a single data point.
        """
        dim_out = np.array(self.indexes_out_).shape[0]
        covariance_out = np.zeros((dim_out, dim_out, self.n_components))

        for gg in range(self.n_components):
            covariance = self.dpgmm.covariances_[gg, :, :]
            covariance_out[:, :, gg] = (
                covariance[self.indexes_out_, :][:, self.indexes_out_]
                - covariance[self.indexes_out_, :][:, self.indexes_in_]
                @ np.linalg.pinv(covariance[self.indexes_in_, :][:, self.indexes_in_])
                @ covariance[self.indexes_in_, :][:, self.indexes_out_]
            )
        return covariance_out


def plot_ellipses(gmm, ax):
    import matplotlib as mpl

    for n in range(gmm.n_components):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2],
            v[0],
            v[1],
            180 + angle,
            # color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


def test_sinus_regression(visualize=False):
    RANDOM_SEED = 2
    np.random.seed(RANDOM_SEED)

    X = 5 * np.random.rand(1000, 1)

    # Add noise to targets
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(X.shape[0] // 5))
    y = y.reshape(-1, 1)

    tt_ratio = 2 / 3
    all_index = np.arange(X.shape[0])
    train_index, test_index = train_test_split(all_index, test_size=(1 - tt_ratio))

    X_train = X[train_index, :]
    X_test = X[test_index, :]

    y_train = y[train_index, :]
    y_test = y[test_index, :]

    regression_model = GaussianMixtureRegression(n_components=5)
    regression_model.fit(X_train, y_train)

    # X_check = np.array([5]).reshape(1, -1)
    # y_check = regression_model.predict(X_check)

    y_predict = regression_model.predict(X_test)

    error = np.sum((y_test - y_predict) ** 2) / y_predict.shape[0]

    if visualize:
        import matplotlib.pyplot as plt

        plt.ion()
        # plt.close("all")

        fig, ax = plt.subplots()
        ax.plot(X, y, ".", label="Original")

        ax.plot(X_test, y_predict, ".", color="red")
        ax.plot(X_test, y_test, ".", color="green")

        plot_ellipses(regression_model.dpgmm, ax=ax)

    assert error < 0.5, "Error is too large."


if (__name__) == "__main__":
    test_sinus_regression()
