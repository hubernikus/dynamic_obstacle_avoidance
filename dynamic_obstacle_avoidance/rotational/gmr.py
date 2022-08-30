""" Gaussian Mixture Regression"""

import warnings

import numpy as np
from numpy import linalg as LA

# from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture


class GMR:
    def __init__(self):
        pass

    def fit(self, n_gaussian=5, covariance_type="full", tt_ratio=0.75):
        """Regress based on the data given."""
        a_label = np.zeros(self.num_samples)
        all_index = np.arange(self.num_samples)

        train_index, test_index = train_test_split(all_index, test_size=(1 - tt_ratio))

        X_train = self.X[train_index, :]
        X_test = self.X[test_index, :]

        y_train = a_label[train_index]
        y_test = a_label[test_index]

        self.dpgmm = mixture.BayesianGaussianMixture(
            n_components=n_gaussian, covariance_type=covariance_type
        )

        # sample dataset
        reference_dataset = 0
        n_start = 0
        for it_set in range(reference_dataset):
            n_start += self.dataset["data"][0, it_set].shape[1]

        index_sample = [
            int(
                np.round(
                    n_start
                    + self.dataset["data"][0, reference_dataset].shape[1]
                    / n_gaussian
                    * ii
                )
            )
            for ii in range(n_gaussian)
        ]

        self.dpgmm.means_init = self.X[index_sample, :]
        self.dpgmm.means_init = X_train[np.random.choice(np.arange(n_gaussian)), :]

        self.dpgmm.fit(X_train[:, :])

    def _predict(
        self,
        X,
        input_output_normalization=True,
        feat_in=None,
        feat_out=None,
        convergence_attractor=False,
        p_beta=2,
        beta_min=0.5,
        beta_r=0.3,
    ):
        """Evaluate the regress field at all the points X"""
        dim = self.dim_gmm
        n_samples = X.shape[0]
        dim_in = X.shape[1]

        if feat_in is None:
            feat_in = np.arange(dim_in)

        if feat_out is None:
            # Default only the 'direction' at the end; additional -1 for indexing at end
            feat_out = self.dim_gmm - 1 - np.arange(self.dim_space - 1)
        dim_out = np.array(feat_out).shape[0]

        if input_output_normalization:
            X = self.transform_initial_to_normalized(X, dims_ind=feat_in)

        # Gausian Mixture Model Properties
        beta = self.get_mixing_weights(X, feat_in=feat_in, feat_out=feat_out)
        mu_yx = self.get_mean_yx(X, feat_in=feat_in, feat_out=feat_out)

        if convergence_attractor:
            if self.pos_attractor is None:
                raise ValueError("Convergence to attractor without attractor...")

            if dim_in == self.dim_space:
                attractor = self.pos_attractor
            else:
                # Attractor + zero-velocity
                attractor = np.hstack((self.pos_attractor, np.zeros(self.dim_space)))

            dist_attr = np.linalg.norm(X - np.tile(attractor, (n_samples, 1)), axis=1)

            beta = np.vstack((beta, np.zeros(n_samples)))

            # Zero values
            ind_zero = dist_attr == 0
            beta[:, ind_zero] = 0
            beta[-1, ind_zero] = 1

            # Nonzeros values
            ind_nonzero = dist_attr != 0
            beta[-1, ind_nonzero] = (dist_attr[ind_nonzero] / beta_r) ** (
                -p_beta
            ) + beta_min
            beta[:, ind_nonzero] = beta[:, ind_nonzero] / np.tile(
                np.linalg.norm(beta[:, ind_nonzero], axis=0), (self.n_gaussians + 1, 1)
            )

            # Add zero velocity
            mu_yx = np.dstack((mu_yx, np.zeros((dim_out, n_samples, 1))))

        regression_value = np.sum(np.tile(beta.T, (dim_out, 1, 1)) * mu_yx, axis=2).T

        # breakpoint()
        if input_output_normalization:
            regression_value = self.transform_normalized_to_initial(
                regression_value, dims_ind=feat_out
            )

        if False:
            print("shape X", X.shape)
            if X.shape[0] == 1:
                print("X", X)
                print("beta", np.round(beta.T, 2))
                print("mu_yx", np.round(mu_yx, 2))
                print("regression", regression_value)
            breakpoint()

        return regression_value

    def get_mixing_weights(
        self,
        X,
        feat_in,
        feat_out,
        input_needs_normalization=False,
        normalize_probability=False,
        weight_factor=4.0,
    ):
        """Get input positions X of the form [dimension, number of samples]."""
        # TODO: try to learn the 'weight_factor' [optimization problem?]
        if input_needs_normalization:
            X = self.transform_initial_to_normalized(X, feat_in)

        n_samples = X.shape[0]
        dim_in = feat_in.shape[0]

        prob_gaussian = self.get_gaussian_probability(X, feat_in=feat_in)
        sum_probGaussian = np.sum(prob_gaussian, axis=0)

        alpha_times_prob = (
            np.tile(self.dpgmm.weights_, (n_samples, 1)).T * prob_gaussian
        )

        if normalize_probability:
            beta = alpha_times_prob / np.tile(
                np.sum(alpha_times_prob, axis=0), (self.n_gaussians, 1)
            )
        else:
            beta = alpha_times_prob
            max_weight = np.max(self.dpgmm.weights_)
            beta = beta / max_weight * weight_factor**dim_in

            sum_beta = np.sum(beta, axis=0)
            ind_large = sum_beta > 1
            beta[:, ind_large] = beta[:, ind_large] / np.tile(
                sum_beta[ind_large], (self.n_gaussians, 1)
            )
        return beta

    def get_gaussian_probability(
        self, X, feat_in=None, covariance_matrices=None, mean=None
    ):
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
        if covariance_matrices is None:
            covariance_matrices = self.dpgmm.covariances_[:, feat_in, :][:, :, feat_in]
        if mean is None:
            mean = self.dpgmm.means_[:, feat_in]

        n_samples = X.shape[0]
        dim_in = X.shape[1]
        # dim_in = mean.shape[1]

        # Calculate weight (GAUSSIAN ML)
        prob_gauss = np.zeros((self.n_gaussians, n_samples))

        for gg in range(self.n_gaussians):
            # Create function of this
            covariance = covariance_matrices[gg, :, :]
            try:
                fac = 1 / (
                    (2 * pi) ** (dim_in * 0.5) * (np.linalg.det(covariance)) ** (0.5)
                )
            except:
                breakpoint()
            dX = X - np.tile(mean[gg, :], (n_samples, 1))

            val_pow_fac = np.sum(
                np.tile(np.linalg.pinv(covariance), (n_samples, 1, 1))
                * np.swapaxes(np.tile(dX, (dim_in, 1, 1)), 0, 1),
                axis=2,
            )

            val_pow = np.exp(-np.sum(dX * val_pow_fac, axis=1))
            prob_gauss[gg, :] = fac * val_pow

        return prob_gauss

    def get_covariance_out(self, feat_in, feat_out, stretch_input_values=False):
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
        dim_out = np.array(feat_out).shape[0]
        covariance_out = np.zeros((dim_out, dim_out, self.n_gaussians))

        for gg in range(self.n_gaussians):
            covariance = self.dpgmm.covariances_[gg, :, :]
            covariance_out[:, :, gg] = (
                covariance[feat_out, :][:, feat_out]
                - covariance[feat_out, :][:, feat_in]
                @ np.linalg.pinv(covariance[feat_in, :][:, feat_in])
                @ covariance[feat_in, :][:, feat_out]
            )
        return covariance_out

    def get_mean_yx(self, X, feat_in, feat_out, stretch_input_values=False):
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
        dim_out = np.array(feat_out).shape[0]
        n_samples = X.shape[0]

        mu_yx = np.zeros((dim_out, n_samples, self.n_gaussians))
        mu_yx_hat = np.zeros((dim_out, n_samples, self.n_gaussians))

        for gg in range(self.n_gaussians):
            mu_yx[:, :, gg] = np.tile(self.dpgmm.means_[gg, feat_out], (n_samples, 1)).T
            matrix_mult = self.dpgmm.covariances_[gg][feat_out, :][:, feat_in].dot(
                np.linalg.pinv(self.dpgmm.covariances_[gg][feat_in, :][:, feat_in])
            )

            mu_yx[:, :, gg] += matrix_mult.dot(
                (X - np.tile(self.dpgmm.means_[gg, feat_in], (n_samples, 1))).T
            )

            ### START REMOVE ###
            for nn in range(n_samples):  # TODO #speed - batch process!!
                mu_yx_hat[:, nn, gg] = self.dpgmm.means_[
                    gg, feat_out
                ] + self.dpgmm.covariances_[gg][feat_out, :][
                    :, feat_in
                ] @ np.linalg.pinv(
                    self.dpgmm.covariances_[gg][feat_in, :][:, feat_in]
                ) @ (
                    X[nn, :] - self.dpgmm.means_[gg, feat_in]
                )

        if np.sum(mu_yx - mu_yx_hat) > 1e-6:
            breakpoint()
        else:
            # TODO: remove when warning never shows up anymore
            warnings.warn("Remove looped multiplication, since is the same...")
        return mu_yx
