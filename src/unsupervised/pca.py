from typing import Tuple

import numpy as np
from numpy._typing import NDArray
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.extmath import svd_flip


class PCA(TransformerMixin, BaseEstimator):

    def __init__(self, n_components=2):
        self._components = None
        self.n_features = None
        self.n_samples = None
        self._means = None
        self.n_components = n_components

    def _fit(
        self, X: NDArray[float], y=None
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """

        :param X: an ndarray of shape (n_samples, n_features)
        :param y: ignored
        :return:
        """

        self.n_samples, self.n_features = X.shape

        if self.n_components is None:
            self.n_components = min(self.n_samples, self.n_features)
        elif not 0 <= self.n_components <= min(self.n_samples, self.n_features):
            ValueError(
                f"n_components={self.n_components} must be between 0 and "
                f"min(n_samples, self.n_features)={min(self.n_samples, self.n_features)}."
            )

        self._means = X.mean(axis=0)
        X -= self._means

        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.explained_variance = S**2 / (self.n_samples - 1)
        self.percent_variance_explained = (
            self.explained_variance / self.explained_variance.sum()
        )

        U, Vt = svd_flip(U, Vt, u_based_decision=False)

        self._components = Vt[: self.n_components, :]

        return U, S, Vt, X

    def fit(self, X: NDArray, y=None) -> "PCA":
        """
        Fit the principal component analysis model.
        :param X: ndarray of shape (n_samples, n_features)
        :param y: ignored
        :return: PCA instance
        """

        _ = self._fit(X, y)

        return self

    def fit_transform(self, X: NDArray[float], y=None, **fit_params) -> NDArray[float]:
        """

        :param X: ndarray (n_samples, n_features)
        :param y: ignored
        :param fit_params: ignored
        :return: ndarray (n_samples, n_components)
        """

        U, S, Vt, X = self._fit(X, y)

        return self.transform(X)

    def transform(self, X: NDArray) -> NDArray:
        """
        Transform the data using PCA components
        :param X: an ndarray of shape (n_samples, n_features)
        :return: an ndarray of shape (n_samples, n_components)
        """

        assert self.is_fitted(), "Please fit the model before transforming"
        return X @ self._components.T

    def inverse_transform(self, X: NDArray) -> NDArray:
        """
        Perform inverse transformation on X.
        :param X:
        :return:
        """

        return X @ self._components + self._means

    def is_fitted(self):
        """
        Check if the model is fitted or not
        :return: bool
        """

        return self._components is not None
