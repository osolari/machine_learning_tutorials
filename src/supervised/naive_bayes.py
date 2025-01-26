r"""
Mathematical Derivations of the Naive Bayes Classifier:

The Naive Bayes classifier is based on Bayes' Theorem:
    P(C|X) = (P(X|C) * P(C)) / P(X)

Where:
    - P(C|X) is the posterior probability of class C given data X.
    - P(X|C) is the likelihood of data X given class C.
    - P(C) is the prior probability of class C.
    - P(X) is the probability of the data X (evidence).

Assumptions:
The Naive Bayes classifier assumes conditional independence between features:
    P(X|C) = P(x1, x2, ..., xn|C) = P(x1|C) * P(x2|C) * ... * P(xn|C)

For Gaussian Naive Bayes:
    P(xi|C) is modeled as a Gaussian distribution:
    P(xi|C) = (1 / sqrt(2 * \pi * var_Ci)) * exp(- (xi - mean_Ci)^2 / (2 * var_Ci))

Log-Likelihood Computation:
    log(P(C|X)) / propto log(P(C)) + / sum_{i=1}^n log(P(xi|C))

This approach avoids numerical underflow by operating in log-space.
"""

import math

from src.utils.mlutils import softmax
from typing import Union
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_array
from abc import abstractmethod


class _BaseNaiveBayes(ClassifierMixin, BaseEstimator):
    """
    Abstract base class for Naive Bayes classifiers.

    Methods:
    - __compute_prior: Abstract method to compute class priors.
    - fit: Abstract method to fit the model to training data.
    - predict: Abstract method to make predictions on new data.
    - predict_proba: Abstract method to compute class probabilities for new data.
    """

    @abstractmethod
    def __compute_prior(self):
        pass

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass


class GaussianNaiveBayes(_BaseNaiveBayes):
    """
    Gaussian Naive Bayes classifier.

    Parameters:
    - priors (Union[np.ndarray, None]): Class prior probabilities. If None, computed from the data.
    - var_smoothing (float): Small value added to variances for numerical stability.

    Attributes:
    - priors: Class priors.
    - var_smoothing: Variance smoothing parameter.
    - y, X: Training labels and features.
    - _classes: Unique class labels.
    - _class_prior: Computed or provided class priors.
    - _var: Variance of features per class.
    - _theta: Mean of features per class.
    - _n_features_in: Number of features in the training data.
    - _epsilon: Small value for numerical stability.
    """

    def __init__(
        self, priors: Union[np.ndarray, None] = None, var_smoothing: float = 1e-9
    ):
        super().__init__()
        self.priors = priors
        self.var_smoothing = var_smoothing

        self.y = None
        self.X = None
        self._classes = None
        self._class_prior = None
        self._var = None
        self._theta = None
        self._n_features_in = None
        self._epsilon = None

    def _compute_prior(self):
        """
        Compute the class prior probabilities based on the training labels or provided priors.
        """
        self.class_count_ = np.bincount(self.y)
        self._classes = np.arange(len(self.class_count_))

        if self.priors is None:
            self._class_prior = self.class_count_ / self.class_count_.sum()
        else:
            assert len(self.priors) == len(
                self.class_count_
            ), f"priors shape must match class_count shape, {len(self.priors)} != {len(self.class_count_)}"
            assert (
                self.priors.sum() == 1
            ), f"priors must sum to 1, {len(self.priors)} != {1}"
            self._class_prior = self.priors

    def _compute_gaussian_params(
        self, vals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the mean and variance for Gaussian distribution per feature.

        Parameters:
        - vals (np.ndarray): Feature values for a given class.

        Returns:
        - tuple: Mean and variance for each feature.
        """
        mean = np.mean(vals, axis=0)
        var = np.var(vals, axis=0) + self.var_smoothing
        return mean, var

    def _compute_gaussian_loglikelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the log-likelihood of a Gaussian distribution for given data points.

        Parameters:
        - x (np.ndarray): Data points.

        Returns:
        - np.ndarray: Log-likelihood values.
        """
        ss = (x - self._theta) ** 2
        llhd = (-0.5 * (np.log(2 * np.pi) + np.log(self._var) + ss / self._var)).sum(
            axis=1
        )
        llhd += np.log(self._class_prior)
        return llhd

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Gaussian Naive Bayes model to the training data.

        Parameters:
        - X (np.ndarray): Training feature matrix.
        - y (np.ndarray): Training labels.
        """
        self.X, self.y = check_X_y(X, y)
        self._n_features_in = self.X.shape[1]
        self._compute_prior()
        self._theta, self._var = zip(
            *[self._compute_gaussian_params(self.X[self.y == c]) for c in self._classes]
        )
        self._theta, self._var = np.array(self._theta), np.array(self._var)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the input data.

        Parameters:
        - X (np.ndarray): Input feature matrix.

        Returns:
        - np.ndarray: Predicted class probabilities.
        """
        X = check_array(X)
        return np.array([softmax(self._compute_gaussian_loglikelihood(x)) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input data.

        Parameters:
        - X (np.ndarray): Input feature matrix.

        Returns:
        - np.ndarray: Predicted class labels.
        """
        probas = self.predict_proba(X)
        return probas.argmax(axis=1)
