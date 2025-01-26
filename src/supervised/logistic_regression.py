"""
Logistic Regression Classifier
-------------------------------

Theory and Derivation:
----------------------
Logistic Regression is a linear model for binary classification that models the conditional probability of the target variable (y) given the input features (X) using the logistic function:

    P(y=1|X; w) = sigmoid(X @ w) = 1 / (1 + exp(-X @ w))

The parameters `w` (weights) are estimated by minimizing the negative log-likelihood of the observed data, which is equivalent to maximizing the likelihood. For binary classification, the log-likelihood is given by:

    L(w) = sum(y * log(sigmoid(X @ w)) + (1 - y) * log(1 - sigmoid(X @ w)))

Gradient Descent Update Rule:
-----------------------------
The weights are updated iteratively using gradient descent. The gradient of the log-likelihood with respect to the weights is:

    \( \frac{\partial L(w)}{\partial w} = X^T (y - sigmoid(X @ w)) \)

Using this gradient, the weight update rule becomes:

    w_new = w_old + learning_rate * X^T (y - sigmoid(X @ w_old))

Regularization:
---------------
To prevent overfitting, L2 regularization (Ridge) is often added to the loss function. The modified loss function is:

    L(w) = -L(w) + (\lambda / 2) * ||w||^2

Here, \( \lambda \) is the regularization parameter, controlled by the hyperparameter `C` (inverse of \( \lambda \)). The gradient is modified as:

    \( \frac{\partial L(w)}{\partial w} = X^T (y - sigmoid(X @ w)) - \lambda w \)

This implementation optimizes the weights using the gradient descent method described above.

"""

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.utils.validation import check_X_y, check_array

from src.utils.mathutils import sigmoid


def add_intercept_column(X: np.ndarray) -> np.ndarray:
    """Adds an intercept column of ones to the input data array.

    Args:
        X (np.ndarray): Input feature matrix.

    Returns:
        np.ndarray: Feature matrix with an added intercept column.
    """
    return np.hstack((np.ones((X.shape[0], 1)), X))


class LogisticRegression(LinearClassifierMixin, BaseEstimator):
    """
    Logistic Regression classifier implemented from scratch.

    This implementation uses gradient descent for optimization and supports L2 regularization.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        C: float = 1.0,
        learning_rate: float = 0.001,
        max_iter: int = 100,
        tol: float = 1e-4,
    ):
        """
        Initializes the LogisticRegression model.

        Args:
            fit_intercept (bool): Whether to add an intercept term. Defaults to True.
            C (float): Inverse of regularization strength. Defaults to 1.0.
            learning_rate (float): Learning rate for gradient descent. Defaults to 0.001.
            max_iter (int): Maximum number of iterations for optimization. Defaults to 100.
            tol (float): Tolerance for convergence. Defaults to 1e-4.
        """
        self.num_features: Optional[int] = None
        self._coef: Optional[np.ndarray] = None
        self.fit_intercept = fit_intercept
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def _xavier_initialization(self) -> np.ndarray:
        """
        Performs Xavier initialization of the coefficients.

        Returns:
            np.ndarray: Initialized coefficient vector.
        """
        return np.random.uniform(
            -1 / self.num_features, 1 / self.num_features, self.num_features
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Fits the logistic regression model to the training data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).

        Returns:
            LogisticRegression: The fitted model.
        """
        # Validate and preprocess the input data
        X, y = check_X_y(X, y)
        if self.fit_intercept:
            X = add_intercept_column(X)

        self.num_features = X.shape[1]
        coef = self._xavier_initialization()

        # Gradient descent optimization
        for _ in range(self.max_iter):
            yhat = sigmoid(X @ coef)  # Predicted probabilities
            residuals = y - yhat  # Difference between actual and predicted values
            dldcoef = residuals @ X  # Gradient calculation

            old_coef = coef
            coef -= self.learning_rate * dldcoef  # Update coefficients

            if self._converged(old_coef, coef):  # Check for convergence
                break

        self._coef = coef
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts probabilities for the input data.

        Args:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted probabilities for each sample.
        """
        self._is_fitted()
        X = check_array(X)
        if self.fit_intercept:
            X = add_intercept_column(X)
        yhat = sigmoid(X @ self._coef)
        return yhat

    def _converged(self, param: np.ndarray, new_param: np.ndarray) -> bool:
        """
        Checks if the optimization has converged based on the tolerance.

        Args:
            param (np.ndarray): Previous parameter vector.
            new_param (np.ndarray): Updated parameter vector.

        Returns:
            bool: True if the difference is less than the tolerance, False otherwise.
        """
        return np.linalg.norm(new_param - param) < self.tol

    def _is_fitted(self) -> None:
        """
        Ensures the model is fitted before making predictions.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if self._coef is None:
            raise ValueError("LogisticRegression is not fitted.")
