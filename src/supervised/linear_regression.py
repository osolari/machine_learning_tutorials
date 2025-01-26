"""
Regression Models: Theory and Implementation
--------------------------------------------

This module provides a foundation for implementing regression algorithms, including Linear Regression, Lasso, Ridge, and Elastic Net.

Theory and Derivation:
----------------------
Linear regression models aim to predict a continuous target variable \( y \) as a linear combination of input features \( X \):

    y = X @ w + \epsilon

where \( w \) represents the weights (coefficients) and \( \epsilon \) is the error term.

Optimization Objective:
-----------------------
The weights are estimated by minimizing a loss function. Commonly used loss functions include:

1. **Mean Squared Error (MSE)**:

       MSE(w) = \frac{1}{n} \sum_{i=1}^{n} (y_i - X_i @ w)^2

   The gradient of MSE is:

       \( \nabla_w MSE(w) = -\frac{2}{n} X^T (y - X @ w) \)

2. **Regularized Loss Functions**:

   To prevent overfitting, regularization terms are added:

   - **L1 Regularization (Lasso)**:

         Loss(w) = MSE(w) + \alpha ||w||_1

   - **L2 Regularization (Ridge)**:

         Loss(w) = MSE(w) + \alpha ||w||_2^2

   - **Elastic Net** (combines L1 and L2):

         Loss(w) = MSE(w) + \alpha [ratio ||w||_1 + (1 - ratio) ||w||_2^2]

Gradient Descent:
-----------------
Weights are updated iteratively using gradient descent:

    w_new = w_old - learning_rate * \nabla Loss(w)

Convergence:
------------
Convergence is achieved when the change in weights is below a tolerance threshold:

    ||w_new - w_old|| < tol

"""

from typing import Union
import numpy as np
from numpy._typing import NDArray
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y, check_array

from src.test.utils.mlutils import add_intercept_column
from src.utils.regularization import (
    _Regularization,
    L1Regularization,
    L2Regularization,
    ElasticNetRegularization,
)


class _RegressionBase(RegressorMixin):
    """
    Base class for regression models using gradient descent optimization.
    """

    def __init__(
        self,
        fit_intercept: bool,
        max_iter: int,
        learning_rate: float,
        regularization: Union[_Regularization, None] = None,
        tol: float = 1e-8,
    ):
        """
        Initializes the regression base model.

        Args:
            fit_intercept (bool): Whether to add an intercept term.
            max_iter (int): Maximum number of iterations for gradient descent.
            learning_rate (float): Learning rate for optimization.
            regularization (Union[_Regularization, None]): Regularization loss function.
            tol (float): Tolerance for convergence. Defaults to 1e-8.
        """
        self._coef = None
        self.n_features = None
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.tol = tol

    def _xavier_initialization(self) -> NDArray:
        """
        Performs Xavier initialization of the coefficients.

        Returns:
            NDArray: Initialized coefficient vector.
        """
        return np.random.uniform(
            -1 / self.n_features, 1 / self.n_features, self.n_features
        )

    def fit(self, X: NDArray, y: NDArray):
        """
        Fits the regression model to the training data using gradient descent.

        Args:
            X (NDArray): Feature matrix of shape (n_samples, n_features).
            y (NDArray): Target vector of shape (n_samples,).
        """
        X, y = check_X_y(X, y)
        if self.fit_intercept:
            X = add_intercept_column(X)
        n_samples, self.n_features = X.shape

        coef = self._xavier_initialization()

        for _ in range(self.max_iter):
            yhat = X @ coef
            residuals = yhat - y
            dldcoef = (
                X.T @ residuals / n_samples + self.regularization.compute_gradient(coef)
                if self.regularization
                else 0
            )

            old_coef_ = coef
            coef -= self.learning_rate * dldcoef

            if self._converged(old_coef_, coef):
                print("CONVERGED!")
                break

        self._coef = coef

        return self

    def predict(self, X: NDArray) -> NDArray:
        """
        Predicts target values for the given input data.

        Args:
            X (NDArray): Feature matrix of shape (n_samples, n_features).

        Returns:
            NDArray: Predicted target values.
        """
        X = check_array(X)
        if self.fit_intercept:
            X = add_intercept_column(X)
        return X @ self._coef

    def _converged(self, param: NDArray, new_param: NDArray) -> bool:
        """
        Checks if the optimization has converged based on the tolerance.

        Args:
            param (NDArray): Previous parameter vector.
            new_param (NDArray): Updated parameter vector.

        Returns:
            bool: True if the change is below the tolerance, False otherwise.
        """
        return np.linalg.norm(new_param - param) < self.tol

    def _is_fitted(self):
        """
        Ensures the model is fitted before making predictions.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if self._coef is None:
            raise ValueError("LinearRegression is not fitted")


class LinearRegression(_RegressionBase):
    """
    Linear Regression model with optional closed-form solution.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        closed_form: bool = True,
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ):
        """
        Initializes the LinearRegression model.

        Args:
            fit_intercept (bool): Whether to add an intercept term. Defaults to True.
            closed_form (bool): Whether to use the closed-form solution. Defaults to True.
            learning_rate (float): Learning rate for gradient descent. Defaults to 0.001.
            max_iter (int): Maximum number of iterations for optimization. Defaults to 1000.
            tol (float): Tolerance for convergence. Defaults to 1e-8.
        """
        super().__init__(
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            learning_rate=learning_rate,
            regularization=None,
            tol=tol,
        )
        self.closed_form = closed_form

    def fit(self, X: NDArray, y: NDArray):
        """
        Fits the Linear Regression model to the training data.

        Args:
            X (NDArray): Feature matrix of shape (n_samples, n_features).
            y (NDArray): Target vector of shape (n_samples,).
        """
        if self.closed_form:
            X, y = check_X_y(X, y)
            if self.fit_intercept:
                X = np.hstack([np.ones((X.shape[0], 1)), X])
            n_samples, self.n_features = X.shape
            XtX = X.T @ X
            U, S, Vt = np.linalg.svd(XtX)
            XtXinv = U @ np.diag(1.0 / S) @ Vt
            self._coef = XtXinv @ (X.T @ y)
        else:
            super().fit(X, y)


class Lasso(_RegressionBase):
    """
    Lasso Regression model with L1 regularization.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        alpha: float = 0.1,
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ):
        """
        Initializes the Lasso Regression model.

        Args:
            fit_intercept (bool): Whether to add an intercept term. Defaults to True.
            alpha (float): Regularization strength. Defaults to 0.1.
            learning_rate (float): Learning rate for gradient descent. Defaults to 0.001.
            max_iter (int): Maximum number of iterations for optimization. Defaults to 1000.
            tol (float): Tolerance for convergence. Defaults to 1e-8.
        """
        self.regularization = L1Regularization(alpha=alpha)
        super().__init__(
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            learning_rate=learning_rate,
            regularization=self.regularization,
            tol=tol,
        )


class Ridge(_RegressionBase):
    """
    Ridge Regression model with L2 regularization.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        alpha: float = 0.1,
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ):
        """
        Initializes the Ridge Regression model.

        Args:
            fit_intercept (bool): Whether to add an intercept term. Defaults to True.
            alpha (float): Regularization strength. Defaults to 0.1.
            learning_rate (float): Learning rate for gradient descent. Defaults to 0.001.
            max_iter (int): Maximum number of iterations for optimization. Defaults to 1000.
            tol (float): Tolerance for convergence. Defaults to 1e-8.
        """
        self.regularization = L2Regularization(alpha=alpha)
        super().__init__(
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            learning_rate=learning_rate,
            regularization=self.regularization,
            tol=tol,
        )


class ElasticNet(_RegressionBase):
    """
    Elastic Net Regression model combining L1 and L2 regularization.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        alpha: float = 0.1,
        ratio: float = 1,
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        tol: float = 1e-8,
    ):
        """
        Initializes the Elastic Net Regression model.

        Args:
            fit_intercept (bool): Whether to add an intercept term. Defaults to True.
            alpha (float): Regularization strength. Defaults to 0.1.
            ratio (float): Mixing ratio between L1 and L2. Defaults to 1.
            learning_rate (float): Learning rate for gradient descent. Defaults to 0.001.
            max_iter (int): Maximum number of iterations for optimization. Defaults to 1000.
            tol (float): Tolerance for convergence. Defaults to 1e-8.
        """
        self.regularization = ElasticNetRegularization(alpha=alpha, ratio=ratio)
        super().__init__(
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            learning_rate=learning_rate,
            regularization=self.regularization,
            tol=tol,
        )
