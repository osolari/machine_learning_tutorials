import copy
from random import shuffle

import cvxpy as cp
import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from typing import Tuple, Union, List, Optional

from src.utils.datautils import aligned_shuffle
from src.utils.kernels import polynomial, rbf, linear
import logging

logger = logging.getLogger(__name__)

r"""
Mathematical Derivations of the SVM Loss Function:

The Support Vector Machine (SVM) seeks to find a hyperplane that maximizes the margin between classes. This involves solving a quadratic optimization problem.

1. **Primal Formulation**:
   The primal objective function is defined as:

   \[
   \min_{w, b, xi} frac{1}{2} ||w||^2 + C \sum_{i=1}^N xi_i
   \]
   Subject to:
   \[
   y_i (w^T x_i + b) \geq 1 - xi_i, \quad xi_i \geq 0 \quad forall i
   \]

   Where:
   - \( w \): Weight vector defining the hyperplane.
   - \( b \): Bias term.
   - \( xi_i \): Slack variables allowing soft-margin violations.
   - \( C \): Regularization parameter controlling the trade-off between maximizing the margin and minimizing classification errors.

2. **Dual Formulation**:
   Using Lagrange multipliers, the optimization problem is transformed into the dual form:

   \[
   \max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j)
   \]
   Subject to:
   \[
   0 \leq \alpha_i \leq C, \quad \sum_{i=1}^N \alpha_i y_i = 0
   \]

   Where:
   - \( \alpha_i \): Lagrange multipliers.
   - \( K(x_i, x_j) \): Kernel function mapping inputs into a higher-dimensional space.

3. **Kernel Trick**:
   The kernel function \( K(x_i, x_j) \) allows computation in the feature space without explicitly performing the transformation:
   - Linear: \( K(x_i, x_j) = x_i^T x_j \)
   - Polynomial: \( K(x_i, x_j) = (x_i^T x_j + 1)^p \)
   - RBF: \( K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2) \)

This dual formulation is solved efficiently using quadratic programming techniques, as implemented in this SVM.
"""


class _BaseSVM(BaseEstimator, ClassifierMixin):
    """
    Abstract Base Class for Support Vector Machines (SVM).

    Attributes:
        C (float): Regularization parameter. Balances margin size and misclassification.
        kernel (str): Kernel type ('linear', 'rbf', or 'polynomial').
        p (int): Degree for polynomial kernel.
        gamma (float): Coefficient for RBF kernel.
        verbose (bool): Verbosity mode for debugging and logs.
        phi (callable): Kernel function.
    """

    __KERNELS__ = dict(linear=linear, rbf=rbf, polynomial=polynomial)

    def __init__(
        self,
        C: float,
        kernel: str,
        p: int,
        gamma: float,
        verbose: bool,
    ):
        if kernel not in _BaseSVM.__KERNELS__:
            raise ValueError(f"kernel must be one of {_BaseSVM.__KERNELS__.keys()}")
        self.kernel = kernel
        self.phi = self.__KERNELS__[kernel](p=p, gamma=gamma)

        self.C = C
        self.p = p
        self.gamma = gamma
        self.verbose = verbose

        self.X, self.y = None, None
        self._alphas = None

        self.multiclass = False
        self._classifiers = []

    def is_fitted(self) -> bool:
        """
        Check if the model has been fitted.

        Returns:
            bool: True if the model is fitted, False otherwise.
        """
        return self._alphas is not None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Abstract method to fit the SVM model to training data.

        Parameters:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels.
        """
        raise NotImplementedError("Fit method must be implemented by subclass.")


class SVMCVX(_BaseSVM):
    """
    SVM implementation using CVXPY for solving the quadratic programming problem.

    Attributes:
        num_classes (int): Number of unique classes in the dataset.
        classes (set): Set of unique class labels.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "linear",
        p: int = 2,
        gamma: float = 1.0,
        verbose: bool = False,
    ):
        super().__init__(C, kernel, p, gamma, verbose)
        self.num_classes = None
        self.classes = None

    def _setup_cvx_problem(self):
        """
        Set up the quadratic programming problem for SVM optimization.

        Returns:
            SVMCVX: Instance with problem initialized.
        """
        logger.info("Computing the kernel matrix ..")
        self.K = self.phi(self.X, self.X)

        P = self.y @ self.y.T * self.K
        q = -np.ones((self.num_samples, 1))

        A = self.y.T
        b = np.zeros(1)

        G = np.concatenate(
            (-np.identity(self.num_samples), np.identity(self.num_samples))
        )
        h = np.concatenate(
            (np.zeros((self.num_samples, 1)), self.C * np.ones((self.num_samples, 1)))
        )

        self._alphas = cp.Variable(self.num_samples)

        self.prob = cp.Problem(
            cp.Minimize(0.5 * cp.quad_form(self._alphas, P) + q.T @ self._alphas),
            [G @ self._alphas <= h, A @ self._alphas == b],
        )

        return self

    def _fit_binary(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM for binary classification.

        Parameters:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels (binary).

        Returns:
            SVMCVX: Fitted SVM instance.
        """
        self.num_samples, self.num_features = X.shape
        self.X, self.y = X, y
        self._setup_cvx_problem()

        logger.info("Solving the QP problem ..")
        self.prob.solve(solver="CVXOPT", verbose=self.verbose)

        if self.prob.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")

        self._alphas = self._alphas.value

        self._is_sv = ((self._alphas > 1e-3) & (self._alphas <= self.C)).squeeze()
        self._margin_sv = np.argmax(
            (0 < self._alphas - 1e-3) & (self._alphas < self.C - 1e-3)
        )
        self.__is_sv = self.prob.constraints[0].dual_value

        return self

    def _fit_multiclass(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM for multi-class classification using one-vs-rest strategy.

        Parameters:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels.
        """
        self.num_samples, self.num_features = X.shape
        for i in range(self.num_classes):
            Xs, Ys = X, copy.copy(y)
            Ys[Ys != i], Ys[Ys == i] = -1, +1
            clf = SVMCVX(
                kernel=self.kernel,
                C=self.C,
            )
            clf._fit_binary(Xs, Ys)
            self._classifiers.append(clf)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM model to the training data.

        Parameters:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels.
        """
        X, y = check_X_y(X, y)
        y = y.reshape(-1, 1).astype(np.double)

        self.classes = set(y.flatten())
        self.num_classes = len(self.classes)
        if self.num_classes > 2:
            self.multiclass = True
            self._fit_multiclass(X=X, y=y)
        elif self.num_classes == 2:
            self.multiclass = False
            if self.classes == {0, 1}:
                y[y == 0] = -1
            self._fit_binary(X=X, y=y)
        else:
            raise ValueError(
                f"Number of classes must be larger than 2: {self.num_classes}"
            )

    def predict(self, X: np.ndarray) -> ndarray | tuple[ndarray, ndarray]:
        """
        Predict class labels for the input data.

        Parameters:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        X = check_array(X)
        if self.multiclass:
            return self._predict_multiclass(X)
        else:
            return self._predict_binary(X)

    def _predict_binary(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class labels for binary classification.

        Parameters:
            X (np.ndarray): Input feature matrix.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted class labels and decision scores.
        """
        Xs, ys = self.X[self._margin_sv, np.newaxis], self.y[self._margin_sv]
        alphas, Xt, yt = (
            self._alphas[self._is_sv],
            self.X[self._is_sv],
            self.y[self._is_sv],
        )
        b = ys - np.sum(alphas * yt * self.phi(Xt, Xs), axis=0)
        score = np.sum(alphas * yt * self.phi(Xt, X), axis=0) + b

        return np.sign(score).astype(int), score

    def _predict_multiclass(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for multi-class classification.

        Parameters:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        predictions = np.zeros((X.shape[0], self.num_classes))
        for i, clf in enumerate(self._classifiers):
            _, predictions[:, i] = clf._predict_binary(X)

        return np.argmax(predictions, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the SVM model on a test dataset.

        Parameters:
            X (np.ndarray): Test feature matrix.
            y (np.ndarray): True labels.

        Returns:
            float: Accuracy of the model.
        """
        outputs, _ = self.predict(X)
        accuracy = np.sum(outputs == y) / len(y)
        return round(accuracy, 2)


r"""
Mathematical Derivation of Stochastic Gradient Descent (SGD) for SVM:

Stochastic Gradient Descent (SGD) is used to optimize the loss function of Support Vector Machines (SVM).

1. **Hinge Loss**:
   The hinge loss is defined as:
   \[
   L(w, b) = \frac{1}{2} ||w||^2 + C \sum_{i=1}^N \max(0, 1 - y_i (w^T x_i + b))
   \]
   Where:
   - \( w \): Weight vector.
   - \( b \): Bias term.
   - \( C \): Regularization parameter.
   - \( y_i \): Label for sample \( i \).
   - \( x_i \): Feature vector for sample \( i \).

2. **Gradient of the Loss**:
   The gradient of the hinge loss with respect to \( w \) is computed as:
   \[
   \nabla_w L(w, b) = w - C \sum_{i=1}^N \mathbb{1}(1 - y_i (w^T x_i + b) > 0) (-y_i x_i)
   \]
   Where \( \mathbb{1} \) is an indicator function that is 1 if the condition is true, and 0 otherwise.

3. **SGD Update Rule**:
   For a single training example \( (x_i, y_i) \), the update rule is:
   \[
   w \leftarrow w - \eta \nabla_w L(w, b)
   \]
   Where:
   - \( \eta \): Learning rate.

   Breaking it into cases:
   - If \( 1 - y_i (w^T x_i + b) \leq 0 \):
     \[
     \nabla_w L(w, b) = w
     \]
   - Otherwise:
     \[
     \nabla_w L(w, b) = w - C y_i x_i
     \]

4. **Bias Term (Optional)**:
   The bias term \( b \) can be updated similarly using SGD:
   \[
   b \leftarrow b - \eta \nabla_b L(w, b)
   \]
   Where the gradient with respect to \( b \) is:
   \[
   \nabla_b L(w, b) = -C \sum_{i=1}^N \mathbb{1}(1 - y_i (w^T x_i + b) > 0) (-y_i)
   \]

SGD iteratively updates \( w \) and \( b \) using these gradients until convergence.

This module implements an SVM optimized via SGD, utilizing these principles.
"""


class SVMSGD(_BaseSVM):
    """
    Support Vector Machine using Stochastic Gradient Descent (SGD).

    Attributes:
        C (float): Regularization parameter.
        bias (bool): Whether to include a bias term in the model.
        kernel (str): Kernel type ('linear', 'rbf', 'polynomial').
        p (int): Degree for polynomial kernel.
        gamma (float): Coefficient for RBF kernel.
        lr (float): Learning rate for SGD.
        max_iter (int): Maximum number of SGD iterations.
        tol (float): Tolerance for convergence.
        verbose (bool): Verbosity mode for logging.
    """

    def __init__(
        self,
        C: float = 1.0,
        bias: bool = True,
        kernel: str = "linear",
        p: int = 2,
        gamma: float = 1.0,
        lr: float = 1e-3,
        max_iter: int = 1000,
        tol: float = 1e-3,
        verbose: bool = False,
    ):
        super().__init__(C, kernel, p, gamma, verbose)

        self.bias = bias
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

        self.num_features = None
        self.num_samples = None
        self.num_classes = None
        self.classes = None

    def _compute_loss(self, X: np.ndarray, y: np.ndarray, alpha: np.ndarray) -> float:
        """
        Compute the hinge loss for the current model.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
            alpha (np.ndarray): Model weights.

        Returns:
            float: Computed loss.
        """
        self.num_samples, self.num_features = X.shape
        dist = 1 - y * (X @ alpha)
        dist[dist < 0] = 0  # Only positive margins contribute to the hinge loss
        hinge_loss = self.C * dist.sum() / self.num_samples
        loss = 0.5 * alpha.T @ alpha + hinge_loss  # Regularization + hinge loss
        return loss

    def compute_gradient(
        self, X: np.ndarray, y: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function with respect to model weights.

        Parameters:
            X (np.ndarray): Feature vector or matrix.
            y (np.ndarray): Labels.
            alpha (np.ndarray): Current model weights.

        Returns:
            np.ndarray: Gradient of the loss.
        """
        if isinstance(y, float):  # Handle single example case
            y = np.array([y])
            X = np.array([X])
        distance = 1 - (y * np.dot(X, alpha))
        dw = np.zeros_like(alpha)
        for ind, d in enumerate(distance):
            if max(0, d) == 0:  # No contribution if the margin is satisfied
                di = alpha
            else:
                di = alpha - (self.C * y[ind] * X[ind])
            dw += di
        dw /= len(y)  # Average the gradient
        return dw

    def _stochastic_gradient_descent(
        self, X: np.ndarray, y: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        """
        Perform one epoch of stochastic gradient descent.

        Parameters:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Labels.
            alpha (np.ndarray): Current model weights.

        Returns:
            np.ndarray: Updated model weights.
        """
        X, y = aligned_shuffle(X, y)  # Shuffle the data
        for xi, yi in zip(X, y):
            descent = self.compute_gradient(xi, yi, alpha)
            alpha -= self.lr * descent  # Update weights
        return alpha

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the SVM model to the training data.

        Parameters:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels.
        """
        X, y = check_X_y(X, y)
        y = y.reshape(-1, 1).astype(np.double)

        self.classes = set(y.flatten())
        self.num_classes = len(self.classes)
        self.num_samples, self.num_features = X.shape

        if self.bias:
            X = np.concatenate([np.ones([self.num_samples, 1]), X], axis=1)

        alpha = np.zeros(self.num_features)
        old_loss = float("inf")

        for i in range(self.max_iter):
            alpha = self._stochastic_gradient_descent(X, y, alpha)

            if self.verbose and (i % 100 == 0 or i == self.max_iter - 1):
                loss = self._compute_loss(X, y, alpha)
                logger.info(f"Iteration {i}: {loss}")
                if self.converged(loss, old_loss):
                    logger.info(f"*** Converged at iteration {i}: {loss} ***")
                    self._alpha = alpha
                    return
                old_loss = loss
        self._alpha = alpha

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input data.

        Parameters:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted class labels.
        """
        if self.bias:
            X = np.concatenate([np.ones([X.shape[0], 1]), X], axis=1)
        yhat = X @ self._alpha
        return np.sign(yhat).astype(int)

    def converged(self, loss: float, old_loss: float) -> bool:
        """
        Check for convergence of the SGD algorithm.

        Parameters:
            loss (float): Current loss.
            old_loss (float): Previous loss.

        Returns:
            bool: True if the algorithm has converged, False otherwise.
        """
        return abs((loss - old_loss) / old_loss) < self.tol


r"""
Mathematical Derivation of Pegasos SVM:

The Pegasos (Primal Estimated sub-GrAdient SOlver for SVM) algorithm minimizes the hinge loss and L2 regularization penalty:

Objective function:
\[
L(w) = \frac{1}{2} ||w||^2 + C \sum_{i=1}^N \max(0, 1 - y_i (w^T \phi(x_i)))
\]

Where:
- \( w \): Weight vector.
- \( C \): Regularization parameter.
- \( y_i \): Label for sample \( i \).
- \( \phi(x_i) \): Kernel mapping for sample \( i \).

Update Rule:
1. Compute learning rate \( \eta_t = \frac{1}{\lambda t} \).
2. For each misclassified sample \( i \) (i.e., \( y_i (w^T \phi(x_i)) < 1 \)):
   \[
   \alpha_i \leftarrow \alpha_i + \eta_t (y_i - \lambda \alpha_i)
   \]

Convergence:
The algorithm converges when the change in \( \alpha \) values between iterations is below a specified tolerance.
"""


class SVMPegasos(_BaseSVM):
    """
    Support Vector Machine using the Pegasos algorithm for optimization.

    Attributes:
        kernel (str): Kernel type ('linear', 'rbf', or 'polynomial').
        C (float): Regularization parameter.
        fit_intercept (bool): Whether to include an intercept term.
        p (int): Degree for polynomial kernel.
        gamma (float): Coefficient for RBF kernel.
        max_iter (int): Maximum number of iterations for training.
        tol (float): Tolerance for convergence.
        verbose (bool): Verbosity mode for logging.
    """

    def __init__(
        self,
        kernel: str = "linear",
        C: float = 100,
        fit_intercept: bool = True,
        p: int = 2,
        gamma: float = 1,
        max_iter: int = 100,
        tol: float = 1e-3,
        verbose: bool = False,
    ):
        """
        Initialize the Pegasos SVM model.

        Parameters:
            kernel (str): Kernel type ('linear', 'rbf', 'polynomial').
            C (float): Regularization parameter.
            fit_intercept (bool): Whether to include an intercept term.
            p (int): Degree for polynomial kernel.
            gamma (float): Coefficient for RBF kernel.
            max_iter (int): Maximum number of iterations.
            tol (float): Tolerance for convergence.
            verbose (bool): Verbosity mode for debugging.
        """
        super().__init__(C=C, kernel=kernel, p=p, gamma=gamma, verbose=verbose)

        self._alphas = None
        self.K = None
        self.num_features = None
        self.num_samples = None
        if kernel not in _BaseSVM.__KERNELS__:
            raise ValueError(f"kernel must be one of {_BaseSVM.__KERNELS__.keys()}")
        self.kernel = kernel
        self.phi = self.__KERNELS__[kernel](p=p, gamma=gamma)

        self.C = C
        self.fit_intercept = fit_intercept
        self.p = p
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        if self.gamma > 0:
            self._lambda = 1 / self.gamma
        else:
            self._lambda = 1

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the Pegasos SVM model.

        Parameters:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels.

        Returns:
            SVMPegasos: The trained Pegasos SVM instance.
        """
        X, y = check_X_y(X, y)
        if self.fit_intercept:
            X = np.c_[X, np.ones((X.shape[0], 1))]  # Add bias column

        self.X, self.y = X, y
        self.num_samples, self.num_features = X.shape
        self.K = self.phi(X, X)  # Kernel matrix

        alphas = np.zeros(self.num_samples)
        alphas_old = np.array([float("inf")] * self.num_samples)
        for t in range(1, self.max_iter + 1):
            eta = 1 / (self._lambda * t)  # Learning rate
            decision = (alphas * y) @ self.K  # Decision function

            # Identify misclassified samples
            mistakes = y * decision < 1
            alphas[mistakes] += eta * (y[mistakes] - self._lambda * alphas[mistakes])

            if self.converged(alphas, alphas_old):
                logger.info(f"*** Converged at iteration {t} ***")
                break

            alphas_old = alphas.copy()

        self._alphas = alphas
        return self

    def converged(self, alpha: np.ndarray, alpha_old: np.ndarray) -> bool:
        """
        Check for convergence of the Pegasos algorithm.

        Parameters:
            alpha (np.ndarray): Current alpha values.
            alpha_old (np.ndarray): Previous alpha values.

        Returns:
            bool: True if the algorithm has converged, False otherwise.
        """
        return ((alpha - alpha_old) ** 2).sum() < self.tol

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input data.

        Parameters:
            X (np.ndarray): Input feature matrix.

        Returns:
            np.ndarray: Predicted class labels.

        Raises:
            AssertionError: If the model has not been fitted before calling predict.
        """
        assert self.is_fitted(), f"Model has not been fitted."

        # Validate and preprocess input data
        X = check_array(X)
        if self.fit_intercept:
            X = np.c_[X, np.ones((X.shape[0], 1))]  # Add bias column

        # Compute kernel matrix between input data and training data
        K = self.phi(X, self.X)

        # Compute decision function
        decision = (self._alphas * self.y) @ K.T

        # Return predicted labels based on the sign of the decision function
        return np.sign(decision)
