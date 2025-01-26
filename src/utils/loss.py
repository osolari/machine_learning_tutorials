from abc import abstractmethod

import numpy as np

from src.utils.activation import Sigmoid


class _Loss:
    """
    Base class for loss functions.

    Parameters
    ----------
    reduction : str, optional
        Specifies the reduction to apply to the output: 'mean' (default) or 'sum'.

    Attributes
    ----------
    reduction : str
        Reduction method for the computed loss.
    """

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    @abstractmethod
    def __call__(self, y, ypred):

        pass

    @abstractmethod
    def compute_gradient(self, y, ypred):

        pass

    @abstractmethod
    def compute_hessian(self, y, ypred):

        pass


class LogisticLoss(_Loss):
    """
    Logistic loss function with gradient and Hessian computations.

    The logistic loss is used for binary classification tasks and is defined as:

    .. math::
        \mathcal{L}(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]

    where:
        - \(y\) are the ground truth labels (0 or 1),
        - \(\hat{y}\) are the predicted probabilities,
        - \(N\) is the number of samples.

    The gradient and Hessian are derived as follows:

    Gradient:
    .. math::
        \frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \hat{y}_i - y_i

    Hessian:
    .. math::
        \frac{\partial^2 \mathcal{L}}{\partial \hat{y}_i^2} = \hat{y}_i (1 - \hat{y}_i)

    Methods
    -------
    __call__(y, y_pred)
        Compute the logistic loss.
    gradient(y, y_pred)
        Compute the gradient of the logistic loss with respect to predictions.
    hess(y, y_pred)
        Compute the Hessian of the logistic loss with respect to predictions.

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([0, 1, 1, 0])
    >>> y_pred = np.array([0.1, 0.9, 0.8, 0.3])
    >>> loss_func = LogisticLoss()
    >>> loss = loss_func(y_true, y_pred)
    >>> print("Logistic Loss:", loss)
    >>> grad = loss_func.compute_gradient(y_true, y_pred)
    >>> print("Gradient:", grad)
    >>> hess = loss_func.compute_hessian(y_true, y_pred)
    >>> print("Hessian:", compute_hessian)
    """

    def __init__(self):
        """
        Initialize the LogisticLoss object.
        """
        super().__init__()
        sigmoid = Sigmoid()
        self.func = sigmoid  # Sigmoid function
        self.gradient_func = sigmoid.gradient  # Gradient of the sigmoid function

    def __call__(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the logistic loss.

        Parameters
        ----------
        y : np.ndarray
            Ground truth labels, shape (n_samples,).
        y_pred : np.ndarray
            Predicted probabilities, shape (n_samples,).

        Returns
        -------
        np.ndarray
            Logistic loss for each sample.
        """
        # Clip predictions to avoid numerical instability
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p = self.func(y_pred)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    def compute_gradient(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the logistic loss with respect to predictions.

        Parameters
        ----------
        y : np.ndarray
            Ground truth labels, shape (n_samples,).
        y_pred : np.ndarray
            Predicted probabilities, shape (n_samples,).

        Returns
        -------
        np.ndarray
            Gradient of the logistic loss for each sample.

        Examples
        --------
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0.1, 0.9, 0.8, 0.3])
        >>> loss_func = LogisticLoss()
        >>> grad = loss_func.compute_gradient(y_true, y_pred)
        >>> print("Gradient:", grad)
        """
        p = self.func(y_pred)
        return -(y - p)

    def compute_hessian(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the Hessian of the logistic loss with respect to predictions.

        Parameters
        ----------
        y : np.ndarray
            Ground truth labels, shape (n_samples,).
        y_pred : np.ndarray
            Predicted probabilities, shape (n_samples,).

        Returns
        -------
        np.ndarray
            Hessian diagonal of the logistic loss for each sample.

        Examples
        --------
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0.1, 0.9, 0.8, 0.3])
        >>> loss_func = LogisticLoss()
        >>> hess = loss_func.compute_hessian(y_true, y_pred)
        >>> print("Hessian:", hess)
        """
        p = self.func(y_pred)
        return p * (1 - p)
