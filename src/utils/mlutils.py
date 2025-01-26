from collections import Counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp
from sklearn.utils import check_consistent_length, check_array
from typing import Union, List, Any


def euclidean_distance(point1: ArrayLike, point2: ArrayLike) -> float:
    """
    Computes the Euclidean distance between two points.
    :param point1: a length n vector
    :param point2: a length n vector
    :return: float
    """

    point1 = np.array(point1)
    point2 = np.array(point2)

    check_consistent_length(point1, point2)

    return np.sqrt(sum((point1 - point2) ** 2))


def matrix_euclidean_distance(X1: ArrayLike, X2: ArrayLike) -> NDArray:
    """
    Computes the Euclidean distance matrix between two matrices.
    :param X1: n1xp matrix
    :param X2: n2xp matrix
    :return: a n1xn2 matrix
    """

    X1 = np.array(X1)
    X2 = np.array(X2)

    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same number of columns")

    return np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=-1))


def compute_entropy(vals: Union[NDArray, list]) -> float:
    """
    Computes the entropy of a given set of values.

    Entropy is a measure of uncertainty or impurity in the data, calculated as:
        H = -Σ(p * log(p))
    where `p` is the probability of each unique value in the data.

    Args:
        vals (Union[NDArray, list]): An array or list of values for which entropy is computed.

    Returns:
        float: The entropy of the input values.

    Raises:
        ValueError: If the input array or list is empty.

    Example:
        >>> compute_entropy([1, 1, 2, 2, 2])
        0.6730116670092565
        >>> compute_entropy([1, 1, 1, 1])
        0.0
    """
    if len(vals) == 0:
        raise ValueError("Input 'vals' must not be empty.")

    # Count occurrences of each unique value
    counts = Counter(vals)
    total = len(vals)

    # Calculate probabilities of each unique value
    p = np.array([count / total for count in counts.values()])

    # Compute entropy using the formula H = -Σ(p * log(p))
    return -np.sum(p * np.log(p))


def compute_information_gain(y: NDArray, yl: NDArray, yr: NDArray) -> float:
    """
    Compute the information gain resulting from splitting a dataset.

    Information gain measures the reduction in entropy achieved by splitting a dataset
    into two subsets. It is used to select the optimal split in decision tree algorithms.

    Theory
    ------
    Information Gain (IG) is calculated as:
        IG = H(y) - [ (|yl| / |y|) * H(yl) + (|yr| / |y|) * H(yr) ]
    where:
        - H(y): Entropy of the original dataset.
        - H(yl): Entropy of the left subset after the split.
        - H(yr): Entropy of the right subset after the split.
        - |y|: Total number of samples in the dataset.
        - |yl|, |yr|: Number of samples in the left and right subsets, respectively.

    Parameters
    ----------
    y : NDArray
        Original target variable (before splitting).
    yl : NDArray
        Target variable of the left subset (after splitting).
    yr : NDArray
        Target variable of the right subset (after splitting).

    Returns
    -------
    float
        The information gain resulting from the split. If one of the subsets is empty,
        the information gain is 0.

    Examples
    --------
    >>> y = np.array([1, 1, 0, 0, 1])
    >>> yl = np.array([1, 1])
    >>> yr = np.array([0, 0, 1])
    >>> compute_information_gain(y, yl, yr)
    0.3219280948873623
    """
    n = len(y)
    total_entropy = compute_entropy(y)

    nl, nr = len(yl), len(yr)

    # If one of the subsets is empty, return 0 (invalid split)
    if not (nl and nr):
        return 0

    # Compute entropy for the left and right subsets
    ent_l = compute_entropy(yl)
    ent_r = compute_entropy(yr)

    # Weighted average of subset entropies
    weighted_entropy = (nl * ent_l + nr * ent_r) / n

    # Information gain is the reduction in entropy
    return total_entropy - weighted_entropy


def compute_majority_class(y: np.ndarray) -> Any:
    """
    Compute the majority class from the target values.

    Parameters
    ----------
    y : np.ndarray
        Target values, shape (n_samples,).

    Returns
    -------
    Any
        The majority class in the target values.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0, 1, 1, 1, 0])
    >>> compute_majority_class(y)
    1

    Notes
    -----
    - If there is a tie, the smallest class label is returned.
    """
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]


def compute_variance_reduction(y: np.ndarray, yl: np.ndarray, yr: np.ndarray) -> float:
    """
    Compute the variance reduction achieved by splitting a dataset into two subsets.

    Variance reduction is a commonly used metric in decision tree algorithms to evaluate the quality
    of a split. It measures the decrease in variance of the target variable after splitting the dataset
    into two subsets.

    Parameters
    ----------
    y : np.ndarray
        The original dataset of target values. It should be a 1D or 2D array.
    yl : np.ndarray
        The subset of target values corresponding to the left child after a split. It should have the same
        dimensionality as `y`.
    yr : np.ndarray
        The subset of target values corresponding to the right child after a split. It should have the same
        dimensionality as `y`.

    Returns
    -------
    float
        The total variance reduction achieved by the split. It is a scalar value even for multi-dimensional inputs.

    Theory
    ------
    Variance reduction is calculated as:

        VarianceReduction = TotalVariance - WeightedAverageVariance

    where:
        - TotalVariance is the variance of the target variable before splitting.
        - WeightedAverageVariance is the weighted average of the variances of the left and right subsets,
          with weights proportional to the sizes of the subsets.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> yl = np.array([1, 2, 3])
    >>> yr = np.array([4, 5])
    >>> compute_variance_reduction(y, yl, yr)
    2.0

    >>> y = np.array([[1, 2], [3, 4], [5, 6]])
    >>> yl = np.array([[1, 2], [3, 4]])
    >>> yr = np.array([[5, 6]])
    >>> compute_variance_reduction(y, yl, yr)
    5.333333333333333

    Notes
    -----
    - This function assumes that the input arrays are properly split and their dimensions are consistent.
    - It calculates variance along the first axis (rows).
    - The result is aggregated using `sum`, producing a scalar output regardless of the dimensionality of `y`.

    """
    # Compute total variance of the original dataset
    total_variance = y.var(axis=0)

    # Get the total number of elements in the original dataset
    n = len(y)

    # Compute variances of the left and right subsets
    var_l, var_r = yl.var(axis=0), yr.var(axis=0)

    # Get the sizes of the left and right subsets
    n_l, n_r = len(yl), len(yr)

    # Compute and return the variance reduction, summed to produce a scalar
    return sum(total_variance - (n_l * var_l + n_r * var_r) / n)


def log_softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute the log-softmax of an input array.

    Parameters:
    x (np.ndarray): Input array for which to compute the log-softmax.

    Returns:
    np.ndarray: Array with the log-softmax values of the input.

    Theory:
    The log-softmax function is defined as:
        log_softmax(x)_i = x_i - log(sum(exp(x)))
    This is useful for numerical stability when working with probabilities in log-space.
    """
    x = x - x.max()  # Shift for numerical stability
    _log_sum_exp = logsumexp(x)  # Compute log(sum(exp(x)))
    x -= _log_sum_exp  # Subtract the log of the sum of exponentials
    return x


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute the softmax of an input array.

    Parameters:
    x (np.ndarray): Input array for which to compute the softmax.

    Returns:
    np.ndarray: Array with the softmax values of the input.

    Theory:
    The softmax function is defined as:
        softmax(x)_i = exp(x_i) / sum(exp(x))
    This is a common function in machine learning to convert scores into probabilities.
    """
    x = log_softmax(x)  # Use log-softmax for numerical stability
    return np.exp(x)


def mean_squared_error(y_true, y_pred):
    """Returns the mean squared error between y_true and y_pred"""
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


import numpy as np
from sklearn.utils.validation import check_array


def compute_covariance_matrix(X: np.ndarray, centered: bool = False) -> np.ndarray:
    """
    Compute the covariance matrix of a dataset.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix of shape (n_samples, n_features).
    centered : bool, optional
        If False (default), the function centers the data by subtracting the mean.
        If True, it assumes that the data is already centered.

    Returns
    -------
    np.ndarray
        The covariance matrix of shape (n_features, n_features).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> compute_covariance_matrix(X)
    array([[4., 4.],
           [4., 4.]])

    >>> X_centered = X - X.mean(axis=0)
    >>> compute_covariance_matrix(X_centered, centered=True)
    array([[4., 4.],
           [4., 4.]])

    Notes
    -----
    - The function assumes the input is a 2D NumPy array.
    - The computed covariance matrix is normalized by the number of samples - 1,
      following the unbiased estimator convention.
    """
    X = check_array(X, ensure_2d=True)
    if not centered:
        X = X - np.mean(X, axis=0, keepdims=True)

    # Compute the covariance matrix using the unbiased estimator (n_samples - 1)
    cov = (X.T @ X) / (X.shape[0] - 1)
    return cov
