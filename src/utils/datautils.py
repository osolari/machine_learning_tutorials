import random
from itertools import combinations_with_replacement

import numpy as np
from typing import Tuple, Union, Optional, List

from sklearn.base import (
    TransformerMixin,
    BaseEstimator,
)
from sklearn.utils import check_array


def bisect_array_on_feature(
    X: np.ndarray,
    feature_idx: int,
    cutoff: Union[float, int, str],
    return_mask_only: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Split a NumPy array into two subsets based on a condition applied to a specific feature column.

    The function divides the input array `X` into two subsets using a condition defined by the `cutoff` value
    applied to the feature at index `feature_idx`. Optionally, it can return only the mask indicating the split.

    Parameters
    ----------
    X : np.ndarray
        The input 2D array to split. Each row corresponds to an observation, and each column corresponds to a feature.
    feature_idx : int
        The index of the feature column to apply the cutoff condition.
    cutoff : Union[float, int, str]
        The cutoff value used to split the array. For numeric types (float or int), the condition is "greater than
        or equal to". For non-numeric types (e.g., strings), the condition is equality.
    return_mask_only : bool, optional
        If `True`, only the mask indicating which rows satisfy the condition is returned. Default is `False`.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]
        If `return_mask_only` is `True`, returns a 1D boolean array (mask) indicating which rows satisfy the condition.
        Otherwise, returns a tuple containing:
            - The subset of rows satisfying the condition.
            - The subset of rows not satisfying the condition.
            - `None` (as a placeholder for future extension).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> bisect_array_on_feature(X, feature_idx=0, cutoff=3)
    (array([[3, 4],
           [5, 6]]), array([[1, 2]]), None)

    >>> bisect_array_on_feature(X, feature_idx=1, cutoff=4, return_mask_only=True)
    array([False,  True,  True])

    >>> X = np.array([['apple', 'red'], ['banana', 'yellow'], ['apple', 'green']])
    >>> bisect_array_on_feature(X, feature_idx=0, cutoff='apple')
    (array([['apple', 'red'],
           ['apple', 'green']], dtype='<U6'),
     array([['banana', 'yellow']], dtype='<U6'), None)

    Notes
    -----
    - The input array `X` is assumed to have consistent data types along each column.
    - The function supports both numeric and string cutoffs. Ensure the data in the feature column matches the cutoff type.
    """

    # Determine the condition function based on the cutoff type
    if isinstance(cutoff, (float, int)):
        fn = lambda x: x >= cutoff  # Numeric cutoff: greater than or equal
    else:
        fn = lambda x: x == cutoff  # Non-numeric cutoff: equality

    # Apply the condition to split the array
    mask = fn(X[:, feature_idx])

    # Return the two subsets and optionally the mask
    if return_mask_only:
        return mask
    else:
        return X[mask], X[~mask], None


def one_hot_encoding(x: np.ndarray, n_classes: int = None) -> np.ndarray:
    """
    Perform one-hot encoding for a given array of class labels.

    Parameters
    ----------
    x : np.ndarray
        Array of class labels, shape (n_samples,). Each element should be an integer representing a class index.
    n_classes : int, optional
        The total number of classes. If None, it is inferred as `X.max() + 1`.

    Returns
    -------
    np.ndarray
        One-hot encoded array of shape (n_samples, n_classes). Each row corresponds to a sample, and each column
        corresponds to a class.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 2, 1, 3])
    >>> one_hot_encoding(x)
    array([[1., 0., 0., 0.],
           [0., 0., 1., 0.],
           [0., 1., 0., 0.],
           [0., 0., 0., 1.]])

    >>> x = np.array([1, 2, 3])
    >>> one_hot_encoding(x, n_classes=5)
    array([[0., 1., 0., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.]])

    Notes
    -----
    - The input array `X` should contain integer values representing class indices starting from 0.
    - If `n_classes` is smaller than `X.max() + 1`, an error will occur because some class indices cannot be encoded.
    - The function uses NumPy for efficient indexing and matrix creation.
    """
    if n_classes is None:
        # Infer the number of classes from the maximum value in X
        n_classes = x.max() + 1

    # Initialize the one-hot encoded matrix with zeros
    x_one_hot = np.zeros((x.shape[0], n_classes))

    # Set the appropriate indices to 1 for one-hot encoding
    x_one_hot[np.arange(x.shape[0]), x] = 1

    return x_one_hot


def aligned_shuffle(
    arr1: Union[List, np.ndarray], arr2: Union[List, np.ndarray], inplace: bool = False
) -> Union[None, Tuple[np.ndarray, np.ndarray]]:
    """
    Shuffle two arrays in the same random order.

    Parameters:
        arr1 (Union[List, np.ndarray]): The first array to shuffle.
        arr2 (Union[List, np.ndarray]): The second array to shuffle.
        inplace (bool): If True, shuffle arrays in place. If False, return new shuffled arrays.

    Returns:
        None if `inplace` is True.
        Tuple[np.ndarray, np.ndarray]: Shuffled copies of `arr1` and `arr2` if `inplace` is False.

    Raises:
        ValueError: If the input arrays have different lengths.

    Examples:
        >>> arr1 = [1, 2, 3, 4]
        >>> arr2 = ['a', 'b', 'c', 'd']
        >>> shuffled_arr1, shuffled_arr2 = aligned_shuffle(arr1, arr2)
        >>> len(arr1) == len(arr2)  # True

        >>> aligned_shuffle(arr1, arr2, inplace=True)
        >>> arr1, arr2  # Shuffled in place.
    """
    if len(arr1) != len(arr2):
        raise ValueError("Both arrays must have the same length.")

    combined = list(zip(arr1, arr2))  # Combine the arrays
    random.shuffle(combined)  # Shuffle the combined list

    if inplace:
        arr1[:], arr2[:] = zip(*combined)  # Modify original arrays in place
    else:
        arr1, arr2 = zip(*combined)  # Unzip into new arrays
        return np.array(arr1), np.array(arr2)


class PolynomialTransformer(TransformerMixin, BaseEstimator):
    """
    A transformer that generates polynomial features of the input data.

    This transformer computes all combinations of the input features up to a specified degree, including
    interactions and powers of individual features.

    Parameters
    ----------
    degree : int, optional
        The degree of the polynomial features to generate. Default is 1.

    Attributes
    ----------
    num_input_features : int
        The number of input features in the training data.
    _index_combinations : list of tuple
        The index combinations representing polynomial terms.
    _num_output_features : int
        The number of polynomial features generated.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data and compute index combinations.
    transform(X, y=None)
        Transform the input data into polynomial features.
    is_fitted()
        Check if the transformer has been fitted.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.validation import check_array
    >>> X = np.array([[1, 2], [3, 4]])
    >>> transformer = PolynomialTransformer(degree=2)
    >>> transformer.fit(X)
    PolynomialTransformer(degree=2)
    >>> transformer.transform(X)
    array([[ 1.,  2.,  1.,  2.,  4.],
           [ 3.,  4.,  9., 12., 16.]])
    """

    def __init__(self, degree: int = 1):
        if degree < 1:
            raise ValueError("Polynomial degree must be at least 1.")
        elif not isinstance(degree, int):
            raise ValueError("Polynomial degree must be an integer.")
        self.degree = degree

        self._index_combinations = None

    def _compute_index_combinations(self):
        """
        Compute all combinations of feature indices up to the specified degree.
        """
        combs = [
            combinations_with_replacement(range(self.num_input_features), i)
            for i in range(0, self.degree + 1)
        ]

        self._index_combinations = [item for sublist in combs for item in sublist]
        self._num_output_features = len(self._index_combinations)

    def fit(self, X, y=None):
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = check_array(X)

        num_samples, self.num_input_features = X.shape
        self._compute_index_combinations()

        return self

    def transform(self, X, y=None):
        """
        Transform the input data into polynomial features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.
        y : None
            Ignored.

        Returns
        -------
        X_transformed : array of shape (n_samples, n_output_features)
            Transformed data with polynomial features.
        """
        assert self.is_fitted(), "Transformer has not been fitted yet."

        X = check_array(X)
        if X.shape[1] != self.num_input_features:
            raise ValueError(
                "Input data must have the same number of features as the fitted data."
            )
        return np.hstack(
            [np.prod(X[:, i], axis=1, keepdims=True) for i in self._index_combinations]
        )

    def is_fitted(self):
        """
        Check if the transformer has been fitted.

        Returns
        -------
        bool
            True if the transformer has been fitted, False otherwise.
        """
        return self._index_combinations is not None
