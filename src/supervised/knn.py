"""
K-Nearest Neighbors (KNN) Classifier
------------------------------------

Theory and Explanation:
-----------------------
KNN is a non-parametric, instance-based learning algorithm used for classification. It predicts the label of a sample based on the labels of its K nearest neighbors in the feature space.

Algorithm Steps:
----------------
1. **Training**:
   - Simply store the training data and labels, as KNN does not explicitly learn a model.

2. **Prediction**:
   - For a given input sample, calculate the distance to all training samples.
   - Select the K closest samples based on the chosen distance metric (commonly Euclidean distance).
   - Predict the label by aggregating the labels of the K nearest neighbors (e.g., majority voting).

Advantages:
-----------
- Simple to understand and implement.
- No training phase, which makes it efficient for small datasets.

Disadvantages:
--------------
- Computationally expensive for large datasets due to the need to compute distances for all training samples.
- Sensitive to the choice of K and the distance metric.

"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y, check_consistent_length, check_array
from src.utils.mlutils import matrix_euclidean_distance


class KNN(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors Classifier.
    """

    def __init__(self, n_neighbors: int = 3):
        """
        Initializes the K-Nearest Neighbors Classifier.

        Args:
            n_neighbors (int): Number of nearest neighbors to consider. Defaults to 3.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        """
        Fits the KNN Classifier to the training data.

        Args:
            X (ArrayLike): A matrix of shape (n_samples, n_features) representing the training features.
            y (ArrayLike): An array of shape (n_samples,) representing the training labels.
        """
        X, y = check_X_y(X, y)

        self.X_train = X
        self.y_train = y

    def predict(self, X: ArrayLike) -> NDArray:
        """
        Predicts labels for the input data.

        Args:
            X (ArrayLike): A matrix of shape (n_samples, n_features) representing the input features.

        Returns:
            NDArray: Predicted labels for the input data.
        """
        X_checked = check_array(X)
        dist = matrix_euclidean_distance(self.X_train, X_checked)

        assert dist.shape == (
            self.X_train.shape[0],
            X_checked.shape[0],
        ), f"Distance matrix must be of the shape {(self.X_train.shape[0], X_checked.shape[0])}"

        labels = self.infer_labels(dist)

        return labels

    def infer_labels(self, dist: NDArray) -> NDArray:
        """
        Infers labels based on the distance matrix.

        Args:
            dist (NDArray): A matrix of shape (n_train_samples, n_test_samples) where dist[i, j] is the distance
                           between training sample i and test sample j.

        Returns:
            NDArray: Predicted labels for the test samples.
        """
        # Sort distances to find the indices of the K nearest neighbors
        dist = dist.argsort(axis=0)
        # Select the labels of the K nearest neighbors
        dist = self.y_train[dist][: self.n_neighbors, :]

        # Aggregate labels using bincount for majority voting
        labels = np.array(
            [
                np.bincount(dist.astype(int)[:, i], minlength=self.n_neighbors)
                for i in range(dist.shape[1])
            ]
        ).argmax(axis=1)

        return labels
