"""
K-Means Clustering Implementation
---------------------------------

Theory and Explanation:
-----------------------
K-Means is an unsupervised learning algorithm used for clustering data into a predefined number of clusters (K). It aims to partition the dataset into K clusters such that the within-cluster sum of squares (WCSS) is minimized.

Algorithm Steps:
----------------
1. **Initialization**:
   - Randomly select K initial centroids from the dataset.

2. **Iteration**:
   - **Assignment Step**: Assign each data point to the nearest centroid based on a distance metric (commonly Euclidean distance).
   - **Update Step**: Compute new centroids as the mean of the points assigned to each cluster.

3. **Convergence**:
   - The algorithm stops if the centroids do not change significantly between iterations or if the maximum number of iterations is reached.

Advantages:
-----------
- Simple to understand and implement.
- Scales well to large datasets.

Disadvantages:
--------------
- Sensitive to the choice of initial centroids (may converge to a local minimum).
- Requires the number of clusters (K) to be specified beforehand.
- Not suitable for non-spherical clusters or clusters of varying densities.

"""

import numpy as np
from sklearn.base import TransformerMixin, ClusterMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array
from numpy.typing import NDArray, ArrayLike
from src.utils.mlutils import matrix_euclidean_distance
import seaborn as sns
import matplotlib.pyplot as plt


class KMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """
    K-Means Clustering Algorithm.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        max_iter: int = 100,
        tol: float = 0.0001,
        random_state: int = None,
    ):
        """
        Initializes the K-Means clustering model.

        Args:
            n_clusters (int): Number of clusters. Defaults to 2.
            max_iter (int): Maximum number of iterations. Defaults to 100.
            tol (float): Tolerance for convergence. Defaults to 0.0001.
            random_state (int): Seed for random number generator. Defaults to None.
        """
        self._labels = None
        self._centroids = None

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: ArrayLike) -> "KMeans":
        """
        Fits the K-Means model to the input data.

        Args:
            X (ArrayLike): Input data of shape (n_samples, n_features).

        Returns:
            KMeans: The fitted model.
        """
        X = check_array(X)
        centroids = self._init_centroids_from_random_samples(X)

        for _ in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)

            old_centroids = centroids
            centroids = self._compute_centroids(X, labels)

            if self._converged(old_centroids, centroids):
                print("CONVERGED!")
                break

        self._centroids = centroids
        self._labels = self._assign_clusters(X, centroids)

        return self

    def predict(self, X: NDArray) -> NDArray:
        """
        Predicts the cluster labels for the input data.

        Args:
            X (NDArray): Input data of shape (n_samples, n_features).

        Returns:
            NDArray: Cluster labels for each sample.
        """
        self._is_fitted()
        X = check_array(X)

        return self._assign_clusters(X, self._centroids)

    def _init_centroids_from_random_samples(self, X: NDArray) -> NDArray:
        """
        Initializes centroids by randomly sampling data points.

        Args:
            X (NDArray): Input data of shape (n_samples, n_features).

        Returns:
            NDArray: Initial centroids of shape (n_clusters, n_features).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)

        return X[idx]

    def _assign_clusters(self, X: NDArray, centroids: NDArray) -> NDArray:
        """
        Assigns each data point to the nearest cluster.

        Args:
            X (NDArray): Input data of shape (n_samples, n_features).
            centroids (NDArray): Centroids of shape (n_clusters, n_features).

        Returns:
            NDArray: Cluster labels for each sample.
        """
        assert len(centroids) == self.n_clusters

        return matrix_euclidean_distance(X, centroids).argmin(axis=1).reshape([-1, 1])

    @staticmethod
    def _compute_centroids(X: NDArray, labels: NDArray) -> NDArray:
        """
        Computes the new centroids based on current cluster assignments.

        Args:
            X (NDArray): Input data of shape (n_samples, n_features).
            labels (NDArray): Cluster labels for each sample.

        Returns:
            NDArray: New centroids of shape (n_clusters, n_features).
        """
        encoder = OneHotEncoder(sparse_output=False)
        labels = encoder.fit_transform(labels)
        sums = labels.sum(axis=0)
        sums[sums == 0] = 1  # Avoid division by zero
        labels /= sums

        return labels.T @ X

    @staticmethod
    def _converged(old_centroids: NDArray, new_centroids: NDArray) -> bool:
        """
        Checks if the centroids have converged.

        Args:
            old_centroids (NDArray): Centroids from the previous iteration.
            new_centroids (NDArray): Centroids from the current iteration.

        Returns:
            bool: True if centroids have not changed significantly, False otherwise.
        """
        return np.allclose(old_centroids, new_centroids, atol=1e-8)

    def _is_fitted(self) -> None:
        """
        Ensures that the model is fitted before making predictions.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if self._centroids is None:
            raise ValueError("KMeans is not fitted")

    @staticmethod
    def plot_2d(ax: plt.Axes, X: NDArray, **kwargs) -> plt.Axes:
        """
        Plots the K-Means clustering results in 2D.

        Args:
            ax (plt.Axes): Matplotlib Axes object for plotting.
            X (NDArray): Input data of shape (n_samples, 2).
            **kwargs: Additional arguments for the KMeans model.

        Returns:
            plt.Axes: The updated Axes object with the plot.
        """
        if X.shape[1] != 2:
            raise ValueError(
                "plot_2d only supports 2-dimensional data. Consider dimensionality reduction or selecting two features."
            )

        kmeans = KMeans(**kwargs)
        kmeans.fit(X)

        sns.scatterplot(
            x=X[:, 0], y=X[:, 1], hue=kmeans._labels.flatten(), ax=ax, palette="viridis"
        )

        return ax
