import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils import check_X_y, check_array

from src.utils.mlutils import compute_covariance_matrix, softmax


class _BaseDiscriminantAnalysis(
    BaseEstimator,
    ClassifierMixin,
    TransformerMixin,
):
    def __init__(self, n_components) -> None:

        self.n_components = n_components


class LinearDiscriminantAnalysis:
    """
    Linear Discriminant Analysis (LDA) classifier.

    LDA is a statistical method used for classification that projects high-dimensional data onto
    a lower-dimensional space with maximum class separability. It assumes that data is normally distributed
    with identical covariance matrices across classes.

    Theory
    ------
    LDA models the distribution of each class using a Gaussian distribution:

    .. math::
        P(X \mid y = k) \sim \mathcal{N}(\mu_k, \Sigma)

    The decision rule for classification is based on the posterior probability computed using Bayes' theorem:

    .. math::
        P(y = k \mid X) \propto P(X \mid y = k) P(y = k)

    Parameters
    ----------
    n_components : int, optional
        Number of components to keep. If None, all components are kept.
    tol : float, optional
        Tolerance for rank estimation of the covariance matrix. Default is 1e-4.

    Attributes
    ----------
    _class_priors : np.ndarray
        Prior probabilities of each class.
    _class_means : np.ndarray
        Means of each class.
    _covariance : np.ndarray
        Pooled covariance matrix.
    _coef : np.ndarray
        Coefficients for the linear decision function.
    _intercept : np.ndarray
        Intercepts for the linear decision function.

    Methods
    -------
    fit(X, y)
        Fit the LDA model to the training data.
    decision_function(X)
        Compute the linear decision scores for the input samples.
    predict_proba(X)
        Compute class probabilities using softmax on decision scores.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([0, 1, 0])
    >>> lda = LinearDiscriminantAnalysis()
    >>> lda.fit(X, y)
    >>> lda.predict_proba(X)
    array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    """

    def __init__(
        self,
        n_components=None,
        tol=1e-4,
    ):
        self.n_components = n_components
        self.tol = tol
        self._class_priors = None
        self._class_means = None
        self._covariance = None
        self._classes = None

    def fit(self, X, y=None):
        """
        Fit the LDA model.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray
            Target labels of shape (n_samples,).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y)
        self._compute_priors(y)
        self._compute_class_means(X, y)
        self._compute_weighted_class_covariances(X, y)
        self._compute_coefficients()
        return self

    def _compute_priors(self, y):
        """
        Compute class priors based on the label distribution.

        Priors are calculated as:

        .. math::
            P(y = k) = \frac{\text{count of class k}}{\text{total samples}}

        """
        self._class_count = np.bincount(y)
        self._classes = np.arange(len(self._class_count))
        self._class_priors = self._class_count / self._class_count.sum()

    def _compute_class_means(self, X, y):
        """
        Compute class means for each feature.

        The mean for each class is calculated as:

        .. math::
            \mu_k = \frac{1}{N_k} \sum_{i=1}^{N_k} X_i

        """
        self._class_means = np.vstack([X[y == yk].mean(axis=0) for yk in self._classes])

    def _compute_weighted_class_covariances(self, X, y):
        """
        Compute the pooled covariance matrix.

        The pooled covariance matrix is computed as:

        .. math::
            \Sigma = \sum_{k} P(y = k) \cdot \text{Cov}(X_k)
        """
        assert self._class_means is not None
        assert self._class_priors is not None

        self._covariance = sum(
            [
                self._class_priors[k]
                * self._compute_covariance(X[y == k] - self._class_means[k])
                for k in self._classes
            ]
        )

    @staticmethod
    def _compute_covariance(X):
        """
        Compute the covariance matrix of the given dataset.

        The covariance matrix is given by:

        .. math::
            \Sigma = \frac{1}{n} X^T X
        """
        cov = compute_covariance_matrix(X)
        return cov

    def decision_function(self, X):
        """
        Compute the decision function for input samples.

        Decision function computes the score:

        .. math::
            X W^T + b
        """
        X = check_array(X)
        return X @ self._coef.T + self._intercept

    def predict_proba(self, X):
        """
        Predict class probabilities.

        The predicted probabilities are computed using softmax transformation:

        .. math::
            P(y=k \mid X) = \frac{e^{f_k(X)}}{\sum_j e^{f_j(X)}}
        """
        X = check_array(X)
        scores = self.decision_function(X)
        return np.apply_along_axis(softmax, 1, scores)


class QuadraticDiscriminantAnalysis(_BaseDiscriminantAnalysis):
    def __init__(self, n_component=None):
        self.n_components = n_component
        super().__init__(n_component)
