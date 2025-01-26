import logging
from types import NoneType
from typing import Optional

import numpy as np
from sklearn.base import DensityMixin, BaseEstimator
from sklearn.utils import check_array

from src.utils.dist import MultiVariateGaussian

logging.basicConfig(format="[%(asctime)s]-[%(name)-1s-%(levelname)2s]: %(message)s")
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

"""
https://github.com/zf109/algorithm_practice/blob/master/gaussian_mixture_model/gaussian_mixture_model/gaussian_mixture_model.pdf
"""


class _BaseMixtureModel:
    """
    Base class for mixture models.

    Parameters
    ----------
    n_components : int
        Number of mixture components.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.
    n_init : int
        Number of initializations to perform.
    init_params : str
        Method for initializing parameters ('kmeans', 'random').
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress messages.
    """

    def __init__(
        self,
        n_components,
        tol,
        max_iter,
        n_init,
        init_params,
        random_state,
        verbose,
    ):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.verbose = verbose


class GMM(_BaseMixtureModel):
    """
    Gaussian Mixture Model (GMM) implementation using the Expectation-Maximization (EM) algorithm.

    GMM models a dataset as a mixture of multiple Gaussian distributions, allowing for clustering and density estimation.

    The log-likelihood function for GMM is given by:

    .. math::
        \log P(X \mid \theta) = \sum_{i=1}^{N} \log \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)

    where:
        - \( \pi_k \) are the mixture weights,
        - \( \mathcal{N}(x_i \mid \mu_k, \Sigma_k) \) is the Gaussian density function,
        - \( \mu_k \) and \( \Sigma_k \) are the mean and covariance of component \( k \).

    The Expectation-Maximization algorithm consists of two main steps:

    1. **Expectation (E-step):** Compute the responsibilities for each data point using current estimates of parameters.

       .. math::
            \gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}

    2. **Maximization (M-step):** Update parameters using the computed responsibilities.

       .. math::
            \mu_k = \frac{\sum_{i=1}^{N} \gamma_{ik} x_i}{\sum_{i=1}^{N} \gamma_{ik}}

       .. math::
            \Sigma_k = \frac{\sum_{i=1}^{N} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^{N} \gamma_{ik}}

       .. math::
            \pi_k = \frac{\sum_{i=1}^{N} \gamma_{ik}}{N}

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components (clusters).
    tol : float, default=1e-3
        Convergence threshold for stopping criteria.
    max_iter : int, default=100
        Maximum number of EM iterations.
    n_init : int, default=1
        Number of initializations to perform.
    init_params : str, default="kmeans"
        Initialization method ('kmeans' or 'random').
    random_state : Optional[int], optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Verbosity mode for debugging output.
    """

    def __init__(
        self,
        n_components: int = 1,
        tol: float = 1e-3,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state: Optional[int] = None,
        verbose: bool = False,
    ):
        super(GMM, self).__init__(
            n_components, tol, max_iter, n_init, init_params, random_state, verbose
        )
        self._mixtures: Optional[np.ndarray] = None  # Mixture weights
        self._means: Optional[np.ndarray] = None  # Mean of each Gaussian component
        self._covariances: Optional[np.ndarray] = (
            None  # Covariance matrices for each component
        )
        self.num_features: Optional[int] = None  # Number of features in the dataset

    import numpy as np
    from sklearn.utils.validation import check_array
    from typing import Optional, Union

    class GMM(_BaseMixtureModel):
        """
        Gaussian Mixture Model (GMM) implementation using the Expectation-Maximization (EM) algorithm.

        GMM models a dataset as a mixture of multiple Gaussian distributions, allowing for clustering and density estimation.

        The log-likelihood function for GMM is given by:

        .. math::
            \log P(X \mid \theta) = \sum_{i=1}^{N} \log \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)

        where:
            - \( \pi_k \) are the mixture weights,
            - \( \mathcal{N}(x_i \mid \mu_k, \Sigma_k) \) is the Gaussian density function,
            - \( \mu_k \) and \( \Sigma_k \) are the mean and covariance of component \( k \).

        Parameters
        ----------
        n_components : int, default=1
            The number of mixture components (clusters).
        tol : float, default=1e-3
            Convergence threshold for stopping criteria.
        max_iter : int, default=100
            Maximum number of EM iterations.
        n_init : int, default=1
            Number of initializations to perform.
        init_params : str, default="kmeans"
            Initialization method ('kmeans' or 'random').
        random_state : Optional[int], optional
            Random seed for reproducibility.
        verbose : bool, default=False
            Verbosity mode for debugging output.
        """

        def __init__(
            self,
            n_components: int = 1,
            tol: float = 1e-3,
            max_iter: int = 100,
            n_init: int = 1,
            init_params: str = "kmeans",
            random_state: Optional[int] = None,
            verbose: bool = False,
        ):
            super(GMM, self).__init__(
                n_components, tol, max_iter, n_init, init_params, random_state, verbose
            )
            self._mixtures: Optional[np.ndarray] = None  # Mixture weights
            self._means: Optional[np.ndarray] = None  # Mean of each Gaussian component
            self._covariances: Optional[np.ndarray] = (
                None  # Covariance matrices for each component
            )
            self.num_features: Optional[int] = None  # Number of features in the dataset

        @staticmethod
        def _compute_components_loglikelihood(
            x: np.ndarray,
            means: np.ndarray,
            covariances: np.ndarray,
            seed: Optional[int],
        ) -> np.ndarray:
            """
            Compute the log-likelihood of each component for a given data point.

            The log-likelihood for a Gaussian component is calculated using:

            .. math::
                \log P(x \mid \mu, \Sigma) = -\frac{1}{2} \left[ (x - \mu)^T \Sigma^{-1} (x - \mu) + \log |\Sigma| + d \log (2\pi) \right]

            Parameters
            ----------
            x : np.ndarray
                The data point.
            means : np.ndarray
                Mean vectors of Gaussian components.
            covariances : np.ndarray
                Covariance matrices of Gaussian components.
            seed : Optional[int]
                Random seed for reproducibility.

            Returns
            -------
            np.ndarray
                Log-likelihood of each component.
            """
            llhds = [
                MultiVariateGaussian(
                    mean=means[i], covariance=covariances[i], seed=seed
                ).loglikelihood(x)
                for i in range(len(means))
            ]
            return np.array(llhds)

    @staticmethod
    def _compute_components_likelihood(
        x: np.ndarray, means: np.ndarray, covariances: np.ndarray, seed: Optional[int]
    ) -> np.ndarray:
        """
        Compute the likelihood of each component for a given data point.

        The likelihood for a Gaussian component is calculated using the multivariate normal distribution:

        .. math::
            P(x \mid \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)

        Parameters
        ----------
        x : np.ndarray
            The data point.
        means : np.ndarray
            Mean vectors of Gaussian components.
        covariances : np.ndarray
            Covariance matrices of Gaussian components.
        seed : Optional[int]
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Likelihood of each component.
        """
        lhds = [
            MultiVariateGaussian(
                mean=means[i], covariance=covariances[i], seed=seed
            ).likelihood(x)
            for i in range(len(means))
        ]
        return np.array(lhds)

    @staticmethod
    def _compute_data_likelihood(
        X: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
        mixtures: np.ndarray,
        reduction: Optional[str] = "sum",
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the overall likelihood of the dataset.

        The likelihood for a dataset is computed as the weighted sum of Gaussian component likelihoods:

        .. math::
            P(X \mid \theta) = \sum_{k=1}^{K} \pi_k P(X \mid \mu_k, \Sigma_k)

        Parameters
        ----------
        X : np.ndarray
            Input data matrix of shape (n_samples, n_features).
        means : np.ndarray
            Mean vectors of Gaussian components.
        covariances : np.ndarray
            Covariance matrices of Gaussian components.
        mixtures : np.ndarray
            Mixture weights for each Gaussian component.
        reduction : Optional[str], default="sum"
            Specifies how to reduce the computed likelihoods ('sum' for total likelihood, 'probs' for normalized probabilities).
        seed : Optional[int]
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Matrix containing the computed likelihoods per sample.
        """
        lhd_matrix = np.vstack(
            [
                GMM._compute_components_likelihood(x, means, covariances, seed=seed)
                * mixtures
                for x in X
            ]
        )

        if reduction == "sum":
            lhd_matrix = lhd_matrix.sum(axis=1)
        elif reduction == "probs":
            lhd_matrix /= lhd_matrix.sum(axis=1, keepdims=True)

        return lhd_matrix

    def _expectation(
        self,
        X: np.ndarray,
        mixtures: np.ndarray,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> np.ndarray:
        """
        E-step: Compute posterior probabilities (responsibilities).

        The responsibilities are computed using:

        .. math::
            \gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}

        Parameters
        ----------
        X : np.ndarray
            Input data matrix.
        mixtures : np.ndarray
            Mixture weights for each Gaussian component.
        means : np.ndarray
            Mean vectors of Gaussian components.
        covariances : np.ndarray
            Covariance matrices of Gaussian components.

        Returns
        -------
        np.ndarray
            Matrix of posterior probabilities.
        """
        prob_matrix = self._compute_data_likelihood(
            X,
            means,
            covariances,
            mixtures,
            reduction="probs",
        )
        return prob_matrix

    def _maximization(
        self, X: np.ndarray, prob_matrix: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        M-step: Update mixture weights, means, and covariances.

        The updated parameters are calculated as follows:

        .. math::
            \mu_k = \frac{\sum_{i=1}^{N} \gamma_{ik} x_i}{\sum_{i=1}^{N} \gamma_{ik}}

        .. math::
            \Sigma_k = \frac{\sum_{i=1}^{N} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^{N} \gamma_{ik}}

        .. math::
            \pi_k = \frac{\sum_{i=1}^{N} \gamma_{ik}}{N}

        Parameters
        ----------
        X : np.ndarray
            Input data matrix.
        prob_matrix : np.ndarray
            Matrix of posterior probabilities (responsibilities).

        Returns
        -------
        means : np.ndarray
            Updated mean vectors of Gaussian components.
        covariances : np.ndarray
            Updated covariance matrices of Gaussian components.
        mixtures : np.ndarray
            Updated mixture weights.
        """
        mixtures = prob_matrix.mean(axis=0)
        means = prob_matrix.T @ X / prob_matrix.sum(axis=0)

        covariances = np.stack(
            [
                GMM._compute_component_covariance(X, means[k], prob_matrix[:, k])
                for k in range(self.n_components)
            ]
        )
        return means, covariances, mixtures

    @staticmethod
    def _compute_component_covariance(
        X: np.ndarray, means: np.ndarray, prob_vector: np.ndarray
    ) -> np.ndarray:
        """
        Compute the covariance matrix for a single Gaussian component.

        The covariance is computed as follows:

        .. math::
            \Sigma_k = \frac{\sum_{i=1}^{N} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^{N} \gamma_{ik}}

        Parameters
        ----------
        X : np.ndarray
            Input data matrix.
        means : np.ndarray
            Mean vector of the component.
        prob_vector : np.ndarray
            Posterior probability vector for the component.

        Returns
        -------
        np.ndarray
            The computed covariance matrix.
        """
        X -= means
        return (prob_vector[:, np.newaxis] * X).T @ X / prob_vector.sum()

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Fit the Gaussian Mixture Model to the data using the EM algorithm.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix of shape (n_samples, n_features).
        y : Optional[np.ndarray], default=None
            Ignored, exists for compatibility with sklearn interface.
        """
        X = check_array(X)
        num_samples, self.num_features = X.shape

        means, covariances, mixtures = self._initialize_parameters()

        for i in range(self.max_iter):
            prob_matrix = self._expectation(X, mixtures, means, covariances)
            old_mixtures = mixtures
            means, covariances, mixtures = self._maximization(
                X, prob_matrix=prob_matrix
            )
            if self.is_converged(mixtures, old_mixtures):
                _logger.info("Converged")
                break

        self._means, self._covariances, self._mixtures = means, covariances, mixtures

    def _initialize_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize the GMM parameters.

        Returns
        -------
        means : np.ndarray
            Randomly initialized means.
        covariances : np.ndarray
            Identity matrices as initial covariances.
        mixtures : np.ndarray
            Equal weights for each component.
        """
        np.random.seed(self.random_state)
        means = np.random.randn(self.n_components, self.num_features)
        covariances = np.stack(
            [np.eye(self.num_features) for _ in range(self.n_components)]
        )
        mixtures = np.ones(self.n_components) / self.n_components

        return means, covariances, mixtures

    def is_converged(self, mixtures: np.ndarray, old_mixtures: np.ndarray) -> bool:
        """
        Check convergence based on the change in mixture weights.

        Parameters
        ----------
        mixtures : np.ndarray
            Current mixture weights.
        old_mixtures : np.ndarray
            Previous iteration's mixture weights.

        Returns
        -------
        bool
            True if the change is below the tolerance threshold, False otherwise.
        """
        return sum(abs(mixtures - old_mixtures)) < self.tol

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict posterior probabilities for each Gaussian component.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix.

        Returns
        -------
        np.ndarray
            Posterior probability matrix.
        """
        X = check_array(X)
        self.is_fitted()
        probs = self._expectation(X, self._mixtures, self._means, self._covariances)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the most likely Gaussian component for each sample.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix.

        Returns
        -------
        np.ndarray
            Predicted cluster labels.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def is_fitted(self) -> None:
        """
        Check if the model has been fitted.

        Raises
        ------
        ValueError
            If the model is not fitted.
        """
        if self._mixtures is None:
            raise ValueError(f"Model is not fitted!")
