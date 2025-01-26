from typing import Union

import numpy as np

from src.supervised.decision_trees import _BaseDecisionTree
from src.supervised.randomforests import _BaseTreeEnsemble
from src.utils.datautils import one_hot_encoding
from src.utils.loss import LogisticLoss, _Loss


class XGBoostDecisionTree(_BaseDecisionTree):
    """
    Implementation of a Decision Tree optimized for use in XGBoost, with gradient and Hessian-based splitting.

    This class extends a base decision tree and integrates gradient and Hessian computations to
    determine the best splits and approximate leaf values. It is specifically designed to work
    with XGBoost's boosting framework.

    Methods
    -------
    fit(X, y)
        Fit the decision tree to the data.
    _compute_score_gain(y, ypred)
        Compute the score gain based on gradients and Hessians for a given split.
    _compute_score_gain_taylor_expansion(y, yl, yr)
        Compute the score gain using a Taylor series approximation for child nodes.
    _compute_leaf_value_approximate(y)
        Approximate the leaf value using gradients and Hessians.

    """

    def __init__(
        self,
        max_depth: float = float("inf"),
        min_sample_split: int = 2,
        min_impurity: float = 1e-7,
        loss: Union[_Loss, None] = None,
    ):
        """
        Initialize the XGBoostDecisionTree.

        This sets up the impurity and leaf value computation methods using Taylor expansion.
        """
        super().__init__(
            max_depth=max_depth,
            min_sample_split=min_sample_split,
            min_impurity=min_impurity,
            loss=loss,
        )

        self._compute_impurity = self._compute_score_gain_taylor_expansion
        self._compute_leaf_value = self._compute_leaf_value_approximate

    def _bisect_y(self, y):
        """
        Split the target array into gradients and Hessians.

        Parameters
        ----------
        y : np.ndarray
            Target array containing both gradients and Hessians.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Gradients (first half) and Hessians (second half).
        """
        idx = y.shape[1] // 2
        return y[:, :idx], y[:, idx:]

    def _compute_score_gain(self, y, y_pred):
        """
        Compute the score gain for a given split using gradients and Hessians.

        Parameters
        ----------
        y : np.ndarray
            Target values (gradients).
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        float
            Score gain for the split.
        """
        G = y * self.loss.compute_gradient(y, y_pred).sum()
        H = self.loss.compute_hessian(y, y_pred).sum()
        return G / (2 * H)

    def _compute_score_gain_taylor_expansion(self, y, yl, yr):
        """
        Compute the score gain using a Taylor expansion approximation.

        Parameters
        ----------
        y : np.ndarray
            Target values (gradients and Hessians).
        yl : np.ndarray
            Left child target values (gradients and Hessians).
        yr : np.ndarray
            Right child target values (gradients and Hessians).

        Returns
        -------
        float
            Total gain for the split.
        """
        y, y_pred = self._bisect_y(y)
        yl, yl_pred = self._bisect_y(yl)
        yr, yr_pred = self._bisect_y(yr)

        gain = self._compute_score_gain(y, y_pred)
        l_gain = self._compute_score_gain(yl, yl_pred)
        r_gain = self._compute_score_gain(yr, yr_pred)

        return l_gain + r_gain - gain

    def _compute_leaf_value_approximate(self, y):
        """
        Compute the approximate value of a leaf node using gradients and Hessians.

        Parameters
        ----------
        y : np.ndarray
            Target values (gradients and Hessians).

        Returns
        -------
        np.ndarray
            Approximate leaf value.
        """
        y, y_pred = self._bisect_y(y)

        G = (y * self.loss.compute_gradient(y, y_pred)).sum(axis=0)
        H = self.loss.compute_hessian(y, y_pred).sum(axis=0)

        return G / H

    def fit(self, X, y):
        """
        Fit the decision tree to the input data.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix, shape (n_samples, n_features).
        y : np.ndarray
            Target array containing gradients and Hessians, shape (n_samples, 2).

        Returns
        -------
        None
        """
        super(XGBoostDecisionTree, self).fit(X, y)


class XGBoostClassifier(_BaseTreeEnsemble):
    """
    XGBoost Classifier implementing a gradient-boosted decision tree ensemble for multi-class classification.

    Parameters
    ----------
    n_estimators : int, optional
        Number of boosting stages to perform. Default is 200.
    learning_rate : float, optional
        Learning rate shrinks the contribution of each tree. Default is 0.001.
    min_samples_split : int, optional
        The minimum number of samples required to split an internal node. Default is 2.
    min_impurity : float, optional
        Minimum impurity required to perform a split. Default is 1e-7.
    max_depth : int, optional
        Maximum depth of the individual regression estimators. Default is 2.

    Attributes
    ----------
    trees : list[XGBoostDecisionTree]
        List of decision trees in the ensemble.
    loss : LogisticLoss
        Loss function for classification (logistic loss).

    Methods
    -------
    fit(X, y)
        Fit the classifier to the training data.
    predict(X)
        Predict class labels for the input data.

    Examples
    --------
    >>> import numpy as np
    >>> from src.supervised.xgboost import XGBoostClassifier
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    >>> y = np.array([0, 1, 0, 1])
    >>> clf = XGBoostClassifier(n_estimators=10, learning_rate=0.1)
    >>> clf.fit(X, y)
    >>> y_pred = clf.predict(X)
    >>> print(y_pred)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.001,
        min_samples_split: int = 2,
        min_impurity: float = 1e-7,
        max_depth: int = 2,
    ):
        """
        Initialize the XGBoostClassifier.

        Parameters
        ----------
        n_estimators : int, optional
            Number of boosting stages to perform. Default is 200.
        learning_rate : float, optional
            Learning rate shrinks the contribution of each tree. Default is 0.001.
        min_samples_split : int, optional
            The minimum number of samples required to split an internal node. Default is 2.
        min_impurity : float, optional
            Minimum impurity required to perform a split. Default is 1e-7.
        max_depth : int, optional
            Maximum depth of the individual regression estimators. Default is 2.
        """
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            min_impurity=min_impurity,
            max_depth=max_depth,
        )

        self.loss = LogisticLoss()

        # Initialize decision trees for boosting
        self.trees = []
        for _ in range(n_estimators):
            tree = XGBoostDecisionTree(
                max_depth=max_depth,
                min_sample_split=self.min_samples_split,
                min_impurity=self.min_impurity,
                loss=self.loss,
            )
            self.trees.append(tree)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the XGBoost classifier to the training data.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix, shape (n_samples, n_features).
        y : np.ndarray
            Class labels, shape (n_samples,).

        Returns
        -------
        None
        """
        # Perform one-hot encoding of class labels
        y = one_hot_encoding(y)

        # Initialize predictions to zero
        y_pred = np.zeros(np.shape(y))

        for i in range(self.n_estimators):
            tree = self.trees[i]
            # Concatenate actual and predicted labels for gradient and Hessian computation
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(X, y_and_pred)

            # Update predictions
            update_pred = tree.predict(X)
            y_pred -= self.learning_rate * update_pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for the input data.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels, shape (n_samples,).
        """
        y_pred = None

        for tree in self.trees:
            update_pred = tree.predict(X)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred -= np.multiply(self.learning_rate, update_pred)

        # Convert logits to probabilities and predict the class with the highest probability
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
