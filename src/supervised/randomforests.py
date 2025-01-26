from abc import abstractmethod


class _BaseTreeEnsemble:

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate=0.001,
        min_samples_split=2,
        min_impurity=1e-7,
        max_depth=2,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
