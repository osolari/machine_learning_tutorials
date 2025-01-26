from abc import abstractmethod

import numpy as np

from src.utils.mathutils import sigmoid


class Activation:

    def __init__(self, gamma: float = None):

        self.gamma = gamma

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:

        pass

    @abstractmethod
    def gradient(self, x: np.ndarray) -> np.ndarray:

        pass


class Sigmoid(Activation):
    """
    Sigmoid activation function and its gradient.

    Methods
    -------
    __call__(z)
        Compute the sigmoid of input `z`.
    gradient(z)
        Compute the gradient of the sigmoid function.
    """

    def __init__(self):

        super().__init__()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the sigmoid activation function.

        Parameters
        ----------
        x : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Sigmoid activation of input `z`.
        """
        return 1 / (1 + np.exp(-x))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the sigmoid function.

        Parameters
        ----------
        x : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Gradient of the sigmoid function.
        """
        sigmoid = self.__call__(x)
        return sigmoid * (1 - sigmoid)
