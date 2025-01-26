import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Applies the sigmoid function element-wise.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Transformed array with sigmoid applied.
    """
    return 1 / (1 + np.exp(-x))
