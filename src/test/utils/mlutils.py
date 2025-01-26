import numpy as np
from numpy._typing import NDArray
from sklearn.utils import check_array


def add_intercept_column(x: NDArray):

    x = check_array(x)
    return np.hstack((np.ones((x.shape[1], 1)), x))
