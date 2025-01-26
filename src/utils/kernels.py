import numpy as np
from scipy.spatial import distance


def linear(**kwargs):
    r"""
    Linear kernel function.

    The linear kernel is defined as:

    .. math::
        K(u, v) = u \cdot v

    where \( u \) and \( v \) are input vectors.

    This kernel computes the dot product between two vectors and is equivalent to using
    the original feature space for machine learning algorithms.

    Parameters
    ----------
    kwargs : dict
        Additional keyword arguments (not used in the linear kernel).

    Returns
    -------
    callable
        A function that computes the linear kernel between two vectors.

    Examples
    --------
    >>> import numpy as np
    >>> u = np.array([1, 2])
    >>> v = np.array([3, 4])
    >>> linear_kernel = linear()
    >>> linear_kernel(u, v)
    11
    """

    def f(u, v):
        return u @ v.T

    return f


def polynomial(p=3, **kwargs):
    """
    Polynomial kernel function.

    The polynomial kernel is defined as:

    .. math::
        K(u, v) = (1 + u \cdot v)^p

    where \( u \) and \( v \) are input vectors, and \( p \) is the degree of the polynomial.

    This kernel maps the original feature space into a higher-dimensional space,
    which can capture non-linear relationships.

    Parameters
    ----------
    p : int, optional
        The degree of the polynomial. Default is 3.
    kwargs : dict
        Additional keyword arguments (not used in the polynomial kernel).

    Returns
    -------
    callable
        A function that computes the polynomial kernel between two vectors.

    Examples
    --------
    >>> import numpy as np
    >>> u = np.array([1, 2])
    >>> v = np.array([3, 4])
    >>> poly_kernel = polynomial(p=2)
    >>> poly_kernel(u, v)
    196
    """

    def f(u, v):
        return (1 + u @ v.T) ** p

    return f


def rbf(gamma, **kwargs):
    """
    Radial Basis Function (RBF) kernel function.

    The RBF kernel is defined as:

    .. math::
        K(u, v) = \exp(-\gamma \|u - v\|^2)

    where \( u \) and \( v \) are input vectors, \( \|u - v\| \) is the Euclidean distance between \( u \) and \( v \),
    and \( \gamma \) controls the spread of the kernel.

    The RBF kernel maps input vectors into an infinite-dimensional feature space,
    making it suitable for capturing highly complex relationships.

    Parameters
    ----------
    gamma : float
        The gamma parameter for the RBF kernel. Controls the spread of the kernel.
    kwargs : dict
        Additional keyword arguments (not used in the RBF kernel).

    Returns
    -------
    callable
        A function that computes the RBF kernel between two vectors.

    Examples
    --------
    >>> import numpy as np
    >>> u = np.array([1, 2])
    >>> v = np.array([3, 4])
    >>> rbf_kernel = rbf(gamma=0.5)
    >>> rbf_kernel(u, v)
    0.01831563888873418
    """

    def f(u, v):
        dist = distance.cdist(u, v, "sqeuclidean")
        return np.exp(-gamma * dist)

    return f
