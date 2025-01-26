import numpy as np


class _BaseDistribution:

    def __init__(self, loc, scale, seed, *args, **kwargs):

        self.loc = loc
        self.scale = scale
        self.seed = seed


class _BaseContinousDistribution(_BaseDistribution):

    def __init__(self, loc, scale, seed, *args, **kwargs):

        super().__init__(loc=loc, scale=scale, seed=seed, *args, **kwargs)


class MultiVariateGaussian(_BaseContinousDistribution):

    def __init__(self, mean, covariance, seed, *args, **kwargs):

        if (
            covariance.shape[0] != mean.shape[0]
            or covariance.shape[0] != covariance.shape[1]
        ):
            raise ValueError("covariance must be square!")

        super().__init__(loc=mean, scale=covariance, seed=seed)

        self.mean = mean
        self.num_rvs = self.mean.shape[0]
        self.covariance = covariance
        self.det = np.linalg.det(self.covariance)
        self.inv_cov = np.linalg.inv(self.covariance)

        self.seed = seed

    def likelihood(self, x):

        llhd = self.loglikelihood(x)

        return np.exp(llhd)

    def loglikelihood(self, x):

        n = len(x)

        exp = -0.5 * (x - self.mean).T @ self.inv_cov @ (x - self.mean)
        const = -0.5 * (n * np.log(2 * np.pi) + np.log(self.det))

        return const + exp
