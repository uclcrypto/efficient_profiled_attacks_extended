import numpy as np
from utils import get_HW, recombine_fwht


class LeakageOracle:
    def __init__(self, sigma, d, b, f=0.0):
        """
        parameters:
            sigma (int)
                noise standard deviation
            d (int)
                number of shares
            b (int)
                number of bits for sensible variables
            f (float)
                first order leakage magnitude
        """
        self.sigma = sigma
        self.d = d
        self.b = b
        self.f = f
        self.ndim = d

        if self.f != 0.0:
            self.ndim += 1

    def get(self, n):
        """
        parameters:
            n (int)
                number of random samples

        return:
            leakage (array n x ndim)
                leakage samples
            shares (array n x d)
                shares values
            secret (array n)
                secret values
        """
        shares = np.random.randint(0, 2 ** self.b, (n, self.d), dtype=np.uint8)
        leakage = np.zeros((n, self.ndim))

        leakage[:, : self.d] = get_HW(shares)
        secret = np.bitwise_xor.reduce(shares, axis=1)
        if self.f != 0.0:
            leakage[:, self.d] = self.f * get_HW(secret)

        leakage = np.random.normal(leakage, scale=self.sigma)
        return leakage, shares, secret

    def pmf(self, leakage):

        n, _ = leakage.shape

        prs_shares = np.zeros((n, self.d, 2 ** self.b))
        for d in range(self.d):
            for x in range(2 ** self.b):
                prs_shares[:, d, x] = np.exp(
                    -0.5 * ((leakage[:, d] - get_HW(x)) / self.sigma) ** 2
                )

        prs = recombine_fwht(prs_shares.T).T

        if self.f != 0.0:
            for x in range(2 ** self.b):
                prs[:, x] *= np.exp(
                    -0.5
                    * ((leakage[:, self.d] - (self.f * get_HW(x))) / self.sigma) ** 2
                )

        return (prs.T / np.sum(prs, axis=1)).T


if __name__ == "__main__":
    n = 100
    d = 2
    loracle = LeakageOracle(sigma=1, d=d, b=4)
    leakage, shares, secret = loracle.get(n)

    assert leakage.shape == (n, d)
    assert shares.shape == (n, d)

    loracle = LeakageOracle(sigma=1, d=d, b=4, f=0.1)
    leakage, shares, secret = loracle.get(n)

    assert leakage.shape == (n, d + 1)
    assert shares.shape == (n, d)
