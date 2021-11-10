import numpy as np
#import scipy

# define hamming weight (HW) table.
HW = 0
r = np.arange(256).astype(np.uint8)
while np.max(r) > 0:
    HW += r & 0x1
    r = r >> 1


def get_HW(matrix):
    return HW[matrix]


def fwht(a):
    h = 1
    x = np.zeros(a[0].shape).astype(np.float64)
    y = np.zeros(a[0].shape).astype(np.float64)
    while h < len(a):
        for i in range(0,len(a),h*2):
            for j in range(i,i+h):
                x[:] = a[j]
                y[:] = a[j+h]
                a[j] = x+y
                a[j+h] = x-y
        h *= 2

def recombine_fwht(pr):
    """
        pr is of size (Nk x D x  Nt):
            Nk size of the field
            D number of shares 
            Nt  number of traces
    """
    pr = pr.astype(np.float64)
    pr_fft = pr.copy()
    fwht(pr_fft)
    pr = np.prod(pr_fft,axis=1)
    fwht(pr)
    return pr


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
        data = np.zeros((n, self.ndim), dtype=np.uint8)
        
        data[:, : self.d] = shares
        secret = np.bitwise_xor.reduce(shares, axis=1)
        if self.f != 0.0:
            data[:, self.d] = secret

        leakage = np.random.normal(get_HW(data), scale=self.sigma)
        return leakage, shares, secret

    def pmf(self,leakage):

        n,_ = leakage.shape
        
        if self.f == 0.0:
            prs_shares = np.zeros((n,self.d,2**self.b))
            for d in range(self.d):
                for x in range(2** self.b):
                    #prs[:,d,:] = scipy.stats.norm.pdf(leakage[:,d],loc=get_HW(x),scale=self.sigma) 
                    prs_shares[:,d,x] = np.exp(-0.5*((leakage[:,d]-get_HW(x))/self.sigma)**2)

                prs_shares[:,d,:] = (prs_shares[:,d,:].T/np.sum(prs_shares[:,d,:],axis=1)).T
                prs = recombine_fwht(prs_shares.T).T
        else:
            print("Not implemented yet pmf for f!=0.0")
            exit(-1)

        
        return (prs.T / np.sum(prs,axis=1)).T

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
