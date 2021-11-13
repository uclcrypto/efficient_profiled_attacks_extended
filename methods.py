import numpy as np
from utils import recombine_fwht

def gt_sasca(leakage,shares,d,b):
    mean_shares =  np.zeros((d,2**b))
    std_shares = np.zeros(d)
    for i in range(d):
        for x in range(2**b):
            indexes = np.where(shares[:,i]==x)[0]
            mean_shares[i,x] = np.mean(leakage[indexes,i])
        std_shares[i] = np.std(leakage[:,i] - mean_shares[i,shares[:,i]])

    def pmf(leakage):
        n,ndim = leakage.shape
        prs_shares = np.zeros((n,d,2**b))
        for i in range(d):
            for x in range(2** b):
                prs_shares[:,i,x] = np.exp(-0.5*((leakage[:,i]-mean_shares[i,x])/std_shares[i])**2)

        prs = recombine_fwht(prs_shares.T).T
        prs = (prs.T / np.sum(prs,axis=1)).T
        prs[np.where(prs<1E-100)] = 1E-100
        return prs 

    return pmf

if __name__ == "__main__":
    
    from leakage_oracle import LeakageOracle
    from it_sampling import it_sampling
    import matplotlib.pyplot as plt

    # parameter of the implementation
    d = 2
    b = 4
    f = 0.1
    sigma = 1

    loracle = LeakageOracle(d=d,b=b,f=f,sigma=sigma)
    mi,mi_std = it_sampling(loracle.pmf,loracle,rp=0.005)

    n_profile = np.logspace(2,6,20,dtype=int)
    pi_sasca = np.zeros(len(n_profile))
    pi_std = np.zeros(len(n_profile))
    
    leakage,shares,secret = loracle.get(np.max(n_profile))
    for i,n  in enumerate(n_profile):
        print(i)
        pmf_sasca = gt_sasca(leakage[:n],shares[:n],d=d,b=b)
        pi_sasca[i],pi_std[i] = it_sampling(pmf_sasca,loracle,rp=0.005,floor=mi/10)

    plt.figure()
    plt.loglog(n_profile,pi_sasca)
    plt.fill_between(n_profile,pi_sasca+pi_std,pi_sasca-pi_std)
    plt.axhline(mi,color="r")
    plt.fill_between(n_profile,mi+mi_std,mi-mi_std,color="r")
    plt.grid(True,which="both",ls="--")
    plt.show()
