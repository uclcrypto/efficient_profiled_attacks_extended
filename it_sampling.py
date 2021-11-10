import numpy as np

def it_sampling(pmf,oracle,rp=1E-1,floor=0.0):
    """ compute information extracted by a pmf from 
    a leakage oracle

    inputs:
        pmf (function)
            outputs probabilities when given leakage.
        oracle (objection)
            a LeakageOracle representing the implementation.
        rp (float)
            relative precision of estimation of it metric.
        floor (floor)
            stop estimating if it is smaller than floor.

    output:
        it (float)
            it metric to be estimated.
        esti_std (float)
            standard deviation of it estimator.
    """
    s = 0; s2 = 0; step = int(1E4); n = 0;

    while n<1E9:
        leakage,shares,secret = oracle.get(step)

        prs_k_l = pmf(leakage)
        logs = np.log2(prs_k_l[np.arange(step),secret])

        s += np.sum(logs)
        s2 += np.sum(logs**2)
        n += step

        std_logs = np.sqrt(s2/n - (s/n)**2)
        esti_std = std_logs / np.sqrt(n) 
        it = oracle.b + s/n

        # break if relative precision is met
        if esti_std < rp * (it) and it > 0:
            break

        # break if it too small compared to floor 
        if it + esti_std*3 < floor:
            break

    return it,esti_std

if __name__ == "__main__":
    print("it_sampling")
    from leakage_oracle import LeakageOracle
    import matplotlib.pyplot as plt
    
    n = 20
    ds = [1,2,3]
    fs = [0.0,0.5]
    b = 2
    rp = 0.1

    var = np.logspace(np.log10(0.01),np.log10(10),n)
    mis = np.zeros(n)
    esti_std = np.zeros(n)

    plt.figure()
    for f in fs:
        for d in ds:
            print("%d shares - %.3f flaw"%(d,f))
            for i,sigma in enumerate(np.sqrt(var)):
                loracle = LeakageOracle(d=d,b=b,f=f,sigma=sigma)
                mis[i],esti_std[i] = it_sampling(loracle.pmf,loracle,rp=rp)

            plt.loglog(var,mis,label="%d-shares - {%.3f} flaw"%(d,f))
            plt.fill_between(var,mis-esti_std,mis+esti_std,alpha=0.1)
    plt.grid(True,which="both",ls="--")
    plt.xlabel("noise variance")
    plt.ylabel("IT")
    plt.legend()
    plt.show()
