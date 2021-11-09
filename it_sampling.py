import numpy as np

def it_sampling(pmf,oracle,rp=1E-1):
    s = 0; s2 = 0; step = int(1E5); n = 0;
    while n<1E7:
        leakage,shares,secret = oracle.get(step)


        prs_k_l = pmf(leakage)
        logs = np.log2(prs_k_l[np.arange(nstep),secret])

        s += np.sum(logs)
        s2 += np.sum(logs**2)
        n += step

        std_logs = np.sqrt(s2/n - (s/n)**2)
        esti_std = std_logs / np.sqrt(n) 
        pi = oracle.b + s/n

        if esti_std < e * (pi) and pi > 0:
            break

    return pi,esti_std
