import numpy as np
from methods import gt_sasca
import os
from leakage_oracle import LeakageOracle
from it_sampling import it_sampling
import glob
import matplotlib.pyplot as plt
import tikzplotlib

d = 2
b = 4
f = 0.00
sigma =  1

rp = 0.005

dir_name = "d%d_b%d_f%.3f_sigma%.3f"%(d,b,f,sigma)
os.makedirs(dir_name,exist_ok=True)

methods = [{"label":"gt-esasca",
                "func":gt_sasca,
                "n_profile":np.logspace(2,6,20,dtype=int),
                "repeat":10}
        ]

def it_computation():
    loracle = LeakageOracle(d=d,b=b,f=f,sigma=sigma)
    mi,mi_std = it_sampling(loracle.pmf,loracle,rp=rp)

    coin = np.random.randint(0,2**32)
    fname = os.path.join(dir_name,"mi"+"_%d.npz"%(coin))
    np.savez(fname,mi=mi,mi_std=mi_std)
    

    for method in methods:
        n_profile = method["n_profile"]
        pi = np.zeros(len(n_profile))
        pi_std = np.zeros(len(n_profile))
       
        for _ in range(method["repeat"]):
            coin = np.random.randint(0,2**32)
            fname = os.path.join(dir_name,method["label"]+"_%d.npz"%(coin))

            print(fname)
            leakage,shares,secret = loracle.get(np.max(n_profile))
            for i,n  in enumerate(n_profile):
                pmf = method["func"](leakage[:n],shares[:n],d=d,b=b)
                pi[i],pi_std[i] = it_sampling(pmf,loracle,rp=rp,floor=mi/50)

            np.savez(fname,pi=pi,pi_std=pi_std,n_profile=n_profile)

def summarize(labels):
    # get MI best estimation
    mi = []
    for fname in glob.glob(os.path.join(dir_name,"mi_*.npz")):
        mi.append(np.load(fname)["mi"])
    mi = np.mean(np.array(mi))
    
    curves = []
    for label in labels:
        n_profile = None
        pi = []

        for fname in glob.glob(os.path.join(dir_name,label+"_*.npz")):
            data = np.load(fname)
            if n_profile is None:
                n_profile = data["n_profile"]
            
            assert np.array_equal(n_profile,data["n_profile"])
            pi.append(data["pi"])
        pi = np.mean(np.array(pi),axis=0)

        curves.append({"pi":pi,"n_profile":n_profile,"label":label})

    np.savez(os.path.join(dir_name,"summary.npz"),curves=curves,mi=mi,allow_pickle=True)

def plot_summary(labels):
    summary = np.load(os.path.join(dir_name,"summary.npz"),allow_pickle=True)
    mi = summary["mi"]
    curves = summary["curves"]
    plt.figure()
    plt.axhline(mi,color="r",ls="--",label="MI")
    for curve in curves:
        if curve["label"] not in labels: 
            break
        plt.loglog(curve["n_profile"],curve["pi"],label=curve["label"])

#    plt.xlabel("profiling data")
#    plt.ylabel("Information Metrics")
#    plt.grid(True,which="both",ls="--")
    plt.legend()
    plt.savefig(os.path.join(dir_name,"summary.pdf"))
    tikzplotlib.save(os.path.join(dir_name,"summary.tex"))
    plt.show()
if __name__ == "__main__":
    print("Computing MI for the implementation")
    #it_computation()

    summarize(["gt-esasca"])
    plot_summary(["gt-esasca"])
