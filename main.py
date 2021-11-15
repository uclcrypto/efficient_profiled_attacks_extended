import numpy as np
from distinguishers import gt_sasca, rf, mlp
import os
from leakage_oracle import LeakageOracle
from it_sampling import it_sampling
import glob
import matplotlib.pyplot as plt

## arguments
import argparse

parser = argparse.ArgumentParser(description="Comparing distinguishers convergence.")
parser.add_argument("--shares", "-d", default=2, type=int, help="Number of shares.")
parser.add_argument(
    "--bits", "-b", default=4, type=int, help="Number of bits in target bits."
)
parser.add_argument("--std", default=1.0, type=float, help="Noise standard deviation.")
parser.add_argument("--flaw", "-f", default=0.0, type=float, help="Flaw magnitude.")
parser.add_argument(
    "--repeat", "-r", default=2, type=int, help="Number of repeated experiments"
)


args = parser.parse_args()
d = args.shares
b = args.bits
f = args.flaw
sigma = args.std
repeat = args.repeat

rp = 0.01
n_profile = np.logspace(2.5, 6, 15, dtype=int)
dir_name = "d%d_b%d_f%.3f_sigma%.3f" % (d, b, f, sigma)
os.makedirs(dir_name, exist_ok=True)

methods = [
    {
        "label": "gt-esasca",
        "func": gt_sasca,
        "n_profile": n_profile,
        "repeat": repeat,
    },
    {
        "label": "rf",
        "func": rf,
        "n_profile": n_profile,
        "repeat": repeat,
    },
    {
        "label": "mlp",
        "func": mlp,
        "n_profile": n_profile,
        "repeat": repeat,
    },
]


def it_computation(methods):
    print("Computing MI for the implementation")
    loracle = LeakageOracle(d=d, b=b, f=f, sigma=sigma)
    mi, mi_std = it_sampling(loracle.pmf, loracle, rp=rp)

    coin = np.random.randint(0, 2 ** 32)
    fname = os.path.join(dir_name, "mi" + "_%d.npz" % (coin))
    np.savez(fname, mi=mi, mi_std=mi_std)

    print("Evaluate PI for different methods")
    for method in methods:
        n_profile = method["n_profile"]
        pi = np.zeros(len(n_profile))
        pi_std = np.zeros(len(n_profile))

        print("----> Evaluate ", method["label"])
        for _ in range(method["repeat"]):
            coin = np.random.randint(0, 2 ** 32)
            fname = os.path.join(dir_name, method["label"] + "_%d.npz" % (coin))
            leakage, shares, secret = loracle.get(np.max(n_profile))
            for i, n in enumerate(n_profile):
                pmf = method["func"](leakage[:n], shares[:n], d=d, b=b)
                pi[i], pi_std[i] = it_sampling(pmf, loracle, rp=rp, floor=mi / 50)
                print(
                    "--- "
                    + f"{method['label']}: pi = {pi[i]:.4f} : n = {n:9d} : mi = {mi:.4f}"
                )
            np.savez(fname, pi=pi, pi_std=pi_std, n_profile=n_profile)


def summarize(labels):
    """Summarize the output of it_computation to generate curves"""

    # get MI best estimation
    mi = []
    for fname in glob.glob(os.path.join(dir_name, "mi_*.npz")):
        mi.append(np.load(fname)["mi"])
    mi = np.mean(np.array(mi))

    curves = []
    for label in labels:
        n_profile = None
        pi = []

        for fname in glob.glob(os.path.join(dir_name, label + "_*.npz")):
            data = np.load(fname)
            if n_profile is None:
                n_profile = data["n_profile"]

            assert np.array_equal(n_profile, data["n_profile"])
            pi.append(data["pi"])
        pi = np.mean(np.array(pi), axis=0)

        curves.append({"pi": pi, "n_profile": n_profile, "label": label})

    np.savez(
        os.path.join(dir_name, "summary.npz"), curves=curves, mi=mi, allow_pickle=True
    )


def plot_summary(labels):
    """Plot the summarized curved"""
    summary = np.load(os.path.join(dir_name, "summary.npz"), allow_pickle=True)
    mi = summary["mi"]
    curves = summary["curves"]
    plt.figure()
    plt.axhline(mi, color="r", ls="--", label="MI")
    for curve in curves:
        if curve["label"] not in labels:
            break
        plt.loglog(curve["n_profile"], curve["pi"], label=curve["label"])

    plt.legend()
    plt.savefig(os.path.join(dir_name, "summary.pdf"))
    plt.show()


if __name__ == "__main__":

    it_computation(methods)
    summarize(["gt-esasca", "mlp"])
    plot_summary(["gt-esasca", "mlp"])
