import numpy as np


def it_sampling(pmf, oracle, rp=1e-1, floor=0.0):
    """compute information extracted by a pmf from
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
    s = 0
    s2 = 0
    step = int(1e4)
    n = 0

    while n < 1e9:
        leakage, shares, secret = oracle.get(step)

        prs_k_l = pmf(leakage)
        prs_k_l[np.where(~np.isfinite(prs_k_l))] = 1e-100
        prs_k_l[np.where(prs_k_l < 1e-100)] = 1e-100
        logs = np.log2(prs_k_l[np.arange(step), secret])

        s += np.sum(logs)
        s2 += np.sum(logs ** 2)
        n += step

        std_logs = np.sqrt(s2 / n - (s / n) ** 2)
        esti_std = std_logs / np.sqrt(n)
        it = oracle.b + s / n

        # break if relative precision is met
        if esti_std < rp * (it) and it > 0:
            break

        # break if it too small compared to floor
        if it + esti_std * 3 < floor:
            break

        print(
            f"n = {n:7d} : floor = {floor:.4f} : it = {it:.5f} : esti_std = {esti_std:.5f} (target {rp*(it):.5f})",
            end="\r",
        )
    print(" " * 100, end="\r")
    return it, esti_std
