import numpy as np

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


