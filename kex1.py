import numpy as np

def kex1(TeP):
    m = len(TeP)

    r = np.zeros(m)
    for i in range(m):
        r[i] = 3 * (TeP[i] / 11605) / 2

    x1 = [1e-21, 4, 4.5, 5, 5.1, 5.3, 5.5, 5.7, 6.3, 7, 7.5, 8, 8.8, 11, 13.6, 16]
    y1 = [1e-21, 1e-14, 10**(-13.2), 10**(-12.4), 1e-12, 10**(-11.3), 1e-11, 10**(-10.7), 1e-10, 10**(-9.6), 10**(-9.4), 10**(-9.2), 1e-9, 10**(-8.8), 10**(-8.7), 10**(-8.6)]

    y3 = np.log(y1)

    d = np.zeros(m)
    for i in range(m):
        h = 1
        while r[i] > x1[h] and h < len(x1) - 1:
            h += 1
        if h == 1:
            d[i] = y3[0]
        elif h < len(x1):
            d[i] = y3[h-1] + (r[i] - x1[h-1]) * (y3[h] - y3[h-1]) / (x1[h] - x1[h-1])
        else:
            d[i] = y3[-1]

    vd = np.exp(d) * 10**(-6)
    return vd