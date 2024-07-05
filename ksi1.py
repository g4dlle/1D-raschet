import numpy as np

def ksi1(TeP):
    m = len(TeP)
    
    r = np.zeros(m)
    for i in range(m):
        r[i] = 3 * (TeP[i] / 11605) / 2

    x2 = [1e-21, 1.75, 2.5, 3.5, 5.15, 7.15, 10.8, 16]
    y1 = [1e-21, 10**(-11.6), 10**(-9), 10**(-7.96), 10**(-7.5), 10**(-7.1), 10**(-6.97), 10**(-6.91)]
    
    y3 = np.log(y1)
    
    d = np.zeros(m)
    for i in range(m):
        h = 1
        while r[i] > x2[h] and h < len(x2) - 1:
            h += 1
        if h == 1:
            d[i] = y3[0]
        elif h < len(x2):
            d[i] = y3[h-1] + (r[i] - x2[h-1]) * (y3[h] - y3[h-1]) / (x2[h] - x2[h-1])
        else:
            d[i] = y3[-1]
    
    vd = np.exp(d) * 10**(-6)
    return vd