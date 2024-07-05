import numpy as np

def ki1(TeP):
    m = len(TeP)
    r = np.zeros(m)
    
    # Вычисление значений r
    for i in range(m):
        r[i] = 3 * (TeP[i] / 11605) / 2

    # Задание массивов x2 и y1
    x2 = np.array([5.5, 5.7, 6, 6.9, 7.5, 9.1, 11, 14, 16])
    y1 = np.array([1e-14, 1e-13, 10**-11.7, 10**-10.6, 10**-9.8, 10**-8.9, 10**-8.4, 10**-8, 10**-7.9])
    
    # Вычисление логарифмов y1
    y3 = np.log(y1)
    
    d = np.zeros(m)
    
    # Интерполяция значений
    for i in range(m):
        h = 1
        while (r[i] > x2[h]) and (h < len(x2) - 1):
            h += 1
        if h == 1:
            d[i] = y3[0]
        elif (h > 1) and (h < len(x2)):
            d[i] = y3[h-1] + (r[i] - x2[h-1]) * (y3[h] - y3[h-1]) / (x2[h] - x2[h-1])
        elif h >= len(x2):
            d[i] = y3[-1]

    # Окончательное вычисление vd
    vd = np.exp(d) * 10**-6
    
    return vd