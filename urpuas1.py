import numpy as np

def progonka(am, bm, cm, fm):
    """
    Решение трёхдиагональной матрицы методом прогонки.
    A*x(i-1) + C*x(i) + B*x(i+1) = f
    """
    n = len(cm)
    al = np.zeros(n)
    be = np.zeros(n)
    
    al[0] = -bm[0] / cm[0]
    be[0] = fm[0] / cm[0]

    for i in range(1, n-1):
        denom = am[i] * al[i-1] + cm[i]
        al[i] = -bm[i] / denom
        be[i] = (fm[i] - am[i] * be[i-1]) / denom

    xv = np.zeros(n)
    xv[-1] = (fm[-1] - am[-1] * be[-2]) / (am[-1] * al[-2] + cm[-1])

    for i in range(n-2, -1, -1):
        xv[i] = al[i] * xv[i+1] + be[i]

    return xv

def urpuas1(xh, s, y0, yn):
    """
    Решение краевой задачи для декартовых координат.
    """
    m = len(s)

    am = np.zeros(m)
    bm = np.zeros(m)
    cm = np.zeros(m)
    fm = np.zeros(m)

    am[0] = 0
    bm[0] = 0
    cm[0] = -1
    fm[0] = -y0

    hquadr = xh * xh

    for i in range(1, m-1):
        fm[i] = -s[i]
        am[i] = 1 / hquadr
        cm[i] = -2 / hquadr
        bm[i] = 1 / hquadr

    bm[-1] = 0
    am[-1] = 0
    cm[-1] = 1
    fm[-1] = yn

    df1 = progonka(am, bm, cm, fm)
    df5 = df1

    return df5