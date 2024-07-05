import numpy as np

def progonka(am, bm, cm, fm):
    """
    A*x(i-1) + C*x(i) + B*x(i+1) = f
    Метод прогонки
    """
    
    n = len(cm)
    al = np.zeros(n)
    be = np.zeros(n)
    x = np.zeros(n)

    al[0] = -bm[0] / cm[0]
    be[0] = fm[0] / cm[0]

    for i in range(1, n - 1):
        denominator = am[i] * al[i - 1] + cm[i]
        al[i] = -bm[i] / denominator
        be[i] = (fm[i] - am[i] * be[i - 1]) / denominator

    x[n - 1] = (fm[n - 1] - am[n - 1] * be[n - 2]) / (am[n - 1] * al[n - 2] + cm[n - 1])

    for i in range(n - 2, -1, -1):
        x[i] = al[i] * x[i + 1] + be[i]

    return x