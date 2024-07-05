import numpy as np

def Potok(delta_x, N, D, Vdr):
    """
    Программа вычисляет поток на слое в алгоритме Shhaefter-Gummel 
    N - значение концентрации, заданная в целых узлах 
    D - коэффициент диффузии заданный в полуцелых узлах
    Vdr - дрейфовая скорость, задана в полуцелых узлах 
    delta_x - шаг пространственной сетки
    """
    
    n = len(N)
    n1 = n - 1

    Gamma = np.zeros(n1)

    N1 = N[0]

    for i in range(n1):
        N2 = N[i + 1]
        
        if Vdr[i] != 0:
            alpha = delta_x * Vdr[i] / D[i]
            exp_alpha = np.exp(alpha)
            hI0 = (exp_alpha - 1) / alpha * delta_x
            Gamma[i] = D[i] * (N1 - exp_alpha * N2) / hI0
        else:
            Gamma[i] = D[i] * (N1 - N2) / delta_x

        N1 = N2

    return Gamma