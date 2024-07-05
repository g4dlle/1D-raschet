import numpy as np

def Potok_point_i(i, delta_x, N, D, Vdr):
    """
    Программа вычисляет поток в точке i в алгоритме Shhaefter-Gummel
    i - номер точки, в которой вычисляется поток (1 <= i <= n-1)
        i = 1 соответствует первому полуцелому узлу
        i = n - 1 соответствует последнему полуцелому узлу
    N - значение концентрации, заданная в целых узлах 
    D - коэффициент диффузии, заданный в полуцелых узлах
    Vdr - дрейфовая скорость, заданная в полуцелых узлах 
    delta_x - шаг пространственной сетки
    """
    
    n = len(N)
    n1 = n - 1

    Gamma = np.zeros(n1)

    N1 = N[0]

    for idx in range(n1):
        N2 = N[idx + 1]
        
        alpha = delta_x * Vdr[idx] / D[idx]
        exp_alpha = np.exp(alpha)
        
        hI0 = (exp_alpha - 1) / alpha * delta_x
        Gamma[idx] = D[idx] * (N1 - exp_alpha * N2) / hI0
        
        N1 = N2

    return Gamma[i-1]  # Возвращаем значение потока в точке i